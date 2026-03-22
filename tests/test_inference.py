"""
test_inference.py — cocotb 2.0 testbench for mlp_accel.v

DUT: mlp_accel (contains mac + FSM + $readmemh weight memories)
Testbench drives: clk, rst, start, pix_data
Testbench reads:  done, result, pix_addr

No BRAM model needed — weights/biases are inside the DUT.
"""
import os
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly, Timer

L1_IN=784; L1_OUT=64; L2_IN=64; L2_OUT=32; L3_IN=32; L3_OUT=10
FRAC=4; SAT_LIM=32767<<8; ACC_MAX=2**31-1; ACC_MIN=-(2**31)


# ── Reference model ──────────────────────────────────────────────────────────
def relu_ref(x):
    if x <= 0:       return 0
    if x > SAT_LIM:  return 32767
    return (x >> 8) & 0xFFFF

def mac_ref(a_ints, b_vals, bias_q8_8):
    acc = int(bias_q8_8) << 8
    for a, b in zip(a_ints, b_vals):
        acc = max(ACC_MIN, min(ACC_MAX,
            acc + (int(np.int32(int(np.int32(a)) * int(np.int32(np.int16(b))))) >> FRAC)))
    return acc

def mlp_forward(pixels, W1, b1, W2, b2, W3, b3):
    pix  = [int(p) for p in pixels]   # uint8 → positive int (0-255)
    act1 = [relu_ref(mac_ref(pix,  W1[n].tolist(), int(b1[n]))) for n in range(L1_OUT)]
    act2 = [relu_ref(mac_ref(act1, W2[n].tolist(), int(b2[n]))) for n in range(L2_OUT)]
    return int(np.argmax([mac_ref(act2, W3[n].tolist(), int(b3[n])) for n in range(L3_OUT)]))


# ── Load weights from hex files (same files the DUT reads) ──────────────────
def load_weights():
    hex_dir = os.environ.get("HEX_DIR", "hex")

    def read_hex16(fname, shape):
        path = os.path.join(hex_dir, fname)
        vals = []
        with open(path) as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("//"):
                    v = int(s, 16) & 0xFFFF
                    vals.append(v - 0x10000 if v >= 0x8000 else v)
        return np.array(vals, dtype=np.int16).reshape(shape)

    return (read_hex16("weight_l1.hex", (L1_OUT, L1_IN)),
            read_hex16("bias_l1.hex",   (L1_OUT,)),
            read_hex16("weight_l2.hex", (L2_OUT, L2_IN)),
            read_hex16("bias_l2.hex",   (L2_OUT,)),
            read_hex16("weight_l3.hex", (L3_OUT, L3_IN)),
            read_hex16("bias_l3.hex",   (L3_OUT,)))


# ── Pixel driver ─────────────────────────────────────────────────────────────
async def drive_pixels(dut, pixels):
    """Respond to pix_addr with correct pixel byte each cycle."""
    while True:
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")          # wait for NBA (pix_addr) to settle
        addr = dut.pix_addr.value.to_unsigned()
        dut.pix_data.value = int(pixels[addr]) if 0 <= addr < L1_IN else 0


async def wait_done(dut, timeout=300_000):
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.done.value) == 1:
            return True
    return False


# ── Test 1: Full inference ────────────────────────────────────────────────────
@cocotb.test()
async def test_full_inference(dut):
    """10 synthetic images — HW prediction must match Python reference."""
    cocotb.start_soon(Clock(dut.clk, 40, unit="ns").start())

    try:
        W1,b1,W2,b2,W3,b3 = load_weights()
        dut._log.info(f"Weights loaded from: {os.environ.get('HEX_DIR','hex')}")
    except Exception as e:
        dut._log.warning(f"Could not load weights: {e}")
        dut._log.warning("Using random weights — correctness not verified")
        rng = np.random.default_rng(42)
        W1=rng.integers(-50,50,(L1_OUT,L1_IN),dtype=np.int16)
        b1=rng.integers(-5,5,(L1_OUT,),dtype=np.int16)
        W2=rng.integers(-50,50,(L2_OUT,L2_IN),dtype=np.int16)
        b2=rng.integers(-5,5,(L2_OUT,),dtype=np.int16)
        W3=rng.integers(-50,50,(L3_OUT,L3_IN),dtype=np.int16)
        b3=rng.integers(-5,5,(L3_OUT,),dtype=np.int16)

    # Reset
    dut.rst.value = 1; dut.start.value = 0; dut.pix_data.value = 0
    for _ in range(5): await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    images = np.random.default_rng(0xDEADBEEF).integers(0,256,(10,L1_IN),dtype=np.uint8)

    for idx, pixels in enumerate(images):
        ref = mlp_forward(pixels, W1,b1,W2,b2,W3,b3)

        pix_task = cocotb.start_soon(drive_pixels(dut, pixels))
        await RisingEdge(dut.clk)   # one cycle for first pixel to appear

        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0

        ok = await wait_done(dut)
        pix_task.cancel()
        assert ok, f"Image {idx}: timeout"

        hw = int(dut.result.value)
        dut._log.info(f"Image {idx:02d}: HW={hw} Ref={ref}  "
                      f"{'MATCH' if hw==ref else 'MISMATCH'}")
        assert hw == ref, f"Image {idx}: HW={hw} Ref={ref}"
        await RisingEdge(dut.clk)

    dut._log.info("All 10 images PASS")


# ── Test 2: Timing ────────────────────────────────────────────────────────────
@cocotb.test()
async def test_inference_timing(dut):
    """Cycle count within 10% of theoretical."""
    # PRE(1)+MAC(L_IN)+DRAIN(2)+STORE(1) = L_IN+4 per neuron, +3 overhead
    # PRE(1)+PRE2(1)+MAC(L_IN)+DRAIN(3)+STORE(1) = L_IN+6 per neuron, +3 overhead
    EXPECTED = L1_OUT*(L1_IN+6) + L2_OUT*(L2_IN+6) + L3_OUT*(L3_IN+6) + 3

    cocotb.start_soon(Clock(dut.clk, 40, unit="ns").start())
    dut.rst.value=1; dut.start.value=0; dut.pix_data.value=0
    for _ in range(5): await RisingEdge(dut.clk)
    dut.rst.value=0
    await RisingEdge(dut.clk)

    dut.start.value=1
    await RisingEdge(dut.clk)
    dut.start.value=0

    count = 0
    for _ in range(EXPECTED * 2):
        await RisingEdge(dut.clk)
        await ReadOnly()
        count += 1
        if int(dut.done.value) == 1:
            break

    dut._log.info(f"Cycles: {count}  expected: ~{EXPECTED}")
    assert abs(count-EXPECTED) < EXPECTED*0.10
    dut._log.info("PASS")
