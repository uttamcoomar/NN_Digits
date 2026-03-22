# mlp_accel — MNIST MLP Accelerator on ECP5 FPGA

A fully open-source neural network inference accelerator implemented on the
Radiona ULX3S development board (Lattice ECP5 85F). Recognises handwritten
digits (0–9) from the MNIST dataset at **94% accuracy** and **476 inferences
per second**, using only 6.7% of the FPGA's logic resources.

## Demo

Hold a handwritten digit in front of your webcam. The FPGA classifies it in
~2 ms. Results appear live on screen and on the board's LEDs.

```
Host PC  ──USB──►  ULX3S ECP5  ──LED[3:0]──►  predicted digit (binary)
         ◄──────   (115200 baud)
```

## Results

| Metric               | Value                        |
|----------------------|------------------------------|
| Test accuracy        | **94%** (100 MNIST images)   |
| Inference latency    | ~2.1 ms                      |
| Throughput           | ~476 inferences/sec          |
| LUT4                 | 2,864                        |
| EBR block RAM (18Kb) | 49                           |
| DSP18                | 1                            |
| Total cells          | 5,649 / 83,640 (6.7%)        |
| Clock                | 25 MHz                       |
| UART baud rate       | 115,200                      |

## Network Architecture

```
Input (784)  →  Dense 64 (ReLU)  →  Dense 32 (ReLU)  →  Dense 10 (argmax)
```

**Fixed-point quantization:**
- Weights:      Q4.12 signed 16-bit  (scale = 4096)
- Activations:  Q8.8  unsigned 16-bit (scale = 256)
- Accumulator:  Q8.16 signed 32-bit
- Biases:       Q8.8  signed 16-bit

## Project Layout

```
mlp_accel/
├── Makefile                      ← all build targets
├── README.md
├── rtl/
│   ├── mlp_accel.v               ← entire accelerator: MAC + FSM + weight BRAMs
│   ├── top.v                     ← ULX3S top level: UART + pixel buffer + MLP
│   ├── top_loopback.v            ← UART loopback test (diagnostics)
│   └── uart/
│       ├── uart_rx.v             ← 8N1 UART receiver
│       └── uart_tx.v             ← 8N1 UART transmitter
├── syn/
│   ├── ulx3s.lpf                 ← pin constraints (ECP5 85F CABGA381)
│   ├── synth.ys                  ← Yosys synthesis script
│   └── synth_loopback.ys         ← Yosys script for loopback test
├── training/
│   ├── train.py                  ← PyTorch training (MNIST, 784→64→32→10)
│   └── quantize.py               ← weight quantization → hex file export
├── tests/
│   ├── test_inference.py         ← cocotb 2.0 simulation testbench
│   └── sim/infer/
│       └── Makefile              ← cocotb simulation entry point
└── host/
    └── webcam_detect.py          ← live webcam inference via UART
```

## Prerequisites

**Hardware:**
- Radiona ULX3S v2.x / v3.x (ECP5 85F, CABGA381)
- USB cable (US1 port — the FTDI JTAG/UART port)

**Software:**
```bash
# FPGA toolchain
sudo apt install yosys nextpnr-ecp5 ecppack openFPGALoader

# Python
pip install torch torchvision numpy cocotb pyserial opencv-python
```

## Quick Start

```bash
# 1. Train the model (~5 min)
make train

# 2. Export quantized weights as hex files
make quantize

# 3. Simulate (requires iverilog + cocotb 2.0) — should show 2/2 PASS
make sim

# 4. Synthesise → place & route → pack → flash
make synth
make pnr
make pack
make prog

# 5. Test on hardware
python3 - << 'PYEOF'
import serial, numpy as np, time
from torchvision import datasets, transforms
ds  = datasets.MNIST("training/data", train=False, download=True,
                     transform=transforms.ToTensor())
img, label = ds[0]
pix = (img.squeeze().numpy()*255).astype(np.uint8).flatten()
ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=10)
time.sleep(0.5); ser.reset_input_buffer()
ser.write(pix.tobytes())
resp = ser.read(1)
if resp: print(f"Label={label}  FPGA={resp[0]}  {'PASS' if resp[0]==label else 'FAIL'}")
ser.close()
PYEOF

# 6. Live webcam demo
python3 host/webcam_detect.py --port /dev/ttyUSB0
```

## Make Targets

| Target      | Description                                      |
|-------------|--------------------------------------------------|
| `make train`    | Train PyTorch model, saves `training/weights/mlp_mnist.pt` |
| `make quantize` | Export Q4.12/Q8.8 weights to `rtl/hex/*.hex`    |
| `make sim`      | Run cocotb simulation (2 tests, should be 2/2 PASS) |
| `make synth`    | Yosys synthesis → `build/top.json`              |
| `make pnr`      | nextpnr place & route → `build/top.config`      |
| `make pack`     | ecppack bitstream → `build/top.bit`             |
| `make prog`     | Flash ULX3S via openFPGALoader                  |
| `make loopback` | Build and flash UART loopback test              |
| `make clean`    | Remove build artefacts                          |

## Host Protocol

The FPGA acts as a simple inference server over UART (115200 8N1):

1. Host sends exactly **784 bytes** — raw uint8 pixel values, row-major
2. FPGA buffers all 784 bytes, runs inference (~2.1 ms)
3. FPGA sends **1 byte** — predicted digit 0–9

## Hardware Design

### mlp_accel.v

Contains three modules in one file:

**`weight_bram`** — synchronous-read block RAM with no reset port. The
no-reset constraint is required for Yosys to infer ECP5 EBR (block RAM)
rather than LUT-based distributed RAM. Weights are initialised from hex
files via `$readmemh` at elaboration time.

**`mac`** — pipelined signed multiply-accumulate unit. Computes
`acc += (a × b) >> 4` with signed saturation. Two-cycle pipeline:
stage 1 multiplies, stage 2 accumulates.

**`mlp_accel`** — one-hot FSM that sequences all three layers through
the shared MAC unit. Uses a registered `base_addr` accumulator instead
of a multiplier for BRAM addresses, eliminating the critical path multiply
that previously limited Fmax.

### FSM State Sequence (per neuron)

```
IDLE → L?_PRE → L?_PRE2 → L?_MAC (×L_IN) → L?_DRAIN (×2) → L?_STORE → ...
```

- **PRE**: present BRAM address (in_ctr=0, nrn=N)
- **PRE2**: registered BRAM data valid; load MAC with bias; assert mac_clr
- **MAC**: stream L_IN weight×activation products into MAC
- **DRAIN**: flush 2-cycle MAC pipeline
- **STORE**: ReLU + write to activation buffer

## Webcam Demo

```bash
python3 host/webcam_detect.py --port /dev/ttyUSB0
```

- Green box shows the crop region — keep your digit centred inside it
- Second window shows the 28×28 image sent to the FPGA
- Press **Space** to run inference, **Q** to quit
- Result overlaid on camera feed, latency ~70 ms (dominated by UART TX)

## Toolchain Versions Tested

| Tool              | Version       |
|-------------------|---------------|
| Yosys             | 0.38+         |
| nextpnr-ecp5      | 0.7+          |
| Python            | 3.12          |
| cocotb            | 2.0.1         |
| iverilog          | 12.0          |
| openFPGALoader    | 0.12+         |
| PyTorch           | 2.x           |

## License

MIT
