# mlp_accel/Makefile

PYTHON  ?= python3
SIM     ?= icarus
WAVES   ?= 0
DEVICE  ?= 85k
PACKAGE ?= CABGA381
BOARD   ?= ulx3s

.PHONY: train quantize sim synth pnr pack prog clean

# ── Software flow ─────────────────────────────────────────────────────────────
train:
	@echo ">>> Training..."
	cd training && $(PYTHON) train.py

quantize:
	@echo ">>> Exporting hex files..."
	cd training && $(PYTHON) quantize.py

sim: quantize
	@echo ">>> Running inference simulation..."
	cd tests/sim/infer && make SIM=$(SIM) WAVES=$(WAVES)

# ── Hardware flow ─────────────────────────────────────────────────────────────
synth:
	@echo ">>> Yosys synthesis..."
	@mkdir -p build
	@echo "Copying hex files for Yosys $readmemh..."
	@mkdir -p syn/hex && cp rtl/hex/*.hex syn/hex/
	cd syn && yosys synth.ys 2>&1 | tee ../build/synth.log
	@echo "--- Resource summary ---"
	@grep -E "LUT|FF|BRAM|DSP|EBR|DP16" build/synth.log | tail -10

pnr: synth
	@echo ">>> nextpnr place and route..."
	nextpnr-ecp5 \
	    --$(DEVICE) --package $(PACKAGE) \
	    --lpf syn/ulx3s.lpf \
	    --json build/top.json \
	    --textcfg build/top.config \
	    --freq 25 \
	    --timing-allow-fail \
	    2>&1 | tee build/pnr.log
	@grep -i "max frequency\|slack" build/pnr.log | tail -5

pack: pnr
	@echo ">>> ecppack bitstream..."
	ecppack --input build/top.config --bit build/top.bit
	@ls -lh build/top.bit

prog: pack
	@echo ">>> Flashing ULX3S..."
	openFPGALoader --board $(BOARD) build/top.bit

clean:
	rm -rf build/
	rm -rf tests/sim/infer/sim_build tests/sim/infer/results.xml
	rm -rf tests/sim/infer/hex
	rm -rf training/weights training/data

loopback:
	@echo ">>> Building UART loopback test..."
	@mkdir -p build
	cd syn && yosys synth_loopback.ys 2>&1 | tee ../build/synth_loopback.log
	nextpnr-ecp5 --85k --package CABGA381 \
	    --lpf syn/ulx3s.lpf \
	    --json build/top_loopback.json \
	    --textcfg build/top_loopback.config \
	    --freq 25 --timing-allow-fail 2>&1 | tee build/pnr_loopback.log
	ecppack --input build/top_loopback.config --bit build/top_loopback.bit
	openFPGALoader --board ulx3s build/top_loopback.bit
