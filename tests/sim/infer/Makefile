# tests/sim/infer/Makefile

SIM                 ?= icarus
TOPLEVEL_LANG        = verilog
TOPLEVEL             = mlp_accel
COCOTB_TEST_MODULES  = test_inference

THIS_DIR   := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJ_ROOT  := $(abspath $(THIS_DIR)../../..)
RTL_DIR    := $(PROJ_ROOT)/rtl
TESTS_DIR  := $(PROJ_ROOT)/tests
HEX_SRC    := $(RTL_DIR)/hex
HEX_DST    := $(THIS_DIR)hex

VERILOG_SOURCES = $(RTL_DIR)/mlp_accel.v

# No plusargs or compile-time defines needed.
# mlp_accel.v uses hardcoded "hex/..." relative paths.
# vvp runs from THIS directory, so hex/ resolves to HEX_DST.

export PYTHONPATH := $(TESTS_DIR)
export HEX_DIR    := $(HEX_DST)

COCOTB_REDUCED_LOG_FMT = 1

ifeq ($(WAVES),1)
SIM_ARGS += -fst
endif

# Copy hex files into THIS directory before simulation
$(shell mkdir -p $(HEX_DST) && cp $(HEX_SRC)/*.hex $(HEX_DST)/ 2>/dev/null || true)

include $(shell cocotb-config --makefiles)/Makefile.sim
