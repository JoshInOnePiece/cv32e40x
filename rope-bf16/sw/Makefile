# =============================================================================
# Bare-metal build for the BF16 RoPE software test (Milestone 7).
#
#   make            build test_rope.elf + .hex + .dis
#   make verify     check that the .insn encoding gcc emitted is what rope_pkg decodes
#   make clean
#
# Uses the CORE-V toolchain if present, else a generic riscv32 gcc.
# =============================================================================

HERE   := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
ROOT   := $(abspath $(HERE)/..)
PYTHON ?= python3

# Toolchain autodetect. The CV32E40X is RV32IMC; the RoPE unit needs no F extension,
# because the coprocessor consumes BF16 in INTEGER registers (two per GPR).
TOOLCHAIN_DIR ?= $(HOME)/corev-openhw-gcc-ubuntu2204-20240530/bin
CROSS         ?= $(if $(wildcard $(TOOLCHAIN_DIR)/riscv32-corev-elf-gcc),$(TOOLCHAIN_DIR)/riscv32-corev-elf-,riscv32-unknown-elf-)

CC      := $(CROSS)gcc
OBJDUMP := $(CROSS)objdump
OBJCOPY := $(CROSS)objcopy

# zicsr is required explicitly: since Zicsr was split out of the base ISA, `csrw mtvec`
# in crt0.S will not assemble without it.
ARCH  ?= rv32imc_zicsr
ABI   ?= ilp32

CFLAGS := -march=$(ARCH) -mabi=$(ABI) -Os -g \
          -ffunction-sections -fdata-sections -fno-builtin \
          -Wall -Wextra -std=c11 -I$(HERE)

LDFLAGS := -march=$(ARCH) -mabi=$(ABI) -nostdlib -nostartfiles \
           -T$(HERE)/link.ld -Wl,--gc-sections -Wl,-Map=$(HERE)/test_rope.map

SRCS := $(HERE)/crt0.S $(HERE)/test_rope.c
ELF  := $(HERE)/test_rope.elf
HEX  := $(HERE)/test_rope.hex
DIS  := $(HERE)/test_rope.dis
VEC  := $(HERE)/rope_vectors.h

.PHONY: all clean verify vectors

all: $(DIS) $(HEX)

vectors: $(VEC)

$(VEC): $(ROOT)/model/gen_c_vectors.py $(ROOT)/model/rope_ref.py $(ROOT)/model/bf16.py
	cd $(ROOT)/model && $(PYTHON) gen_c_vectors.py --out $@

$(ELF): $(SRCS) $(VEC) $(HERE)/rope_intrin.h $(HERE)/link.ld
	$(CC) $(CFLAGS) $(LDFLAGS) $(HERE)/crt0.S $(HERE)/test_rope.c -o $@

$(DIS): $(ELF)
	$(OBJDUMP) -d -S $< > $@

# Flat little-endian hex image, one 32-bit word per line, for $readmemh in the testbench.
$(HEX): $(ELF)
	$(OBJCOPY) -O verilog --verilog-data-width=4 $< $@

# ---------------------------------------------------------------------------
# Encoding verification
#
# The whole software path hinges on `.insn r 0x0B, 0, 0, rd, rs1, rs2` assembling to an
# instruction the coprocessor decodes. Rather than trust that, disassemble and check the
# fields: opcode must be 0x0B, funct3 0, funct7 0.
# ---------------------------------------------------------------------------
verify: $(DIS)
	@$(PYTHON) $(HERE)/verify_encoding.py $(DIS)

clean:
	rm -f $(ELF) $(HEX) $(DIS) $(HERE)/test_rope.map
