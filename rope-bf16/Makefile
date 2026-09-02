# =============================================================================
# BF16 RoPE coprocessor for CV32E40X -- build and verification
#
# Everything is self-contained inside the cv32e40x repository: CVFPU is COPIED into
# rtl/vendor/ (never symlinked), and the golden model needs only numpy.
#
# Common targets:
#   make gen          regenerate the theta ROM and sine LUT from gen/
#   make model        run the Python golden-model test suites
#   make lint         Verilator lint over all RTL
#   make test         run every RTL testbench (M1..M5)
#   make characterize regenerate docs/error_budget.md
#   make all          gen + model + lint + test
#
# Per-milestone: make m1 m2 m3 m4 m5
# =============================================================================

ROOT      := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
CV32E40X  := $(abspath $(ROOT)/..)

PYTHON    ?= python3
VERILATOR ?= verilator
BUILD     := $(ROOT)/build

# Number of random vectors per testbench. Override for a quick smoke run:
#   make test NVEC=2000
# steps.md asks for 10^7 on the primitives and the datapath; that is NVEC=10000000,
# which is slow in Verilator but supported -- see `make m1-full`.
NVEC      ?= 200000

# ---------------------------------------------------------------------------
# Source lists. fpnew_pkg MUST be compiled before anything that references it, and
# rope_pkg before the RoPE RTL, so these lists are order-significant.
# ---------------------------------------------------------------------------

CC_SRC := \
  $(ROOT)/rtl/vendor/common_cells/src/cf_math_pkg.sv \
  $(ROOT)/rtl/vendor/common_cells/src/lzc.sv \
  $(ROOT)/rtl/vendor/common_cells/src/rr_arb_tree.sv

FPNEW_SRC := \
  $(ROOT)/rtl/vendor/cvfpu/src/fpnew_pkg.sv \
  $(ROOT)/rtl/vendor/cvfpu/src/fpnew_classifier.sv \
  $(ROOT)/rtl/vendor/cvfpu/src/fpnew_rounding.sv \
  $(ROOT)/rtl/vendor/cvfpu/src/fpnew_fma.sv \
  $(ROOT)/rtl/vendor/cvfpu/src/fpnew_noncomp.sv \
  $(ROOT)/rtl/vendor/cvfpu/src/fpnew_opgroup_fmt_slice.sv \
  $(ROOT)/rtl/vendor/cvfpu/src/fpnew_opgroup_block.sv \
  $(ROOT)/rtl/vendor/cvfpu/src/fpnew_top.sv

ROPE_SRC := \
  $(ROOT)/rtl/rope_pkg.sv \
  $(ROOT)/rtl/rope_theta_rom.sv \
  $(ROOT)/rtl/rope_sin_rom.sv \
  $(ROOT)/rtl/rope_sin_lut.sv \
  $(ROOT)/rtl/rope_phase_gen.sv \
  $(ROOT)/rtl/rope_bf16_unit.sv \
  $(ROOT)/rtl/rope_bf16_mul.sv \
  $(ROOT)/rtl/rope_bf16_add.sv \
  $(ROOT)/rtl/rope_datapath.sv

XIF_SRC := \
  $(CV32E40X)/rtl/cv32e40x_if_xif.sv \
  $(ROOT)/rtl/rope_xif_coproc.sv

ALL_RTL := $(CC_SRC) $(FPNEW_SRC) $(ROPE_SRC) $(XIF_SRC)

# ---------------------------------------------------------------------------
# Verilator flags
#
# --binary builds a standalone simulation executable (Verilator >= 5).
# fpnew and common_cells are third-party: silence style-only warnings from them rather
# than editing vendored code, but keep real checks on. Our own RTL is held to -Wall.
# ---------------------------------------------------------------------------

# Waivers. Every one of these fires inside the vendored CVFPU / common_cells sources,
# whose style Verilator dislikes but which are correct and widely taped out. We do not
# edit vendored code, so they are waived globally rather than patched.
#
#   BLKANDNBLK   fpnew_fma drives inp_pipe_valid_q[0] with assign and the rest with
#                non-blocking assignments in a generate loop. Verilator flags the mix as
#                unsupported; it is legal SV and the elements are disjoint.
#   WIDTHCONCAT  fpnew_pkg's own fpu_features_t literals (Width: 64 etc.) are unsized;
#                assignment patterns to packed structs then look like concatenations.
#   UNSIGNED /   constant-comparison and empty-pin style in rr_arb_tree.
#   PINCONNECTEMPTY
VLT_WAIVE := \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNOPTFLAT -Wno-UNUSEDSIGNAL \
  -Wno-UNUSEDPARAM -Wno-DECLFILENAME -Wno-VARHIDDEN -Wno-SYNCASYNCNET \
  -Wno-CASEINCOMPLETE -Wno-ASCRANGE -Wno-GENUNNAMED -Wno-PINCONNECTEMPTY \
  -Wno-BLKANDNBLK -Wno-WIDTHCONCAT -Wno-UNSIGNED -Wno-CMPCONST

# rtl/vendor/common_cells/include is required: fpnew_fma.sv includes
# "common_cells/registers.svh" for its FF/FFL flop macros.
VLT_INC := -I$(ROOT)/rtl -I$(ROOT)/tb -I$(ROOT)/rtl/vendor/common_cells/include

VLT_FLAGS := --timing -Wall $(VLT_WAIVE) $(VLT_INC) --assert -j 0

# Testbench-only waivers. Scoreboards legitimately use blocking assignments for
# bookkeeping counters inside clocked processes, and stimulus tasks are procedural.
VLT_TB_WAIVE := -Wno-BLKSEQ -Wno-INITIALDLY -Wno-MULTIDRIVEN -Wno-STMTDLY

VLT_SIM   := $(VERILATOR) --binary $(VLT_FLAGS) $(VLT_TB_WAIVE)

.PHONY: all gen model lint test characterize clean m1 m2 m3 m4 m5 m1-full m4-full dirs

all: gen model lint test

dirs:
	@mkdir -p $(BUILD)

# ---------------------------------------------------------------------------
# Generated RTL
# ---------------------------------------------------------------------------

gen: $(ROOT)/rtl/rope_theta_rom.sv $(ROOT)/rtl/rope_sin_rom.sv

$(ROOT)/rtl/rope_theta_rom.sv: $(ROOT)/gen/gen_theta_rom.py $(ROOT)/model/rope_ref.py
	$(PYTHON) $(ROOT)/gen/gen_theta_rom.py --out $@

$(ROOT)/rtl/rope_sin_rom.sv: $(ROOT)/gen/gen_sin_lut.py $(ROOT)/model/bf16.py
	$(PYTHON) $(ROOT)/gen/gen_sin_lut.py --out $@

# ---------------------------------------------------------------------------
# Golden model
# ---------------------------------------------------------------------------

model:
	cd $(ROOT)/model && $(PYTHON) test_bf16.py
	cd $(ROOT)/model && $(PYTHON) test_rope_ref.py

characterize:
	cd $(ROOT) && $(PYTHON) model/characterize.py --out docs/error_budget.md

# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------

# Linting the coprocessor standalone leaves the CPU-driven half of the XIF interface
# (issue_valid, commit_*, result_ready, most of issue_req) undriven by construction, so
# UNDRIVEN is waived HERE ONLY. The `lint-integration` target below has the real core
# driving those signals and does not waive it.
lint: gen
	$(VERILATOR) --lint-only $(VLT_FLAGS) -Wno-UNDRIVEN $(ALL_RTL) --top-module rope_xif_coproc

# Full integration lint: the real cv32e40x core + interface + coprocessor.
# This is the steps.md Section 10.1 step 4 check.
#
# cv32e40x_sim_clock_gate.sv comes from bhv/ -- the core instantiates cv32e40x_clock_gate
# and ships only a behavioural model of it, so the technology-independent build needs it.
#
# IMPLICIT is waived because the three occurrences are in the CORE's own upstream RTL
# (cv32e40x_debug_triggers.sv, cv32e40x_ex_stage.sv), not in anything written here.
CORE_SRC := \
  $(CV32E40X)/rtl/include/cv32e40x_pkg.sv \
  $(CV32E40X)/bhv/cv32e40x_sim_clock_gate.sv

# All core RTL EXCEPT cv32e40x_if_xif.sv, which XIF_SRC already provides -- listing it
# twice is a duplicate-module error, not a harmless repeat.
CORE_RTL := $(filter-out $(CV32E40X)/rtl/cv32e40x_if_xif.sv,$(wildcard $(CV32E40X)/rtl/*.sv))

# COMBDLY is waived for the same reason as IMPLICIT: it fires only inside the core's own
# behavioural clock-gate model.
lint-integration: gen
	$(VERILATOR) --lint-only $(VLT_FLAGS) -Wno-UNDRIVEN -Wno-IMPLICIT -Wno-COMBDLY \
	  -I$(CV32E40X)/rtl -I$(CV32E40X)/rtl/include -I$(CV32E40X)/bhv \
	  $(CORE_SRC) \
	  $(CC_SRC) $(FPNEW_SRC) $(ROPE_SRC) $(XIF_SRC) \
	  $(CORE_RTL) \
	  $(ROOT)/rtl/rope_cv32e40x_wrapper.sv --top-module rope_cv32e40x_wrapper

# ---------------------------------------------------------------------------
# Testbenches
# ---------------------------------------------------------------------------

test: m1 m2 m3 m4 m5

# Vector filenames embed NVEC. This is load-bearing, not cosmetic: with a fixed name,
# `make m1 NVEC=1000000` sees an existing up-to-date vectors file and silently reuses the
# SMALLER previous one, so the run reports a pass over the wrong vector count. Encoding
# the count in the name makes a change to NVEC regenerate, and caches each size.
VEC_BF16     := $(BUILD)/vectors_bf16_$(NVEC).hex
VEC_PHASE    := $(BUILD)/vectors_phase_$(NVEC).hex
VEC_LUT      := $(BUILD)/vectors_lut_$(NVEC).hex
VEC_DATAPATH := $(BUILD)/vectors_datapath_$(NVEC).hex
VEC_XIF      := $(BUILD)/vectors_xif_$(NVEC).hex

# --- M1: BF16 primitives -----------------------------------------------------
$(VEC_BF16): $(ROOT)/model/gen_vectors.py $(ROOT)/model/bf16.py | dirs
	cd $(ROOT)/model && $(PYTHON) gen_vectors.py bf16 --n $(NVEC) --out $@

m1: gen $(VEC_BF16)
	cd $(BUILD) && $(VLT_SIM) --Mdir obj_m1 -o sim_m1 \
	  $(CC_SRC) $(FPNEW_SRC) $(ROOT)/rtl/rope_pkg.sv \
	  $(ROOT)/rtl/rope_bf16_unit.sv $(ROOT)/rtl/rope_bf16_mul.sv $(ROOT)/rtl/rope_bf16_add.sv \
	  $(ROOT)/tb/tb_rope_bf16_unit.sv --top-module tb_rope_bf16_unit
	cd $(BUILD) && ./obj_m1/sim_m1 +vectors=$(notdir $(VEC_BF16))

# steps.md Section 5.3 asks for 10^7 random pairs with zero mismatches.
m1-full:
	$(MAKE) m1 NVEC=10000000

# --- M2: integer phase generator --------------------------------------------
$(VEC_PHASE): $(ROOT)/model/gen_vectors.py $(ROOT)/model/rope_ref.py | dirs
	cd $(ROOT)/model && $(PYTHON) gen_vectors.py phase --n $(NVEC) --out $@

m2: gen $(VEC_PHASE)
	cd $(BUILD) && $(VLT_SIM) --Mdir obj_m2 -o sim_m2 \
	  $(ROOT)/rtl/rope_pkg.sv $(ROOT)/rtl/rope_theta_rom.sv $(ROOT)/rtl/rope_phase_gen.sv \
	  $(ROOT)/tb/tb_rope_phase_gen.sv --top-module tb_rope_phase_gen
	cd $(BUILD) && ./obj_m2/sim_m2 +vectors=$(notdir $(VEC_PHASE))

# --- M3: sine LUT ------------------------------------------------------------
$(VEC_LUT): $(ROOT)/model/gen_vectors.py $(ROOT)/model/rope_ref.py | dirs
	cd $(ROOT)/model && $(PYTHON) gen_vectors.py lut --n $(NVEC) --out $@

m3: gen $(VEC_LUT)
	cd $(BUILD) && $(VLT_SIM) --Mdir obj_m3 -o sim_m3 \
	  $(ROOT)/rtl/rope_pkg.sv $(ROOT)/rtl/rope_sin_rom.sv $(ROOT)/rtl/rope_sin_lut.sv \
	  $(ROOT)/tb/tb_rope_sin_lut.sv --top-module tb_rope_sin_lut
	cd $(BUILD) && ./obj_m3/sim_m3 +vectors=$(notdir $(VEC_LUT))

# --- M4: full BF16 datapath --------------------------------------------------
$(VEC_DATAPATH): $(ROOT)/model/gen_vectors.py $(ROOT)/model/rope_ref.py | dirs
	cd $(ROOT)/model && $(PYTHON) gen_vectors.py datapath --n $(NVEC) --out $@

m4: gen $(VEC_DATAPATH)
	cd $(BUILD) && $(VLT_SIM) --Mdir obj_m4 -o sim_m4 \
	  $(CC_SRC) $(FPNEW_SRC) $(ROPE_SRC) \
	  $(ROOT)/tb/tb_rope_datapath.sv --top-module tb_rope_datapath
	cd $(BUILD) && ./obj_m4/sim_m4 +vectors=$(notdir $(VEC_DATAPATH))

m4-full:
	$(MAKE) m4 NVEC=10000000

# --- M5: XIF coprocessor wrapper --------------------------------------------
$(VEC_XIF): $(ROOT)/model/gen_vectors.py $(ROOT)/model/rope_ref.py | dirs
	cd $(ROOT)/model && $(PYTHON) gen_vectors.py xif --n $(NVEC) --out $@

m5: gen $(VEC_XIF)
	cd $(BUILD) && $(VLT_SIM) --Mdir obj_m5 -o sim_m5 \
	  $(CC_SRC) $(FPNEW_SRC) $(ROPE_SRC) $(XIF_SRC) \
	  $(ROOT)/tb/tb_rope_xif_coproc.sv --top-module tb_rope_xif_coproc
	cd $(BUILD) && ./obj_m5/sim_m5 +vectors=$(notdir $(VEC_XIF))

# --- M7: the C test on the integrated core ----------------------------------
#
# Builds the bare-metal image, then runs it on the real core + coprocessor. This closes
# the loop: golden model -> compiled vectors -> real instructions -> core -> XIF -> RoPE.
.PHONY: sw
sw:
	$(MAKE) -C $(ROOT)/sw
	$(MAKE) -C $(ROOT)/sw verify

m7: gen sw | dirs
	cd $(BUILD) && $(VLT_SIM) -Wno-IMPLICIT -Wno-COMBDLY \
	  -I$(CV32E40X)/rtl -I$(CV32E40X)/rtl/include -I$(CV32E40X)/bhv \
	  --Mdir obj_m7 -o sim_m7 \
	  $(CORE_SRC) $(CC_SRC) $(FPNEW_SRC) $(ROPE_SRC) $(XIF_SRC) $(CORE_RTL) \
	  $(ROOT)/rtl/rope_cv32e40x_wrapper.sv \
	  $(ROOT)/tb/tb_rope_core.sv --top-module tb_rope_core
	cd $(BUILD) && ./obj_m7/sim_m7 +hex=$(ROOT)/sw/test_rope.hex

# ---------------------------------------------------------------------------
# Demo -- a single screen-recordable command that runs every check and prints
# a live metrics summary. See presentation.md.
#
#   make demo-prep    compile everything first (RUN THIS BEFORE RECORDING)
#   make demo         the recorded run
#   make demo-quick   same, 20k vectors instead of 200k
# ---------------------------------------------------------------------------
.PHONY: demo demo-quick demo-prep

# Build every simulator binary and the software image, so the recorded run spends
# its time producing results rather than invoking the C++ compiler.
demo-prep: gen sw
	@echo "Pre-building simulators (this is the slow part, done before recording)..."
	$(MAKE) m1 NVEC=$(NVEC) > $(BUILD)/prep_m1.log 2>&1 || (tail -20 $(BUILD)/prep_m1.log; false)
	$(MAKE) m2 NVEC=$(NVEC) > $(BUILD)/prep_m2.log 2>&1 || (tail -20 $(BUILD)/prep_m2.log; false)
	$(MAKE) m3 NVEC=$(NVEC) > $(BUILD)/prep_m3.log 2>&1 || (tail -20 $(BUILD)/prep_m3.log; false)
	$(MAKE) m4 NVEC=$(NVEC) > $(BUILD)/prep_m4.log 2>&1 || (tail -20 $(BUILD)/prep_m4.log; false)
	$(MAKE) m5 NVEC=$(NVEC) > $(BUILD)/prep_m5.log 2>&1 || (tail -20 $(BUILD)/prep_m5.log; false)
	$(MAKE) m7              > $(BUILD)/prep_m7.log 2>&1 || (tail -20 $(BUILD)/prep_m7.log; false)
	@echo "Ready. Now run: make demo"

demo:
	@NVEC=$(NVEC) bash $(ROOT)/demo.sh

demo-quick:
	@NVEC=20000 bash $(ROOT)/demo.sh

clean:
	rm -rf $(BUILD)
	$(MAKE) -C $(ROOT)/sw clean
