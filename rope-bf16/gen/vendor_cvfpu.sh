#!/usr/bin/env bash
# Vendor the minimal CVFPU (FPnew) + common_cells source set into this repository.
#
# Per the project constraint, CVFPU is COPIED, never symlinked: rope-bf16 must be
# self-contained inside cv32e40x. Re-run this script only if you need to refresh the
# vendored sources from an upstream checkout.
#
# Usage: vendor_cvfpu.sh [path-to-checkout-containing-fpnew-and-common_cells]
#
# Only the files needed for an ADDMUL-only, FP16ALT-only, PARALLEL configuration are
# copied (see docs/notes.md "Vendored CVFPU subset"). DIVSQRT is deliberately excluded:
# fpnew_divsqrt_multi.sv depends on the external fpu_div_sqrt_mvp repository, and the
# RoPE datapath needs neither division nor square root (steps.md Section 5.1).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
SRC="${1:-$HOME/cv32e40p/rtl/vendor}"

FPNEW_SRC="$SRC/pulp_platform_fpnew"
CC_SRC="$SRC/pulp_platform_common_cells"

if [[ ! -d "$FPNEW_SRC/src" ]]; then
  echo "ERROR: no FPnew sources at $FPNEW_SRC/src" >&2
  echo "Pass the directory containing pulp_platform_fpnew/ and pulp_platform_common_cells/" >&2
  exit 1
fi

# Minimal FPnew set for ADDMUL / PARALLEL / FP16ALT.
FPNEW_FILES=(
  fpnew_pkg                    # formats, opgroups, helper functions  (compile FIRST)
  fpnew_classifier             # operand classification
  fpnew_rounding               # RNE and friends
  fpnew_fma                    # the ADD/MUL workhorse
  fpnew_noncomp                # NONCOMP opgroup (disabled, kept for completeness)
  fpnew_opgroup_fmt_slice      # per-format lane wrapper
  fpnew_opgroup_block          # per-opgroup wrapper, gates on FmtUnitTypes
  fpnew_top                    # top level
)

# FPnew's only common_cells dependencies in this configuration:
#   lzc          <- fpnew_fma (leading-zero count for normalization)
#   rr_arb_tree  <- fpnew_top, fpnew_opgroup_block (output arbitration)
#   cf_math_pkg  <- lzc (idx_width)
CC_FILES=(cf_math_pkg lzc rr_arb_tree)

mkdir -p "$ROOT/rtl/vendor/cvfpu/src" "$ROOT/rtl/vendor/common_cells/src" \
         "$ROOT/rtl/vendor/common_cells/include/common_cells"

# fpnew_fma.sv and friends do `\`include "common_cells/registers.svh"` for the FF/FFL
# flop macros, so the include tree has to come along too. Compiles need
# -Irtl/vendor/common_cells/include (the Makefile passes this).
for h in registers.svh assertions.svh; do
  cp -f "$CC_SRC/include/common_cells/$h" \
        "$ROOT/rtl/vendor/common_cells/include/common_cells/$h" 2>/dev/null || true
done

for f in "${FPNEW_FILES[@]}"; do
  cp -f "$FPNEW_SRC/src/$f.sv" "$ROOT/rtl/vendor/cvfpu/src/$f.sv"
done

for f in "${CC_FILES[@]}"; do
  cp -f "$CC_SRC/src/$f.sv" "$ROOT/rtl/vendor/common_cells/src/$f.sv"
done

# Licenses travel with the code (SolderPad 0.51 / Apache-2.0).
cp -f "$FPNEW_SRC/LICENSE.solderpad" "$ROOT/rtl/vendor/cvfpu/" 2>/dev/null || true
cp -f "$FPNEW_SRC/LICENSE.apache"    "$ROOT/rtl/vendor/cvfpu/" 2>/dev/null || true
cp -f "$FPNEW_SRC/README.license.md" "$ROOT/rtl/vendor/cvfpu/" 2>/dev/null || true
for l in LICENSE LICENSE.md LICENSE.solderpad; do
  cp -f "$CC_SRC/$l" "$ROOT/rtl/vendor/common_cells/" 2>/dev/null || true
done

echo "Vendored FPnew -> rtl/vendor/cvfpu/src:"
ls -1 "$ROOT/rtl/vendor/cvfpu/src"
echo "Vendored common_cells -> rtl/vendor/common_cells/src:"
ls -1 "$ROOT/rtl/vendor/common_cells/src"
