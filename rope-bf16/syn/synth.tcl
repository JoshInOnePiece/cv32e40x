# =============================================================================
# synth.tcl -- Genus synthesis script for the standalone BF16 RoPE coprocessor.
#
#   genus -batch -f syn/synth.tcl
#
# Produces the Milestone 9 measurements: area, timing and the area breakdown
# (sine LUT vs theta ROM vs CVFPU units vs phase logic vs XIF wrapper).
#
# STATUS: NOT YET RUN. This project has no library set up here, so the script is
# untested end to end -- it encodes the intended flow and the reporting that the
# writeup needs, and will need the library paths below pointed at a real PDK.
# Everything above Milestone 9 is verified in simulation and does not depend on this.
# =============================================================================

# -----------------------------------------------------------------------------
# Library setup -- EDIT THESE
# -----------------------------------------------------------------------------
# Mugi reports 45 nm at 400 MHz; matching that node makes the comparison meaningful.
if {![info exists env(ROPE_LIB_PATH)]} {
  puts "ERROR: set ROPE_LIB_PATH to the directory holding the .lib files"
  exit 1
}
set_db init_lib_search_path $env(ROPE_LIB_PATH)
set_db library [glob -directory $env(ROPE_LIB_PATH) *.lib]

set ROOT   [file normalize [file dirname [info script]]/..]
set OUTDIR $ROOT/syn/out
file mkdir $OUTDIR

# -----------------------------------------------------------------------------
# Read RTL
#
# Order matters: fpnew_pkg before anything referencing it, rope_pkg before the RoPE RTL.
# rtl/vendor/common_cells/include must be on the include path for registers.svh.
# -----------------------------------------------------------------------------
set_db init_hdl_search_path [list \
  $ROOT/rtl \
  $ROOT/rtl/vendor/common_cells/include \
]

set RTL_FILES [list \
  $ROOT/rtl/vendor/common_cells/src/cf_math_pkg.sv \
  $ROOT/rtl/vendor/common_cells/src/lzc.sv \
  $ROOT/rtl/vendor/common_cells/src/rr_arb_tree.sv \
  $ROOT/rtl/vendor/cvfpu/src/fpnew_pkg.sv \
  $ROOT/rtl/vendor/cvfpu/src/fpnew_classifier.sv \
  $ROOT/rtl/vendor/cvfpu/src/fpnew_rounding.sv \
  $ROOT/rtl/vendor/cvfpu/src/fpnew_fma.sv \
  $ROOT/rtl/vendor/cvfpu/src/fpnew_noncomp.sv \
  $ROOT/rtl/vendor/cvfpu/src/fpnew_opgroup_fmt_slice.sv \
  $ROOT/rtl/vendor/cvfpu/src/fpnew_opgroup_block.sv \
  $ROOT/rtl/vendor/cvfpu/src/fpnew_top.sv \
  $ROOT/rtl/rope_pkg.sv \
  $ROOT/rtl/rope_theta_rom.sv \
  $ROOT/rtl/rope_sin_rom.sv \
  $ROOT/rtl/rope_sin_lut.sv \
  $ROOT/rtl/rope_phase_gen.sv \
  $ROOT/rtl/rope_bf16_unit.sv \
  $ROOT/rtl/rope_bf16_mul.sv \
  $ROOT/rtl/rope_bf16_add.sv \
  $ROOT/rtl/rope_datapath.sv \
]

read_hdl -language sv $RTL_FILES

# Synthesise the datapath rather than the XIF wrapper: the wrapper's ports are a
# SystemVerilog interface, which complicates a standalone run, and the datapath plus
# phase/LUT is what the area and energy numbers are actually about.
elaborate rope_datapath

read_sdc $ROOT/syn/rope.sdc

check_design -unresolved
report_timing -lint

# -----------------------------------------------------------------------------
# Synthesis
# -----------------------------------------------------------------------------
set_db syn_generic_effort high
set_db syn_map_effort     high
set_db syn_opt_effort     high

syn_generic
syn_map
syn_opt

# -----------------------------------------------------------------------------
# Reports -- these ARE the Milestone 9 deliverables
# -----------------------------------------------------------------------------
report_timing            > $OUTDIR/timing.rpt
report_area              > $OUTDIR/area.rpt
report_power             > $OUTDIR/power.rpt
report_gates             > $OUTDIR/gates.rpt
report_qor               > $OUTDIR/qor.rpt

# Area breakdown by block, which is the table the writeup needs:
#   sine LUT (2 KB) vs theta ROM (256 B) vs CVFPU units vs phase logic vs wrapper.
report_area -depth 3     > $OUTDIR/area_hierarchy.rpt

foreach inst {i_rom_rand i_rom_seq gen_mul i_sub_x i_add_y} {
  if {[llength [get_db insts -quiet *$inst*]] > 0} {
    report_area [get_db insts *$inst*] >> $OUTDIR/area_breakdown.rpt
  }
}

write_hdl                > $OUTDIR/rope_datapath_netlist.v
write_sdc                > $OUTDIR/rope_datapath.sdc

puts "Synthesis complete. Reports in $OUTDIR"
exit
