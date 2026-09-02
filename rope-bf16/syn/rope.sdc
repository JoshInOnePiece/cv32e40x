# =============================================================================
# rope.sdc -- timing constraints for the standalone BF16 RoPE coprocessor.
#
# Derived from cv32e40x/constraints/cv32e40x_core.sdc, as steps.md Section 9.3
# recommends, with the CORE-V-XIF timing budget applied on top.
#
# Target: 400 MHz (2.5 ns), matching Mugi's 45 nm operating point so the area/timing
# numbers are directly comparable. Start at 100 MHz on FPGA and tighten from there.
# =============================================================================

set CLK_PERIOD   2.5
set CLK_NAME     clk_i
set RST_NAME     rst_ni

create_clock -name $CLK_NAME -period $CLK_PERIOD [get_ports $CLK_NAME]

# Realistic uncertainty and transition until a real PLL/CTS model is available.
set_clock_uncertainty [expr {$CLK_PERIOD * 0.03}] [get_clocks $CLK_NAME]
set_clock_transition  [expr {$CLK_PERIOD * 0.03}] [get_clocks $CLK_NAME]

# Reset is asynchronous and externally synchronised.
set_false_path -from [get_ports $RST_NAME]

# -----------------------------------------------------------------------------
# CORE-V-XIF budget (steps.md Section 9.3)
# -----------------------------------------------------------------------------
#
# General split for XIF signals: 20% processor / 20% interconnect / 60% coprocessor.
set XIF_INPUT_DELAY  [expr {$CLK_PERIOD * 0.40}]
set XIF_OUTPUT_DELAY [expr {$CLK_PERIOD * 0.40}]

# The issue-request source operands are the exception and the reason
# rope_xif_coproc flops rs1/rs2 on arrival with ZERO combinational logic in front of
# them: CV32E40X drives them straight from its register-file bypass network, so the
# coprocessor sees only a small fraction of the period. Budget them tightly (85% gone
# before they arrive) so that a violation here shows up in synthesis rather than as a
# mysterious silicon failure.
set XIF_RS_INPUT_DELAY [expr {$CLK_PERIOD * 0.85}]

if {[llength [get_ports -quiet xif_issue_if*rs*]] > 0} {
  set_input_delay -clock $CLK_NAME $XIF_RS_INPUT_DELAY [get_ports xif_issue_if*rs*]
}

# Everything else on the XIF gets the general budget.
foreach pat {xif_issue_if* xif_commit_if* xif_result_if*} {
  set in_ports [get_ports -quiet $pat -filter {direction == in}]
  if {[llength $in_ports] > 0} {
    set_input_delay -clock $CLK_NAME $XIF_INPUT_DELAY $in_ports
  }
  set out_ports [get_ports -quiet $pat -filter {direction == out}]
  if {[llength $out_ports] > 0} {
    set_output_delay -clock $CLK_NAME $XIF_OUTPUT_DELAY $out_ports
  }
}

# -----------------------------------------------------------------------------
# Design rules
# -----------------------------------------------------------------------------

set_max_fanout 20 [current_design]
set_max_transition [expr {$CLK_PERIOD * 0.15}] [current_design]

# The sine LUT is a 1024x16 constant table. Left as logic deliberately (2 KB is small,
# and a macro would add a pipeline stage plus a memory-compiler dependency), but it is
# the widest combinational cone in the design, so keep an eye on it in the QoR report.
