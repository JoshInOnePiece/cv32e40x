// -----------------------------------------------------------------------------
// rope_bf16_unit.sv -- shared thin wrapper over one CVFPU (FPnew) FP16ALT unit.
//
// Configured per steps.md Section 5.1:
//   * FP16ALT (= bfloat16) is the ONLY enabled format -- everything else is masked off
//     to keep area minimal. In FPnew nomenclature bfloat16 is FP16ALT; there is no
//     "BF16" anywhere in the vendored sources.
//   * Only the ADDMUL operation group is generated. DIVSQRT, NONCOMP and CONV are
//     DISABLED: the RoPE datapath needs none of them, DivSqrt is a large block, and
//     disabling it also avoids FPnew's external fpu_div_sqrt_mvp dependency (which is
//     deliberately not vendored -- see gen/vendor_cvfpu.sh).
//   * Rounding mode is tied STATICALLY to RNE (INV-1). It is not an input.
//
// Package scoping is explicit (`fpnew_pkg::`) throughout; the package is never
// wildcard-imported, per steps.md Section 16.
//
// Both ports use AXI-like valid/ready, which composes naturally with XIF.
// -----------------------------------------------------------------------------

module rope_bf16_unit #(
  // Number of pipeline registers inside the FPnew unit. 1 is a reasonable default;
  // raise it if timing closure needs more stages.
  parameter int unsigned NumPipeRegs = 1,
  // Which ADDMUL operation this instance performs.
  parameter fpnew_pkg::operation_e Op = fpnew_pkg::MUL,
  // op_mod inverts the sign of operand C, turning ADD into SUB.
  parameter logic OpMod = 1'b0
) (
  input  logic        clk_i,
  input  logic        rst_ni,

  // Operands are BF16 bit patterns. Their meaning depends on Op -- see the mapping
  // comment at `operands` below, which is where the FPnew ADD/MUL asymmetry is handled.
  input  logic [15:0] operand_a_i,
  input  logic [15:0] operand_b_i,

  input  logic        in_valid_i,
  output logic        in_ready_o,
  input  logic        flush_i,

  output logic [15:0] result_o,
  output logic        out_valid_o,
  input  logic        out_ready_i,
  output logic        busy_o
);

  // ---------------------------------------------------------------------------
  // FPnew configuration
  // ---------------------------------------------------------------------------

  // Width 16 with a 16-bit format means exactly one lane and no NaN-boxing (FPnew skips
  // the box check when FP_WIDTH == WIDTH, so EnableNanBox is immaterial here).
  localparam fpnew_pkg::fpu_features_t RopeFeatures = '{
    Width:         rope_pkg::Bf16Width,
    EnableVectors: 1'b0,
    EnableNanBox:  1'b0,
    FpFmtMask:     rope_pkg::FpFmtMaskBf16Only,   // FP16ALT only
    IntFmtMask:    rope_pkg::IntFmtMaskNone
  };

  // UnitTypes is indexed [ADDMUL, DIVSQRT, NONCOMP, CONV]. Only ADDMUL is PARALLEL;
  // the rest are DISABLED, which is what gates their generate blocks away entirely
  // (fpnew_opgroup_block keys on FmtUnitTypes[fmt] == PARALLEL).
  localparam fpnew_pkg::fpu_implementation_t RopeImpl = '{
    PipeRegs:   '{default: NumPipeRegs},
    UnitTypes:  '{'{default: fpnew_pkg::PARALLEL},  // ADDMUL  -- the only one we need
                  '{default: fpnew_pkg::DISABLED},  // DIVSQRT -- not needed, large
                  '{default: fpnew_pkg::DISABLED},  // NONCOMP -- not needed
                  '{default: fpnew_pkg::DISABLED}}, // CONV    -- no conversions (INV-1)
    PipeConfig: fpnew_pkg::BEFORE                   // registers at the inputs
  };

  // ---------------------------------------------------------------------------
  // Operand mapping -- the FPnew ADD/MUL asymmetry
  // ---------------------------------------------------------------------------
  //
  // FPnew's ADDMUL group is an FMA computing  operands[0]*operands[1] + operands[2].
  // fpnew_fma then rewrites the operands per opcode:
  //
  //   MUL: operand_c is forced to +/-0, so the result is operands[0] * operands[1].
  //        -> a, b go in slots 0 and 1.
  //
  //   ADD: operand_a is forced to +1.0, so the result is operands[1] + operands[2].
  //        -> a, b go in slots 1 and 2. Slot 0 is IGNORED.
  //
  // Putting the addends in slots 0 and 1 for an ADD would silently compute
  // `1.0 * b + 0` and throw away operand a. This is handled once, here.
  //
  //   op_mod inverts the sign of operand_c, so SUB is ADD with OpMod=1:
  //   operands[1] - operands[2].
  localparam int unsigned NumOperands = 3;
  logic [NumOperands-1:0][rope_pkg::Bf16Width-1:0] operands;

  always_comb begin
    operands = '0;
    if (Op == fpnew_pkg::ADD) begin
      operands[0] = rope_pkg::Bf16One;  // ignored by fpnew_fma (forced to +1.0)
      operands[1] = operand_a_i;
      operands[2] = operand_b_i;
    end else begin  // MUL (and any other ADDMUL op using slots 0/1)
      operands[0] = operand_a_i;
      operands[1] = operand_b_i;
      operands[2] = rope_pkg::Bf16PosZero;  // forced to +/-0 by fpnew_fma
    end
  end

  fpnew_pkg::status_t status_unused;
  logic               tag_unused;

  fpnew_top #(
    .Features       ( RopeFeatures ),
    .Implementation ( RopeImpl     ),
    // PulpDivsqrt only selects between DivSqrt implementations; DIVSQRT is DISABLED
    // above so nothing is generated either way.
    .PulpDivsqrt    ( 1'b0         ),
    .TagType        ( logic        )
  ) i_fpnew (
    .clk_i,
    .rst_ni,
    .operands_i     ( operands                     ),
    // RNE, statically. Not an input: INV-1 requires round-to-nearest-even everywhere.
    .rnd_mode_i     ( fpnew_pkg::RNE               ),
    .op_i           ( Op                           ),
    .op_mod_i       ( OpMod                        ),
    .src_fmt_i      ( fpnew_pkg::FP16ALT           ),
    .dst_fmt_i      ( fpnew_pkg::FP16ALT           ),
    .int_fmt_i      ( fpnew_pkg::INT8              ),  // unused, CONV disabled
    .vectorial_op_i ( 1'b0                         ),
    .tag_i          ( 1'b0                         ),
    .simd_mask_i    ( 1'b1                         ),
    .in_valid_i     ( in_valid_i                   ),
    .in_ready_o     ( in_ready_o                   ),
    .flush_i        ( flush_i                      ),
    .result_o       ( result_o                     ),
    .status_o       ( status_unused                ),
    .tag_o          ( tag_unused                   ),
    .out_valid_o    ( out_valid_o                  ),
    .out_ready_i    ( out_ready_i                  ),
    .busy_o         ( busy_o                       )
  );

endmodule : rope_bf16_unit
