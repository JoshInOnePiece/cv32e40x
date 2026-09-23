// -----------------------------------------------------------------------------
// rope_bf16_add.sv -- BF16 adder/subtractor, rounded RNE to BF16.
//
//   Sub = 0 : result = a + b
//   Sub = 1 : result = a - b
//
// Subtraction uses FPnew's op_mod (which inverts the sign of operand C) rather than a
// separate negation stage: it is exact and free. Note that a BF16 sign flip is a pure
// XOR on bit 15, so even an explicit negation would introduce no rounding -- but going
// through op_mod keeps the operand count down.
// -----------------------------------------------------------------------------

module rope_bf16_add #(
  parameter int unsigned NumPipeRegs = 0,
  parameter logic        Sub         = 1'b0
) (
  input  logic        clk_i,
  input  logic        rst_ni,

  input  logic [15:0] a_i,
  input  logic [15:0] b_i,

  input  logic        in_valid_i,
  output logic        in_ready_o,
  input  logic        flush_i,

  output logic [15:0] result_o,
  output logic        out_valid_o,
  input  logic        out_ready_i,
  output logic        busy_o
);

  rope_bf16_unit #(
    .NumPipeRegs ( NumPipeRegs    ),
    .Op          ( fpnew_pkg::ADD ),
    .OpMod       ( Sub            )
  ) i_unit (
    .clk_i,
    .rst_ni,
    // rope_bf16_unit maps these into FPnew's slots 1 and 2, because FPnew's ADD forces
    // operand slot 0 to +1.0. See the operand-mapping comment there.
    .operand_a_i ( a_i                 ),
    .operand_b_i ( b_i                 ),
    .operand_c_i ( rope_pkg::Bf16PosZero ),  // unused: ADD takes slots 1 and 2
    .in_valid_i,
    .in_ready_o,
    .flush_i,
    .result_o,
    .out_valid_o,
    .out_ready_i,
    .busy_o
  );

endmodule : rope_bf16_add
