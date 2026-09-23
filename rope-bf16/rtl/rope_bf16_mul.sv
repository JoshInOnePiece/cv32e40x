// -----------------------------------------------------------------------------
// rope_bf16_mul.sv -- BF16 multiplier. result = a * b, rounded RNE to BF16.
//
// A named wrapper over rope_bf16_unit so the datapath reads as what it is. All the
// CVFPU configuration detail lives in rope_bf16_unit.sv.
// -----------------------------------------------------------------------------

module rope_bf16_mul #(
  parameter int unsigned NumPipeRegs = 0
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
    .NumPipeRegs ( NumPipeRegs        ),
    .Op          ( fpnew_pkg::MUL     ),
    .OpMod       ( 1'b0               )
  ) i_unit (
    .clk_i,
    .rst_ni,
    .operand_a_i ( a_i                 ),
    .operand_b_i ( b_i                 ),
    .operand_c_i ( rope_pkg::Bf16PosZero ),  // FPnew forces slot 2 to +/-0 for MUL
    .in_valid_i,
    .in_ready_o,
    .flush_i,
    .result_o,
    .out_valid_o,
    .out_ready_i,
    .busy_o
  );

endmodule : rope_bf16_mul
