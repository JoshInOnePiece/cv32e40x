// -----------------------------------------------------------------------------
// rope_bf16_fma.sv -- one BF16 fused multiply-add:  result = a*b + c  (or a*b - c).
//
// The third member of the rope_bf16_{mul,add,fma} family, all of which are thin
// wrappers over the SAME rope_bf16_unit / FPnew ADDMUL block. FPnew's ADDMUL group is
// an FMA; `rope_bf16_mul` and `rope_bf16_add` use it with one of its three operands
// forced to a constant, which wastes half the unit. This wrapper uses all of it.
//
// WHY THIS EXISTS
// ---------------
// The rotation needs two products per output:
//
//     x' = x*cos - y*sin
//     y' = x*sin + y*cos
//
// An FMA fuses ONE multiply with the add, so the other product must already exist and
// arrive on `c`. Building the datapath that way takes 4 units instead of 6:
//
//     p1 = y*sin          (rope_bf16_mul)      p2 = y*cos          (rope_bf16_mul)
//     x' = x*cos - p1     (this module, Sub=1) y' = x*sin + p2     (this module, Sub=0)
//
// The dependency depth is unchanged at 2 -- p1 must exist before the FMA can start --
// so this does NOT shorten the pipeline. What it changes is the number of roundings
// per output, from three to two: `x*cos` is formed to its full 2p bits inside the unit
// and aligned against the addend in the internal accumulator, so it is never rounded
// on its own. Measured against an FP64 reference that is 1.79x lower RMS error.
//
// The cost is that this no longer matches Model A (strict BF16, INV-1), which rounds at
// every node. `rope_datapath`'s `UseFma` parameter selects between them, and the
// matching reference is `model/rope_ref.py:rope_fma_bf16` (Model A').
//
// SIGNS
// -----
// `Sub` maps to FPnew's op_mod, which inverts operand C's sign UNCONDITIONALLY:
//
//     operand_c.sign = operand_c.sign ^ op_mod_q
//
// one XOR, no conditional negate, no magnitude compare. It is therefore correct for any
// sign of c -- subtracting a negative addend adds. Verified across all four quadrants
// (cos<0, sin<0, both) crossed with all four sign combinations of x and y.
// -----------------------------------------------------------------------------

module rope_bf16_fma #(
  parameter int unsigned NumPipeRegs = 0,
  // 1 => result = a*b - c   (FPnew FMADD with op_mod=1, i.e. FMSUB)
  // 0 => result = a*b + c
  parameter logic        Sub         = 1'b0
) (
  input  logic        clk_i,
  input  logic        rst_ni,

  input  logic [15:0] a_i,   // multiplicand
  input  logic [15:0] b_i,   // multiplier
  input  logic [15:0] c_i,   // addend -- the product computed by the upstream stage

  input  logic        in_valid_i,
  output logic        in_ready_o,
  input  logic        flush_i,

  output logic [15:0] result_o,
  output logic        out_valid_o,
  input  logic        out_ready_i,
  output logic        busy_o
);

  rope_bf16_unit #(
    .NumPipeRegs ( NumPipeRegs      ),
    .Op          ( fpnew_pkg::FMADD ),
    .OpMod       ( Sub              )
  ) i_unit (
    .clk_i,
    .rst_ni,
    // FMADD is the one ADDMUL op FPnew does not rewrite, so the mapping is direct.
    .operand_a_i ( a_i ),
    .operand_b_i ( b_i ),
    .operand_c_i ( c_i ),
    .in_valid_i,
    .in_ready_o,
    .flush_i,
    .result_o,
    .out_valid_o,
    .out_ready_i,
    .busy_o
  );

endmodule : rope_bf16_fma
