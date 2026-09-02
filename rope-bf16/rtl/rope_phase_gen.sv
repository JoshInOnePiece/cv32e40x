// -----------------------------------------------------------------------------
// rope_phase_gen.sv -- integer phase generator (steps.md Milestone 2).
//
// This is the piece that makes BF16 RoPE possible at all (INV-2).
//
//   P = (m * Phi_i) mod 2^32          <- the mod is FREE: natural wraparound
//
//     P[31:30] = quadrant  (2 bits)
//     P[29:20] = LUT index (10 bits)
//     P[19:0]  = unused (reserved for future interpolation)
//
// Everything here is integer arithmetic. No value in this module ever enters a BF16
// multiplier or adder: the phase word is an ADDRESS, not data.
//
// Two operating modes (steps.md Section 6.3):
//
//   Random access (`ROPE.ROT`)  -- compute m * Phi_i directly. m is 13 bits and Phi is
//                                  32 bits, so this is a 13x32 multiplier: cheap.
//   Sequential decode           -- hold per-pair phase state and add Phi_i once per
//                                  token. One integer add, and ZERO drift because
//                                  integer addition is exact.
//
// NOT USED, deliberately: the trigonometric recurrence
//     cos((m+1)th) = cos(m th)cos(th) - sin(m th)sin(th)
// In BF16 that drifts catastrophically -- the error compounds multiplicatively and the
// state wanders off the unit circle within a few dozen tokens at 7 mantissa bits. The
// integer accumulator is both exact and cheaper.
// -----------------------------------------------------------------------------

module rope_phase_gen
  import rope_pkg::*;
#(
  parameter int unsigned NumPairs = rope_pkg::NumPairsMax,
  parameter int unsigned IdxBits  = rope_pkg::PairIdxBits,
  parameter int unsigned PosBits  = rope_pkg::PosBits
) (
  input  logic                 clk_i,
  input  logic                 rst_ni,

  // ---- random-access mode (combinational) --------------------------------
  input  logic [PosBits-1:0]   pos_i,        // m
  input  logic [IdxBits-1:0]   pair_idx_i,   // i
  output logic [PhaseBits-1:0] phase_o,      // (m * Phi_i) mod 2^32

  // ---- sequential mode --------------------------------------------------
  // seq_clear_i resets the accumulator for pair seq_idx_i to zero (position 0).
  // seq_step_i  advances it by one token: acc += Phi_i, exactly.
  input  logic                 seq_clear_i,
  input  logic                 seq_step_i,
  input  logic [IdxBits-1:0]   seq_idx_i,
  output logic [PhaseBits-1:0] seq_phase_o
);

  // ---------------------------------------------------------------------------
  // Phi lookup
  // ---------------------------------------------------------------------------

  logic [PhaseBits-1:0] phi_rand;
  logic [PhaseBits-1:0] phi_seq;

  rope_theta_rom i_rom_rand (
    .pair_idx_i ( pair_idx_i ),
    .phi_o      ( phi_rand   )
  );

  rope_theta_rom i_rom_seq (
    .pair_idx_i ( seq_idx_i ),
    .phi_o      ( phi_seq   )
  );

  // ---------------------------------------------------------------------------
  // Random access: P = m * Phi_i, truncated to PhaseBits.
  //
  // Truncation IS the mod-2^32 reduction, which is exactly the "mod 2*pi is free"
  // property: the phase word is a fixed-point fraction of a full turn, so discarding
  // the overflow discards whole turns and nothing else.
  // ---------------------------------------------------------------------------

  logic [PosBits+PhaseBits-1:0] product;

  assign product = {{PhaseBits{1'b0}}, pos_i} * {{PosBits{1'b0}}, phi_rand};
  assign phase_o = product[PhaseBits-1:0];

  // ---------------------------------------------------------------------------
  // Sequential mode: one exact integer add per token, per pair.
  // ---------------------------------------------------------------------------

  logic [PhaseBits-1:0] acc_q [NumPairs];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int unsigned p = 0; p < NumPairs; p++) acc_q[p] <= '0;
    end else if (seq_clear_i) begin
      acc_q[seq_idx_i] <= '0;
    end else if (seq_step_i) begin
      // Wraparound is the desired behaviour, not an overflow bug.
      acc_q[seq_idx_i] <= acc_q[seq_idx_i] + phi_seq;
    end
  end

  assign seq_phase_o = acc_q[seq_idx_i];

  // ---------------------------------------------------------------------------
  // Field extraction helpers, exported for the LUT stage.
  // ---------------------------------------------------------------------------
  // (Consumers slice phase_o directly; these exist to document the layout in one place
  //  and are checked by the testbench.)

endmodule : rope_phase_gen
