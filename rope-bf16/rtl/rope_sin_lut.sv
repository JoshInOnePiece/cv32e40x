// -----------------------------------------------------------------------------
// rope_sin_lut.sv -- sin and cos from an integer phase word (steps.md Milestone 3).
//
// ONE quarter-wave table serves both functions:
//
//     sin(P)  = quadrant/reflection logic below
//     cos(P)  = sin(P + pi/2) = sin(P + 2^30)      -- same table, phase offset
//
// Two reads of the same (combinational) table give both per cycle.
//
// Quadrant logic (steps.md Section 7.4):
//
//     quad 0:  sin = +T[idx]
//     quad 1:  sin = +T[N-1-idx]
//     quad 2:  sin = -T[idx]
//     quad 3:  sin = -T[N-1-idx]
//
// The reflection index is exactly N-1-idx because the table is MIDPOINT sampled. With
// edge sampling it would be N-idx, which overflows the table at idx=0 and would force
// either a 1025th entry or a special case.
//
// Sign application is a BF16 sign-bit XOR (bit 15) -- exact, no arithmetic, no
// rounding. This is one of the few places BF16's format layout helps directly.
//
// NOTE: this module is purely combinational. It is the caller's job to register the
// outputs if the LUT read is on a critical path.
// -----------------------------------------------------------------------------

module rope_sin_lut
  import rope_pkg::*;
#(
  parameter int unsigned LutBitsP = rope_pkg::LutBits,
  parameter int unsigned LutNP    = rope_pkg::LutN
) (
  input  logic [PhaseBits-1:0] phase_i,
  output logic [15:0]          sin_o,
  output logic [15:0]          cos_o
);

  // cos(x) = sin(x + pi/2). Wraparound is intended and free.
  logic [PhaseBits-1:0] phase_cos;
  assign phase_cos = phase_i + QuarterTurn;

  // ---------------------------------------------------------------------------
  // One lookup path, instantiated twice
  // ---------------------------------------------------------------------------

  logic [15:0] sin_raw, cos_raw;

  rope_sin_lut_path #(
    .LutBitsP ( LutBitsP ),
    .LutNP    ( LutNP    )
  ) i_path_sin (
    .phase_i ( phase_i ),
    .val_o   ( sin_raw )
  );

  rope_sin_lut_path #(
    .LutBitsP ( LutBitsP ),
    .LutNP    ( LutNP    )
  ) i_path_cos (
    .phase_i ( phase_cos ),
    .val_o   ( cos_raw   )
  );

  assign sin_o = sin_raw;
  assign cos_o = cos_raw;

endmodule : rope_sin_lut


// -----------------------------------------------------------------------------
// rope_sin_lut_path -- one phase word in, one signed BF16 sine value out.
// -----------------------------------------------------------------------------

module rope_sin_lut_path
  import rope_pkg::*;
#(
  parameter int unsigned LutBitsP = rope_pkg::LutBits,
  parameter int unsigned LutNP    = rope_pkg::LutN
) (
  input  logic [PhaseBits-1:0] phase_i,
  output logic [15:0]          val_o
);

  // Field extraction. The low IdxShift bits of the phase are deliberately ignored
  // (reserved for future interpolation).
  logic [QuadBits-1:0]    quad;
  logic [LutBitsP-1:0]    idx;
  logic [LutBitsP-1:0]    idx_eff;
  logic                   reflect;
  logic                   negate;
  logic [15:0]            tab_val;

  assign quad = phase_i[PhaseBits-1 -: QuadBits];          // phase[31:30]
  assign idx  = phase_i[IdxShift +: LutBitsP];             // phase[29:20]

  // Quadrants 1 and 3 read the table backwards.
  assign reflect = (quad == 2'd1) || (quad == 2'd3);
  // Quadrants 2 and 3 are negative.
  assign negate  = quad[1];

  // Exactly N-1-idx thanks to midpoint sampling. With LutNP a power of two this is a
  // bitwise complement, so it costs nothing.
  assign idx_eff = reflect ? (LutBitsP'(LutNP - 1) - idx) : idx;

  rope_sin_rom #(
    .LutBits ( LutBitsP ),
    .LutN    ( LutNP    )
  ) i_rom (
    .idx_i ( idx_eff ),
    .sin_o ( tab_val )
  );

  // Sign application: XOR on the BF16 sign bit. Exact.
  //
  // Note this means sin(phase) for a phase in quadrant 2 or 3 that reads T[k]==0 would
  // produce -0. The table is midpoint sampled so it contains no exact zero, and -0 is
  // in any case the numerically correct signed zero here.
  assign val_o = {tab_val[15] ^ negate, tab_val[14:0]};

endmodule : rope_sin_lut_path
