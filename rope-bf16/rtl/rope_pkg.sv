// -----------------------------------------------------------------------------
// rope_pkg.sv -- parameters, types and opcode constants for the BF16 RoPE unit.
//
// Design invariants this package encodes (see steps.md Section 0):
//   INV-1  the arithmetic datapath is BF16 only, RNE at every node, no conversions
//   INV-2  the phase index is an INTEGER -- addressing, not arithmetic
//   INV-3  no approximation of the interface contract
//   INV-4  rotate before quantize
// -----------------------------------------------------------------------------

package rope_pkg;

  // ---------------------------------------------------------------------------
  // BF16 format (INV-1)
  // ---------------------------------------------------------------------------

  // bfloat16: 1 sign, 8 exponent, 7 stored mantissa bits. Bit-identical to the upper
  // 16 bits of IEEE binary32, which is why no conversion stage is needed anywhere.
  localparam int unsigned Bf16Width   = 16;
  localparam int unsigned Bf16ExpBits = 8;
  localparam int unsigned Bf16ManBits = 7;

  localparam logic [15:0] Bf16PosZero = 16'h0000;
  localparam logic [15:0] Bf16PosInf  = 16'h7F80;
  localparam logic [15:0] Bf16QNaN    = 16'h7FC0;  // canonical quiet NaN
  localparam logic [15:0] Bf16One     = 16'h3F80;

  // In CVFPU/FPnew nomenclature bfloat16 is FP16ALT (format index 4). Grepping the
  // vendored sources for "bf16" finds nothing -- see steps.md Section 5.1.
  // FpFmtMask is `logic [0:4]`, so FP16ALT-only is 5'b00001.
  localparam logic [0:4] FpFmtMaskBf16Only = 5'b00001;
  localparam logic [0:3] IntFmtMaskNone    = 4'b0000;

  // ---------------------------------------------------------------------------
  // Integer phase (INV-2)
  // ---------------------------------------------------------------------------

  // 32-bit phase word P, representing angle = 2*pi * P / 2^32.
  //
  //   P[31:30] = quadrant  (2 bits)
  //   P[29:20] = LUT index (10 bits -> 1024 entries per quadrant)
  //   P[19:0]  = unused (reserved for future interpolation)
  //
  // 32 bits rather than 24: at 32 bits the slowest frequency (theta ~ 1e-4) still gets
  // Phi_min ~ 79000 (relative quantisation ~1e-5), versus Phi_min ~ 267 (~0.4%) at 24
  // bits. 32 is also the natural width on RV32. `mod 2*pi` is free (wraparound).
  localparam int unsigned PhaseBits = 32;
  localparam int unsigned QuadBits  = 2;
  localparam int unsigned LutBits   = 10;
  localparam int unsigned LutN      = 1 << LutBits;          // 1024
  localparam int unsigned QuadShift = PhaseBits - QuadBits;   // 30
  localparam int unsigned IdxShift  = QuadShift - LutBits;    // 20

  // cos(x) = sin(x + pi/2) -> a phase offset of exactly one quarter turn.
  // ONE table, two lookups (steps.md Section 7.2).
  localparam logic [PhaseBits-1:0] QuarterTurn = 1 << QuadShift;  // 2^30

  // ---------------------------------------------------------------------------
  // Head geometry
  // ---------------------------------------------------------------------------

  localparam int unsigned HeadDimMax  = 128;
  localparam int unsigned NumPairsMax = HeadDimMax / 2;             // 64
  localparam int unsigned PairIdxBits = $clog2(NumPairsMax);        // 6

  // Sequence position m.
  //
  // 16 bits, matching the rs2[31:16] field of ROPE.ROT exactly. steps.md Section 6.3
  // suggests 13 bits ("m is ~13 bits ... a 13x32 multiplier, cheap"), and 13 is indeed
  // enough for an 8192-token context -- but the *instruction* carries 16 bits, so a
  // 13-bit register silently truncates: m=10000 would be taken as 10000 & 0x1FFF = 1808
  // and produce a confidently wrong rotation with no error anywhere. Since 8192-token
  // contexts are ordinary, that failure is reachable in normal use.
  //
  // Three extra multiplier bits (16x32 instead of 13x32) is a negligible price for
  // removing a silent-wrongness class, so the width follows the ISA rather than the
  // expected sequence length. Narrow it only if you also range-check m.
  localparam int unsigned PosBits = 16;

  // ---------------------------------------------------------------------------
  // Pairing convention (steps.md Section 12)
  // ---------------------------------------------------------------------------
  //
  // The two conventions are equivalent up to a fixed permutation of the head dimension,
  // so trained weights port between them -- but mixing them up makes the error look like
  // garbage rather than like a convention mismatch. llama.cpp carries an explicit NEOX
  // flag; so do we.
  //
  // Default is HALF_SPLIT because HuggingFace Transformers (which Mugi's authors used
  // for all their models) uses rotate_half.
  typedef enum logic {
    STRIDE_HALF_SPLIT  = 1'b0,  // (x_i, x_{i+d/2})  -- HF LLaMA / GPT-NeoX  [DEFAULT]
    STRIDE_INTERLEAVED = 1'b1   // (x_2i, x_2i+1)    -- GPT-J / original RoFormer
  } stride_mode_e;

  // ---------------------------------------------------------------------------
  // Accumulator width (steps.md Section 8.3)
  // ---------------------------------------------------------------------------
  //
  // 16 = strict BF16 (DEFAULT, per INV-1).
  // 32 = wide internal accumulate, MEASUREMENT ONLY -- exists solely to quantify the
  //      error of the strict-BF16 choice for the writeup. Do not ship it enabled.
  localparam int unsigned AccWidthStrict = 16;
  localparam int unsigned AccWidthWide   = 32;

  // ---------------------------------------------------------------------------
  // Instruction encoding (steps.md Section 9.1)
  // ---------------------------------------------------------------------------
  //
  //   ROPE.ROT rd, rs1, rs2      # custom-0
  //     rs1 = {x_bf16[15:0], y_bf16[15:0]}     native BF16 pair, NO conversion
  //     rs2 = {m[15:0], i[15:0]}               position and pair index (integers)
  //     rd  = {x'_bf16[15:0], y'_bf16[15:0]}   native BF16 pair out
  //
  // BF16 packs two-per-GPR on RV32, so a single source pair and a single destination
  // suffice: no dual writeback, X_RFW_WIDTH stays 32.
  //
  // The opcode is a parameter so it can be reassigned if it ever collides with another
  // coprocessor -- an explicit CV-X-IF recommendation.
  localparam logic [6:0] RopeOpcodeCustom0 = 7'b0001011;  // 0x0B, RISC-V custom-0
  localparam logic [2:0] RopeFunct3Rot     = 3'b000;
  localparam logic [6:0] RopeFunct7Rot     = 7'b0000000;

  // R-type field extraction helpers.
  function automatic logic [6:0] instr_opcode(logic [31:0] instr);
    return instr[6:0];
  endfunction

  function automatic logic [2:0] instr_funct3(logic [31:0] instr);
    return instr[14:12];
  endfunction

  function automatic logic [6:0] instr_funct7(logic [31:0] instr);
    return instr[31:25];
  endfunction

  function automatic logic [4:0] instr_rd(logic [31:0] instr);
    return instr[11:7];
  endfunction

  // Decode: is this a ROPE.ROT?
  function automatic logic is_rope_rot(logic [31:0] instr);
    return (instr_opcode(instr) == RopeOpcodeCustom0) &&
           (instr_funct3(instr) == RopeFunct3Rot)     &&
           (instr_funct7(instr) == RopeFunct7Rot);
  endfunction

  // ---------------------------------------------------------------------------
  // Operand packing helpers
  // ---------------------------------------------------------------------------
  //
  // A BF16 pair lives in one 32-bit GPR as {x, y} with x in the HIGH half. Keeping this
  // in one place stops the halves being swapped somewhere between the RTL, the
  // intrinsics header and the golden model.

  function automatic logic [15:0] pair_x(logic [31:0] packed_pair);
    return packed_pair[31:16];
  endfunction

  function automatic logic [15:0] pair_y(logic [31:0] packed_pair);
    return packed_pair[15:0];
  endfunction

  function automatic logic [31:0] pack_pair(logic [15:0] x, logic [15:0] y);
    return {x, y};
  endfunction

  // rs2 = {m[15:0], i[15:0]}
  function automatic logic [15:0] rs2_pos(logic [31:0] rs2);
    return rs2[31:16];
  endfunction

  function automatic logic [15:0] rs2_pair_idx(logic [31:0] rs2);
    return rs2[15:0];
  endfunction

endpackage : rope_pkg
