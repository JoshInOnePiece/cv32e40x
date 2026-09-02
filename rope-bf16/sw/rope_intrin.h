/* ---------------------------------------------------------------------------
 * rope_intrin.h -- inline-asm wrappers for the BF16 RoPE coprocessor.
 *
 * ROPE.ROT rd, rs1, rs2       # custom-0, opcode 0x0B, funct3=0, funct7=0
 *   rs1 = {x_bf16[31:16], y_bf16[15:0]}    native BF16 pair, NO conversion
 *   rs2 = {m[31:16], i[15:0]}              position and pair index (integers)
 *   rd  = {x'_bf16[31:16], y'_bf16[15:0]}  native BF16 pair out
 *
 * The packing MUST match rope_pkg.sv (pair_x/pair_y/pack_pair): x lives in the HIGH
 * half of the word. Getting this backwards is the single easiest way to produce
 * "correct-looking but wrong" results, so it is stated in both places and checked by
 * sw/test_rope.c.
 *
 * Later: patch binutils/LLVM with the encoding so real mnemonics replace `.insn`.
 * Not needed to make progress.
 * ------------------------------------------------------------------------- */

#ifndef ROPE_INTRIN_H
#define ROPE_INTRIN_H

#include <stdint.h>

/* Instruction fields, kept in sync with rope_pkg.sv. */
#define ROPE_OPCODE_CUSTOM0 0x0B
#define ROPE_FUNCT3_ROT     0
#define ROPE_FUNCT7_ROT     0

/* ---------------------------------------------------------------------------
 * BF16 <-> FP32 conversion
 *
 * BF16 is bit-identical to the upper 16 bits of FP32, so widening is EXACT and needs
 * no library call and no rounding. Narrowing is the only direction that rounds.
 * ------------------------------------------------------------------------- */

/* Exact BF16 -> float. No rounding whatsoever. */
static inline float bf16_to_f32(uint16_t b) {
    union { uint32_t u; float f; } c;
    c.u = (uint32_t)b << 16;
    return c.f;
}

/* float -> BF16, round-to-nearest-even.
 *
 * RNE, not truncation: truncating gives a systematic downward bias that shows up much
 * later as an unexplained perplexity gap. NaN is canonicalised rather than shifted,
 * because a NaN whose payload lives entirely in the low 16 bits would otherwise become
 * an infinity. This mirrors round_f32_to_bf16() in model/bf16.py exactly. */
static inline uint16_t f32_to_bf16(float f) {
    union { uint32_t u; float f; } c;
    uint32_t u, lsb;
    c.f = f;
    u = c.u;

    /* NaN -> canonical quiet NaN. */
    if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu) != 0u) {
        return 0x7FC0u;
    }

    lsb = (u >> 16) & 1u;
    return (uint16_t)((u + 0x7FFFu + lsb) >> 16);
}

/* ---------------------------------------------------------------------------
 * Pair packing -- must match rope_pkg::pack_pair / pair_x / pair_y
 * ------------------------------------------------------------------------- */

static inline uint32_t rope_pack(uint16_t x, uint16_t y) {
    return ((uint32_t)x << 16) | (uint32_t)y;
}

static inline uint16_t rope_unpack_x(uint32_t xy) { return (uint16_t)(xy >> 16); }
static inline uint16_t rope_unpack_y(uint32_t xy) { return (uint16_t)(xy & 0xFFFFu); }

/* ---------------------------------------------------------------------------
 * The instruction
 * ------------------------------------------------------------------------- */

/* Rotate one BF16 pair at sequence position m, frequency index i.
 *
 * `xy` is a packed BF16 pair {x, y}; the result is the packed rotated pair {x', y'}.
 *
 *     x' = x*cos(m*theta_i) - y*sin(m*theta_i)
 *     y' = x*sin(m*theta_i) + y*cos(m*theta_i)
 */
static inline uint32_t rope_rot(uint32_t xy, uint16_t m, uint16_t i) {
    uint32_t rd;
    uint32_t rs2 = ((uint32_t)m << 16) | (uint32_t)i;
    /* `__asm__ __volatile__` rather than `asm volatile`: under -std=c11 (strict ISO)
       the unprefixed spellings are not keywords. */
    __asm__ __volatile__(".insn r %[opc], %[f3], %[f7], %[rd], %[rs1], %[rs2]"
                         : [rd] "=r"(rd)
                         : [rs1] "r"(xy), [rs2] "r"(rs2),
                           [opc] "i"(ROPE_OPCODE_CUSTOM0),
                           [f3] "i"(ROPE_FUNCT3_ROT),
                           [f7] "i"(ROPE_FUNCT7_ROT));
    return rd;
}

/* Convenience wrapper: rotate a pair given as floats, returning floats.
 *
 * Note this rounds the inputs to BF16 on the way in. The coprocessor itself performs no
 * conversion at all (INV-1) -- the rounding happens here, in software, and is visible. */
static inline void rope_rot_f32(float x, float y, uint16_t m, uint16_t i,
                                float *x_out, float *y_out) {
    uint32_t packed = rope_pack(f32_to_bf16(x), f32_to_bf16(y));
    uint32_t res    = rope_rot(packed, m, i);
    *x_out = bf16_to_f32(rope_unpack_x(res));
    *y_out = bf16_to_f32(rope_unpack_y(res));
}

/* ---------------------------------------------------------------------------
 * Whole-vector application
 *
 * Both pairing conventions from steps.md Section 12. Default to half-split, because
 * HuggingFace Transformers (which Mugi's authors used) uses rotate_half.
 * ------------------------------------------------------------------------- */

typedef enum {
    ROPE_HALF_SPLIT = 0,  /* (x_i, x_{i+d/2})  -- HF LLaMA / GPT-NeoX  [DEFAULT] */
    ROPE_INTERLEAVED = 1  /* (x_2i, x_2i+1)    -- GPT-J / original RoFormer */
} rope_stride_mode_t;

/* Rotate one head vector of `d` BF16 elements in place.
 *
 * `vec` points to d uint16_t BF16 values. `d` must be even.
 *
 * Under half-split the two elements of a pair are d/2 apart -- for a 64-element head
 * that is two separate cache lines per pair. Interleaved gets both elements of a pair in
 * one 32-bit word, which is why X_MEM_WIDTH = 32 lines up perfectly for interleaved and
 * needs two transactions for half-split.
 */
static inline void rope_rotate_head(uint16_t *vec, unsigned d, uint16_t m,
                                    rope_stride_mode_t mode) {
    unsigned half = d / 2;
    unsigned k;
    for (k = 0; k < half; k++) {
        unsigned lo, hi;
        uint32_t res;
        if (mode == ROPE_HALF_SPLIT) {
            lo = k;
            hi = k + half;
        } else {
            lo = 2 * k;
            hi = 2 * k + 1;
        }
        res = rope_rot(rope_pack(vec[lo], vec[hi]), m, (uint16_t)k);
        vec[lo] = rope_unpack_x(res);
        vec[hi] = rope_unpack_y(res);
    }
}

#endif /* ROPE_INTRIN_H */
