/* ---------------------------------------------------------------------------
 * test_rope.c -- bare-metal test for the BF16 RoPE coprocessor (Milestone 7).
 *
 * Runs on the integrated core (rope_cv32e40x_wrapper). Vectors are compiled in from
 * rope_vectors.h, which model/gen_c_vectors.py emits from the same golden model the RTL
 * testbenches use -- so a pass here means the *software* view (instruction encoding,
 * operand packing, register allocation) agrees with the model too, not just the RTL.
 *
 * Output goes to a memory-mapped putchar address; on a mismatch the test halts with a
 * nonzero exit code so a simulator can detect failure without parsing text.
 *
 * Build:  make -C sw
 * ------------------------------------------------------------------------- */

#include <stdint.h>

#include "rope_intrin.h"
#include "rope_vectors.h"

/* ---------------------------------------------------------------------------
 * Minimal bare-metal I/O.
 *
 * Addresses are overridable so this can be retargeted at a different testbench
 * memory map without editing the source.
 * ------------------------------------------------------------------------- */

#ifndef ROPE_PUTCHAR_ADDR
#define ROPE_PUTCHAR_ADDR 0x10000000u
#endif

#ifndef ROPE_EXIT_ADDR
#define ROPE_EXIT_ADDR 0x20000000u
#endif

static void put_char(char c) {
    *(volatile uint32_t *)ROPE_PUTCHAR_ADDR = (uint32_t)(unsigned char)c;
}

static void put_str(const char *s) {
    while (*s) put_char(*s++);
}

static void put_hex(uint32_t v, int digits) {
    static const char hexd[] = "0123456789abcdef";
    int i;
    for (i = digits - 1; i >= 0; i--) {
        put_char(hexd[(v >> (4 * i)) & 0xF]);
    }
}

static void put_dec(uint32_t v) {
    char buf[11];
    int n = 0;
    if (v == 0) { put_char('0'); return; }
    while (v) { buf[n++] = (char)('0' + (v % 10)); v /= 10; }
    while (n--) put_char(buf[n]);
}

static void test_exit(uint32_t code) {
    *(volatile uint32_t *)ROPE_EXIT_ADDR = code;
    /* If the testbench does not act on the exit write, spin rather than fall through
       into whatever follows in memory. */
    for (;;) { }
}

/* ---------------------------------------------------------------------------
 * Tests
 * ------------------------------------------------------------------------- */

static uint32_t failures = 0;

static void check_u32(const char *what, uint32_t got, uint32_t want, uint32_t idx) {
    if (got != want) {
        failures++;
        if (failures <= 20) {
            put_str("FAIL ");
            put_str(what);
            put_str(" [");
            put_dec(idx);
            put_str("]: got=0x");
            put_hex(got, 8);
            put_str(" want=0x");
            put_hex(want, 8);
            put_str("\n");
        }
    }
}

/* 1. The packing convention must match the RTL exactly.
 *
 * Checked before anything numerical: if x and y were swapped, every later comparison
 * would fail in a confusing way that looks numerical rather than structural. */
static void test_packing(void) {
    uint32_t p = rope_pack(0x1234u, 0xABCDu);
    check_u32("pack", p, 0x1234ABCDu, 0);
    check_u32("unpack_x", rope_unpack_x(p), 0x1234u, 0);
    check_u32("unpack_y", rope_unpack_y(p), 0xABCDu, 0);
    put_str("test_packing done\n");
}

/* 2. BF16 <-> FP32 conversion. Widening must be exact; narrowing must be RNE. */
static void test_conversion(void) {
    /* 1.0f == 0x3F80 in BF16, and widening is a pure shift. */
    check_u32("f32_to_bf16(1.0)", f32_to_bf16(1.0f), 0x3F80u, 0);
    check_u32("f32_to_bf16(-1.0)", f32_to_bf16(-1.0f), 0xBF80u, 0);
    check_u32("f32_to_bf16(2.0)", f32_to_bf16(2.0f), 0x4000u, 0);
    check_u32("f32_to_bf16(0.5)", f32_to_bf16(0.5f), 0x3F00u, 0);

    /* Exact widening: BF16 IS the top 16 bits of FP32. */
    {
        union { uint32_t u; float f; } c;
        c.f = bf16_to_f32(0x3F80u);
        check_u32("bf16_to_f32(0x3F80) bits", c.u, 0x3F800000u, 0);
        c.f = bf16_to_f32(0xC0A0u);
        check_u32("bf16_to_f32(0xC0A0) bits", c.u, 0xC0A00000u, 0);
    }

    /* RNE on an exact tie: the midpoint between 1.0 (0x3F80) and 1.0078125 (0x3F81)
       must round to the EVEN significand, i.e. back to 0x3F80. A truncating
       implementation happens to agree here, so also check a tie that must round UP. */
    {
        union { uint32_t u; float f; } c;
        c.u = 0x3F808000u;  /* exactly halfway between 0x3F80 and 0x3F81 */
        check_u32("RNE tie -> even (down)", f32_to_bf16(c.f), 0x3F80u, 0);
        c.u = 0x3F818000u;  /* halfway between 0x3F81 (odd) and 0x3F82 -> round up */
        check_u32("RNE tie -> even (up)", f32_to_bf16(c.f), 0x3F82u, 0);
    }

    /* A NaN whose payload lives only in the low 16 bits must stay a NaN, not become
       an infinity. */
    {
        union { uint32_t u; float f; } c;
        c.u = 0x7F800001u;
        check_u32("NaN with low payload stays NaN", f32_to_bf16(c.f), 0x7FC0u, 0);
    }
    put_str("test_conversion done\n");
}

/* 3. The instruction itself, against the golden-model vectors. */
static void test_rope_rot(void) {
    uint32_t k;
    for (k = 0; k < ROPE_NUM_VECTORS; k++) {
        uint32_t rs1 = rope_vec_rs1[k];
        uint32_t rs2 = rope_vec_rs2[k];
        uint32_t got = rope_rot(rs1, (uint16_t)(rs2 >> 16), (uint16_t)(rs2 & 0xFFFFu));
        check_u32("rope_rot", got, rope_vec_rd[k], k);
    }
    put_str("test_rope_rot done: ");
    put_dec(ROPE_NUM_VECTORS);
    put_str(" vectors\n");
}

/* 4. Back-to-back dependent instructions.
 *
 * Feeding one ROPE.ROT's result straight into the next exercises the registered-operand
 * path (steps.md Section 9.3 rule 1) and the core's forwarding network together. A
 * rotation by m twice must equal a rotation by 2m, up to BF16 rounding -- so rather than
 * assert bit-equality (which rounding forbids), assert against the model-provided
 * expected chain value. */
static void test_dependent_chain(void) {
    uint32_t acc = rope_vec_rs1[0];
    uint32_t k;
    for (k = 0; k < ROPE_CHAIN_STEPS; k++) {
        acc = rope_rot(acc, ROPE_CHAIN_M, ROPE_CHAIN_I);
    }
    check_u32("dependent chain", acc, ROPE_CHAIN_EXPECTED, 0);
    put_str("test_dependent_chain done\n");
}

/* 5. Whole-head rotation in both conventions. */
static void test_head_vector(void) {
    uint16_t vec[ROPE_HEAD_DIM];
    uint32_t k;

    for (k = 0; k < ROPE_HEAD_DIM; k++) vec[k] = rope_head_in[k];
    rope_rotate_head(vec, ROPE_HEAD_DIM, ROPE_HEAD_M, ROPE_HALF_SPLIT);
    for (k = 0; k < ROPE_HEAD_DIM; k++) {
        check_u32("head half_split", vec[k], rope_head_out_half_split[k], k);
    }

    for (k = 0; k < ROPE_HEAD_DIM; k++) vec[k] = rope_head_in[k];
    rope_rotate_head(vec, ROPE_HEAD_DIM, ROPE_HEAD_M, ROPE_INTERLEAVED);
    for (k = 0; k < ROPE_HEAD_DIM; k++) {
        check_u32("head interleaved", vec[k], rope_head_out_interleaved[k], k);
    }
    put_str("test_head_vector done\n");
}

int main(void) {
    put_str("=== BF16 RoPE coprocessor test ===\n");

    test_packing();
    test_conversion();
    test_rope_rot();
    test_dependent_chain();
    test_head_vector();

    if (failures) {
        put_str("RESULT: FAIL, ");
        put_dec(failures);
        put_str(" mismatches\n");
        test_exit(1);
    }
    put_str("RESULT: PASS\n");
    test_exit(0);
    return 0;
}
