#!/usr/bin/env python3
"""Milestone 0 exit criteria for the RoPE golden model.

Run:  python3 model/test_rope_ref.py

Covers the correctness half of Milestone 0. The *quantitative* error budget (Model A vs
Model C, strict vs wide accumulate, the cancellation study) lives in characterize.py so
that the numbers land in one reviewable report rather than in pass/fail assertions.
"""

import sys

import numpy as np

from bf16 import bf16_to_f32, is_nan, to_bf16
from rope_ref import (
    DEFAULT_BASE,
    IDX_SHIFT,
    LUT,
    LUT_N,
    PHASE_MASK,
    QUAD_SHIFT,
    QUARTER_TURN,
    hf_apply_rotary_pos_emb,
    lut_cos,
    lut_sin,
    pair_indices,
    permutation_interleaved_to_half_split,
    phase_of,
    phi_table,
    rope_exact,
    rope_strict_bf16,
    rope_vector_exact,
    rope_vector_strict_bf16,
)

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# INV-2: the numerical proof that the phase cannot be BF16
# ---------------------------------------------------------------------------


def _bf16_spacing(v):
    """Distance from BF16(v) to its next representable neighbour."""
    a = to_bf16(v)
    nxt = np.uint16(int(a) + 1)
    return float(bf16_to_f32(nxt)) - float(bf16_to_f32(a))


def test_inv2_bf16_phase_is_unrepresentable():
    """Demonstrate, not assert on faith, that a BF16 angle is useless at large m.

    For i=0, theta_0 = 1.0 rad, so the angle at position m is m radians.

    NOTE ON steps.md INV-2: the plan states that at m=1000 "the representable-value
    spacing exceeds a full period of 2*pi", citing 1000*2^-8 ~= 3.9 rad. The measured
    BF16 spacing at 1000 is 4.0 rad, which exceeds *pi* (a half period) but not 2*pi.
    The full-period claim becomes true at m >= 1024, where the spacing is 8.0 rad.
    The invariant itself is unaffected -- 4 rad of quantisation on an angle already
    destroys all correlation with the true value -- but the example is off by ~1.6x.
    Both the correct weaker claim at m=1000 and the strong claim at m=1024 are asserted
    here so the writeup quotes a number that survives checking.
    """
    sp1000 = _bf16_spacing(1000.0)
    check(
        f"m=1000: BF16 spacing {sp1000:.3f} rad > pi ({np.pi:.3f}) -- half a period lost",
        sp1000 > np.pi,
        f"spacing={sp1000}",
    )
    sp1024 = _bf16_spacing(1024.0)
    check(
        f"m=1024: BF16 spacing {sp1024:.3f} rad > 2*pi ({2*np.pi:.3f}) -- a full period",
        sp1024 > 2 * np.pi,
        f"spacing={sp1024}",
    )

    # The consequence that actually matters: the induced angle error is O(1) radians,
    # so cos/sin of the rounded angle are uncorrelated with the true values.
    worst_angle_err = sp1000 / 2
    check(
        f"induced angle error at m=1000 is {worst_angle_err:.3f} rad (>= 1 rad: garbage)",
        worst_angle_err >= 1.0,
    )

    # Argument reduction budget from steps.md: log2(4096) + 10 ~= 22 bits needed,
    # BF16 supplies 8 significand bits.
    need = np.log2(4096) + 10
    check(f"m<=4096 with ~10-bit angle precision needs ~{need:.0f} bits > BF16's 8", need > 8)

    # And the integer accumulator has no such problem: exact for all m.
    phi = phi_table(128)
    m = 1000
    got = int(phase_of(m, 0, phi))
    want = (m * int(phi[0])) & PHASE_MASK
    check("integer phase exact at m=1000", got == want)


# ---------------------------------------------------------------------------
# Phase generator (mirrors Milestone 2 exit criteria)
# ---------------------------------------------------------------------------


def test_phase_table():
    phi = phi_table(128, DEFAULT_BASE)
    check("phi_table(128) has 64 entries", len(phi) == 64)

    # Phi_0 corresponds to theta=1.0 rad -> 2^32/(2*pi).
    want0 = round((1 << 32) / (2 * np.pi))
    check(f"Phi_0 == round(2^32/2pi) == {want0}", int(phi[0]) == want0)

    # Slowest frequency: theta ~ 1e-4 -> Phi ~ 68000, the sizing argument for 32 bits.
    check(
        f"Phi_min = {int(phi[-1])} (>= 60000, justifies 32-bit phase over 24-bit)",
        int(phi[-1]) >= 60000,
    )
    check("Phi strictly decreasing with i", np.all(np.diff(phi.astype(np.int64)) < 0))


def test_phase_wraparound():
    """Wraparound must be exact modular arithmetic across the 2^32 boundary."""
    phi = phi_table(128)
    # Pick m values that straddle multiples of 2^32 for the fastest frequency.
    p0 = int(phi[0])
    ms = [(1 << 32) // p0 + k for k in (-2, -1, 0, 1, 2)]
    ok = True
    for m in ms:
        got = int(phase_of(m, 0, phi))
        want = (m * p0) & PHASE_MASK
        if got != want:
            ok = False
    check(f"phase wraps exactly at m in {ms}", ok)

    # Sequential accumulation must equal random access -- zero drift (steps.md 6.3).
    acc = 0
    ok_seq = True
    for m in range(0, 8193):
        if int(phase_of(m, 3, phi)) != acc:
            ok_seq = False
            break
        acc = (acc + int(phi[3])) & PHASE_MASK
    check("sequential accumulate == random access for m in [0,8192]", ok_seq)


# ---------------------------------------------------------------------------
# Sine LUT (mirrors Milestone 3 exit criteria)
# ---------------------------------------------------------------------------


def test_lut_sizing_argument():
    """The table must be finer than BF16 can represent (steps.md 7.1)."""
    ang_res = (np.pi / 2) / LUT_N
    bf16_ulp_near_1 = 2.0**-7
    check(
        f"LUT angle resolution {ang_res:.5f} < BF16 ulp near 1.0 ({bf16_ulp_near_1:.5f})",
        ang_res < bf16_ulp_near_1,
    )
    check(f"LUT is {LUT_N} entries x 2 bytes = {LUT_N*2/1024:.0f} KB", LUT_N * 2 == 2048)


def test_lut_reflection_symmetry():
    """Q1/Q3 reflection must be exactly LUT_N-1-idx, with no endpoint special case.

    This is the payoff of midpoint sampling. Test it at the quadrant boundaries
    specifically, where edge sampling would have overflowed the table.
    """
    # sin(pi/2 - t) == cos(t): compare across a quadrant boundary using phase words.
    # phase p in Q0 and its mirror (QUARTER_TURN*2 - 1 - p) ... use the index algebra
    # directly: for idx k in Q0, Q1 at mirrored position must read LUT[LUT_N-1-k].
    ok = True
    for k in (0, 1, 2, LUT_N // 2, LUT_N - 2, LUT_N - 1):
        p_q0 = (0 << QUAD_SHIFT) | (k << IDX_SHIFT)
        p_q1 = (1 << QUAD_SHIFT) | (k << IDX_SHIFT)
        got_q1 = int(lut_sin(np.uint32(p_q1)))
        want_q1 = int(LUT[LUT_N - 1 - k])
        if got_q1 != want_q1:
            ok = False
        # Q2 is the exact sign flip of Q0, Q3 of Q1.
        p_q2 = (2 << QUAD_SHIFT) | (k << IDX_SHIFT)
        if int(lut_sin(np.uint32(p_q2))) != (int(lut_sin(np.uint32(p_q0))) ^ 0x8000):
            ok = False
    check("Q1 reflection == LUT[1023-idx]; Q2 == sign-flip of Q0 (incl. idx 0 and 1023)", ok)

    # No index can ever escape the table -- the thing edge sampling would break.
    idx_all = (np.arange(LUT_N, dtype=np.uint32))
    ridx = np.uint32(LUT_N - 1) - idx_all
    check("reflected index stays in [0, 1023] for all idx", bool(np.all(ridx < LUT_N)))


def test_lut_cos_is_sin_plus_quarter_turn():
    rng = np.random.default_rng(3)
    p = rng.integers(0, 1 << 32, size=100_000, dtype=np.uint64).astype(np.uint32)
    lhs = lut_cos(p)
    rhs = lut_sin(((p.astype(np.uint64) + QUARTER_TURN) & PHASE_MASK).astype(np.uint32))
    check("cos(P) == sin(P + 2^30), one table two lookups", np.array_equal(lhs, rhs))


def test_lut_accuracy_vs_true_sin():
    """LUT output must be within the expected quantisation + BF16 rounding budget."""
    rng = np.random.default_rng(5)
    p = rng.integers(0, 1 << 32, size=200_000, dtype=np.uint64).astype(np.uint32)
    ang = 2 * np.pi * (p.astype(np.float64) / 2.0**32)

    got_s = bf16_to_f32(lut_sin(p)).astype(np.float64)
    got_c = bf16_to_f32(lut_cos(p)).astype(np.float64)

    err_s = np.abs(got_s - np.sin(ang))
    err_c = np.abs(got_c - np.cos(ang))

    # Budget: half a LUT bucket (|d sin/d theta| <= 1) plus half a BF16 ulp near 1.
    budget = 0.5 * (np.pi / 2) / LUT_N + 0.5 * 2.0**-7 + 1e-9
    check(
        f"max |lut_sin - sin| = {err_s.max():.6f} <= budget {budget:.6f}",
        err_s.max() <= budget,
    )
    check(
        f"max |lut_cos - cos| = {err_c.max():.6f} <= budget {budget:.6f}",
        err_c.max() <= budget,
    )
    print(f"        RMS sin err = {np.sqrt(np.mean(err_s**2)):.6f}")


def test_sin2_plus_cos2():
    """Documented, not asserted to be 1.0 -- it cannot be, in BF16 (steps.md 7.5)."""
    rng = np.random.default_rng(11)
    p = rng.integers(0, 1 << 32, size=200_000, dtype=np.uint64).astype(np.uint32)
    s = bf16_to_f32(lut_sin(p)).astype(np.float64)
    c = bf16_to_f32(lut_cos(p)).astype(np.float64)
    n = s * s + c * c
    dev = np.abs(n - 1.0)
    print(
        f"        sin^2+cos^2: max dev = {dev.max():.6f}, "
        f"mean = {np.mean(n):.6f}, RMS dev = {np.sqrt(np.mean(dev**2)):.6f}"
    )
    # Expected to deviate by roughly a BF16 ulp; flag only if it is wildly off, which
    # would indicate a real bug (bad reflection, wrong quadrant sign) rather than
    # expected quantisation.
    check(f"sin^2+cos^2 within 4 BF16 ulp of 1.0 (max dev {dev.max():.5f})", dev.max() < 4 * 2.0**-7)


# ---------------------------------------------------------------------------
# Model A behaviour
# ---------------------------------------------------------------------------


def test_model_a_deterministic():
    phi = phi_table(128)
    rng = np.random.default_rng(21)
    x = rng.integers(0, 1 << 16, size=5000, dtype=np.uint16)
    y = rng.integers(0, 1 << 16, size=5000, dtype=np.uint16)
    m = rng.integers(0, 4096, size=5000)
    i = rng.integers(0, 64, size=5000)
    a1 = rope_strict_bf16(x, y, m, i, phi)
    a2 = rope_strict_bf16(x, y, m, i, phi)
    check(
        "Model A reproduces itself bit-exactly",
        np.array_equal(a1[0], a2[0]) and np.array_equal(a1[1], a2[1]),
    )


def test_model_a_identity_at_m0():
    """At m=0 the rotation is near-identity, but deliberately NOT exactly the identity.

    This is the documented cost of midpoint sampling (steps.md 7.3): the table stores
    bucket *centres*, so sin(0) reads T[0] = sin(0.5 * (pi/2)/1024) ~= 7.67e-4 rather
    than 0, and cos(0) reads T[1023] which does round to exactly 1.0.

    Consequence worth recording: the absolute error in x' is bounded by |y|*sin(0),
    so the *relative* error in x' is amplified by |y|/|x| and is NOT bounded by a
    small constant. With x=0.25, y=8.0 the relative error is ~2.3%.

    Midpoint sampling buys exact reflection symmetry (no 1025th entry, no endpoint
    special case) and halves the mean sampling error; the price is that phase 0 and
    phase pi/2 are never exact. See docs/notes.md.
    """
    phi = phi_table(128)
    c0 = float(bf16_to_f32(lut_cos(np.uint32(0))))
    s0 = float(bf16_to_f32(lut_sin(np.uint32(0))))
    print(f"        at m=0: cos={c0:.8f} (ideal 1.0), sin={s0:.8f} (ideal 0.0)")
    check("cos(0) rounds to exactly 1.0", c0 == 1.0)
    half_bucket = 0.5 * (np.pi / 2) / LUT_N
    check(
        f"sin(0) == {s0:.2e}, within a BF16 rounding of the half-bucket {half_bucket:.2e}",
        abs(s0 - half_bucket) < 2.0**-7 * half_bucket + 1e-9,
    )

    x = to_bf16([1.0, 2.0, -3.5, 0.25])
    y = to_bf16([0.5, -1.0, 2.0, 8.0])
    i = np.array([0, 1, 2, 3])
    xr, yr = rope_strict_bf16(x, y, 0, i, phi)

    xf = bf16_to_f32(x).astype(np.float64)
    yf = bf16_to_f32(y).astype(np.float64)
    abserr = np.abs(bf16_to_f32(xr).astype(np.float64) - xf)

    # The absolute error must be explained by the |y|*sin(0) offset plus one BF16
    # rounding of the result -- that is the real, bounded property.
    budget = np.abs(yf) * s0 + 2.0**-8 * np.maximum(np.abs(xf), 1e-30) * 2
    check(
        "m=0 absolute error in x' is explained by |y|*sin(0) + one BF16 rounding",
        bool(np.all(abserr <= budget)),
        f"abserr={abserr}, budget={budget}",
    )
    relerr = abserr / np.abs(xf)
    print(
        f"        m=0 rel err in x': max {relerr.max():.5f} "
        f"(amplified by |y|/|x|, worst case here {np.max(np.abs(yf)/np.abs(xf)):.1f}x)"
    )
    # With a balanced ratio the rotation really is near-identity.
    bal = np.abs(yf) <= np.abs(xf)
    check(
        f"m=0 rel err < 0.5% when |y| <= |x| (max {relerr[bal].max():.5f})",
        relerr[bal].max() < 0.005,
    )


def test_model_a_specials_propagate():
    """NaN/Inf must propagate rather than silently becoming finite numbers."""
    phi = phi_table(128)
    nan = np.uint16(0x7FC0)
    inf = np.uint16(0x7F80)
    one = to_bf16(1.0)

    xr, yr = rope_strict_bf16(np.array([nan]), np.array([one]), 5, np.array([0]), phi)
    check("NaN input -> NaN outputs", bool(is_nan(xr)[0]) and bool(is_nan(yr)[0]))

    xr, yr = rope_strict_bf16(np.array([inf]), np.array([one]), 5, np.array([0]), phi)
    finite_free = (int(xr[0]) & 0x7F80) == 0x7F80 and (int(yr[0]) & 0x7F80) == 0x7F80
    check("Inf input -> Inf or NaN outputs (not finite)", finite_free)


# ---------------------------------------------------------------------------
# Section 12: the pairing-convention trap
# ---------------------------------------------------------------------------


def test_half_split_matches_huggingface():
    """Our half-split FP64 path must match HF's rotate_half formulation exactly."""
    d = 128
    rng = np.random.default_rng(31)
    vec = rng.integers(0, 1 << 16, size=d, dtype=np.uint16)
    # Keep it to sane finite values so the comparison is about convention, not specials.
    vals = rng.uniform(-4, 4, size=d)
    vec = to_bf16(vals)

    for m in (0, 1, 7, 100, 1000, 4095):
        ours = rope_vector_exact(vec, m, d=d, convention="half_split")
        theirs = hf_apply_rotary_pos_emb(bf16_to_f32(vec).astype(np.float64), m)
        if not np.allclose(ours, theirs, rtol=0, atol=1e-12):
            check(f"half_split == HF apply_rotary_pos_emb at m={m}", False,
                  f"max diff {np.abs(ours-theirs).max():.3e}")
            return
    check("half_split == HF apply_rotary_pos_emb (m in 0,1,7,100,1000,4095)", True)


def test_conventions_related_by_permutation():
    """Interleaved and half-split must differ by exactly the fixed permutation."""
    d = 64
    rng = np.random.default_rng(41)
    v_inter = to_bf16(rng.uniform(-4, 4, size=d))
    perm = permutation_interleaved_to_half_split(d)
    phi = phi_table(d)

    m = 137
    out_inter = rope_vector_strict_bf16(v_inter, m, phi, convention="interleaved")
    out_hs = rope_vector_strict_bf16(v_inter[perm], m, phi, convention="half_split")
    check(
        "rope_interleaved(v)[perm] == rope_half_split(v[perm]), bit-exact",
        np.array_equal(out_inter[perm], out_hs),
    )

    # And confirm the two are genuinely different in place -- i.e. the trap is real.
    out_hs_same_layout = rope_vector_strict_bf16(v_inter, m, phi, convention="half_split")
    check(
        "the two conventions really do differ on the same layout (trap is real)",
        not np.array_equal(out_inter, out_hs_same_layout),
    )


def test_pair_indices_cover_vector():
    for conv in ("half_split", "interleaved"):
        lo, hi, pi_ = pair_indices(128, conv)
        touched = np.concatenate([lo, hi])
        check(
            f"{conv}: pairs cover all 128 elements exactly once",
            len(np.unique(touched)) == 128 and touched.max() == 127,
        )
        check(f"{conv}: frequency index runs 0..63", np.array_equal(pi_, np.arange(64)))


def test_rotation_preserves_norm_approximately():
    """A rotation must preserve the pair norm; large drift would mean a real bug."""
    phi = phi_table(128)
    rng = np.random.default_rng(51)
    x = to_bf16(rng.uniform(-4, 4, size=20000))
    y = to_bf16(rng.uniform(-4, 4, size=20000))
    m = rng.integers(0, 4096, size=20000)
    i = rng.integers(0, 64, size=20000)

    xr, yr = rope_strict_bf16(x, y, m, i, phi)
    n_in = bf16_to_f32(x).astype(np.float64) ** 2 + bf16_to_f32(y).astype(np.float64) ** 2
    n_out = bf16_to_f32(xr).astype(np.float64) ** 2 + bf16_to_f32(yr).astype(np.float64) ** 2
    rel = np.abs(np.sqrt(n_out) - np.sqrt(n_in)) / np.sqrt(n_in)
    print(f"        pair-norm rel err: max {rel.max():.5f}, RMS {np.sqrt(np.mean(rel**2)):.5f}")
    check(f"norm preserved to <5% worst case (max {rel.max():.4f})", rel.max() < 0.05)


def main():
    print("=== RoPE golden model tests (Milestone 0) ===")
    test_inv2_bf16_phase_is_unrepresentable()
    test_phase_table()
    test_phase_wraparound()
    test_lut_sizing_argument()
    test_lut_reflection_symmetry()
    test_lut_cos_is_sin_plus_quarter_turn()
    test_lut_accuracy_vs_true_sin()
    test_sin2_plus_cos2()
    test_model_a_deterministic()
    test_model_a_identity_at_m0()
    test_model_a_specials_propagate()
    test_half_split_matches_huggingface()
    test_conventions_related_by_permutation()
    test_pair_indices_cover_vector()
    test_rotation_preserves_norm_approximately()

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("All golden-model tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
