#!/usr/bin/env python3
"""Validate the BF16 primitives in bf16.py.

Plain Python, no pytest — run directly:  python3 model/test_bf16.py

The point of this file is to prove that `round_f32_to_bf16` (fast, bit-twiddling) agrees
with `_round_f32_to_bf16_reference_scalar` (slow, derived from first principles) so that
every downstream bit-exactness claim rests on a rounding function that was actually
checked rather than assumed.
"""

import sys

import numpy as np

from bf16 import (
    BF16_MAX_NORMAL,
    BF16_MIN_NORMAL,
    BF16_MIN_SUBNORMAL,
    BF16_NEG_INF,
    BF16_POS_INF,
    BF16_QNAN,
    _round_f32_to_bf16_reference_scalar,
    bf16_add,
    bf16_mul,
    bf16_sub,
    bf16_to_f32,
    is_inf,
    is_nan,
    is_subnormal,
    round_f32_to_bf16,
    to_bf16,
)

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


# ---------------------------------------------------------------------------


def test_roundtrip_all_bf16():
    """Every one of the 65536 BF16 patterns must survive bits -> f32 -> bits."""
    allbits = np.arange(1 << 16, dtype=np.uint16)
    back = round_f32_to_bf16(bf16_to_f32(allbits))
    # NaN patterns legitimately canonicalise to BF16_QNAN; everything else is identity.
    nanmask = is_nan(allbits)
    ok_non_nan = np.array_equal(back[~nanmask], allbits[~nanmask])
    ok_nan = np.all(back[nanmask] == BF16_QNAN)
    check("bits->f32->bits identity (non-NaN, all 65536)", ok_non_nan)
    check("NaN patterns canonicalise to 0x7FC0", ok_nan)


def test_upper_16_bits_identity():
    """bf16_to_f32(b) must equal bitcast<float>(b << 16) — the steps.md Section 4.1 fact."""
    allbits = np.arange(1 << 16, dtype=np.uint16)
    expect = (allbits.astype(np.uint32) << 16).view(np.float32)
    got = bf16_to_f32(allbits)
    check(
        "bf16_to_f32 == bitcast<float>(bits<<16), exact",
        np.array_equal(got.view(np.uint32), expect.view(np.uint32)),
    )


def test_rne_vs_reference_random():
    """Fast RNE vs independent scalar reference over random FP32 bit patterns."""
    rng = np.random.default_rng(12345)
    n = 200_000
    u = rng.integers(0, 1 << 32, size=n, dtype=np.uint64).astype(np.uint32)
    xf = u.view(np.float32)
    fast = round_f32_to_bf16(xf)
    bad = 0
    first = None
    for i in range(n):
        ref = _round_f32_to_bf16_reference_scalar(xf[i])
        if int(fast[i]) != ref:
            bad += 1
            if first is None:
                first = (hex(int(u[i])), float(xf[i]), hex(int(fast[i])), hex(ref))
    check(
        f"RNE vs reference, {n} random FP32 patterns",
        bad == 0,
        f"{bad} mismatches, first={first}",
    )


def test_rne_vs_reference_ties():
    """Directed exact-tie test — the case truncation would silently get wrong.

    An exact tie is an FP32 whose low 16 mantissa bits are exactly 0x8000. RNE must
    round to the even neighbour; truncation would always round down and introduce the
    systematic negative bias steps.md Section 5.3 warns about.
    """
    # Build ties across the whole exponent range and both parities of the retained LSB.
    ties = []
    for exp in range(1, 255):
        for man_hi in (0x00, 0x01, 0x02, 0x03, 0x7E, 0x7F):
            for sign in (0, 1):
                u = (sign << 31) | (exp << 23) | (man_hi << 16) | 0x8000
                ties.append(u)
    u = np.array(ties, dtype=np.uint32)
    xf = u.view(np.float32)
    fast = round_f32_to_bf16(xf)

    bad = 0
    first = None
    for i in range(len(u)):
        ref = _round_f32_to_bf16_reference_scalar(xf[i])
        if int(fast[i]) != ref:
            bad += 1
            if first is None:
                first = (hex(int(u[i])), hex(int(fast[i])), hex(ref))
    check(f"RNE on {len(u)} exact ties", bad == 0, f"{bad} mismatches, first={first}")

    # And assert the ties-to-even property explicitly: result significand must be even.
    check(
        "every exact tie rounds to an even significand",
        np.all((fast & np.uint16(1)) == 0),
    )

    # A truncating implementation would differ from ours on ties with odd LSB -> prove
    # our function is genuinely not truncation.
    trunc = (u >> 16).astype(np.uint16)
    check(
        "RNE differs from truncation on odd-LSB ties (not silently truncating)",
        np.any(fast != trunc),
    )


def test_no_bias_on_ties():
    """Ties must round up and down about equally — a bias check, not just a spot check."""
    rng = np.random.default_rng(7)
    exps = rng.integers(60, 200, size=20000)
    mans = rng.integers(0, 0x80, size=20000)
    u = ((exps.astype(np.uint32) << 23) | (mans.astype(np.uint32) << 16) | 0x8000).astype(
        np.uint32
    )
    xf = u.view(np.float32)
    got = round_f32_to_bf16(xf)
    trunc = (u >> 16).astype(np.uint16)
    up = int(np.sum(got > trunc))
    down = int(np.sum(got == trunc))
    # With uniformly random LSB parity, ~half the ties round up.
    check(
        f"tie rounding unbiased (up={up}, down={down})",
        abs(up - down) < 0.1 * len(u),
        f"up={up} down={down}",
    )


def test_specials():
    check("+Inf survives", int(round_f32_to_bf16(np.float32(np.inf))) == int(BF16_POS_INF))
    check("-Inf survives", int(round_f32_to_bf16(np.float32(-np.inf))) == int(BF16_NEG_INF))
    check("NaN -> canonical qNaN", int(round_f32_to_bf16(np.float32(np.nan))) == int(BF16_QNAN))
    check("+0 -> +0", int(round_f32_to_bf16(np.float32(0.0))) == 0x0000)
    check("-0 -> -0", int(round_f32_to_bf16(np.float32(-0.0))) == 0x8000)

    # NaN with payload only in the low 16 bits: the naive shift would yield +Inf.
    sneaky = np.array([0x7F800001], dtype=np.uint32).view(np.float32)
    check(
        "NaN with low-16-bit payload does NOT become Inf",
        int(round_f32_to_bf16(sneaky)[0]) == int(BF16_QNAN),
    )

    # Overflow: a value above BF16's max normal must go to Inf, not wrap.
    big = np.float32(3.5e38)
    check("overflow -> +Inf", int(round_f32_to_bf16(big)) == int(BF16_POS_INF))
    check(
        "max normal 0x7F7F round-trips",
        int(round_f32_to_bf16(bf16_to_f32(BF16_MAX_NORMAL))) == int(BF16_MAX_NORMAL),
    )


def test_subnormals():
    check(
        "min subnormal 2^-133 round-trips",
        int(round_f32_to_bf16(bf16_to_f32(BF16_MIN_SUBNORMAL))) == int(BF16_MIN_SUBNORMAL),
    )
    check("2^-133 classified subnormal", bool(is_subnormal(BF16_MIN_SUBNORMAL)))
    check("2^-126 classified normal", not bool(is_subnormal(BF16_MIN_NORMAL)))

    # Flush-to-zero policy must preserve sign and leave normals alone.
    neg_sub = np.uint16(0x8001)
    ftz = round_f32_to_bf16(bf16_to_f32(neg_sub), flush_subnormals=True)
    check("FTZ maps -subnormal to -0", int(ftz) == 0x8000)
    keep = round_f32_to_bf16(bf16_to_f32(neg_sub), flush_subnormals=False)
    check("no-FTZ keeps -subnormal", int(keep) == 0x8001)


def test_arithmetic_single_rounding():
    """Arithmetic helpers must equal 'exact result, rounded once' computed in FP64."""
    rng = np.random.default_rng(99)
    n = 300_000
    a = rng.integers(0, 1 << 16, size=n, dtype=np.uint16)
    b = rng.integers(0, 1 << 16, size=n, dtype=np.uint16)

    af = bf16_to_f32(a).astype(np.float64)
    bf = bf16_to_f32(b).astype(np.float64)

    for name, hw, exact in (
        ("mul", bf16_mul(a, b), af * bf),
        ("add", bf16_add(a, b), af + bf),
        ("sub", bf16_sub(a, b), af - bf),
    ):
        # Round the FP64-exact result once. FP64->FP32 is itself exact here (see the
        # module docstring), so this is a genuine independent path.
        ref = round_f32_to_bf16(exact.astype(np.float32))
        finite = ~(is_nan(hw) | is_nan(ref))
        ok = np.array_equal(hw[finite], ref[finite])
        nan_ok = np.array_equal(is_nan(hw), is_nan(ref))
        check(f"bf16_{name} == round_once(exact) on {n} random pairs", ok and nan_ok)


def test_known_values():
    """Hand-computed values, so a systematic error can't hide behind self-consistency."""
    cases = [
        (1.0, 0x3F80),
        (-1.0, 0xBF80),
        (2.0, 0x4000),
        (0.5, 0x3F00),
        (3.0, 0x4040),
        (1.0078125, 0x3F81),  # 1 + 2^-7, exactly the next BF16 after 1.0
        (100.0, 0x42C8),
        (10000.0, 0x461C),
    ]
    ok = True
    for val, want in cases:
        got = int(to_bf16(val))
        if got != want:
            ok = False
            print(f"        {val}: got 0x{got:04X} want 0x{want:04X}")
    check("hand-checked BF16 encodings", ok)

    # The midpoint between 1.0 and 1.0078125 is an exact tie -> must go to even (1.0).
    mid = 1.0 + 2.0**-8
    check(
        "tie between 1.0 and 1.0078125 rounds to even (1.0)",
        int(to_bf16(mid)) == 0x3F80,
        f"got 0x{int(to_bf16(mid)):04X}",
    )
    # 1.0 + 2^-8 is exactly representable as a sum of two BF16 values, and BF16 add
    # must reproduce that tie behaviour.
    check(
        "bf16_add(1.0, 2^-8) ties to even",
        int(bf16_add(to_bf16(1.0), to_bf16(2.0**-8))) == 0x3F80,
    )


def main():
    print("=== BF16 primitive tests ===")
    test_upper_16_bits_identity()
    test_roundtrip_all_bf16()
    test_known_values()
    test_rne_vs_reference_random()
    test_rne_vs_reference_ties()
    test_no_bias_on_ties()
    test_specials()
    test_subnormals()
    test_arithmetic_single_rounding()

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("All BF16 primitive tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
