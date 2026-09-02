"""BF16 (bfloat16) helpers with exact round-to-nearest-even semantics.

Everything here works on **uint16 bit patterns**, not on a numpy extension dtype.
That is deliberate: the hardware produces bit patterns, so a bit-pattern model makes
"bit-exact" comparison unambiguous and needs no third-party float16-alt library.

Key fact (steps.md Section 4.1):

    BF16 is bit-identical to the upper 16 bits of FP32.
    bf16_to_f32(b) == bitcast<float>(b << 16), exactly, with no rounding.

Rounding policy
---------------
Round-to-nearest-even (RNE) at every node, per INV-1. `round_f32_to_bf16` is the single
rounding primitive; every arithmetic helper funnels through it exactly once, so no
double-rounding can occur.

Why an FP32 intermediate is safe (no double rounding)
-----------------------------------------------------
For BF16 inputs, FP32 holds the *exact* result of a single multiply or add closely
enough that rounding FP32->BF16 equals rounding the exact result->BF16:

* multiply: two 8-bit significands give a 16-bit product, exact in FP32's 24 bits.
* add/sub: exact whenever the exponent difference is <= 16. Beyond that the smaller
  operand lies entirely below FP32's rounding position, and the FP32 result cannot
  land on a BF16 tie point (a BF16 tie point needs bit 8 of the significand set,
  which a rounding increment at bit 23 cannot produce), so the tie-break is never
  corrupted.
* BF16 and FP32 share the 8-bit exponent field and bias 127, so BF16's subnormal
  range (2^-133 .. 2^-126) sits well inside FP32's, with 16 bits of headroom.

Subnormal policy
----------------
Selectable, because the *hardware's* policy is what decides this and must be measured
rather than assumed (see docs/notes.md). `flush_subnormals=False` keeps gradual
underflow; `True` flushes subnormal results to zero, preserving sign.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------

BF16_SIGN_MASK = np.uint16(0x8000)
BF16_EXP_MASK = np.uint16(0x7F80)
BF16_MAN_MASK = np.uint16(0x007F)
BF16_MAN_BITS = 7  # stored mantissa bits (8 significand bits with the implicit one)
BF16_EXP_BITS = 8
BF16_BIAS = 127

BF16_POS_INF = np.uint16(0x7F80)
BF16_NEG_INF = np.uint16(0xFF80)
BF16_QNAN = np.uint16(0x7FC0)  # canonical quiet NaN (matches FPnew's canonical NaN)
BF16_POS_ZERO = np.uint16(0x0000)
BF16_NEG_ZERO = np.uint16(0x8000)

BF16_MAX_NORMAL = np.uint16(0x7F7F)  # ~3.3895e38
BF16_MIN_NORMAL = np.uint16(0x0080)  # 2^-126
BF16_MIN_SUBNORMAL = np.uint16(0x0001)  # 2^-133


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def bf16_to_f32(bits):
    """BF16 bit pattern -> FP32. EXACT, no rounding: BF16 is FP32's top 16 bits."""
    u = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return u.view(np.float32)


def round_f32_to_bf16(x, flush_subnormals=False):
    """Round FP32 -> BF16 bit pattern, round-to-nearest-even.

    NaN inputs map to the canonical quiet NaN rather than being truncated: the naive
    "add 0x7FFF and shift" would turn e.g. 0x7F800001 (a NaN with all its payload in
    the low 16 bits) into +Inf, which is wrong and would silently corrupt NaN
    propagation tests.
    """
    xf = np.asarray(x, dtype=np.float32)
    u = xf.view(np.uint32)

    exp_all_ones = (u & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
    man_nonzero = (u & np.uint32(0x007FFFFF)) != np.uint32(0)
    is_nan = exp_all_ones & man_nonzero

    # RNE: add half an LSB, plus one more if the retained LSB is odd.
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    rounded = ((u + np.uint32(0x7FFF) + lsb) >> np.uint32(16)).astype(np.uint16)

    out = np.where(is_nan, BF16_QNAN, rounded).astype(np.uint16)

    if flush_subnormals:
        is_sub = ((out & BF16_EXP_MASK) == 0) & ((out & BF16_MAN_MASK) != 0)
        out = np.where(is_sub, out & BF16_SIGN_MASK, out).astype(np.uint16)

    return out


def f32_to_bf16(x, flush_subnormals=False):
    """Alias for round_f32_to_bf16 (kept for readability at call sites)."""
    return round_f32_to_bf16(x, flush_subnormals=flush_subnormals)


def to_bf16(x, flush_subnormals=False):
    """Round any real input (FP32/FP64/python float) -> BF16 bit pattern, RNE."""
    return round_f32_to_bf16(
        np.asarray(x, dtype=np.float64).astype(np.float32),
        flush_subnormals=flush_subnormals,
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def is_nan(bits):
    b = np.asarray(bits, dtype=np.uint16)
    return ((b & BF16_EXP_MASK) == BF16_EXP_MASK) & ((b & BF16_MAN_MASK) != 0)


def is_inf(bits):
    b = np.asarray(bits, dtype=np.uint16)
    return ((b & BF16_EXP_MASK) == BF16_EXP_MASK) & ((b & BF16_MAN_MASK) == 0)


def is_zero(bits):
    b = np.asarray(bits, dtype=np.uint16)
    return (b & np.uint16(0x7FFF)) == 0


def is_subnormal(bits):
    b = np.asarray(bits, dtype=np.uint16)
    return ((b & BF16_EXP_MASK) == 0) & ((b & BF16_MAN_MASK) != 0)


# ---------------------------------------------------------------------------
# Arithmetic — each rounds exactly once, RNE
# ---------------------------------------------------------------------------


def bf16_neg(bits):
    """Exact sign flip (XOR on bit 15). No arithmetic, no rounding (steps.md 7.4)."""
    return (np.asarray(bits, dtype=np.uint16) ^ BF16_SIGN_MASK).astype(np.uint16)


# Inf*0, Inf-Inf and overflow are *intended, tested* behaviours here (NaN/Inf
# propagation), so numpy's warnings about them are noise rather than signal.
_SPECIALS_OK = {"invalid": "ignore", "over": "ignore", "under": "ignore"}


def bf16_mul(a, b, flush_subnormals=False):
    """BF16 x BF16 -> BF16, single RNE rounding."""
    with np.errstate(**_SPECIALS_OK):
        prod = bf16_to_f32(a) * bf16_to_f32(b)
        return round_f32_to_bf16(prod, flush_subnormals=flush_subnormals)


def bf16_add(a, b, flush_subnormals=False):
    """BF16 + BF16 -> BF16, single RNE rounding."""
    with np.errstate(**_SPECIALS_OK):
        s = bf16_to_f32(a) + bf16_to_f32(b)
        return round_f32_to_bf16(s, flush_subnormals=flush_subnormals)


def bf16_sub(a, b, flush_subnormals=False):
    """BF16 - BF16 -> BF16, single RNE rounding."""
    with np.errstate(**_SPECIALS_OK):
        s = bf16_to_f32(a) - bf16_to_f32(b)
        return round_f32_to_bf16(s, flush_subnormals=flush_subnormals)


# ---------------------------------------------------------------------------
# Independent (slow) reference for validating the rounding primitive
# ---------------------------------------------------------------------------


def _round_f32_to_bf16_reference_scalar(xf):
    """Deliberately naive RNE rounding, derived from first principles.

    Enumerates the two candidate BF16 neighbours and picks the nearer, breaking exact
    ties toward the even significand. Used only by the test suite to cross-check
    `round_f32_to_bf16`; it shares no logic with it on purpose.
    """
    xf = np.float32(xf)
    u = int(np.float32(xf).view(np.uint32))

    if (u & 0x7F800000) == 0x7F800000:
        if u & 0x007FFFFF:
            return int(BF16_QNAN)
        return (u >> 16) & 0xFFFF  # +/-Inf passes through

    lo = (u >> 16) & 0xFFFF  # truncated candidate
    # The other candidate is the next BF16 away from zero.
    hi = (lo + 1) & 0xFFFF

    lo_val = float(bf16_to_f32(np.uint16(lo)))

    # If `hi` is the infinity encoding, its value for *comparison* purposes is not
    # infinity: IEEE 754-2019 Section 4.3.1 rounds as if the exponent range were
    # unbounded and only then overflows. That unbounded value is 2^128 for BF16
    # (emax=127). Treating it as literal Inf would make the upper candidate
    # infinitely far away and wrongly clamp every overflow case to MAX_NORMAL.
    if (hi & 0x7FFF) == 0x7F80:
        hi_val = -(2.0**128) if (hi & 0x8000) else 2.0**128
    else:
        hi_val = float(bf16_to_f32(np.uint16(hi)))

    x = float(xf)

    if x == lo_val:
        return lo

    d_lo = abs(x - lo_val)
    d_hi = abs(hi_val - x)

    if d_lo < d_hi:
        return lo
    if d_hi < d_lo:
        return hi
    # Exact tie -> pick the even significand.
    return lo if (lo & 1) == 0 else hi
