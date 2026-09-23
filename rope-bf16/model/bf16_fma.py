"""Exact BF16 fused multiply-add, to model CVFPU's `fpnew_fma` in FMADD/FMSUB mode.

`fpnew_fma` computes `a*b + c` with a SINGLE round-to-nearest-even at the end: the
product is formed to its full 2p bits and the addend is aligned against it in a wide
internal accumulator (`sum_raw`, 3p+5 bits) before rounding once. Modelling that with
float64 is *nearly* always right but not provably so -- when the product and the addend
sit far apart in exponent, the addend can fall off the end of float64's 53-bit
significand, and rounding the inexact float64 value is a double rounding that can
disagree with the hardware on a tie.

So the reference here is exact integer arithmetic, with a fast float64 path that is
cross-checked against it. Every BF16 value is exactly `(-1)**s * M * 2**E` for integers
M and E, so `a*b + c` is exact in Python integers with no precision limit at all.
"""

import numpy as np

from bf16 import (
    BF16_QNAN,
    bf16_to_f32,
    to_bf16,
)

__all__ = ["bf16_fma_exact", "bf16_fma", "decompose", "MAN_BITS"]

MAN_BITS = 7
SIG_BITS = MAN_BITS + 1          # 8 significand bits including the implicit one
BIAS = 127
EXP_MASK = 0x7F80
MAN_MASK = 0x007F


def decompose(u):
    """BF16 bit pattern -> (sign, significand M, exponent E) with value = (-1)^s * M * 2^E.

    Normals:    M = 128+mantissa, E = e - BIAS - MAN_BITS
    Subnormals: M = mantissa,     E = 1 - BIAS - MAN_BITS
    Zero falls out of the subnormal case with M = 0.
    """
    u = int(u) & 0xFFFF
    s = (u >> 15) & 1
    e = (u & EXP_MASK) >> MAN_BITS
    m = u & MAN_MASK
    if e == 0:
        return s, m, 1 - BIAS - MAN_BITS
    return s, (1 << MAN_BITS) | m, e - BIAS - MAN_BITS


def _round_exact_to_bf16(sign, num, exp):
    """Round the exact value (-1)^sign * num * 2^exp to BF16, RNE, in ONE rounding.

    `num` is a non-negative Python int of arbitrary size, so nothing has been discarded
    before this point. The normal and subnormal ranges use different grids, so which one
    the result lands in is decided BEFORE rounding -- rounding to the significand first
    and then onto the subnormal grid would be a double rounding.
    """
    if num == 0:
        return np.uint16(0x8000 if sign else 0x0000)

    SUB_EXP = -133                      # smallest subnormal is 1 * 2**-133

    # Where would the result land if the exponent range were unbounded?
    nbits = num.bit_length()
    e_unbiased = exp + nbits - 1        # exponent of the leading set bit
    e_biased = e_unbiased + BIAS

    if e_biased <= 0:
        # ---- subnormal grid: uniform spacing of 2**SUB_EXP, round directly onto it ----
        shift = SUB_EXP - exp
        if shift <= 0:
            M = num << (-shift)
        else:
            if shift > nbits + 2:       # far below half the smallest subnormal
                return np.uint16(sign << 15)
            M = num >> shift
            rem = num & ((1 << shift) - 1)
            half = 1 << (shift - 1)
            if rem > half or (rem == half and (M & 1)):
                M += 1
        if M >= (1 << MAN_BITS):
            # rounded up out of the subnormal range into the smallest normal
            return np.uint16((sign << 15) | (1 << MAN_BITS))
        return np.uint16((sign << 15) | M)

    # ---- normal range: round to SIG_BITS significand bits ----
    shift = nbits - SIG_BITS
    if shift > 0:
        keep = num >> shift
        rem = num & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if rem > half or (rem == half and (keep & 1)):
            keep += 1
            if keep >> SIG_BITS:        # carried out of the significand
                keep >>= 1
                shift += 1
    else:
        keep = num << (-shift)

    e = exp + shift + MAN_BITS + BIAS
    if e >= 0xFF:                       # overflow -> infinity
        return np.uint16((sign << 15) | 0x7F80)
    if e <= 0:
        # The round-up pushed it back down a binade into the subnormal range; redo on
        # the subnormal grid from the exact value rather than from `keep`.
        shift2 = SUB_EXP - exp
        if shift2 > nbits + 2:
            return np.uint16(sign << 15)
        M = num >> shift2 if shift2 > 0 else num << (-shift2)
        if shift2 > 0:
            rem2 = num & ((1 << shift2) - 1)
            half2 = 1 << (shift2 - 1)
            if rem2 > half2 or (rem2 == half2 and (M & 1)):
                M += 1
        if M >= (1 << MAN_BITS):
            return np.uint16((sign << 15) | (1 << MAN_BITS))
        return np.uint16((sign << 15) | M)

    return np.uint16((sign << 15) | (e << MAN_BITS) | (keep & MAN_MASK))


def bf16_fma_exact(a, b, c, sub=False):
    """Scalar, exact `a*b + c` (or `a*b - c` when `sub`), rounded once to BF16.

    Mirrors fpnew_fma: op_mod inverts the sign of operand C unconditionally, which is
    why `sub` works for any sign of c (subtracting a negative adds).
    """
    a, b, c = int(a) & 0xFFFF, int(b) & 0xFFFF, int(c) & 0xFFFF

    # Special values first -- IEEE ordering: NaN wins, then Inf*0 invalid, then Inf.
    def is_nan(u):
        return (u & EXP_MASK) == EXP_MASK and (u & MAN_MASK) != 0

    def is_inf(u):
        return (u & EXP_MASK) == EXP_MASK and (u & MAN_MASK) == 0

    def is_zero(u):
        return (u & 0x7FFF) == 0

    csign = ((c >> 15) & 1) ^ (1 if sub else 0)

    if is_nan(a) or is_nan(b) or is_nan(c):
        return BF16_QNAN
    if (is_inf(a) and is_zero(b)) or (is_inf(b) and is_zero(a)):
        return BF16_QNAN                       # Inf * 0 -> invalid
    psign = ((a >> 15) & 1) ^ ((b >> 15) & 1)
    if is_inf(a) or is_inf(b):
        if is_inf(c) and csign != psign:
            return BF16_QNAN                   # Inf - Inf -> invalid
        return np.uint16((psign << 15) | 0x7F80)
    if is_inf(c):
        return np.uint16((csign << 15) | 0x7F80)

    sa, Ma, Ea = decompose(a)
    sb, Mb, Eb = decompose(b)
    _, Mc, Ec = decompose(c)

    Mp, Ep = Ma * Mb, Ea + Eb                  # exact product, no rounding

    if Mp == 0 and Mc == 0:                    # both zero: sign per IEEE (RNE -> +0)
        return np.uint16(0x8000 if (psign and csign) else 0x0000)

    e_min = min(Ep, Ec)
    total = (Mp << (Ep - e_min)) * (-1 if psign else 1) \
          + (Mc << (Ec - e_min)) * (-1 if csign else 1)

    if total == 0:
        return np.uint16(0x0000)               # exact cancellation -> +0 under RNE
    return _round_exact_to_bf16(1 if total < 0 else 0, abs(total), e_min)


def _exp_of(u):
    """Unbiased exponent of the leading significand bit, for risk estimation only."""
    e = (u & EXP_MASK) >> MAN_BITS
    return np.where(e == 0, -126 - MAN_BITS, e.astype(np.int32) - BIAS)


def bf16_fma(a, b, c, sub=False, check=False):
    """Vectorised `a*b + c` (or `a*b - c`), rounded exactly once to BF16.

    float64 holds the exact value of `a*b + c` only while the product and the addend are
    close enough in magnitude that the addend does not fall off the end of its 53-bit
    significand. When it does fall off, rounding the float64 result is a DOUBLE rounding
    and can disagree with the hardware -- specifically when the product lands exactly on
    a rounding tie that the discarded addend should have broken. That is not theoretical:
    it occurs on ~0.2% of uniformly random BF16 triples.

    So the float64 path is used only where it is provably exact, and every other case
    falls back to `bf16_fma_exact`. `check=True` verifies the whole array against the
    exact model.
    """
    a = np.asarray(a, dtype=np.uint16)
    b = np.asarray(b, dtype=np.uint16)
    c = np.asarray(c, dtype=np.uint16)
    a, b, c = np.broadcast_arrays(a, b, c)
    shape = a.shape
    a, b, c = np.ravel(a), np.ravel(b), np.ravel(c)

    af = bf16_to_f32(a).astype(np.float64)
    bf = bf16_to_f32(b).astype(np.float64)
    cf = bf16_to_f32(c).astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        out = to_bf16(af * bf + (-cf if sub else cf))

    # A case is safe for float64 only if every operand is a plain normal number AND the
    # product and addend are within 20 binades -- comfortably inside the 53-bit budget
    # (16 product bits + 8 addend bits + alignment).
    def is_normal(u):
        e = (u & EXP_MASK) >> MAN_BITS
        return (e != 0) & (e != 0xFF)

    prod_exp = _exp_of(a) + _exp_of(b)
    risky = ~(is_normal(a) & is_normal(b) & is_normal(c))
    risky |= np.abs(prod_exp - _exp_of(c)) > 20

    idx = np.flatnonzero(risky)
    for k in idx:
        out[k] = bf16_fma_exact(a[k], b[k], c[k], sub)

    if check:
        exact = np.array([bf16_fma_exact(a[k], b[k], c[k], sub) for k in range(a.size)],
                         dtype=np.uint16)
        bad = int(np.count_nonzero(exact != out))
        if bad:
            raise AssertionError(
                "bf16_fma disagrees with the exact model on %d of %d cases"
                % (bad, a.size))

    return out.reshape(shape)
