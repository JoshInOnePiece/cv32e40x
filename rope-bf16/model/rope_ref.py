"""Golden models for the BF16 RoPE unit (steps.md Milestone 0).

Three models, all needed:

  A. `rope_strict_bf16`  — strict BF16 at every node. This is what the hardware does
                           (INV-1) and the target for bit-exact comparison.
  B. `rope_wide_acc`     — BF16 in/out, FP32 internal accumulate. Exists only to
                           quantify what strict BF16 costs (the `AccWidth` study).
  C. `rope_exact`        — FP64 with libm sin/cos. Measures the total error of A and B.

Shared, exact-by-construction pieces (used by A and B, and mirrored in RTL):

  * `phi_table`  — integer phase increments, INV-2. Phase is *addressing*, not
                   arithmetic; it never touches a BF16 unit.
  * `sin_lut_bf16` / `lut_sin` / `lut_cos` — one midpoint-sampled quarter-wave table.

Both pairing conventions from steps.md Section 12 are implemented, because getting this
wrong makes a convention mismatch look like a numerical bug:

  * "interleaved" (GPT-J / original RoFormer): pairs are (x0,x1), (x2,x3), ...
  * "half_split"  (GPT-NeoX / HuggingFace LLaMA): pairs are (x_i, x_{i+d/2})  [DEFAULT]
"""

import numpy as np

from bf16 import (
    bf16_add,
    bf16_mul,
    bf16_neg,
    bf16_sub,
    bf16_to_f32,
    to_bf16,
)

# ---------------------------------------------------------------------------
# Format / sizing constants — must match rope_pkg.sv
# ---------------------------------------------------------------------------

PHASE_BITS = 32  # integer phase accumulator width (INV-2)
LUT_BITS = 10  # 1024 entries per quadrant
LUT_N = 1 << LUT_BITS
QUAD_SHIFT = PHASE_BITS - 2  # 30: phase[31:30] is the quadrant
IDX_SHIFT = PHASE_BITS - 2 - LUT_BITS  # 20: phase[29:20] is the LUT index

PHASE_MASK = (1 << PHASE_BITS) - 1
QUARTER_TURN = 1 << QUAD_SHIFT  # +pi/2 as a phase increment

DEFAULT_BASE = 10000.0  # LLaMA / RoFormer default

CONVENTIONS = ("half_split", "interleaved")


# ---------------------------------------------------------------------------
# Integer phase (INV-2)
# ---------------------------------------------------------------------------


def phi_table(d, base=DEFAULT_BASE):
    """Integer phase increments Phi_i = round(2^32 * theta_i / (2*pi)), as uint64.

    theta_i = base^(-2i/d) for i in [0, d/2).

    Returned as uint64 (not uint32) so that `m * Phi_i` can be formed without overflow
    before the mod-2^32 wrap; the wrap itself is free in hardware.
    """
    if d % 2:
        raise ValueError(f"head dimension d must be even, got {d}")
    i = np.arange(d // 2, dtype=np.float64)
    theta = base ** (-2.0 * i / d)
    phi = np.round((theta / (2.0 * np.pi)) * (1 << PHASE_BITS))
    return phi.astype(np.uint64) & np.uint64(PHASE_MASK)


def phase_of(m, i, phi):
    """Phase word for sequence position m and pair index i: (m * Phi_i) mod 2^32.

    Exact integer arithmetic — zero drift, and `mod 2*pi` is natural wraparound.
    """
    m64 = np.asarray(m, dtype=np.uint64)
    p = (m64 * np.asarray(phi, dtype=np.uint64)[i]) & np.uint64(PHASE_MASK)
    return p.astype(np.uint32)


# ---------------------------------------------------------------------------
# Sine LUT (Milestone 3)
# ---------------------------------------------------------------------------


def sin_lut_bf16():
    """Quarter-wave sine table, BF16 bit patterns, MIDPOINT sampled.

    T[k] = sin((k + 0.5) * (pi/2) / LUT_N)

    Midpoint sampling makes the Q1/Q3 reflection index exactly (LUT_N-1-idx) with no
    endpoint special case, and halves the worst-case sampling error (steps.md 7.3).
    """
    k = np.arange(LUT_N, dtype=np.float64)
    ang = (k + 0.5) * (np.pi / 2.0) / LUT_N
    return to_bf16(np.sin(ang))


LUT = sin_lut_bf16()


def lut_sin(phase):
    """sin() from a 32-bit integer phase word. Returns BF16 bit patterns.

    Sign application is an exact BF16 sign-bit XOR, never an arithmetic negation.
    """
    p = np.asarray(phase, dtype=np.uint32)
    quad = (p >> np.uint32(QUAD_SHIFT)) & np.uint32(0x3)
    idx = (p >> np.uint32(IDX_SHIFT)) & np.uint32(LUT_N - 1)

    reflect = (quad == 1) | (quad == 3)
    ridx = np.where(reflect, np.uint32(LUT_N - 1) - idx, idx)

    val = LUT[ridx]
    negate = quad >= 2
    return np.where(negate, bf16_neg(val), val).astype(np.uint16)


def lut_cos(phase):
    """cos(x) = sin(x + pi/2) -> phase + 2^30. ONE table, two lookups."""
    p = np.asarray(phase, dtype=np.uint32)
    return lut_sin((p + np.uint32(QUARTER_TURN)) & np.uint32(PHASE_MASK))


# ---------------------------------------------------------------------------
# MODEL A: strict BF16 — matches the hardware (INV-1)
# ---------------------------------------------------------------------------


def rope_strict_bf16(x, y, m, i, phi, flush_subnormals=False):
    """Rotate one BF16 pair. Inputs/outputs are BF16 *bit patterns*.

        x' = x*cos - y*sin
        y' = x*sin + y*cos

    Four BF16 multiplies and two BF16 adds, rounded RNE to BF16 at every node.
    """
    ph = phase_of(m, i, phi)
    c = lut_cos(ph)
    s = lut_sin(ph)

    xb = np.asarray(x, dtype=np.uint16)
    yb = np.asarray(y, dtype=np.uint16)

    kw = {"flush_subnormals": flush_subnormals}
    p0 = bf16_mul(xb, c, **kw)  # x*cos
    p1 = bf16_mul(yb, s, **kw)  # y*sin
    p2 = bf16_mul(xb, s, **kw)  # x*sin
    p3 = bf16_mul(yb, c, **kw)  # y*cos

    return bf16_sub(p0, p1, **kw), bf16_add(p2, p3, **kw)


# ---------------------------------------------------------------------------
# MODEL B: BF16 I/O, FP32 internal accumulate (AccWidth=32 comparison only)
# ---------------------------------------------------------------------------


def rope_wide_acc(x, y, m, i, phi):
    """Same rotation, but the products are summed in FP32 before one final rounding.

    This is what every other BF16 accelerator does (multiply narrow, accumulate wide).
    Not shipped — it exists to measure what INV-1 costs.
    """
    ph = phase_of(m, i, phi)
    c = bf16_to_f32(lut_cos(ph)).astype(np.float32)
    s = bf16_to_f32(lut_sin(ph)).astype(np.float32)

    xf = bf16_to_f32(x).astype(np.float32)
    yf = bf16_to_f32(y).astype(np.float32)

    return to_bf16(xf * c - yf * s), to_bf16(xf * s + yf * c)


# ---------------------------------------------------------------------------
# MODEL C: FP64 exact reference
# ---------------------------------------------------------------------------


def rope_exact(x, y, m, i, d, base=DEFAULT_BASE):
    """FP64 reference with true sin/cos. Inputs are BF16 bit patterns (exactly
    representable in FP64); outputs are FP64 reals, deliberately NOT rounded to BF16.

    This is the yardstick: it includes no LUT quantisation and no BF16 rounding, so the
    difference A-vs-C is the *total* error of the strict-BF16 design.
    """
    th = float(base) ** (-2.0 * np.asarray(i, dtype=np.float64) / d)
    a = np.asarray(m, dtype=np.float64) * th
    xf = bf16_to_f32(x).astype(np.float64)
    yf = bf16_to_f32(y).astype(np.float64)
    ca, sa = np.cos(a), np.sin(a)
    return xf * ca - yf * sa, xf * sa + yf * ca


# ---------------------------------------------------------------------------
# Whole-vector application, both pairing conventions (steps.md Section 12)
# ---------------------------------------------------------------------------


def pair_indices(d, convention="half_split"):
    """Return (lo_idx, hi_idx, pair_i) arrays describing the d/2 rotated pairs.

    `pair_i` is the *frequency* index used to look up Phi, which is what makes the two
    conventions differ: the same theta_i is applied to different element positions.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")
    half = d // 2
    pair_i = np.arange(half)
    if convention == "half_split":
        # HuggingFace LLaMA / GPT-NeoX: element i pairs with element i + d/2.
        return pair_i, pair_i + half, pair_i
    # Interleaved (GPT-J / original RoFormer): adjacent elements pair up.
    return 2 * pair_i, 2 * pair_i + 1, pair_i


def rope_vector_strict_bf16(vec, m, phi, convention="half_split", flush_subnormals=False):
    """Apply strict-BF16 RoPE to one head vector of BF16 bit patterns.

    `vec` has shape (..., d). Returns the rotated vector, same shape and dtype.
    """
    v = np.asarray(vec, dtype=np.uint16)
    d = v.shape[-1]
    lo, hi, pi_ = pair_indices(d, convention)

    out = v.copy()
    xs, ys = v[..., lo], v[..., hi]
    xr, yr = rope_strict_bf16(xs, ys, m, pi_, phi, flush_subnormals=flush_subnormals)
    out[..., lo] = xr
    out[..., hi] = yr
    return out


def rope_vector_exact(vec, m, d=None, base=DEFAULT_BASE, convention="half_split"):
    """FP64 reference applied to a whole head vector. Returns FP64."""
    v = np.asarray(vec, dtype=np.uint16)
    d = v.shape[-1] if d is None else d
    lo, hi, pi_ = pair_indices(d, convention)

    out = bf16_to_f32(v).astype(np.float64)
    xr, yr = rope_exact(v[..., lo], v[..., hi], m, pi_, d, base)
    out[..., lo] = xr
    out[..., hi] = yr
    return out


# ---------------------------------------------------------------------------
# HuggingFace-equivalent reference (for the convention cross-check)
# ---------------------------------------------------------------------------


def hf_apply_rotary_pos_emb(x, m, base=DEFAULT_BASE):
    """Reimplementation of HuggingFace's `apply_rotary_pos_emb` + `rotate_half`, FP64.

    Mirrors transformers/models/llama/modeling_llama.py:

        inv_freq = 1 / (base ** (arange(0, d, 2) / d))
        freqs    = m * inv_freq            # length d/2
        emb      = cat(freqs, freqs)       # length d
        cos, sin = emb.cos(), emb.sin()
        rotate_half(x) = cat(-x[d/2:], x[:d/2])
        out = x * cos + rotate_half(x) * sin

    Written out explicitly so the half-split convention can be verified against the
    structure HF actually uses, rather than against our own restatement of it.
    """
    xf = np.asarray(x, dtype=np.float64)
    d = xf.shape[-1]
    half = d // 2

    inv_freq = 1.0 / (base ** (np.arange(0, d, 2, dtype=np.float64) / d))
    freqs = float(m) * inv_freq
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos, sin = np.cos(emb), np.sin(emb)

    rot = np.concatenate([-xf[..., half:], xf[..., :half]], axis=-1)
    return xf * cos + rot * sin


def permutation_interleaved_to_half_split(d):
    """Index permutation mapping an interleaved-layout vector to half-split layout.

    The two conventions are equivalent up to this fixed permutation of the head
    dimension, which is why trained weights port between them (steps.md Section 12).
    """
    half = d // 2
    perm = np.empty(d, dtype=np.int64)
    perm[:half] = np.arange(0, d, 2)
    perm[half:] = np.arange(1, d, 2)
    return perm
