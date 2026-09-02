#!/usr/bin/env python3
"""Error characterisation for strict-BF16 RoPE (steps.md Milestones 0 and 7).

Run:  python3 model/characterize.py [--out docs/error_budget.md]

Produces the quantitative results the writeup needs:

  1. Model A (strict BF16) vs Model C (FP64 exact)  -- the total error budget
  2. Model A vs Model B (wide accumulate)           -- what INV-1 actually costs
  3. Error vs sequence position m, and vs pair index i
  4. The catastrophic-cancellation study (steps.md 11.2): x ~ y with m*theta ~ 45 deg
  5. Significant-bit loss statistics

This is the measurement steps.md argues has not been published: every BF16 accelerator
multiplies in BF16 but accumulates wider, so the cost of *strict* BF16 for RoPE
specifically is unquantified.
"""

import argparse
import sys

import numpy as np

from bf16 import bf16_to_f32, to_bf16
from rope_ref import (
    DEFAULT_BASE,
    lut_cos,
    lut_sin,
    phase_of,
    phi_table,
    rope_exact,
    rope_strict_bf16,
    rope_wide_acc,
)


def rel_err(got, want):
    """Relative error with a sane definition at want==0 (fall back to absolute)."""
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    denom = np.abs(want)
    scale = np.where(denom > 0, denom, 1.0)
    return np.abs(got - want) / scale


def pair_rel_err(xr, yr, xe, ye):
    """Relative error of the rotated pair, measured on the pair's vector norm.

    Per-component relative error is misleading under cancellation: a component whose
    true value is near zero has an unbounded relative error while carrying almost no
    information. Normalising by the pair norm (which rotation preserves) is the
    meaningful measure, so both are reported.
    """
    xr = np.asarray(xr, dtype=np.float64)
    yr = np.asarray(yr, dtype=np.float64)
    xe = np.asarray(xe, dtype=np.float64)
    ye = np.asarray(ye, dtype=np.float64)
    err = np.sqrt((xr - xe) ** 2 + (yr - ye) ** 2)
    norm = np.sqrt(xe**2 + ye**2)
    return err / np.where(norm > 0, norm, 1.0)


def bits_retained(got, want):
    """How many significant bits survive: -log2(relative error), clipped at 24."""
    r = rel_err(got, want)
    with np.errstate(divide="ignore"):
        b = -np.log2(np.maximum(r, 2.0**-24))
    # `+ 0.0` normalises the -0.0 that clipping a small negative produces, which would
    # otherwise print as "-0.0 bits kept".
    return np.clip(b, 0, 24) + 0.0


def gen_inputs(rng, n, lo=-4.0, hi=4.0):
    """Random BF16 operands in a range typical of post-GEMM activations."""
    return to_bf16(rng.uniform(lo, hi, size=n)), to_bf16(rng.uniform(lo, hi, size=n))


# ---------------------------------------------------------------------------
# 1 + 2: broad random sweep, A vs C and B vs C
# ---------------------------------------------------------------------------


def study_broad(d=128, base=DEFAULT_BASE, n=400_000, max_m=4096, seed=1):
    rng = np.random.default_rng(seed)
    phi = phi_table(d, base)

    x, y = gen_inputs(rng, n)
    m = rng.integers(0, max_m, size=n)
    i = rng.integers(0, d // 2, size=n)

    xa, ya = rope_strict_bf16(x, y, m, i, phi)
    xb, yb = rope_wide_acc(x, y, m, i, phi)
    xe, ye = rope_exact(x, y, m, i, d, base)

    a_pair = pair_rel_err(bf16_to_f32(xa), bf16_to_f32(ya), xe, ye)
    b_pair = pair_rel_err(bf16_to_f32(xb), bf16_to_f32(yb), xe, ye)

    a_comp = np.concatenate([rel_err(bf16_to_f32(xa), xe), rel_err(bf16_to_f32(ya), ye)])
    b_comp = np.concatenate([rel_err(bf16_to_f32(xb), xe), rel_err(bf16_to_f32(yb), ye)])

    exact_match = int(np.sum((xa == xb) & (ya == yb)))

    return {
        "n": n,
        "a_pair": a_pair,
        "b_pair": b_pair,
        "a_comp": a_comp,
        "b_comp": b_comp,
        "exact_match_frac": exact_match / n,
    }


# ---------------------------------------------------------------------------
# 3: error vs m and vs i
# ---------------------------------------------------------------------------


def study_vs_m(d=128, base=DEFAULT_BASE, per_bucket=4000, seed=2):
    rng = np.random.default_rng(seed)
    phi = phi_table(d, base)
    rows = []
    for m in (0, 1, 2, 4, 8, 16, 64, 256, 1024, 2048, 4095):
        x, y = gen_inputs(rng, per_bucket)
        i = rng.integers(0, d // 2, size=per_bucket)
        xa, ya = rope_strict_bf16(x, y, m, i, phi)
        xe, ye = rope_exact(x, y, m, i, d, base)
        e = pair_rel_err(bf16_to_f32(xa), bf16_to_f32(ya), xe, ye)
        rows.append((m, e.max(), np.sqrt(np.mean(e**2))))
    return rows


def study_vs_i(d=128, base=DEFAULT_BASE, per_bucket=4000, max_m=4096, seed=3):
    rng = np.random.default_rng(seed)
    phi = phi_table(d, base)
    rows = []
    for i in (0, 1, 2, 4, 8, 16, 32, 48, 63):
        x, y = gen_inputs(rng, per_bucket)
        m = rng.integers(0, max_m, size=per_bucket)
        xa, ya = rope_strict_bf16(x, y, m, i, phi)
        xe, ye = rope_exact(x, y, m, i, d, base)
        e = pair_rel_err(bf16_to_f32(xa), bf16_to_f32(ya), xe, ye)
        theta = base ** (-2.0 * i / d)
        rows.append((i, theta, e.max(), np.sqrt(np.mean(e**2))))
    return rows


# ---------------------------------------------------------------------------
# 4: the cancellation study
# ---------------------------------------------------------------------------


def find_m_near_45deg(phi_i, target_deg=45.0):
    """Smallest m whose phase lands nearest `target_deg` for this frequency."""
    target_phase = (target_deg / 360.0) * 2.0**32
    best, best_err = 1, None
    for m in range(1, 20000):
        ph = (m * int(phi_i)) & 0xFFFFFFFF
        err = min(abs(ph - target_phase), abs(ph + 2.0**32 - target_phase))
        if best_err is None or err < best_err:
            best, best_err = m, err
        if best_err < 2.0**20:  # within one LUT bucket
            break
    return best


def study_cancellation(d=128, base=DEFAULT_BASE, seed=4):
    """x == y exactly, phase swept through 45 deg -- the worst case for x*cos - y*sin.

    At exactly 45 degrees cos == sin, so with x == y the two products cancel completely
    and x' should be 0. With 7 stored mantissa bits the difference retains essentially
    no significant digits: this is where strict BF16 is at its worst.
    """
    phi = phi_table(d, base)
    out = {}

    # Sweep the phase finely across 45 degrees using the phase word directly, which
    # isolates the cancellation from any question of which m reaches that angle.
    n = 20001
    centre = int((45.0 / 360.0) * 2.0**32)
    span = int((2.0 / 360.0) * 2.0**32)  # +/- 2 degrees
    phases = np.linspace(centre - span, centre + span, n).astype(np.int64)
    phases = (phases & 0xFFFFFFFF).astype(np.uint32)

    rng = np.random.default_rng(seed)
    results = {}
    for label, ratio in (("x==y", 1.0), ("y=1.01x", 1.01), ("y=1.1x", 1.1)):
        # IMPORTANT: x must have a non-trivial BF16 mantissa. If x were exactly 1.0 (or
        # any power of two) then x*cos and y*sin would be *exact* products, the strict
        # and wide models would agree by construction, and the comparison would measure
        # nothing. Draw x uniformly from [1,2) so all 7 stored mantissa bits are in use.
        xv = rng.uniform(1.0, 2.0, size=n)
        x = to_bf16(xv)
        # y = ratio*x, rounded to BF16. For ratio == 1.0 this makes y bit-identical to x,
        # which is the exact-cancellation case.
        y = to_bf16(bf16_to_f32(x).astype(np.float64) * ratio)
        c = lut_cos(phases)
        s = lut_sin(phases)

        # Strict BF16, node by node (mirrors rope_strict_bf16 but with a given phase).
        from bf16 import bf16_add, bf16_mul, bf16_sub

        p0 = bf16_mul(x, c)
        p1 = bf16_mul(y, s)
        p2 = bf16_mul(x, s)
        p3 = bf16_mul(y, c)
        xa = bf16_sub(p0, p1)
        ya = bf16_add(p2, p3)

        # Wide accumulate.
        xf = bf16_to_f32(x).astype(np.float32)
        yf = bf16_to_f32(y).astype(np.float32)
        cf = bf16_to_f32(c).astype(np.float32)
        sf = bf16_to_f32(s).astype(np.float32)
        xb = to_bf16(xf * cf - yf * sf)
        yb = to_bf16(xf * sf + yf * cf)

        # Exact, using the true angle the phase word represents.
        ang = 2 * np.pi * (phases.astype(np.float64) / 2.0**32)
        xd = bf16_to_f32(x).astype(np.float64)
        yd = bf16_to_f32(y).astype(np.float64)
        xe = xd * np.cos(ang) - yd * np.sin(ang)
        ye = xd * np.sin(ang) + yd * np.cos(ang)

        ra = rel_err(bf16_to_f32(xa), xe)
        rb = rel_err(bf16_to_f32(xb), xe)
        ba = bits_retained(bf16_to_f32(xa), xe)
        bb = bits_retained(bf16_to_f32(xb), xe)
        results[label] = {
            "strict_max_rel": ra.max(),
            "strict_rms_rel": np.sqrt(np.mean(ra**2)),
            "wide_max_rel": rb.max(),
            "wide_rms_rel": np.sqrt(np.mean(rb**2)),
            "strict_bits_min": ba.min(),
            "strict_bits_mean": ba.mean(),
            "wide_bits_min": bb.min(),
            "wide_bits_mean": bb.mean(),
            "strict_frac_lost_half": float(np.mean(ba < 4)),
            "wide_frac_lost_half": float(np.mean(bb < 4)),
        }
    out["phase_sweep"] = results

    # Ratio sweep across the BF16 exponent range at the worst phase.
    worst_phase = np.uint32(centre)
    ratios = []
    from bf16 import bf16_mul, bf16_sub

    for exp in (-20, -10, -4, 0, 4, 10, 20):
        # Again, scale by a full-mantissa value rather than a bare power of two, so the
        # products genuinely round. 1.4140625 = 1.0110101b uses all 7 mantissa bits.
        scale = (2.0**exp) * 1.4140625
        x = to_bf16(np.full(2001, scale))
        rats = np.linspace(0.99, 1.01, 2001)
        y = to_bf16(scale * rats)
        c = lut_cos(np.full(2001, worst_phase, dtype=np.uint32))
        s = lut_sin(np.full(2001, worst_phase, dtype=np.uint32))
        xa = bf16_sub(bf16_mul(x, c), bf16_mul(y, s))
        ang = 2 * np.pi * (float(worst_phase) / 2.0**32)
        xe = bf16_to_f32(x).astype(np.float64) * np.cos(ang) - bf16_to_f32(y).astype(
            np.float64
        ) * np.sin(ang)
        b = bits_retained(bf16_to_f32(xa), xe)
        ratios.append((exp, b.min(), b.mean()))
    out["exponent_sweep"] = ratios

    # Which m actually reaches ~45 degrees for a few frequencies.
    out["m_at_45deg"] = [(i, find_m_near_45deg(phi[i])) for i in (0, 1, 8, 32, 63)]
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def pct(x):
    return f"{100*x:.4f}%"


def build_report(d, base, broad, vs_m, vs_i, canc):
    L = []
    w = L.append
    w("# Strict-BF16 RoPE — Error Budget")
    w("")
    w(f"Generated by `model/characterize.py` (d={d}, base={base:g}).")
    w("")
    w("Definitions:")
    w("")
    w("* **Model A** — strict BF16, rounded RNE at every node. This is the hardware (INV-1).")
    w("* **Model B** — BF16 in/out, FP32 internal accumulate (`AccWidth=32`, measurement only).")
    w("* **Model C** — FP64 with libm `sin`/`cos`. The yardstick.")
    w("* **Pair-relative error** — `|(x',y')_A - (x',y')_C| / |(x',y')_C|`, i.e. error")
    w("  normalised by the pair norm. Preferred over per-component relative error, which")
    w("  is unbounded when a component's true value is near zero under cancellation.")
    w("")

    w("## 1. Overall error (random operands in [-4, 4])")
    w("")
    w(f"{broad['n']} random `(x, y, m, i)` tuples, m in [0, 4096), i in [0, {d//2}).")
    w("")
    w("| Metric | Model A (strict) | Model B (wide acc) |")
    w("|---|---|---|")
    w(
        f"| Pair-relative error, max | {broad['a_pair'].max():.6e} | "
        f"{broad['b_pair'].max():.6e} |"
    )
    w(
        f"| Pair-relative error, RMS | {np.sqrt(np.mean(broad['a_pair']**2)):.6e} | "
        f"{np.sqrt(np.mean(broad['b_pair']**2)):.6e} |"
    )
    w(
        f"| Pair-relative error, mean | {np.mean(broad['a_pair']):.6e} | "
        f"{np.mean(broad['b_pair']):.6e} |"
    )
    w(
        f"| Component-relative error, max | {broad['a_comp'].max():.6e} | "
        f"{broad['b_comp'].max():.6e} |"
    )
    w(
        f"| Component-relative error, RMS | {np.sqrt(np.mean(broad['a_comp']**2)):.6e} | "
        f"{np.sqrt(np.mean(broad['b_comp']**2)):.6e} |"
    )
    w("")
    ratio = np.sqrt(np.mean(broad["a_pair"] ** 2)) / np.sqrt(np.mean(broad["b_pair"] ** 2))
    w(
        f"**Cost of INV-1 (strict vs wide), RMS pair error: {ratio:.3f}x.** "
        f"Strict BF16 and wide accumulate produce bit-identical results on "
        f"{pct(broad['exact_match_frac'])} of these random cases."
    )
    w("")
    w(
        "Interpretation: for generic operands the dominant error term is the shared "
        "LUT quantisation plus the final BF16 rounding of the result, which both models "
        "pay. The extra intermediate roundings that INV-1 accepts contribute comparatively "
        "little away from cancellation — which is why the cancellation study below is the "
        "measurement that matters."
    )
    w("")

    w("## 2. Error vs sequence position m")
    w("")
    w("| m | pair-rel max | pair-rel RMS |")
    w("|---|---|---|")
    for m, mx, rms in vs_m:
        w(f"| {m} | {mx:.6e} | {rms:.6e} |")
    w("")
    w(
        "Flat in m, as intended: the integer phase accumulator is exact, so there is no "
        "drift with sequence position (contrast the trigonometric recurrence, which "
        "steps.md Section 6.3 rejects for exactly this reason)."
    )
    w("")

    w("## 3. Error vs pair index i")
    w("")
    w("| i | theta_i | pair-rel max | pair-rel RMS |")
    w("|---|---|---|---|")
    for i, th, mx, rms in vs_i:
        w(f"| {i} | {th:.6e} | {mx:.6e} | {rms:.6e} |")
    w("")

    w("## 4. Catastrophic cancellation (the headline measurement)")
    w("")
    w(
        "`x' = x*cos - y*sin`. At 45 degrees `cos == sin`, so with `x == y` the two "
        "products cancel exactly and the true result is 0. Phase swept +/-2 degrees "
        "around 45 degrees in 20001 steps."
    )
    w("")
    w("| operands | model | max rel err | RMS rel err | min bits kept | mean bits kept | frac. losing >half |")
    w("|---|---|---|---|---|---|---|")
    for label, r in canc["phase_sweep"].items():
        w(
            f"| {label} | strict BF16 | {r['strict_max_rel']:.3e} | {r['strict_rms_rel']:.3e} | "
            f"{r['strict_bits_min']:.1f} | {r['strict_bits_mean']:.1f} | {pct(r['strict_frac_lost_half'])} |"
        )
        w(
            f"| {label} | wide acc | {r['wide_max_rel']:.3e} | {r['wide_rms_rel']:.3e} | "
            f"{r['wide_bits_min']:.1f} | {r['wide_bits_mean']:.1f} | {pct(r['wide_frac_lost_half'])} |"
        )
    w("")
    w(
        "'bits kept' is `-log2(relative error)`, clipped to [0, 24]; 'frac. losing >half' "
        "is the fraction of cases retaining fewer than 4 of BF16's 8 significand bits."
    )
    w("")
    w("### Cancellation across the exponent range (at 45 degrees, y/x in [0.99, 1.01])")
    w("")
    w("| x magnitude | min bits kept | mean bits kept |")
    w("|---|---|---|")
    for exp, bmin, bmean in canc["exponent_sweep"]:
        w(f"| 2^{exp} | {bmin:.1f} | {bmean:.1f} |")
    w("")
    w(
        "Scale-invariant, as expected for a floating-point format: cancellation depends "
        "on the ratio y/x and the angle, not on the absolute magnitude."
    )
    w("")
    w("### Which positions actually hit 45 degrees")
    w("")
    w("| i | smallest m within one LUT bucket of 45 deg |")
    w("|---|---|")
    for i, m in canc["m_at_45deg"]:
        w(f"| {i} | {m} |")
    w("")
    w(
        "Low-i (high-frequency) pairs reach the cancellation-worst angle within the first "
        "few positions, so this is not a rare corner: it is hit constantly in normal use."
    )
    w("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--base", type=float, default=DEFAULT_BASE)
    ap.add_argument("--n", type=int, default=400_000)
    ap.add_argument("--out", default=None, help="write the markdown report here")
    args = ap.parse_args()

    print(f"characterising d={args.d} base={args.base:g} n={args.n} ...", file=sys.stderr)
    broad = study_broad(args.d, args.base, args.n)
    vs_m = study_vs_m(args.d, args.base)
    vs_i = study_vs_i(args.d, args.base)
    canc = study_cancellation(args.d, args.base)

    report = build_report(args.d, args.base, broad, vs_m, vs_i, canc)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
