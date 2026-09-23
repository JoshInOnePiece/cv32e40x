#!/usr/bin/env python3
"""Validate the exact BF16 FMA model, with particular attention to signs.

Three questions:
  1. Does the float64 fast path agree with exact integer arithmetic everywhere?
  2. Is every combination of signs on x, y, sin and cos handled -- all four quadrants?
  3. Is the fused form actually closer to the exact answer than the 3-rounding form?
"""
import sys

import numpy as np

import bf16
from bf16_fma import bf16_fma, bf16_fma_exact
import rope_ref as R

FAIL = 0


def check(cond, msg):
    global FAIL
    if cond:
        print("  PASS  %s" % msg)
    else:
        FAIL += 1
        print("  FAIL  %s" % msg)


# ---------------------------------------------------------------------------
print("1. float64 fast path vs exact integer FMA")
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)

SPECIALS = np.array([
    0x0000, 0x8000,                  # +/-0
    0x0001, 0x8001,                  # smallest subnormal
    0x007F, 0x807F,                  # largest subnormal
    0x0080, 0x8080,                  # smallest normal
    0x3F80, 0xBF80,                  # +/-1
    0x7F7F, 0xFF7F,                  # largest finite
    0x7F80, 0xFF80,                  # +/-inf
    0x7FC0,                          # qNaN
], dtype=np.uint16)

n = 40000
a = rng.integers(0, 1 << 16, n, dtype=np.uint16)
b = rng.integers(0, 1 << 16, n, dtype=np.uint16)
c = rng.integers(0, 1 << 16, n, dtype=np.uint16)

# Cross product of specials, so the boundary cases are covered densely.
sa, sb, sc = np.meshgrid(SPECIALS, SPECIALS, SPECIALS, indexing="ij")
a = np.concatenate([a, sa.ravel()])
b = np.concatenate([b, sb.ravel()])
c = np.concatenate([c, sc.ravel()])

for sub in (False, True):
    fast = bf16_fma(a, b, c, sub=sub)
    exact = np.array([bf16_fma_exact(a[k], b[k], c[k], sub) for k in range(a.size)],
                     dtype=np.uint16)
    mism = np.flatnonzero(fast != exact)
    check(mism.size == 0,
          "sub=%-5s : %d triples, %d mismatches" % (sub, a.size, mism.size))
    if mism.size:
        for k in mism[:5]:
            print("        a=0x%04x b=0x%04x c=0x%04x  fast=0x%04x exact=0x%04x"
                  % (a[k], b[k], c[k], fast[k], exact[k]))

# ---------------------------------------------------------------------------
print("\n2. sign coverage across all four quadrants")
# ---------------------------------------------------------------------------
D = 128
phi = R.phi_table(D)

rng2 = np.random.default_rng(11)
N = 200000
m = rng2.integers(0, 65536, N, dtype=np.uint32)
i = rng2.integers(0, D // 2, N, dtype=np.uint32)
x = rng2.integers(0, 1 << 16, N, dtype=np.uint16)
y = rng2.integers(0, 1 << 16, N, dtype=np.uint16)

# Keep the sign study on finite inputs; specials are covered in part 1.
finite = (~bf16.is_nan(x)) & (~bf16.is_inf(x)) & (~bf16.is_nan(y)) & (~bf16.is_inf(y))
m, i, x, y = m[finite], i[finite], x[finite], y[finite]

ph = R.phase_of(m, i, phi)
c_lut = R.lut_cos(ph)
s_lut = R.lut_sin(ph)

quad = (ph >> 30).astype(int)
for q in range(4):
    sel = quad == q
    cn = ((c_lut[sel] >> 15) & 1).astype(bool)
    sn = ((s_lut[sel] >> 15) & 1).astype(bool)
    lbl = {0: "cos>0 sin>0", 1: "cos<0 sin>0", 2: "cos<0 sin<0", 3: "cos>0 sin<0"}[q]
    print("  Q%d (%s): %7d cases, cos<0 %5.1f%%, sin<0 %5.1f%%"
          % (q, lbl, sel.sum(), 100 * cn.mean(), 100 * sn.mean()))

# Four sign combinations of (x, y) crossed with all quadrants, checked exactly.
xs = x.copy()
ys = y.copy()
combos = 0
for xneg in (False, True):
    for yneg in (False, True):
        xx = (xs | np.uint16(0x8000)) if xneg else (xs & np.uint16(0x7FFF))
        yy = (ys | np.uint16(0x8000)) if yneg else (ys & np.uint16(0x7FFF))
        # Fused form: x' = fma(x, cos, -round(y*sin)),  y' = fma(x, sin, +round(y*cos))
        p1 = bf16.bf16_mul(yy, s_lut)
        p2 = bf16.bf16_mul(yy, c_lut)
        xo_fast = bf16_fma(xx, c_lut, p1, sub=True)
        yo_fast = bf16_fma(xx, s_lut, p2, sub=False)
        # Spot-check a slice against the exact integer model.
        sl = slice(0, 4000)
        xo_ex = np.array([bf16_fma_exact(xx[k], c_lut[k], p1[k], True)
                          for k in range(*sl.indices(xx.size))], dtype=np.uint16)
        yo_ex = np.array([bf16_fma_exact(xx[k], s_lut[k], p2[k], False)
                          for k in range(*sl.indices(xx.size))], dtype=np.uint16)
        ok = np.array_equal(xo_fast[sl], xo_ex) and np.array_equal(yo_fast[sl], yo_ex)
        check(ok, "x%s y%s : fused form exact over all quadrants"
              % ("<0" if xneg else ">0", "<0" if yneg else ">0"))
        combos += 1

# ---------------------------------------------------------------------------
print("\n3. accuracy: fused (2 roundings) vs current (3 roundings)")
# ---------------------------------------------------------------------------
sel = slice(0, 200000)
xx, yy = x[sel], y[sel]
cc, ss = c_lut[sel], s_lut[sel]

xf = bf16.bf16_to_f32(xx).astype(np.float64)
yf = bf16.bf16_to_f32(yy).astype(np.float64)
cf = bf16.bf16_to_f32(cc).astype(np.float64)
sf = bf16.bf16_to_f32(ss).astype(np.float64)
ref = xf * cf - yf * sf                       # FP64 truth

now = bf16.bf16_sub(bf16.bf16_mul(xx, cc), bf16.bf16_mul(yy, ss))
fma = bf16_fma(xx, cc, bf16.bf16_mul(yy, ss), sub=True)

nowf = bf16.bf16_to_f32(now).astype(np.float64)
fmaf = bf16.bf16_to_f32(fma).astype(np.float64)

good = np.isfinite(ref) & np.isfinite(nowf) & np.isfinite(fmaf) & (np.abs(ref) > 0)
rel_now = np.abs(nowf[good] - ref[good]) / np.abs(ref[good])
rel_fma = np.abs(fmaf[good] - ref[good]) / np.abs(ref[good])

rms_now = float(np.sqrt(np.mean(rel_now ** 2)))
rms_fma = float(np.sqrt(np.mean(rel_fma ** 2)))

print("  cases compared        : %d" % good.sum())
print("  RMS rel error, current: %.4e" % rms_now)
print("  RMS rel error, fused  : %.4e" % rms_fma)
print("  improvement           : %.3fx" % (rms_now / rms_fma))
print("  identical results     : %.1f%%" % (100.0 * np.mean(now[good] == fma[good])))
check(rms_fma < rms_now, "fused form is closer to the exact answer")

print()
if FAIL:
    print("FAILURES: %d" % FAIL)
    sys.exit(1)
print("All BF16 FMA model tests passed.")
