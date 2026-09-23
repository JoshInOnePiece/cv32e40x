#!/usr/bin/env python3
"""Generate RTL test-vector files from the golden model.

The testbenches are self-checking: each vector carries the expected result computed by
the Python golden model, so the SV side never re-derives anything.

Subcommands:
    bf16      -> <op> <a> <b> <expected>            for tb_rope_bf16_unit
    phase     -> <m> <i> <expected_phase>           for tb_rope_phase_gen
    lut       -> <phase> <exp_sin> <exp_cos>        for tb_rope_sin_lut
    datapath  -> <x> <y> <m> <i> <exp_x> <exp_y>    for tb_rope_datapath
    xif       -> <rs1> <rs2> <exp_rd>               for tb_rope_xif_coproc

All fields are hex without a 0x prefix, one vector per line.
"""

import argparse
import sys

import numpy as np

from bf16 import bf16_add, bf16_mul, bf16_sub, to_bf16
from rope_ref import (
    DEFAULT_BASE,
    lut_cos,
    lut_sin,
    phase_of,
    phi_table,
    rope_fma_bf16,
    rope_strict_bf16,
)

# Interesting BF16 patterns that random sampling would rarely produce but that break
# naive implementations: zeros, infinities, NaNs, subnormals, the max/min normals, and
# exact powers of two (whose products are exact and so hide rounding bugs).
SPECIAL_BF16 = [
    0x0000,  # +0
    0x8000,  # -0
    0x7F80,  # +Inf
    0xFF80,  # -Inf
    0x7FC0,  # +qNaN
    0xFFC0,  # -qNaN
    0x7FC1,  # another NaN payload
    0x7F81,  # sNaN-ish payload
    0x0001,  # +min subnormal
    0x8001,  # -min subnormal
    0x007F,  # +max subnormal
    0x0080,  # +min normal
    0x8080,  # -min normal
    0x7F7F,  # +max normal
    0xFF7F,  # -max normal
    0x3F80,  # +1.0
    0xBF80,  # -1.0
    0x3F00,  # +0.5
    0x4000,  # +2.0
    0x3F81,  # 1 + 2^-7, the next BF16 after 1.0
    0x3FB5,  # 1.4140625, full mantissa
]


def _rng(seed):
    return np.random.default_rng(seed)


def _random_bf16(rng, n):
    """Random BF16 patterns, weighted toward ordinary finite values.

    A purely uniform draw over 2^16 patterns would spend ~0.4% of vectors on NaN/Inf and
    would rarely hit the near-cancellation cases that matter, so mix three populations.
    """
    n_uniform = n // 2
    n_narrow = n - n_uniform

    uniform = rng.integers(0, 1 << 16, size=n_uniform, dtype=np.uint16)
    # Values in a range typical of post-GEMM activations.
    narrow = to_bf16(rng.uniform(-8.0, 8.0, size=n_narrow))
    out = np.concatenate([uniform, narrow])
    rng.shuffle(out)
    return out.astype(np.uint16)


# ---------------------------------------------------------------------------


def gen_bf16(n, seed, f):
    """MUL / ADD / SUB vectors, including an exhaustive special-value cross product."""
    rng = _rng(seed)
    count = 0

    # 1. Every special x every special, for all three ops. This is the NaN/Inf/subnormal
    #    propagation check and the exact-tie check rolled together.
    for a in SPECIAL_BF16:
        for b in SPECIAL_BF16:
            av = np.array([a], dtype=np.uint16)
            bv = np.array([b], dtype=np.uint16)
            for op, fn in ((0, bf16_mul), (1, bf16_add), (2, bf16_sub)):
                e = int(fn(av, bv)[0])
                f.write(f"{op:x} {a:04x} {b:04x} {e:04x}\n")
                count += 1

    # 2. Directed exact ties. An FP32 sum/product landing exactly on a BF16 midpoint is
    #    where truncation-instead-of-RNE shows up, so hit it deliberately: x + x*2^-8
    #    style pairs across the exponent range.
    for exp in range(-100, 101, 4):
        base = 2.0**exp
        for mant in (1.0, 1.0078125, 1.5, 1.9921875):
            a = to_bf16(base * mant)
            b = to_bf16(base * mant * 2.0**-8)
            av = np.array([a], dtype=np.uint16)
            bv = np.array([b], dtype=np.uint16)
            for op, fn in ((0, bf16_mul), (1, bf16_add), (2, bf16_sub)):
                e = int(fn(av, bv)[0])
                f.write(f"{op:x} {int(a):04x} {int(b):04x} {e:04x}\n")
                count += 1

    # 3. Bulk random.
    a = _random_bf16(rng, n)
    b = _random_bf16(rng, n)
    ops = rng.integers(0, 3, size=n)
    r_mul = bf16_mul(a, b)
    r_add = bf16_add(a, b)
    r_sub = bf16_sub(a, b)
    for k in range(n):
        op = int(ops[k])
        e = int((r_mul, r_add, r_sub)[op][k])
        f.write(f"{op:x} {int(a[k]):04x} {int(b[k]):04x} {e:04x}\n")
        count += 1

    return count


def gen_phase(n, seed, f, d=128, base=DEFAULT_BASE):
    """Phase generator vectors: exhaustive small m, all i, plus wraparound cases."""
    phi = phi_table(d, base)
    half = d // 2
    count = 0

    def emit(m, i):
        nonlocal count
        p = int(phase_of(m, i, phi))
        f.write(f"{m:04x} {i:02x} {p:08x}\n")
        count += 1

    # All i for a spread of m, including 0 and the top of the ISA range.
    #
    # Values above 8191 are deliberately included: rope_pkg::PosBits is 16 to match the
    # instruction's rs2[31:16] field, and an earlier 13-bit version silently truncated
    # here (m=10000 -> 1808). These vectors are what would catch that regression.
    for m in list(range(0, 64)) + [
        100, 255, 256, 1000, 1023, 1024, 2048, 4095, 4096,
        8191, 8192, 8193, 10000, 16384, 32768, 60000, 65535,
    ]:
        for i in range(half):
            emit(m, i)

    # Wraparound: m values straddling multiples of 2^32 / Phi_i for the fast frequencies.
    for i in (0, 1, 2, 3):
        p0 = int(phi[i])
        m_wrap = (1 << 32) // p0
        for dm in (-2, -1, 0, 1, 2, 3):
            m = m_wrap + dm
            if 0 <= m <= 8191:
                emit(m, i)

    # Random fill.
    rng = _rng(seed)
    # Full 16-bit m range, matching the instruction field (see PosBits in rope_pkg.sv).
    ms = rng.integers(0, 65536, size=n)
    ids = rng.integers(0, half, size=n)
    for k in range(n):
        emit(int(ms[k]), int(ids[k]))

    return count


def gen_lut(n, seed, f):
    """Sine LUT vectors: exhaustive on the top 12 bits, plus quadrant boundaries."""
    count = 0
    phases = []

    # Exhaustive on the top 12 bits (quadrant + 10-bit index) with the unused low bits
    # zero: this covers every distinct table entry in every quadrant.
    for top in range(1 << 12):
        phases.append(top << 20)

    # Quadrant boundaries specifically, where reflection logic breaks if it is wrong.
    for q in range(4):
        for k in (0, 1, 2, 511, 512, 1021, 1022, 1023):
            base = (q << 30) | (k << 20)
            for low in (0, 1, 0xFFFFF, 0x80000):
                phases.append((base | low) & 0xFFFFFFFF)

    # Random fill over the whole 2^32 space, including nonzero low bits (which must be
    # ignored by the index extraction).
    rng = _rng(seed)
    phases.extend(int(v) for v in rng.integers(0, 1 << 32, size=n, dtype=np.uint64))

    ph = np.array(phases, dtype=np.uint32)
    s = lut_sin(ph)
    c = lut_cos(ph)
    for k in range(len(ph)):
        f.write(f"{int(ph[k]):08x} {int(s[k]):04x} {int(c[k]):04x}\n")
        count += 1
    return count


def gen_datapath(n, seed, f, d=128, base=DEFAULT_BASE, fma=False):
    """Full datapath vectors vs Model A, including the cancellation corner."""
    phi = phi_table(d, base)
    half = d // 2
    rng = _rng(seed)
    count = 0

    xs, ys, ms, ids = [], [], [], []

    # Specials against ordinary values.
    for a in SPECIAL_BF16:
        for b in (0x3F80, 0x3FB5, 0x0000, 0x7F80, 0x7FC0):
            xs.append(a)
            ys.append(b)
            ms.append(int(rng.integers(0, 4096)))
            ids.append(int(rng.integers(0, half)))

    # The cancellation corner: x == y (and near-equal) at phases near 45 degrees.
    # Find, per frequency, an m that lands near 45 degrees.
    for i in (0, 1, 2, 8, 32):
        target = int((45.0 / 360.0) * 2.0**32)
        best_m, best_e = 1, None
        for m in range(1, 20000):
            p = (m * int(phi[i])) & 0xFFFFFFFF
            e = min(abs(p - target), abs(p + 2**32 - target))
            if best_e is None or e < best_e:
                best_m, best_e = m, e
            if best_e < 2**20:
                break
        for mult in (1.0, 1.0 + 2**-7, 1.0 - 2**-7, 1.01, 0.99):
            xv = to_bf16(rng.uniform(1.0, 2.0, size=64))
            from bf16 import bf16_to_f32

            yv = to_bf16(bf16_to_f32(xv).astype(np.float64) * mult)
            for k in range(len(xv)):
                xs.append(int(xv[k]))
                ys.append(int(yv[k]))
                ms.append(best_m)
                ids.append(i)

    # m = 0 and small m for every i.
    for m in (0, 1, 2, 3):
        for i in range(half):
            xs.append(int(to_bf16(1.4140625)))
            ys.append(int(to_bf16(-0.7071)))
            ms.append(m)
            ids.append(i)

    # Bulk random.
    xr = _random_bf16(rng, n)
    yr = _random_bf16(rng, n)
    mr = rng.integers(0, 65536, size=n)  # full 16-bit m range, per PosBits
    ir = rng.integers(0, half, size=n)
    xs.extend(int(v) for v in xr)
    ys.extend(int(v) for v in yr)
    ms.extend(int(v) for v in mr)
    ids.extend(int(v) for v in ir)

    x = np.array(xs, dtype=np.uint16)
    y = np.array(ys, dtype=np.uint16)
    m = np.array(ms, dtype=np.uint64)
    i = np.array(ids, dtype=np.int64)

    xr_, yr_ = (rope_fma_bf16(x, y, m, i, phi) if fma
               else rope_strict_bf16(x, y, m, i, phi))
    for k in range(len(x)):
        f.write(
            f"{int(x[k]):04x} {int(y[k]):04x} {int(m[k]):04x} {int(i[k]):02x} "
            f"{int(xr_[k]):04x} {int(yr_[k]):04x}\n"
        )
        count += 1
    return count


def gen_xif(n, seed, f, d=128, base=DEFAULT_BASE):
    """XIF-level vectors: packed rs1/rs2 -> packed rd, matching rope_pkg's packing.

    rs1 = {x, y}, rs2 = {m, i}, rd = {x', y'}  (x in the HIGH half throughout).
    """
    phi = phi_table(d, base)
    half = d // 2
    rng = _rng(seed)

    x = _random_bf16(rng, n)
    y = _random_bf16(rng, n)
    m = rng.integers(0, 65536, size=n)  # full 16-bit m range, per PosBits
    i = rng.integers(0, half, size=n)

    xr, yr = rope_strict_bf16(x, y, m.astype(np.uint64), i, phi)
    for k in range(n):
        rs1 = (int(x[k]) << 16) | int(y[k])
        rs2 = (int(m[k]) << 16) | int(i[k])
        rd = (int(xr[k]) << 16) | int(yr[k])
        f.write(f"{rs1:08x} {rs2:08x} {rd:08x}\n")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["bf16", "phase", "lut", "datapath", "xif"])
    ap.add_argument("--n", type=int, default=100_000, help="number of random vectors")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fma", action="store_true",
                    help="datapath vectors for the UseFma=1 variant (Model A')")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--base", type=float, default=DEFAULT_BASE)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.out, "w") as f:
        if args.kind == "bf16":
            n = gen_bf16(args.n, args.seed, f)
        elif args.kind == "phase":
            n = gen_phase(args.n, args.seed, f, args.d, args.base)
        elif args.kind == "lut":
            n = gen_lut(args.n, args.seed, f)
        elif args.kind == "datapath":
            n = gen_datapath(args.n, args.seed, f, args.d, args.base, args.fma)
        else:
            n = gen_xif(args.n, args.seed, f, args.d, args.base)

    print(f"wrote {args.out}: {n} vectors", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
