# BF16-Native RoPE Coprocessor for CV32E40X

A synthesizable, **BF16-only** Rotary Positional Embedding unit attached to CV32E40X over
CORE-V-XIF — the external RoPE offload path that the Mugi paper (arXiv:2601.10823,
Section 7.1) names but leaves unimplemented.

Implements the plan in [`../steps.md`](../steps.md). Everything is **self-contained inside
this `cv32e40x` checkout**: CVFPU is copied into `rtl/vendor/` (never symlinked), the XIF
interface comes from the core's own `cv32e40x_if_xif.sv`, and the golden model needs
nothing beyond numpy.

## Status

| Milestone | | |
|---|---|---|
| M0 | Golden model (3 models) + error budget | **PASS** |
| M1 | BF16 primitives (CVFPU FP16ALT) | **PASS** — zero mismatches |
| M2 | Integer phase generator | **PASS** — bit-exact, zero drift |
| M3 | BF16 sine LUT | **PASS** — bit-exact, exact reflection |
| M4 | BF16 datapath (4 mul + 2 add) | **PASS** — bit-exact vs Model A |
| M5 | CORE-V-XIF wrapper | **PASS** — kill, throughput, backpressure, SVA |
| M6 | Core integration | **PASS** — lint clean with the real core |
| M7 | C test on the integrated core | **PASS** |
| M8 | Streaming variant | not started |
| M9 | Synthesis (area/timing/power) | scripts written, **not run** (no PDK here) |

Measured: **3-cycle datapath latency**, one rotated pair per cycle after fill; 4 cycles
issue→writeback. Not yet run: the `core-v-verif` baseline regression and RVFI/Spike
co-simulation — see [`docs/notes.md`](docs/notes.md) §4 for exactly what that leaves open.

## Quick start

```bash
make model              # golden-model test suites (numpy only)
make test               # RTL testbenches M1..M5
make m7                 # C test on the integrated core
make lint-integration   # real core + XIF + coprocessor, lint clean
make characterize       # regenerate docs/error_budget.md
```

`make test` uses `NVEC=200000` random vectors per testbench on top of the directed and
exhaustive sets. Raise it with `make m1 NVEC=1000000`, or `make m1-full` / `make m4-full`
for the 10^7 runs `steps.md` asks for.

## The instruction

```
ROPE.ROT rd, rs1, rs2       # custom-0, opcode 0x0B, funct3=0, funct7=0
  rs1 = {x_bf16, y_bf16}    # native BF16 pair in,  NO conversion
  rs2 = {m[15:0], i[15:0]}  # sequence position and pair index (integers)
  rd  = {x'_bf16, y'_bf16}  # native BF16 pair out
```

BF16 packs two-per-GPR on RV32, so one source pair and one destination suffice: no dual
writeback, `X_RFW_WIDTH` stays 32. From C:

```c
#include "rope_intrin.h"
uint32_t out = rope_rot(rope_pack(x_bf16, y_bf16), m, i);
```

## How it works

```
        m, i (integers)                        x, y (BF16)
             |                                      |
      [rope_theta_rom]  Phi_i                       |
             |                                      |
      [rope_phase_gen]  P = (m * Phi_i) mod 2^32    |     <- INTEGER, exact (INV-2)
             |                                      |
      [rope_sin_lut]    sin, cos (BF16)             |     <- one quarter-wave table
             |                                      |
             +--------------> [rope_datapath] <-----+     <- 4 BF16 mul, 2 BF16 add
                                    |                        RNE at every node (INV-1)
                               x', y' (BF16)
```

Three ideas carry the design:

**The phase is an integer.** A BF16 angle is unusable: at m=1024 the spacing between
representable BF16 values near 1024 rad is 8.0 rad — larger than a full period, so the
angle is not approximated but destroyed. A 32-bit integer phase accumulator is exact, has
zero drift, and makes `mod 2π` free (natural wraparound). `phase[31:30]` is the quadrant,
`phase[29:20]` the LUT index.

**One small table, and BF16 bounds its size.** 1024 entries over a quarter turn give
0.00153 rad resolution, already ~5× finer than a BF16 ulp near 1.0 (0.0078) — so BF16's
own coarseness caps the useful table size. `cos(P) = sin(P + 2^30)`: same table, two
lookups. Midpoint sampling makes the Q1/Q3 reflection index exactly `1023-idx`, with no
endpoint special case. Total: **2 KB**, versus ~1 MB for the naive precomputed
`seq_len × d/2 × {sin,cos}` table — a ~500× reduction.

**Strict BF16, and only ~2 roundings.** Four multiplies and two adds, rounded RNE at every
node. That is far fewer rounding events than CORDIC's ~16 sequential stages, which is why
LUT+multiply is *more* accurate in BF16, not less.

## Layout

```
rope-bf16/
├── Makefile                    # build, lint, and all test targets
├── rtl/
│   ├── rope_pkg.sv             # params, opcode, packing helpers
│   ├── rope_theta_rom.sv       # GENERATED: integer Phi_i table (256 B)
│   ├── rope_sin_rom.sv         # GENERATED: BF16 quarter-wave table (2 KB)
│   ├── rope_sin_lut.sv         # quadrant decode, reflection, sign
│   ├── rope_phase_gen.sv       # integer phase: random-access + sequential
│   ├── rope_bf16_unit.sv       # CVFPU FP16ALT wrapper (all config lives here)
│   ├── rope_bf16_mul.sv        # BF16 multiply
│   ├── rope_bf16_add.sv        # BF16 add / subtract
│   ├── rope_datapath.sv        # 4 mul + 2 add, strict BF16
│   ├── rope_xif_coproc.sv      # CORE-V-XIF wrapper
│   ├── rope_cv32e40x_wrapper.sv# core + interface + coprocessor
│   └── vendor/                 # COPIED CVFPU + common_cells (not symlinked)
├── gen/
│   ├── gen_theta_rom.py        # emits rope_theta_rom.sv
│   ├── gen_sin_lut.py          # emits rope_sin_rom.sv
│   └── vendor_cvfpu.sh         # records/refreshes the vendored CVFPU subset
├── model/
│   ├── bf16.py                 # BF16 RNE by bit manipulation (no ml_dtypes needed)
│   ├── rope_ref.py             # Models A (strict) / B (wide acc) / C (FP64 exact)
│   ├── characterize.py         # -> docs/error_budget.md
│   ├── gen_vectors.py          # RTL test vectors
│   ├── gen_c_vectors.py        # -> sw/rope_vectors.h
│   └── test_*.py               # golden-model suites
├── tb/
│   ├── tb_rope_bf16_unit.sv    # M1
│   ├── tb_rope_phase_gen.sv    # M2
│   ├── tb_rope_sin_lut.sv      # M3
│   ├── tb_rope_datapath.sv     # M4
│   ├── tb_rope_xif_coproc.sv   # M5
│   ├── tb_rope_core.sv         # M7, C test on the integrated core
│   └── rope_assertions.svh     # XIF protocol SVA
├── sw/
│   ├── rope_intrin.h           # .insn wrappers, BF16 conversion, whole-head loops
│   ├── test_rope.c             # bare-metal test
│   ├── crt0.S, link.ld         # minimal startup and memory map
│   └── verify_encoding.py      # checks the assembled encoding against rope_pkg
├── syn/
│   ├── rope.sdc                # 400 MHz + the XIF timing budget
│   └── synth.tcl               # Genus flow (untested: no PDK here)
└── docs/
    ├── notes.md                # design decisions, steps.md corrections, gotchas
    └── error_budget.md         # GENERATED: the quantitative results
```

## Results worth knowing

From [`docs/error_budget.md`](docs/error_budget.md) (d=128, base=10000, 400k random tuples):

* Strict BF16 vs FP64 exact: **2.73e-3 RMS** pair-relative error, 9.6e-3 max.
* **Cost of strict BF16 over wide accumulate: 1.27× RMS.** They are bit-identical on 45%
  of random cases — away from cancellation, the shared LUT quantisation and the final
  rounding dominate, and both models pay those.
* Error is **flat in sequence position** m — the integer accumulator does not drift.
* Under cancellation (`x ≈ y`, phase ≈ 45°) strict BF16 keeps **3.5 bits on average** and
  loses more than half its significand 63% of the time, versus 3.7 bits / 60% for wide
  accumulate. Cancellation is scale-invariant, and the high-frequency pairs reach 45°
  within the first couple of hundred positions — so this is the common case, not a corner.

That strict-vs-wide comparison for RoPE specifically is the measurement `steps.md` argues
has not been published.

## Read this before changing anything

[`docs/notes.md`](docs/notes.md) records the non-obvious things, including two traps in
vendored CVFPU that produce silent wrongness rather than errors:

* FPnew's **ADD forces operand slot 0 to +1.0**, so an adder's operands belong in slots 1
  and 2. Using slots 0 and 1 computes `1.0*b + 0` and discards `a`.
* FPnew's **`in_ready` is gated by its own `in_valid`**, so the natural
  `valid <= in_valid & in_ready` input register is a combinational cycle that deadlocks
  the datapath at reset.

It also documents three places where `steps.md`'s own numbers needed correcting (INV-2's
worked example, `Phi_min`, and the fact that midpoint sampling makes m=0 a *near*-identity
rather than an identity), and the deliberate deviations from its dependency plan.
