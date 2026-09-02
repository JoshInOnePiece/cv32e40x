# BF16 RoPE Coprocessor — Design Notes

Decisions, measurements and deviations. Anything here that contradicts `steps.md` is
called out explicitly, with the evidence.

---

## 1. Subnormal policy — MEASURED, not assumed

**Policy: gradual underflow. Subnormals are preserved on input and output. No
flush-to-zero.**

`steps.md` Section 5.3 asks for a subnormal policy to be "chosen and documented", and
notes that flush-to-zero is defensible and cheaper. The policy is not actually ours to
choose freely: it is whatever the vendored CVFPU FP16ALT unit does, and that had to be
determined empirically rather than assumed.

How it was established: `model/gen_vectors.py` emits an exhaustive cross product of 21
special BF16 values (both zeros, both infinities, several NaN payloads, min/max
subnormal, min/max normal, exact powers of two, and full-mantissa values) against each
other for MUL, ADD and SUB — including cases whose *results* are subnormal, e.g.
`0x0080 (2^-126) * 0x3F00 (0.5) = 0x0040`, a subnormal. `tb_rope_bf16_unit` compares all
of them bit-exactly against `model/bf16.py` configured with `flush_subnormals=False`,
and gets **zero mismatches**. Had FPnew flushed, every subnormal-result case would have
disagreed.

Consequences:

* The golden model's default (`flush_subnormals=False`) is the correct model of the
  hardware, so `to_bf16()` and friends need no FTZ flag in normal use.
* This differs from Mugi's post-processing block, which muxes Zero/INF/NaN explicitly.
  If consistency with Mugi is later judged more important than consistency with CVFPU,
  the change belongs in `rope_bf16_unit.sv` (or a post-stage), and
  `flush_subnormals=True` in the model would then match it — the model already supports
  that path and it is tested (`test_bf16.py::test_subnormals`).

**NaN canonicalisation:** FPnew emits the canonical quiet NaN `0x7FC0` for every NaN
result, and this too is confirmed bit-exactly (including the awkward case of a NaN whose
payload lives entirely in the low 16 bits of the FP32 pattern, which a naive
"add 0x7FFF and shift" narrowing turns into an *infinity*). Both `model/bf16.py` and
`sw/rope_intrin.h` handle that case explicitly.

---

## 2. Corrections to `steps.md`

### 2.1 INV-2's numerical example is overstated (the invariant is still right)

`steps.md` Section 0, INV-2 states:

> For pair `i=0`, `theta_0 = 1.0` rad. At sequence position `m = 1000` the angle is
> `1000` rad. […] `1000 * 2^-8 ~= 3.9 radians` […] The representable-value spacing
> exceeds a full period of 2*pi.

**3.9 does not exceed 2π ≈ 6.283.** Measured BF16 spacings (see
`test_rope_ref.py::test_inv2_bf16_phase_is_unrepresentable`, which asserts these):

| m | BF16 spacing | > π/2 | > π | > 2π |
|---|---|---|---|---|
| 100 | 0.5 rad | no | no | no |
| 500 | 2.0 rad | yes | no | no |
| 1000 | **4.0 rad** | yes | yes | **no** |
| 1024 | **8.0 rad** | yes | yes | **yes** |
| 4096 | 32.0 rad | yes | yes | yes |

So at m=1000 the spacing is 4.0 rad, which exceeds a *half* period; the full-period claim
becomes true from **m ≥ 1024**. The correct statement for a writeup is either "exceeds a
half period at m=1000" or "exceeds a full period from m=1024".

**The invariant is unaffected.** A 4 rad quantum on an angle means the induced angle
error is up to ±2 rad (±115°), so the resulting sin/cos are uncorrelated with the true
values — the output is noise, not a degraded result. The argument-reduction budget in
`steps.md` (`log2(4096) + 10 ≈ 22` bits needed, BF16 supplies 8) is correct as written and
is the more robust way to state the case.

### 2.2 Phi_min is ~79000, not ~68000

`steps.md` Section 6.1 justifies 32 bits over 24 with "`Phi_min ~ 68000` at 32 bits".
The measured value for d=128, base=10000 is **78937** (`test_phase_table`). The
conclusion — 32 bits gives ~1e-5 relative quantisation versus ~0.4% at 24 bits — holds.

### 2.3 Midpoint sampling makes m=0 a near-identity, not an identity

Not an error in `steps.md`, but a consequence it does not mention and which will look
like a bug to the next reader.

Because the table stores bucket *centres*, `sin(0)` reads `T[0] = sin(0.5·(π/2)/1024) ≈
7.667e-4` rather than 0. (`cos(0)` reads `T[1023]` and does round to exactly 1.0.) So at
sequence position 0 the rotation is not the identity:

```
x' = x·1.0 − y·7.667e-4
```

The absolute error is bounded by `|y|·sin(0)`, so the *relative* error in `x'` is
amplified by `|y|/|x|` and is **not** bounded by a small constant: with x=0.25, y=8.0 the
relative error is 2.3%. With `|y| ≤ |x|` it is under 0.5%.

This is the price of the free reflection (`N-1-idx` exactly, no 1025th entry, no endpoint
special case) and of halving the mean sampling error. It is the right trade, but if exact
behaviour at m=0 ever matters, the alternatives are edge sampling with a 1025-entry table
or a special case for phase 0.

### 2.4 `PosBits` is 16, not 13 — the ISA field wins over the expected sequence length

`steps.md` Section 6.3 justifies a narrow position register: "`m` is ~13 bits, `Phi` is 32
bits — a 13x32 multiplier, cheap." 13 bits does cover an 8192-token context, and that is
where this design started.

It is the wrong width, because **the instruction carries 16 bits of `m`** (`rs2[31:16]`).
With a 13-bit register the coprocessor silently truncates: `m = 10000` is taken as
`10000 & 0x1FFF = 1808`, producing a confidently wrong rotation with no error signalled
anywhere — not an exception, not a flag. And 8192-token contexts are ordinary, so this is
reachable in normal use rather than a corner case.

`rope_pkg::PosBits` is therefore **16**, matching the ISA field exactly. Three extra
multiplier bits (16×32 rather than 13×32) is a negligible price for deleting a whole class
of silent wrongness. The vector generators now sweep the full 16-bit range and include
`m ∈ {8191, 8192, 8193, 10000, 16384, 32768, 60000, 65535}` explicitly, so the 13-bit
behaviour would now fail M2/M4/M5/M7 rather than pass unnoticed.

If a future configuration really wants a narrower multiplier, narrow `PosBits` *and*
range-check `m` — do not rely on callers behaving.

### 2.5 Repository layout and dependencies

Deviations from `steps.md` Section 3 / 2.3, all in service of the project being
self-contained inside the `cv32e40x` checkout:

| `steps.md` says | What was done | Why |
|---|---|---|
| `deps/` with 4 cloned repos | nothing cloned | CVFPU sources are **copied** into `rtl/vendor/` (never symlinked). The core and XIF definitions already live in this repository. |
| `Bender.yml` with `cvfpu` as a git dependency; "don't vendor it" | vendored, no Bender | Directly contradicts the requirement that all changes be self-contained in this repo. `gen/vendor_cvfpu.sh` records exactly which files were copied and can refresh them. |
| use `deps/core-v-xif/src/core_v_xif.sv` | use `cv32e40x/rtl/cv32e40x_if_xif.sv` | The core ships an equivalent interface definition. Using the core's own copy makes every struct type-compatible by construction and needs no external clone. |
| `rope_sin_lut.sv` is generated | `rope_sin_rom.sv` generated; `rope_sin_lut.sv` hand-written | Keeps generated code trivially auditable (a bare table) and the interesting logic (quadrant decode, reflection, sign) reviewable. |
| `ml_dtypes`, `torch`, `cocotb`, `pytest` | numpy only | `ml_dtypes` could not be installed (PEP 668) and is not needed: `model/bf16.py` implements BF16 RNE by bit manipulation and is cross-checked against an independently written scalar reference. No pytest/cocotb, so the test suites are plain runnable Python and self-checking SystemVerilog. |

### 2.6 `AccWidth=32` lives in the model, not the RTL

`steps.md` Section 8.3 asks for an `AccWidth` parameter on `rope_datapath` with a
`32` setting for the wide-accumulate comparison. **The RTL implements strict BF16 only**
and `rope_datapath` fails elaboration with a clear message if `AccWidth != 16`.

Reason: a wide-accumulate hardware path needs FP32 adders plus BF16↔FP32 casts, i.e.
enabling FPnew's CONV opgroup and adding exactly the conversion stages INV-1 forbids —
for a configuration that is explicitly never shipped. The comparison itself is what
matters and it *is* produced, by `model/rope_ref.py::rope_wide_acc` (Model B) and
reported in `docs/error_budget.md`. The parameter is retained so the intent stays visible
and a future implementer is not misled into thinking `32` works.

---

## 3. Two bugs worth remembering

Both were found by tests that existed specifically to check assumptions about vendored
code, and neither would have produced an obvious symptom.

### 3.1 FPnew's ADD ignores operand slot 0

FPnew's ADDMUL group is an FMA computing `operands[0]*operands[1] + operands[2]`, and
`fpnew_fma` rewrites operands per opcode:

* `MUL`: forces `operand_c = ±0` → result is `operands[0] * operands[1]`.
* `ADD`: forces **`operand_a = +1.0`** → result is `operands[1] + operands[2]`.

So an adder must place its addends in slots **1 and 2**. Putting them in 0 and 1 — the
obvious choice — silently computes `1.0 * b + 0`, discarding operand `a` entirely. Handled
once, in `rope_bf16_unit.sv`, with the reasoning written down next to the code.

`SUB` uses `op_mod`, which inverts the sign of operand C. That is exact and free.

### 3.2 FPnew's `in_ready` is gated by its own `in_valid` — a deadlock trap

`fpnew_top` drives:

```systemverilog
assign in_ready_o = in_valid_i & opgrp_in_ready[get_opgroup(op_i)];
```

Ready depends on valid. So the natural-looking datapath input register

```systemverilog
valid_s0 <= in_valid_i & in_ready_o;   // WRONG
```

is a combinational cycle through the FPnew instances: at reset `valid=0` → `ready=0` →
`valid` stays 0 forever. The datapath deadlocks and never produces a single result. This
is what the first `tb_rope_datapath` run hit (`sent=0` at the watchdog).

Fix and its justification: `in_ready_o` is a constant 1, because `out_ready_i` is tied to
1 on every FPnew instance, which makes every input-pipeline stage unconditionally ready,
so a unit accepts an operand on every cycle it is offered one. Since that is an assumption
about vendored code, `rope_datapath.sv` **asserts it every cycle**: whenever `valid_s0` is
high, all four multipliers must report `in_ready`. If a CVFPU bump ever introduces a
stall, that assertion fires instead of pairs being silently dropped.

The back-pressure the design does need is handled one level up, by the credit counter in
`rope_xif_coproc.sv` — which is also what the XIF rules force, since `commit_valid` and
`mem_result_valid` have no ready signal and the coprocessor must always be able to accept.

---

## 4. Verification status

| Milestone | What | Result |
|---|---|---|
| M0 | golden model, 3 models, LUT/phase properties, HF convention cross-check | **PASS** (`make model`) |
| M0 | error budget, strict vs wide, cancellation study | **DONE** (`docs/error_budget.md`) |
| M1 | BF16 mul/add/sub vs model, incl. exhaustive specials + directed exact ties | **PASS**, zero mismatches (**1,001,935** vectors) |
| M2 | integer phase gen: random-access, sequential, wraparound | **PASS**, bit-exact |
| M3 | sine LUT: bit-exact, reflection symmetry, quadrant signs, low-bit isolation | **PASS** |
| M4 | full datapath vs Model A, incl. cancellation corner, at full throughput | **PASS**, zero mismatches (**1,001,961** vectors) |
| M5 | XIF: decode, writeback, kill-mid-operation, back-to-back, backpressure, SVA | **PASS** |
| M6 | integration lint: real core + XIF + coprocessor | **PASS**, zero warnings |
| M7 | C test on the integrated core (`make m7`) | **PASS** |
| M7 | `.insn` encoding verified against `rope_pkg`'s decode | **PASS** |

Measured datapath latency: **3 cycles**, one rotated pair per cycle after fill.
Coprocessor latency issue→writeback: 4 cycles.

The largest runs completed so far are 10^6 vectors each on M1 and M4 (both zero
mismatches). `steps.md` asks for 10^7; `make m1-full` / `make m4-full` do that and are
expected to take roughly ten times as long — the vector generation and `$fscanf` parsing
dominate, not the simulation.

### A Makefile trap that was fixed (worth knowing if you script runs)

Vector filenames embed `NVEC` (`vectors_bf16_1000000.hex`). Originally they did not, and
`make m1 NVEC=1000000` then found the existing 200k-vector file up to date, silently
reused it, and reported a confident pass over the wrong vector count. If you add a new
testbench, keep the count in the filename.

### Not done, and why

* **`core-v-verif` baseline regression pre/post integration** (M6 exit criterion) and
  **RVFI/Spike co-simulation.** `core-v-verif` is present at `~/core-v-verif` but its
  regression flow is UVM-based and expects a commercial simulator plus a configured
  toolchain/testbench environment; standing that up is a separate task from building the
  unit. This is the most important remaining verification gap: it is what would prove the
  coprocessor has not perturbed core state. What *is* established meanwhile: the
  integration elaborates cleanly, and the coprocessor is provably inert for
  non-ROPE instructions (`a_accept_matches_decode` in `tb/rope_assertions.svh` checks that
  `accept` is asserted for exactly the ROPE.ROT encodings and nothing else, on every
  completed handshake).
* **Milestone 8 (streaming variant).** Not started. The memory channels are tied off and
  `a_no_memory_transactions` guards the tie-off.
* **Milestone 9 (synthesis: area/timing/power).** `syn/rope.sdc` and `syn/synth.tcl` are
  written but **have not been run** — no PDK/library is configured in this environment.
  The script needs `ROPE_LIB_PATH` pointing at a 45 nm library to reproduce Mugi's
  operating point.
* **Comparison against real HuggingFace LLaMA tensors.** The half-split convention is
  verified against a faithful reimplementation of `apply_rotary_pos_emb` /
  `rotate_half` in `model/rope_ref.py` (matching to 1e-12 in FP64), and the two
  conventions are proven equivalent up to the fixed permutation. Testing against actual
  extracted weights needs `torch` + network access + gated model access, none of which
  are available here.

---

## 5. Sizing summary

| Item | Size | Note |
|---|---|---|
| Sine LUT | 1024 × 16 bit = **2 KB** | one quarter wave, serves both sin and cos |
| Theta ROM | 64 × 32 bit = **256 B** | d=128 |
| Naive precomputed table | ~1 MB | `max_seq_len × d/2 × {sin,cos}`, d=128, seq=4096 |

The phase accumulator buys roughly a **500×** table reduction. For scale, Mugi's iSRAM
is 64 KB, so the LUT is ~3% of it.

LUT resolution is `(π/2)/1024 = 0.00153` rad, versus a BF16 ulp near 1.0 of
`2^-7 = 0.00781` — the table is already ~5× finer than BF16 can represent, so BF16's own
coarseness bounds the table size. 512 entries would also suffice.

Measured `sin²+cos²` deviation from 1.0: max **0.00528**, RMS 0.00213 (≈ 0.68 BF16 ulp).
Not 1.0, and that is expected in BF16 (`steps.md` 7.5). RTL and model agree exactly on
this figure.

---

## 6. Reproducing everything

```bash
cd rope-bf16

make model          # golden-model suites (numpy only)
make characterize   # regenerate docs/error_budget.md
make gen            # regenerate the theta ROM and sine LUT from gen/
make lint           # coprocessor lint
make lint-integration   # real core + XIF + coprocessor
make test           # RTL testbenches M1..M5
make m7             # C test on the integrated core

# Heavier runs (steps.md asks for 10^7 on the primitives and the datapath):
make m1 NVEC=1000000
make m1-full        # 10^7
make m4-full        # 10^7
```

`make test` defaults to `NVEC=200000` random vectors per testbench on top of the directed
and exhaustive sets, which runs in a couple of minutes.
