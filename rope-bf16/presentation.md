# BF16 RoPE Coprocessor — Two-Slide Presentation

Slide content is below as ready-to-paste bullets. The screen-recording command is in
[Slide 2](#slide-2--metrics--proof) and the run instructions are in
[How to record](#how-to-record).

**One-line framing if you need it:** *A hardware accelerator for the position-encoding
step of a transformer, built entirely in BF16, driven by a new custom RISC-V instruction.*

---
---

# SLIDE 1 — Accomplished / Upcoming

## Accomplished

- **Built a working BF16-only RoPE coprocessor** for the CV32E40X RISC-V core,
  attached over CORE-V-XIF — the external offload path the Mugi paper (arXiv:2601.10823
  §7.1) names but does not implement.

- **Defined and implemented a new RISC-V custom instruction: `ROPE.ROT`**
  — R-type, **custom-0, opcode `0x0B`**, `funct3 = 0`, `funct7 = 0`.
  BF16 packs two-per-register on RV32, so one source and one destination register
  suffice — no dual-register writeback, interface stays 32-bit.

- **Solved the core numerical problem: the rotation angle cannot exist in BF16.**
  At token 1024, consecutive representable BF16 values are **8.0 radians** apart while a
  full circle is 6.28 — the angle is destroyed, not approximated. Replaced it with a
  **32-bit integer phase**, which makes `mod 2π` free (integer overflow *is* the modulo),
  exact, and drift-free.

- **Cut the trigonometric table from ~1 MB to 2 KB (~500×)** — one midpoint-sampled
  quarter-wave sine table serving both `sin` and `cos`, with exact mirror symmetry and
  sign applied as a single XOR.

- **Verified bit-exactly against a reference model: 10⁶ vectors per unit, zero
  mismatches** — no tolerance, identical bit patterns. Includes an exhaustive
  special-value cross product (NaN / Inf / subnormals) and directed exact-tie tests.

- **Integrated with the real core and ran real software on it** — a C program using the
  new instruction executes correctly on the actual CV32E40X, lint-clean with **0 errors
  and 0 warnings**.

- **Produced the first characterisation of what *strict* BF16 costs for RoPE** — every
  other BF16 accelerator accumulates in wider precision; nobody had measured the strict
  case. Answer: **1.27× RMS error**, and only ~3.5 of 8 significand bits survive the
  cancellation that high-frequency pairs hit within the first few hundred tokens.

- **Found and fixed 10 defects**, most of the silent-wrong-answer kind — including two
  traps in the third-party FPU and a position-field truncation that would have corrupted
  results past token 8191 with no error reported.

- **Self-contained and reproducible** — 49 new files, zero upstream files modified,
  one `make` command runs every check.

## Upcoming

- **Run the core's own regression suite (`core-v-verif`) and Spike co-simulation** —
  *highest priority.* Proves the coprocessor has not perturbed the CPU itself. This is
  the gap between "the unit is correct" and "the system is correct".

- **Synthesis for area, timing and power** — scripts are written (400 MHz / 45 nm
  constraints, area-breakdown reporting) but **have never been run**: no PDK available.
  Until then, **no area or frequency claims.**

- **Four more instructions for the streaming variant** — today one instruction rotates
  one pair, so a 128-element head costs 64 instructions and issue overhead dominates.
  Planned, same custom-0 opcode space, distinguished by `funct3`/`funct7`:
  `ROPE.CFG` · `ROPE.POS` · `ROPE.START` · `ROPE.WAIT`
  Groundwork is in place: the phase generator already implements the sequential mode
  this needs, and the memory channels are tied off behind a guard assertion.

- **Scale verification to 10⁷ vectors** — targets are wired up (`make m1-full`,
  `make m4-full`); 10⁶ is done at zero mismatches.

- **Validate against real LLaMA weights** — currently verified against a faithful
  reimplementation of HuggingFace's `apply_rotary_pos_emb` (agreeing to 1e-12); real
  extracted tensors need PyTorch and gated model access.

- **FPGA bring-up at 100 MHz** as a reality check before any ASIC target.

- **Read RoME (arXiv:2604.09742)** — the only operator-level hardware analysis of RoPE;
  may change where the boundary between this unit and the main array should sit.

---
---

# SLIDE 2 — Metrics / Proof

## The new instruction

```
ROPE.ROT rd, rs1, rs2          RISC-V R-type, custom-0

  31       25 24    20 19    15 14  12 11   7 6         0
 +----------+--------+--------+------+------+-----------+
 |  funct7  |  rs2   |  rs1   |funct3|  rd  |  opcode   |
 | 0000000  | xxxxx  | xxxxx  | 000  |xxxxx | 0001011   |
 +----------+--------+--------+------+------+-----------+
    7 bits   5 bits   5 bits  3 bits 5 bits   7 bits
    = 0x00                    = 0x0          = 0x0B (custom-0)

Register contents — BF16 packs two-per-register on RV32:
  rs1 = { x  [31:16] , y  [15:0] }    BF16 pair in   (no conversion)
  rs2 = { m  [31:16] , i  [15:0] }    position, pair index (integers)
  rd  = { x' [31:16] , y' [15:0] }    rotated BF16 pair out

Verified from the compiled binary:  0x0117878b
  -> opcode 0x0B, funct3 0, funct7 0x00, rd=x15, rs1=x15, rs2=x17
```

## Correctness

| Check | Largest run completed | Shown in the recording |
|---|---|---|
| BF16 arithmetic primitives | **bit-exact**, 1,001,935 vectors, **0** mismatches | 201,935 vectors, 0 mismatches |
| Integer phase generator | **bit-exact**; sequential == random access (zero drift) | 205,208 vectors, 0 errors |
| Sine table | **bit-exact**; mirror symmetry exact at all 1024 indices | 204,224 vectors, 0 errors |
| Rotation datapath | **bit-exact**, 1,001,961 vectors, **0** mismatches | 201,961 vectors, 0 mismatches |
| XIF coprocessor | **6/6** tests, all **7** formal properties held | same |
| Integration with real core | **0 errors, 0 warnings** | same |
| C program on the core | **PASS** in 13,993 cycles | same |

> **Keep these two columns straight if asked.** `make demo` runs **200k vectors per test**
> so it finishes in ~16 s on camera. The **10⁶** figures come from a longer campaign
> already completed (also zero mismatches). To make the recording match the left column
> exactly, run `make demo NVEC=1000000` — it takes a few minutes because it regenerates
> the vector files.

## Performance and size

| Metric | Value |
|---|---|
| Datapath latency | **3 cycles** |
| Throughput | **1 rotated pair / cycle** after fill |
| Sine LUT | **2 KB** (1024 × 16-bit, quarter wave, serves sin *and* cos) |
| Theta ROM | **256 B** (64 × 32-bit integer phase increments) |
| Naive table avoided | ~1 MB → **~500× reduction** |
| `sin²+cos²` deviation | 0.005280 (~0.68 BF16 ulp — cannot be exact in BF16) |

## Accuracy (400k random tuples, vs FP64 reference)

| Metric | Value |
|---|---|
| Strict BF16 RMS error | 2.73e-03 (pair-relative) |
| **Cost of strict BF16** | **1.27× RMS** vs conventional wide accumulate |
| Bit-identical to wide accumulate | 45.2% of random cases |
| Under cancellation | **3.5 of 8** significand bits retained (wide: 3.7) |

## Scale

| | |
|---|---|
| Hand-written RTL | 1,543 lines, 9 modules |
| Golden model + generators | 1,516 lines (Python, numpy only) |
| Testbenches + assertions | 1,899 lines, 6 testbenches |
| Vendored CVFPU | 3,013 lines (copied in, not symlinked) |
| **Upstream files modified** | **0** |

---

## ▶ The command to screen-record

```bash
cd ~/cv32e40x/rope-bf16 && make demo
```

**Runtime: ~15 seconds** (measured). Runs all eight verification stages live, then prints
the instruction encoding — decoded from the actual compiled binary — and the full metrics
table. Nothing is hard-coded: every number on screen is produced by the tools during the
run, and the script exits non-zero if any stage fails.

---
---

# How to record

## Before you hit record

```bash
cd ~/cv32e40x/rope-bf16
make demo-prep          # ~4 minutes: compiles all 6 simulators + the RISC-V binary
```

This exists so the recording shows **results**, not a C++ compiler scrolling for four
minutes. It is a one-time step; `make demo` afterwards is the real verification run.

## Record this

```bash
cd ~/cv32e40x/rope-bf16 && make demo
```

- **~15 seconds** (measured), fits comfortably in a slide transition
- Output is **161 lines × 99 columns**. Use a terminal **at least 100 columns** wide or
  the ASCII bit-field diagram wraps and looks broken. It will scroll — that reads fine on
  a recording, but if you want it all on one screen, drop the font size to give ~165 rows,
  or use `make demo-quick` which is shorter
- Exits **0** on success, non-zero if anything fails
- Ends with a green **`ALL CHECKS PASSED`**
- Order on screen: 8 verification stages → the instruction encoding → the metrics table.
  The last two screens are the ones worth pausing on

## Variants

| Command | Vectors per test | Measured runtime |
|---|---|---|
| `make demo` | 200,000 | **~15 s** — the one to record |
| `make demo-quick` | 20,000 | ~9 s — if pressed for time |
| `make demo NVEC=1000000` | 1,000,000 | a few minutes — matches the 10⁶ column exactly |

## If a step fails on the day

Every step writes a full log to `build/demo/*.log` (`m1.log`, `m4.log`, `lint.log`, …).
The demo prints the last 15 lines of any failing step inline, so you are not guessing on
camera.

## What the run shows, stage by stage

| Stage | What it proves |
|---|---|
| 1 | Reference model self-validates — two independent BF16 rounding implementations agree; exhaustive round-trip over all 65,536 BF16 patterns |
| 2 | **M1** BF16 multiply/add/subtract bit-exact, incl. NaN/Inf/subnormal cross product |
| 3 | **M2** Integer phase exact; prints the measured BF16 spacing that motivates the whole design |
| 4 | **M3** Sine table bit-exact; mirror symmetry and quadrant signs exact |
| 5 | **M4** Full datapath bit-exact at one pair per cycle; reports 3-cycle latency |
| 6 | **M5** CPU interface: rejects foreign instructions, handles cancellation, survives backpressure |
| 7 | **M6** Real cv32e40x core + interface + coprocessor: 0 errors, 0 warnings |
| 8 | **M7** C program runs on the core; instruction encoding verified against the RTL decoder |

---

# Backup — likely questions

**"Does this run an LLM?"**
No, and I would not claim it. CV32E40X is a small embedded core; it cannot run LLaMA.
The contribution is the unit and its characterisation — which is exactly the gap the
Mugi paper leaves open.

**"Why is BF16-only hard? Doesn't everyone use BF16?"**
Everyone *multiplies* in BF16 and then **accumulates in FP32** — GPU tensor cores
included. Keeping it strictly BF16 end to end is the aggressive choice, and its cost for
RoPE had not been published. That measurement is the novel result: 1.27× RMS.

**"How do you know it's actually correct?"**
Bit-exact comparison — identical bit patterns, not a tolerance — against a reference
model built *before* the hardware, over 10⁶ vectors per unit with zero mismatches. Plus
a C program producing correct results on the real core.

**"What's the one thing that could still be wrong?"**
System-level interference with the CPU. The coprocessor is verified and provably inert
for instructions that aren't its own, but I have not yet run the core's own regression
suite before-and-after. That is the top of the upcoming list.

**"Can you quote area or clock frequency?"**
No. The synthesis scripts are written but have never been run — no PDK. Any number there
would be invented.

**"Why 3 cycles / why 2 KB?"**
3 cycles = one operand register + one multiplier stage + one adder stage. 2 KB because
1024 entries over a quarter turn is already ~5× finer than BF16 can represent — BF16's
own coarseness caps the useful table size, which is the inverse of the usual concern.

---

*Full technical detail: `docmentaiton.md`. Design decisions and gotchas: `docs/notes.md`.
Generated measurements: `docs/error_budget.md`.*
