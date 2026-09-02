# BF16-Native RoPE Coprocessor for CV32E40X — Implementation Plan

Target: a synthesizable, BF16-only Rotary Positional Embedding unit attached to
CV32E40X over CORE-V-XIF, intended as the external RoPE offload path that the
Mugi paper (arXiv:2601.10823) leaves unimplemented.

---

## 0. Design Invariants

These are non-negotiable properties of the design. Anything that violates one is a bug,
not a tradeoff.

### INV-1: The arithmetic datapath is BF16 only. No conversions.

Every value that enters a multiplier or an adder is BF16. Every value that leaves one is
BF16. There is no FP32, no FP16, no fixed-point, and no conversion stage anywhere in the
compute path — not at the interface, not internally.

- LUT contents: BF16
- All 4 products: BF16 in, BF16 out
- Both sums: BF16 in, BF16 out
- Rounding: round-to-nearest-even at **every** node
- `AccWidth` parameter exists **only** to quantify the error of this choice for the
  writeup. Default is strict BF16. Do not ship it enabled.

### INV-2: The phase index is an integer. This is addressing, not arithmetic.

Sequence position `m` and pair index `i` are integers in every RoPE implementation in
existence. The phase word is an **address** computed from them, used to index the sin
LUT. It is never multiplied by, added to, or otherwise mixed with BF16 data.

**Why it cannot be BF16 — the numerical proof.** For pair `i=0`, `theta_0 = 1.0` rad.
At sequence position `m = 1000` the angle is `1000` rad. BF16 has 7 stored mantissa bits,
so the spacing of representable values near 1000 is:

```
1000 * 2^-8  ~=  3.9 radians
```

The representable-value spacing exceeds a full period of 2*pi. The angle is not
approximated, it is unrepresentable — the output is uncorrelated noise, not a degraded
result. Argument reduction fundamentally requires extra precision: for `m` up to 4096 and
~10 bits of final angle precision you need `log2(4096) + 10 ~= 22` bits. BF16 provides 8.

A 32-bit integer phase accumulator solves this exactly, with zero drift, and makes
`mod 2*pi` free (natural wraparound). See Section 6.

### INV-3: No approximation of the interface contract.

BF16 in from Mugi's GEMM output, BF16 out to the KV-cache quantizer. Bit-exact match
against the golden model, or the milestone is not complete.

### INV-4: Rotate before quantize.

RoPE must be applied to BF16 Q/K **before** INT4 KV-cache quantization. Rotation mixes
element pairs; rotating already-quantized keys is mathematically wrong. This constrains
where the unit sits in the dataflow (Section 13).

---

## 1. Scope and Honest Expectations

**What this project delivers:**
- A correct, synthesizable BF16 RoPE unit
- Measured area, timing, and energy on a real core
- Bit-exact verification against PyTorch/HuggingFace
- A characterized error budget for strict-BF16 RoPE (novel — nobody has published this)
- An argued integration story with Mugi

**What it does not deliver:** end-to-end LLM inference. CV32E40X is a small embedded
core. It cannot run LLaMA. Do not promise this in a paper or a proposal. The contribution
is the unit and its characterization, which is exactly the gap Mugi's Section 7.1 leaves.

---

## 2. Prerequisites

### 2.1 Toolchain

```bash
# Verilator (primary simulator — fast, free, good SV support)
sudo apt install verilator
verilator --version    # want >= 5.x

# RISC-V toolchain
sudo apt install gcc-riscv64-unknown-elf
# or build riscv-gnu-toolchain with --with-arch=rv32imc --with-abi=ilp32

# Bender (dependency manager — CVFPU ships support for it)
cargo install bender
# or grab a release binary from github.com/pulp-platform/bender

# Python env for golden model + LUT generation
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy ml_dtypes cocotb pytest
```

### 2.2 VS Code extensions

- **Verilog-HDL/SystemVerilog** (mshr-h) — syntax + linting, point it at `verilator --lint-only`
- **WaveTrace** or **Surfer** — inline VCD/FST waveform viewing
- **Even Better TOML** — for `Bender.yml` editing
- **Python** — for the golden model and generators

Add to `.vscode/settings.json`:

```json
{
  "verilog.linting.linter": "verilator",
  "verilog.linting.verilator.arguments": "--lint-only -Wall --timing",
  "files.associations": { "*.sv": "systemverilog", "*.svh": "systemverilog" }
}
```

### 2.3 Repositories to clone

```bash
mkdir -p deps && cd deps
git clone https://github.com/openhwgroup/cv32e40x.git
git clone https://github.com/openhwgroup/core-v-xif.git
git clone https://github.com/openhwgroup/cvfpu.git
git clone https://github.com/openhwgroup/core-v-verif.git   # large; verification env
cd ..
```

---

## 3. Repository Layout

```
rope-bf16/
├── Bender.yml
├── rtl/
│   ├── rope_pkg.sv                 # params, types, opcode constants
│   ├── rope_theta_rom.sv           # generated: Phi_i table (integer)
│   ├── rope_sin_lut.sv             # generated: BF16 quarter-wave sine table
│   ├── rope_phase_gen.sv           # integer phase accumulator + quadrant decode
│   ├── rope_bf16_mul.sv            # thin wrapper over CVFPU FP16ALT
│   ├── rope_bf16_add.sv            # thin wrapper over CVFPU FP16ALT
│   ├── rope_datapath.sv            # 4 mul + 2 add, strict BF16
│   ├── rope_xif_coproc.sv          # CORE-V-XIF wrapper
│   └── rope_stream_fsm.sv          # milestone 8 only
├── gen/
│   ├── gen_sin_lut.py              # emits rope_sin_lut.sv
│   └── gen_theta_rom.py            # emits rope_theta_rom.sv
├── model/
│   ├── bf16.py                     # BF16 helpers, RNE rounding
│   ├── rope_ref.py                 # golden model (strict BF16 + fp32 reference)
│   └── rope_ref.c                  # same, for DPI-C
├── tb/
│   ├── tb_rope_datapath.sv
│   ├── tb_rope_xif_coproc.sv
│   ├── rope_assertions.svh         # SVA for XIF protocol
│   └── cocotb/
│       └── test_rope.py
├── sw/
│   ├── rope_intrin.h               # .insn inline asm wrappers
│   └── test_rope.c
├── syn/
│   ├── rope.sdc
│   └── synth.tcl
└── docs/
    └── notes.md
```

---

## 4. Milestone 0 — Golden Model First

**Do this before writing any RTL.** You cannot verify what you cannot compare against.

### 4.1 `model/bf16.py`

```python
import numpy as np
import ml_dtypes   # provides bfloat16 with correct RNE semantics

def to_bf16(x):
    """Round FP32/FP64 -> BF16, round-to-nearest-even."""
    return np.asarray(x, dtype=np.float32).astype(ml_dtypes.bfloat16)

def bf16_bits(x):
    """BF16 value -> uint16 bit pattern."""
    return to_bf16(x).view(np.uint16)

def bits_to_bf16(u):
    """uint16 bit pattern -> BF16 value."""
    return np.asarray(u, dtype=np.uint16).view(ml_dtypes.bfloat16)

def bf16_to_f32_exact(u16):
    """BF16 bits -> FP32, EXACT (no rounding). BF16 is the top 16 bits of FP32."""
    return (np.asarray(u16, dtype=np.uint32) << 16).view(np.float32)
```

> **Key fact worth internalizing:** BF16 is bit-identical to the upper 16 bits of FP32.
> `bf16_to_f32(x) == bitcast<float>(x << 16)`, exactly, with no rounding. This makes
> hardware/model comparison trivial — no conversion library, no ambiguity about who
> rounded what. Use this everywhere in the testbench.

### 4.2 `model/rope_ref.py`

Implement **three** models. You need all three.

```python
import numpy as np
from bf16 import to_bf16

PHASE_BITS = 32
LUT_BITS   = 10          # 1024 entries per quadrant
LUT_N      = 1 << LUT_BITS

def phi_table(d, base=10000.0):
    """Integer phase increments Phi_i = round(2^32 * theta_i / (2*pi))."""
    i = np.arange(d // 2)
    theta = base ** (-2.0 * i / d)
    return np.round((theta / (2*np.pi)) * (1 << PHASE_BITS)).astype(np.uint64)

def sin_lut_bf16():
    """Quarter-wave sine table, BF16, MIDPOINT sampled.
    Midpoint sampling makes the Q1/Q3 reflection index exactly (LUT_N-1-idx)
    with no endpoint special case. Do not use edge sampling."""
    k = np.arange(LUT_N)
    ang = (k + 0.5) * (np.pi/2) / LUT_N
    return to_bf16(np.sin(ang))

LUT = sin_lut_bf16()

def lut_sin(phase):
    """sin() from 32-bit integer phase, BF16 out."""
    phase = np.uint32(phase)
    quad = (phase >> 30) & 0x3
    idx  = (phase >> (PHASE_BITS - 2 - LUT_BITS)) & (LUT_N - 1)
    ridx = np.where((quad == 1) | (quad == 3), LUT_N - 1 - idx, idx)
    val  = LUT[ridx]
    return np.where(quad >= 2, -val, val)

def lut_cos(phase):
    """cos(x) = sin(x + pi/2)  ->  phase + 2^30. ONE table, two lookups."""
    return lut_sin(np.uint32(phase) + np.uint32(1 << 30))

# ---- MODEL A: strict BF16 (matches the hardware, INV-1) ----
def rope_strict_bf16(x, y, m, i, phi):
    ph = np.uint32((np.uint64(m) * phi[i]) & 0xFFFFFFFF)
    c, s = to_bf16(lut_cos(ph)), to_bf16(lut_sin(ph))
    x, y = to_bf16(x), to_bf16(y)
    p0 = to_bf16(x * c);  p1 = to_bf16(y * s)      # round after every op
    p2 = to_bf16(x * s);  p3 = to_bf16(y * c)
    return to_bf16(p0 - p1), to_bf16(p2 + p3)

# ---- MODEL B: BF16 I/O, wide internal accumulate (for AccWidth comparison) ----
def rope_wide_acc(x, y, m, i, phi):
    ph = np.uint32((np.uint64(m) * phi[i]) & 0xFFFFFFFF)
    c  = np.float32(lut_cos(ph)); s = np.float32(lut_sin(ph))
    xf = np.float32(to_bf16(x));  yf = np.float32(to_bf16(y))
    return to_bf16(xf*c - yf*s), to_bf16(xf*s + yf*c)

# ---- MODEL C: FP64 exact reference (measures total error of A and B) ----
def rope_exact(x, y, m, i, d, base=10000.0):
    th = base ** (-2.0 * i / d)
    a  = m * th
    return x*np.cos(a) - y*np.sin(a), x*np.sin(a) + y*np.cos(a)
```

### 4.3 Exit criteria for Milestone 0

- [ ] Model A reproduces itself deterministically
- [ ] Model A vs Model C: plot relative error vs `m` for all `i`. Record max and RMS.
      **This is your error budget and a headline result.** Expect worst case near
      `m*theta ~ 45 deg` with `x ~ y` (catastrophic cancellation — see Section 11).
- [ ] Model A vs HuggingFace `apply_rotary_pos_emb` on real extracted tensors, after
      resolving the pairing convention (Section 12). Document the observed delta.
- [ ] `model/rope_ref.c` written and cross-checked against `rope_ref.py` bit-for-bit
      (needed later for DPI-C)

---

## 5. Milestone 1 — BF16 Arithmetic Primitives

Do not write BF16 multipliers or adders from scratch. Use **CVFPU (FPnew)**.

### 5.1 Critical detail: BF16 is called `FP16ALT`

In FPnew's nomenclature, bfloat16 is **`FP16ALT`**, not `BF16` or `BFLOAT16`. Grepping
the repo for "bf16" will find nothing. Configure via `fpnew_pkg`:

- Enable format `FP16ALT` in your `Features` struct
- Disable every other format to keep area minimal
- Enable only `ADD` and `MUL` operation groups (no DIV/SQRT — you don't need them, and
  DivSqrt is a large block)
- Set `rnd_mode_i` to round-to-nearest-even, **statically**

### 5.2 Integration notes

- Instantiate `fpnew_top`. Compile `fpnew_pkg` **before** any file referencing it.
- Scope explicitly: `fpnew_pkg::foo`. Do not wildcard-import the package.
- FPnew deliberately avoids SystemVerilog interfaces and uses packed arrays throughout
  (poor EDA tool support for the alternatives). It will not match `core_v_xif.sv`'s
  interface style — adapt at the boundary, in `rope_xif_coproc.sv`.
- Both FPnew ports use valid/ready handshakes, AXI-like. This composes naturally with
  XIF and with your streaming FSM.
- License: SolderPad (permissive, Apache-2.0-based). Safe to combine with your work.
- Bender support ships in the repo — add it as a dependency, don't vendor it.

`Bender.yml`:

```yaml
package:
  name: rope-bf16
dependencies:
  cvfpu: { git: "https://github.com/openhwgroup/cvfpu.git", rev: master }
sources:
  - rtl/rope_pkg.sv
  - rtl/rope_theta_rom.sv
  - rtl/rope_sin_lut.sv
  - rtl/rope_phase_gen.sv
  - rtl/rope_bf16_mul.sv
  - rtl/rope_bf16_add.sv
  - rtl/rope_datapath.sv
  - rtl/rope_xif_coproc.sv
```

### 5.3 Exit criteria

- [ ] `rope_bf16_mul` passes an exhaustive-ish random test vs Model A's `to_bf16(a*b)`
      — 10^7 random pairs, **zero** mismatches
- [ ] `rope_bf16_add` same
- [ ] RNE confirmed on tie cases specifically. Build a directed test of exact-tie
      mantissas. If you truncate instead of RNE you get a systematic downward bias that
      shows up much later as an unexplained perplexity gap — catch it here.
- [ ] NaN/Inf propagation verified. Subnormal policy chosen and **documented** in
      `docs/notes.md` (flush-to-zero is defensible and cheaper; Mugi's PP block muxes
      Zero/INF/NaN explicitly, so mirroring that keeps you consistent with the target).

---

## 6. Milestone 2 — Integer Phase Generator

Per INV-2. This is the piece that makes BF16 RoPE possible at all.

### 6.1 Structure

```
32-bit phase word P = (m * Phi_i) mod 2^32     <- mod is FREE (wraparound)

  P[31:30] = quadrant  (2 bits)
  P[29:20] = LUT index (10 bits, 1024 entries)
  P[19:0]  = unused (reserve for future interpolation)
```

Why 32 bits and not 24: the slowest frequency has `theta ~ 1e-4`, giving
`Phi_min ~ 68000` at 32 bits (relative quantization error ~1e-5) versus `Phi_min ~ 267`
at 24 bits (~0.4% error — too coarse). 32 bits is also the natural width on RV32.

### 6.2 `gen/gen_theta_rom.py`

Emits `rtl/rope_theta_rom.sv` — a `d/2` entry x 32-bit integer ROM of
`Phi_i = round(2^32 * theta_i / (2*pi))`, with `d` and `base` as script arguments so you
can regenerate for different models (LLaMA base=10000).

Size for `d=128`: 64 entries x 32 bits = **256 bytes**.

### 6.3 Two operating modes

- **Random access** (`ROPE.ROT`): compute `m * Phi_i` with a small multiplier.
  `m` is ~13 bits, `Phi` is 32 bits — a 13x32 multiplier, cheap.
- **Sequential decode**: hold per-pair phase state, add `Phi_i` once per token.
  One integer add. **Zero drift**, because integer addition is exact.

> **Do NOT use the trigonometric recurrence**
> `cos((m+1)th) = cos(m*th)cos(th) - sin(m*th)sin(th)`.
> In BF16 this drifts catastrophically — errors compound multiplicatively and the state
> wanders off the unit circle within a few dozen tokens at 7 mantissa bits. The integer
> accumulator is exact and cheaper.

### 6.4 Exit criteria

- [ ] `rope_phase_gen` matches Python phase computation bit-exactly for
      `m` in [0, 8192], all `i`
- [ ] Wraparound verified at `m` values that cross `2^32`
- [ ] Sequential and random-access modes agree for the same `(m, i)`

---

## 7. Milestone 3 — BF16 Sine LUT

### 7.1 Sizing argument (why the table is tiny)

With 1024 entries over a quarter turn, angle resolution is
`(pi/2)/1024 ~= 0.0015` rad. Since `|d(sin)/d(theta)| <= 1`, worst-case value error is
`~0.0015`. BF16's ulp near 1.0 is `2^-7 ~= 0.0078`.

**The table is already 5x finer than BF16 can represent.** BF16's own coarseness bounds
the table size — this is the inverse of the usual concern. 512 entries would also suffice.

```
1024 entries x 2 bytes (BF16) = 2 KB
```

For comparison: Mugi's iSRAM is 64 KB. And the naive precomputed
`max_seq_len x d/2 x {sin,cos}` table would be ~1 MB for `d=128, seq=4096` —
completely out of budget. The phase accumulator buys a ~500x table reduction.

### 7.2 One table, both functions

- `sin(P)`: quadrant + reflection logic below
- `cos(P) = sin(P + pi/2)` = `sin(P + 2^30)`. **Same table, phase offset.**

Use a dual-port ROM (or two reads) to get both per cycle.

### 7.3 Midpoint sampling — do not skip this

Store `T[k] = sin((k + 0.5) * (pi/2) / 1024)`, **not** `sin(k * (pi/2) / 1024)`.

With midpoint sampling the Q1/Q3 reflection index is exactly `1023 - idx`:

```
angle = (k+0.5)*D  where D = (pi/2)/1024
pi/2 - angle = 1024*D - (k+0.5)*D = ((1023-k)+0.5)*D   -> index 1023-k, exactly
```

Edge sampling would need index `1024 - idx`, which overflows the table at `idx=0` and
forces either a 1025th entry or a special case. Midpoint sampling also halves the max
sampling error. Free correctness and free accuracy.

### 7.4 Quadrant logic

```
quad 0:  sin = +T[idx]            cos via phase+2^30
quad 1:  sin = +T[1023-idx]
quad 2:  sin = -T[idx]
quad 3:  sin = -T[1023-idx]
```

Sign application is a **BF16 sign-bit flip** (XOR on bit 15) — exact, no arithmetic, no
rounding. Note this is one of the few places where BF16's format layout helps you
directly.

### 7.5 Exit criteria

- [ ] `rope_sin_lut` output matches Python `lut_sin` / `lut_cos` bit-exactly across all
      `2^32` phase values (sample densely; exhaustive on the top 12 bits)
- [ ] Reflection symmetry verified at quadrant boundaries specifically
- [ ] `sin^2 + cos^2` checked; document the deviation from 1.0 (it will not be exactly
      1.0 in BF16, and that is expected and fine)

---

## 8. Milestone 4 — BF16 Datapath

### 8.1 Structure

```
      x (BF16)   y (BF16)      cos (BF16)   sin (BF16)
         |          |              |            |
         +----+-----+------+-------+------+-----+
              |            |              |
         [BF16 mul]   [BF16 mul]    [BF16 mul]   [BF16 mul]
          x*cos        y*sin         x*sin        y*cos
              |            |              |            |
              +-----+------+              +-----+------+
                    |                           |
               [BF16 sub]                  [BF16 add]
                    |                           |
                 x' (BF16)                   y' (BF16)
```

4 BF16 multiplies, 2 BF16 adds. Rounded to BF16 at every node (INV-1).

Only ~2 rounding events on the data path, versus ~16 for a CORDIC implementation — this
is why the LUT+multiply approach is *more* accurate in BF16, not less.

### 8.2 Pipeline

Target one pair per cycle after fill. Depth is set by CVFPU's configured pipeline
registers plus the LUT read. Expect roughly 4-6 stages.

### 8.3 `AccWidth` parameter

```systemverilog
parameter int unsigned AccWidth = 16   // 16 = strict BF16 (DEFAULT, per INV-1)
                                       // 32 = wide internal accumulate (MEASUREMENT ONLY)
```

Default strict. The wide setting exists solely so you can produce the
"error under strict BF16 vs wide accumulate" comparison for the writeup. Ship strict.

### 8.4 Exit criteria

- [ ] Bit-exact vs Model A on 10^7 random `(x, y, m, i)` tuples — **zero** mismatches
- [ ] Bit-exact vs Model A on real Q/K tensors extracted from HuggingFace LLaMA
- [ ] Directed cancellation tests pass (Section 11)
- [ ] `AccWidth=32` build matches Model B, and the A-vs-B error delta is recorded

---

## 9. Milestone 5 — XIF Coprocessor Wrapper

### 9.1 Instruction encoding

BF16 packs two-per-GPR on RV32. This is the cleanest possible fit:

```
ROPE.ROT  rd, rs1, rs2          # custom-0, opcode 0x0B
  rs1 = {x_bf16[15:0], y_bf16[15:0]}    # native BF16 pair, NO conversion
  rs2 = {m[15:0], i[15:0]}              # position and pair index (integers)
  rd  = {x'_bf16[15:0], y'_bf16[15:0]}  # native BF16 pair out
```

Single source pair, single destination. **No dual writeback needed — `X_RFW_WIDTH`
stays 32.** Make the opcode a parameter in `rope_pkg.sv` so it can be reassigned if it
ever collides with another coprocessor (this is an explicit CV-X-IF recommendation).

### 9.2 XIF channels used

| Channel | Use | Notes |
|---|---|---|
| Compressed | No | Not needed |
| Issue | Yes | Decode + accept, receive rs operands |
| Commit | **Yes, mandatory** | Kill handling — see below |
| Memory req/resp | Milestone 8 | CV32E40X implements these (CVA6 does not) |
| Memory result | Milestone 8 | |
| Result | Yes | Writeback |

### 9.3 Three protocol rules that will burn you

**1. Register the operand inputs.** The coprocessor receives only a small fraction of the
timing budget on paths through `xif_issue_if.issue_req.rs*`, because CV32E40X sources
those operands directly from its register-file bypass network to avoid stalls. Flop
`rs1`/`rs2` on arrival and do **zero** combinational logic on them in the issue cycle.
Violating this blows timing in a way that is hard to diagnose.

General budget for signals not called out: 20% processor / 20% interconnect / 60%
coprocessor. Start from `deps/cv32e40x/constraints/cv32e40x_core.sdc`.

**2. `commit_valid` has no ready signal** (implicit, assumed 1). The coprocessor must be
able to observe `commit_valid` and `commit_kill` at any time coincident with or after
issue. For multi-cycle operations: track state per `id`, and **never** issue a store
before seeing a non-kill commit for that `id`.

**3. `mem_result_valid` has no ready signal.** You must always be able to sink a
returning load. Size the input buffer accordingly.

### 9.4 SVA — `tb/rope_assertions.svh`

Protocol bugs are far more likely than math bugs here. Assert:

- [ ] No `result_valid` for an `id` that was never issued
- [ ] No state update after `commit_kill` for that `id`
- [ ] No memory transaction before non-kill commit
- [ ] Transactions with an earlier issued `id` never depend on a later issued `id`
      (explicit CV-X-IF ordering rule — a coprocessor may not delay `result_valid`
      because it wants to see `commit_valid` for a newer instruction)
- [ ] `issue_ready` deassertion behavior is legal
- [ ] Every accepted issue eventually produces exactly one result or one kill

### 9.5 Parameters — lock these on day one

Mismatches between the core, the `cv32e40x_if_xif` instance, and the coprocessor cause
confusing elaboration-time breakage. Declare as localparams **above** `cv32e40x_core`:

```systemverilog
localparam int X_NUM_RS     = 2;    // 3 if you want R4-type later
localparam int X_ID_WIDTH   = 4;
localparam int X_MEM_WIDTH  = 32;   // 32 = exactly one interleaved BF16 pair/txn
localparam int X_RFR_WIDTH  = 32;
localparam int X_RFW_WIDTH  = 32;   // 32 is sufficient — no dual writeback
```

### 9.6 Exit criteria

- [ ] Standalone XIF testbench passes with all SVA enabled
- [ ] Kill-mid-operation test: assert `commit_kill`, verify no writeback and no state
      corruption
- [ ] Back-to-back `ROPE.ROT` at full throughput
- [ ] Dependent-instruction test: `ROPE.ROT` immediately consuming a preceding
      non-offloaded ALU result (this is what the registered-operand rule protects)

---

## 10. Milestone 6 — Core Integration

### 10.1 Steps

1. Instantiate `cv32e40x_core` + `cv32e40x_if_xif` + `rope_xif_coproc` in a wrapper
2. Connect all six XIF channel bundles (tie off compressed)
3. Use `deps/core-v-xif/src/core_v_xif.sv` for the interface/struct definitions rather
   than hand-rolling them — keeps you type-compatible with the core
4. Verilator lint clean: `verilator --lint-only -Wall`
5. Run the `core-v-verif` baseline regression **before** adding the coprocessor, save
   results, then **after**. They must match.
6. Keep RVFI-based Spike co-simulation green throughout — this is your guarantee that the
   coprocessor has not corrupted core state

### 10.2 Exit criteria

- [ ] Baseline `core-v-verif` regression identical pre- and post-integration
- [ ] RVFI co-simulation green
- [ ] No timing violations at target frequency (start at 100 MHz on FPGA; Mugi
      synthesizes at 400 MHz, so that is a reasonable eventual ASIC target)

---

## 11. Milestone 7 — Software and Error Characterization

### 11.1 `sw/rope_intrin.h`

```c
#include <stdint.h>

/* BF16 pair packed into one uint32_t: {x, y} */
static inline uint32_t rope_rot(uint32_t xy, uint16_t m, uint16_t i) {
    uint32_t rd, rs2 = ((uint32_t)m << 16) | i;
    asm volatile (".insn r 0x0B, 0, 0, %0, %1, %2"
                  : "=r"(rd) : "r"(xy), "r"(rs2));
    return rd;
}

/* Exact BF16 -> float. No rounding: BF16 IS the top 16 bits of FP32. */
static inline float bf16_to_f32(uint16_t b) {
    union { uint32_t u; float f; } c = { .u = (uint32_t)b << 16 };
    return c.f;
}
```

Later: patch binutils/LLVM with the encodings so you get real mnemonics instead of
`.insn`. Not needed to make progress.

### 11.2 The cancellation study — do this properly, it is a result

`x*cos - y*sin` has a catastrophic cancellation hazard: when `m*theta ~ 45 deg` and
`x ~ y`, the two products nearly cancel, and with 7 mantissa bits the difference retains
almost no significant digits.

Directed test sweep:

- [ ] `x == y` exactly, sweep `m*theta` through 45 deg at fine resolution
- [ ] `x/y` ratios near 1.0 across the BF16 exponent range
- [ ] Record max relative error, RMS error, and how often the result loses more than
      half its significant bits
- [ ] Compare strict BF16 (`AccWidth=16`) vs wide accumulate (`AccWidth=32`)

**This comparison has not been published.** Every BF16 accelerator — including Mugi's own
vector array and every BF16 tensor core — multiplies in BF16 but accumulates wider. You
are characterizing what strict BF16 actually costs for RoPE specifically. That is the
most novel measurement in the project. Report it either way, whichever direction it goes.

### 11.3 Exit criteria

- [ ] C test passes on the integrated core in Verilator
- [ ] Full error characterization table produced (strict vs wide, per pair index `i`,
      across `m`)
- [ ] Error vs Model C plotted; worst case identified and explained

---

## 12. The Pairing-Convention Trap

**This will cost you a week if you miss it.** Two incompatible conventions exist:

| Convention | Pairs | Used by | HW stride |
|---|---|---|---|
| **Interleaved** (GPT-J, original RoFormer) | `(x0,x1), (x2,x3), ...` | Original paper, GPT-J | 1 (adjacent) |
| **Half-split** (`rotate_half`, GPT-NeoX) | `(x_i, x_{i+d/2})` | **HuggingFace LLaMA**, GPT-NeoX | `d/2` |

They are equivalent up to a fixed permutation of the head dimension, so trained weights
port between them — but if your RTL uses one and your golden model uses the other, the
comparison fails and the error looks like garbage rather than like a convention mismatch.

`llama.cpp` handles this with an explicit NEOX-vs-normal RoPE type flag. **Do the same:**
add a `stride_mode` bit to the config. It costs almost nothing and makes the unit
reusable.

Hardware consequence: half-split means the two elements of a pair are `d/2` apart, so for
a 64-element head that is two separate cache lines per pair. Interleaved gets both
elements in one 32-bit word — which is why `X_MEM_WIDTH = 32` aligns perfectly for
interleaved and needs two transactions for half-split.

**Since Mugi's authors used HuggingFace Transformers for all their models, half-split is
your primary target.** Support both; default to half-split.

---

## 13. Milestone 8 — Streaming Variant

Once `ROPE.ROT` is correct, amortize the issue overhead. A per-pair instruction means
`d/2` instructions per vector per head — issue cost dominates.

```
ROPE.CFG   rs1, rs2   # rs1 = base address, rs2 = {d, stride_mode, flags}
ROPE.POS   rs1        # starting sequence position m
ROPE.START rd         # stream d/2 pairs via XIF memory channel
ROPE.WAIT  rd         # poll/block until drained
```

Uses the XIF memory req/resp + memory result channels, which CV32E40X implements. Per
Section 9.3 rule 2: buffer the results and do not store until commit is confirmed
non-kill.

### 13.1 Placement constraint (INV-4)

RoPE is applied after the Q/K projection GEMM and before the attention GEMM. Mugi
quantizes the KV cache to INT4. Therefore the ordering must be **rotate, then quantize** —
your unit needs to see Mugi's BF16 GEMM output *before* the KV-cache quantizer. That
argues for the unit sitting on the path between Mugi's output buffer and the cache write,
not as a general-purpose memory-to-memory engine.

### 13.2 GQA asymmetry

Under GQA (LLaMA-2-70B uses group size 8 in Mugi's setup) there are 8x fewer K heads than
Q heads. Your Q-side and K-side rotation workloads are asymmetric — parameterize
throughput so you can size to whichever side dominates.

---

## 14. Milestone 9 — Measurement

- [ ] Synthesize standalone: area, timing, power. Mugi uses 45 nm at 400 MHz — match
      that node and frequency for a comparable number.
- [ ] Area breakdown: LUT (2 KB) vs Theta ROM (256 B) vs CVFPU units vs phase logic vs
      XIF wrapper
- [ ] Cycles per token for a realistic head dimension, both conventions
- [ ] Energy per rotated pair
- [ ] Compare against the alternative Mugi mentioned (approximating sin/cos on its VLP
      array, where utilization would be low due to RoPE's sparse nature) — that
      comparison is the one the paper never ran

---

## 15. Milestone 10 — Mugi Integration Argument

Mugi's Section 7.1 explicitly names both options: approximate sin/cos on the VLP array
(low utilization, sparse), or offload to external hardware as existing GEMM accelerators
do. You are building option 2. Frame the writeup as closing that gap.

Read **arXiv:2604.09742 (RoME)** — the only work analyzing RoPE at the operator level for
hardware efficiency. It reformulates RoPE as a matrix operation, replacing the
rearrange/chunk/cat/flatten overhead with a single matrix multiply and fusing the
mul-add-mul sequence into one composite operator. Relevant two ways:

1. Related-work positioning
2. **Possibly your datapath** — if the rotation reduces to a fused matrix form, part of
   it might map onto Mugi's array after all, which sharpens the argument about *where*
   the boundary between the two units should sit

---

## 16. Gotcha Checklist

Print this. Every item has cost someone a day.

- [ ] BF16 is `FP16ALT` in CVFPU. Grepping for "bf16" finds nothing.
- [ ] RNE, not truncation. Truncation = systematic bias = mystery perplexity gap.
- [ ] Register `rs1`/`rs2` on arrival. Zero combinational logic in the issue cycle.
- [ ] `commit_valid` and `mem_result_valid` have no ready. Always be able to accept.
- [ ] Never store before non-kill commit.
- [ ] Pairing convention: HuggingFace uses half-split, original paper uses interleaved.
- [ ] Midpoint-sample the LUT so reflection is exactly `1023-idx`.
- [ ] Phase is integer. BF16 angles are unrepresentable past `m ~ 100` (INV-2).
- [ ] No trigonometric recurrence — it drifts in BF16.
- [ ] Compile `fpnew_pkg` first; scope explicitly, never wildcard-import.
- [ ] All five XIF parameters must match across core, interface, and coprocessor.
- [ ] Rotate before quantize, never after (INV-4).
- [ ] `bf16_to_f32(x) == bitcast<float>(x << 16)`, exact. Use it in every testbench.

---

## 17. References

### Papers
- **Mugi: Value Level Parallelism For Efficient LLMs** — arXiv:2601.10823 (ASPLOS '26).
  Section 7.1 states the RoPE gap this project fills.
- **RoFormer: Enhanced Transformer with Rotary Position Embedding** — arXiv:2104.09864
- **Efficient Matrix Implementation for RoPE (RoME)** — arXiv:2604.09742. Operator-level
  hardware analysis of RoPE.
- **Carat: Unlocking Value-Level Parallelism for Multiplier-Free GEMMs** — ASPLOS '24.
  Mugi's predecessor.
- **FPnew: An Open-Source Multi-Format FPU Architecture** — arXiv:2007.01530.
  Documents FP16alt = bfloat16.

### RTL
- `openhwgroup/cv32e40x` — the core
- `openhwgroup/core-v-xif` — spec; `src/core_v_xif.sv` has the SV interface definitions
- `openhwgroup/cvfpu` — BF16 arithmetic (SolderPad license)
- `openhwgroup/core-v-verif` — verification environment
- CV32E40X user manual: https://cv32e40x-user-manual.readthedocs.io
- CV-X-IF spec: https://docs.openhwgroup.org/projects/openhw-group-core-v-xif

### Software RoPE references
- **HuggingFace Transformers** `LlamaRotaryEmbedding` / `apply_rotary_pos_emb` /
  `rotate_half` — **primary golden model** (Mugi's authors used HF Transformers)
- `ZhuiyiTechnology/roformer` — original authors' repo, canonical math
- `lucidrains/rotary-embedding-torch` — most-used standalone PyTorch package
- `llama.cpp` `ggml_rope` — C, integer-friendly, explicit NEOX-vs-normal variant flag
- labml.ai annotated RoPE: https://nn.labml.ai/transformers/rope/
- `aju22/RoPE-PyTorch` — educational implementation

### Deliberately not used
- **CORDIC** (ZipCPU/cordic, etc.) — a fixed-point algorithm. In BF16 the shifts are free
  (exponent decrement) but every stage needs a full FP add, and ~16 sequential BF16
  roundings at 7 mantissa bits gives worse error than the format itself. Violates INV-1
  without a conversion stage. Rejected.
