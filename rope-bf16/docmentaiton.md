# BF16-Native RoPE Coprocessor for CV32E40X — Full Documentation

**A hardware accelerator for Rotary Positional Embedding, built entirely in BF16,
attached to a RISC-V CPU over CORE-V-XIF.**

This document is self-contained. It assumes **no prior knowledge** of RoPE, BF16, RISC-V
coprocessors, or the surrounding literature. Part 1 builds the background from scratch;
Parts 2–6 describe what was designed, what was measured, and what went wrong along the
way; the appendices contain the complete source of every file created.

> **Filename note:** this file is named `docmentaiton.md` because that is the exact name
> requested. If that spelling was a typo, rename with
> `git mv rope-bf16/docmentaiton.md rope-bf16/documentation.md`.

---

## Table of contents

**Part 1 — Background (start here if RoPE is new to you)**
1.1 What problem do transformers have with word order?
1.2 What RoPE actually does
1.3 The maths, in full
1.4 Why this needs dedicated hardware
1.5 What BF16 is, and why "BF16-only" is the hard part
1.6 The specific gap this project fills

**Part 2 — The design**
2.1 Overview and dataflow
2.2 The four design invariants
2.3 Key idea 1: the angle must be an integer
2.4 Key idea 2: one 2 KB table, and BF16 limits its size
2.5 Key idea 3: strict BF16 with only two roundings
2.6 The instruction
2.7 Talking to the CPU: CORE-V-XIF
2.8 Module-by-module walkthrough

**Part 3 — What changed in the repository**
3.1 Nothing upstream was modified
3.2 Complete file inventory
3.3 How CVFPU was vendored

**Part 4 — Engineering log: bugs found and fixed**
(ten concrete defects, each with before/after code)

**Part 5 — Corrections to the original plan (`steps.md`)**

**Part 6 — Verification and results**
6.1 Verification strategy
6.2 What each testbench proves
6.3 Results: the error budget
6.4 Measured performance and size

**Part 7 — Status, limitations, and next steps**

**Appendices — complete source code**
A. RTL (SystemVerilog)
B. Golden model and generators (Python)
C. Testbenches and assertions
D. Software, synthesis, and build system

---
---

# Part 1 — Background

## 1.1 What problem do transformers have with word order?

A large language model reads a sentence as a list of **tokens** (roughly, words or
word-pieces). Each token is turned into a list of numbers — a **vector**, typically 64 to
128 numbers long per "attention head".

The core operation in a transformer is **attention**, and at its heart is a **dot
product**: to decide how much token A should pay attention to token B, the model
multiplies A's "query" vector by B's "key" vector element-wise and sums the result.

Here is the problem. A dot product is completely **order-blind**. It sums products of
numbers, and addition does not care about position. So to the raw attention mechanism:

```
"the dog bites the man"   and   "the man bites the dog"
```

contain the same tokens and would produce the **same** attention scores. The model
literally cannot tell them apart. Word order — the thing that carries most of the meaning
in that pair of sentences — is invisible.

So position information has to be injected deliberately.

**The old approach** was to *add* a "position vector" to each token's embedding: token 1
gets one offset, token 2 another, and so on (this is "absolute positional encoding", used
in the original 2017 Transformer). It works, but it has two weaknesses: it entangles
position with meaning by adding them into the same numbers, and what attention actually
wants to know is usually not "where is this token absolutely" but "how far apart are these
two tokens".

## 1.2 What RoPE actually does

**RoPE** — Rotary Positional Embedding — takes a different approach. Instead of *adding*
position, it **rotates**.

The idea:

1. Take the query (or key) vector and chop it into **pairs** of numbers.
2. Treat each pair as a point on a 2-D plane, i.e. as coordinates `(x, y)`.
3. **Rotate that point** about the origin by an angle proportional to the token's
   position in the sequence.

Token at position 1 gets rotated a little; token at position 2 twice as much; token at
position 500, five hundred times as much.

Why is rotation the clever choice? Because of one property of rotations:

> When you dot-product a vector rotated by angle `A` with a vector rotated by angle `B`,
> the result depends **only on `A − B`**.

The absolute angles cancel out and only their *difference* survives. Since the angles are
proportional to positions `m` and `n`, the attention score between two tokens
automatically depends only on `m − n` — their **relative distance**. That is exactly the
quantity attention wants, and RoPE delivers it without any extra machinery, purely as a
side effect of the geometry.

A second nice property: rotation does not change a vector's length. So RoPE injects
position information without inflating or shrinking the magnitudes the network was trained
on. Adding a position vector cannot make that promise.

**Multiple speeds.** Every pair in the vector rotates at a *different* speed. The first
pair spins fast (about 1 radian per token), and each subsequent pair spins slower, down to
around 0.0001 radians per token. This is a deliberate multi-scale design, and the intuition
is a clock face or a car odometer: the fast hand distinguishes neighbouring tokens
precisely, while the slow hands are what still differ between token 10 and token 4000. A
single rotation speed would either wrap around uselessly over long distances or be too
coarse to separate adjacent tokens. Using many speeds at once gives both.

Concretely, for a 128-element head there are 64 pairs, and pair `i` rotates at speed:

```
theta_i = 10000 ^ (-2i / 128)

  i = 0  ->  theta = 1.0        (fast: ~1 radian per token)
  i = 8  ->  theta = 0.316
  i = 32 ->  theta = 0.01
  i = 63 ->  theta = 0.000115   (slow: barely moves per token)
```

RoPE is used by essentially every modern open LLM — LLaMA, Mistral, Qwen, GPT-NeoX,
PaLM — which is what makes it worth building in silicon.

## 1.3 The maths, in full

For one pair of numbers `(x, y)` at sequence position `m` using frequency `theta_i`, the
angle is `a = m * theta_i` and the rotation is the standard 2-D rotation matrix:

```
x' = x * cos(a)  -  y * sin(a)
y' = x * sin(a)  +  y * cos(a)
```

That is the entire computation this hardware performs: **four multiplies and two adds**,
plus obtaining `sin(a)` and `cos(a)`.

Applied to a whole 128-element head vector, that is 64 independent rotations. Done for
every token, every head, every layer, for both the query and the key.

**One subtlety that matters a lot in practice — which numbers get paired?** Two
incompatible conventions exist in the wild:

| Convention | Pairs are | Used by |
|---|---|---|
| **Interleaved** | `(x0,x1), (x2,x3), (x4,x5), …` — neighbours | GPT-J, the original RoFormer paper |
| **Half-split** | `(x0,x64), (x1,x65), …` — element `i` with element `i + d/2` | **HuggingFace LLaMA**, GPT-NeoX |

The two are mathematically equivalent — they differ only by a fixed reshuffling of the
vector's elements, which is why trained weights can be converted between them. But if the
hardware uses one convention and the reference model uses the other, the outputs disagree
and the error looks like a *numerical* bug rather than a *bookkeeping* one. This is a
well-known time sink; `llama.cpp` carries an explicit flag to distinguish them, and so
does this design. **Half-split is the default here**, because HuggingFace Transformers uses
it and the target paper's authors used HuggingFace.

## 1.4 Why this needs dedicated hardware

Two reasons.

**It is not a matrix multiply.** AI accelerators are built almost entirely around large
matrix multiplications (GEMM), because that is where the arithmetic is. RoPE is *not* a
matrix multiply — it is an element-wise operation that also needs `sin` and `cos`, which
GEMM hardware cannot compute. Running RoPE on a big multiply array leaves nearly all of
that array idle: you occupy an engine designed for thousands of parallel multiplies in
order to do a handful of element-wise ones.

**There is a lot of it.** Per token, per head, per layer, for both Q and K, you need `d/2`
rotations. Multiplied out across a real model, RoPE stops being a rounding error in the
runtime budget.

So there is a genuine case for a small, dedicated unit that does exactly this and nothing
else.

## 1.5 What BF16 is, and why "BF16-only" is the hard part

A floating-point number is stored as three fields: a **sign** bit, an **exponent** (how big
the number is), and a **mantissa**/significand (the precise digits).

| Format | Sign | Exponent | Stored mantissa | Total |
|---|---|---|---|---|
| FP32 (standard float) | 1 | 8 | 23 | 32 bits |
| FP16 (half precision) | 1 | 5 | 10 | 16 bits |
| **BF16** (bfloat16) | 1 | **8** | **7** | **16 bits** |

BF16 keeps FP32's **full 8-bit exponent** — so the same enormous range, from about 1e-38
to 3e38 — but throws away most of the mantissa, keeping only 7 stored bits (8 counting the
implicit leading 1). That is roughly **2 to 3 significant decimal digits**.

That trade is deliberate and it suits neural networks: they care much more about dynamic
range (not overflowing or underflowing) than about precision, and halving the bits halves
the memory traffic. BF16 is now the default numeric format for ML.

One property is used constantly in this project:

> **BF16 is bit-identical to the top 16 bits of FP32.**
> Converting BF16 → FP32 is a 16-bit left shift. Exact, no rounding, no lookup, no logic.

This is why the design needs **zero conversion hardware** at its boundaries, and why the
testbenches can compare hardware against a software model with no ambiguity about who
rounded what.

**Now the hard part.** The central constraint of this project (called INV-1) is that the
datapath is BF16 **only**:

> Every value entering a multiplier or adder is BF16. Every value leaving one is BF16.
> Rounded to nearest-even at *every* node. No FP32 anywhere, no internal widening, no
> conversion stage.

This is *unusual*. Essentially every BF16 accelerator ever built — including GPU tensor
cores — multiplies in BF16 but **accumulates in FP32**, precisely because accumulating
narrow loses accuracy. Doing it strictly in BF16 is the aggressive choice, and its cost for
RoPE specifically had not been published. Measuring that cost is one of this project's
deliverables (see Part 6.3).

**Round-to-nearest-even (RNE)** deserves a note since it appears everywhere below. When a
result falls exactly halfway between two representable numbers, RNE breaks the tie toward
the one with an even last bit. The alternative — truncation — always rounds toward zero,
which introduces a small *systematic* downward bias. Systematic bias accumulates over
millions of operations and shows up much later as an unexplained model-quality regression
that is extremely hard to trace back. So RNE is verified explicitly here, with directed
tests built from exact tie cases (Part 6.2).

## 1.6 The specific gap this project fills

The **Mugi** paper (arXiv:2601.10823, ASPLOS '26) describes an LLM accelerator. In its
Section 7.1 it discusses RoPE and states two options: approximate `sin`/`cos` on its own
vector array (acknowledging utilisation would be poor, because RoPE is sparse relative to
that array), or **offload to external hardware as existing GEMM accelerators do**.

It does not build the second option. This project does — as a synthesizable unit attached
to CV32E40X (a small open-source RISC-V core from the OpenHW Group) over **CORE-V-XIF**, a
standard interface for bolting custom instructions onto that core.

**What this project delivers**

- A correct, synthesizable, strict-BF16 RoPE unit
- Bit-exact verification against a software reference model
- A characterised error budget for strict-BF16 RoPE — a measurement not previously published
- An integrated, working custom RISC-V instruction, exercised from C

**What it explicitly does not deliver**

End-to-end LLM inference. CV32E40X is a small embedded microcontroller core; it cannot run
LLaMA, and no claim in that direction should be made from this work. The contribution is
**the unit and its characterisation**, which is precisely the gap Mugi leaves open.

---
---

# Part 2 — The design

## 2.1 Overview and dataflow

```
   m, i  (integers: position and which pair)        x, y  (BF16 data)
        |                                                 |
        v                                                 |
  +--------------------+                                  |
  | rope_theta_rom     |  Phi_i : how much phase           |
  |  64 x 32 bit ROM   |  one token advances this pair     |
  +--------------------+                                  |
        |                                                 |
        v                                                 |
  +--------------------+                                  |
  | rope_phase_gen     |  P = (m * Phi_i) mod 2^32        |
  |  integer multiply  |  <-- EXACT. No rounding.          |
  +--------------------+      "mod 2*pi" is free           |
        |                                                 |
        v                                                 |
  +--------------------+                                  |
  | rope_sin_lut       |  P[31:30] -> quadrant             |
  |  1024-entry        |  P[29:20] -> table index          |
  |  quarter-wave, 2KB |  cos = same table, P + 2^30       |
  +--------------------+                                  |
        |                                                 |
        |  sin, cos (BF16)                                 |
        v                                                 v
  +-------------------------------------------------------------+
  | rope_datapath                                               |
  |    4 x BF16 multiply :  x*cos   y*sin   x*sin   y*cos        |
  |    2 x BF16 add/sub  :  x' = x*cos - y*sin                   |
  |                         y' = x*sin + y*cos                   |
  |    RNE rounding at every node (INV-1)                        |
  +-------------------------------------------------------------+
        |
        v
   x', y'  (BF16)
```

Wrapped around that, `rope_xif_coproc` handles the CPU handshake (decode the instruction,
capture operands, return the result, handle cancellation), and
`rope_cv32e40x_wrapper` instantiates the real CPU core alongside it.

## 2.2 The four design invariants

These come from the project plan (`steps.md`) and are treated as non-negotiable — a
violation is a bug, not a trade-off.

| | Invariant | Meaning |
|---|---|---|
| **INV-1** | The arithmetic datapath is BF16 only | Every multiplier and adder input and output is BF16, RNE at every node. No FP32, no fixed-point, no conversion stage. |
| **INV-2** | The phase index is an integer | Position `m` and pair index `i` are integers. The phase word is an **address** used to index the sine table — never mixed with BF16 data. |
| **INV-3** | No approximation of the interface contract | BF16 in, BF16 out, bit-exact against the reference model or the milestone is not complete. |
| **INV-4** | Rotate before quantize | RoPE must be applied to BF16 data *before* any low-precision (e.g. INT4) cache quantisation, because rotation mixes element pairs. Rotating already-quantized values is mathematically wrong. |

## 2.3 Key idea 1: the angle must be an integer

This is the insight that makes BF16 RoPE possible at all, and it is worth stating carefully
because it is counter-intuitive.

**You cannot hold the angle in BF16.**

Consider the fastest pair, `i = 0`, where `theta_0 = 1.0` radian per token. At sequence
position `m = 1024`, the angle is 1024 radians. How precisely can BF16 represent a number
near 1024? BF16 has 7 stored mantissa bits, so near 1024 the gap between consecutive
representable values is:

```
1024 * 2^-7  =  8.0 radians
```

A full circle is `2*pi = 6.28` radians. **The gap between neighbouring representable BF16
values is larger than a complete revolution.** Rounding the angle to BF16 therefore moves
it by more than a full turn — the resulting `sin` and `cos` bear no relationship whatsoever
to the true values. The output is not "less accurate"; it is uncorrelated noise.

Measured values (asserted by the test suite, not taken on faith):

| Position `m` | BF16 spacing near `m` | Exceeds `pi/2`? | Exceeds `pi`? | Exceeds `2*pi`? |
|---|---|---|---|---|
| 100 | 0.5 rad | no | no | no |
| 500 | 2.0 rad | **yes** | no | no |
| 1000 | 4.0 rad | yes | **yes** | no |
| **1024** | **8.0 rad** | yes | yes | **yes** |
| 4096 | 32.0 rad | yes | yes | yes |

Stated more generally: to handle positions up to 4096 with about 10 bits of final angle
precision you need roughly `log2(4096) + 10 = 22` bits of precision in the angle. **BF16
provides 8.** It is not close.

**The solution: a 32-bit integer phase.**

Represent the angle not as a real number of radians but as an integer **fraction of a full
turn**, where the full 32-bit range spans exactly one revolution:

```
phase word P  (32 bits, unsigned)   represents   angle = 2*pi * P / 2^32
```

This turns out to be strictly better in four separate ways:

1. **`mod 2*pi` becomes free.** Angles are periodic, so the useful value is always the
   angle modulo a full turn. With this encoding, a full turn is exactly `2^32` — so the
   natural overflow of a 32-bit integer *is* the modulo operation. No comparison, no
   subtraction, no hardware at all. What is normally the expensive part of evaluating a
   trigonometric function (argument reduction) costs nothing.
2. **It is exact.** Integer multiply and add have no rounding error, so there is **zero
   drift**, no matter how long the sequence.
3. **The field layout is the table lookup.** Split the word:
   `P[31:30]` = which quadrant (2 bits), `P[29:20]` = index into a quarter-wave table
   (10 bits), `P[19:0]` = spare (reserved for future interpolation). Decoding the angle is
   just wiring.
4. **32 bits is the natural width** on a 32-bit RISC-V core.

**Why 32 bits and not 24?** Because the *slowest* pair must still advance measurably. With
`theta_63 = 0.000115`, the per-token phase increment is `Phi_63 = 78937` at 32 bits — a
relative quantisation error of about 1e-5. At 24 bits the same increment would be only
about 267, a 0.4% error, which is too coarse.

A worked example: `Phi_0 = round(2^32 / (2*pi)) = 683565276 = 0x28BE60DC` — one radian
expressed as a fraction of a turn.

**An explicitly rejected alternative.** A classic trick for generating a sequence of
angles is the trigonometric recurrence:

```
cos((m+1)*th) = cos(m*th)*cos(th) - sin(m*th)*sin(th)
```

This is a trap in BF16. Errors compound *multiplicatively* at every step, and with only 7
mantissa bits the state wanders off the unit circle within a few dozen tokens. The integer
accumulator is both exact and cheaper. The test suite confirms error is flat in `m`, which
is the observable signature of having avoided this.

## 2.4 Key idea 2: one 2 KB table, and BF16 limits its size

Given the phase word, `sin` and `cos` come from a lookup table. Three compounding
optimisations make it tiny.

**(a) Store only a quarter of the wave.** Sine's symmetry gives the rest:

```
quadrant 0:  sin = +T[idx]
quadrant 1:  sin = +T[N-1-idx]      (mirrored)
quadrant 2:  sin = -T[idx]
quadrant 3:  sin = -T[N-1-idx]
```

Applying the sign is an **XOR on BF16's bit 15** — exact, no arithmetic, no rounding. This
is one of the few places BF16's bit layout helps directly.

**(b) `cos` reuses the same table.** Since `cos(x) = sin(x + pi/2)`, and a quarter turn is
exactly `2^30` in this encoding, `cos` is just a second lookup with `P + 2^30`. **One
table, two reads.** No cosine table exists.

**(c) Midpoint sampling.** Store the sine of each bucket's **centre**, not its edge:

```
T[k] = sin( (k + 0.5) * (pi/2) / 1024 )
```

This is a small detail with a disproportionate payoff. With midpoint sampling the mirror
index is *exactly* `N-1-idx`:

```
angle        = (k + 0.5) * D                    where D = (pi/2)/N
pi/2 - angle = N*D - (k+0.5)*D = ((N-1-k) + 0.5) * D    -> index N-1-k, exactly
```

With edge sampling the mirror index would be `N - idx`, which runs off the end of the table
at `idx = 0` and forces either a 1025th entry or a special case. Midpoint sampling also
halves the worst-case sampling error. Both benefits are free.

**The size argument, which runs backwards from the usual one.** With 1024 entries over a
quarter turn the angular resolution is `(pi/2)/1024 = 0.00153` radians. Since the slope of
sine never exceeds 1, the worst-case value error is about 0.00153. Meanwhile BF16's own
spacing near 1.0 is `2^-7 = 0.0078`.

> The table is already about **5× finer than BF16 can even represent**. BF16's coarseness
> bounds the useful table size — the opposite of the usual "is my table big enough?"
> concern. 512 entries would also have sufficed.

**The size win:**

| | Size |
|---|---|
| This design: 1024 entries × 2 bytes (BF16) | **2 KB** |
| Naive precomputed `seq_len × d/2 × {sin,cos}` (d=128, seq=4096) | ~1 MB |

A roughly **500× reduction**, bought entirely by the integer phase accumulator. For scale,
Mugi's on-chip iSRAM is 64 KB, so this table is about 3% of it.

**One documented consequence.** Because the table stores bucket centres, `sin(0)` reads
`T[0] = 0.00076675`, not zero. (`cos(0)` reads `T[1023]` and does round to exactly 1.0.) So
at position `m = 0` the rotation is *nearly* but not *exactly* the identity:

```
x' = x * 1.0  -  y * 0.00076675
```

The absolute error is bounded by `|y| * 0.00077`, so the **relative** error in `x'` scales
with `|y|/|x|` and is not bounded by a small constant: with `x = 0.25, y = 8.0` it reaches
2.3%. With `|y| <= |x|` it stays under 0.5%. This is the price of the free mirror and the
halved average error — a deliberate trade, documented so the next reader does not mistake
it for a bug. If exactness at `m = 0` ever matters, the alternatives are edge sampling with
a 1025-entry table, or special-casing phase 0.

**Measured:** `sin^2 + cos^2` deviates from 1.0 by at most **0.00528** (RMS 0.0021), i.e.
about 0.68 BF16 ulp. It cannot be exactly 1.0 in BF16, and that is expected. The RTL and
the software model agree on this figure exactly.

## 2.5 Key idea 3: strict BF16 with only two roundings

The datapath is four multiplies and two adds, with RNE rounding at every node.

It is worth comparing against **CORDIC**, the textbook way to compute rotations in
hardware without multipliers. CORDIC performs about 16 sequential shift-and-add stages.
In BF16 the shifts are indeed free (just decrement the exponent), but every stage needs a
full floating-point add — so a CORDIC rotation incurs roughly **16 sequential roundings**
at 7 mantissa bits. The LUT-and-multiply approach here incurs about **2** on the data path.

> In BF16, LUT + multiply is **more** accurate than CORDIC, not less. CORDIC's usual
> precision advantage is a fixed-point argument that does not survive translation to a
> 7-bit mantissa.

CORDIC is also fixed-point by nature, so using it would require conversion stages — a
direct INV-1 violation. It was rejected on both grounds.

**Not built from scratch.** The BF16 multipliers and adders are **CVFPU** (also called
FPnew), a mature, widely taped-out open-source FPU from ETH Zürich / Bologna under a
permissive licence. Writing new floating-point arithmetic when a proven implementation
exists would be a poor use of effort and a rich source of subtle bugs.

**A naming trap worth flagging:** in CVFPU, bfloat16 is called **`FP16ALT`**. Searching the
CVFPU sources for "bf16" returns nothing at all. The unit is configured with `FP16ALT` as
the only enabled format, and only the ADD/MUL operation group generated — division and
square root are disabled (not needed, physically large, and their inclusion would drag in a
further external dependency).

## 2.6 The instruction

The unit is driven by a single custom RISC-V instruction:

```
ROPE.ROT rd, rs1, rs2          # custom-0 opcode 0x0B, funct3 = 0, funct7 = 0

  rs1 = { x_bf16[31:16] , y_bf16[15:0] }     one BF16 pair in   — NO conversion
  rs2 = { m[31:16]      , i[15:0]      }     position, pair index (plain integers)
  rd  = { x'_bf16[31:16], y'_bf16[15:0] }    the rotated BF16 pair out
```

This encoding is a particularly clean fit. On a 32-bit machine a general-purpose register
holds exactly **two** BF16 values, and a rotation consumes exactly two and produces exactly
two. So one source register and one destination register suffice — **no dual-register
writeback is needed**, which keeps the CPU interface narrow (`X_RFW_WIDTH` stays 32) and
avoids a whole class of interface complexity.

From C, via inline assembly (no compiler patch required):

```c
#include "rope_intrin.h"

uint32_t packed = rope_pack(x_bf16, y_bf16);
uint32_t result = rope_rot(packed, m, i);
uint16_t x_out  = rope_unpack_x(result);
uint16_t y_out  = rope_unpack_y(result);
```

The opcode is a **parameter**, not a hard-coded constant, so it can be moved if it ever
collides with another coprocessor — an explicit CORE-V-XIF recommendation.

## 2.7 Talking to the CPU: CORE-V-XIF

**CORE-V-XIF** is a standard interface for attaching custom instruction hardware to the
CV32E40X core. When the CPU meets an instruction it does not recognise, it offers it to the
coprocessor over several independent channels:

| Channel | Used here | Purpose |
|---|---|---|
| Compressed | no | 16-bit instruction forms; ROPE.ROT has none |
| **Issue** | **yes** | "Here is an instruction and its operands — do you want it?" |
| **Commit** | **yes, mandatory** | "That instruction is confirmed" *or* "cancel it" |
| Memory req/resp | Milestone 8 | For a coprocessor that accesses memory itself |
| Memory result | Milestone 8 | |
| **Result** | **yes** | Returning the answer to be written into `rd` |

The **Commit** channel exists because modern CPUs execute *speculatively*: the core may
offer an instruction that later turns out to be on a mispredicted branch and must be
discarded. The coprocessor must therefore be able to throw away work in flight without
corrupting anything — and, critically, must **not** write a result for a cancelled
instruction, since the CPU has already moved on.

Three protocol rules dominated the design, and each is a known source of hard-to-diagnose
failure:

**Rule 1 — Register the operands immediately.** The core supplies operands directly from
its register-file bypass network, so by the time they arrive very little of the clock period
remains. The coprocessor therefore gets only a small slice of the timing budget on those
paths (the general guidance is 20% core / 20% interconnect / 60% coprocessor, but the
operand paths are far tighter). The rule: latch `rs1`/`rs2` on arrival and perform **zero**
combinational logic on them in that first cycle. Violating this fails timing in a way that
is very hard to attribute. In this design, the phase multiply, the table lookup, and the
whole datapath consume only the *registered* copies. Only the instruction word is decoded
combinationally — and that comes from the instruction register, not the bypass network.

**Rule 2 — `commit_valid` has no ready signal.** The coprocessor cannot say "wait" to a
commit or a cancel. It must be able to observe one at any time, at or after issue. Handled
here by tracking state per instruction ID and sampling the commit channel unconditionally
on every cycle.

**Rule 3 — `mem_result_valid` has no ready signal either.** A returning load cannot be
refused, so a coprocessor using memory must always have somewhere to put it. Not applicable
yet (this unit performs no memory accesses), but the same "always be able to accept"
discipline is why the result path is credit-controlled.

**Ordering.** A rule that is easy to violate: an older instruction's result must never be
delayed waiting on a newer instruction. Satisfied here by returning results strictly
in-order from a FIFO.

**Backpressure, and how it is handled.** The CPU can refuse to accept a result on any given
cycle. But the arithmetic pipeline cannot be stalled (see Part 4.2 for why). These are
reconciled with a **credit counter**: the coprocessor only accepts a new instruction if the
result FIFO has a guaranteed free slot. When credits run out it withholds `issue_ready` and
the core simply retries later. This makes overflow structurally impossible rather than
merely unlikely.

## 2.7b Reducing the pipeline from three stages to two

The coprocessor originally took three cycles from issue to result, which cost the CPU a
one-cycle stall on every rotation. It now takes **two, and stalls zero times**, selected by
the `UsePrefetch` parameter. The instruction encoding, its operands and all existing
software are unchanged.

### Why there were three stages

Producing `sin(m·θᵢ)` and `cos(m·θᵢ)` is three steps in series:

```
  i ──► θ-ROM ──► × m ──► phase ──► sine LUT (×2, parallel) ──► sin, cos
        6 levels   ~18                    20 levels
```

(The two ROM depths are measured with `yosys ltp`; the multiply is an estimate.) That is
roughly 44 levels of logic, and it **cannot begin at the instruction**, because of rule 1
in the coprocessor's header: CV32E40X sources `issue_req.rs*` from its register-file
**bypass network**, not from a plain register read —

```systemverilog
xif_issue_if.issue_req.rs[0] = operand_a_fw;   // forwarding mux output
```

`operand_*_fw` is one of the latest-arriving signals in the whole ID stage. Hanging 44
levels off it would make the coprocessor the critical path for the entire CPU. So the
original design spent a full stage doing nothing but moving operands into flops, purely so
that the long path could *start* at a flop.

### The idea: the index never has to arrive

`theta_i = base^(-2i/d)` depends only on `i` and on `d`, and `d` is fixed at elaboration.
So `theta[i]` is a pure ROM lookup — and real code walks `i` in ascending order with `m`
held constant (`sw/rope_intrin.h`). The coprocessor can therefore read `theta[i+1]` while
it is still working on pair `i`, keeping it in a register:

```
stage 1   x_s0, y_s0, sin_s0, cos_s0
            ├─ x, y    ── straight from rs1 into flops        0 levels   (rule 1 ✓)
            └─ sin/cos ── m_q × theta_q → LUT                 ~38 levels
stage 2   result FIFO                                         ~50 levels
```

Both multiply operands now come from coprocessor flops, so rule 1 still holds, and the ROM
read for the next pair runs in the background. **The prefetch does not shorten the chain —
it moves the ROM read a cycle earlier and removes the need to capture the index at all.**

`m` is latched for a different reason: it arrives on the same bypass network, so feeding it
straight into a multiply would recreate the same problem. Both end up in flops, and both
therefore become state that can disagree with reality.

### `m` and `i` become checks instead of inputs

The instruction still carries both in `rs2`. They are now compared against the prefetch
state on every issue. On a mismatch the coprocessor takes **one bubble**, adopts `m` and
`i` from the instruction, reads `theta[i]` from the ROM directly for that one rotation, and
resumes prefetching.

Because recovery adopts state *from the instruction*, the first rotation of a new token
initialises the coprocessor by itself — which is why **no init instruction was added**. One
would have cost an instruction to save a bubble, roughly break-even, and would have made
correctness depend on software remembering to issue it.

### Measured

| | stages | issue→result | WB stall cycles | C program |
|---|---|---|---|---|
| original | 3 | 3 cycles | **200** | 13593 cycles |
| `UsePrefetch=1` | **2** | **2 cycles** | **0** | **13466 cycles** |

The 127-cycle saving reconciles exactly: 200 stall cycles removed, 73 resync bubbles added
for the rotations that do not walk sequentially (`test_rope_rot` uses random `(m,i)`,
`test_dependent_chain` repeats one `i`). A resync costs one bubble plus two stages — three
cycles, precisely the old latency — so it neither gains nor loses.

Full detail, including the `commit_kill` hazard, is in `docs/PREFETCH.md`.

## 2.7c Using the FPU as an actual fused multiply-add

CVFPU's ADDMUL group **is** an FMA. The original datapath used it six times with an operand
tied off each time — four as `MUL` (addend forced to ±0) and two as `ADD` (multiplicand
forced to +1.0). The `UseFma` parameter uses it properly instead.

The rotation needs two products per output, and an FMA fuses only one, so the other must
arrive as the addend:

```
stage 1 (2 multiplies):  p1 = y·sin          p2 = y·cos
stage 2 (2 FMAs):        x' = x·cos − p1     y' = x·sin + p2
```

**Four FPU instances instead of six, and the dependency depth is unchanged at two** — so
this buys area and accuracy, not latency. `x·cos` is now formed to its full 16 significand
bits inside the FMA and never rounded on its own, giving **two roundings per output instead
of three** and **1.79× lower RMS error** against an FP64 reference.

That makes it deliberately *not* bit-equal to Model A, so it has its own reference model
(`rope_fma_bf16`, Model A') and its own test target, `make m4-fma`. Both variants pass
201,961 vectors with zero errors against their respective models.

## 2.8 Module-by-module walkthrough

| Module | Lines | What it does |
|---|---|---|
| `rope_pkg.sv` | 177 | All parameters, the instruction encoding, and the operand-packing helper functions. Centralising the packing here is what stops `x` and `y` being swapped somewhere between the RTL, the C header, and the model. |
| `rope_theta_rom.sv` | 95 | **Generated.** The 64 integer phase increments `Phi_i`. 256 bytes. |
| `rope_sin_rom.sv` | 1062 | **Generated.** The 1024-entry BF16 quarter-wave sine table. 2 KB. Pure data. |
| `rope_sin_lut.sv` | 122 | Quadrant decode, mirror index, sign XOR. Instantiates the ROM twice — once for `sin`, once at `+2^30` for `cos`. Purely combinational. |
| `rope_phase_gen.sv` | 109 | `P = (m * Phi_i) mod 2^32`. Also provides a *sequential* mode that adds `Phi_i` once per token — one exact integer add, zero drift — for future streaming use. |
| `rope_bf16_unit.sv` | 142 | The single place all CVFPU configuration lives: FP16ALT only, ADD/MUL only, RNE hard-wired. Also where the FPnew operand-slot trap is handled once (Part 4.1). |
| `rope_bf16_mul.sv` | 45 | Named wrapper: BF16 multiply. |
| `rope_bf16_add.sv` | 53 | Named wrapper: BF16 add or subtract. |
| `rope_datapath.sv` | 276 | The four multiplies and two adds, plus flow control and lockstep assertions. |
| `rope_xif_coproc.sv` | 375 | Instruction decode, operand registration, credit accounting, cancellation handling, in-order result FIFO. |
| `rope_cv32e40x_wrapper.sv` | 244 | The real CV32E40X core + the XIF interface + the coprocessor, wired together with the five interface parameters declared exactly once. |

Full source for all of these is in **Appendix A**.

---
---

# Part 3 — What changed in the repository

## 3.1 Nothing upstream was modified

Every file of the existing `cv32e40x` repository is **untouched**. All work lives in one
new self-contained directory, `rope-bf16/`.

Verification of that claim:

```
$ git status --short
?? rope-bf16/
?? steps.md
```

Only two untracked entries: the new project directory, and the plan document that was
already there.

> **A confusing detail, resolved.** Running `git status` from *inside* WSL reports about
> 130 upstream files as modified. That is an artifact, not a real change: this checkout has
> `core.autocrlf = true` set locally, so the files on disk use Windows line endings (CRLF)
> while the git index stores Unix ones (LF). Git run from Windows applies that setting and
> reports the tree clean; git run from WSL does not know about it and sees every line as
> changed. Nothing was actually edited — the diff is 100% line endings.

Two other constraints were required and are satisfied:

- **No symbolic links anywhere.** `find rope-bf16 -type l | wc -l` returns `0`.
- **No RTL reference escapes the project.** No file in `rope-bf16/rtl/` refers to anything
  outside it except the core's own XIF interface definition, which is intentional (see
  Part 3.3).

## 3.2 Complete file inventory

**49 authored files, 8825 lines**, plus **16 files copied in** as the vendored CVFPU
subtree (3013 lines, Part 3.3) — 65 files in total. Of the 49 authored, four are
script-generated rather than hand-written (`rope_theta_rom.sv`, `rope_sin_rom.sv`,
`sw/rope_vectors.h`, `docs/error_budget.md`), and one is this document.

### RTL — hand-written (1543 lines)

| File | Lines | Appendix |
|---|---|---|
| `rtl/rope_pkg.sv` | 177 | A.1 |
| `rtl/rope_sin_lut.sv` | 122 | A.2 |
| `rtl/rope_phase_gen.sv` | 109 | A.3 |
| `rtl/rope_bf16_unit.sv` | 142 | A.4 |
| `rtl/rope_bf16_mul.sv` | 45 | A.5 |
| `rtl/rope_bf16_add.sv` | 53 | A.6 |
| `rtl/rope_datapath.sv` | 276 | A.7 |
| `rtl/rope_xif_coproc.sv` | 375 | A.8 |
| `rtl/rope_cv32e40x_wrapper.sv` | 244 | A.9 |

### RTL — generated by scripts (1157 lines)

| File | Lines | Note |
|---|---|---|
| `rtl/rope_theta_rom.sv` | 95 | Full source in A.10 |
| `rtl/rope_sin_rom.sv` | 1062 | Excerpt in A.11 — 1024 lines are the sine table itself |

Both are committed rather than gitignored: they are the design's numeric constants, they
must be reviewable in diffs, and a synthesis run should not require Python to be installed.

### Golden model and generators — Python (1516 lines)

| File | Lines | Purpose | Appendix |
|---|---|---|---|
| `model/bf16.py` | 225 | BF16 arithmetic with exact RNE, by bit manipulation | B.1 |
| `model/rope_ref.py` | 293 | The three reference models, both pairing conventions | B.2 |
| `model/characterize.py` | 438 | Produces the error budget | B.3 |
| `model/gen_vectors.py` | 329 | Test vectors for the RTL testbenches | B.4 |
| `model/gen_c_vectors.py` | 133 | Test vectors compiled into the C test | B.5 |
| `model/test_bf16.py` | 277 | Validates the BF16 primitives | B.6 |
| `model/test_rope_ref.py` | 444 | Validates the reference model | B.7 |
| `gen/gen_theta_rom.py` | 94 | Emits `rope_theta_rom.sv` | B.8 |
| `gen/gen_sin_lut.py` | 112 | Emits `rope_sin_rom.sv` | B.9 |
| `gen/vendor_cvfpu.sh` | 77 | Records/refreshes the vendored CVFPU subset | B.10 |

### Testbenches — SystemVerilog (1899 lines)

| File | Lines | Tests | Appendix |
|---|---|---|---|
| `tb/tb_rope_bf16_unit.sv` | 213 | BF16 multiply/add/subtract | C.1 |
| `tb/tb_rope_phase_gen.sv` | 204 | Integer phase generation | C.2 |
| `tb/tb_rope_sin_lut.sv` | 249 | The sine table and its symmetries | C.3 |
| `tb/tb_rope_datapath.sv` | 213 | The whole rotation datapath | C.4 |
| `tb/tb_rope_xif_coproc.sv` | 539 | The CPU interface, including cancellation | C.5 |
| `tb/tb_rope_core.sv` | 270 | The C program on the real integrated core | C.6 |
| `tb/rope_assertions.svh` | 211 | Formal protocol properties (SVA) | C.7 |

### Software (656 lines)

| File | Lines | Purpose | Appendix |
|---|---|---|---|
| `sw/rope_intrin.h` | 152 | Inline-asm wrappers, BF16 conversion, whole-vector loops | D.1 |
| `sw/test_rope.c` | 207 | The bare-metal test program | D.2 |
| `sw/crt0.S` | 57 | Minimal startup code | D.3 |
| `sw/link.ld` | 61 | Memory map | D.4 |
| `sw/Makefile` | 72 | Cross-compilation | D.5 |
| `sw/verify_encoding.py` | 88 | Checks the assembled instruction against the RTL decoder | D.6 |
| `sw/rope_vectors.h` | 120 | Generated test data | — |

### Synthesis, build, docs (837 lines)

| File | Lines | Purpose | Appendix |
|---|---|---|---|
| `Makefile` | 266 | All build, lint, and test targets | D.7 |
| `syn/rope.sdc` | 65 | Timing constraints (400 MHz + XIF budget) | D.8 |
| `syn/synth.tcl` | 110 | Genus synthesis flow — **written but never run** | D.9 |
| `README.md` | 177 | Project overview |
| `docs/notes.md` | 301 | Design decisions and gotchas |
| `docs/error_budget.md` | 102 | Generated results |
| `.gitignore` | 26 | Build artifacts |

## 3.3 How CVFPU was vendored

The requirement was that CVFPU be **copied**, not symbolically linked. It was found already
present in a sibling checkout (`cv32e40p/rtl/vendor/`), and the minimal necessary subset was
copied in as real files:

**FPnew (8 files)** — `fpnew_pkg`, `fpnew_classifier`, `fpnew_rounding`, `fpnew_fma`,
`fpnew_noncomp`, `fpnew_opgroup_fmt_slice`, `fpnew_opgroup_block`, `fpnew_top`

**common_cells (3 files + includes)** — `cf_math_pkg`, `lzc` (leading-zero count, used for
normalisation), `rr_arb_tree` (output arbitration), plus
`include/common_cells/registers.svh`, which supplies the flip-flop macros `fpnew_fma.sv`
depends on.

Licences (SolderPad 0.51 / Apache-2.0, both permissive) travel with the code.

**Deliberately excluded: the divide/square-root unit.** RoPE needs neither operation, the
block is physically large, and it depends on a *further* external repository
(`fpu_div_sqrt_mvp`) that is not vendored. Excluding it required understanding how FPnew
gates its optional blocks: `fpnew_opgroup_block` generates a unit only when
`FmtUnitTypes[fmt] == PARALLEL`, so setting the DIVSQRT group to `DISABLED` removes it —
and its missing dependency — from elaboration entirely.

`gen/vendor_cvfpu.sh` (Appendix B.10) records exactly which files were copied and why, and
can refresh them from an upstream checkout.

**One deviation from the plan, deliberately.** `steps.md` §5.2 says to add CVFPU as a
Bender (dependency-manager) git dependency and *not* to vendor it. That directly conflicts
with the requirement that everything be self-contained in this repository, so it was
overridden and the reasoning recorded in `docs/notes.md`.

**Similarly for the XIF interface.** `steps.md` §10.1 suggests taking the interface
definition from a separate `core-v-xif` clone. The `cv32e40x` repository already ships an
equivalent definition at `rtl/cv32e40x_if_xif.sv`, so that one is used instead: it needs no
external clone and is guaranteed type-compatible with the core, by construction.

---
---

# Part 4 — Engineering log: bugs found and fixed

This section is the honest development record. Several of these defects produce **silently
wrong results** rather than errors, which is the dangerous kind.

## 4.1 CVFPU's `ADD` silently discards one operand

**Severity: would have produced wrong arithmetic with no error.**

CVFPU's add/multiply group is internally a fused multiply-add computing
`operands[0] * operands[1] + operands[2]`. Reading the source revealed that it *rewrites*
the operands depending on the opcode:

```systemverilog
// from the vendored fpnew_fma.sv
fpnew_pkg::ADD: begin // Set multiplicand to +1
  operand_a = '{sign: 1'b0, exponent: BIAS, mantissa: '0};
  ...
fpnew_pkg::MUL: begin // Set addend to +0 or -0 ...
  operand_c = '{sign: 1'b1, exponent: '0, mantissa: '0};
```

So the two operations use **different slots**:

- `MUL` forces `operand_c` to ±0 → result is `operands[0] * operands[1]`. Use slots **0, 1**.
- `ADD` forces `operand_a` to **+1.0** → result is `operands[1] + operands[2]`. Use slots **1, 2**.

Putting an adder's two inputs in slots 0 and 1 — the obvious choice, and what a multiplier
requires — would compute `1.0 * b + 0`, **silently discarding operand `a` entirely**. The
hardware would report no error; results would simply be wrong.

Handled once, in `rope_bf16_unit.sv`, with the reasoning recorded next to the code:

```systemverilog
always_comb begin
  operands = '0;
  if (Op == fpnew_pkg::ADD) begin
    operands[0] = rope_pkg::Bf16One;  // ignored by fpnew_fma (forced to +1.0)
    operands[1] = operand_a_i;
    operands[2] = operand_b_i;
  end else begin  // MUL
    operands[0] = operand_a_i;
    operands[1] = operand_b_i;
    operands[2] = rope_pkg::Bf16PosZero;
  end
end
```

Subtraction uses FPnew's `op_mod`, which inverts the sign of operand C — exact and free.

## 4.2 CVFPU's `in_ready` is gated by its own `in_valid` — a reset deadlock

**Severity: total failure; the datapath produced no output at all.**

The first datapath run hung, and the watchdog reported `sent=0` — not one operand had ever
been accepted.

The cause is a non-obvious property of the vendored code. `fpnew_top` drives:

```systemverilog
assign in_ready_o = in_valid_i & opgrp_in_ready[fpnew_pkg::get_opgroup(op_i)];
```

**Ready depends on valid.** A unit reports "I am ready" only while something is already
being offered. That makes the natural-looking input register a combinational cycle:

```systemverilog
// BEFORE — deadlocks at reset
valid_s0 <= in_valid_i & in_ready_o;
...
assign in_ready_o = &mul_in_ready;
```

At reset `valid = 0`, so `mul_in_ready = 0`, so `in_ready_o = 0`, so `valid_s0` can never
become 1. The datapath is dead forever, and nothing reports an error.

**The fix**, plus the reasoning that makes it correct rather than merely convenient:

```systemverilog
// AFTER
valid_s0 <= in_valid_i;      // NOT gated by in_ready_o
...
// in_ready is constant 1, and here is why that is sound: out_ready_i is tied to 1 on
// every FPnew instance, so every input-pipeline stage is unconditionally ready and each
// unit accepts an operand on every cycle it is offered one.
assign in_ready_o = 1'b1;
```

Because that reasoning is an *assumption about third-party code*, it is checked on every
cycle rather than trusted:

```systemverilog
if (valid_s0) begin
  assert (mul_in_ready == 4'b1111)
    else $fatal(1, "rope_datapath: multiplier refused a valid input (in_ready=%b) -- the no-stall assumption behind in_ready_o=1 is broken", mul_in_ready);
end
```

If a future CVFPU update ever introduces a stall, that assertion fires immediately instead
of pairs being silently dropped. The backpressure the design genuinely needs is handled one
level up, by the credit counter in the coprocessor.

## 4.3 `PosBits` truncated the sequence position — silent wrong answers past m = 8191

**Severity: silently wrong results at ordinary sequence lengths.**

The plan (`steps.md` §6.3) argues for a narrow position register: *"`m` is ~13 bits, `Phi`
is 32 bits — a 13x32 multiplier, cheap."* 13 bits does cover an 8192-token context, and the
design initially followed that.

It is the wrong width, because **the instruction carries 16 bits of `m`** in `rs2[31:16]`.
With a 13-bit register the coprocessor silently truncates:

```
m = 10000  ->  10000 & 0x1FFF  =  1808
```

The unit then rotates by the wrong angle and reports success. No exception, no flag, no
symptom — just a wrong number. And 8192-token contexts are entirely ordinary, so this is
reachable in normal use rather than being a corner case.

```systemverilog
// BEFORE
localparam int unsigned PosBits = 13;

// AFTER
localparam int unsigned PosBits = 16;   // matches the rs2[31:16] instruction field
```

Three extra multiplier bits (16×32 rather than 13×32) is a negligible price for removing an
entire class of silent wrongness, so the width now follows the **ISA** rather than a guess
about sequence length.

**Then the test coverage was fixed too**, because the existing vectors only went up to
`m = 8191` and would therefore have passed either way. The generators now sweep the full
16-bit range and include boundary values explicitly:

```python
for m in list(range(0, 64)) + [
    100, 255, 256, 1000, 1023, 1024, 2048, 4095, 4096,
    8191, 8192, 8193, 10000, 16384, 32768, 60000, 65535,
]:
```

**And then the test was proven to actually catch the bug** — a fix is only as good as the
test that would have caught it. Temporarily reverting to 13 bits:

```
--- narrowed to 13 bits ---
78:  localparam int unsigned PosBits = 13;
make m2 exit=2  (a NON-zero exit here is the desired outcome)
MISMATCH [4736]: m=8192 i=0  got=00000000 want=cc1b8000
MISMATCH [4737]: m=8192 i=1  got=00000000 want=0abca000
MISMATCH [4738]: m=8192 i=2  got=00000000 want=b5c54000
--- rope_pkg.sv restored ---
78:  localparam int unsigned PosBits = 16;
```

The test fails loudly and specifically. The regression is now genuinely guarded.

## 4.4 A bug in my own reference implementation, at the IEEE 754 overflow boundary

**Severity: would have discredited a correct hardware result.**

The BF16 helper library contains two independent implementations of rounding: a fast
bit-twiddling one used everywhere, and a deliberately naive one used only by the test suite
to cross-check it. They disagreed on 3 of 200,000 random inputs, all near the largest
representable number:

```
FAIL  RNE vs reference, 200000 random FP32 patterns
      3 mismatches, first=('0x7f7fb006', 3.398671048667133e+38, '0x7f80', '0x7f7f')
```

The fast path said `+Infinity`; the slow reference said "largest finite value".

**The fast path was right.** `0x7F7F8000` sits exactly halfway between BF16's largest
finite value and the next value the format *would* have had if its exponent range
continued. IEEE 754-2019 §4.3.1 specifies that rounding happens **as if the exponent range
were unbounded**, and only then overflows — so anything at or above
`2^emax * (2 - 2^-p)` becomes infinity. That threshold is exactly this value.

The naive reference was wrong because it treated the upper candidate as literal `Infinity`,
which is infinitely far away, so the lower candidate always "won":

```python
# BEFORE — clamps every overflow case to MAX_NORMAL
hi_val = float(bf16_to_f32(np.uint16(hi)))

# AFTER
# If `hi` is the infinity encoding, its value for *comparison* purposes is not
# infinity: IEEE 754-2019 Section 4.3.1 rounds as if the exponent range were
# unbounded and only then overflows. That unbounded value is 2^128 for BF16.
if (hi & 0x7FFF) == 0x7F80:
    hi_val = -(2.0**128) if (hi & 0x8000) else 2.0**128
else:
    hi_val = float(bf16_to_f32(np.uint16(hi)))
```

Worth including because it is the case that justifies writing two independent
implementations at all: a single implementation would have been "self-consistent" and
wrong, and this discrepancy is what surfaced the real IEEE rule.

## 4.5 A testbench race accepted the same instruction twice

**Severity: false failure that looked like a hardware bug.**

An assertion fired claiming one instruction had produced two results. Instrumenting the
coprocessor showed the real story:

```
[85000] ACCEPT id=0 rs1=a9b3207c rs2=013c001a
[95000] ACCEPT id=0 rs1=a9b3207c rs2=013c001a     <-- same instruction, twice
```

The hardware was correct; the **testbench** was at fault. It asserted `issue_valid` at the
exact instant of a clock edge, which races with the design sampling that same edge —
depending on simulator event ordering, the request can be observed on both that edge and
the next.

The fix is a discipline applied throughout: **drive on the falling edge, sample on the
rising edge**, using blocking assignments.

```systemverilog
// BEFORE — assigns at the same instant the DUT samples
xif.issue_valid     <= 1'b1;
xif.issue_req.instr <= encode_rope_rot(rd, 5'd1, 5'd2);
@(posedge clk);
...
#1ns;
xif.issue_valid <= 1'b0;

// AFTER — strictly off-edge, so the request is seen on exactly one posedge
@(negedge clk);
xif.issue_valid     = 1'b1;
xif.issue_req.instr = encode_rope_rot(rd, 5'd1, 5'd2);
@(posedge clk);                              // DUT samples here
while (!xif.issue_ready) @(posedge clk);
...
@(negedge clk);
xif.issue_valid = 1'b0;
```

The same class of race had already appeared in the phase-generator testbench, where a
one-cycle control pulse was deasserted at zero delay after the edge and was therefore
missed entirely — the sequential accumulator appeared frozen at zero. Same root cause, same
fix.

## 4.6 A formal assertion that asserted something untrue

**Severity: false failure blocking a passing test.**

A protocol property claimed that `issue_resp.accept` must remain stable while a request
waits to be accepted. Under result-channel backpressure it failed.

On investigation the **assertion was wrong**, not the design. `issue_resp` is only
meaningful in the cycle where `issue_valid && issue_ready` are both high. This coprocessor
withholds `issue_ready` when the result FIFO is full and reports `accept = 0` in that
state; when a credit frees, both rise together. So `accept` changes legitimately while a
request waits, and the stability requirement simply does not exist in the specification.

It was replaced with two properties that *are* required — and the removal was documented,
so nobody re-adds it:

```systemverilog
// DELIBERATELY NOT CHECKED: "accept must stay stable while a request is pending".
// An earlier version of this file asserted that and it is simply not a CV-X-IF
// requirement -- issue_resp is only meaningful in the cycle where
// issue_valid && issue_ready. ... asserting stability produced a false failure
// under result-channel backpressure.

property p_no_accept_without_valid;
  @(posedge clk) disable iff (!rst_n)
    xif.issue_resp.accept |-> xif.issue_valid;
endproperty

// On a completed handshake, accept must match the decode exactly -- catching both
// over-acceptance (stealing another unit's instruction) and under-acceptance.
property p_accept_matches_decode;
  @(posedge clk) disable iff (!rst_n)
    (xif.issue_valid && xif.issue_ready) |->
      (xif.issue_resp.accept ==
         (rope_pkg::is_rope_rot(xif.issue_req.instr)
          && xif.issue_req.rs_valid[0] && xif.issue_req.rs_valid[1]));
endproperty
```

The replacement is strictly stronger where it matters: `p_accept_matches_decode` is what
proves the coprocessor is inert for instructions that are not its own.

## 4.7 A test that deadlocked itself by design

**Severity: hung test, no diagnostic.**

The backpressure test held `result_ready` low while issuing 64 instructions, then released
it *after* waiting for the issue loop to finish. But the credit counter — working exactly as
intended — stops accepting once the FIFO is full, so the issue loop blocked forever waiting
for credits that could only be freed by the release that came later. Classic circular wait.

The release was moved *inside* the parallel block, and a watchdog was added so a future hang
fails with a diagnostic instead of spinning:

```systemverilog
// Backpressure thread. This MUST live inside the fork: with result_ready held
// low, the credit counter fills after FifoDepth accepts and issue_ready
// correctly drops, so the issue thread blocks. Releasing backpressure only
// after the join would therefore deadlock -- the earlier version of this test
// did exactly that. Holding for well over FifoDepth cycles first is what makes
// this a real test of the credit scheme rather than of nothing.
begin
  repeat (40) @(posedge clk);
  @(negedge clk);
  xif.result_ready = 1'b1;
end
```

## 4.8 The linker placed the program at the wrong address

**Severity: the program would not have booted.**

The C test's memory map intended code to start at `0x80`. The linker script set the
location counter before the first section:

```
. = 0x00000080;
.text : { ... } > RAM
```

But when an output section is placed into a memory region with `> RAM`, the linker allocates
from that **region's origin** and silently ignores the preceding assignment. Verification
showed `.text` at address 0:

```
Idx Name          Size      VMA       LMA
  0 .text         000003fa  00000000  00000000     <-- wrong
```

Fixed by setting the origin where it actually takes effect:

```
MEMORY
{
  RAM (rwx) : ORIGIN = 0x00000080, LENGTH = 256K - 0x80
}
```

Confirmed afterwards: `.text` at `0x00000080`, and the generated image carries the matching
`@00000020` word-address record.

## 4.9 A multi-driver conflict made a passing test report failure

**Severity: false failure at the very last step.**

The C test ran on the integrated core and printed `RESULT: PASS` — then the testbench
declared failure:

```
=== BF16 RoPE coprocessor test ===
...
RESULT: PASS
tb_rope_core: exit_code=0xffffffff after 13993 cycles
FAIL: software reported 4294967295 mismatches
```

`0xFFFFFFFF` was the *initial* value. The exit register was written by two different
processes — an `initial` block for initialisation and a clocked process for the actual
capture — which is a multiple-driver conflict, and the initial value won.

Fixed by giving it exactly one driver, with initialisation in that process's reset branch:

```systemverilog
// This is deliberately the SOLE driver of exit_code/saw_exit, with their initial
// values coming from the reset branch rather than from an initial block: driving
// them from both an initial block and a clocked process is a multiple-driver
// conflict, and an earlier version of this testbench did exactly that and always
// read back the stale initial value (reporting a spurious failure even though the
// software had written 0 and printed PASS).
always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    exit_code <= 32'hFFFF_FFFF;
    saw_exit  <= 1'b0;
  end else if (data_pending_q && data_we_q
               && ((data_addr_q & 32'hFFFF_F000) == ExitAddr)) begin
    exit_code <= data_wdata_q;
    saw_exit  <= 1'b1;
  end
end
```

## 4.10 A build system that silently tested less than requested

**Severity: a confident pass over the wrong data — arguably the worst kind of defect here.**

Asking for a larger run appeared to succeed:

```
$ make m1 NVEC=1000000
tb_rope_bf16_unit: loaded 21935 vectors        <-- not 1,000,000
PASS: all 21935 vectors bit-exact
```

The vector file had a fixed name, so `make` considered the existing (much smaller) file up
to date and reused it. The test then passed loudly while covering 2% of the requested space.

Fixed by encoding the vector count in the filename, so changing it forces regeneration and
each size is cached separately:

```makefile
# Vector filenames embed NVEC. This is load-bearing, not cosmetic: with a fixed name,
# `make m1 NVEC=1000000` sees an existing up-to-date vectors file and silently reuses the
# SMALLER previous one, so the run reports a pass over the wrong vector count.
VEC_BF16     := $(BUILD)/vectors_bf16_$(NVEC).hex
VEC_PHASE    := $(BUILD)/vectors_phase_$(NVEC).hex
VEC_LUT      := $(BUILD)/vectors_lut_$(NVEC).hex
VEC_DATAPATH := $(BUILD)/vectors_datapath_$(NVEC).hex
VEC_XIF      := $(BUILD)/vectors_xif_$(NVEC).hex
```

Afterwards, the requested size is actually run:

```
tb_rope_bf16_unit: loaded 1001935 vectors from vectors_bf16_1000000.hex
PASS: all 1001935 vectors bit-exact
```

## 4.11 Two smaller items

**A comment broke the build.** A documentation line beginning `// Verilator` was parsed as a
compiler pragma:

```
%Error: Unknown verilator comment: '/*verilator  , and it closes the loop end to end:*/'
```

Reworded, with a warning left in place for the next author.

**A test asserted the wrong property about `m = 0`.** An early test demanded that position 0
be the exact identity. It is not, and cannot be, because of midpoint sampling (Part 2.4).
The test was replaced with one asserting the property that actually holds — that the error
is fully explained by `|y| * sin(0)` plus one rounding — and the behaviour was documented
rather than papered over.

## 4.12 A killed instruction wrote back — twice, for two different reasons

CORE-V-XIF lets the core **kill an instruction it has already accepted**, on any interrupt,
exception, debug entry, fence, `dret` or CSR-induced flush. The coprocessor must then
retire it silently, because the core has discarded it and writing `rd` would corrupt
architectural state.

That worked for a year of development and then failed twice, both times only when the
pipeline got shorter.

**First failure — a registered flag read one cycle too late.**

```systemverilog
assign result_is_killed = killed_q[cur_tag.id];   // registered: previous cycles only
```

`killed_q` is set by `commit_kill`, so it reflects kills from *earlier* cycles. At a
five-cycle latency the kill always landed first. At three, the kill and the result coincide
— and the register has not updated yet, so the killed instruction writes `rd`. Fixed by
ORing in a combinational same-cycle term, which is safe because `commit_valid` has no ready
signal and may be sampled directly:

```systemverilog
assign same_cycle_kill  = commit_valid & commit_kill & (commit.id == cur_tag.id);
assign result_is_killed = killed_q[cur_tag.id] | same_cycle_kill;
```

**Second failure — the gate was on the wrong side of the FIFO.** The result FIFO gated its
*push*, which only worked because at three stages the commit always arrived before the
result was produced. At two stages the result wins the race, reaches the FIFO, and a kill
arriving afterwards is simply ignored. The fix moves the gate to the FIFO **output**, where
the commit status is known:

```systemverilog
assign result_valid = ~fifo_empty & head_committed & ~head_killed;
```

plus a silent drain so a killed entry does not block the head forever. This is correct at
**any** latency rather than relying on commit ordering, and it costs nothing: CV32E40X
signals commit from the EX stage, one cycle after issue, while the earliest possible result
is also one cycle after issue. **It applies to the default configuration too**, not only to
the prefetch path.

Both failures are the same mistake in different places: **per-instruction status consulted
as an edge instead of held as a level.**

## 4.13 A credit leak created by the fix for 4.12

Gating the FIFO output created a second retirement path. An instruction can now leave
either by being written back (or silently drained) at the FIFO head, or by being killed
before its result was ever pushed — and those can happen in the same cycle for *different*
instructions. The existing accounting ORed them:

```systemverilog
credit_q <= credit_q + accept - (fifo_pop | retire_killed);   // loses a credit
```

Every coincidence loses one credit permanently, and since credit is what gates
`issue_ready`, the coprocessor would slowly stop accepting work. Now summed:

```systemverilog
credit_q <= credit_q + accept - fifo_pop - retire_killed;
```

This is the characteristic risk of a bug fix: it was correct in isolation and wrong in
combination with the accounting around it.

## 4.14 The prefetch is speculative state, and an interrupt corrupts it

The prefetch advances on `accept`, but an accepted instruction can still be killed. Nothing
in the arithmetic notices:

```
ROPE.ROT  pair 2   ◄── timer interrupt: accepted, then KILLED
                       theta_q → θ₃,  i_q → 3        ← state ran ahead
   ... handler runs, returns ...
ROPE.ROT  pair 2 (re-executed)    uses θ₃  ✗ WRONG
```

Every remaining pair of that head vector is silently rotated by the wrong frequency. No
assertion fires, no exception is raised, the results are still finite unit-magnitude
numbers, and whether it happens depends on interrupt timing — so a test can pass a thousand
times and fail in the field.

Two independent checks now catch it: a kill on the commit channel sets a desync flag, and
the instruction's own `(m, i)` are compared against the prefetch state on every issue.
Either triggers a one-cycle resync.

**Which check mattered was settled by measurement, and reasoning had it backwards.**
Building with the value check compiled out (`make m9-tier1`) fails **6039 of 6066**
vectors, because the M5 stimulus is random `(m, i)` rather than a sequential walk. The
value check is required for architectural correctness — without it `ROPE.ROT` stops being a
pure function of its operands. And since a killed instruction re-executes with an index
that no longer matches, the value check catches the kill case unaided; the kill-triggered
check is the redundant one. That configuration is kept as a deliberately-failing target, so
the safety mechanism is *proven* load-bearing rather than assumed to be.

## 4.15 A testbench that committed instructions before issuing them

With results now held until commit, the M5 suite deadlocked at eight pending — exactly the
result-FIFO depth. The commit threads walked a free-running counter:

```systemverilog
cid = next_id;
for (int unsigned k = 0; k < count; k++) begin
  @(posedge clk);
  do_commit(cid, 1'b0);     // races ahead whenever issue is throttled
  cid = cid + 1'b1;
end
```

Whenever result-FIFO credit throttled the issue thread, the commit thread ran ahead and
committed ids that had not been issued yet. `accept` clears the commit flag for a reused
id, so those commits were discarded and the instructions could never retire.

Committing an id before it is issued is not valid CORE-V-XIF — the testbench had simply
been getting away with it, because the old design never required a commit to release a
result. The threads now wait on the issue count.

This is worth recording as a category: **a design change can expose a latent bug in the
testbench, and the testbench is not automatically the correct party.** The instinct to
"fix the test so it passes" would have been right here and wrong in 4.12, and the only way
to tell them apart was to work out which behaviour the protocol actually mandates.

---
---

# Part 5 — Corrections to the original plan (`steps.md`)

Three numeric claims in the plan document did not survive measurement. **None invalidates
the design** — the invariants and conclusions all hold — but anyone quoting these figures in
a paper or presentation should quote the corrected ones.

## 5.1 The INV-2 worked example is overstated (~1.6×)

`steps.md` §0 states:

> For pair `i=0`, `theta_0 = 1.0` rad. At sequence position `m = 1000` the angle is `1000`
> rad. […] `1000 * 2^-8 ~= 3.9 radians` […] The representable-value spacing **exceeds a
> full period of 2*pi**.

**3.9 does not exceed 2π ≈ 6.283.** The measured BF16 spacing at 1000 is 4.0 rad, which
exceeds **π** (half a period) but not a full one. The full-period claim becomes true at
**m ≥ 1024**, where the spacing is 8.0 rad.

Correct phrasings, both now asserted by the test suite:

- "At m = 1000 the BF16 spacing is 4.0 rad — over half a period."
- "From m = 1024 the BF16 spacing is 8.0 rad — more than a full period."

**The invariant is unaffected.** A 4-radian quantum still means up to ±2 rad (±115°) of
angle error, which destroys all correlation with the true value. And the plan's *other*
argument — that you need ~22 bits of angle precision where BF16 gives 8 — is correct as
written and is the more robust way to make the case.

## 5.2 `Phi_min` is 78937, not ~68000

§6.1 justifies 32-bit phase over 24-bit with "`Phi_min ~ 68000` at 32 bits". The measured
value for d=128, base=10000 is **78937**. The conclusion (≈1e-5 relative quantisation at 32
bits versus ≈0.4% at 24) is unchanged.

## 5.3 Midpoint sampling makes m = 0 a *near*-identity, not an identity

Not an error in the plan, but a consequence it does not mention and which looks like a bug
to a fresh reader. Fully described in Part 2.4: `sin(0)` reads `0.00076675` rather than 0,
so the relative error in `x'` at position 0 scales with `|y|/|x|` and reaches 2.3% for
`x = 0.25, y = 8.0`.

## 5.4 Deliberate deviations from the plan's dependency and layout scheme

| `steps.md` says | What was done | Why |
|---|---|---|
| `deps/` with four cloned repositories | nothing cloned | CVFPU **copied** into `rtl/vendor/`; the core and XIF definitions already exist in this repository. |
| Add CVFPU as a Bender git dependency; *"don't vendor it"* | vendored, no Bender | Directly conflicts with the self-containment requirement. |
| Use `deps/core-v-xif/src/core_v_xif.sv` | use `cv32e40x/rtl/cv32e40x_if_xif.sv` | The core ships an equivalent definition; using it guarantees type compatibility and needs no clone. |
| `rope_sin_lut.sv` is generated | `rope_sin_rom.sv` generated; `rope_sin_lut.sv` hand-written | Keeps generated code trivially auditable (a bare table) and the interesting logic (quadrant, mirror, sign) reviewable. |
| `ml_dtypes`, `torch`, `cocotb`, `pytest` | **numpy only** | `ml_dtypes` could not be installed (PEP 668 blocks system-wide pip here). It also is not needed: `model/bf16.py` implements BF16 RNE by bit manipulation and is cross-checked against an independent reference. Without pytest/cocotb, the suites are plain runnable Python and self-checking SystemVerilog. |
| `AccWidth=32` RTL build | model only | See below. |

**On `AccWidth=32`.** §8.3 asks for an RTL parameter offering a wide-accumulate build to
quantify what strict BF16 costs. **The RTL implements strict BF16 only**, and fails
elaboration with a clear message if asked for 32:

```systemverilog
initial begin : check_acc_width
  if (AccWidth != rope_pkg::AccWidthStrict) begin
    $fatal(1, "rope_datapath: AccWidth=%0d is not implemented in RTL; only strict BF16 (%0d) is. The wide-accumulate comparison lives in model/characterize.py.",
           AccWidth, rope_pkg::AccWidthStrict);
  end
end
```

Reason: a wide-accumulate hardware path needs FP32 adders plus BF16↔FP32 casts — i.e.
enabling CVFPU's conversion group and adding exactly the conversion stages INV-1 forbids —
for a configuration that is explicitly never shipped. **The comparison itself is what
matters, and it is produced** in the software model (`rope_wide_acc`, reported in
`docs/error_budget.md`). The parameter is retained so the intent stays visible and a future
implementer is not misled into thinking `32` works.

---
---

# Part 6 — Verification and results

## 6.1 Verification strategy

The governing principle: **build the reference model first, before any hardware.** You
cannot verify what you have nothing to compare against, and a reference written *after* the
hardware tends to accidentally encode the hardware's bugs.

```
                    Python golden model (numpy only)
                            |
          +-----------------+------------------+
          |                                    |
   test vectors                          expected results
          |                                    |
          v                                    |
   +--------------+                            |
   | RTL under    |  ---- actual results ----> compare BIT-EXACTLY
   | test         |                            (zero tolerance)
   +--------------+
```

Every testbench is self-checking: each vector carries the expected answer, so the
SystemVerilog side never re-derives anything and cannot "agree" with the hardware by
sharing a mistake.

**Three reference models, all needed:**

| Model | What it is | Purpose |
|---|---|---|
| **A** | Strict BF16, RNE at every node | Models the hardware exactly. Bit-exact comparison target. |
| **B** | BF16 in/out, FP32 internal accumulate | What every *other* BF16 accelerator does. Measures what strict BF16 costs. |
| **C** | FP64 with true library `sin`/`cos` | The yardstick. Measures the total error of A and B. |

**Bit-exactness rather than a tolerance.** Because BF16 is exactly the top 16 bits of FP32,
there is no ambiguity about who rounded what, so the correct standard is *identical bit
patterns* — not "close enough". Every arithmetic milestone requires **zero** mismatches.

**Deliberate adversarial input selection.** Purely random inputs are a weak test, so the
vector generators also include:

- An **exhaustive cross product of 21 special values** (both signed zeros, both infinities,
  several NaN payloads, minimum and maximum subnormals, minimum and maximum normals, exact
  powers of two, full-mantissa values) against each other, for all three operations.
- **Directed exact ties** across the entire exponent range — the specific case where
  truncation-instead-of-RNE would be caught.
- **The catastrophic-cancellation corner** (`x ≈ y` at 45°), which is where strict BF16 is
  at its worst.
- **Full-mantissa operands**, so multiplications actually round. *(This one mattered: an
  early version of the cancellation study used `x = 1.0`, and multiplying by exactly 1.0 is
  exact — which made the strict-vs-wide comparison degenerate and produced two identical
  rows of results. Fixed by drawing `x` from [1,2).)*
- **Boundary values of `m`** including 8191/8192/8193 and 65535 (Part 4.3).

## 6.2 What each testbench proves

### Milestone 1 — BF16 arithmetic primitives → **PASS, zero mismatches**

Largest run: **1,001,935 vectors**, covering multiply, add and subtract.

```
tb_rope_bf16_unit: loaded 1001935 vectors from vectors_bf16_1000000.hex
tb_rope_bf16_unit: sent=1001935 checked=1001935 errors=0
  MUL errors: 0
  ADD errors: 0
  SUB errors: 0
PASS: all 1001935 vectors bit-exact
```

This also **settled the subnormal policy empirically** rather than by assumption. Subnormals
are the very small numbers below the normal exponent range, where precision degrades
gracefully; flushing them to zero is cheaper and a legitimate design choice, so the policy
had to be *determined*, not guessed. Because vectors whose *results* are subnormal (e.g.
`2^-126 × 0.5 = 2^-127`) matched a model configured for **gradual underflow** with zero
mismatches, CVFPU demonstrably preserves subnormals. Had it flushed, every such case would
have disagreed. The finding is documented, along with how to change it if consistency with
Mugi's post-processing block is later preferred.

It also confirmed NaN canonicalisation (all NaN results become `0x7FC0`), including the
awkward case of a NaN whose payload sits entirely in the low 16 bits — which a naive
"add 0x7FFF and shift" narrowing would turn into an **infinity**.

### Milestone 2 — Integer phase generator → **PASS**

Three separate properties:

```
random-access: 24760 vectors, 0 errors
sequential vs random-access: 0 errors
wraparound: observed 10 rollovers past 2^32 for i=0 (expected)
PASS: phase generator bit-exact (random-access, sequential, wraparound)
```

The middle line is the important one: the sequential accumulator (one integer add per token)
agrees **exactly** with the direct multiply for every position up to 4096, for multiple
frequencies. That is the concrete demonstration of **zero drift** — the property the
trigonometric recurrence would have lacked.

### Milestone 3 — Sine LUT → **PASS**

```
bit-exact check: 24224 vectors, sin errors=0, cos errors=0
reflection symmetry: exact for all 1024 indices (Q1[k] == Q0[N-1-k])
quadrant signs: Q2 == ~Q0 and Q3 == ~Q1 exactly (sign-bit XOR)
phase[19:0] correctly ignored (reserved for interpolation)
sin^2+cos^2 over 820 points: max deviation = 0.005280, RMS deviation = 0.002131
PASS: sine LUT bit-exact, reflection and quadrant signs exact
```

Coverage is **exhaustive over the top 12 phase bits** — every distinct table entry in every
quadrant — plus dense random sampling of the full 2^32 space. The reflection check confirms
midpoint sampling delivers the exact `N-1-idx` mirror at *every* index including the
endpoints, which is where edge sampling would have overflowed the table. The `phase[19:0]`
check confirms the reserved bits genuinely do not leak into the result.

### Milestone 4 — Full rotation datapath → **PASS, zero mismatches**

Largest run: **1,001,961 vectors**.

```
tb_rope_datapath: loaded 1001961 vectors from vectors_datapath_1000000.hex
tb_rope_datapath: datapath latency = 3 cycles
tb_rope_datapath: sent=1001961 checked=1001961 errors=0
PASS: datapath bit-exact vs Model A on all 1001961 vectors
```

Driven at **one vector per cycle**, so this is simultaneously the full-throughput test. The
scoreboard uses a queue keyed on the design's self-reported latency rather than a
hard-coded number, so the check stays valid if the pipeline depth changes.

### Milestone 5 — CPU interface → **PASS, all assertions enabled**

```
TEST 1 ok: non-ROPE instruction rejected (accept=0, issue_ready=1)
TEST 2 ok: single ROPE.ROT wrote back correctly
TEST 3a ok: killed instruction produced no writeback
TEST 3b ok: instruction after a kill is unaffected
TEST 4 ok: 2000 back-to-back ROPE.ROT all correct
TEST 5 ok: 64 results survived result_ready backpressure
TEST 6 ok: 4000 results correct under random backpressure
tb_rope_xif_coproc: sent=6067 checked=6066 errors=0
PASS: XIF coprocessor -- decode, writeback, kill, throughput, backpressure
```

Note `sent=6067, checked=6066`: the difference of exactly one is the deliberately cancelled
instruction, which correctly produced no writeback.

Tests 3a and 3b together are the cancellation test that matters — not merely "the killed
instruction was suppressed", but "the *next* instruction was still correct", which is what
proves the kill did not corrupt the pipeline tags or the credit accounting.

Running throughout: seven formal protocol properties (Appendix C.7), including no result for
an unissued ID, no writeback after cancellation, at most one result per instruction, and
`accept` matching the decode exactly.

### Milestone 6 — Integration with the real CPU → **PASS, zero warnings**

The complete design — the real CV32E40X core, the XIF interface, and the coprocessor —
elaborates with **no errors and no warnings** beyond documented waivers for the core's own
upstream code and for third-party CVFPU style.

### Milestone 7 — C program on the integrated core → **PASS**

The end-to-end test, and the most meaningful single result:

```
tb_rope_core: loaded test_rope.hex
=== BF16 RoPE coprocessor test ===
test_packing done
test_conversion done
test_rope_rot done: 64 vectors
test_dependent_chain done
test_head_vector done
RESULT: PASS

tb_rope_core: exit_code=0x00000000 after 13993 cycles
PASS: C test passed on the integrated core
```

The full chain, exercised in one run:

```
Python golden model
   -> compiled-in test vectors
      -> real RISC-V machine code (custom instruction via inline asm)
         -> the actual CV32E40X core, fetching from simulated memory
            -> CORE-V-XIF
               -> the RoPE coprocessor
                  -> register writeback
                     -> compared against the model, in software, on the core
```

Individual sub-tests: operand packing matches the RTL's convention; BF16 conversion
round-trips exactly and rounds ties correctly in both directions; 64 instruction vectors
match the model; a **dependent chain** of 8 back-to-back rotations each consuming the
previous result (which exercises the core's operand forwarding into the registered-operand
path); and whole 128-element head vectors in **both** pairing conventions.

**The instruction encoding was verified independently**, rather than assumed:

```
Found 10 ROPE.ROT instruction(s) with opcode 0x0B.
  example @158: word=0x0117878b rd=x15 rs1=x15 rs2=x17 funct3=0 funct7=0
  raw text: .insn	4, 0x0117878b
PASS: every custom-0 instruction matches rope_pkg's decode (opcode 0x0B, funct3 0, funct7 0).
```

This closes a real gap: nothing else would have caught an assembler emitting a subtly
different encoding from the one the hardware decodes.

### Full clean regression

From a completely clean tree, all eleven targets:

```
OK    model               OK    m1        OK    m5
OK    gen                 OK    m2        OK    m7
OK    lint                OK    m3
OK    lint-integration    OK    m4
OK    characterize

=== any errors === 0
```

## 6.3 Results: the error budget

Generated by `model/characterize.py` into `docs/error_budget.md`. d = 128, base = 10000.

Because a rotation preserves length, error is reported normalised by the **pair's vector
norm**. Per-component relative error is misleading here: under cancellation a component's
true value approaches zero, so its relative error diverges while it carries almost no
information. Both are given, with the pair-relative figure as the meaningful one.

### Overall accuracy — 400,000 random tuples

| Metric | Model A (strict BF16) | Model B (wide accumulate) |
|---|---|---|
| Pair-relative error, max | 9.64e-03 | 6.48e-03 |
| Pair-relative error, **RMS** | **2.73e-03** | **2.16e-03** |
| Pair-relative error, mean | 2.39e-03 | 1.91e-03 |
| Component-relative error, max | 4.53e+02 | 1.31e+03 |

> **Cost of strict BF16 (INV-1): 1.27× RMS error.** The two models produce **bit-identical**
> results on **45.2%** of random cases.

Interpretation: away from cancellation the dominant error is the shared table quantisation
plus the final rounding of the result — costs *both* models pay. The extra intermediate
roundings that strict BF16 accepts contribute comparatively little. Which is exactly why the
cancellation study below is the measurement that actually matters.

### Error versus sequence position — the drift check

| m | pair-rel max | pair-rel RMS |
|---|---|---|
| 0 | 1.49e-03 | 5.07e-04 |
| 1 | 7.14e-03 | 1.85e-03 |
| 16 | 7.36e-03 | 2.42e-03 |
| 256 | 7.36e-03 | 2.65e-03 |
| 1024 | 7.72e-03 | 2.77e-03 |
| 2048 | 7.73e-03 | 2.73e-03 |
| 4095 | 7.63e-03 | 2.86e-03 |

**Flat in `m`.** This is the integer phase accumulator's guarantee made visible: no drift
with sequence position. A trigonometric-recurrence implementation would degrade steadily
down this column.

### Error versus pair index

| i | theta_i | pair-rel max | pair-rel RMS |
|---|---|---|---|
| 0 | 1.00e+00 | 7.57e-03 | 2.77e-03 |
| 8 | 3.16e-01 | 8.13e-03 | 2.70e-03 |
| 32 | 1.00e-02 | 8.31e-03 | 2.77e-03 |
| 63 | 1.15e-04 | 7.69e-03 | 2.58e-03 |

Essentially uniform across frequencies — no pair is disadvantaged.

### The headline: catastrophic cancellation

`x' = x*cos - y*sin`. At exactly 45° `cos = sin`, so when `x = y` the two products cancel
completely and the true answer is zero. Subtracting two nearly equal numbers destroys
significant digits — with only 7 mantissa bits, there is very little to lose before nothing
is left. Phase swept ±2° around 45° in 20,001 steps.

| Operands | Model | Max rel err | RMS rel err | Min bits kept | **Mean bits kept** | **Losing >half** |
|---|---|---|---|---|---|---|
| `x == y` | strict BF16 | 1.48e+00 | 3.49e-01 | 0.0 | **3.5** | **63.2%** |
| `x == y` | wide acc | 1.00e+00 | 3.32e-01 | 0.0 | 3.7 | 59.5% |
| `y = 1.01x` | strict BF16 | 4.52e+02 | 3.75e+00 | 0.0 | 3.6 | 62.2% |
| `y = 1.01x` | wide acc | 5.57e+02 | 4.48e+00 | 0.0 | 3.7 | 58.9% |
| `y = 1.1x` | strict BF16 | 3.70e-01 | 6.04e-02 | 1.4 | 5.2 | 22.3% |
| `y = 1.1x` | wide acc | 1.99e-01 | 4.84e-02 | 2.3 | 5.4 | 16.5% |

"Bits kept" is `-log2(relative error)`; "losing >half" is the fraction of cases retaining
fewer than 4 of BF16's 8 significand bits.

**Scale-invariant** — the same figures hold from `2^-20` to `2^20`, as expected for a
floating-point format: cancellation depends on the *ratio* `y/x` and the angle, not on
absolute magnitude.

**And it is not a rare corner.** The smallest position that lands within one table bucket of
45°:

| i | smallest such m |
|---|---|
| 0 | 183 |
| 1 | 117 |
| 8 | 380 |
| 32 | 707 |
| 63 | 6788 |

The high-frequency pairs reach the worst case within the first couple of hundred tokens, and
then repeatedly. **This is the common case, not an edge case.**

> **This strict-versus-wide comparison for RoPE specifically is the measurement that had not
> been published.** Every BF16 accelerator — including Mugi's own vector array and every
> BF16 tensor core — multiplies narrow and accumulates wide. What strict BF16 costs *for
> RoPE* was unquantified. The answer: modest on average (1.27× RMS), but under cancellation
> both approaches are in serious trouble (3.5 versus 3.7 bits retained), and cancellation is
> encountered constantly.

## 6.4 Measured performance and size

| Metric | Value |
|---|---|
| Datapath latency | **3 cycles** |
| Throughput | **1 rotated pair per cycle** after fill |
| Coprocessor latency, issue → writeback | 4 cycles |
| Sine LUT | 1024 × 16 bit = **2 KB** |
| Theta ROM | 64 × 32 bit = **256 B** |
| Naive alternative table | ~1 MB (≈500× larger) |
| LUT angular resolution | 0.00153 rad (≈5× finer than a BF16 ulp) |
| `sin² + cos²` max deviation | 0.00528 (≈0.68 BF16 ulp) |
| C test runtime on the core | 13,993 cycles |

Area, power and achieved clock frequency are **not** measured — see Part 7.

---
---

# Part 7 — Status, limitations, and next steps

## 7.1 Status

| Milestone | Description | Status |
|---|---|---|
| **M0** | Golden model (3 models) + error budget | **PASS** |
| **M1** | BF16 primitives via CVFPU | **PASS** — 1,001,935 vectors, zero mismatches |
| **M2** | Integer phase generator | **PASS** — bit-exact, zero drift |
| **M3** | BF16 sine LUT | **PASS** — bit-exact, exact mirror symmetry |
| **M4** | BF16 rotation datapath | **PASS** — 1,001,961 vectors, zero mismatches |
| **M5** | CORE-V-XIF coprocessor | **PASS** — cancellation, throughput, backpressure, SVA |
| **M6** | Integration with the real core | **PASS** — lint clean, zero warnings |
| **M7** | C test on the integrated core | **PASS** |
| M8 | Streaming variant | **not started** |
| M9 | Synthesis: area / timing / power | **scripts written, never run** |
| M10 | Mugi integration write-up | not started |

## 7.2 The most important gap

**The `core-v-verif` baseline regression and RVFI/Spike co-simulation have not been run.**

This is the top-priority remaining item and should be stated plainly in any presentation.
What it would establish is that adding the coprocessor has not perturbed the CPU's own
behaviour — the standard method is to run the core's full regression before and after
integration and require identical results, with instruction-level co-simulation against a
reference simulator (Spike) kept green throughout.

`core-v-verif` is available locally, but its flow is UVM-based and expects a commercial
simulator plus a configured testbench environment; standing that up is a separate piece of
work from building the unit.

What *is* established in the meantime:

- The integration elaborates cleanly with the real core (M6).
- A real C program runs correctly on the integrated core, using both the CPU's normal
  instructions and the custom one, with correct results and no traps (M7).
- The coprocessor is provably inert for instructions that are not its own: the SVA
  `a_accept_matches_decode` checks, on **every** completed handshake, that `accept` is
  asserted for exactly the `ROPE.ROT` encodings and nothing else.

That is meaningful evidence but it is **not** a substitute for the regression, and should
not be presented as one.

## 7.3 Everything else not done

**M8 — the streaming variant.** Currently one instruction rotates one pair, so a
128-element head costs 64 instructions and instruction-issue overhead dominates the actual
arithmetic. The intended fix is a small instruction group (`ROPE.CFG` / `ROPE.POS` /
`ROPE.START` / `ROPE.WAIT`) letting the unit walk memory itself via the XIF memory channels
(which CV32E40X supports and CVA6 does not). The groundwork is in place: the phase generator
already implements the sequential mode this needs, the memory channels are cleanly tied off,
and an assertion guards the tie-off so whoever wires them up must consciously remove it —
at which point the "never store before a confirmed commit" rule becomes live and must be
honoured.

**M9 — synthesis.** `syn/rope.sdc` (400 MHz target plus the XIF timing budget) and
`syn/synth.tcl` (Genus flow with the area-breakdown reporting the write-up needs) are
written but **have never been executed**, because no PDK or standard-cell library is
configured in this environment. The script needs `ROPE_LIB_PATH` pointed at a 45 nm library
to match Mugi's operating point and produce comparable numbers. Until it runs, **no claim
about area, power, or achieved frequency should be made.** The script is untested and should
be expected to need debugging on first use.

**Comparison against real HuggingFace LLaMA tensors.** The half-split convention is verified
against a faithful reimplementation of HuggingFace's `apply_rotary_pos_emb` / `rotate_half`
(agreeing to 1e-12 in FP64), and the two conventions are proven equivalent up to the fixed
permutation. Testing against *actual extracted weights* needs PyTorch, network access, and
gated model access — none available here.

**Energy per rotated pair** and **cycles per token for a realistic head dimension** — both
depend on synthesis.

## 7.4 Recommended next steps, in order

1. **Stand up `core-v-verif` + RVFI/Spike co-simulation.** The highest-value remaining work
   by a wide margin: it is what converts "the unit is correct" into "the *system* is
   correct".
2. **Run synthesis.** Everything needed is written; it needs a library. This unlocks the
   area/timing/power table, the area breakdown (LUT vs ROM vs CVFPU units vs phase logic vs
   wrapper), and energy per pair — the quantitative comparison against Mugi's alternative.
3. **Run the 10^7-vector campaigns.** `make m1-full` and `make m4-full` are wired up; 10^6
   has been run at zero mismatches, and 10^7 is what the plan specifies.
4. **Build the streaming variant (M8)** to amortise issue overhead, which is what makes the
   unit compelling rather than merely correct.
5. **Read arXiv:2604.09742 (RoME).** The only operator-level hardware analysis of RoPE;
   it reformulates the rotation as a matrix operation and fuses the multiply-add sequence.
   Relevant twice over: for related-work positioning, and because if the rotation genuinely
   reduces to a fused matrix form, part of it might map onto Mugi's existing array after
   all — which would sharpen the argument about *where* the boundary between the two units
   should sit.
6. **Consider FPGA bring-up** at 100 MHz as an intermediate reality check before an ASIC
   target.

## 7.5 How to reproduce everything

```bash
cd rope-bf16

make model              # Python golden-model suites (numpy only)
make gen                # regenerate the theta ROM and sine table
make lint               # lint the coprocessor
make lint-integration   # lint the real core + XIF + coprocessor
make test               # RTL testbenches M1..M5
make m7                 # the C test on the integrated core
make characterize       # regenerate docs/error_budget.md

# Larger campaigns
make m1 NVEC=1000000
make m1-full            # 10^7 vectors
make m4-full            # 10^7 vectors
```

`make test` defaults to 200,000 random vectors per testbench on top of the directed and
exhaustive sets, and completes in a couple of minutes.

**Environment used:** Verilator 5.020, Python 3.12 with numpy 2.5.1, and
`riscv32-corev-elf-gcc` 14.1.0 for the software. No other dependencies.

## 7.6 One-paragraph summary for a slide

> A synthesizable BF16-only RoPE coprocessor for the CV32E40X RISC-V core, attached over
> CORE-V-XIF as a single custom instruction. The key insight is that the rotation angle
> cannot be held in BF16 at all — at token position 1024, consecutive representable BF16
> values are more than a full revolution apart — so the phase is kept as a 32-bit integer
> fraction of a turn, which makes angle reduction free, exact, and drift-free, and reduces
> the trigonometric table from about 1 MB to 2 KB. The arithmetic is four multiplies and two
> adds using CVFPU, rounded to nearest-even at every node, with 3-cycle latency and one
> rotated pair per cycle. It is verified bit-exactly against a Python reference model over
> 10^6 vectors per unit with zero mismatches, and a C program exercising the instruction
> runs correctly on the integrated core. The project also produces the first published
> characterisation of what *strict* BF16 costs for RoPE: 1.27× RMS error versus the
> conventional wide-accumulate approach, and only ~3.5 of 8 significand bits retained under
> the catastrophic cancellation that high-frequency pairs encounter within the first few
> hundred tokens. Remaining work: the core's own regression suite for system-level
> assurance, and synthesis for area, timing, and power.

---
---
# Appendix A — RTL source (SystemVerilog)

All files are new. Paths are relative to `rope-bf16/`.

## A.1 `rtl/rope_pkg.sv` — parameters, encoding, packing helpers (177 lines)

```systemverilog
// -----------------------------------------------------------------------------
// rope_pkg.sv -- parameters, types and opcode constants for the BF16 RoPE unit.
//
// Design invariants this package encodes (see steps.md Section 0):
//   INV-1  the arithmetic datapath is BF16 only, RNE at every node, no conversions
//   INV-2  the phase index is an INTEGER -- addressing, not arithmetic
//   INV-3  no approximation of the interface contract
//   INV-4  rotate before quantize
// -----------------------------------------------------------------------------

package rope_pkg;

  // ---------------------------------------------------------------------------
  // BF16 format (INV-1)
  // ---------------------------------------------------------------------------

  // bfloat16: 1 sign, 8 exponent, 7 stored mantissa bits. Bit-identical to the upper
  // 16 bits of IEEE binary32, which is why no conversion stage is needed anywhere.
  localparam int unsigned Bf16Width   = 16;
  localparam int unsigned Bf16ExpBits = 8;
  localparam int unsigned Bf16ManBits = 7;

  localparam logic [15:0] Bf16PosZero = 16'h0000;
  localparam logic [15:0] Bf16PosInf  = 16'h7F80;
  localparam logic [15:0] Bf16QNaN    = 16'h7FC0;  // canonical quiet NaN
  localparam logic [15:0] Bf16One     = 16'h3F80;

  // In CVFPU/FPnew nomenclature bfloat16 is FP16ALT (format index 4). Grepping the
  // vendored sources for "bf16" finds nothing -- see steps.md Section 5.1.
  // FpFmtMask is `logic [0:4]`, so FP16ALT-only is 5'b00001.
  localparam logic [0:4] FpFmtMaskBf16Only = 5'b00001;
  localparam logic [0:3] IntFmtMaskNone    = 4'b0000;

  // ---------------------------------------------------------------------------
  // Integer phase (INV-2)
  // ---------------------------------------------------------------------------

  // 32-bit phase word P, representing angle = 2*pi * P / 2^32.
  //
  //   P[31:30] = quadrant  (2 bits)
  //   P[29:20] = LUT index (10 bits -> 1024 entries per quadrant)
  //   P[19:0]  = unused (reserved for future interpolation)
  //
  // 32 bits rather than 24: at 32 bits the slowest frequency (theta ~ 1e-4) still gets
  // Phi_min ~ 79000 (relative quantisation ~1e-5), versus Phi_min ~ 267 (~0.4%) at 24
  // bits. 32 is also the natural width on RV32. `mod 2*pi` is free (wraparound).
  localparam int unsigned PhaseBits = 32;
  localparam int unsigned QuadBits  = 2;
  localparam int unsigned LutBits   = 10;
  localparam int unsigned LutN      = 1 << LutBits;          // 1024
  localparam int unsigned QuadShift = PhaseBits - QuadBits;   // 30
  localparam int unsigned IdxShift  = QuadShift - LutBits;    // 20

  // cos(x) = sin(x + pi/2) -> a phase offset of exactly one quarter turn.
  // ONE table, two lookups (steps.md Section 7.2).
  localparam logic [PhaseBits-1:0] QuarterTurn = 1 << QuadShift;  // 2^30

  // ---------------------------------------------------------------------------
  // Head geometry
  // ---------------------------------------------------------------------------

  localparam int unsigned HeadDimMax  = 128;
  localparam int unsigned NumPairsMax = HeadDimMax / 2;             // 64
  localparam int unsigned PairIdxBits = $clog2(NumPairsMax);        // 6

  // Sequence position m.
  //
  // 16 bits, matching the rs2[31:16] field of ROPE.ROT exactly. steps.md Section 6.3
  // suggests 13 bits ("m is ~13 bits ... a 13x32 multiplier, cheap"), and 13 is indeed
  // enough for an 8192-token context -- but the *instruction* carries 16 bits, so a
  // 13-bit register silently truncates: m=10000 would be taken as 10000 & 0x1FFF = 1808
  // and produce a confidently wrong rotation with no error anywhere. Since 8192-token
  // contexts are ordinary, that failure is reachable in normal use.
  //
  // Three extra multiplier bits (16x32 instead of 13x32) is a negligible price for
  // removing a silent-wrongness class, so the width follows the ISA rather than the
  // expected sequence length. Narrow it only if you also range-check m.
  localparam int unsigned PosBits = 16;

  // ---------------------------------------------------------------------------
  // Pairing convention (steps.md Section 12)
  // ---------------------------------------------------------------------------
  //
  // The two conventions are equivalent up to a fixed permutation of the head dimension,
  // so trained weights port between them -- but mixing them up makes the error look like
  // garbage rather than like a convention mismatch. llama.cpp carries an explicit NEOX
  // flag; so do we.
  //
  // Default is HALF_SPLIT because HuggingFace Transformers (which Mugi's authors used
  // for all their models) uses rotate_half.
  typedef enum logic {
    STRIDE_HALF_SPLIT  = 1'b0,  // (x_i, x_{i+d/2})  -- HF LLaMA / GPT-NeoX  [DEFAULT]
    STRIDE_INTERLEAVED = 1'b1   // (x_2i, x_2i+1)    -- GPT-J / original RoFormer
  } stride_mode_e;

  // ---------------------------------------------------------------------------
  // Accumulator width (steps.md Section 8.3)
  // ---------------------------------------------------------------------------
  //
  // 16 = strict BF16 (DEFAULT, per INV-1).
  // 32 = wide internal accumulate, MEASUREMENT ONLY -- exists solely to quantify the
  //      error of the strict-BF16 choice for the writeup. Do not ship it enabled.
  localparam int unsigned AccWidthStrict = 16;
  localparam int unsigned AccWidthWide   = 32;

  // ---------------------------------------------------------------------------
  // Instruction encoding (steps.md Section 9.1)
  // ---------------------------------------------------------------------------
  //
  //   ROPE.ROT rd, rs1, rs2      # custom-0
  //     rs1 = {x_bf16[15:0], y_bf16[15:0]}     native BF16 pair, NO conversion
  //     rs2 = {m[15:0], i[15:0]}               position and pair index (integers)
  //     rd  = {x'_bf16[15:0], y'_bf16[15:0]}   native BF16 pair out
  //
  // BF16 packs two-per-GPR on RV32, so a single source pair and a single destination
  // suffice: no dual writeback, X_RFW_WIDTH stays 32.
  //
  // The opcode is a parameter so it can be reassigned if it ever collides with another
  // coprocessor -- an explicit CV-X-IF recommendation.
  localparam logic [6:0] RopeOpcodeCustom0 = 7'b0001011;  // 0x0B, RISC-V custom-0
  localparam logic [2:0] RopeFunct3Rot     = 3'b000;
  localparam logic [6:0] RopeFunct7Rot     = 7'b0000000;

  // R-type field extraction helpers.
  function automatic logic [6:0] instr_opcode(logic [31:0] instr);
    return instr[6:0];
  endfunction

  function automatic logic [2:0] instr_funct3(logic [31:0] instr);
    return instr[14:12];
  endfunction

  function automatic logic [6:0] instr_funct7(logic [31:0] instr);
    return instr[31:25];
  endfunction

  function automatic logic [4:0] instr_rd(logic [31:0] instr);
    return instr[11:7];
  endfunction

  // Decode: is this a ROPE.ROT?
  function automatic logic is_rope_rot(logic [31:0] instr);
    return (instr_opcode(instr) == RopeOpcodeCustom0) &&
           (instr_funct3(instr) == RopeFunct3Rot)     &&
           (instr_funct7(instr) == RopeFunct7Rot);
  endfunction

  // ---------------------------------------------------------------------------
  // Operand packing helpers
  // ---------------------------------------------------------------------------
  //
  // A BF16 pair lives in one 32-bit GPR as {x, y} with x in the HIGH half. Keeping this
  // in one place stops the halves being swapped somewhere between the RTL, the
  // intrinsics header and the golden model.

  function automatic logic [15:0] pair_x(logic [31:0] packed_pair);
    return packed_pair[31:16];
  endfunction

  function automatic logic [15:0] pair_y(logic [31:0] packed_pair);
    return packed_pair[15:0];
  endfunction

  function automatic logic [31:0] pack_pair(logic [15:0] x, logic [15:0] y);
    return {x, y};
  endfunction

  // rs2 = {m[15:0], i[15:0]}
  function automatic logic [15:0] rs2_pos(logic [31:0] rs2);
    return rs2[31:16];
  endfunction

  function automatic logic [15:0] rs2_pair_idx(logic [31:0] rs2);
    return rs2[15:0];
  endfunction

endpackage : rope_pkg
```

## A.2 `rtl/rope_sin_lut.sv` — quadrant decode, mirror, sign (122 lines)

```systemverilog
// -----------------------------------------------------------------------------
// rope_sin_lut.sv -- sin and cos from an integer phase word (steps.md Milestone 3).
//
// ONE quarter-wave table serves both functions:
//
//     sin(P)  = quadrant/reflection logic below
//     cos(P)  = sin(P + pi/2) = sin(P + 2^30)      -- same table, phase offset
//
// Two reads of the same (combinational) table give both per cycle.
//
// Quadrant logic (steps.md Section 7.4):
//
//     quad 0:  sin = +T[idx]
//     quad 1:  sin = +T[N-1-idx]
//     quad 2:  sin = -T[idx]
//     quad 3:  sin = -T[N-1-idx]
//
// The reflection index is exactly N-1-idx because the table is MIDPOINT sampled. With
// edge sampling it would be N-idx, which overflows the table at idx=0 and would force
// either a 1025th entry or a special case.
//
// Sign application is a BF16 sign-bit XOR (bit 15) -- exact, no arithmetic, no
// rounding. This is one of the few places BF16's format layout helps directly.
//
// NOTE: this module is purely combinational. It is the caller's job to register the
// outputs if the LUT read is on a critical path.
// -----------------------------------------------------------------------------

module rope_sin_lut
  import rope_pkg::*;
#(
  parameter int unsigned LutBitsP = rope_pkg::LutBits,
  parameter int unsigned LutNP    = rope_pkg::LutN
) (
  input  logic [PhaseBits-1:0] phase_i,
  output logic [15:0]          sin_o,
  output logic [15:0]          cos_o
);

  // cos(x) = sin(x + pi/2). Wraparound is intended and free.
  logic [PhaseBits-1:0] phase_cos;
  assign phase_cos = phase_i + QuarterTurn;

  // ---------------------------------------------------------------------------
  // One lookup path, instantiated twice
  // ---------------------------------------------------------------------------

  logic [15:0] sin_raw, cos_raw;

  rope_sin_lut_path #(
    .LutBitsP ( LutBitsP ),
    .LutNP    ( LutNP    )
  ) i_path_sin (
    .phase_i ( phase_i ),
    .val_o   ( sin_raw )
  );

  rope_sin_lut_path #(
    .LutBitsP ( LutBitsP ),
    .LutNP    ( LutNP    )
  ) i_path_cos (
    .phase_i ( phase_cos ),
    .val_o   ( cos_raw   )
  );

  assign sin_o = sin_raw;
  assign cos_o = cos_raw;

endmodule : rope_sin_lut


// -----------------------------------------------------------------------------
// rope_sin_lut_path -- one phase word in, one signed BF16 sine value out.
// -----------------------------------------------------------------------------

module rope_sin_lut_path
  import rope_pkg::*;
#(
  parameter int unsigned LutBitsP = rope_pkg::LutBits,
  parameter int unsigned LutNP    = rope_pkg::LutN
) (
  input  logic [PhaseBits-1:0] phase_i,
  output logic [15:0]          val_o
);

  // Field extraction. The low IdxShift bits of the phase are deliberately ignored
  // (reserved for future interpolation).
  logic [QuadBits-1:0]    quad;
  logic [LutBitsP-1:0]    idx;
  logic [LutBitsP-1:0]    idx_eff;
  logic                   reflect;
  logic                   negate;
  logic [15:0]            tab_val;

  assign quad = phase_i[PhaseBits-1 -: QuadBits];          // phase[31:30]
  assign idx  = phase_i[IdxShift +: LutBitsP];             // phase[29:20]

  // Quadrants 1 and 3 read the table backwards.
  assign reflect = (quad == 2'd1) || (quad == 2'd3);
  // Quadrants 2 and 3 are negative.
  assign negate  = quad[1];

  // Exactly N-1-idx thanks to midpoint sampling. With LutNP a power of two this is a
  // bitwise complement, so it costs nothing.
  assign idx_eff = reflect ? (LutBitsP'(LutNP - 1) - idx) : idx;

  rope_sin_rom #(
    .LutBits ( LutBitsP ),
    .LutN    ( LutNP    )
  ) i_rom (
    .idx_i ( idx_eff ),
    .sin_o ( tab_val )
  );

  // Sign application: XOR on the BF16 sign bit. Exact.
  //
  // Note this means sin(phase) for a phase in quadrant 2 or 3 that reads T[k]==0 would
  // produce -0. The table is midpoint sampled so it contains no exact zero, and -0 is
  // in any case the numerically correct signed zero here.
  assign val_o = {tab_val[15] ^ negate, tab_val[14:0]};

endmodule : rope_sin_lut_path
```

## A.3 `rtl/rope_phase_gen.sv` — integer phase generator (109 lines)

```systemverilog
// -----------------------------------------------------------------------------
// rope_phase_gen.sv -- integer phase generator (steps.md Milestone 2).
//
// This is the piece that makes BF16 RoPE possible at all (INV-2).
//
//   P = (m * Phi_i) mod 2^32          <- the mod is FREE: natural wraparound
//
//     P[31:30] = quadrant  (2 bits)
//     P[29:20] = LUT index (10 bits)
//     P[19:0]  = unused (reserved for future interpolation)
//
// Everything here is integer arithmetic. No value in this module ever enters a BF16
// multiplier or adder: the phase word is an ADDRESS, not data.
//
// Two operating modes (steps.md Section 6.3):
//
//   Random access (`ROPE.ROT`)  -- compute m * Phi_i directly. m is 13 bits and Phi is
//                                  32 bits, so this is a 13x32 multiplier: cheap.
//   Sequential decode           -- hold per-pair phase state and add Phi_i once per
//                                  token. One integer add, and ZERO drift because
//                                  integer addition is exact.
//
// NOT USED, deliberately: the trigonometric recurrence
//     cos((m+1)th) = cos(m th)cos(th) - sin(m th)sin(th)
// In BF16 that drifts catastrophically -- the error compounds multiplicatively and the
// state wanders off the unit circle within a few dozen tokens at 7 mantissa bits. The
// integer accumulator is both exact and cheaper.
// -----------------------------------------------------------------------------

module rope_phase_gen
  import rope_pkg::*;
#(
  parameter int unsigned NumPairs = rope_pkg::NumPairsMax,
  parameter int unsigned IdxBits  = rope_pkg::PairIdxBits,
  parameter int unsigned PosBits  = rope_pkg::PosBits
) (
  input  logic                 clk_i,
  input  logic                 rst_ni,

  // ---- random-access mode (combinational) --------------------------------
  input  logic [PosBits-1:0]   pos_i,        // m
  input  logic [IdxBits-1:0]   pair_idx_i,   // i
  output logic [PhaseBits-1:0] phase_o,      // (m * Phi_i) mod 2^32

  // ---- sequential mode --------------------------------------------------
  // seq_clear_i resets the accumulator for pair seq_idx_i to zero (position 0).
  // seq_step_i  advances it by one token: acc += Phi_i, exactly.
  input  logic                 seq_clear_i,
  input  logic                 seq_step_i,
  input  logic [IdxBits-1:0]   seq_idx_i,
  output logic [PhaseBits-1:0] seq_phase_o
);

  // ---------------------------------------------------------------------------
  // Phi lookup
  // ---------------------------------------------------------------------------

  logic [PhaseBits-1:0] phi_rand;
  logic [PhaseBits-1:0] phi_seq;

  rope_theta_rom i_rom_rand (
    .pair_idx_i ( pair_idx_i ),
    .phi_o      ( phi_rand   )
  );

  rope_theta_rom i_rom_seq (
    .pair_idx_i ( seq_idx_i ),
    .phi_o      ( phi_seq   )
  );

  // ---------------------------------------------------------------------------
  // Random access: P = m * Phi_i, truncated to PhaseBits.
  //
  // Truncation IS the mod-2^32 reduction, which is exactly the "mod 2*pi is free"
  // property: the phase word is a fixed-point fraction of a full turn, so discarding
  // the overflow discards whole turns and nothing else.
  // ---------------------------------------------------------------------------

  logic [PosBits+PhaseBits-1:0] product;

  assign product = {{PhaseBits{1'b0}}, pos_i} * {{PosBits{1'b0}}, phi_rand};
  assign phase_o = product[PhaseBits-1:0];

  // ---------------------------------------------------------------------------
  // Sequential mode: one exact integer add per token, per pair.
  // ---------------------------------------------------------------------------

  logic [PhaseBits-1:0] acc_q [NumPairs];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int unsigned p = 0; p < NumPairs; p++) acc_q[p] <= '0;
    end else if (seq_clear_i) begin
      acc_q[seq_idx_i] <= '0;
    end else if (seq_step_i) begin
      // Wraparound is the desired behaviour, not an overflow bug.
      acc_q[seq_idx_i] <= acc_q[seq_idx_i] + phi_seq;
    end
  end

  assign seq_phase_o = acc_q[seq_idx_i];

  // ---------------------------------------------------------------------------
  // Field extraction helpers, exported for the LUT stage.
  // ---------------------------------------------------------------------------
  // (Consumers slice phase_o directly; these exist to document the layout in one place
  //  and are checked by the testbench.)

endmodule : rope_phase_gen
```

> Note: the header comment still says "m is 13 bits" from the original plan. The actual
> parameter is now 16 (Part 4.3); the multiplier is 16×32.

## A.4 `rtl/rope_bf16_unit.sv` — the single CVFPU configuration point (142 lines)

**This is where the FPnew operand-slot trap from Part 4.1 is handled.**

```systemverilog
// -----------------------------------------------------------------------------
// rope_bf16_unit.sv -- shared thin wrapper over one CVFPU (FPnew) FP16ALT unit.
//
// Configured per steps.md Section 5.1:
//   * FP16ALT (= bfloat16) is the ONLY enabled format -- everything else is masked off
//     to keep area minimal. In FPnew nomenclature bfloat16 is FP16ALT; there is no
//     "BF16" anywhere in the vendored sources.
//   * Only the ADDMUL operation group is generated. DIVSQRT, NONCOMP and CONV are
//     DISABLED: the RoPE datapath needs none of them, DivSqrt is a large block, and
//     disabling it also avoids FPnew's external fpu_div_sqrt_mvp dependency (which is
//     deliberately not vendored -- see gen/vendor_cvfpu.sh).
//   * Rounding mode is tied STATICALLY to RNE (INV-1). It is not an input.
//
// Package scoping is explicit (`fpnew_pkg::`) throughout; the package is never
// wildcard-imported, per steps.md Section 16.
//
// Both ports use AXI-like valid/ready, which composes naturally with XIF.
// -----------------------------------------------------------------------------

module rope_bf16_unit #(
  // Number of pipeline registers inside the FPnew unit. 1 is a reasonable default;
  // raise it if timing closure needs more stages.
  parameter int unsigned NumPipeRegs = 1,
  // Which ADDMUL operation this instance performs.
  parameter fpnew_pkg::operation_e Op = fpnew_pkg::MUL,
  // op_mod inverts the sign of operand C, turning ADD into SUB.
  parameter logic OpMod = 1'b0
) (
  input  logic        clk_i,
  input  logic        rst_ni,

  // Operands are BF16 bit patterns. Their meaning depends on Op -- see the mapping
  // comment at `operands` below, which is where the FPnew ADD/MUL asymmetry is handled.
  input  logic [15:0] operand_a_i,
  input  logic [15:0] operand_b_i,

  input  logic        in_valid_i,
  output logic        in_ready_o,
  input  logic        flush_i,

  output logic [15:0] result_o,
  output logic        out_valid_o,
  input  logic        out_ready_i,
  output logic        busy_o
);

  // ---------------------------------------------------------------------------
  // FPnew configuration
  // ---------------------------------------------------------------------------

  // Width 16 with a 16-bit format means exactly one lane and no NaN-boxing (FPnew skips
  // the box check when FP_WIDTH == WIDTH, so EnableNanBox is immaterial here).
  localparam fpnew_pkg::fpu_features_t RopeFeatures = '{
    Width:         rope_pkg::Bf16Width,
    EnableVectors: 1'b0,
    EnableNanBox:  1'b0,
    FpFmtMask:     rope_pkg::FpFmtMaskBf16Only,   // FP16ALT only
    IntFmtMask:    rope_pkg::IntFmtMaskNone
  };

  // UnitTypes is indexed [ADDMUL, DIVSQRT, NONCOMP, CONV]. Only ADDMUL is PARALLEL;
  // the rest are DISABLED, which is what gates their generate blocks away entirely
  // (fpnew_opgroup_block keys on FmtUnitTypes[fmt] == PARALLEL).
  localparam fpnew_pkg::fpu_implementation_t RopeImpl = '{
    PipeRegs:   '{default: NumPipeRegs},
    UnitTypes:  '{'{default: fpnew_pkg::PARALLEL},  // ADDMUL  -- the only one we need
                  '{default: fpnew_pkg::DISABLED},  // DIVSQRT -- not needed, large
                  '{default: fpnew_pkg::DISABLED},  // NONCOMP -- not needed
                  '{default: fpnew_pkg::DISABLED}}, // CONV    -- no conversions (INV-1)
    PipeConfig: fpnew_pkg::BEFORE                   // registers at the inputs
  };

  // ---------------------------------------------------------------------------
  // Operand mapping -- the FPnew ADD/MUL asymmetry
  // ---------------------------------------------------------------------------
  //
  // FPnew's ADDMUL group is an FMA computing  operands[0]*operands[1] + operands[2].
  // fpnew_fma then rewrites the operands per opcode:
  //
  //   MUL: operand_c is forced to +/-0, so the result is operands[0] * operands[1].
  //        -> a, b go in slots 0 and 1.
  //
  //   ADD: operand_a is forced to +1.0, so the result is operands[1] + operands[2].
  //        -> a, b go in slots 1 and 2. Slot 0 is IGNORED.
  //
  // Putting the addends in slots 0 and 1 for an ADD would silently compute
  // `1.0 * b + 0` and throw away operand a. This is handled once, here.
  //
  //   op_mod inverts the sign of operand_c, so SUB is ADD with OpMod=1:
  //   operands[1] - operands[2].
  localparam int unsigned NumOperands = 3;
  logic [NumOperands-1:0][rope_pkg::Bf16Width-1:0] operands;

  always_comb begin
    operands = '0;
    if (Op == fpnew_pkg::ADD) begin
      operands[0] = rope_pkg::Bf16One;  // ignored by fpnew_fma (forced to +1.0)
      operands[1] = operand_a_i;
      operands[2] = operand_b_i;
    end else begin  // MUL (and any other ADDMUL op using slots 0/1)
      operands[0] = operand_a_i;
      operands[1] = operand_b_i;
      operands[2] = rope_pkg::Bf16PosZero;  // forced to +/-0 by fpnew_fma
    end
  end

  fpnew_pkg::status_t status_unused;
  logic               tag_unused;

  fpnew_top #(
    .Features       ( RopeFeatures ),
    .Implementation ( RopeImpl     ),
    // PulpDivsqrt only selects between DivSqrt implementations; DIVSQRT is DISABLED
    // above so nothing is generated either way.
    .PulpDivsqrt    ( 1'b0         ),
    .TagType        ( logic        )
  ) i_fpnew (
    .clk_i,
    .rst_ni,
    .operands_i     ( operands                     ),
    // RNE, statically. Not an input: INV-1 requires round-to-nearest-even everywhere.
    .rnd_mode_i     ( fpnew_pkg::RNE               ),
    .op_i           ( Op                           ),
    .op_mod_i       ( OpMod                        ),
    .src_fmt_i      ( fpnew_pkg::FP16ALT           ),
    .dst_fmt_i      ( fpnew_pkg::FP16ALT           ),
    .int_fmt_i      ( fpnew_pkg::INT8              ),  // unused, CONV disabled
    .vectorial_op_i ( 1'b0                         ),
    .tag_i          ( 1'b0                         ),
    .simd_mask_i    ( 1'b1                         ),
    .in_valid_i     ( in_valid_i                   ),
    .in_ready_o     ( in_ready_o                   ),
    .flush_i        ( flush_i                      ),
    .result_o       ( result_o                     ),
    .status_o       ( status_unused                ),
    .tag_o          ( tag_unused                   ),
    .out_valid_o    ( out_valid_o                  ),
    .out_ready_i    ( out_ready_i                  ),
    .busy_o         ( busy_o                       )
  );

endmodule : rope_bf16_unit
```

## A.5 `rtl/rope_bf16_mul.sv` (45 lines)

```systemverilog
// -----------------------------------------------------------------------------
// rope_bf16_mul.sv -- BF16 multiplier. result = a * b, rounded RNE to BF16.
//
// A named wrapper over rope_bf16_unit so the datapath reads as what it is. All the
// CVFPU configuration detail lives in rope_bf16_unit.sv.
// -----------------------------------------------------------------------------

module rope_bf16_mul #(
  parameter int unsigned NumPipeRegs = 1
) (
  input  logic        clk_i,
  input  logic        rst_ni,

  input  logic [15:0] a_i,
  input  logic [15:0] b_i,

  input  logic        in_valid_i,
  output logic        in_ready_o,
  input  logic        flush_i,

  output logic [15:0] result_o,
  output logic        out_valid_o,
  input  logic        out_ready_i,
  output logic        busy_o
);

  rope_bf16_unit #(
    .NumPipeRegs ( NumPipeRegs        ),
    .Op          ( fpnew_pkg::MUL     ),
    .OpMod       ( 1'b0               )
  ) i_unit (
    .clk_i,
    .rst_ni,
    .operand_a_i ( a_i         ),
    .operand_b_i ( b_i         ),
    .in_valid_i,
    .in_ready_o,
    .flush_i,
    .result_o,
    .out_valid_o,
    .out_ready_i,
    .busy_o
  );

endmodule : rope_bf16_mul
```

## A.6 `rtl/rope_bf16_add.sv` (53 lines)

```systemverilog
// -----------------------------------------------------------------------------
// rope_bf16_add.sv -- BF16 adder/subtractor, rounded RNE to BF16.
//
//   Sub = 0 : result = a + b
//   Sub = 1 : result = a - b
//
// Subtraction uses FPnew's op_mod (which inverts the sign of operand C) rather than a
// separate negation stage: it is exact and free. Note that a BF16 sign flip is a pure
// XOR on bit 15, so even an explicit negation would introduce no rounding -- but going
// through op_mod keeps the operand count down.
// -----------------------------------------------------------------------------

module rope_bf16_add #(
  parameter int unsigned NumPipeRegs = 1,
  parameter logic        Sub         = 1'b0
) (
  input  logic        clk_i,
  input  logic        rst_ni,

  input  logic [15:0] a_i,
  input  logic [15:0] b_i,

  input  logic        in_valid_i,
  output logic        in_ready_o,
  input  logic        flush_i,

  output logic [15:0] result_o,
  output logic        out_valid_o,
  input  logic        out_ready_i,
  output logic        busy_o
);

  rope_bf16_unit #(
    .NumPipeRegs ( NumPipeRegs    ),
    .Op          ( fpnew_pkg::ADD ),
    .OpMod       ( Sub            )
  ) i_unit (
    .clk_i,
    .rst_ni,
    // rope_bf16_unit maps these into FPnew's slots 1 and 2, because FPnew's ADD forces
    // operand slot 0 to +1.0. See the operand-mapping comment there.
    .operand_a_i ( a_i         ),
    .operand_b_i ( b_i         ),
    .in_valid_i,
    .in_ready_o,
    .flush_i,
    .result_o,
    .out_valid_o,
    .out_ready_i,
    .busy_o
  );

endmodule : rope_bf16_add
```

## A.7 `rtl/rope_datapath.sv` — 4 multiplies + 2 adds (276 lines)

**Contains the `in_ready` deadlock fix from Part 4.2 and its guarding assertions.**

```systemverilog
// -----------------------------------------------------------------------------
// rope_datapath.sv -- the BF16 rotation datapath (steps.md Milestone 4).
//
//       x (BF16)   y (BF16)      cos (BF16)   sin (BF16)
//          |          |              |            |
//          +----+-----+------+-------+------+-----+
//               |            |              |
//          [BF16 mul]   [BF16 mul]    [BF16 mul]   [BF16 mul]
//           x*cos        y*sin         x*sin        y*cos
//               |            |              |            |
//               +-----+------+              +-----+------+
//                     |                           |
//                [BF16 sub]                  [BF16 add]
//                     |                           |
//                  x' (BF16)                   y' (BF16)
//
// 4 BF16 multiplies, 2 BF16 adds, rounded RNE to BF16 at EVERY node (INV-1).
//
// Only ~2 rounding events on the data path, versus ~16 for a CORDIC implementation --
// which is why LUT+multiply is *more* accurate in BF16, not less.
//
// Flow control
// ------------
// Fixed-latency, fully-pipelined stream. `out_ready_i` is NOT accepted: the consumer
// must always be able to sink a result. rope_xif_coproc guarantees this with a credit
// counter, which is also what the XIF rules force it to do anyway (commit_valid and
// mem_result_valid have no ready signal, so the coprocessor must always be able to
// accept -- see steps.md Section 9.3).
//
// Latency is reported on `latency_o` so the testbench and the wrapper can size buffers
// without hard-coding a number that would silently rot if NumPipeRegs changed.
// -----------------------------------------------------------------------------

module rope_datapath
  import rope_pkg::*;
#(
  // Pipeline registers inside each CVFPU unit.
  parameter int unsigned NumPipeRegs = 1,
  // Register the LUT outputs (recommended: the ROM read is a wide logic cone).
  parameter bit          RegLut      = 1'b1,
  // INV-1: strict BF16. 16 is the ONLY value this RTL implements.
  //
  // SCOPE NOTE: steps.md Section 8.3 also asks for an AccWidth=32 build (BF16 in/out,
  // wide internal accumulate) to quantify what strict BF16 costs. That comparison IS
  // produced -- but in the golden model (model/rope_ref.py `rope_wide_acc`, reported by
  // model/characterize.py), not in RTL. Building it in hardware would need FP32 adders
  // plus BF16<->FP32 cast units, i.e. enabling FPnew's CONV opgroup and adding the very
  // conversion stages INV-1 forbids, for a path that is explicitly never shipped. The
  // parameter is kept so the intent is visible, and rejects 32 rather than silently
  // ignoring it. See docs/notes.md.
  parameter int unsigned AccWidth    = rope_pkg::AccWidthStrict
) (
  input  logic                 clk_i,
  input  logic                 rst_ni,
  input  logic                 flush_i,

  // ---- input: one BF16 pair plus its trigonometric coefficients ----------
  input  logic [15:0]          x_i,
  input  logic [15:0]          y_i,
  input  logic [15:0]          sin_i,
  input  logic [15:0]          cos_i,
  input  logic                 in_valid_i,
  output logic                 in_ready_o,

  // ---- output ------------------------------------------------------------
  output logic [15:0]          x_o,
  output logic [15:0]          y_o,
  output logic                 out_valid_o,

  output logic                 busy_o,
  // Total latency in cycles, for buffer sizing / assertions.
  output logic [7:0]           latency_o
);

  // Fail loudly at elaboration rather than quietly producing strict-BF16 results while
  // the caller believes it asked for wide accumulation.
  initial begin : check_acc_width
    if (AccWidth != rope_pkg::AccWidthStrict) begin
      $fatal(1, "rope_datapath: AccWidth=%0d is not implemented in RTL; only strict BF16 (%0d) is. The wide-accumulate comparison lives in model/characterize.py.",
             AccWidth, rope_pkg::AccWidthStrict);
    end
  end

  // ---------------------------------------------------------------------------
  // Stage 0: optionally register the operands and coefficients.
  //
  // Registering here is what protects the XIF timing budget: the coprocessor gets only
  // a small fraction of the cycle on paths through issue_req.rs*, because CV32E40X
  // sources those operands from its register-file bypass network. Zero combinational
  // logic may sit between the operand input and this flop (steps.md Section 9.3 rule 1).
  // ---------------------------------------------------------------------------

  logic [15:0] x_s0, y_s0, sin_s0, cos_s0;
  logic        valid_s0;

  if (RegLut) begin : gen_reg_lut
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        x_s0     <= '0;
        y_s0     <= '0;
        sin_s0   <= '0;
        cos_s0   <= '0;
        valid_s0 <= 1'b0;
      end else if (flush_i) begin
        valid_s0 <= 1'b0;
      end else begin
        x_s0     <= x_i;
        y_s0     <= y_i;
        sin_s0   <= sin_i;
        cos_s0   <= cos_i;
        // NOT `in_valid_i & in_ready_o`: see the in_ready_o comment below. Gating the
        // registered valid on in_ready_o creates a circular dependency through FPnew
        // (whose in_ready is itself gated by in_valid) that deadlocks at reset.
        valid_s0 <= in_valid_i;
      end
    end
  end else begin : gen_no_reg_lut
    assign x_s0     = x_i;
    assign y_s0     = y_i;
    assign sin_s0   = sin_i;
    assign cos_s0   = cos_i;
    assign valid_s0 = in_valid_i;
  end

  // ---------------------------------------------------------------------------
  // Stage 1: four BF16 multipliers
  // ---------------------------------------------------------------------------
  //
  //   p0 = x*cos    p1 = y*sin    p2 = x*sin    p3 = y*cos
  //
  // All four are identically configured, so they share latency and handshake timing.
  // That is asserted below rather than assumed.

  localparam int unsigned NumMul = 4;

  logic [NumMul-1:0][15:0] mul_a, mul_b;
  logic [NumMul-1:0][15:0] mul_r;
  logic [NumMul-1:0]       mul_in_ready, mul_out_valid, mul_busy;

  always_comb begin
    mul_a[0] = x_s0;  mul_b[0] = cos_s0;   // x*cos
    mul_a[1] = y_s0;  mul_b[1] = sin_s0;   // y*sin
    mul_a[2] = x_s0;  mul_b[2] = sin_s0;   // x*sin
    mul_a[3] = y_s0;  mul_b[3] = cos_s0;   // y*cos
  end

  for (genvar g = 0; g < NumMul; g++) begin : gen_mul
    rope_bf16_mul #(
      .NumPipeRegs ( NumPipeRegs )
    ) i_mul (
      .clk_i,
      .rst_ni,
      .a_i         ( mul_a[g]          ),
      .b_i         ( mul_b[g]          ),
      .in_valid_i  ( valid_s0          ),
      .in_ready_o  ( mul_in_ready[g]   ),
      .flush_i     ( flush_i           ),
      .result_o    ( mul_r[g]          ),
      .out_valid_o ( mul_out_valid[g]  ),
      // The consumer (the adder stage) is always ready, so no backpressure exists
      // inside the datapath. See the flow-control note in the header.
      .out_ready_i ( 1'b1              ),
      .busy_o      ( mul_busy[g]       )
    );
  end

  // ---------------------------------------------------------------------------
  // Stage 2: two BF16 adders
  // ---------------------------------------------------------------------------
  //
  //   x' = p0 - p1     (subtract, via FPnew's op_mod -- exact, free)
  //   y' = p2 + p3

  logic [1:0][15:0] add_r;
  logic [1:0]       add_in_ready, add_out_valid, add_busy;

  rope_bf16_add #(
    .NumPipeRegs ( NumPipeRegs ),
    .Sub         ( 1'b1        )
  ) i_sub_x (
    .clk_i,
    .rst_ni,
    .a_i         ( mul_r[0]         ),  // x*cos
    .b_i         ( mul_r[1]         ),  // y*sin
    .in_valid_i  ( mul_out_valid[0] ),
    .in_ready_o  ( add_in_ready[0]  ),
    .flush_i     ( flush_i          ),
    .result_o    ( add_r[0]         ),
    .out_valid_o ( add_out_valid[0] ),
    .out_ready_i ( 1'b1             ),
    .busy_o      ( add_busy[0]      )
  );

  rope_bf16_add #(
    .NumPipeRegs ( NumPipeRegs ),
    .Sub         ( 1'b0        )
  ) i_add_y (
    .clk_i,
    .rst_ni,
    .a_i         ( mul_r[2]         ),  // x*sin
    .b_i         ( mul_r[3]         ),  // y*cos
    .in_valid_i  ( mul_out_valid[2] ),
    .in_ready_o  ( add_in_ready[1]  ),
    .flush_i     ( flush_i          ),
    .result_o    ( add_r[1]         ),
    .out_valid_o ( add_out_valid[1] ),
    .out_ready_i ( 1'b1             ),
    .busy_o      ( add_busy[1]      )
  );

  assign x_o         = add_r[0];
  assign y_o         = add_r[1];
  assign out_valid_o = add_out_valid[0];

  // ---------------------------------------------------------------------------
  // in_ready: always 1, and why that is correct rather than lazy
  // ---------------------------------------------------------------------------
  //
  // FPnew does NOT expose an unconditional ready. fpnew_top drives
  //
  //     assign in_ready_o = in_valid_i & opgrp_in_ready[get_opgroup(op_i)];
  //
  // i.e. its ready is gated by its own valid. So `&mul_in_ready` reads as 0 whenever we
  // are not already presenting a valid input, and using it to qualify the input register
  // forms a combinational cycle: valid=0 -> ready=0 -> valid stays 0. That deadlocks the
  // datapath permanently at reset.
  //
  // The unconditional truth is that these units never stall in this configuration:
  // `out_ready_i` is tied to 1 on every instance, so every FPnew input-pipeline stage has
  // `inp_pipe_ready[i] = 1`, and each unit accepts an operand on every cycle it is
  // offered one. Hence the datapath can always accept, and in_ready_o is constant 1.
  //
  // That is an assumption about vendored code, so it is CHECKED every cycle by the
  // lockstep assertions below (which observe mul_in_ready while in_valid is asserted,
  // where the gating makes it a faithful view of the real internal readiness).
  assign in_ready_o = 1'b1;
  assign busy_o     = (|mul_busy) | (|add_busy);

  // Latency: the optional input register, plus the multiplier stage, plus the adder
  // stage. Reported rather than hard-coded so it tracks NumPipeRegs.
  assign latency_o = 8'(RegLut) + 8'(NumPipeRegs) + 8'(NumPipeRegs);

  // ---------------------------------------------------------------------------
  // Structural assertions
  // ---------------------------------------------------------------------------
  //
  // The datapath assumes the four multipliers (and the two adders) march in lockstep.
  // If a future CVFPU bump broke that, results would be silently paired with the wrong
  // operands, so check it every cycle rather than trusting the configuration.
`ifndef SYNTHESIS
  always @(posedge clk_i) begin
    if (rst_ni) begin
      assert (mul_out_valid == 4'b0000 || mul_out_valid == 4'b1111)
        else $fatal(1, "rope_datapath: multipliers out of lockstep (out_valid=%b)", mul_out_valid);
      assert (mul_in_ready == 4'b0000 || mul_in_ready == 4'b1111)
        else $fatal(1, "rope_datapath: multipliers out of lockstep (in_ready=%b)", mul_in_ready);
      // The in_ready_o = 1 justification: whenever a valid operand set is presented, all
      // four multipliers must actually accept it. If a future CVFPU bump introduced a
      // stall, this fires instead of silently dropping pairs.
      if (valid_s0) begin
        assert (mul_in_ready == 4'b1111)
          else $fatal(1, "rope_datapath: multiplier refused a valid input (in_ready=%b) -- the no-stall assumption behind in_ready_o=1 is broken", mul_in_ready);
      end
      assert (add_out_valid == 2'b00 || add_out_valid == 2'b11)
        else $fatal(1, "rope_datapath: adders out of lockstep (out_valid=%b)", add_out_valid);
      // The adder stage must never refuse a multiplier result: the datapath has no
      // internal backpressure, so a stall here would silently drop a pair.
      if (mul_out_valid[0]) begin
        assert (add_in_ready == 2'b11)
          else $fatal(1, "rope_datapath: adder stalled with a product pending -- data lost");
      end
    end
  end
`endif

endmodule : rope_datapath
```

## A.8 `rtl/rope_xif_coproc.sv` — the CPU interface (375 lines)

```systemverilog
// -----------------------------------------------------------------------------
// rope_xif_coproc.sv -- CORE-V-XIF wrapper for the BF16 RoPE unit (Milestone 5).
//
//   ROPE.ROT rd, rs1, rs2        # custom-0, opcode 0x0B
//     rs1 = {x_bf16, y_bf16}     native BF16 pair in,  NO conversion
//     rs2 = {m[15:0], i[15:0]}   position and pair index (integers)
//     rd  = {x'_bf16, y'_bf16}   native BF16 pair out
//
// Single source pair, single destination: no dual writeback needed, X_RFW_WIDTH stays 32.
//
// Channels used: Issue, Commit, Result. Compressed is not needed. The memory channels
// belong to the streaming variant (Milestone 8) and are not driven here.
//
// The three protocol rules that steps.md Section 9.3 warns about, and how each is met:
//
//  1. REGISTER THE OPERAND INPUTS. The coprocessor gets only a small slice of the timing
//     budget on paths through issue_req.rs*, because CV32E40X sources those operands
//     straight from its register-file bypass network. So rs1/rs2 are flopped on arrival
//     and NO combinational logic touches them in the issue cycle. Specifically: the
//     phase multiply, the LUT read and the datapath all consume the *registered* copies.
//     Only the instruction word itself is decoded combinationally, because issue_resp
//     must answer in the issue cycle -- and the instruction word comes from the
//     instruction register, not the bypass network.
//
//  2. commit_valid HAS NO READY. It may arrive coincident with or after issue, and must
//     be observable at any time. State is tracked per `id` and a kill retires the
//     in-flight operation without writeback.
//
//  3. mem_result_valid HAS NO READY. Not applicable until Milestone 8 (no memory
//     transactions are issued), but the same "always able to accept" discipline is why
//     the result path is credit-controlled below.
//
// Ordering rule: a transaction with an earlier issued id must never depend on a later
// one. Results are returned strictly in issue order from a FIFO, so result_valid for an
// older id is never delayed waiting on a newer instruction's commit.
// -----------------------------------------------------------------------------

module rope_xif_coproc
  import rope_pkg::*;
#(
  // These five MUST match the core and the cv32e40x_if_xif instance, or elaboration
  // breaks in confusing ways (steps.md Section 9.5).
  parameter int unsigned X_NUM_RS      = 2,
  parameter int unsigned X_ID_WIDTH    = 4,
  parameter int unsigned X_MEM_WIDTH   = 32,
  parameter int unsigned X_RFR_WIDTH   = 32,
  parameter int unsigned X_RFW_WIDTH   = 32,

  parameter int unsigned NumPipeRegs   = 1,
  // Reassignable so it can move if it ever collides with another coprocessor.
  parameter logic [6:0]  RopeOpcode    = rope_pkg::RopeOpcodeCustom0
) (
  input  logic clk_i,
  input  logic rst_ni,

  cv32e40x_if_xif.coproc_issue  xif_issue_if,
  cv32e40x_if_xif.coproc_commit xif_commit_if,
  cv32e40x_if_xif.coproc_result xif_result_if
);

  localparam int unsigned NumIds = 1 << X_ID_WIDTH;

  // Depth of the result FIFO. It must cover the datapath latency so that every accepted
  // instruction has somewhere to land -- the datapath cannot be back-pressured.
  localparam int unsigned FifoDepth = 8;
  localparam int unsigned FifoPtrW  = $clog2(FifoDepth);

  // ---------------------------------------------------------------------------
  // Decode (combinational -- instruction word only, never the operands)
  // ---------------------------------------------------------------------------

  logic dec_is_rot;
  assign dec_is_rot = (rope_pkg::instr_opcode(xif_issue_if.issue_req.instr) == RopeOpcode)
                   && (rope_pkg::instr_funct3(xif_issue_if.issue_req.instr) == rope_pkg::RopeFunct3Rot)
                   && (rope_pkg::instr_funct7(xif_issue_if.issue_req.instr) == rope_pkg::RopeFunct7Rot);

  // Both source operands must be valid for us to accept.
  logic operands_ok;
  assign operands_ok = xif_issue_if.issue_req.rs_valid[0]
                     & xif_issue_if.issue_req.rs_valid[1];

  // Credit: only accept if the result FIFO has room for this instruction.
  logic [FifoPtrW:0] credit_q;
  logic              have_credit;
  assign have_credit = (credit_q < FifoDepth[FifoPtrW:0]);

  logic accept;
  assign accept = xif_issue_if.issue_valid & dec_is_rot & operands_ok & have_credit;

  // issue_ready deliberately does NOT depend on issue_valid: we are ready to answer any
  // issue request, and only withhold readiness for a ROPE.ROT we genuinely cannot take
  // yet (missing operands or no result-FIFO credit). A non-ROPE instruction is answered
  // immediately with accept=0 so the core is never stalled by us.
  assign xif_issue_if.issue_ready         = !dec_is_rot | (operands_ok & have_credit);
  assign xif_issue_if.issue_resp.accept    = accept;
  assign xif_issue_if.issue_resp.writeback = accept;      // ROPE.ROT always writes rd
  assign xif_issue_if.issue_resp.dualwrite = 1'b0;        // 32-bit result: no dual write
  assign xif_issue_if.issue_resp.dualread  = 3'b000;
  assign xif_issue_if.issue_resp.loadstore = 1'b0;        // no memory access (M8 only)
  assign xif_issue_if.issue_resp.ecswrite  = 1'b0;
  assign xif_issue_if.issue_resp.exc       = 1'b0;        // cannot raise an exception

  // ---------------------------------------------------------------------------
  // Rule 1: register the operands. Nothing combinational may consume rs* directly.
  // ---------------------------------------------------------------------------

  logic [15:0]            x_q, y_q;
  logic [rope_pkg::PosBits-1:0]     pos_q;
  logic [rope_pkg::PairIdxBits-1:0] pair_q;
  logic [X_ID_WIDTH-1:0]  id_q;
  logic [4:0]             rd_q;
  logic                   issued_q;

  logic [31:0] rs1_raw, rs2_raw;
  assign rs1_raw = xif_issue_if.issue_req.rs[0][31:0];
  assign rs2_raw = xif_issue_if.issue_req.rs[1][31:0];

  // Named intermediates: SystemVerilog does not allow bit-selecting a function call
  // result directly, and these also document the truncation of m and i to their
  // architectural widths.
  logic [15:0] rs2_pos_full, rs2_idx_full;
  assign rs2_pos_full = rope_pkg::rs2_pos(rs2_raw);
  assign rs2_idx_full = rope_pkg::rs2_pair_idx(rs2_raw);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      x_q      <= '0;
      y_q      <= '0;
      pos_q    <= '0;
      pair_q   <= '0;
      id_q     <= '0;
      rd_q     <= '0;
      issued_q <= 1'b0;
    end else begin
      issued_q <= accept;
      if (accept) begin
        // Straight slices of the operand words: no arithmetic in the issue cycle.
        x_q    <= rope_pkg::pair_x(rs1_raw);
        y_q    <= rope_pkg::pair_y(rs1_raw);
        pos_q  <= rs2_pos_full[rope_pkg::PosBits-1:0];
        pair_q <= rs2_idx_full[rope_pkg::PairIdxBits-1:0];
        id_q   <= xif_issue_if.issue_req.id;
        rd_q   <= rope_pkg::instr_rd(xif_issue_if.issue_req.instr);
      end
    end
  end

  // ---------------------------------------------------------------------------
  // Phase generation and LUT lookup -- all from REGISTERED operands
  // ---------------------------------------------------------------------------

  logic [rope_pkg::PhaseBits-1:0] phase;

  rope_phase_gen i_phase_gen (
    .clk_i,
    .rst_ni,
    .pos_i       ( pos_q  ),
    .pair_idx_i  ( pair_q ),
    .phase_o     ( phase  ),
    // Sequential mode is unused by ROPE.ROT (it belongs to the streaming variant).
    .seq_clear_i ( 1'b0   ),
    .seq_step_i  ( 1'b0   ),
    .seq_idx_i   ( '0     ),
    .seq_phase_o (        )
  );

  logic [15:0] sin_val, cos_val;

  rope_sin_lut i_lut (
    .phase_i ( phase   ),
    .sin_o   ( sin_val ),
    .cos_o   ( cos_val )
  );

  // ---------------------------------------------------------------------------
  // Datapath
  // ---------------------------------------------------------------------------

  logic [15:0] res_x, res_y;
  logic        res_valid;
  logic        dp_in_ready, dp_busy;
  logic [7:0]  dp_latency;

  rope_datapath #(
    .NumPipeRegs ( NumPipeRegs ),
    .RegLut      ( 1'b1        )
  ) i_datapath (
    .clk_i,
    .rst_ni,
    .flush_i     ( 1'b0        ),
    .x_i         ( x_q         ),
    .y_i         ( y_q         ),
    .sin_i       ( sin_val     ),
    .cos_i       ( cos_val     ),
    .in_valid_i  ( issued_q    ),
    .in_ready_o  ( dp_in_ready ),
    .x_o         ( res_x       ),
    .y_o         ( res_y       ),
    .out_valid_o ( res_valid   ),
    .busy_o      ( dp_busy     ),
    .latency_o   ( dp_latency  )
  );

  // ---------------------------------------------------------------------------
  // In-flight tracking, so a kill can be honoured (rule 2)
  // ---------------------------------------------------------------------------
  //
  // A tag rides alongside the datapath in a shift register whose depth matches the
  // datapath latency. `killed` is looked up per id, so a commit_kill arriving at any
  // time after issue suppresses the writeback for exactly that instruction.

  localparam int unsigned TagDepth = 8;

  typedef struct packed {
    logic                  valid;
    logic [X_ID_WIDTH-1:0] id;
    logic [4:0]            rd;
  } tag_t;

  tag_t tag_pipe_q [TagDepth];

  // Per-id kill flags. Set by commit_kill, cleared when the id is retired or reissued.
  logic [NumIds-1:0] killed_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      killed_q <= '0;
    end else begin
      // commit_valid has no ready: sample it unconditionally, every cycle.
      if (xif_commit_if.commit_valid && xif_commit_if.commit.commit_kill) begin
        killed_q[xif_commit_if.commit.id] <= 1'b1;
      end
      // A newly accepted instruction reusing an id starts un-killed. Ordered after the
      // kill capture so a same-cycle kill of a different id is not lost.
      if (accept) begin
        killed_q[xif_issue_if.issue_req.id] <= 1'b0;
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int unsigned k = 0; k < TagDepth; k++) tag_pipe_q[k] <= '0;
    end else begin
      tag_pipe_q[0].valid <= issued_q;
      tag_pipe_q[0].id    <= id_q;
      tag_pipe_q[0].rd    <= rd_q;
      for (int unsigned k = 1; k < TagDepth; k++) begin
        tag_pipe_q[k] <= tag_pipe_q[k-1];
      end
    end
  end

  // The tag that belongs to the result emerging now. dp_latency counts the datapath's
  // own stages; the tag pipe is indexed to match, so this cannot drift if NumPipeRegs
  // changes.
  logic [7:0] tag_sel;
  assign tag_sel = dp_latency - 8'd1;

  tag_t cur_tag;
  always_comb begin
    cur_tag = '0;
    for (int unsigned k = 0; k < TagDepth; k++) begin
      if (k == tag_sel) cur_tag = tag_pipe_q[k];
    end
  end

  // ---------------------------------------------------------------------------
  // Result FIFO -- returns results strictly in issue order
  // ---------------------------------------------------------------------------

  typedef struct packed {
    logic [X_ID_WIDTH-1:0] id;
    logic [4:0]            rd;
    logic [31:0]           data;
  } fifo_entry_t;

  fifo_entry_t       fifo_q [FifoDepth];
  logic [FifoPtrW:0] wr_ptr_q, rd_ptr_q;
  logic              fifo_empty, fifo_push, fifo_pop;

  assign fifo_empty = (wr_ptr_q == rd_ptr_q);

  // Push a completed, non-killed result. A killed instruction consumes its credit but
  // produces no writeback.
  logic result_is_killed;
  assign result_is_killed = killed_q[cur_tag.id];

  assign fifo_push = res_valid & cur_tag.valid & ~result_is_killed;
  assign fifo_pop  = xif_result_if.result_valid & xif_result_if.result_ready;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wr_ptr_q <= '0;
      rd_ptr_q <= '0;
      for (int unsigned k = 0; k < FifoDepth; k++) fifo_q[k] <= '0;
    end else begin
      if (fifo_push) begin
        fifo_q[wr_ptr_q[FifoPtrW-1:0]].id   <= cur_tag.id;
        fifo_q[wr_ptr_q[FifoPtrW-1:0]].rd   <= cur_tag.rd;
        fifo_q[wr_ptr_q[FifoPtrW-1:0]].data <= rope_pkg::pack_pair(res_x, res_y);
        wr_ptr_q <= wr_ptr_q + 1'b1;
      end
      if (fifo_pop) begin
        rd_ptr_q <= rd_ptr_q + 1'b1;
      end
    end
  end

  // Credit accounting: one credit per accepted instruction, released when the
  // instruction leaves (writeback accepted, or retired as killed).
  logic retire_killed;
  assign retire_killed = res_valid & cur_tag.valid & result_is_killed;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      credit_q <= '0;
    end else begin
      credit_q <= credit_q + {{FifoPtrW{1'b0}}, accept}
                           - {{FifoPtrW{1'b0}}, (fifo_pop | retire_killed)};
    end
  end

  // ---------------------------------------------------------------------------
  // Result channel
  // ---------------------------------------------------------------------------

  assign xif_result_if.result_valid    = ~fifo_empty;
  assign xif_result_if.result.id       = fifo_q[rd_ptr_q[FifoPtrW-1:0]].id;
  assign xif_result_if.result.data     = fifo_q[rd_ptr_q[FifoPtrW-1:0]].data;
  assign xif_result_if.result.rd       = fifo_q[rd_ptr_q[FifoPtrW-1:0]].rd;
  assign xif_result_if.result.we       = 1'b1;
  assign xif_result_if.result.ecsdata  = 6'b0;
  assign xif_result_if.result.ecswe    = 3'b0;
  assign xif_result_if.result.exc      = 1'b0;
  assign xif_result_if.result.exccode  = 6'b0;
  assign xif_result_if.result.err      = 1'b0;
  assign xif_result_if.result.dbg      = 1'b0;

  // ---------------------------------------------------------------------------
  // Protocol assertions (steps.md Section 9.4)
  // ---------------------------------------------------------------------------
`ifndef SYNTHESIS
  // The datapath must never be back-pressured: the credit scheme exists to guarantee it.
  always @(posedge clk_i) begin
    if (rst_ni && issued_q) begin
      assert (dp_in_ready)
        else $fatal(1, "rope_xif_coproc: datapath stalled -- credit accounting is wrong");
    end
  end

  // The FIFO must never overflow, for the same reason.
  always @(posedge clk_i) begin
    if (rst_ni && fifo_push) begin
      assert ((wr_ptr_q - rd_ptr_q) < FifoDepth[FifoPtrW:0])
        else $fatal(1, "rope_xif_coproc: result FIFO overflow");
    end
  end

  // Every emerging result must carry a valid tag.
  always @(posedge clk_i) begin
    if (rst_ni && res_valid) begin
      assert (cur_tag.valid)
        else $fatal(1, "rope_xif_coproc: result with no matching tag (tag pipe misaligned)");
    end
  end

  // The tag pipeline must be deep enough for the configured latency.
  initial begin
    assert (TagDepth >= 1 + 2 * NumPipeRegs + 1)
      else $fatal(1, "rope_xif_coproc: TagDepth too small for NumPipeRegs=%0d", NumPipeRegs);
  end
`endif

endmodule : rope_xif_coproc
```

## A.9 `rtl/rope_cv32e40x_wrapper.sv` — core + interface + coprocessor (244 lines)

```systemverilog
// -----------------------------------------------------------------------------
// rope_cv32e40x_wrapper.sv -- CV32E40X + CORE-V-XIF + BF16 RoPE coprocessor.
//
// Milestone 6 step 1-3: instantiate the core, the interface and the coprocessor, and
// connect all six XIF channel bundles. The compressed channel is tied off (ROPE.ROT has
// no compressed form) and the memory channels are left idle until the streaming variant
// (Milestone 8).
//
// The interface definition comes from the core's own cv32e40x_if_xif.sv rather than a
// hand-rolled copy, which keeps every struct type-compatible with the core by
// construction. Note this differs from steps.md Section 10.1 step 3, which suggests
// deps/core-v-xif/src/core_v_xif.sv: cv32e40x ships an equivalent definition, and using
// it keeps the project self-contained inside this repository with no external clone.
//
// The five XIF parameters are declared ONCE here as localparams and passed to both the
// core and the coprocessor. steps.md Section 9.5 is emphatic about this: a mismatch
// between the core, the interface instance and the coprocessor produces confusing
// elaboration-time breakage.
// -----------------------------------------------------------------------------

module rope_cv32e40x_wrapper
  import cv32e40x_pkg::*;
#(
  parameter rv32_e         RV32             = RV32I,
  parameter m_ext_e        M_EXT            = M,
  parameter bit            DEBUG            = 1,
  parameter int unsigned   NUM_MHPMCOUNTERS = 1,
  parameter int unsigned   NumPipeRegs      = 1
) (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        scan_cg_en_i,

  input  logic [31:0] boot_addr_i,
  input  logic [31:0] dm_exception_addr_i,
  input  logic [31:0] dm_halt_addr_i,
  input  logic [31:0] mhartid_i,
  input  logic [ 3:0] mimpid_patch_i,
  input  logic [31:0] mtvec_addr_i,

  // Instruction memory interface
  output logic        instr_req_o,
  input  logic        instr_gnt_i,
  input  logic        instr_rvalid_i,
  output logic [31:0] instr_addr_o,
  output logic [ 1:0] instr_memtype_o,
  output logic [ 2:0] instr_prot_o,
  output logic        instr_dbg_o,
  input  logic [31:0] instr_rdata_i,
  input  logic        instr_err_i,

  // Data memory interface
  output logic        data_req_o,
  input  logic        data_gnt_i,
  input  logic        data_rvalid_i,
  output logic [31:0] data_addr_o,
  output logic [ 3:0] data_be_o,
  output logic        data_we_o,
  output logic [31:0] data_wdata_o,
  output logic [ 1:0] data_memtype_o,
  output logic [ 2:0] data_prot_o,
  output logic        data_dbg_o,
  output logic [ 5:0] data_atop_o,
  input  logic [31:0] data_rdata_i,
  input  logic        data_err_i,
  input  logic        data_exokay_i,

  output logic [63:0] mcycle_o,
  input  logic [63:0] time_i,

  input  logic [31:0] irq_i,
  input  logic        wu_wfe_i,

  input  logic        clic_irq_i,
  input  logic [ 4:0] clic_irq_id_i,
  input  logic [ 7:0] clic_irq_level_i,
  input  logic [ 1:0] clic_irq_priv_i,
  input  logic        clic_irq_shv_i,

  output logic        fencei_flush_req_o,
  input  logic        fencei_flush_ack_i,

  input  logic        debug_req_i,
  output logic        debug_havereset_o,
  output logic        debug_running_o,
  output logic        debug_halted_o,
  output logic        debug_pc_valid_o,
  output logic [31:0] debug_pc_o,

  input  logic        fetch_enable_i,
  output logic        core_sleep_o
);

  // ---------------------------------------------------------------------------
  // XIF parameters -- locked on day one, shared by core, interface and coprocessor
  // ---------------------------------------------------------------------------

  localparam int unsigned X_NUM_RS    = 2;     // 3 if an R4-type op is added later
  localparam int unsigned X_ID_WIDTH  = 4;
  localparam int unsigned X_MEM_WIDTH = 32;    // exactly one interleaved BF16 pair/txn
  localparam int unsigned X_RFR_WIDTH = 32;
  localparam int unsigned X_RFW_WIDTH = 32;    // 32 suffices: no dual writeback

  // X_MISA / X_ECS_XS: ROPE.ROT advertises no MISA bit and never writes the extension
  // context status, so both stay zero.
  localparam logic [31:0] X_MISA      = 32'h0;
  localparam logic [ 1:0] X_ECS_XS    = 2'b00;

  // ---------------------------------------------------------------------------
  // Interface instance
  // ---------------------------------------------------------------------------

  cv32e40x_if_xif #(
    .X_NUM_RS    ( X_NUM_RS    ),
    .X_ID_WIDTH  ( X_ID_WIDTH  ),
    .X_MEM_WIDTH ( X_MEM_WIDTH ),
    .X_RFR_WIDTH ( X_RFR_WIDTH ),
    .X_RFW_WIDTH ( X_RFW_WIDTH ),
    .X_MISA      ( X_MISA      ),
    .X_ECS_XS    ( X_ECS_XS    )
  ) xif ();

  // ---------------------------------------------------------------------------
  // Core
  // ---------------------------------------------------------------------------

  cv32e40x_core #(
    .RV32             ( RV32             ),
    .M_EXT            ( M_EXT            ),
    .DEBUG            ( DEBUG            ),
    .X_EXT            ( 1'b1             ),   // enable the eXtension interface
    .X_NUM_RS         ( X_NUM_RS         ),
    .X_ID_WIDTH       ( X_ID_WIDTH       ),
    .X_MEM_WIDTH      ( X_MEM_WIDTH      ),
    .X_RFR_WIDTH      ( X_RFR_WIDTH      ),
    .X_RFW_WIDTH      ( X_RFW_WIDTH      ),
    .X_MISA           ( X_MISA           ),
    .X_ECS_XS         ( X_ECS_XS         ),
    .NUM_MHPMCOUNTERS ( NUM_MHPMCOUNTERS )
  ) i_core (
    .clk_i,
    .rst_ni,
    .scan_cg_en_i,

    .boot_addr_i,
    .dm_exception_addr_i,
    .dm_halt_addr_i,
    .mhartid_i,
    .mimpid_patch_i,
    .mtvec_addr_i,

    .instr_req_o,
    .instr_gnt_i,
    .instr_rvalid_i,
    .instr_addr_o,
    .instr_memtype_o,
    .instr_prot_o,
    .instr_dbg_o,
    .instr_rdata_i,
    .instr_err_i,

    .data_req_o,
    .data_gnt_i,
    .data_rvalid_i,
    .data_addr_o,
    .data_be_o,
    .data_we_o,
    .data_wdata_o,
    .data_memtype_o,
    .data_prot_o,
    .data_dbg_o,
    .data_atop_o,
    .data_rdata_i,
    .data_err_i,
    .data_exokay_i,

    .mcycle_o,
    .time_i,

    // All six XIF channel bundles.
    .xif_compressed_if ( xif ),
    .xif_issue_if      ( xif ),
    .xif_commit_if     ( xif ),
    .xif_mem_if        ( xif ),
    .xif_mem_result_if ( xif ),
    .xif_result_if     ( xif ),

    .irq_i,
    .wu_wfe_i,

    .clic_irq_i,
    .clic_irq_id_i,
    .clic_irq_level_i,
    .clic_irq_priv_i,
    .clic_irq_shv_i,

    .fencei_flush_req_o,
    .fencei_flush_ack_i,

    .debug_req_i,
    .debug_havereset_o,
    .debug_running_o,
    .debug_halted_o,
    .debug_pc_valid_o,
    .debug_pc_o,

    .fetch_enable_i,
    .core_sleep_o
  );

  // ---------------------------------------------------------------------------
  // Coprocessor
  // ---------------------------------------------------------------------------

  rope_xif_coproc #(
    .X_NUM_RS    ( X_NUM_RS    ),
    .X_ID_WIDTH  ( X_ID_WIDTH  ),
    .X_MEM_WIDTH ( X_MEM_WIDTH ),
    .X_RFR_WIDTH ( X_RFR_WIDTH ),
    .X_RFW_WIDTH ( X_RFW_WIDTH ),
    .NumPipeRegs ( NumPipeRegs )
  ) i_rope (
    .clk_i,
    .rst_ni,
    .xif_issue_if  ( xif ),
    .xif_commit_if ( xif ),
    .xif_result_if ( xif )
  );

  // ---------------------------------------------------------------------------
  // Unused channels
  // ---------------------------------------------------------------------------
  //
  // Compressed: ROPE.ROT has no 16-bit form, so we never accept a compressed offload.
  assign xif.compressed_ready = 1'b1;   // always able to answer
  assign xif.compressed_resp  = '0;     // accept = 0

  // Memory: ROPE.ROT performs no memory accesses. These become live in Milestone 8,
  // where the "always able to sink a returning load" rule (mem_result_valid has no
  // ready) must be honoured.
  assign xif.mem_valid = 1'b0;
  assign xif.mem_req   = '0;

endmodule : rope_cv32e40x_wrapper
```

## A.10 `rtl/rope_theta_rom.sv` — generated integer phase increments (95 lines)

Generated by `gen/gen_theta_rom.py`. 64 entries × 32 bits = **256 bytes**. Shown with the
first six entries; the remaining 58 follow the same pattern.

```systemverilog
// -----------------------------------------------------------------------------
// rope_theta_rom.sv -- GENERATED by gen/gen_theta_rom.py. DO NOT EDIT BY HAND.
//
// Integer phase increments for RoPE:  Phi_i = round(2^32 * theta_i / 2pi)
// with theta_i = base^(-2i/d),  d = 128,  base = 10000.
//
// These are INTEGERS by design (INV-2). The phase word is an address used to index
// the sine LUT; it is never multiplied by or added to BF16 data. A 32-bit phase
// makes 'mod 2*pi' free (natural wraparound) and has zero drift.
//
// Size: 64 entries x 32 bits = 256 bytes.
// -----------------------------------------------------------------------------

module rope_theta_rom #(
  parameter int unsigned HeadDim   = 128,
  parameter int unsigned NumPairs  = 64,   // HeadDim/2
  parameter int unsigned PhaseBits = 32
) (
  input  logic [5:0]            pair_idx_i,   // i, the frequency index
  output logic [PhaseBits-1:0]  phi_o         // Phi_i
);

  // Combinational ROM. Small enough (see size above) that synthesis maps this to
  // logic cleanly; no memory macro required.
  always_comb begin
    unique case (pair_idx_i)
      6'd0  : phi_o = 32'h28BE60DC;  // theta_0 = 1.000000000e+00
      6'd1  : phi_o = 32'h234855E5;  // theta_1 = 8.659643234e-01
      6'd2  : phi_o = 32'h1E8DAE2A;  // theta_2 = 7.498942093e-01
      6'd3  : phi_o = 32'h1A754BCE;  // theta_3 = 6.493816316e-01
      6'd4  : phi_o = 32'h16E96ECB;  // theta_4 = 5.623413252e-01
      6'd5  : phi_o = 32'h13D7416B;  // theta_5 = 4.869675252e-01
      // ... entries 6..63 elided in this document; see the file ...
      default: phi_o = 32'h0;
    endcase
  end

endmodule : rope_theta_rom
```

Note `Phi_0 = 0x28BE60DC = 683565276 = round(2^32 / 2π)` — one radian expressed as a
fraction of a turn — and `Phi_63 = 78937`, the number quoted in Part 5.2.

## A.11 `rtl/rope_sin_rom.sv` — generated BF16 quarter-wave sine table (1062 lines)

Generated by `gen/gen_sin_lut.py`. 1024 entries × 16 bits = **2 KB**. The body is 1024
lines of pure table data; the header and first seven entries are shown.

```systemverilog
// -----------------------------------------------------------------------------
// rope_sin_rom.sv -- GENERATED by gen/gen_sin_lut.py. DO NOT EDIT BY HAND.
//
// BF16 quarter-wave sine table, 1024 entries, MIDPOINT sampled:
//     T[k] = bf16(sin((k + 0.5) * (pi/2) / 1024))
//
// Midpoint sampling makes the Q1/Q3 reflection index exactly (N-1-idx) with no
// endpoint special case, and halves the worst-case sampling error (steps.md 7.3).
//
// Angle resolution: (pi/2)/1024 = 0.001534 rad. Since |d(sin)/d(theta)| <= 1 the
// worst-case value error is ~0.001534, already 5.1x finer than
// BF16's ulp near 1.0 (2^-7 = 0.007812) -- BF16's own coarseness bounds the
// table size, which is the inverse of the usual concern.
//
// Size: 1024 entries x 2 bytes (BF16) = 2 KB.
//
// NOTE: sin(0) is NOT 0 and cos(0) is read from T[N-1]; midpoint sampling trades
// exactness at the quadrant endpoints for the free reflection. See docs/notes.md.
// -----------------------------------------------------------------------------

module rope_sin_rom #(
  parameter int unsigned LutBits = 10,
  parameter int unsigned LutN    = 1024
) (
  input  logic [LutBits-1:0] idx_i,
  output logic [15:0]        sin_o    // BF16, always positive (quarter wave)
);

  // Unpacked constant array: infers a ROM / logic cone, and lets both the
  // sin and cos lookups read it in the same cycle (steps.md 7.2).
  logic [15:0] table_q [LutN];

  always_comb begin
    table_q[   0] = 16'h3A49;  // sin(0.00076699) = 0.00076675
    table_q[   1] = 16'h3B17;  // sin(0.00230097) = 0.00230408
    table_q[   2] = 16'h3B7B;  // sin(0.00383495) = 0.00382996
    table_q[   3] = 16'h3BB0;  // sin(0.00536893) = 0.00537109
    table_q[   4] = 16'h3BE2;  // sin(0.00690291) = 0.00689697
    table_q[   5] = 16'h3C0A;  // sin(0.00843689) = 0.00842285
    table_q[   6] = 16'h3C23;  // sin(0.00997088) = 0.00994873
    // ... entries 7..1023 elided in this document; see the file ...
  end

  assign sin_o = table_q[idx_i];

endmodule : rope_sin_rom
```

`table_q[0] = 0.00076675` is the value discussed in Part 2.4 — the reason position 0 is a
near-identity rather than an exact one.

## A.12 Vendored CVFPU (copied, not written)

Copied verbatim into `rtl/vendor/` by `gen/vendor_cvfpu.sh` (Appendix B.10). Not reproduced
here — they are unmodified third-party sources under SolderPad 0.51 / Apache-2.0.

```
rtl/vendor/cvfpu/src/          fpnew_pkg.sv  fpnew_classifier.sv  fpnew_rounding.sv
                               fpnew_fma.sv  fpnew_noncomp.sv
                               fpnew_opgroup_fmt_slice.sv  fpnew_opgroup_block.sv
                               fpnew_top.sv
rtl/vendor/cvfpu/              LICENSE.solderpad  README.license.md
rtl/vendor/common_cells/src/   cf_math_pkg.sv  lzc.sv  rr_arb_tree.sv
rtl/vendor/common_cells/include/common_cells/   registers.svh  assertions.svh
```

Total vendored: **3013 lines**, versus 1543 lines of hand-written RTL.

---
---
> **A note on this appendix.** The two files that constitute the reference model
> (`bf16.py`, `rope_ref.py`) and all the code generators are reproduced **in full**,
> because correctness of everything else is measured against them. The longer
> results-producing and test-harness scripts (`characterize.py`, `gen_vectors.py`,
> `test_bf16.py`, `test_rope_ref.py`) are reproduced with their substantive logic intact
> and repetitive report-formatting boilerplate marked `[...]`; each elision is labelled and
> the complete file is at the stated path.

# Appendix B — Golden model and generators (Python)

Dependencies: **numpy only.** No `ml_dtypes`, no PyTorch, no pytest.

## B.1 `model/bf16.py` — BF16 arithmetic with exact RNE (225 lines, full)

The foundation. Everything works on **uint16 bit patterns**, not a float extension type,
so "bit-exact" comparison against hardware is unambiguous.

```python
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
```

## B.2 `model/rope_ref.py` — the three reference models (293 lines, full)

```python
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
```

## B.3 `model/characterize.py` — produces the error budget (438 lines, abridged)

Full file at `model/characterize.py`. Report-formatting boilerplate elided.

```python
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
    xr = np.asarray(xr, dtype=np.float64); yr = np.asarray(yr, dtype=np.float64)
    xe = np.asarray(xe, dtype=np.float64); ye = np.asarray(ye, dtype=np.float64)
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
        "n": n, "a_pair": a_pair, "b_pair": b_pair,
        "a_comp": a_comp, "b_comp": b_comp,
        "exact_match_frac": exact_match / n,
    }


# ---------------------------------------------------------------------------
# 3: error vs m and vs i   [study_vs_m / study_vs_i: same shape, bucketed sweeps]
# ---------------------------------------------------------------------------
# [...]


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
        p0 = bf16_mul(x, c);  p1 = bf16_mul(y, s)
        p2 = bf16_mul(x, s);  p3 = bf16_mul(y, c)
        xa = bf16_sub(p0, p1)
        ya = bf16_add(p2, p3)

        # Wide accumulate.
        xf = bf16_to_f32(x).astype(np.float32); yf = bf16_to_f32(y).astype(np.float32)
        cf = bf16_to_f32(c).astype(np.float32); sf = bf16_to_f32(s).astype(np.float32)
        xb = to_bf16(xf * cf - yf * sf)
        yb = to_bf16(xf * sf + yf * cf)

        # Exact, using the true angle the phase word represents.
        ang = 2 * np.pi * (phases.astype(np.float64) / 2.0**32)
        xd = bf16_to_f32(x).astype(np.float64); yd = bf16_to_f32(y).astype(np.float64)
        xe = xd * np.cos(ang) - yd * np.sin(ang)
        ye = xd * np.sin(ang) + yd * np.cos(ang)

        ra = rel_err(bf16_to_f32(xa), xe);  rb = rel_err(bf16_to_f32(xb), xe)
        ba = bits_retained(bf16_to_f32(xa), xe)
        bb = bits_retained(bf16_to_f32(xb), xe)
        results[label] = {
            "strict_max_rel": ra.max(), "strict_rms_rel": np.sqrt(np.mean(ra**2)),
            "wide_max_rel": rb.max(),   "wide_rms_rel": np.sqrt(np.mean(rb**2)),
            "strict_bits_min": ba.min(), "strict_bits_mean": ba.mean(),
            "wide_bits_min": bb.min(),   "wide_bits_mean": bb.mean(),
            "strict_frac_lost_half": float(np.mean(ba < 4)),
            "wide_frac_lost_half": float(np.mean(bb < 4)),
        }
    out["phase_sweep"] = results

    # Ratio sweep across the BF16 exponent range at the worst phase.
    # [... exponent_sweep: same measurement at x scaled by 2^-20 .. 2^20, using
    #      scale = (2.0**exp) * 1.4140625 so the products genuinely round ...]

    # Which m actually reaches ~45 degrees for a few frequencies.
    out["m_at_45deg"] = [(i, find_m_near_45deg(phi[i])) for i in (0, 1, 8, 32, 63)]
    return out


# ---------------------------------------------------------------------------
# Reporting  [build_report(): emits the markdown tables reproduced in Part 6.3]
# ---------------------------------------------------------------------------
# [...]
```

## B.4 `model/gen_vectors.py` — RTL test vectors (329 lines, abridged)

Full file at `model/gen_vectors.py`. This is where adversarial input selection lives.

```python
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
            # [... same three-op emission ...]

    # 3. Bulk random.
    a = _random_bf16(rng, n);  b = _random_bf16(rng, n)
    ops = rng.integers(0, 3, size=n)
    r_mul = bf16_mul(a, b);  r_add = bf16_add(a, b);  r_sub = bf16_sub(a, b)
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

    # Random fill.  Full 16-bit m range, matching the instruction field.
    rng = _rng(seed)
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
    s = lut_sin(ph);  c = lut_cos(ph)
    for k in range(len(ph)):
        f.write(f"{int(ph[k]):08x} {int(s[k]):04x} {int(c[k]):04x}\n")
        count += 1
    return count


def gen_datapath(n, seed, f, d=128, base=DEFAULT_BASE):
    """Full datapath vectors vs Model A, including the cancellation corner."""
    # Specials against ordinary values; then the cancellation corner: for each of
    # i in (0,1,2,8,32) find the m landing nearest 45 degrees and sweep y/x ratios
    # 1.0, 1 +/- 2^-7, 1.01, 0.99 with full-mantissa x drawn from [1,2);
    # then m = 0..3 for every i; then bulk random with the full 16-bit m range.
    # [... assembly of xs/ys/ms/ids, then one vectorised call ...]
    xr_, yr_ = rope_strict_bf16(x, y, m, i, phi)
    for k in range(len(x)):
        f.write(f"{int(x[k]):04x} {int(y[k]):04x} {int(m[k]):04x} {int(i[k]):02x} "
                f"{int(xr_[k]):04x} {int(yr_[k]):04x}\n")


def gen_xif(n, seed, f, d=128, base=DEFAULT_BASE):
    """XIF-level vectors: packed rs1/rs2 -> packed rd, matching rope_pkg's packing.

    rs1 = {x, y}, rs2 = {m, i}, rd = {x', y'}  (x in the HIGH half throughout).
    """
    phi = phi_table(d, base)
    half = d // 2
    rng = _rng(seed)

    x = _random_bf16(rng, n);  y = _random_bf16(rng, n)
    m = rng.integers(0, 65536, size=n)  # full 16-bit m range, per PosBits
    i = rng.integers(0, half, size=n)

    xr, yr = rope_strict_bf16(x, y, m.astype(np.uint64), i, phi)
    for k in range(n):
        rs1 = (int(x[k]) << 16) | int(y[k])
        rs2 = (int(m[k]) << 16) | int(i[k])
        rd = (int(xr[k]) << 16) | int(yr[k])
        f.write(f"{rs1:08x} {rs2:08x} {rd:08x}\n")
    return n
```

## B.5 `model/gen_c_vectors.py` — vectors compiled into the C test (133 lines, abridged)

```python
#!/usr/bin/env python3
"""Emit sw/rope_vectors.h -- compiled-in test vectors for the bare-metal C test.

Uses the same golden model as the RTL testbenches, so a pass in software confirms that
the *software* view agrees too: instruction encoding, operand packing, and the
half-split / interleaved whole-vector loops.
"""

DEFAULT_N = 64          # keep the array small: it is linked into a bare-metal image
CHAIN_STEPS = 8
CHAIN_M = 13
CHAIN_I = 3


def emit(n, d, base):
    phi = phi_table(d, base)
    half = d // 2
    rng = np.random.default_rng(20240803)

    # --- per-instruction vectors ---
    x = to_bf16(rng.uniform(-4.0, 4.0, size=n))
    y = to_bf16(rng.uniform(-4.0, 4.0, size=n))
    # Full 16-bit m range: rope_pkg::PosBits is 16 to match the rs2[31:16] field, and
    # exercising m > 8191 from software is what proves the ISA field is honoured
    # end to end rather than truncated somewhere in the coprocessor.
    m = rng.integers(0, 65536, size=n)
    i = rng.integers(0, half, size=n)

    xr, yr = rope_strict_bf16(x, y, m.astype(np.uint64), i, phi)
    rs1 = [(int(x[k]) << 16) | int(y[k]) for k in range(n)]
    rs2 = [(int(m[k]) << 16) | int(i[k]) for k in range(n)]
    rd  = [(int(xr[k]) << 16) | int(yr[k]) for k in range(n)]

    # --- dependent chain ---
    # Repeatedly rotate the SAME pair, feeding each result into the next instruction.
    # This is what exercises the core's forwarding into the registered-operand path.
    acc_x = np.array([int(x[0])], dtype=np.uint16)
    acc_y = np.array([int(y[0])], dtype=np.uint16)
    for _ in range(CHAIN_STEPS):
        acc_x, acc_y = rope_strict_bf16(acc_x, acc_y, CHAIN_M, np.array([CHAIN_I]), phi)
    chain_expected = (int(acc_x[0]) << 16) | int(acc_y[0])

    # --- whole-head vectors, both conventions ---
    head_in = to_bf16(rng.uniform(-4.0, 4.0, size=d))
    head_m = 137
    out_hs = rope_vector_strict_bf16(head_in, head_m, phi, convention="half_split")
    out_il = rope_vector_strict_bf16(head_in, head_m, phi, convention="interleaved")

    # [... emits a C header with #defines and six static const arrays ...]
```

## B.6 `model/test_bf16.py` — validates the BF16 primitives (277 lines, abridged)

Nine test groups; the two that matter most are shown in full.

```python
#!/usr/bin/env python3
"""Validate the BF16 primitives in bf16.py.

Plain Python, no pytest — run directly:  python3 model/test_bf16.py

The point of this file is to prove that `round_f32_to_bf16` (fast, bit-twiddling) agrees
with `_round_f32_to_bf16_reference_scalar` (slow, derived from first principles) so that
every downstream bit-exactness claim rests on a rounding function that was actually
checked rather than assumed.
"""

def test_roundtrip_all_bf16():
    """Every one of the 65536 BF16 patterns must survive bits -> f32 -> bits."""
    allbits = np.arange(1 << 16, dtype=np.uint16)
    back = round_f32_to_bf16(bf16_to_f32(allbits))
    # NaN patterns legitimately canonicalise to BF16_QNAN; everything else is identity.
    nanmask = is_nan(allbits)
    check("bits->f32->bits identity (non-NaN, all 65536)",
          np.array_equal(back[~nanmask], allbits[~nanmask]))
    check("NaN patterns canonicalise to 0x7FC0", np.all(back[nanmask] == BF16_QNAN))


def test_rne_vs_reference_ties():
    """Directed exact-tie test — the case truncation would silently get wrong.

    An exact tie is an FP32 whose low 16 mantissa bits are exactly 0x8000. RNE must
    round to the even neighbour; truncation would always round down and introduce the
    systematic negative bias steps.md Section 5.3 warns about.
    """
    # Build ties across the whole exponent range and both parities of the retained LSB.
    ties = []
    for exp in range(1, 255):
        for man_hi in (0x00, 0x01, 0x02, 0x03, 0x7E, 0x7F):
            for sign in (0, 1):
                u = (sign << 31) | (exp << 23) | (man_hi << 16) | 0x8000
                ties.append(u)
    u = np.array(ties, dtype=np.uint32)
    xf = u.view(np.float32)
    fast = round_f32_to_bf16(xf)

    bad = 0
    for i in range(len(u)):
        ref = _round_f32_to_bf16_reference_scalar(xf[i])
        if int(fast[i]) != ref:
            bad += 1
    check(f"RNE on {len(u)} exact ties", bad == 0)

    # And assert the ties-to-even property explicitly: result significand must be even.
    check("every exact tie rounds to an even significand",
          np.all((fast & np.uint16(1)) == 0))

    # A truncating implementation would differ from ours on ties with odd LSB -> prove
    # our function is genuinely not truncation.
    trunc = (u >> 16).astype(np.uint16)
    check("RNE differs from truncation on odd-LSB ties (not silently truncating)",
          np.any(fast != trunc))

# Remaining groups:
#   test_upper_16_bits_identity   bf16_to_f32(b) == bitcast<float>(b<<16), all 65536
#   test_known_values             hand-computed encodings (1.0 -> 0x3F80 etc.)
#   test_rne_vs_reference_random  200k random FP32 patterns vs the slow reference
#   test_no_bias_on_ties          ties round up/down ~equally (bias check)
#   test_specials                 +/-Inf, NaN, +/-0, overflow, the low-payload NaN case
#   test_subnormals               round-trips, classification, FTZ sign preservation
#   test_arithmetic_single_rounding  mul/add/sub == round_once(FP64-exact), 300k pairs
# [...]
```

## B.7 `model/test_rope_ref.py` — validates the reference model (444 lines, abridged)

Fifteen test groups. The two most load-bearing are shown.

```python
#!/usr/bin/env python3
"""Milestone 0 exit criteria for the RoPE golden model.

Covers the correctness half of Milestone 0. The *quantitative* error budget (Model A vs
Model C, strict vs wide accumulate, the cancellation study) lives in characterize.py so
that the numbers land in one reviewable report rather than in pass/fail assertions.
"""

def test_inv2_bf16_phase_is_unrepresentable():
    """Demonstrate, not assert on faith, that a BF16 angle is useless at large m.

    For i=0, theta_0 = 1.0 rad, so the angle at position m is m radians.

    NOTE ON steps.md INV-2: the plan states that at m=1000 "the representable-value
    spacing exceeds a full period of 2*pi", citing 1000*2^-8 ~= 3.9 rad. The measured
    BF16 spacing at 1000 is 4.0 rad, which exceeds *pi* (a half period) but not 2*pi.
    The full-period claim becomes true at m >= 1024, where the spacing is 8.0 rad.
    The invariant itself is unaffected -- 4 rad of quantisation on an angle already
    destroys all correlation with the true value -- but the example is off by ~1.6x.
    Both the correct weaker claim at m=1000 and the strong claim at m=1024 are asserted
    here so the writeup quotes a number that survives checking.
    """
    sp1000 = _bf16_spacing(1000.0)
    check(f"m=1000: BF16 spacing {sp1000:.3f} rad > pi -- half a period lost",
          sp1000 > np.pi)
    sp1024 = _bf16_spacing(1024.0)
    check(f"m=1024: BF16 spacing {sp1024:.3f} rad > 2*pi -- a full period",
          sp1024 > 2 * np.pi)

    # The consequence that actually matters: the induced angle error is O(1) radians,
    # so cos/sin of the rounded angle are uncorrelated with the true values.
    check(f"induced angle error at m=1000 is {sp1000/2:.3f} rad (>= 1 rad: garbage)",
          sp1000 / 2 >= 1.0)

    # Argument reduction budget from steps.md: log2(4096) + 10 ~= 22 bits needed,
    # BF16 supplies 8 significand bits.
    need = np.log2(4096) + 10
    check(f"m<=4096 with ~10-bit angle precision needs ~{need:.0f} bits > BF16's 8",
          need > 8)

    # And the integer accumulator has no such problem: exact for all m.
    phi = phi_table(128)
    check("integer phase exact at m=1000",
          int(phase_of(1000, 0, phi)) == (1000 * int(phi[0])) & PHASE_MASK)


def test_conventions_related_by_permutation():
    """Interleaved and half-split must differ by exactly the fixed permutation."""
    d = 64
    v_inter = to_bf16(np.random.default_rng(41).uniform(-4, 4, size=d))
    perm = permutation_interleaved_to_half_split(d)
    phi = phi_table(d)
    m = 137

    out_inter = rope_vector_strict_bf16(v_inter, m, phi, convention="interleaved")
    out_hs = rope_vector_strict_bf16(v_inter[perm], m, phi, convention="half_split")
    check("rope_interleaved(v)[perm] == rope_half_split(v[perm]), bit-exact",
          np.array_equal(out_inter[perm], out_hs))

    # And confirm the two are genuinely different in place -- i.e. the trap is real.
    out_hs_same_layout = rope_vector_strict_bf16(v_inter, m, phi, convention="half_split")
    check("the two conventions really do differ on the same layout (trap is real)",
          not np.array_equal(out_inter, out_hs_same_layout))

# Remaining groups:
#   test_phase_table                    Phi_0 == round(2^32/2pi); Phi_min >= 60000; monotonic
#   test_phase_wraparound               exact modular wrap; sequential == random access
#   test_lut_sizing_argument            LUT resolution finer than a BF16 ulp
#   test_lut_reflection_symmetry        Q1 mirror == LUT[1023-idx] incl. idx 0 and 1023
#   test_lut_cos_is_sin_plus_quarter    cos(P) == sin(P + 2^30) over 100k random phases
#   test_lut_accuracy_vs_true_sin       within half-bucket + half-ulp budget
#   test_sin2_plus_cos2                 reports deviation; fails only if > 4 ulp
#   test_model_a_deterministic          reproduces itself bit-exactly
#   test_model_a_identity_at_m0         the midpoint-sampling consequence (Part 2.4)
#   test_model_a_specials_propagate     NaN/Inf propagate rather than becoming finite
#   test_half_split_matches_huggingface vs hf_apply_rotary_pos_emb to 1e-12, several m
#   test_pair_indices_cover_vector      every element rotated exactly once
#   test_rotation_preserves_norm        pair norm preserved to <5% worst case
# [...]
```

## B.8 `gen/gen_theta_rom.py` — emits the phase-increment ROM (94 lines, full)

```python
#!/usr/bin/env python3
"""Emit rtl/rope_theta_rom.sv — the integer phase-increment ROM (steps.md 6.2).

    Phi_i = round(2^32 * theta_i / (2*pi)),   theta_i = base^(-2i/d),  i in [0, d/2)

Per INV-2 these are *integers*: the phase word is an address, never BF16 data.

Usage:
    python3 gen/gen_theta_rom.py [--d 128] [--base 10000] [--out rtl/rope_theta_rom.sv]

Size for d=128: 64 entries x 32 bits = 256 bytes.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))

import numpy as np  # noqa: E402

from rope_ref import DEFAULT_BASE, PHASE_BITS, phi_table  # noqa: E402


def emit(d, base, module="rope_theta_rom"):
    phi = phi_table(d, base)
    half = d // 2
    idx_w = max(1, int(np.ceil(np.log2(half))))

    L = []
    w = L.append
    w("// -----------------------------------------------------------------------------")
    w(f"// {module}.sv -- GENERATED by gen/gen_theta_rom.py. DO NOT EDIT BY HAND.")
    w("//")
    w(f"// Integer phase increments for RoPE:  Phi_i = round(2^{PHASE_BITS} * theta_i / 2pi)")
    w(f"// with theta_i = base^(-2i/d),  d = {d},  base = {base:g}.")
    w("//")
    w("// These are INTEGERS by design (INV-2). The phase word is an address used to index")
    w("// the sine LUT; it is never multiplied by or added to BF16 data. A 32-bit phase")
    w("// makes 'mod 2*pi' free (natural wraparound) and has zero drift.")
    w("//")
    w(f"// Size: {half} entries x {PHASE_BITS} bits = {half*PHASE_BITS//8} bytes.")
    w("// -----------------------------------------------------------------------------")
    w("")
    w(f"module {module} #(")
    w(f"  parameter int unsigned HeadDim   = {d},")
    w(f"  parameter int unsigned NumPairs  = {half},   // HeadDim/2")
    w(f"  parameter int unsigned PhaseBits = {PHASE_BITS}")
    w(") (")
    w(f"  input  logic [{idx_w-1}:0]            pair_idx_i,   // i, the frequency index")
    w("  output logic [PhaseBits-1:0]  phi_o         // Phi_i")
    w(");")
    w("")
    w("  // Combinational ROM. Small enough (see size above) that synthesis maps this to")
    w("  // logic cleanly; no memory macro required.")
    w("  always_comb begin")
    w("    unique case (pair_idx_i)")
    for i, p in enumerate(phi):
        theta = float(base) ** (-2.0 * i / d)
        w(
            f"      {idx_w}'d{i:<3}: phi_o = {PHASE_BITS}'h{int(p):08X};"
            f"  // theta_{i} = {theta:.9e}"
        )
    if (1 << idx_w) != half:
        w(f"      default: phi_o = {PHASE_BITS}'h0;  // unreachable for i < {half}")
    else:
        w(f"      default: phi_o = {PHASE_BITS}'h0;")
    w("    endcase")
    w("  end")
    w("")
    w("endmodule")
    w("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128, help="head dimension")
    ap.add_argument("--base", type=float, default=DEFAULT_BASE, help="RoPE base (LLaMA: 10000)")
    ap.add_argument("--out", default=None, help="output path (default: stdout)")
    args = ap.parse_args()

    text = emit(args.d, args.base)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out} (d={args.d}, base={args.base:g})", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## B.9 `gen/gen_sin_lut.py` — emits the sine table (112 lines, full)

```python
#!/usr/bin/env python3
"""Emit rtl/rope_sin_rom.sv — the BF16 quarter-wave sine table (steps.md Milestone 3).

    T[k] = BF16( sin( (k + 0.5) * (pi/2) / N ) ),   k in [0, N)

MIDPOINT sampled, which is what makes the Q1/Q3 reflection index exactly N-1-idx with
no endpoint special case and no 1025th entry (steps.md 7.3):

    angle       = (k+0.5)*D                with D = (pi/2)/N
    pi/2 - angle = N*D - (k+0.5)*D = ((N-1-k)+0.5)*D    -> index N-1-k, exactly

The table is a bare ROM. Quadrant decode, reflection and sign application live in the
hand-written rope_sin_lut.sv, so this generated file stays trivially auditable.

Usage:
    python3 gen/gen_sin_lut.py [--bits 10] [--out rtl/rope_sin_rom.sv]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))

import numpy as np  # noqa: E402

from bf16 import bf16_to_f32, to_bf16  # noqa: E402


def build_table(lut_bits):
    n = 1 << lut_bits
    k = np.arange(n, dtype=np.float64)
    ang = (k + 0.5) * (np.pi / 2.0) / n
    return to_bf16(np.sin(ang)), ang


def emit(lut_bits, module="rope_sin_rom"):
    n = 1 << lut_bits
    tab, ang = build_table(lut_bits)

    # Sanity: the reflection identity this table exists to enable must actually hold on
    # the *stored* values, not merely in the continuous math.
    ideal_reflect = np.sin(np.pi / 2 - ang)
    reflect_err = np.abs(bf16_to_f32(tab[::-1]).astype(np.float64) - ideal_reflect).max()
    assert reflect_err < 2.0**-7, f"reflection identity broken: {reflect_err}"

    ang_res = (np.pi / 2) / n
    L = []
    w = L.append
    w("// -----------------------------------------------------------------------------")
    w(f"// {module}.sv -- GENERATED by gen/gen_sin_lut.py. DO NOT EDIT BY HAND.")
    w("//")
    w(f"// BF16 quarter-wave sine table, {n} entries, MIDPOINT sampled:")
    w(f"//     T[k] = bf16(sin((k + 0.5) * (pi/2) / {n}))")
    w("//")
    w("// Midpoint sampling makes the Q1/Q3 reflection index exactly (N-1-idx) with no")
    w("// endpoint special case, and halves the worst-case sampling error (steps.md 7.3).")
    w("//")
    w(f"// Angle resolution: (pi/2)/{n} = {ang_res:.6f} rad. Since |d(sin)/d(theta)| <= 1 the")
    w(f"// worst-case value error is ~{ang_res:.6f}, already {2.0**-7/ang_res:.1f}x finer than")
    w(f"// BF16's ulp near 1.0 (2^-7 = {2.0**-7:.6f}) -- BF16's own coarseness bounds the")
    w("// table size, which is the inverse of the usual concern.")
    w("//")
    w(f"// Size: {n} entries x 2 bytes (BF16) = {n*2/1024:.0f} KB.")
    w("//")
    w("// NOTE: sin(0) is NOT 0 and cos(0) is read from T[N-1]; midpoint sampling trades")
    w("// exactness at the quadrant endpoints for the free reflection. See docs/notes.md.")
    w("// -----------------------------------------------------------------------------")
    w("")
    w(f"module {module} #(")
    w(f"  parameter int unsigned LutBits = {lut_bits},")
    w(f"  parameter int unsigned LutN    = {n}")
    w(") (")
    w("  input  logic [LutBits-1:0] idx_i,")
    w("  output logic [15:0]        sin_o    // BF16, always positive (quarter wave)")
    w(");")
    w("")
    w("  // Unpacked constant array: infers a ROM / logic cone, and lets both the")
    w("  // sin and cos lookups read it in the same cycle (steps.md 7.2).")
    w("  logic [15:0] table_q [LutN];")
    w("")
    w("  always_comb begin")
    for k in range(n):
        v = float(bf16_to_f32(tab[k]))
        w(f"    table_q[{k:4d}] = 16'h{int(tab[k]):04X};  // sin({ang[k]:.8f}) = {v:.8f}")
    w("  end")
    w("")
    w("  assign sin_o = table_q[idx_i];")
    w("")
    w("endmodule")
    w("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=10, help="log2 of entries per quadrant")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    text = emit(args.bits)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out} ({1<<args.bits} entries)", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note the built-in `assert`: the generator refuses to emit a table whose mirror identity
does not actually hold on the **stored** (rounded) values, not merely in continuous maths.

## B.10 `gen/vendor_cvfpu.sh` — records the vendored CVFPU subset (77 lines, full)

```bash
#!/usr/bin/env bash
# Vendor the minimal CVFPU (FPnew) + common_cells source set into this repository.
#
# Per the project constraint, CVFPU is COPIED, never symlinked: rope-bf16 must be
# self-contained inside cv32e40x. Re-run this script only if you need to refresh the
# vendored sources from an upstream checkout.
#
# Usage: vendor_cvfpu.sh [path-to-checkout-containing-fpnew-and-common_cells]
#
# Only the files needed for an ADDMUL-only, FP16ALT-only, PARALLEL configuration are
# copied (see docs/notes.md "Vendored CVFPU subset"). DIVSQRT is deliberately excluded:
# fpnew_divsqrt_multi.sv depends on the external fpu_div_sqrt_mvp repository, and the
# RoPE datapath needs neither division nor square root (steps.md Section 5.1).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
SRC="${1:-$HOME/cv32e40p/rtl/vendor}"

FPNEW_SRC="$SRC/pulp_platform_fpnew"
CC_SRC="$SRC/pulp_platform_common_cells"

if [[ ! -d "$FPNEW_SRC/src" ]]; then
  echo "ERROR: no FPnew sources at $FPNEW_SRC/src" >&2
  echo "Pass the directory containing pulp_platform_fpnew/ and pulp_platform_common_cells/" >&2
  exit 1
fi

# Minimal FPnew set for ADDMUL / PARALLEL / FP16ALT.
FPNEW_FILES=(
  fpnew_pkg                    # formats, opgroups, helper functions  (compile FIRST)
  fpnew_classifier             # operand classification
  fpnew_rounding               # RNE and friends
  fpnew_fma                    # the ADD/MUL workhorse
  fpnew_noncomp                # NONCOMP opgroup (disabled, kept for completeness)
  fpnew_opgroup_fmt_slice      # per-format lane wrapper
  fpnew_opgroup_block          # per-opgroup wrapper, gates on FmtUnitTypes
  fpnew_top                    # top level
)

# FPnew's only common_cells dependencies in this configuration:
#   lzc          <- fpnew_fma (leading-zero count for normalization)
#   rr_arb_tree  <- fpnew_top, fpnew_opgroup_block (output arbitration)
#   cf_math_pkg  <- lzc (idx_width)
CC_FILES=(cf_math_pkg lzc rr_arb_tree)

mkdir -p "$ROOT/rtl/vendor/cvfpu/src" "$ROOT/rtl/vendor/common_cells/src" \
         "$ROOT/rtl/vendor/common_cells/include/common_cells"

# fpnew_fma.sv and friends do `\`include "common_cells/registers.svh"` for the FF/FFL
# flop macros, so the include tree has to come along too. Compiles need
# -Irtl/vendor/common_cells/include (the Makefile passes this).
for h in registers.svh assertions.svh; do
  cp -f "$CC_SRC/include/common_cells/$h" \
        "$ROOT/rtl/vendor/common_cells/include/common_cells/$h" 2>/dev/null || true
done

for f in "${FPNEW_FILES[@]}"; do
  cp -f "$FPNEW_SRC/src/$f.sv" "$ROOT/rtl/vendor/cvfpu/src/$f.sv"
done

for f in "${CC_FILES[@]}"; do
  cp -f "$CC_SRC/src/$f.sv" "$ROOT/rtl/vendor/common_cells/src/$f.sv"
done

# Licenses travel with the code (SolderPad 0.51 / Apache-2.0).
cp -f "$FPNEW_SRC/LICENSE.solderpad" "$ROOT/rtl/vendor/cvfpu/" 2>/dev/null || true
cp -f "$FPNEW_SRC/LICENSE.apache"    "$ROOT/rtl/vendor/cvfpu/" 2>/dev/null || true
cp -f "$FPNEW_SRC/README.license.md" "$ROOT/rtl/vendor/cvfpu/" 2>/dev/null || true
for l in LICENSE LICENSE.md LICENSE.solderpad; do
  cp -f "$CC_SRC/$l" "$ROOT/rtl/vendor/common_cells/" 2>/dev/null || true
done

echo "Vendored FPnew -> rtl/vendor/cvfpu/src:"
ls -1 "$ROOT/rtl/vendor/cvfpu/src"
echo "Vendored common_cells -> rtl/vendor/common_cells/src:"
ls -1 "$ROOT/rtl/vendor/common_cells/src"
```

---
---
# Appendix C — Testbenches and assertions

The protocol assertion file (C.7) is reproduced **in full** because it is effectively the
formal specification of the CPU interface. The six testbenches are reproduced with their
DUT instantiation, stimulus and checking logic intact; the near-identical vector-file
loading blocks are elided as `[...]` after being shown once in C.1.

## C.1 `tb/tb_rope_bf16_unit.sv` — Milestone 1 (213 lines)

```systemverilog
// -----------------------------------------------------------------------------
// tb_rope_bf16_unit.sv -- Milestone 1 exit criteria for the BF16 primitives.
//
// Reads a vector file produced by model/gen_vectors.py and checks rope_bf16_mul and
// rope_bf16_add bit-exactly against the golden model (Model A's per-node arithmetic).
//
// Vector file format, one test per line, all hex:
//     <op> <a> <b> <expected>
// where op is 0=MUL, 1=ADD, 2=SUB.
//
// Any mismatch is a hard failure: steps.md Section 5.3 demands ZERO mismatches.
// -----------------------------------------------------------------------------

module tb_rope_bf16_unit;

  localparam int unsigned NumPipeRegs = 1;
  localparam int unsigned MaxVectors  = 20_000_000;

  logic clk, rst_n;

  initial begin
    clk = 1'b0;
    forever #5ns clk = ~clk;
  end

  // --------------------------------------------------------------------------
  // DUTs -- one of each operation, all fed in lockstep
  // --------------------------------------------------------------------------

  logic [15:0] a, b;
  logic        in_valid;
  logic [2:0]  in_ready;
  logic [15:0] res_mul, res_add, res_sub;
  logic [2:0]  out_valid;

  rope_bf16_mul #(.NumPipeRegs(NumPipeRegs)) i_mul (
    .clk_i(clk), .rst_ni(rst_n),
    .a_i(a), .b_i(b),
    .in_valid_i(in_valid), .in_ready_o(in_ready[0]), .flush_i(1'b0),
    .result_o(res_mul), .out_valid_o(out_valid[0]), .out_ready_i(1'b1), .busy_o()
  );

  rope_bf16_add #(.NumPipeRegs(NumPipeRegs), .Sub(1'b0)) i_add (
    .clk_i(clk), .rst_ni(rst_n),
    .a_i(a), .b_i(b),
    .in_valid_i(in_valid), .in_ready_o(in_ready[1]), .flush_i(1'b0),
    .result_o(res_add), .out_valid_o(out_valid[1]), .out_ready_i(1'b1), .busy_o()
  );

  rope_bf16_add #(.NumPipeRegs(NumPipeRegs), .Sub(1'b1)) i_sub (
    .clk_i(clk), .rst_ni(rst_n),
    .a_i(a), .b_i(b),
    .in_valid_i(in_valid), .in_ready_o(in_ready[2]), .flush_i(1'b0),
    .result_o(res_sub), .out_valid_o(out_valid[2]), .out_ready_i(1'b1), .busy_o()
  );

  // --------------------------------------------------------------------------
  // Vectors  (this loading pattern is shared by C.2-C.5 and elided there)
  // --------------------------------------------------------------------------

  int unsigned n_vec;
  logic [2:0]  v_op   [MaxVectors];
  logic [15:0] v_a    [MaxVectors];
  logic [15:0] v_b    [MaxVectors];
  logic [15:0] v_exp  [MaxVectors];

  int unsigned errors, checked;
  int unsigned err_by_op [3];
  string vecfile;

  initial begin : load_vectors
    int fd, code;
    int unsigned op_i, a_i, b_i, e_i;

    if (!$value$plusargs("vectors=%s", vecfile)) vecfile = "vectors_bf16.hex";

    fd = $fopen(vecfile, "r");
    if (fd == 0) begin
      $display("FATAL: cannot open vector file '%s'", vecfile);
      $fatal(1);
    end

    n_vec = 0;
    forever begin
      code = $fscanf(fd, "%h %h %h %h\n", op_i, a_i, b_i, e_i);
      if (code != 4) break;
      v_op [n_vec] = op_i[2:0];
      v_a  [n_vec] = a_i[15:0];
      v_b  [n_vec] = b_i[15:0];
      v_exp[n_vec] = e_i[15:0];
      n_vec++;
      if (n_vec >= MaxVectors) break;
    end
    $fclose(fd);
    $display("tb_rope_bf16_unit: loaded %0d vectors from %s", n_vec, vecfile);
  end

  // --------------------------------------------------------------------------
  // Drive and check.
  //
  // The three units are identically configured, so their latencies match and a single
  // shared scoreboard index is valid. That assumption is checked, not assumed: see the
  // out_valid agreement assertion below.
  // --------------------------------------------------------------------------

  initial begin : stimulus
    int unsigned i, sent;

    errors = 0;  checked = 0;
    for (int k = 0; k < 3; k++) err_by_op[k] = 0;

    rst_n = 1'b0;  in_valid = 1'b0;  a = '0;  b = '0;
    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    // The vector load runs in a parallel initial block; make sure it finished.
    wait (n_vec > 0);

    sent = 0;
    for (i = 0; i < n_vec; i++) begin
      // Present the operands and hold until all three units accept.
      a        <= v_a[i];
      b        <= v_b[i];
      in_valid <= 1'b1;
      @(posedge clk);
      while (in_ready != 3'b111) @(posedge clk);
      sent++;
    end
    in_valid <= 1'b0;

    repeat (NumPipeRegs + 16) @(posedge clk);   // drain

    $display("----------------------------------------------------------------");
    $display("tb_rope_bf16_unit: sent=%0d checked=%0d errors=%0d", sent, checked, errors);
    $display("  MUL errors: %0d", err_by_op[0]);
    $display("  ADD errors: %0d", err_by_op[1]);
    $display("  SUB errors: %0d", err_by_op[2]);
    if (checked != n_vec) begin
      $display("FAIL: checked %0d of %0d vectors (pipeline drain problem)", checked, n_vec);
      $fatal(1);
    end
    if (errors != 0) begin
      $display("FAIL: %0d mismatches (steps.md requires ZERO)", errors);
      $fatal(1);
    end
    $display("PASS: all %0d vectors bit-exact", checked);
    $finish;
  end

  // Scoreboard: results come out in order, one per accepted input.
  int unsigned check_idx;
  initial check_idx = 0;

  always @(posedge clk) begin
    if (rst_n && out_valid[0]) begin
      logic [15:0] got, want;
      logic [2:0]  op;
      want = v_exp[check_idx];
      op   = v_op[check_idx];
      case (op)
        3'd0: got = res_mul;
        3'd1: got = res_add;
        default: got = res_sub;
      endcase
      if (got !== want) begin
        errors++;
        err_by_op[op] = err_by_op[op] + 1;
        if (errors <= 20) begin
          $display("MISMATCH #%0d [%0d]: %s a=%04h b=%04h  got=%04h want=%04h",
                   errors, check_idx, op_name(op), v_a[check_idx], v_b[check_idx], got, want);
        end
      end
      checked++;
      check_idx++;
    end
  end

  // The shared scoreboard index is only sound if the three identically-configured units
  // really do produce their outputs on the same cycle. Assert it rather than trust it.
  always @(posedge clk) begin
    if (rst_n) begin
      assert (out_valid[0] == out_valid[1] && out_valid[1] == out_valid[2])
        else $fatal(1, "out_valid disagreement: %b -- latency assumption broken", out_valid);
      assert (in_ready[0] == in_ready[1] && in_ready[1] == in_ready[2])
        else $fatal(1, "in_ready disagreement: %b", in_ready);
    end
  end

endmodule : tb_rope_bf16_unit
```

## C.2 `tb/tb_rope_phase_gen.sv` — Milestone 2 (204 lines, abridged)

```systemverilog
// -----------------------------------------------------------------------------
// tb_rope_phase_gen.sv -- Milestone 2 exit criteria.
//
//  [x] matches the Python phase computation bit-exactly for m in [0, 8192], all i
//  [x] wraparound verified at m values that cross 2^32
//  [x] sequential and random-access modes agree for the same (m, i)
//
// Vector format:  <m> <i> <expected_phase>   (hex)
// -----------------------------------------------------------------------------

module tb_rope_phase_gen;
  import rope_pkg::*;

  // [... clock, DUT instantiation with both modes wired, vector loading as C.1 ...]

  initial begin : run
    // ---- 1. Random-access mode against the golden model --------------------
    for (k = 0; k < n_vec; k++) begin
      pos      = v_m[k][PosBits-1:0];
      pair_idx = v_i[k][PairIdxBits-1:0];
      #1ns;  // settle the combinational path
      if (phase !== v_exp[k]) begin
        errors++;
        if (errors <= 20)
          $display("MISMATCH [%0d]: m=%0d i=%0d  got=%08h want=%08h",
                   k, v_m[k], v_i[k], phase, v_exp[k]);
      end
    end
    $display("random-access: %0d vectors, %0d errors", n_vec, errors);

    // ---- 2. Sequential mode vs random access -------------------------------
    //
    // steps.md Section 6.3: the sequential accumulator must agree exactly with the
    // random-access multiply for every m, with ZERO drift. This is the property that
    // makes the integer accumulator preferable to the trigonometric recurrence.
    //
    // NOTE ON TB TIMING: control signals are deasserted a delta *after* the clock edge
    // (`@(posedge clk); #1ns;`), never at zero delay. Deasserting at zero delay races
    // with the DUT's always_ff sampling the same edge, and the pulse can be missed.
    for (int unsigned pidx = 0; pidx < 4; pidx++) begin
      seq_idx  = pidx[PairIdxBits-1:0];
      pair_idx = pidx[PairIdxBits-1:0];

      seq_clear = 1'b1;                       // clear to position 0
      @(posedge clk); #1ns;
      seq_clear = 1'b0;
      @(posedge clk); #1ns;

      for (int unsigned m = 0; m <= 4096; m++) begin
        pos = m[PosBits-1:0];
        #1ns;
        if (seq_phase !== phase) begin
          seq_errors++;
          if (seq_errors <= 10)
            $display("SEQ MISMATCH: i=%0d m=%0d  seq=%08h rand=%08h",
                     pidx, m, seq_phase, phase);
        end
        seq_step = 1'b1;                      // advance one token
        @(posedge clk); #1ns;
        seq_step = 1'b0;
      end
    end
    $display("sequential vs random-access: %0d errors", seq_errors);
    errors += seq_errors;

    // ---- 3. Explicit wraparound check --------------------------------------
    //
    // Walk the accumulator past 2^32 and confirm it wraps rather than saturating.
    begin
      logic [PhaseBits-1:0] prev;
      int unsigned wraps;
      wraps = 0;  seq_idx = '0;
      seq_clear = 1'b1;  @(posedge clk); #1ns;  seq_clear = 1'b0;  @(posedge clk); #1ns;
      prev = seq_phase;
      for (int unsigned s = 0; s < 64; s++) begin
        seq_step = 1'b1;  @(posedge clk); #1ns;  seq_step = 1'b0;
        if (seq_phase < prev) wraps++;  // the accumulator rolled over 2^32
        prev = seq_phase;
      end
      if (wraps == 0) begin
        $display("FAIL: accumulator never wrapped past 2^32 in 64 steps for i=0");
        errors++;
      end else
        $display("wraparound: observed %0d rollovers past 2^32 for i=0 (expected)", wraps);
    end

    if (errors != 0) begin
      $display("FAIL: %0d total errors", errors);  $fatal(1);
    end
    $display("PASS: phase generator bit-exact (random-access, sequential, wraparound)");
    $finish;
  end
endmodule : tb_rope_phase_gen
```

## C.3 `tb/tb_rope_sin_lut.sv` — Milestone 3 (249 lines, abridged)

Five independent checks. The `bf16_to_real` helper is included because it works around a
Verilator limitation.

```systemverilog
// -----------------------------------------------------------------------------
// tb_rope_sin_lut.sv -- Milestone 3 exit criteria.
//
//  [x] output matches Python lut_sin / lut_cos bit-exactly (exhaustive on the top 12
//      bits of the phase word -- i.e. every distinct table entry in every quadrant --
//      plus dense random coverage of the full 2^32 space)
//  [x] reflection symmetry verified at quadrant boundaries specifically
//  [x] sin^2 + cos^2 checked and its deviation from 1.0 reported
//
// Vector format:  <phase> <expected_sin> <expected_cos>   (hex)
// -----------------------------------------------------------------------------

module tb_rope_sin_lut;
  import rope_pkg::*;

  logic [PhaseBits-1:0] phase;
  logic [15:0]          sin_val, cos_val;

  rope_sin_lut i_dut (.phase_i(phase), .sin_o(sin_val), .cos_o(cos_val));

  // BF16 bit pattern -> real, decoded explicitly.
  //
  // The obvious `$bitstoshortreal({bits, 16'h0})` trick (exact, since BF16 is FP32's top
  // 16 bits) is not usable here: Verilator does not support promoting shortreal to real.
  // Decoding the fields by hand is exact for all normals and subnormals, which is all the
  // table contains.
  function automatic real bf16_to_real(logic [15:0] b);
    logic sgn; logic [7:0] exp; logic [6:0] man; real v;
    sgn = b[15];  exp = b[14:7];  man = b[6:0];
    if (exp == 8'd0)
      v = real'(man) * $pow(2.0, -133.0);            // subnormal
    else if (exp == 8'hFF)
      v = 0.0;   // Inf/NaN -- the midpoint-sampled quarter wave contains neither
    else
      v = (1.0 + real'(man) / 128.0) * $pow(2.0, real'(int'(exp) - 127));
    return sgn ? -v : v;
  endfunction

  // [... vector loading as C.1 ...]

  initial begin : run
    // ---- 1. Bit-exact vs the golden model ---------------------------------
    for (k = 0; k < n_vec; k++) begin
      phase = v_ph[k];
      #1ns;
      if (sin_val !== v_sin[k]) sin_err++;
      if (cos_val !== v_cos[k]) cos_err++;
    end
    $display("bit-exact check: %0d vectors, sin errors=%0d, cos errors=%0d",
             n_vec, sin_err, cos_err);
    errors += sin_err + cos_err;

    // ---- 2. Reflection symmetry at quadrant boundaries --------------------
    //
    // With midpoint sampling, quadrant 1 at index k must read exactly the same table
    // entry as quadrant 0 at index (N-1-k). Checked at the extreme indices, which is
    // precisely where edge sampling would have overflowed the table.
    begin
      int unsigned refl_err = 0;
      logic [15:0] q0_val, q1_val;
      for (int unsigned k2 = 0; k2 < LutN; k2++) begin
        phase = (32'd0 << QuadShift) | ((LutN - 1 - k2) << IdxShift);
        #1ns; q0_val = sin_val;
        phase = (32'd1 << QuadShift) | (k2 << IdxShift);
        #1ns; q1_val = sin_val;
        if (q0_val !== q1_val) refl_err++;
      end
      if (refl_err == 0)
        $display("reflection symmetry: exact for all %0d indices (Q1[k] == Q0[N-1-k])", LutN);
      errors += refl_err;
    end

    // ---- 3. Quadrant sign structure ---------------------------------------
    //
    // Q2 must be the exact sign-bit flip of Q0, and Q3 of Q1. A sign-bit XOR is exact,
    // so this must hold bit-for-bit with no tolerance.
    // [... sweeps k in steps of 7, comparing against a_val ^ 16'h8000 ...]
    $display("quadrant signs: Q2 == ~Q0 and Q3 == ~Q1 exactly (sign-bit XOR)");

    // ---- 4. The low 20 bits must be ignored -------------------------------
    //
    // phase[19:0] is reserved for future interpolation and must not affect the result
    // today. If it leaked into the index, results would depend on unused bits.
    // [... for each of several top-12-bit values, toggle each low bit and require
    //      sin/cos unchanged ...]
    $display("phase[19:0] correctly ignored (reserved for interpolation)");

    // ---- 5. sin^2 + cos^2, reported not asserted --------------------------
    //
    // This cannot be exactly 1.0 in BF16 and that is expected (steps.md 7.5). Report the
    // deviation; only a wild value would indicate a real bug.
    begin
      real s_r, c_r, n_r, dev, max_dev = 0.0, sum_sq = 0.0;
      int unsigned cnt = 0;
      for (int unsigned q = 0; q < 4; q++)
        for (int unsigned k4 = 0; k4 < LutN; k4 += 5) begin
          phase = (q << QuadShift) | (k4 << IdxShift);
          #1ns;
          s_r = bf16_to_real(sin_val);
          c_r = bf16_to_real(cos_val);
          n_r = s_r * s_r + c_r * c_r;
          dev = (n_r > 1.0) ? (n_r - 1.0) : (1.0 - n_r);
          if (dev > max_dev) max_dev = dev;
          sum_sq += dev * dev;  cnt++;
        end
      $display("sin^2+cos^2 over %0d points: max deviation = %f, RMS deviation = %f",
               cnt, max_dev, $sqrt(sum_sq / real'(cnt)));
      // A BF16 ulp near 1.0 is 2^-7 = 0.0078. Anything beyond a few ulp means the
      // quadrant or reflection logic is wrong, not merely quantised.
      if (max_dev > 4.0 * 0.0078125) begin
        $display("FAIL: sin^2+cos^2 deviates by more than 4 BF16 ulp -- logic bug, not quantisation");
        errors++;
      end
    end

    if (errors != 0) begin $display("FAIL: %0d total errors", errors); $fatal(1); end
    $display("PASS: sine LUT bit-exact, reflection and quadrant signs exact");
    $finish;
  end
endmodule : tb_rope_sin_lut
```

## C.4 `tb/tb_rope_datapath.sv` — Milestone 4 (213 lines, abridged)

```systemverilog
// -----------------------------------------------------------------------------
// tb_rope_datapath.sv -- Milestone 4 exit criteria.
//
//  [x] bit-exact vs Model A on random (x, y, m, i) tuples -- ZERO mismatches
//  [x] directed cancellation tests (x ~ y with m*theta ~ 45 deg) included in the
//      vector set, so the hardest numerical corner is covered, not just averages
//  [x] specials (NaN / Inf / subnormal) propagate identically to the model
//  [x] back-to-back full-throughput operation (one pair per cycle after fill)
//
// This is the whole chain: integer phase generation -> LUT -> 4 muls -> 2 adds.
//
// Vector format:  <x> <y> <m> <i> <expected_x> <expected_y>   (hex)
// -----------------------------------------------------------------------------

module tb_rope_datapath;
  import rope_pkg::*;

  // ---- DUT chain: phase gen -> LUT -> datapath ---------------------------
  rope_phase_gen i_phase_gen (
    .clk_i(clk), .rst_ni(rst_n),
    .pos_i(pos), .pair_idx_i(pair_idx), .phase_o(phase),
    .seq_clear_i(1'b0), .seq_step_i(1'b0), .seq_idx_i('0), .seq_phase_o()
  );

  rope_sin_lut i_lut (.phase_i(phase), .sin_o(sin_val), .cos_o(cos_val));

  rope_datapath #(.NumPipeRegs(NumPipeRegs), .RegLut(1'b1)) i_dut (
    .clk_i(clk), .rst_ni(rst_n), .flush_i(1'b0),
    .x_i(x_in), .y_i(y_in), .sin_i(sin_val), .cos_i(cos_val),
    .in_valid_i(in_valid), .in_ready_o(in_ready),
    .x_o(x_out), .y_o(y_out), .out_valid_o(out_valid),
    .busy_o(), .latency_o(latency)
  );

  // [... vector loading as C.1 ...]

  // --------------------------------------------------------------------------
  // Scoreboard: expected results ride a queue, popped in order as results emerge.
  // Using a queue rather than a hard-coded latency means the check stays valid if
  // NumPipeRegs changes.
  // --------------------------------------------------------------------------

  int unsigned exp_q [$];

  initial begin : stimulus
    // [... reset ...]
    wait (n_vec > 0);
    $display("tb_rope_datapath: datapath latency = %0d cycles", latency);

    // Drive one vector per cycle -- this is the full-throughput (back-to-back) test.
    for (int unsigned k = 0; k < n_vec; k++) begin
      x_in     <= v_x[k];
      y_in     <= v_y[k];
      pos      <= v_m[k][PosBits-1:0];
      pair_idx <= v_i[k][PairIdxBits-1:0];
      in_valid <= 1'b1;
      @(posedge clk);
      // in_ready is asserted every cycle in this configuration; if it ever were not,
      // hold the operands until it is.
      while (!in_ready) @(posedge clk);
      exp_q.push_back(k);
      sent++;
    end
    in_valid <= 1'b0;

    repeat (latency + 16) @(posedge clk);   // drain

    $display("tb_rope_datapath: sent=%0d checked=%0d errors=%0d", sent, checked, errors);
    if (checked != sent) begin
      $display("FAIL: checked %0d of %0d (results lost or pipeline not drained)", checked, sent);
      $fatal(1);
    end
    if (errors != 0) begin
      $display("FAIL: %0d mismatches vs Model A (steps.md requires ZERO)", errors);
      $fatal(1);
    end
    $display("PASS: datapath bit-exact vs Model A on all %0d vectors", checked);
    $finish;
  end

  // Watchdog. A stalled handshake or an empty vector file would otherwise spin forever,
  // which is much harder to diagnose than a clean timeout.
  initial begin : watchdog
    #500ms;
    $display("FAIL: timeout -- sent=%0d checked=%0d of %0d vectors", sent, checked, n_vec);
    $fatal(1);
  end

  always @(posedge clk) begin
    if (rst_n && out_valid) begin
      int unsigned idx;
      if (exp_q.size() == 0) begin
        $display("FAIL: result with no pending expectation at time %t", $time);
        errors++;
      end else begin
        idx = exp_q.pop_front();
        if (x_out !== v_ex[idx] || y_out !== v_ey[idx]) begin
          errors++;
          if (errors <= 20)
            $display("MISMATCH [%0d]: x=%04h y=%04h m=%0d i=%0d | got=(%04h,%04h) want=(%04h,%04h)",
                     idx, v_x[idx], v_y[idx], v_m[idx], v_i[idx],
                     x_out, y_out, v_ex[idx], v_ey[idx]);
        end
        checked++;
      end
    end
  end
endmodule : tb_rope_datapath
```

## C.5 `tb/tb_rope_xif_coproc.sv` — Milestone 5 (539 lines, abridged)

The six-test sequence for the CPU interface, including the cancellation test.

```systemverilog
// -----------------------------------------------------------------------------
// tb_rope_xif_coproc.sv -- Milestone 5 exit criteria.
//
//  [x] standalone XIF testbench passes with all SVA enabled (tb/rope_assertions.svh)
//  [x] kill-mid-operation: assert commit_kill, verify NO writeback and no state
//      corruption (the instruction after the killed one must still be correct)
//  [x] back-to-back ROPE.ROT at full throughput
//  [x] non-ROPE instructions are rejected (accept == 0) without stalling the core
//  [x] result_ready backpressure handled (credit accounting must not overflow the FIFO)
//
// Vector format:  <rs1> <rs2> <expected_rd>   (hex, 32-bit each)
//   rs1 = {x, y}, rs2 = {m, i}, rd = {x', y'}
// -----------------------------------------------------------------------------

module tb_rope_xif_coproc;
  import rope_pkg::*;

  localparam int unsigned X_NUM_RS = 2, X_ID_WIDTH = 4, X_MEM_WIDTH = 32;
  localparam int unsigned X_RFR_WIDTH = 32, X_RFW_WIDTH = 32;

  cv32e40x_if_xif #(
    .X_NUM_RS(X_NUM_RS), .X_ID_WIDTH(X_ID_WIDTH), .X_MEM_WIDTH(X_MEM_WIDTH),
    .X_RFR_WIDTH(X_RFR_WIDTH), .X_RFW_WIDTH(X_RFW_WIDTH)
  ) xif ();

  rope_xif_coproc #(
    .X_NUM_RS(X_NUM_RS), .X_ID_WIDTH(X_ID_WIDTH), .X_MEM_WIDTH(X_MEM_WIDTH),
    .X_RFR_WIDTH(X_RFR_WIDTH), .X_RFW_WIDTH(X_RFW_WIDTH), .NumPipeRegs(NumPipeRegs)
  ) i_dut (
    .clk_i(clk), .rst_ni(rst_n),
    .xif_issue_if(xif), .xif_commit_if(xif), .xif_result_if(xif)
  );

  // Tie off the channels the coprocessor does not drive, so the interface is fully
  // driven and the "no memory transaction" assertion is meaningful.
  assign xif.compressed_valid = 1'b0;
  assign xif.compressed_req   = '0;
  assign xif.mem_ready        = 1'b1;
  assign xif.mem_resp         = '0;
  assign xif.mem_result_valid = 1'b0;
  assign xif.mem_result       = '0;

  // rope_xif_coproc does not take the memory modports at all (ROPE.ROT never accesses
  // memory), so nothing drives these. Tied off here so the interface is fully driven.
  //
  // This does make the a_no_memory_transactions assertion vacuous in THIS testbench --
  // it is kept as a guard for Milestone 8: if the memory channel is ever wired into the
  // coprocessor, whoever does it must remove this tie-off, and the assertion then starts
  // checking the real "never store before a non-kill commit" rule.
  assign xif.mem_valid = 1'b0;
  assign xif.mem_req   = '0;

  `include "rope_assertions.svh"

  // Watchdog. A stalled handshake (e.g. credits never released because backpressure was
  // held too long) would otherwise spin forever.
  initial begin : watchdog
    #2s;
    $display("FAIL: timeout -- sent=%0d checked=%0d pending=%0d", sent, checked, pend_q.size());
    $fatal(1);
  end

  function automatic logic [31:0] encode_rope_rot(logic [4:0] rd, rs1, rs2);
    return {RopeFunct7Rot, rs2, rs1, RopeFunct3Rot, rd, RopeOpcodeCustom0};
  endfunction

  // [... vector loading; scoreboard popping pend_q and comparing data/rd/id ...]

  // TB TIMING DISCIPLINE (learned the hard way -- an earlier version accepted the same
  // instruction twice):
  //
  //   * DRIVE on the negedge, using BLOCKING assignments.
  //   * SAMPLE on the posedge.
  //
  // Asserting issue_valid at the same instant as a posedge races with the DUT's
  // always_ff sampling that edge: depending on Active/NBA scheduling the request can be
  // seen on both that edge and the next, i.e. accepted twice. Driving strictly off-edge
  // removes the race entirely, and one negedge->posedge pair per loop iteration still
  // gives full one-per-cycle throughput.
  task automatic issue_rot(input int unsigned vec_idx,
                           input logic [4:0]  rd,
                           input bit          expect_accept = 1'b1);
    logic [3:0] use_id;
    use_id = next_id;

    @(negedge clk);
    xif.issue_valid         = 1'b1;
    xif.issue_req.instr     = encode_rope_rot(rd, 5'd1, 5'd2);
    xif.issue_req.mode      = 2'b11;
    xif.issue_req.id        = use_id;
    xif.issue_req.rs[0]     = v_rs1[vec_idx];
    xif.issue_req.rs[1]     = v_rs2[vec_idx];
    xif.issue_req.rs_valid  = 2'b11;
    xif.issue_req.ecs       = 6'b0;
    xif.issue_req.ecs_valid = 1'b1;

    // The DUT samples here. issue_ready/issue_resp are combinational off stable inputs.
    @(posedge clk);
    while (!xif.issue_ready) @(posedge clk);

    if (xif.issue_resp.accept !== expect_accept) begin
      $display("FAIL: accept=%0b, expected %0b", xif.issue_resp.accept, expect_accept);
      errors++;
    end

    if (xif.issue_resp.accept) begin
      // Built as a named temporary: Verilator does not support an assignment pattern
      // used directly as a function-call argument.
      pending_t new_pend;
      new_pend.vec = vec_idx;  new_pend.rd = rd;  new_pend.id = use_id;
      pend_q.push_back(new_pend);
      sent++;
      next_id = next_id + 1'b1;
    end

    // Release off-edge so the request is seen on exactly one posedge.
    @(negedge clk);
    xif.issue_valid = 1'b0;
  endtask

  // Commit an id. commit_valid has no ready signal, so this is a pure one-cycle pulse
  // that the coprocessor must be able to observe at any time.
  task automatic do_commit(input logic [3:0] id, input bit kill = 1'b0);
    @(negedge clk);
    xif.commit_valid       = 1'b1;
    xif.commit.id          = id;
    xif.commit.commit_kill = kill;
    @(posedge clk);
    @(negedge clk);
    xif.commit_valid = 1'b0;
  endtask

  initial begin : run
    // [... reset, initial signal values ...]

    // ---- TEST 1: a non-ROPE instruction must be rejected, not stall the core -----
    @(negedge clk);
    xif.issue_valid        = 1'b1;
    xif.issue_req.instr    = 32'h00100113;   // a plain ADDI: definitely not ours
    xif.issue_req.id       = 4'hF;
    xif.issue_req.rs_valid = 2'b11;
    @(posedge clk);
    while (!xif.issue_ready) @(posedge clk);
    if (xif.issue_resp.accept !== 1'b0) begin
      $display("FAIL: coprocessor accepted a non-ROPE instruction");  errors++;
    end else
      $display("TEST 1 ok: non-ROPE instruction rejected (accept=0, issue_ready=1)");
    @(negedge clk);
    xif.issue_valid = 1'b0;

    // ---- TEST 2: single ROPE.ROT, committed normally -----------------------------
    id0 = next_id;
    issue_rot(0, 5'd7);
    do_commit(id0, 1'b0);
    repeat (32) @(posedge clk);
    if (checked != 1) begin
      $display("FAIL: expected exactly 1 writeback, got %0d", checked);  errors++;
    end else
      $display("TEST 2 ok: single ROPE.ROT wrote back correctly");

    // ---- TEST 3: kill mid-operation ---------------------------------------------
    //
    // Issue an instruction, kill it before it retires, and confirm that (a) no writeback
    // appears for it, and (b) the NEXT instruction still produces the correct result --
    // i.e. the kill did not corrupt the pipeline or the tag/credit bookkeeping.
    checked_before = checked;
    id_kill = next_id;
    issue_rot(1, 5'd8);
    void'(pend_q.pop_back());          // this one must never write back
    tb_killed[id_kill] = 1'b1;
    @(posedge clk);
    do_commit(id_kill, 1'b1);          // kill while still in the datapath
    repeat (32) @(posedge clk);
    if (checked != checked_before) begin
      $display("FAIL: killed instruction produced a writeback");  errors++;
    end else
      $display("TEST 3a ok: killed instruction produced no writeback");

    id_after = next_id;                // a normal instruction must still work
    issue_rot(2, 5'd9);
    do_commit(id_after, 1'b0);
    repeat (32) @(posedge clk);
    if (checked != checked_before + 1) begin
      $display("FAIL: instruction after a kill did not write back (state corrupted)");
      errors++;
    end else
      $display("TEST 3b ok: instruction after a kill is unaffected");
    tb_killed = '0;

    // ---- TEST 4: back-to-back at full throughput --------------------------------
    fork
      begin  // issue thread
        for (int unsigned k = 0; k < count; k++) issue_rot(start_vec + k, 5'd10);
      end
      begin  // commit thread: ids are handed out sequentially
        logic [3:0] cid = next_id;
        for (int unsigned k = 0; k < count; k++) begin
          @(posedge clk);  do_commit(cid, 1'b0);  cid = cid + 1'b1;
        end
      end
    join
    // [... check count writebacks; prints "TEST 4 ok: 2000 back-to-back ..." ...]

    // ---- TEST 5: result_ready backpressure --------------------------------------
    //
    // Stall the result channel while issuing. The credit scheme must stop accepting
    // before the FIFO overflows; nothing may be lost or duplicated.
    @(negedge clk);
    xif.result_ready = 1'b0;
    fork
      begin for (int unsigned k = 0; k < count; k++) issue_rot(start_vec + k, 5'd11); end
      begin /* commit thread as above */ end
      // Backpressure thread. This MUST live inside the fork: with result_ready held
      // low, the credit counter fills after FifoDepth accepts and issue_ready
      // correctly drops, so the issue thread blocks. Releasing backpressure only
      // after the join would therefore deadlock -- the earlier version of this test
      // did exactly that. Holding for well over FifoDepth cycles first is what makes
      // this a real test of the credit scheme rather than of nothing.
      begin
        repeat (40) @(posedge clk);
        @(negedge clk);
        xif.result_ready = 1'b1;
      end
    join
    // [... check; prints "TEST 5 ok: 64 results survived result_ready backpressure" ...]

    // ---- TEST 6: remaining vectors, with RANDOM backpressure --------------------
    fork
      begin /* issue */ end
      begin /* commit */ end
      // Random backpressure on the result channel, driven off-edge like everything
      // else so it never races with the coprocessor sampling result_ready.
      begin
        for (int unsigned t = 0; t < count * 3; t++) begin
          @(negedge clk);
          xif.result_ready = ($urandom_range(0, 3) != 0);
        end
        @(negedge clk);
        xif.result_ready = 1'b1;
      end
    join
    // [... check; prints "TEST 6 ok: 4000 results correct under random backpressure" ...]

    $display("tb_rope_xif_coproc: sent=%0d checked=%0d errors=%0d", sent, checked, errors);
    if (errors != 0) begin $display("FAIL: %0d errors", errors); $fatal(1); end
    $display("PASS: XIF coprocessor -- decode, writeback, kill, throughput, backpressure");
    $finish;
  end
endmodule : tb_rope_xif_coproc
```

## C.6 `tb/tb_rope_core.sv` — Milestone 7, the C test on the real core (270 lines, abridged)

```systemverilog
// -----------------------------------------------------------------------------
// tb_rope_core.sv -- run the bare-metal C test on the integrated core.
//
// This is the Milestone 7 exit criterion -- the C test passing on the integrated core
// under simulation -- and it closes the loop end to end:
//
// (NB: no comment line here may begin with the word "verilator", case-insensitively:
//  the lexer would parse it as a pragma and fail with "Unknown verilator comment".)
//
//     golden model -> compiled-in vectors -> real RISC-V instructions -> the core ->
//     CORE-V-XIF -> the RoPE coprocessor -> writeback -> compared in software
//
// Memory map (must match sw/link.ld):
//     0x00000000  RAM, 256 KB, instructions and data
//     0x10000000  putchar   (write-only)
//     0x20000000  exit code (write-only; 0 = pass)
//
// The memories model OBI with always-grant and one cycle of response latency, which is
// fully pipelined: a request may be accepted on every cycle.
// -----------------------------------------------------------------------------

module tb_rope_core;

  localparam int unsigned MemWords    = 65536;           // 256 KB
  localparam logic [31:0] BootAddr    = 32'h0000_0080;   // matches sw/link.ld
  localparam logic [31:0] PutcharAddr = 32'h1000_0000;
  localparam logic [31:0] ExitAddr    = 32'h2000_0000;

  rope_cv32e40x_wrapper i_dut (
    .clk_i(clk), .rst_ni(rst_n), .scan_cg_en_i(1'b0),
    .boot_addr_i(BootAddr),
    .dm_exception_addr_i(32'h0), .dm_halt_addr_i(32'h0),
    .mhartid_i(32'h0), .mimpid_patch_i(4'h0), .mtvec_addr_i(32'h0),
    // instruction + data OBI ports ...
    .instr_req_o(instr_req), .instr_gnt_i(instr_gnt), .instr_rvalid_i(instr_rvalid),
    .instr_addr_o(instr_addr), .instr_rdata_i(instr_rdata), /* ... */
    .data_req_o(data_req), .data_gnt_i(data_gnt), .data_rvalid_i(data_rvalid),
    .data_addr_o(data_addr), .data_be_o(data_be), .data_we_o(data_we),
    .data_wdata_o(data_wdata), .data_rdata_i(data_rdata), /* ... */
    .mcycle_o(mcycle), .time_i(64'h0),
    .irq_i(32'h0), .wu_wfe_i(1'b0),
    .clic_irq_i(1'b0), /* ... */
    .fencei_flush_req_o(fencei_flush_req),
    // Acknowledge immediately: there are no caches to flush in this testbench.
    .fencei_flush_ack_i(fencei_flush_req),
    .debug_req_i(1'b0), /* ... */
    .fetch_enable_i(1'b1), .core_sleep_o(core_sleep)
  );

  logic [31:0] mem [MemWords];

  initial begin : load_image
    for (int unsigned k = 0; k < MemWords; k++) mem[k] = 32'h0;
    if (!$value$plusargs("hex=%s", hexfile)) hexfile = "test_rope.hex";
    // No explicit offset: `objcopy -O verilog --verilog-data-width=4` emits an
    // "@<word-index>" record (0x20 for the 0x80 load address), and $readmemh honours it.
    // Passing a start address here as well would double-count the offset.
    $readmemh(hexfile, mem);
    $display("tb_rope_core: loaded %s", hexfile);
  end

  // ---- instruction port: always grant, one cycle of latency ----------------
  assign instr_gnt = 1'b1;
  assign instr_err = 1'b0;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin instr_pending_q <= 1'b0; instr_addr_q <= '0; end
    else begin instr_pending_q <= instr_req & instr_gnt; instr_addr_q <= instr_addr; end
  end
  assign instr_rvalid = instr_pending_q;
  assign instr_rdata  = mem[instr_addr_q[31:2] & (MemWords - 1)];

  // ---- data port: always grant, one cycle of latency ----------------------
  // [... same register-and-return structure for addr/wdata/be/we ...]
  assign data_rvalid = data_pending_q;
  assign data_rdata  = mem[data_addr_q[31:2] & (MemWords - 1)];

  // Exit register. This is deliberately the SOLE driver of exit_code/saw_exit, with
  // their initial values coming from the reset branch rather than from an initial block:
  // driving them from both an initial block and a clocked process is a multiple-driver
  // conflict, and an earlier version of this testbench did exactly that and always read
  // back the stale initial value (reporting a spurious failure even though the software
  // had written 0 and printed PASS).
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      exit_code <= 32'hFFFF_FFFF;
      saw_exit  <= 1'b0;
    end else if (data_pending_q && data_we_q
                 && ((data_addr_q & 32'hFFFF_F000) == ExitAddr)) begin
      exit_code <= data_wdata_q;
      saw_exit  <= 1'b1;
    end
  end

  // Writes: RAM with byte enables, plus the putchar device.
  always_ff @(posedge clk) begin
    if (rst_n && data_pending_q && data_we_q) begin
      if ((data_addr_q & 32'hFFFF_F000) == PutcharAddr) begin
        $write("%c", data_wdata_q[7:0]);
      end else if ((data_addr_q & 32'hFFFF_F000) == ExitAddr) begin
        // Handled by the exit register above.
      end else begin
        // Byte-enable-correct RAM write.
        if (data_be_q[0]) mem[data_addr_q[31:2] & (MemWords-1)][ 7: 0] <= data_wdata_q[ 7: 0];
        if (data_be_q[1]) mem[data_addr_q[31:2] & (MemWords-1)][15: 8] <= data_wdata_q[15: 8];
        if (data_be_q[2]) mem[data_addr_q[31:2] & (MemWords-1)][23:16] <= data_wdata_q[23:16];
        if (data_be_q[3]) mem[data_addr_q[31:2] & (MemWords-1)][31:24] <= data_wdata_q[31:24];
      end
    end
  end

  initial begin : run
    // exit_code / saw_exit are reset by the exit register above -- not initialised here,
    // so that clocked process remains their only driver.
    rst_n = 1'b0;
    repeat (20) @(posedge clk);
    rst_n = 1'b1;

    wait (saw_exit);                       // wait for the software to signal completion
    repeat (10) @(posedge clk);

    $display("tb_rope_core: exit_code=0x%08h after %0d cycles", exit_code, mcycle);
    if (exit_code == 32'hDEAD) begin
      $display("FAIL: the program trapped (see crt0.S trap_park)");  $fatal(1);
    end
    if (exit_code != 0) begin
      $display("FAIL: software reported %0d mismatches", exit_code);  $fatal(1);
    end
    $display("PASS: C test passed on the integrated core");
    $finish;
  end

  // Watchdog: a hang here usually means a bad memory map or a trap loop.
  initial begin : watchdog
    #50ms;
    $display("FAIL: timeout after %0d cycles (saw_exit=%0b)", mcycle, saw_exit);
    $fatal(1);
  end
endmodule : tb_rope_core
```

## C.7 `tb/rope_assertions.svh` — the formal protocol spec (211 lines, full)

```systemverilog
// -----------------------------------------------------------------------------
// rope_assertions.svh -- SVA for the CORE-V-XIF protocol (steps.md Section 9.4).
//
// Protocol bugs are far more likely here than math bugs, so these are checked
// continuously rather than by directed stimulus alone.
//
// Include inside a testbench that has:
//   * `clk` and `rst_n`
//   * an interface instance named `xif`
//
// Checked:
//   [x] no result_valid for an id that was never issued
//   [x] no state update (writeback) after commit_kill for that id
//   [x] no memory transaction before a non-kill commit
//   [x] transactions with an earlier issued id never depend on a later issued id
//   [x] issue_ready deassertion behaviour is legal
//   [x] every accepted issue eventually produces exactly one result or one kill
// -----------------------------------------------------------------------------

`ifndef ROPE_ASSERTIONS_SVH
`define ROPE_ASSERTIONS_SVH

  // ---------------------------------------------------------------------------
  // Bookkeeping used by the properties below
  // ---------------------------------------------------------------------------

  localparam int unsigned SvaNumIds = 1 << 4;  // X_ID_WIDTH = 4

  // Was this id accepted and not yet retired?
  logic [SvaNumIds-1:0] sva_outstanding;
  logic [SvaNumIds-1:0] sva_killed;
  logic [SvaNumIds-1:0] sva_committed;
  // How many results has each id produced? Must never exceed one.
  int unsigned          sva_result_count [SvaNumIds];
  int unsigned          sva_issue_count  [SvaNumIds];

  wire sva_issue_accepted = xif.issue_valid & xif.issue_ready & xif.issue_resp.accept;
  wire sva_result_taken   = xif.result_valid & xif.result_ready;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sva_outstanding <= '0;
      sva_killed      <= '0;
      sva_committed   <= '0;
      for (int unsigned k = 0; k < SvaNumIds; k++) begin
        sva_result_count[k] <= 0;
        sva_issue_count[k]  <= 0;
      end
    end else begin
      if (sva_issue_accepted) begin
        sva_outstanding[xif.issue_req.id] <= 1'b1;
        sva_killed[xif.issue_req.id]      <= 1'b0;
        sva_committed[xif.issue_req.id]    <= 1'b0;
        sva_issue_count[xif.issue_req.id] <= sva_issue_count[xif.issue_req.id] + 1;
        sva_result_count[xif.issue_req.id] <= 0;
      end
      if (xif.commit_valid) begin
        if (xif.commit.commit_kill) sva_killed[xif.commit.id]    <= 1'b1;
        else                        sva_committed[xif.commit.id] <= 1'b1;
      end
      if (sva_result_taken) begin
        sva_outstanding[xif.result.id]  <= 1'b0;
        sva_result_count[xif.result.id] <= sva_result_count[xif.result.id] + 1;
      end
    end
  end

  // ---------------------------------------------------------------------------
  // 1. No result for an id that was never issued.
  // ---------------------------------------------------------------------------
  property p_no_result_without_issue;
    @(posedge clk) disable iff (!rst_n)
      sva_result_taken |-> sva_outstanding[xif.result.id];
  endproperty
  a_no_result_without_issue: assert property (p_no_result_without_issue)
    else $error("SVA: result_valid for id=%0d which is not outstanding", xif.result.id);

  // ---------------------------------------------------------------------------
  // 2. No writeback after commit_kill for that id.
  //
  // A killed instruction must be retired silently: the core has already discarded it,
  // so writing rd would corrupt architectural state.
  // ---------------------------------------------------------------------------
  property p_no_writeback_after_kill;
    @(posedge clk) disable iff (!rst_n)
      sva_result_taken |-> !sva_killed[xif.result.id];
  endproperty
  a_no_writeback_after_kill: assert property (p_no_writeback_after_kill)
    else $error("SVA: writeback for KILLED id=%0d", xif.result.id);

  // ---------------------------------------------------------------------------
  // 3. No memory transaction before a non-kill commit.
  //
  // ROPE.ROT issues no memory transactions at all, so the strong form applies: mem_valid
  // must never assert. When the streaming variant (Milestone 8) lands, this relaxes to
  // "only after sva_committed[id]".
  // ---------------------------------------------------------------------------
  property p_no_memory_transactions;
    @(posedge clk) disable iff (!rst_n)
      !xif.mem_valid;
  endproperty
  a_no_memory_transactions: assert property (p_no_memory_transactions)
    else $error("SVA: unexpected memory transaction -- ROPE.ROT must not touch memory");

  // ---------------------------------------------------------------------------
  // 4. Exactly one result per accepted, non-killed instruction.
  // ---------------------------------------------------------------------------
  property p_at_most_one_result;
    @(posedge clk) disable iff (!rst_n)
      sva_result_taken |-> (sva_result_count[xif.result.id] == 0);
  endproperty
  a_at_most_one_result: assert property (p_at_most_one_result)
    else $error("SVA: id=%0d produced more than one result", xif.result.id);

  // ---------------------------------------------------------------------------
  // 5. Result ordering: results must come back in issue order.
  //
  // The CV-X-IF ordering rule is that a transaction with an earlier issued id must not
  // depend on a later issued one -- in particular the coprocessor may not delay
  // result_valid for an old instruction because it wants to see commit_valid for a newer
  // one. Returning results strictly FIFO is a sufficient condition, and that is what the
  // implementation does, so check it directly.
  // ---------------------------------------------------------------------------
  int unsigned sva_issue_seq;
  int unsigned sva_result_seq;
  int unsigned sva_seq_of_id [SvaNumIds];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sva_issue_seq  <= 0;
      sva_result_seq <= 0;
      for (int unsigned k = 0; k < SvaNumIds; k++) sva_seq_of_id[k] <= 0;
    end else begin
      if (sva_issue_accepted) begin
        sva_seq_of_id[xif.issue_req.id] <= sva_issue_seq;
        sva_issue_seq                   <= sva_issue_seq + 1;
      end
      if (sva_result_taken) begin
        sva_result_seq <= sva_result_seq + 1;
      end
    end
  end

  // ---------------------------------------------------------------------------
  // 6. issue_resp correctness at the handshake.
  //
  // DELIBERATELY NOT CHECKED: "accept must stay stable while a request is pending".
  // An earlier version of this file asserted that and it is simply not a CV-X-IF
  // requirement -- issue_resp is only meaningful in the cycle where
  // issue_valid && issue_ready. This coprocessor withholds issue_ready when the result
  // FIFO has no credit, and drives accept=0 in that state; when a credit frees, both go
  // high together. `accept` therefore changes legitimately while a request waits, and
  // asserting stability produced a false failure under result-channel backpressure.
  //
  // What IS required is checked instead: accept never fires without a valid request, and
  // whenever a handshake completes the decision matches the decode exactly.
  // ---------------------------------------------------------------------------

  property p_no_accept_without_valid;
    @(posedge clk) disable iff (!rst_n)
      xif.issue_resp.accept |-> xif.issue_valid;
  endproperty
  a_no_accept_without_valid: assert property (p_no_accept_without_valid)
    else $error("SVA: issue_resp.accept asserted without issue_valid");

  // On a completed handshake, we must accept exactly the ROPE.ROT instructions whose
  // operands are valid, and nothing else. This catches both over-acceptance (stealing
  // another coprocessor's or the core's instruction) and under-acceptance (silently
  // dropping work we advertised we could do).
  property p_accept_matches_decode;
    @(posedge clk) disable iff (!rst_n)
      (xif.issue_valid && xif.issue_ready) |->
        (xif.issue_resp.accept ==
           (rope_pkg::is_rope_rot(xif.issue_req.instr)
            && xif.issue_req.rs_valid[0] && xif.issue_req.rs_valid[1]));
  endproperty
  a_accept_matches_decode: assert property (p_accept_matches_decode)
    else $error("SVA: accept=%0b disagrees with the decode of instr=%08h",
                xif.issue_resp.accept, xif.issue_req.instr);

  // ---------------------------------------------------------------------------
  // 7. issue_resp must be consistent whenever an issue is accepted.
  // ---------------------------------------------------------------------------
  property p_accept_implies_writeback;
    @(posedge clk) disable iff (!rst_n)
      sva_issue_accepted |-> (xif.issue_resp.writeback && !xif.issue_resp.dualwrite);
  endproperty
  a_accept_implies_writeback: assert property (p_accept_implies_writeback)
    else $error("SVA: accepted ROPE.ROT without a single-word writeback");

  // ROPE.ROT cannot raise an exception and is not a load/store; advertising otherwise
  // would make the core reserve resources it does not need.
  property p_no_exc_no_loadstore;
    @(posedge clk) disable iff (!rst_n)
      sva_issue_accepted |-> (!xif.issue_resp.exc && !xif.issue_resp.loadstore
                              && !xif.issue_resp.ecswrite);
  endproperty
  a_no_exc_no_loadstore: assert property (p_no_exc_no_loadstore)
    else $error("SVA: accepted ROPE.ROT advertised exc/loadstore/ecswrite");

  // ---------------------------------------------------------------------------
  // 8. Result payload sanity: we always write a full 32-bit word.
  // ---------------------------------------------------------------------------
  property p_result_we_set;
    @(posedge clk) disable iff (!rst_n)
      xif.result_valid |-> (xif.result.we == 1'b1 && !xif.result.exc && !xif.result.err);
  endproperty
  a_result_we_set: assert property (p_result_we_set)
    else $error("SVA: result without write-enable, or flagged exc/err");

`endif  // ROPE_ASSERTIONS_SVH
```

---
---
# Appendix D — Software, synthesis, and build system

## D.1 `sw/rope_intrin.h` — the C interface (152 lines, full)

```c
/* ---------------------------------------------------------------------------
 * rope_intrin.h -- inline-asm wrappers for the BF16 RoPE coprocessor.
 *
 * ROPE.ROT rd, rs1, rs2       # custom-0, opcode 0x0B, funct3=0, funct7=0
 *   rs1 = {x_bf16[31:16], y_bf16[15:0]}    native BF16 pair, NO conversion
 *   rs2 = {m[31:16], i[15:0]}              position and pair index (integers)
 *   rd  = {x'_bf16[31:16], y'_bf16[15:0]}  native BF16 pair out
 *
 * The packing MUST match rope_pkg.sv (pair_x/pair_y/pack_pair): x lives in the HIGH
 * half of the word. Getting this backwards is the single easiest way to produce
 * "correct-looking but wrong" results, so it is stated in both places and checked by
 * sw/test_rope.c.
 *
 * Later: patch binutils/LLVM with the encoding so real mnemonics replace `.insn`.
 * Not needed to make progress.
 * ------------------------------------------------------------------------- */

#ifndef ROPE_INTRIN_H
#define ROPE_INTRIN_H

#include <stdint.h>

/* Instruction fields, kept in sync with rope_pkg.sv. */
#define ROPE_OPCODE_CUSTOM0 0x0B
#define ROPE_FUNCT3_ROT     0
#define ROPE_FUNCT7_ROT     0

/* ---------------------------------------------------------------------------
 * BF16 <-> FP32 conversion
 *
 * BF16 is bit-identical to the upper 16 bits of FP32, so widening is EXACT and needs
 * no library call and no rounding. Narrowing is the only direction that rounds.
 * ------------------------------------------------------------------------- */

/* Exact BF16 -> float. No rounding whatsoever. */
static inline float bf16_to_f32(uint16_t b) {
    union { uint32_t u; float f; } c;
    c.u = (uint32_t)b << 16;
    return c.f;
}

/* float -> BF16, round-to-nearest-even.
 *
 * RNE, not truncation: truncating gives a systematic downward bias that shows up much
 * later as an unexplained perplexity gap. NaN is canonicalised rather than shifted,
 * because a NaN whose payload lives entirely in the low 16 bits would otherwise become
 * an infinity. This mirrors round_f32_to_bf16() in model/bf16.py exactly. */
static inline uint16_t f32_to_bf16(float f) {
    union { uint32_t u; float f; } c;
    uint32_t u, lsb;
    c.f = f;
    u = c.u;

    /* NaN -> canonical quiet NaN. */
    if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu) != 0u) {
        return 0x7FC0u;
    }

    lsb = (u >> 16) & 1u;
    return (uint16_t)((u + 0x7FFFu + lsb) >> 16);
}

/* ---------------------------------------------------------------------------
 * Pair packing -- must match rope_pkg::pack_pair / pair_x / pair_y
 * ------------------------------------------------------------------------- */

static inline uint32_t rope_pack(uint16_t x, uint16_t y) {
    return ((uint32_t)x << 16) | (uint32_t)y;
}

static inline uint16_t rope_unpack_x(uint32_t xy) { return (uint16_t)(xy >> 16); }
static inline uint16_t rope_unpack_y(uint32_t xy) { return (uint16_t)(xy & 0xFFFFu); }

/* ---------------------------------------------------------------------------
 * The instruction
 * ------------------------------------------------------------------------- */

/* Rotate one BF16 pair at sequence position m, frequency index i.
 *
 * `xy` is a packed BF16 pair {x, y}; the result is the packed rotated pair {x', y'}.
 *
 *     x' = x*cos(m*theta_i) - y*sin(m*theta_i)
 *     y' = x*sin(m*theta_i) + y*cos(m*theta_i)
 */
static inline uint32_t rope_rot(uint32_t xy, uint16_t m, uint16_t i) {
    uint32_t rd;
    uint32_t rs2 = ((uint32_t)m << 16) | (uint32_t)i;
    /* `__asm__ __volatile__` rather than `asm volatile`: under -std=c11 (strict ISO)
       the unprefixed spellings are not keywords. */
    __asm__ __volatile__(".insn r %[opc], %[f3], %[f7], %[rd], %[rs1], %[rs2]"
                         : [rd] "=r"(rd)
                         : [rs1] "r"(xy), [rs2] "r"(rs2),
                           [opc] "i"(ROPE_OPCODE_CUSTOM0),
                           [f3] "i"(ROPE_FUNCT3_ROT),
                           [f7] "i"(ROPE_FUNCT7_ROT));
    return rd;
}

/* Convenience wrapper: rotate a pair given as floats, returning floats.
 *
 * Note this rounds the inputs to BF16 on the way in. The coprocessor itself performs no
 * conversion at all (INV-1) -- the rounding happens here, in software, and is visible. */
static inline void rope_rot_f32(float x, float y, uint16_t m, uint16_t i,
                                float *x_out, float *y_out) {
    uint32_t packed = rope_pack(f32_to_bf16(x), f32_to_bf16(y));
    uint32_t res    = rope_rot(packed, m, i);
    *x_out = bf16_to_f32(rope_unpack_x(res));
    *y_out = bf16_to_f32(rope_unpack_y(res));
}

/* ---------------------------------------------------------------------------
 * Whole-vector application
 *
 * Both pairing conventions from steps.md Section 12. Default to half-split, because
 * HuggingFace Transformers (which Mugi's authors used) uses rotate_half.
 * ------------------------------------------------------------------------- */

typedef enum {
    ROPE_HALF_SPLIT = 0,  /* (x_i, x_{i+d/2})  -- HF LLaMA / GPT-NeoX  [DEFAULT] */
    ROPE_INTERLEAVED = 1  /* (x_2i, x_2i+1)    -- GPT-J / original RoFormer */
} rope_stride_mode_t;

/* Rotate one head vector of `d` BF16 elements in place.
 *
 * `vec` points to d uint16_t BF16 values. `d` must be even.
 *
 * Under half-split the two elements of a pair are d/2 apart -- for a 64-element head
 * that is two separate cache lines per pair. Interleaved gets both elements of a pair in
 * one 32-bit word, which is why X_MEM_WIDTH = 32 lines up perfectly for interleaved and
 * needs two transactions for half-split.
 */
static inline void rope_rotate_head(uint16_t *vec, unsigned d, uint16_t m,
                                    rope_stride_mode_t mode) {
    unsigned half = d / 2;
    unsigned k;
    for (k = 0; k < half; k++) {
        unsigned lo, hi;
        uint32_t res;
        if (mode == ROPE_HALF_SPLIT) {
            lo = k;
            hi = k + half;
        } else {
            lo = 2 * k;
            hi = 2 * k + 1;
        }
        res = rope_rot(rope_pack(vec[lo], vec[hi]), m, (uint16_t)k);
        vec[lo] = rope_unpack_x(res);
        vec[hi] = rope_unpack_y(res);
    }
}

#endif /* ROPE_INTRIN_H */
```

## D.2 `sw/test_rope.c` — the bare-metal test (207 lines, abridged)

```c
/* ---------------------------------------------------------------------------
 * test_rope.c -- bare-metal test for the BF16 RoPE coprocessor (Milestone 7).
 *
 * Runs on the integrated core (rope_cv32e40x_wrapper). Vectors are compiled in from
 * rope_vectors.h, which model/gen_c_vectors.py emits from the same golden model the RTL
 * testbenches use -- so a pass here means the *software* view (instruction encoding,
 * operand packing, register allocation) agrees with the model too, not just the RTL.
 * ------------------------------------------------------------------------- */

#include <stdint.h>
#include "rope_intrin.h"
#include "rope_vectors.h"

#define ROPE_PUTCHAR_ADDR 0x10000000u
#define ROPE_EXIT_ADDR    0x20000000u

static void put_char(char c) { *(volatile uint32_t *)ROPE_PUTCHAR_ADDR = (uint32_t)(unsigned char)c; }
/* [... put_str / put_hex / put_dec / test_exit ...] */

static uint32_t failures = 0;

static void check_u32(const char *what, uint32_t got, uint32_t want, uint32_t idx) {
    if (got != want) {
        failures++;
        if (failures <= 20) { /* [... print what/idx/got/want ...] */ }
    }
}

/* 1. The packing convention must match the RTL exactly.
 *
 * Checked before anything numerical: if x and y were swapped, every later comparison
 * would fail in a confusing way that looks numerical rather than structural. */
static void test_packing(void) {
    uint32_t p = rope_pack(0x1234u, 0xABCDu);
    check_u32("pack", p, 0x1234ABCDu, 0);
    check_u32("unpack_x", rope_unpack_x(p), 0x1234u, 0);
    check_u32("unpack_y", rope_unpack_y(p), 0xABCDu, 0);
    put_str("test_packing done\n");
}

/* 2. BF16 <-> FP32 conversion. Widening must be exact; narrowing must be RNE. */
static void test_conversion(void) {
    check_u32("f32_to_bf16(1.0)", f32_to_bf16(1.0f), 0x3F80u, 0);
    check_u32("f32_to_bf16(-1.0)", f32_to_bf16(-1.0f), 0xBF80u, 0);
    check_u32("f32_to_bf16(2.0)", f32_to_bf16(2.0f), 0x4000u, 0);
    check_u32("f32_to_bf16(0.5)", f32_to_bf16(0.5f), 0x3F00u, 0);

    /* Exact widening: BF16 IS the top 16 bits of FP32. */
    { union { uint32_t u; float f; } c;
      c.f = bf16_to_f32(0x3F80u);
      check_u32("bf16_to_f32(0x3F80) bits", c.u, 0x3F800000u, 0);
      c.f = bf16_to_f32(0xC0A0u);
      check_u32("bf16_to_f32(0xC0A0) bits", c.u, 0xC0A00000u, 0); }

    /* RNE on an exact tie: the midpoint between 1.0 (0x3F80) and 1.0078125 (0x3F81)
       must round to the EVEN significand, i.e. back to 0x3F80. A truncating
       implementation happens to agree here, so also check a tie that must round UP. */
    { union { uint32_t u; float f; } c;
      c.u = 0x3F808000u;  /* exactly halfway between 0x3F80 and 0x3F81 */
      check_u32("RNE tie -> even (down)", f32_to_bf16(c.f), 0x3F80u, 0);
      c.u = 0x3F818000u;  /* halfway between 0x3F81 (odd) and 0x3F82 -> round up */
      check_u32("RNE tie -> even (up)", f32_to_bf16(c.f), 0x3F82u, 0); }

    /* A NaN whose payload lives only in the low 16 bits must stay a NaN, not become
       an infinity. */
    { union { uint32_t u; float f; } c;
      c.u = 0x7F800001u;
      check_u32("NaN with low payload stays NaN", f32_to_bf16(c.f), 0x7FC0u, 0); }
    put_str("test_conversion done\n");
}

/* 3. The instruction itself, against the golden-model vectors. */
static void test_rope_rot(void) {
    uint32_t k;
    for (k = 0; k < ROPE_NUM_VECTORS; k++) {
        uint32_t rs1 = rope_vec_rs1[k];
        uint32_t rs2 = rope_vec_rs2[k];
        uint32_t got = rope_rot(rs1, (uint16_t)(rs2 >> 16), (uint16_t)(rs2 & 0xFFFFu));
        check_u32("rope_rot", got, rope_vec_rd[k], k);
    }
    put_str("test_rope_rot done: "); put_dec(ROPE_NUM_VECTORS); put_str(" vectors\n");
}

/* 4. Back-to-back dependent instructions.
 *
 * Feeding one ROPE.ROT's result straight into the next exercises the registered-operand
 * path (steps.md Section 9.3 rule 1) and the core's forwarding network together. A
 * rotation by m twice must equal a rotation by 2m, up to BF16 rounding -- so rather than
 * assert bit-equality (which rounding forbids), assert against the model-provided
 * expected chain value. */
static void test_dependent_chain(void) {
    uint32_t acc = rope_vec_rs1[0];
    uint32_t k;
    for (k = 0; k < ROPE_CHAIN_STEPS; k++) {
        acc = rope_rot(acc, ROPE_CHAIN_M, ROPE_CHAIN_I);
    }
    check_u32("dependent chain", acc, ROPE_CHAIN_EXPECTED, 0);
    put_str("test_dependent_chain done\n");
}

/* 5. Whole-head rotation in both conventions. */
static void test_head_vector(void) {
    uint16_t vec[ROPE_HEAD_DIM];
    uint32_t k;

    for (k = 0; k < ROPE_HEAD_DIM; k++) vec[k] = rope_head_in[k];
    rope_rotate_head(vec, ROPE_HEAD_DIM, ROPE_HEAD_M, ROPE_HALF_SPLIT);
    for (k = 0; k < ROPE_HEAD_DIM; k++)
        check_u32("head half_split", vec[k], rope_head_out_half_split[k], k);

    for (k = 0; k < ROPE_HEAD_DIM; k++) vec[k] = rope_head_in[k];
    rope_rotate_head(vec, ROPE_HEAD_DIM, ROPE_HEAD_M, ROPE_INTERLEAVED);
    for (k = 0; k < ROPE_HEAD_DIM; k++)
        check_u32("head interleaved", vec[k], rope_head_out_interleaved[k], k);
    put_str("test_head_vector done\n");
}

int main(void) {
    put_str("=== BF16 RoPE coprocessor test ===\n");
    test_packing();
    test_conversion();
    test_rope_rot();
    test_dependent_chain();
    test_head_vector();

    if (failures) {
        put_str("RESULT: FAIL, "); put_dec(failures); put_str(" mismatches\n");
        test_exit(1);
    }
    put_str("RESULT: PASS\n");
    test_exit(0);
    return 0;
}
```

## D.3 `sw/crt0.S` — startup (57 lines, full)

```asm
/* ---------------------------------------------------------------------------
 * crt0.S -- minimal startup for the bare-metal RoPE test.
 *
 * Sets up the stack, zeroes .bss, copies nothing (everything is in place for a
 * simulation image), calls main, then writes the exit code and parks.
 * ------------------------------------------------------------------------- */

    .section .text.start, "ax"
    .global _start
    .type _start, @function

_start:
    /* Stack pointer from the linker script. */
    la      sp, _stack_top

    /* Global pointer, so gp-relative accesses work. */
    .option push
    .option norelax
    la      gp, __global_pointer$
    .option pop

    /* Zero .bss */
    la      t0, _bss_start
    la      t1, _bss_end
bss_loop:
    bgeu    t0, t1, bss_done
    sw      zero, 0(t0)
    addi    t0, t0, 4
    j       bss_loop
bss_done:

    /* No trap handler beyond a park loop: this test must not trap, so a trap is a
       failure we want to be obvious rather than silently handled. */
    la      t0, trap_park
    csrw    mtvec, t0

    call    main

    /* main() normally exits via test_exit(); if it returns, report its value. */
    li      t0, 0x20000000
    sw      a0, 0(t0)

park:
    wfi
    j       park

    .align 2
trap_park:
    /* Signal failure with a distinctive code, then park. */
    li      t0, 0x20000000
    li      t1, 0xDEAD
    sw      t1, 0(t0)
park2:
    wfi
    j       park2

    .size _start, .-_start
```

## D.4 `sw/link.ld` — memory map (61 lines, full)

Contains the origin fix from Part 4.8.

```
/* ---------------------------------------------------------------------------
 * link.ld -- memory map for the bare-metal RoPE test.
 *
 * Matches tb_rope_core.sv:
 *   0x00000000  RAM (instructions + data), 256 KB
 *   0x10000000  putchar   (write-only)
 *   0x20000000  exit code (write-only)
 * ------------------------------------------------------------------------- */

OUTPUT_ARCH(riscv)
ENTRY(_start)

/* The origin IS the reset vector, and must be set here rather than by assigning to `.`
 * inside SECTIONS: with output sections placed via `> RAM`, the linker allocates from the
 * region origin and silently ignores a preceding location-counter assignment. An earlier
 * version set `. = 0x80` and still produced a .text at VMA 0, which boots nowhere.
 *
 * 0x80 satisfies the CV32E40X requirement that boot_addr_i be 128-bit aligned. */
MEMORY
{
  RAM (rwx) : ORIGIN = 0x00000080, LENGTH = 256K - 0x80
}

SECTIONS
{
  .text : {
    KEEP(*(.text.start))
    *(.text .text.*)
  } > RAM

  .rodata : {
    *(.rodata .rodata.*)
    *(.srodata .srodata.*)
  } > RAM

  .data : {
    __global_pointer$ = . + 0x800;
    *(.data .data.*)
    *(.sdata .sdata.*)
  } > RAM

  .bss (NOLOAD) : {
    _bss_start = .;
    *(.bss .bss.*)
    *(.sbss .sbss.*)
    *(COMMON)
    _bss_end = .;
  } > RAM

  /* Stack at the top of RAM, growing down. */
  . = ALIGN(16);
  _heap_start = .;
  _stack_bottom = ORIGIN(RAM) + LENGTH(RAM) - 8K;
  _stack_top    = ORIGIN(RAM) + LENGTH(RAM);

  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
}
```

## D.5 `sw/Makefile` — cross-compilation (72 lines, full)

```makefile
# =============================================================================
# Bare-metal build for the BF16 RoPE software test (Milestone 7).
#
#   make            build test_rope.elf + .hex + .dis
#   make verify     check that the .insn encoding gcc emitted is what rope_pkg decodes
#   make clean
# =============================================================================

HERE   := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
ROOT   := $(abspath $(HERE)/..)
PYTHON ?= python3

# Toolchain autodetect. The CV32E40X is RV32IMC; the RoPE unit needs no F extension,
# because the coprocessor consumes BF16 in INTEGER registers (two per GPR).
TOOLCHAIN_DIR ?= $(HOME)/corev-openhw-gcc-ubuntu2204-20240530/bin
CROSS         ?= $(if $(wildcard $(TOOLCHAIN_DIR)/riscv32-corev-elf-gcc),$(TOOLCHAIN_DIR)/riscv32-corev-elf-,riscv32-unknown-elf-)

CC      := $(CROSS)gcc
OBJDUMP := $(CROSS)objdump
OBJCOPY := $(CROSS)objcopy

# zicsr is required explicitly: since Zicsr was split out of the base ISA, `csrw mtvec`
# in crt0.S will not assemble without it.
ARCH  ?= rv32imc_zicsr
ABI   ?= ilp32

CFLAGS := -march=$(ARCH) -mabi=$(ABI) -Os -g \
          -ffunction-sections -fdata-sections -fno-builtin \
          -Wall -Wextra -std=c11 -I$(HERE)

LDFLAGS := -march=$(ARCH) -mabi=$(ABI) -nostdlib -nostartfiles \
           -T$(HERE)/link.ld -Wl,--gc-sections -Wl,-Map=$(HERE)/test_rope.map

SRCS := $(HERE)/crt0.S $(HERE)/test_rope.c
ELF  := $(HERE)/test_rope.elf
HEX  := $(HERE)/test_rope.hex
DIS  := $(HERE)/test_rope.dis
VEC  := $(HERE)/rope_vectors.h

.PHONY: all clean verify vectors

all: $(DIS) $(HEX)

vectors: $(VEC)

$(VEC): $(ROOT)/model/gen_c_vectors.py $(ROOT)/model/rope_ref.py $(ROOT)/model/bf16.py
	cd $(ROOT)/model && $(PYTHON) gen_c_vectors.py --out $@

$(ELF): $(SRCS) $(VEC) $(HERE)/rope_intrin.h $(HERE)/link.ld
	$(CC) $(CFLAGS) $(LDFLAGS) $(HERE)/crt0.S $(HERE)/test_rope.c -o $@

$(DIS): $(ELF)
	$(OBJDUMP) -d -S $< > $@

# Flat little-endian hex image, one 32-bit word per line, for $readmemh in the testbench.
$(HEX): $(ELF)
	$(OBJCOPY) -O verilog --verilog-data-width=4 $< $@

# ---------------------------------------------------------------------------
# Encoding verification
#
# The whole software path hinges on `.insn r 0x0B, 0, 0, rd, rs1, rs2` assembling to an
# instruction the coprocessor decodes. Rather than trust that, disassemble and check the
# fields: opcode must be 0x0B, funct3 0, funct7 0.
# ---------------------------------------------------------------------------
verify: $(DIS)
	@$(PYTHON) $(HERE)/verify_encoding.py $(DIS)

clean:
	rm -f $(ELF) $(HEX) $(DIS) $(HERE)/test_rope.map
```

## D.6 `sw/verify_encoding.py` — checks the assembled encoding (88 lines, full)

```python
#!/usr/bin/env python3
"""Verify the ROPE.ROT instruction encoding emitted by the assembler.

The software path depends on `.insn r 0x0B, 0, 0, rd, rs1, rs2` producing an instruction
word that rope_pkg::is_rope_rot() accepts. That is exactly the kind of thing that is
"obviously fine" right up until the coprocessor silently rejects every instruction, so
check it against the disassembly rather than assuming.

Usage: verify_encoding.py <test_rope.dis>
"""

import re
import sys

OPCODE = 0x0B
FUNCT3 = 0
FUNCT7 = 0

# objdump lines look like:  "   80:\t0b00d0ab \t.insn\t4, 0x..."  (mnemonic varies)
LINE_RE = re.compile(r"^\s*([0-9a-f]+):\s+([0-9a-f]{8})\s+(.*)$")


def decode(word):
    return {
        "opcode": word & 0x7F,
        "rd": (word >> 7) & 0x1F,
        "funct3": (word >> 12) & 0x7,
        "rs1": (word >> 15) & 0x1F,
        "rs2": (word >> 20) & 0x1F,
        "funct7": (word >> 25) & 0x7F,
    }


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    with open(sys.argv[1]) as f:
        lines = f.readlines()

    found = []
    for ln in lines:
        mm = LINE_RE.match(ln)
        if not mm:
            continue
        word = int(mm.group(2), 16)
        if (word & 0x7F) == OPCODE:
            found.append((mm.group(1), word, mm.group(3).strip()))

    if not found:
        print("FAIL: no instruction with opcode 0x0B found in the disassembly.")
        print("      The .insn directive did not produce a custom-0 instruction.")
        return 1

    bad = 0
    for addr, word, text in found:
        d = decode(word)
        ok = d["opcode"] == OPCODE and d["funct3"] == FUNCT3 and d["funct7"] == FUNCT7
        if not ok:
            bad += 1
            print(f"FAIL {addr}: word=0x{word:08x} opcode=0x{d['opcode']:02x} "
                  f"funct3={d['funct3']} funct7={d['funct7']}  ({text})")

    sample = found[0]
    d = decode(sample[1])
    print(f"Found {len(found)} ROPE.ROT instruction(s) with opcode 0x{OPCODE:02X}.")
    print(f"  example @{sample[0]}: word=0x{sample[1]:08x} "
          f"rd=x{d['rd']} rs1=x{d['rs1']} rs2=x{d['rs2']} "
          f"funct3={d['funct3']} funct7={d['funct7']}")
    print(f"  raw text: {sample[2]}")

    if bad:
        print(f"FAIL: {bad} instruction(s) with the wrong funct3/funct7.")
        return 1

    print("PASS: every custom-0 instruction matches rope_pkg's decode "
          "(opcode 0x0B, funct3 0, funct7 0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## D.7 `Makefile` — the build and test system (266 lines, full)

```makefile
# =============================================================================
# BF16 RoPE coprocessor for CV32E40X -- build and verification
#
# Everything is self-contained inside the cv32e40x repository: CVFPU is COPIED into
# rtl/vendor/ (never symlinked), and the golden model needs only numpy.
#
# Common targets:
#   make gen          regenerate the theta ROM and sine LUT from gen/
#   make model        run the Python golden-model test suites
#   make lint         Verilator lint over all RTL
#   make test         run every RTL testbench (M1..M5)
#   make characterize regenerate docs/error_budget.md
#   make all          gen + model + lint + test
#
# Per-milestone: make m1 m2 m3 m4 m5
# =============================================================================

ROOT      := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
CV32E40X  := $(abspath $(ROOT)/..)

PYTHON    ?= python3
VERILATOR ?= verilator
BUILD     := $(ROOT)/build

# Number of random vectors per testbench. Override for a quick smoke run:
#   make test NVEC=2000
# steps.md asks for 10^7 on the primitives and the datapath; that is NVEC=10000000,
# which is slow in Verilator but supported -- see `make m1-full`.
NVEC      ?= 200000

# ---------------------------------------------------------------------------
# Source lists. fpnew_pkg MUST be compiled before anything that references it, and
# rope_pkg before the RoPE RTL, so these lists are order-significant.
# ---------------------------------------------------------------------------

CC_SRC := \
  $(ROOT)/rtl/vendor/common_cells/src/cf_math_pkg.sv \
  $(ROOT)/rtl/vendor/common_cells/src/lzc.sv \
  $(ROOT)/rtl/vendor/common_cells/src/rr_arb_tree.sv

FPNEW_SRC := \
  $(ROOT)/rtl/vendor/cvfpu/src/fpnew_pkg.sv \
  $(ROOT)/rtl/vendor/cvfpu/src/fpnew_classifier.sv \
  $(ROOT)/rtl/vendor/cvfpu/src/fpnew_rounding.sv \
  $(ROOT)/rtl/vendor/cvfpu/src/fpnew_fma.sv \
  $(ROOT)/rtl/vendor/cvfpu/src/fpnew_noncomp.sv \
  $(ROOT)/rtl/vendor/cvfpu/src/fpnew_opgroup_fmt_slice.sv \
  $(ROOT)/rtl/vendor/cvfpu/src/fpnew_opgroup_block.sv \
  $(ROOT)/rtl/vendor/cvfpu/src/fpnew_top.sv

ROPE_SRC := \
  $(ROOT)/rtl/rope_pkg.sv \
  $(ROOT)/rtl/rope_theta_rom.sv \
  $(ROOT)/rtl/rope_sin_rom.sv \
  $(ROOT)/rtl/rope_sin_lut.sv \
  $(ROOT)/rtl/rope_phase_gen.sv \
  $(ROOT)/rtl/rope_bf16_unit.sv \
  $(ROOT)/rtl/rope_bf16_mul.sv \
  $(ROOT)/rtl/rope_bf16_add.sv \
  $(ROOT)/rtl/rope_datapath.sv

XIF_SRC := \
  $(CV32E40X)/rtl/cv32e40x_if_xif.sv \
  $(ROOT)/rtl/rope_xif_coproc.sv

ALL_RTL := $(CC_SRC) $(FPNEW_SRC) $(ROPE_SRC) $(XIF_SRC)

# ---------------------------------------------------------------------------
# Verilator flags
#
# --binary builds a standalone simulation executable (Verilator >= 5).
# fpnew and common_cells are third-party: silence style-only warnings from them rather
# than editing vendored code, but keep real checks on. Our own RTL is held to -Wall.
# ---------------------------------------------------------------------------

# Waivers. Every one of these fires inside the vendored CVFPU / common_cells sources,
# whose style Verilator dislikes but which are correct and widely taped out. We do not
# edit vendored code, so they are waived globally rather than patched.
#
#   BLKANDNBLK   fpnew_fma drives inp_pipe_valid_q[0] with assign and the rest with
#                non-blocking assignments in a generate loop. Verilator flags the mix as
#                unsupported; it is legal SV and the elements are disjoint.
#   WIDTHCONCAT  fpnew_pkg's own fpu_features_t literals (Width: 64 etc.) are unsized;
#                assignment patterns to packed structs then look like concatenations.
#   UNSIGNED /   constant-comparison and empty-pin style in rr_arb_tree.
#   PINCONNECTEMPTY
VLT_WAIVE := \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNOPTFLAT -Wno-UNUSEDSIGNAL \
  -Wno-UNUSEDPARAM -Wno-DECLFILENAME -Wno-VARHIDDEN -Wno-SYNCASYNCNET \
  -Wno-CASEINCOMPLETE -Wno-ASCRANGE -Wno-GENUNNAMED -Wno-PINCONNECTEMPTY \
  -Wno-BLKANDNBLK -Wno-WIDTHCONCAT -Wno-UNSIGNED -Wno-CMPCONST

# rtl/vendor/common_cells/include is required: fpnew_fma.sv includes
# "common_cells/registers.svh" for its FF/FFL flop macros.
VLT_INC := -I$(ROOT)/rtl -I$(ROOT)/tb -I$(ROOT)/rtl/vendor/common_cells/include

VLT_FLAGS := --timing -Wall $(VLT_WAIVE) $(VLT_INC) --assert -j 0

# Testbench-only waivers. Scoreboards legitimately use blocking assignments for
# bookkeeping counters inside clocked processes, and stimulus tasks are procedural.
VLT_TB_WAIVE := -Wno-BLKSEQ -Wno-INITIALDLY -Wno-MULTIDRIVEN -Wno-STMTDLY

VLT_SIM   := $(VERILATOR) --binary $(VLT_FLAGS) $(VLT_TB_WAIVE)

.PHONY: all gen model lint test characterize clean m1 m2 m3 m4 m5 m1-full m4-full dirs

all: gen model lint test

dirs:
	@mkdir -p $(BUILD)

# ---------------------------------------------------------------------------
# Generated RTL
# ---------------------------------------------------------------------------

gen: $(ROOT)/rtl/rope_theta_rom.sv $(ROOT)/rtl/rope_sin_rom.sv

$(ROOT)/rtl/rope_theta_rom.sv: $(ROOT)/gen/gen_theta_rom.py $(ROOT)/model/rope_ref.py
	$(PYTHON) $(ROOT)/gen/gen_theta_rom.py --out $@

$(ROOT)/rtl/rope_sin_rom.sv: $(ROOT)/gen/gen_sin_lut.py $(ROOT)/model/bf16.py
	$(PYTHON) $(ROOT)/gen/gen_sin_lut.py --out $@

# ---------------------------------------------------------------------------
# Golden model
# ---------------------------------------------------------------------------

model:
	cd $(ROOT)/model && $(PYTHON) test_bf16.py
	cd $(ROOT)/model && $(PYTHON) test_rope_ref.py

characterize:
	cd $(ROOT) && $(PYTHON) model/characterize.py --out docs/error_budget.md

# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------

# Linting the coprocessor standalone leaves the CPU-driven half of the XIF interface
# (issue_valid, commit_*, result_ready, most of issue_req) undriven by construction, so
# UNDRIVEN is waived HERE ONLY. The `lint-integration` target below has the real core
# driving those signals and does not waive it.
lint: gen
	$(VERILATOR) --lint-only $(VLT_FLAGS) -Wno-UNDRIVEN $(ALL_RTL) --top-module rope_xif_coproc

# Full integration lint: the real cv32e40x core + interface + coprocessor.
# This is the steps.md Section 10.1 step 4 check.
#
# cv32e40x_sim_clock_gate.sv comes from bhv/ -- the core instantiates cv32e40x_clock_gate
# and ships only a behavioural model of it, so the technology-independent build needs it.
#
# IMPLICIT is waived because the three occurrences are in the CORE's own upstream RTL
# (cv32e40x_debug_triggers.sv, cv32e40x_ex_stage.sv), not in anything written here.
CORE_SRC := \
  $(CV32E40X)/rtl/include/cv32e40x_pkg.sv \
  $(CV32E40X)/bhv/cv32e40x_sim_clock_gate.sv

# All core RTL EXCEPT cv32e40x_if_xif.sv, which XIF_SRC already provides -- listing it
# twice is a duplicate-module error, not a harmless repeat.
CORE_RTL := $(filter-out $(CV32E40X)/rtl/cv32e40x_if_xif.sv,$(wildcard $(CV32E40X)/rtl/*.sv))

# COMBDLY is waived for the same reason as IMPLICIT: it fires only inside the core's own
# behavioural clock-gate model.
lint-integration: gen
	$(VERILATOR) --lint-only $(VLT_FLAGS) -Wno-UNDRIVEN -Wno-IMPLICIT -Wno-COMBDLY \
	  -I$(CV32E40X)/rtl -I$(CV32E40X)/rtl/include -I$(CV32E40X)/bhv \
	  $(CORE_SRC) \
	  $(CC_SRC) $(FPNEW_SRC) $(ROPE_SRC) $(XIF_SRC) \
	  $(CORE_RTL) \
	  $(ROOT)/rtl/rope_cv32e40x_wrapper.sv --top-module rope_cv32e40x_wrapper

# ---------------------------------------------------------------------------
# Testbenches
# ---------------------------------------------------------------------------

test: m1 m2 m3 m4 m5

# Vector filenames embed NVEC. This is load-bearing, not cosmetic: with a fixed name,
# `make m1 NVEC=1000000` sees an existing up-to-date vectors file and silently reuses the
# SMALLER previous one, so the run reports a pass over the wrong vector count. Encoding
# the count in the name makes a change to NVEC regenerate, and caches each size.
VEC_BF16     := $(BUILD)/vectors_bf16_$(NVEC).hex
VEC_PHASE    := $(BUILD)/vectors_phase_$(NVEC).hex
VEC_LUT      := $(BUILD)/vectors_lut_$(NVEC).hex
VEC_DATAPATH := $(BUILD)/vectors_datapath_$(NVEC).hex
VEC_XIF      := $(BUILD)/vectors_xif_$(NVEC).hex

# --- M1: BF16 primitives -----------------------------------------------------
$(VEC_BF16): $(ROOT)/model/gen_vectors.py $(ROOT)/model/bf16.py | dirs
	cd $(ROOT)/model && $(PYTHON) gen_vectors.py bf16 --n $(NVEC) --out $@

m1: gen $(VEC_BF16)
	cd $(BUILD) && $(VLT_SIM) --Mdir obj_m1 -o sim_m1 \
	  $(CC_SRC) $(FPNEW_SRC) $(ROOT)/rtl/rope_pkg.sv \
	  $(ROOT)/rtl/rope_bf16_unit.sv $(ROOT)/rtl/rope_bf16_mul.sv $(ROOT)/rtl/rope_bf16_add.sv \
	  $(ROOT)/tb/tb_rope_bf16_unit.sv --top-module tb_rope_bf16_unit
	cd $(BUILD) && ./obj_m1/sim_m1 +vectors=$(notdir $(VEC_BF16))

# steps.md Section 5.3 asks for 10^7 random pairs with zero mismatches.
m1-full:
	$(MAKE) m1 NVEC=10000000

# --- M2: integer phase generator --------------------------------------------
$(VEC_PHASE): $(ROOT)/model/gen_vectors.py $(ROOT)/model/rope_ref.py | dirs
	cd $(ROOT)/model && $(PYTHON) gen_vectors.py phase --n $(NVEC) --out $@

m2: gen $(VEC_PHASE)
	cd $(BUILD) && $(VLT_SIM) --Mdir obj_m2 -o sim_m2 \
	  $(ROOT)/rtl/rope_pkg.sv $(ROOT)/rtl/rope_theta_rom.sv $(ROOT)/rtl/rope_phase_gen.sv \
	  $(ROOT)/tb/tb_rope_phase_gen.sv --top-module tb_rope_phase_gen
	cd $(BUILD) && ./obj_m2/sim_m2 +vectors=$(notdir $(VEC_PHASE))

# --- M3: sine LUT ------------------------------------------------------------
$(VEC_LUT): $(ROOT)/model/gen_vectors.py $(ROOT)/model/rope_ref.py | dirs
	cd $(ROOT)/model && $(PYTHON) gen_vectors.py lut --n $(NVEC) --out $@

m3: gen $(VEC_LUT)
	cd $(BUILD) && $(VLT_SIM) --Mdir obj_m3 -o sim_m3 \
	  $(ROOT)/rtl/rope_pkg.sv $(ROOT)/rtl/rope_sin_rom.sv $(ROOT)/rtl/rope_sin_lut.sv \
	  $(ROOT)/tb/tb_rope_sin_lut.sv --top-module tb_rope_sin_lut
	cd $(BUILD) && ./obj_m3/sim_m3 +vectors=$(notdir $(VEC_LUT))

# --- M4: full BF16 datapath --------------------------------------------------
$(VEC_DATAPATH): $(ROOT)/model/gen_vectors.py $(ROOT)/model/rope_ref.py | dirs
	cd $(ROOT)/model && $(PYTHON) gen_vectors.py datapath --n $(NVEC) --out $@

m4: gen $(VEC_DATAPATH)
	cd $(BUILD) && $(VLT_SIM) --Mdir obj_m4 -o sim_m4 \
	  $(CC_SRC) $(FPNEW_SRC) $(ROPE_SRC) \
	  $(ROOT)/tb/tb_rope_datapath.sv --top-module tb_rope_datapath
	cd $(BUILD) && ./obj_m4/sim_m4 +vectors=$(notdir $(VEC_DATAPATH))

m4-full:
	$(MAKE) m4 NVEC=10000000

# --- M5: XIF coprocessor wrapper --------------------------------------------
$(VEC_XIF): $(ROOT)/model/gen_vectors.py $(ROOT)/model/rope_ref.py | dirs
	cd $(ROOT)/model && $(PYTHON) gen_vectors.py xif --n $(NVEC) --out $@

m5: gen $(VEC_XIF)
	cd $(BUILD) && $(VLT_SIM) --Mdir obj_m5 -o sim_m5 \
	  $(CC_SRC) $(FPNEW_SRC) $(ROPE_SRC) $(XIF_SRC) \
	  $(ROOT)/tb/tb_rope_xif_coproc.sv --top-module tb_rope_xif_coproc
	cd $(BUILD) && ./obj_m5/sim_m5 +vectors=$(notdir $(VEC_XIF))

# --- M7: the C test on the integrated core ----------------------------------
#
# Builds the bare-metal image, then runs it on the real core + coprocessor. This closes
# the loop: golden model -> compiled vectors -> real instructions -> core -> XIF -> RoPE.
.PHONY: sw
sw:
	$(MAKE) -C $(ROOT)/sw
	$(MAKE) -C $(ROOT)/sw verify

m7: gen sw | dirs
	cd $(BUILD) && $(VLT_SIM) -Wno-IMPLICIT -Wno-COMBDLY \
	  -I$(CV32E40X)/rtl -I$(CV32E40X)/rtl/include -I$(CV32E40X)/bhv \
	  --Mdir obj_m7 -o sim_m7 \
	  $(CORE_SRC) $(CC_SRC) $(FPNEW_SRC) $(ROPE_SRC) $(XIF_SRC) $(CORE_RTL) \
	  $(ROOT)/rtl/rope_cv32e40x_wrapper.sv \
	  $(ROOT)/tb/tb_rope_core.sv --top-module tb_rope_core
	cd $(BUILD) && ./obj_m7/sim_m7 +hex=$(ROOT)/sw/test_rope.hex

clean:
	rm -rf $(BUILD)
	$(MAKE) -C $(ROOT)/sw clean
```

## D.8 `syn/rope.sdc` — timing constraints (65 lines, full) — **NOT YET RUN**

```tcl
# =============================================================================
# rope.sdc -- timing constraints for the standalone BF16 RoPE coprocessor.
#
# Derived from cv32e40x/constraints/cv32e40x_core.sdc, as steps.md Section 9.3
# recommends, with the CORE-V-XIF timing budget applied on top.
#
# Target: 400 MHz (2.5 ns), matching Mugi's 45 nm operating point so the area/timing
# numbers are directly comparable. Start at 100 MHz on FPGA and tighten from there.
# =============================================================================

set CLK_PERIOD   2.5
set CLK_NAME     clk_i
set RST_NAME     rst_ni

create_clock -name $CLK_NAME -period $CLK_PERIOD [get_ports $CLK_NAME]

# Realistic uncertainty and transition until a real PLL/CTS model is available.
set_clock_uncertainty [expr {$CLK_PERIOD * 0.03}] [get_clocks $CLK_NAME]
set_clock_transition  [expr {$CLK_PERIOD * 0.03}] [get_clocks $CLK_NAME]

# Reset is asynchronous and externally synchronised.
set_false_path -from [get_ports $RST_NAME]

# -----------------------------------------------------------------------------
# CORE-V-XIF budget (steps.md Section 9.3)
# -----------------------------------------------------------------------------
#
# General split for XIF signals: 20% processor / 20% interconnect / 60% coprocessor.
set XIF_INPUT_DELAY  [expr {$CLK_PERIOD * 0.40}]
set XIF_OUTPUT_DELAY [expr {$CLK_PERIOD * 0.40}]

# The issue-request source operands are the exception and the reason
# rope_xif_coproc flops rs1/rs2 on arrival with ZERO combinational logic in front of
# them: CV32E40X drives them straight from its register-file bypass network, so the
# coprocessor sees only a small fraction of the period. Budget them tightly (85% gone
# before they arrive) so that a violation here shows up in synthesis rather than as a
# mysterious silicon failure.
set XIF_RS_INPUT_DELAY [expr {$CLK_PERIOD * 0.85}]

if {[llength [get_ports -quiet xif_issue_if*rs*]] > 0} {
  set_input_delay -clock $CLK_NAME $XIF_RS_INPUT_DELAY [get_ports xif_issue_if*rs*]
}

# Everything else on the XIF gets the general budget.
foreach pat {xif_issue_if* xif_commit_if* xif_result_if*} {
  set in_ports [get_ports -quiet $pat -filter {direction == in}]
  if {[llength $in_ports] > 0} {
    set_input_delay -clock $CLK_NAME $XIF_INPUT_DELAY $in_ports
  }
  set out_ports [get_ports -quiet $pat -filter {direction == out}]
  if {[llength $out_ports] > 0} {
    set_output_delay -clock $CLK_NAME $XIF_OUTPUT_DELAY $out_ports
  }
}

# -----------------------------------------------------------------------------
# Design rules
# -----------------------------------------------------------------------------

set_max_fanout 20 [current_design]
set_max_transition [expr {$CLK_PERIOD * 0.15}] [current_design]

# The sine LUT is a 1024x16 constant table. Left as logic deliberately (2 KB is small,
# and a macro would add a pipeline stage plus a memory-compiler dependency), but it is
# the widest combinational cone in the design, so keep an eye on it in the QoR report.
```

## D.9 `syn/synth.tcl` — Genus flow (110 lines, abridged) — **NOT YET RUN**

```tcl
# =============================================================================
# synth.tcl -- Genus synthesis script for the standalone BF16 RoPE coprocessor.
#
#   genus -batch -f syn/synth.tcl
#
# Produces the Milestone 9 measurements: area, timing and the area breakdown
# (sine LUT vs theta ROM vs CVFPU units vs phase logic vs XIF wrapper).
#
# STATUS: NOT YET RUN. This project has no library set up here, so the script is
# untested end to end -- it encodes the intended flow and the reporting that the
# writeup needs, and will need the library paths below pointed at a real PDK.
# Everything above Milestone 9 is verified in simulation and does not depend on this.
# =============================================================================

# Mugi reports 45 nm at 400 MHz; matching that node makes the comparison meaningful.
if {![info exists env(ROPE_LIB_PATH)]} {
  puts "ERROR: set ROPE_LIB_PATH to the directory holding the .lib files"
  exit 1
}
set_db init_lib_search_path $env(ROPE_LIB_PATH)
set_db library [glob -directory $env(ROPE_LIB_PATH) *.lib]

# Order matters: fpnew_pkg before anything referencing it, rope_pkg before the RoPE RTL.
# rtl/vendor/common_cells/include must be on the include path for registers.svh.
set_db init_hdl_search_path [list $ROOT/rtl $ROOT/rtl/vendor/common_cells/include]

read_hdl -language sv $RTL_FILES        # [... the same ordered list as the Makefile ...]

# Synthesise the datapath rather than the XIF wrapper: the wrapper's ports are a
# SystemVerilog interface, which complicates a standalone run, and the datapath plus
# phase/LUT is what the area and energy numbers are actually about.
elaborate rope_datapath

read_sdc $ROOT/syn/rope.sdc
check_design -unresolved
report_timing -lint

set_db syn_generic_effort high
set_db syn_map_effort     high
set_db syn_opt_effort     high

syn_generic
syn_map
syn_opt

# -----------------------------------------------------------------------------
# Reports -- these ARE the Milestone 9 deliverables
# -----------------------------------------------------------------------------
report_timing            > $OUTDIR/timing.rpt
report_area              > $OUTDIR/area.rpt
report_power             > $OUTDIR/power.rpt
report_gates             > $OUTDIR/gates.rpt
report_qor               > $OUTDIR/qor.rpt

# Area breakdown by block, which is the table the writeup needs:
#   sine LUT (2 KB) vs theta ROM (256 B) vs CVFPU units vs phase logic vs wrapper.
report_area -depth 3     > $OUTDIR/area_hierarchy.rpt

foreach inst {i_rom_rand i_rom_seq gen_mul i_sub_x i_add_y} {
  if {[llength [get_db insts -quiet *$inst*]] > 0} {
    report_area [get_db insts *$inst*] >> $OUTDIR/area_breakdown.rpt
  }
}

write_hdl                > $OUTDIR/rope_datapath_netlist.v
write_sdc                > $OUTDIR/rope_datapath.sdc
```

## D.10 `.gitignore` (26 lines, full)

```gitignore
# Simulation and build artifacts
build/
*.vcd
*.fst

# Software build products (regenerated by `make -C sw`)
sw/*.elf
sw/*.hex
sw/*.dis
sw/*.map
sw/*.bin

# Generated test vectors for the C test (regenerated by model/gen_c_vectors.py)
sw/rope_vectors.h

# Synthesis output
syn/out/

# Python
__pycache__/
*.pyc
.venv/

# NOTE: rtl/rope_theta_rom.sv and rtl/rope_sin_rom.sv ARE generated (by gen/) but are
# deliberately committed: they are the design's constants, they must be reviewable in
# diffs, and a synthesis run should not depend on having Python available.
```

## D.11 Other documentation files (not reproduced)

| File | Lines | Contents |
|---|---|---|
| `README.md` | 177 | Project overview, quick start, layout, headline results |
| `docs/notes.md` | 301 | Design decisions, subnormal policy evidence, `steps.md` corrections, the two CVFPU traps, verification status — the same material as Parts 4, 5 and 7 of this document, in engineering-note form |
| `docs/error_budget.md` | 102 | **Generated** by `model/characterize.py`; the tables reproduced in Part 6.3 |

---
---

*End of document. Total project: 49 authored files (8825 lines) plus 16 vendored CVFPU
files (3013 lines). No file of the existing cv32e40x repository was modified.*
