# Two stages instead of three: the θ prefetch

**Audience:** you are modifying `rope_xif_coproc.sv` or trying to understand why the
coprocessor has two pipeline stages instead of the three it originally had. This document
explains the architecture, why the obvious approaches do not work, and the `commit_kill`
hazard that the design has to defend against.

Everything measured here was run on 2026-09-23.

---

## 1. What changed, in one line

`ROPE.ROT`'s pair index `i` walks `0, 1, 2, …` in every real use, so the coprocessor can
read `theta[i+1]` while it is still working on pair `i`. The index then never needs to
*arrive* with the instruction, and the pipeline stage that existed only to capture it
disappears.

| | before | after |
|---|---|---|
| Pipeline stages | 3 | **2** |
| Issue → result | 3 cycles | **2 cycles** |
| WB stall | 1 cycle | **0** |
| Instruction encoding | — | **unchanged** |
| Software | — | **unchanged** |

Selected by `rope_xif_coproc`'s `UsePrefetch` parameter, default `0`.

---

## 2. Why the coprocessor had three stages

The rotation needs `sin(m·θᵢ)` and `cos(m·θᵢ)`. Producing them is three steps in series:

```
  i ──► θ-ROM ──► × m ──► phase ──► sine LUT (×2, parallel) ──► sin, cos
        6 levels   ~18                    20 levels
```

Measured with `yosys ltp` on the generated ROMs:

```
rope_theta_rom : longest topological path =  6
rope_sin_rom   : longest topological path = 20
```

That whole chain is roughly 44 levels of logic. It cannot start from the instruction,
because of a constraint the coprocessor has obeyed since Milestone 5 — rule 1 in
`rope_xif_coproc.sv`'s header:

> **REGISTER THE OPERAND INPUTS.** The coprocessor gets only a small slice of the timing
> budget on paths through `issue_req.rs*`, because CV32E40X sources those operands
> straight from its register-file bypass network.

The core drives those operands from the **forwarding mux**, not from a plain register
file read — `cv32e40x_id_stage.sv`:

```systemverilog
xif_issue_if.issue_req.rs[0] = operand_a_fw;
xif_issue_if.issue_req.rs[1] = operand_b_fw;
```

`operand_*_fw` is one of the latest-arriving signals in the whole ID stage: it waits on
hazard detection and bypass selection from EX and WB. Hanging 44 levels of ROM and
multiply off it would make the coprocessor the critical path for the entire CPU, not just
for itself.

So the original design spent a full stage doing nothing but moving the operands into
flops:

```
stage 1   x_q, y_q, pos_q, pair_q     ← 0 levels of logic. Capture only.
stage 2   x_s0, y_s0, sin_s0, cos_s0  ← θ-ROM + multiply + LUT, from stage 1's flops
stage 3   result FIFO                 ← BF16 arithmetic
```

Stage 1 exists solely so that stage 2's long path can begin at a flop.

---

## 3. The idea: the index does not have to arrive

`theta_i = base^(-2i/d)` depends only on `i` and on `d`, and `d` is fixed at elaboration
— it is baked into `rope_theta_rom`. So `theta[i]` is a pure ROM lookup, and the sequence
of indices in a head vector is completely predictable. `sw/rope_intrin.h`:

```c
for (k = 0; k < half; k++) {
    res = rope_rot(rope_pack(vec[lo], vec[hi]), m, (uint16_t)k);
}
```

`k` ascends, `m` is constant. So the coprocessor can keep `theta[i]` in a register and
refill it one instruction ahead:

```
stage 1   x_s0, y_s0, sin_s0, cos_s0
            ├─ x, y    ── straight from rs1 into flops        0 levels   (rule 1 ✓)
            └─ sin/cos ── m_q × theta_q → LUT                 ~38 levels
stage 2   result FIFO                                         ~50 levels
```

Both multiply operands now come from coprocessor flops, so rule 1 still holds. The θ-ROM
read for the *next* pair runs in the background, off the critical path. The capture stage
is gone.

**The prefetch does not make the chain shorter.** It moves the ROM read into an earlier
cycle and removes the need to capture the index at all. That distinction matters: a
design that merely computed the same chain combinationally from the issue interface would
be 2 stages too, and would wreck the core's frequency.

### Why `m` is latched and not used in-time

`m` also arrives on `issue_req.rs[1]`, from the same forwarding network. If stage 1's
multiply took `m` directly from the instruction, its 38 levels would again hang off the
bypass mux. So `m` lives in `m_q`, adopted from whichever instruction last resynced.

`theta` and `m` are therefore latched for *different reasons* — one because the ROM read
is slow, the other because the operand arrives late — but they create the same hazard:
state that persists across instructions and can disagree with reality.

### The invariant

```
theta_q == theta[i_q]
```

maintained on every accepted rotation by `theta_q ← theta_rom(i_q + 1)`, `i_q ← i_q + 1`.

---

## 4. The hazard: `commit_kill`

The prefetch advances on `accept`. But CORE-V-XIF lets the core **kill an instruction it
has already accepted**. `cv32e40x_controller_fsm.sv`:

```systemverilog
assign xif_commit_if.commit.commit_kill = xif_csr_error_i || ctrl_fsm_o.kill_ex || kill_rejected;
```

Three sources, and only one of them is benign:

| Source | Meaning | Threat to the prefetch |
|---|---|---|
| `kill_rejected` | the coprocessor declined the instruction | none — we never accepted it |
| `xif_csr_error_i` | an accepted instruction also touched a CSR | none — `ROPE.ROT` does not |
| **`kill_ex`** | **the pipeline was flushed** | **this one** |

`kill_ex` is asserted at seven sites in the controller, every one of them asynchronous to
the instruction stream: **NMI**, **interrupt**, **exception in WB**, **debug entry**,
**`fence`/`fence.i`**, **`dret`**, and a **CSR write that requires a flush**. All of them
kill IF, ID and EX together — every instruction in flight, regardless of whether it did
anything wrong.

### What goes wrong without a defence

```
ROPE.ROT  pair 0     uses θ₀ ✓     theta_q → θ₁,  i_q → 1
ROPE.ROT  pair 1     uses θ₁ ✓     theta_q → θ₂,  i_q → 2
ROPE.ROT  pair 2   ◄── timer interrupt: accepted, then KILLED
                                   theta_q → θ₃,  i_q → 3   ← state ran ahead
   ... handler runs, returns ...
ROPE.ROT  pair 2 (re-executed)     uses θ₃ ✗  WRONG
ROPE.ROT  pair 3                   uses θ₄ ✗  wrong from here on
```

The instruction re-executes correctly from the core's point of view, but the coprocessor
advanced for an instruction that never retired. **Every remaining pair in the head vector
is silently rotated by the wrong frequency.**

This failure is silent at every level: no assertion fires, no exception is raised, the
results are still finite unit-magnitude numbers, and it depends on interrupt timing — so
a test can pass a thousand times and fail in the field.

---

## 5. The defence: two independent checks

**Which of these is load-bearing was settled by measurement, not by reasoning.** Building
with `UseValueCheck=0` (`make m9-tier1`) fails **6039 of 6066** vectors, because the M5
stimulus is random `(m, i)` tuples rather than a sequential walk. So:

| | role | optional? |
|---|---|---|
| **Tier 2** (value check) | keeps `ROPE.ROT` a pure function of its operands for **any** `(m, i)` | **no** - required for correctness |
| **Tier 1** (kill-triggered) | redundant safety net; Tier 2 already catches the kill case | yes - costs nothing, kept anyway |

An earlier draft of this document had that backwards. A killed instruction re-executes
with an index that no longer matches `i_q`, so the value check detects it without help.

### Tier 1 — kill-triggered (always on)

```systemverilog
assign pf_kill_seen = xif_commit_if.commit_valid & xif_commit_if.commit.commit_kill;
...
if (pf_kill_seen) desync_q <= 1'b1;
```

Driven from the commit channel, which is already registered, so it costs nothing on any
timing path. This is what protects **correct** software from interrupts.

### Tier 2 (REQUIRED) — value check (`UseValueCheck`, default on)

```systemverilog
assign pf_value_mismatch = UseValueCheck
                         & ( (rs2_pos_full[PosBits-1:0]     != m_q)
                           | (rs2_idx_full[PairIdxBits-1:0] != i_q) );
```

The instruction still carries `m` and `i` in `rs2`. They stop being *inputs* and become
*assertions about the coprocessor's state*. This catches everything Tier 1 cannot:
context switches, two head vectors interleaved, a debugger stepping through the loop,
an interrupt handler that itself issues a rotation, or software that simply never
established the state.

**Timing note.** This comparator feeds `issue_ready`, which is a path rule 1 warns is
tight. It is acceptable because a comparator is ~5 levels and its result only has to
reach the handshake logic — not traverse a multiplier and a ROM. The parameter
exists to make the failure demonstrable (`make m9-tier1`), **not** as a shippable
configuration: without the check, a program that rotates pairs in any order other than
ascending silently gets the wrong answer. If it ever fails timing, the fix is to drop
`UsePrefetch` altogether rather than to disable the check. **No STA tool or PDK is configured in this repository, so this has
not been verified in silicon terms.**

### Recovery: one bubble

```systemverilog
if (pf_need_resync) begin
  m_q       <= rs2_pos_full[PosBits-1:0];   // adopt the instruction's own view
  i_q       <= rs2_idx_full[PairIdxBits-1:0];
  recover_q <= 1'b1;
  desync_q  <= 1'b0;
end
```

`issue_ready` drops for that one cycle, so the core re-offers the instruction. On the
next cycle `recover_q` selects the ROM path directly, because `theta_q` has not been
refilled yet:

```systemverilog
assign theta_sel = recover_q ? theta_cur : theta_q;   // theta_cur = theta_rom(i_q)
```

That path is ~44 levels rather than ~38 — the longer of the two, and still shorter than
the ~50-level arithmetic stage, so it does not set the clock.

### Why there is no `ROPE.INIT`

Because recovery **adopts state from the instruction**, the first rotation of a new token
initialises the coprocessor by itself. An explicit init instruction would cost one
instruction to save one bubble — roughly break-even — and would make correctness depend
on software remembering to issue it, which is exactly the fragile contract the checks
exist to eliminate.

Reset is handled the same way: `desync_q` resets to **1**, so the first rotation after
reset resyncs. That is cheaper and safer than inventing a `theta[0]` constant in the RTL
that could drift from `gen/gen_theta_rom.py`.

### Multi-head costs nothing

`i_q` is `PairIdxBits` wide and wraps naturally at `NumPairs`, and the prefetch wraps with
it, so the invariant survives the end of a vector. The next head for the same token starts
at `i = 0` with `m` unchanged, **matches**, and runs at full speed. One bubble per *token*,
not per vector.

---

## 6. A second hazard the shorter latency exposed

Reducing the latency broke something that had been latent since Milestone 5.

The original code gated the FIFO **push**:

```systemverilog
assign fifo_push = res_valid & cur_tag.valid & ~result_is_killed;
```

At three cycles the commit always arrived before the result was produced, so this worked.
At two cycles it does not — the result reaches the FIFO first, and a kill arriving
afterwards is ignored, so **the killed instruction writes `rd`**. This was caught by
`tb_rope_xif_coproc`'s TEST 3a immediately on the first run of the new path.

XIF permits the core to kill any outstanding instruction, so the fix is to gate the
FIFO **output** instead, where the commit status is known:

```systemverilog
assign head_committed = committed_q[fifo_q[rd_ptr_q[FifoPtrW-1:0]].id];
assign head_killed    = killed_q   [fifo_q[rd_ptr_q[FifoPtrW-1:0]].id];

assign xif_result_if.result_valid = ~fifo_empty & head_committed & ~head_killed;
```

plus a silent drain for an entry whose instruction was killed after it was pushed:

```systemverilog
assign fifo_drain = ~fifo_empty & head_killed;
assign fifo_pop   = (result_valid & result_ready) | fifo_drain;
```

This is correct at **any** latency, rather than relying on commit happening to arrive
first. It costs nothing in practice: CV32E40X signals commit from the **EX stage**
(`commit_valid` is driven from `id_ex_pipe_i`), one cycle after issue, and the earliest a
result can appear is also one cycle after issue.

`committed_q` is a per-id **level**, not an edge — `commit_valid` has no ready signal and
may arrive at any time, so an edge-triggered capture would lose it. That is the same
mistake, in a different place, as the `result_is_killed` bug fixed on 2026-09-22.

### Credit accounting

Two retirement paths now exist and can fire in the same cycle for *different*
instructions: a drain at the FIFO head, and a kill caught before the result was ever
pushed. They are **summed**, not ORed — ORing loses a credit whenever both coincide, which
slowly starves the coprocessor until it stops accepting work.

---

## 7. Building and testing

```bash
make m9           # the M5 suite against the two-stage path, both checks on
make m9-tier1     # same, with UseValueCheck compiled out (Tier 1 only)
make m5           # the original three-stage path, unchanged
```

`m9` runs **the same testbench and the same vectors** as `m5`. Semantics are identical —
same instruction, same encoding, same operands — so a difference between them is a bug,
not a new reference.

---

### Measured

```
make m5        sent=6067 checked=6066 errors=0     PASS   (3-stage)
make m9        sent=6067 checked=6066 errors=0     PASS   (2-stage, both tiers)
make m9-tier1  sent=6067 checked=6066 errors=6039  FAIL   (deliberately)

make m7        exit_code=0  after 13593 cycles     PASS   (3-stage)
make m9-core   exit_code=0  after 13466 cycles     PASS   (2-stage)
```

`m9-tier1` is a **falsification test**: it proves the value check does real work. A change
that makes it pass has broken the test, not fixed the design.

A probe on `wb_stage_i.xif_waiting` over the same program measures the stall directly:

```
3-stage:  accepts=200  stall_events=200  stall_cycles=200
2-stage:  accepts=200  stall_events=0    stall_cycles=0
```

**Zero.** `stall = stages - 2` is therefore measured at two stages, not interpolated from
the three- and five-stage points.

The two numbers reconcile exactly. 200 stall cycles are removed, and 73 resync bubbles
are added, for the rotations that do not walk sequentially:

| | rotations | behaviour |
|---|---|---|
| `test_rope_rot` (random `m`, `i`) | 64 | resync every time |
| `test_dependent_chain` (fixed `i`, `i_q` advances) | 8 | resync every time |
| `test_head_vector` (ascending `i`) | 128 | 1 resync, then 127 fast |
| | **200** | **73 resyncs, 127 fast** |

`200 - 73 = 127`. A resyncing rotation costs one bubble plus two stages -- three cycles,
exactly the old latency -- so it neither gains nor loses, and the net saving equals the
number of sequential rotations.

The 127-cycle saving on the integrated core matches the model exactly. The C test makes
two passes over a 64-pair head vector. The first differs from the reset state in `m`, so
it costs one resync bubble and then runs 63 rotations at two cycles; the second finds
`i_q` wrapped to 0 with `m` unchanged, matches immediately, and runs all 64 at two cycles.
63 + 64 = 127. The random-access tests (`test_rope_rot`, `test_dependent_chain`) resync on
every instruction and so cost three cycles, exactly the old latency: no gain, no loss.

### Three latent defects this work exposed

Reducing the latency made three pre-existing problems reachable. All three were found by
running the suite, none by inspection:

1. **A killed instruction wrote back.** The FIFO gated its *push* on the kill flag, which
   only worked because at three cycles the commit always arrived first. Fixed by gating
   the FIFO *output* on commit status (section 6).
2. **A credit leak.** That fix created two retirement paths which can fire in the same
   cycle for different instructions; the accounting ORed them. Now summed (section 6).
3. **A testbench protocol violation.** The commit threads walked a free-running counter
   and raced ahead of the issue thread whenever result-FIFO credit throttled it,
   committing ids before they had been issued. `accept` cleared those commits, so the
   instructions could never retire and the suite deadlocked at eight pending. The commit
   threads now wait on `sent`.

---

## 8. What is not verified

- **Stage 1 is estimated at 38–44 levels and stage 2 at ~50.** The ROM depths (6 and 20)
  are measured with `yosys ltp`; the multiply and the FPU are **estimates**, because
  Yosys cannot parse `rope_phase_gen`, `rope_sin_lut` or anything containing `fpnew_fma`
  (`function automatic` inside `rope_pkg` and `cf_math_pkg`). If stage 1 turns out to
  exceed stage 2, two stages is a bad trade and `UsePrefetch=0` is the fallback.
- ~~`stall = stages - 2` was interpolated~~ -- now measured at two stages: zero stall
  cycles over 200 rotations on the integrated core. See section 7.
- The Tier 2 comparator's effect on `issue_ready` has not been timed.
