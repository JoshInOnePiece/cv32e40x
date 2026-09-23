# Verification onramp

**Audience:** you are about to change the CV32E40X core or the RoPE coprocessor RTL by
hand, and you want to know whether it still works before and after. This is the operational
guide: where to stand, what to type, what it does, and what a healthy run prints.

Everything in this file was executed on 2026-09-22, against the working tree described in
§4 (`NumPipeRegs = 0`, same-cycle kill fix applied). The expected
output quoted below is real captured output, not an illustration.

---

## 0. The 30-second version

```bash
cd ~/cv32e40x/rope-bf16
make lint-integration && make m7
```

If both exit 0 and the second prints `PASS: C test passed on the integrated core`, a core
change has not broken the coprocessor path. Everything else in this document is detail
around those two commands.

**Every command below runs from `~/cv32e40x/rope-bf16`** unless the heading says otherwise.
The Makefile computes its own root, so it does not matter how you got there, but relative
paths in these examples assume it.

---

## 1. Prerequisites

| Tool | Needed for | Verified working version |
|---|---|---|
| Verilator | every RTL simulation and both lints | 5.020 (`--binary` requires >= 5) |
| Python 3 + numpy | the golden model and all stimulus generation | 3.12, numpy 2.5.1 |
| RISC-V GCC | `sw`, `m7`, `demo` only | `~/corev-openhw-gcc-ubuntu2204-20240530` |
| Yosys | the two generated ROMs only — see §6 | 0.33 installed; reads the ROMs, not the rest |

Check them:

```bash
verilator --version
python3 -c "import numpy; print(numpy.__version__)"
ls ~/corev-openhw-gcc-ubuntu2204-20240530/bin/riscv32-corev-elf-gcc
```

The RISC-V toolchain does **not** need to be on `PATH`. `sw/Makefile` looks for
`$(HOME)/corev-openhw-gcc-ubuntu2204-20240530/bin/riscv32-corev-elf-gcc` and falls back to a
generic `riscv32-unknown-elf-` prefix only if that is missing.

### Run git from inside WSL

This repository lives in WSL. Running `git status` against the `\\wsl.localhost\...` path
from Windows reports files as modified that are byte-identical, because Git for Windows
applies `core.autocrlf`. Use `wsl -d Ubuntu-24.04 -- bash -lc "cd ~/cv32e40x && git status"`,
or just run git from a WSL shell. This has already caused one real confusion in this repo.

---

## 2. The verification ladder

Eight stages, cheapest first. Each one isolates a smaller set of causes than the one after
it, so when something breaks, run them in order and stop at the first failure — that is the
layer your change broke.

| # | Command | What it exercises | Touches the core? | Needs GCC? |
|---|---|---|---|---|
| 1 | `make gen` | regenerates the two ROMs from Python | no | no |
| 2 | `make model` | the Python golden model, against itself | no | no |
| 3 | `make lint` | Verilator elaboration of the coprocessor alone | no | no |
| 4 | `make lint-integration` | Verilator elaboration of **core + XIF + coprocessor** | **yes** | no |
| 5 | `make m1` … `make m5` | five Verilator RTL simulations of the coprocessor | no | no |
| 6 | `make m7` | Verilator sim of the **real core** running a compiled C program | **yes** | yes |
| 7 | `make demo-quick` | all of the above plus a metrics summary | yes | yes |
| 8 | `make m1-full`, `make m4-full` | the 10^7-vector sweeps | no | no |

Three further targets cover the optional datapath variants. They are **not** part of the
default ladder because both variants are parameterised off; run them if you touch the
datapath or the XIF wrapper.

| Command | What it exercises | Must |
|---|---|---|
| `make m4-fma` | the fused multiply-add datapath (`UseFma=1`), against Model A' | PASS |
| `make m9` | the two-stage theta prefetch (`UsePrefetch=1`), same vectors as `m5` | PASS |
| `make m9-core` | the same, on the integrated core running the M7 C program | PASS |
| `make m9-tier1` | the prefetch with its value check compiled out | **FAIL** |

`m9-tier1` is a **falsification test** and is *expected to fail* — 6039 errors of 6066. It
exists to prove the value check is load-bearing. A change that makes it pass has broken
the test, not fixed the design.

**If you are changing the CV32E40X core, stages 4 and 6 are the ones that matter.** Stages
3 and 5 compile the coprocessor standalone and will happily pass while the core is broken.

---

## 3. Stage by stage

### 1. `make gen` — regenerate the ROMs

```bash
cd ~/cv32e40x/rope-bf16
make gen
```

Runs `gen/gen_theta_rom.py` and `gen/gen_sin_lut.py` to rewrite `rtl/rope_theta_rom.sv` and
`rtl/rope_sin_rom.sv`. Both are generated files under version control; the Makefile rebuilds
them only when the generator or the model it imports is newer.

Expected when nothing changed:

```
make: Nothing to be done for 'gen'.
```

Every other target depends on `gen`, so you rarely invoke it directly. If you edit
`model/bf16.py` or `model/rope_ref.py`, this is what propagates the change into RTL — and a
`git diff` on the two ROM files afterwards is how you see what moved.

### 2. `make model` — the golden model

```bash
make model
```

Runs `model/test_bf16.py` and `model/test_rope_ref.py`. Pure Python and numpy; **no RTL is
involved**. This establishes that the reference the RTL is checked against is itself sane —
BF16 rounding behaviour, NaN/Inf propagation, that the half-split and interleaved RoPE
conventions agree under permutation, that pair norms are preserved.

Expected — a list of `PASS` lines ending in:

```
  PASS  norm preserved to <5% worst case (max 0.0083)

All golden-model tests passed.
```

If this fails, nothing below it means anything: every RTL testbench compares against this
model, so a broken model produces confidently wrong "passes".

### 3. `make lint` — coprocessor elaboration

```bash
make lint
```

`verilator --lint-only -Wall` over CVFPU + common_cells + the RoPE RTL, topped at
`rope_xif_coproc`. Catches width mismatches, undriven signals, latch inference and
combinational loops without building a simulation.

Expected: **the echoed command line and nothing else.** Exit 0, no `%Error` or `%Warning`.

`-Wno-UNDRIVEN` is passed **here only**. Linting the coprocessor standalone leaves the
CPU-driven half of the XIF interface (`issue_valid`, `commit_*`, `result_ready`) undriven by
construction. Stage 4 has the real core driving those and does not waive it — which is why
stage 4 is the one that catches a broken XIF connection.

### 4. `make lint-integration` — core + XIF + coprocessor

```bash
make lint-integration
```

**The first stage that compiles the CV32E40X core.** Elaborates every `rtl/*.sv` plus
`bhv/cv32e40x_sim_clock_gate.sv`, the vendored FPU, the coprocessor and
`rope_cv32e40x_wrapper`, topped at the wrapper.

Expected: exit 0, no `%Error` lines.

```
exit=0
(no %Error lines above = clean)
```

`-Wno-IMPLICIT` and `-Wno-COMBDLY` are waived because they fire only inside the core's own
upstream RTL (`cv32e40x_debug_triggers.sv`, `cv32e40x_ex_stage.sv`, and the behavioural
clock gate) — not in anything written for this project. **If you add a new waiver to make
your change lint, you have almost certainly hidden a real bug.**

This is the cheapest check that a core edit still elaborates. Run it first, every time.

### 5. `make m1` … `make m5` — the coprocessor testbenches

```bash
make test                # all five, at the default NVEC=200000
make test NVEC=2000      # same five, fast
make m4 NVEC=2000        # just one
```

Each target generates stimulus with `model/gen_vectors.py`, compiles a Verilator binary into
`build/obj_m<N>/`, and runs it. **These are real RTL simulations** — the testbench drives the
DUT cycle by cycle and compares every output bit against the golden model.

`NVEC` is the random vector count, default 200000. The vector filename embeds the count
(`build/vectors_bf16_2000.hex`), which is load-bearing: with a fixed name, raising `NVEC`
would silently reuse the smaller previous file and report a pass over the wrong vector count.

Expected output at `NVEC=2000` — all five exit 0:

```
########## m1 ##########
tb_rope_bf16_unit: loaded 3935 vectors from vectors_bf16_2000.hex
tb_rope_bf16_unit: sent=3935 checked=3935 errors=0
PASS: all 3935 vectors bit-exact

########## m2 ##########
tb_rope_phase_gen: loaded 7208 vectors from vectors_phase_2000.hex
PASS: phase generator bit-exact (random-access, sequential, wraparound)

########## m3 ##########
tb_rope_sin_lut: loaded 6224 vectors from vectors_lut_2000.hex
bit-exact check: 6224 vectors, sin errors=0, cos errors=0
PASS: sine LUT bit-exact, reflection and quadrant signs exact

########## m4 ##########
tb_rope_datapath: loaded 3961 vectors from vectors_datapath_2000.hex
tb_rope_datapath: sent=3961 checked=3961 errors=0
PASS: datapath bit-exact vs Model A on all 3961 vectors

########## m5 ##########
tb_rope_xif_coproc: loaded 2000 vectors from vectors_xif_2000.hex
TEST 1 ok: non-ROPE instruction rejected (accept=0, issue_ready=1)
TEST 2 ok: single ROPE.ROT wrote back correctly
TEST 3a ok: killed instruction produced no writeback
TEST 3b ok: instruction after a kill is unaffected
TEST 4 ok: 1997 back-to-back ROPE.ROT all correct
tb_rope_xif_coproc: sent=2000 checked=1999 errors=0
PASS: XIF coprocessor -- decode, writeback, kill, throughput, backpressure
```

Reading these:

- **`loaded N` is larger than `NVEC`.** The generators add directed corner cases
  (zeros, subnormals, NaN, Inf, max exponent) on top of the random draw. 2000 becomes 3935
  for m1, 7208 for m2. That is expected, not a bug.
- **m5 shows `sent=2000 checked=1999`.** The one unchecked instruction is the deliberately
  killed one in TEST 3a — it must produce no writeback, so there is nothing to compare.
  `checked` being exactly one less than `sent` is the healthy result.
- **`errors=0` is the number that matters.** A non-zero count prints the first mismatching
  vector with its index, so you can regenerate just that case.
- Each run ends with `Verilog $finish` and a source line. That is the testbench terminating
  normally, not an error.

What each one isolates:

| Target | Top module | Isolates |
|---|---|---|
| m1 | `tb_rope_bf16_unit` | BF16 multiply and add against CVFPU |
| m2 | `tb_rope_phase_gen` | integer phase accumulation, wraparound, random vs sequential access |
| m3 | `tb_rope_sin_lut` | the quarter-wave sine table, reflection and quadrant sign logic |
| m4 | `tb_rope_datapath` | the whole rotation datapath end to end |
| m5 | `tb_rope_xif_coproc` | CORE-V-XIF decode, writeback, kill, backpressure, plus 9 SVA properties |

m5 is the one that matters for XIF changes; it binds the assertions in
`tb/rope_assertions.svh`.

### 6. `make m7` — the real core running real instructions

```bash
make m7
```

**The most important target if you are touching the processor.** It:

1. builds the bare-metal image (`make -C sw`): compiles `sw/test_rope.c` and `sw/crt0.S`
   with the CORE-V GCC, links against `sw/link.ld`, and objcopies to Verilog hex;
2. runs `make -C sw verify`, which disassembles the ELF and checks that the `.insn` encoding
   GCC actually emitted matches what `rope_pkg` decodes;
3. compiles a Verilator simulation of the **full CV32E40X core** + `cv32e40x_if_xif` +
   the coprocessor + a memory model, and runs the program on it.

This closes the loop: golden model → compiled vectors → real instructions → core → XIF →
coprocessor → writeback.

Expected:

```
Found 10 ROPE.ROT instruction(s) with opcode 0x0B.
  example @158: word=0x0117878b rd=x15 rs1=x15 rs2=x17 funct3=0 funct7=0
PASS: every custom-0 instruction matches rope_pkg's decode (opcode 0x0B, funct3 0, funct7 0).
...
tb_rope_core: loaded /home/jj/cv32e40x/rope-bf16/sw/test_rope.hex
=== BF16 RoPE coprocessor test ===
test_packing done
test_conversion done
test_rope_rot done: 64 vectors
test_dependent_chain done
test_head_vector done
RESULT: PASS

----------------------------------------------------------------
tb_rope_core: exit_code=0x00000000 after 13593 cycles
PASS: C test passed on the integrated core
```

Reading these:

- **`exit_code=0x00000000`** is the pass condition. Anything else is the C program's own
  failure code, and the preceding `test_*` lines tell you which subtest stopped printing.
- **`after 13593 cycles`** is your regression canary. A correctness-neutral core change that
  moves this number by more than a few percent changed the pipeline's behaviour — worth
  understanding before you move on. `test_dependent_chain` in particular is sensitive to
  stall and forwarding logic.
- `ld: warning: ... LOAD segment with RWX permissions` is expected and harmless for a
  bare-metal image with a single flat section.

### 7. `make demo-quick` — everything, with a summary

```bash
make demo-prep     # compile everything first (slow; do this before recording)
make demo-quick    # 20k vectors
make demo          # 200k vectors
```

Runs `demo.sh`, which drives all eight stages and prints a metrics table. Use this to
showcase that the design works; use the individual targets to debug.

`demo-prep` exists because the Verilator C++ compile dominates the wall clock. Pre-build,
and the recorded run is ~7 seconds.

Expected tail:

```
  Correctness
    M1  BF16 primitives      bit-exact, 21935 vectors, 0 mismatches
    M2  phase generator      bit-exact; sequential == random access (zero drift)
    M3  sine table           bit-exact; mirror symmetry exact at all 1024 indices
    M4  rotation datapath    bit-exact, 21961 vectors, 0 mismatches
    M5  XIF coprocessor      6/6 tests, all 7 SVA properties held
    M6  integration lint     0 errors, 0 warnings with the real core
    M7  C test on the core   PASS in 13593 cycles
  ...
  Wall-clock for this run: 7s

  ALL CHECKS PASSED
```

`ALL CHECKS PASSED` is the line to look for.

One cosmetic bug in that output: `demo.sh` prints *"all 7 SVA properties held"*, but
`tb/rope_assertions.svh` now binds **9**. The assertions all pass; only the count in the
banner is stale. Verify with `grep -c 'assert property' tb/rope_assertions.svh`.

### 7b. The datapath variants

```bash
make m4-fma      # UseFma=1     -- 4 FPU instances instead of 6
make m9          # UsePrefetch=1 -- 2 pipeline stages instead of 3
make m9-core     # the same, on the real core
make m9-tier1    # prefetch WITHOUT the value check -- must FAIL
```

Both variants default **off**, so `make test` and `make m7` exercise the original design.

**`UseFma`** replaces two of the four multipliers and both adders with two multipliers and
two FMAs. It is deliberately **not** bit-equal to Model A: it rounds twice per output
instead of three times, because `x*cos` is formed inside the FMA and never rounded on its
own. `m4-fma` therefore checks against **Model A'**
(`model/rope_ref.py:rope_fma_bf16`) with its own vector file. Measured 1.79x lower RMS
error against an FP64 reference.

**`UsePrefetch`** removes the operand-capture stage by holding `theta[i]` in a register
refilled one instruction ahead. The instruction encoding, operands and all existing
software are unchanged — `m` and `i` stop being inputs and become checks. Measured on the
integrated core:

```
make m7        exit_code=0  13593 cycles   200 WB stall cycles
make m9-core   exit_code=0  13466 cycles     0 WB stall cycles
```

The full architecture, the `commit_kill` hazard and the recovery mechanism are in
[`docs/PREFETCH.md`](../docs/PREFETCH.md).

### 8. The full sweeps

```bash
make m1-full     # m1 at NVEC=10000000
make m4-full     # m4 at NVEC=10000000
```

Ten million random vectors against the BF16 primitives and the datapath. Slow — run before
you claim bit-exactness, not in the edit loop.

---

## 4. A suggested loop for manual core changes

```bash
cd ~/cv32e40x/rope-bf16

# Before you touch anything — record the baseline.
make lint-integration && make m7          # note the cycle count

# ... edit rtl/cv32e40x_*.sv ...

make lint-integration                     # does it still elaborate?
make m7                                   # does it still execute correctly?
make test NVEC=2000                       # coprocessor unaffected?
make demo-quick                           # full sweep before committing
```

### The one parameter that changes timing: `NumPipeRegs`

`NumPipeRegs` sets how many pipeline registers sit inside the BF16 datapath, and it is the
single highest-leverage experiment in this design. It is **not** a correctness knob — every
testbench is bit-exact at both settings, because it changes when results appear, not what
they are.

| `NumPipeRegs` | Datapath latency | Issue→result | Stall when it occurs |
|---|---|---|---|
| 1 (previous default) | 3 cycles | 5 cycles | 3 cycles |
| **0** (current) | **1 cycle** | **3 cycles** | **1 cycle** |

The relationship is `stall = issue→result latency − 2`, where the 2 is the instruction's own
`ID → EX → WB` travel, which is free. `make m4` prints the datapath latency directly, so you
can confirm the left-hand column without reasoning about it.

**Two traps, both of which have already cost real debugging time here:**

- **The declaration in `rtl/rope_cv32e40x_wrapper.sv` is whitespace-padded**
  (`NumPipeRegs      = 1`, column-aligned with its neighbours). A natural
  `sed 's/NumPipeRegs = 0/NumPipeRegs = 1/'` silently does not match it, so the coprocessor
  changes and **the core path does not**. The symptom is two "different" builds with
  identical `md5sum` on the simulator binary. Use a whitespace-tolerant pattern and always
  `grep` the result back:
  ```bash
  perl -pi -e 's/NumPipeRegs\s*=\s*\K[01]/0/g' rtl/rope_*.sv tb/tb_*.sv
  grep -rn 'NumPipeRegs.*=' rtl/rope_*.sv | grep -v vendor
  ```
  Do **not** touch `rtl/vendor/cvfpu/**` — those default to 0 and take their value from ours.
- **Setting it to 0 makes the BF16 multiply and add combinational in one cycle**, creating
  the longest path in the design. Verilator has no notion of delay and will not tell you
  whether that closes timing, and there is no PDK or STA tool configured here. The 1-cycle
  stall is real; its frequency cost is **unmeasured**.

Three points worth internalising:

- **Record the m7 cycle count before you start.** It is the only number in the suite that
  detects a change that is correct but slower.
- **`make test` does not compile the core at all.** Passing it proves nothing about a core
  edit. `lint-integration` and `m7` are your core coverage.
- **Verilator caches aggressively per directory** (`build/obj_m1/`, `build/obj_m7/`, …).
  It tracks source dependencies correctly, but if you see a result that cannot be explained
  by your change, `make clean` and re-run before debugging further.

---

## 4b. What implementing the variants actually cost

Five defects surfaced while building these two variants. **Every one was found by running
the suite, none by review** — which is the argument for keeping the ladder cheap enough to
run often. They are recorded here because four of the five are the same shape.

### The recurring shape: speculative per-id state read as an edge

1. **`result_is_killed` read a registered flag.** A `commit_kill` arriving in the *same*
   cycle the result reached the output stage was missed, and the killed instruction wrote
   `rd`. Unreachable at a 5-cycle latency; immediate at 3. Fixed by ORing in a
   combinational same-cycle term (`commit_valid` has no ready, so it is safe to sample).
2. **The result FIFO gated its *push* on the kill flag.** That worked only because at
   three stages the commit always arrived before the result existed. At two stages the
   result wins, and the kill is ignored. Fixed by gating the FIFO **output** on committed
   state instead — correct at any latency. **This change applies to the default
   configuration too.**
3. **A credit leak.** Fix 2 created two retirement paths that can fire in the same cycle
   for *different* instructions — a drain at the FIFO head and a kill caught before the
   push. The accounting ORed them, losing a credit each time they coincided and slowly
   starving the coprocessor until it stopped accepting work. Now summed.
4. **The prefetch itself is speculative per-id state**, advanced on `accept` but killable
   afterwards. Without a defence, an interrupt mid-loop leaves every remaining pair of the
   head vector silently rotated by the wrong frequency. Two checks now catch it; see
   `docs/PREFETCH.md` §4-5.

If you add state that tracks an in-flight instruction, **assume the commit can arrive at
any time relative to your result, including the same cycle**, and hold per-id status as a
*level* rather than capturing an edge.

### And one testbench bug

5. **The M5 commit threads walked a free-running counter.** Whenever result-FIFO credit
   throttled the issue thread, the commit thread raced ahead and committed ids that had
   not been issued. `accept` clears the commit flag for a reused id, so those commits were
   lost and the instructions could never retire — the suite deadlocked at eight pending.
   The threads now `wait (sent > ...)`.

   This was legal-looking only because the old design never required a commit to release a
   result. Committing an id before it is issued is not valid CORE-V-XIF.

### The lesson for the numbers

`m9-tier1` exists because reasoning about which check mattered got it **backwards**. The
value check was assumed to be robustness hardening and the kill check mandatory; measuring
showed the opposite — with the value check compiled out, 6039 of 6066 vectors fail,
because the M5 stimulus is random `(m, i)` rather than a sequential walk. Keep a
configuration around that *proves* a safety mechanism is load-bearing.

---

## 5. When something fails

| Symptom | Where to look |
|---|---|
| `%Error` from Verilator during `lint-integration` | Real elaboration failure. The message names file and line. Do **not** add a waiver. |
| `errors=N` with N > 0 in m1–m5 | Testbench prints the first mismatching vector and its index. Reproduce with `make m4 NVEC=2000` and the vector file in `build/`. |
| m7 `exit_code` non-zero | Read the `test_*` lines above it — the last one printed is the subtest that failed. `sw/test_rope.c` names them. |
| m7 cycle count moved a lot | Pipeline behaviour changed. `test_dependent_chain` stresses stall/forwarding. See the stall model below — a change in coprocessor latency moves this number predictably; anything else moving it is worth explaining. |
| `sw verify` fails | GCC emitted an encoding `rope_pkg` does not decode. Compare `sw/test_rope.dis` against the decode constants in `rtl/rope_pkg.sv`. |
| Simulation hangs | `tb_rope_core.sv` has a cycle-limit watchdog; a hang usually means the coprocessor never asserted `result_valid`. |
| Result makes no sense | `make clean && make test NVEC=2000` |
| `%Error: Internal Error: attempted to destroy locked Thread Pool` | A Verilator 5.020 threading flake, not your design. `VLT_FLAGS` passes `-j 0`; re-run, or re-run with `-j 1` to confirm. Seen once in ~10 lint invocations. |

Full logs from a target land in `build/` (`prep_m1.log`, …) when run through `demo-prep`.
Run a target directly and the output goes to your terminal.

---

## 6. Synthesis — read this before trying Yosys

**Yosys reads the two generated ROMs, and nothing else.** An earlier version of this file
said it could read none of the design. That was too strong, and it mattered — the ROMs are
the part worth measuring.

What works, re-confirmed on 2026-09-20 with Yosys 0.33:

```bash
cd ~/cv32e40x/rope-bf16
yosys -p "read_verilog -sv rtl/rope_theta_rom.sv; synth -top rope_theta_rom; stat"
yosys -p "read_verilog -sv rtl/rope_sin_rom.sv;   synth -top rope_sin_rom;   stat"
```

| Module | Result | Outcome |
|---|---|---|
| `rope_theta_rom` | **synthesises** | 341 cells |
| `rope_sin_rom` | **synthesises** | 3,144 cells |
| `rope_sin_lut` | fails | `rtl/rope_pkg.sv:156: ERROR: syntax error, unexpected TOK_ID` |
| `rope_phase_gen` | fails | same error, same line |
| `rope_datapath` | fails | `rtl/vendor/common_cells/src/cf_math_pkg.sv:24: ERROR: ... TOK_AUTOMATIC` |

Logs for all five are in `build/syn/*.log`.

**This is the only real area data the project has**, and it settles a question the design
rested on. The sine table is 1024 × 16 = 16,384 bits and synthesises to 3,144 cells — about
**0.19 cells per bit**, roughly five times smaller than a naive per-bit estimate, because
the quarter-wave table is highly compressible logic rather than a memory array. The claim
that the LUT is cheap is now measured rather than asserted.

The failures are structural, not a flag you are missing. `rope_pkg.sv:156` is a
`function automatic` declared inside a package, and `cf_math_pkg.sv:24` is the same
construct in a vendored one; Yosys's Verilog-2005 reader elaborates neither. Anything
involving the core additionally needs `cv32e40x_if_xif`, a SystemVerilog **interface**,
which it also cannot read. Confirm the slang plugin is absent for yourself:

```bash
yosys -p "plugin -i slang"
# ERROR: Can't load module `./slang': /usr/lib/yosys/plugins/slang.so: cannot open shared object file
```

There is no `syn`/`yosys` target in the Makefile, and there should not be one until a real
elaborator is available. Four ways forward, in order of effort:

1. **Move `function automatic` out of `rope_pkg`.** The cheapest real unlock. Only two
   constructs block `rope_sin_lut`, `rope_phase_gen` and `rope_datapath`, and neither needs
   to live in a package. Hoisting them would take the synthesisable fraction of this design
   from two generated ROMs to most of the datapath, with no new tooling.
2. **Verilator lint as the structural check.** `make lint` and `make lint-integration`
   already elaborate the full design with a real SystemVerilog frontend. This will not give
   you area or timing, but it catches everything a synthesis elaboration would catch about
   the source. For an iterative RTL edit loop, this is the right tool.
3. **Yosys with the slang plugin** (`yosys-slang`). This is what actually works on this
   design — it is exactly what the ORFS flow was built around, with
   `SYNTH_HDL_FRONTEND = slang`. It requires building the plugin; the system Yosys does not
   have it.
4. **`syn/synth.tcl`** is a **Cadence Genus** script, not a Yosys one. Its own header says
   `STATUS: NOT YET RUN`. It needs `ROPE_LIB_PATH` pointing at a directory of `.lib` files,
   and there is no PDK configured here:
   ```bash
   ROPE_LIB_PATH=/path/to/libs genus -batch -f syn/synth.tcl
   ```
   It encodes the intended Milestone 9 flow — area, timing, and the area breakdown across
   sine LUT / theta ROM / CVFPU / phase logic / XIF wrapper — but it is untested end to end.

An OpenROAD/ORFS flow targeting asap7 was written for this repo and produced three synthesis
targets (`rope_cv32e40x_wrapper`, `cv32e40x_core`, `rope_datapath`). It was **never run** and
is not in the working tree — see §7.

---

## 7. What this suite does not cover

These exist but are currently in `git stash`, not in the working tree:

- **The streaming variant.** `rope_stream_fsm.sv`, `tb_rope_stream.sv` and the `m8` target —
  `ROPE.CFG` / `ROPE.POS` / `ROPE.START` / `ROPE.WAIT` driving a memory model on the XIF
  memory channel. The demo output above says so explicitly: *"Only ONE instruction was added.
  Four more are planned for the streaming variant."*
- **`make compare` and `make bench`** — cycle-count measurements on the real core, 64×
  `ROPE.ROT` versus one `ROPE.START`.
- **The OpenROAD/ORFS flow** (`BUILD`, `MODULE.bazel`, `orfs/`).
- **`docs/ONRAMP.md`** — the conceptual onramp to RoPE itself, which this file is the
  operational counterpart to.

### The same-cycle kill fix (applied 2026-09-22)

Dropping to `NumPipeRegs = 0` exposed a latent correctness bug, which is now fixed in
`rtl/rope_xif_coproc.sv`. Recorded here because the failure mode is instructive: it was
invisible at the old latency and `make m5` passed throughout.

The original code consulted only the registered kill flag:

```systemverilog
// HEAD -- consults only the REGISTERED kill flag
assign result_is_killed = killed_q[cur_tag.id];
```

`killed_q` is set by `commit_kill` and therefore only reflects kills from *previous* cycles.
If a kill arrives in the same cycle the instruction's tag reaches the output stage, it is
missed, and **a killed instruction writes back to `rd`** — architectural corruption. The fix
adds the combinational term (`commit_valid` has no ready signal, so it is safe to sample):

```systemverilog
assign same_cycle_kill  = xif_commit_if.commit_valid
                        & xif_commit_if.commit.commit_kill
                        & (xif_commit_if.commit.id == cur_tag.id);
assign result_is_killed = killed_q[cur_tag.id] | same_cycle_kill;
```

**`make m5` passed at `NumPipeRegs = 1` throughout, and that is the lesson.** At five
cycles of issue→result latency the kill always lands before the result, so the window never
opened. The moment the latency dropped to three, `a_no_writeback_after_kill` fired on the
first run (`SVA: writeback for KILLED id=1`). A green m5 at one latency setting is not
evidence that the kill path is sound at another — the assertions are only as good as the
timing the testbench happens to produce.

### Restoring the stashes

They are split across two, which must be restored **together** — `stash@{1}` holds the
modified `rope-bf16/Makefile` that defines `m8`/`compare`/`bench` (plus `NumPipeRegs = 0`
and the kill fix above), and `stash@{0}` holds the untracked source files those targets
reference:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd ~/cv32e40x && git stash list"
```

Restoring only one of the two leaves the Makefile referencing files that do not exist, or
files with no targets that build them.

**`stash@{1}` is polluted with line-ending churn.** `git stash show --name-status
'stash@{1}'` lists ~150 files as modified, including `LICENSE`, the GitHub issue templates
and the user-manual SVGs — none of which anyone edited. That is the Windows/WSL
`core.autocrlf` problem from §1 captured into a stash. The real content changes are confined
to `rope-bf16/**` and are worth extracting individually rather than popping the whole thing:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd ~/cv32e40x && git diff 'stash@{1}' -- rope-bf16/"
```

### Measured stall behaviour (for interpreting the m7 cycle count)

Not a test — a characterisation, from a per-PC probe on `wb_stage_i.xif_waiting` over
`sw/bench_rope.c`. Recorded here because the m7 cycle count is the suite's only performance
signal and is otherwise hard to interpret.

`ROPE.ROT` stalls in the **WB stage**. The offloaded instruction sits in WB with its result
not yet returned, so [`cv32e40x_wb_stage.sv`](../../rtl/cv32e40x_wb_stage.sv) drops
`wb_ready_o`, freezing EX, ID and IF behind it:

```systemverilog
assign xif_waiting = ex_wb_pipe_i.instr_valid && ex_wb_pipe_i.xif_en
                     && !xif_result_if.result_valid;
assign wb_ready_o  = ctrl_fsm_i.kill_wb || (lsu_ready_i && !xif_waiting && ...);
```

Stalls are **always whole cycles and always the same length** — **1 cycle** in the current
configuration (3 at the old `NumPipeRegs = 1`). What varies is how often:

| Code shape | Stall rate | Why |
|---|---|---|
| dependent chain | **100%** | the next `ROPE.ROT` cannot issue until the previous writes back, so nothing is in flight to absorb the delay |
| 4-way independent | **1 in 3** | a stall freezes IF/ID/EX, and the two instructions already in flight between ID and WB gain the cycle they needed; the third does not |

The 1-in-3 is a property of pipeline depth — there are exactly 2 stages between ID and WB,
so exactly 2 instructions benefit per stall. It is a floor, not an average over a
distribution: in a per-PC probe of the unrolled loop, three specific PCs stalled on
**200 of 200** executions and the other five on **0 of 200**.

This means a fractional "cycles per instruction" figure is a *frequency*, never a duration.
Reaching zero stalls needs issue→result latency ≤ 2. `NumPipeRegs = 0` reaches 3, which is
the floor for this datapath: one stage to register the operands and LUT output, and one for
the result FIFO. Going lower means precomputing the phase and LUT lookup inside the
coprocessor — the unbuilt change that would eliminate the stall rather than shorten it.

Also not covered by anything here: the core's own upstream verification (`tb/`, the UVM
environment, RVFI checks). This suite verifies the coprocessor and its integration, not the
CV32E40X's baseline RISC-V compliance. A core change that breaks something unrelated to the
XIF path will pass every target in this document.

---

## 9. Reproducing the variant results

Every number quoted for the two datapath variants, in the order it was produced. All of
these run from `~/cv32e40x/rope-bf16` and need only Verilator and numpy, except `m7` /
`m9-core`, which also need the RISC-V toolchain.

### 9.1 The baseline must be clean first

```bash
make lint                 # 0 errors, 0 warnings
make lint-integration     # 0 errors, 0 warnings
make model                # includes test_bf16_fma.py
make test                 # M1-M5, the shipping configuration
```

Expected: `errors=0` on every line, `All golden-model tests passed.`

### 9.2 The fused multiply-add datapath

```bash
make m4-fma
```

```
tb_rope_datapath: sent=201961 checked=201961 errors=0
PASS: datapath bit-exact vs Model A' (fused multiply-add) on all 201961 vectors
```

It checks against **Model A'**, not Model A -- the FMA rounds twice per output instead of
three times, so bit-equality with the shipping datapath would mean the variant was not
doing anything. The accuracy claim comes from the model tests:

```bash
cd model && python3 test_bf16_fma.py
```

```
RMS rel error, current: 6.8086e-03
RMS rel error, fused  : 3.8110e-03
improvement           : 1.787x
```

That script also cross-checks the vectorised FMA against exact integer arithmetic over
43,375 triples including a full special-value cross product, and reports quadrant coverage
for the sign study.

### 9.3 The two-stage prefetch

```bash
make m9 NVEC=2000     # fast smoke run
make m9               # full, 200k vectors
```

```
tb_rope_xif_coproc: sent=6067 checked=6066 errors=0
PASS: XIF coprocessor -- decode, writeback, kill, throughput, backpressure
```

`m9` runs **the same testbench and the same vectors as `m5`** -- semantics are unchanged,
so a difference between them is a bug, not a new reference. Compare directly:

```bash
make m5 && make m9
```

### 9.4 The test that must fail

```bash
make m9-tier1
```

```
tb_rope_xif_coproc: sent=6067 checked=6066 errors=6039
FAIL: 6039 errors
```

**This failing is the correct result.** It builds the prefetch with `UseValueCheck=0`, and
6039 of 6066 vectors are wrong because the M5 stimulus is random `(m, i)` rather than a
sequential walk. If it ever passes, the value check has stopped doing its job.

### 9.5 On the integrated core

```bash
make m7          # 3-stage baseline
make m9-core     # 2-stage prefetch
```

```
m7      : exit_code=0x00000000 after 13593 cycles
m9-core : exit_code=0x00000000 after 13466 cycles
```

Both must print `RESULT: PASS` and `exit_code=0x00000000`. The 127-cycle difference is the
headline number.

### 9.6 Measuring the stall directly

The zero-stall claim came from a temporary probe, which is **not** in the tree -- it is
reproduced here so the number can be re-derived rather than trusted.

Add to `tb/tb_rope_core.sv`, just before the `rope_cv32e40x_wrapper` instantiation:

```systemverilog
  int unsigned stall_cycles = 0, stall_events = 0, rot_accepts = 0;
  logic        waiting_d_p  = 1'b0;
  always @(posedge clk) begin
    if (i_dut.i_core.wb_stage_i.xif_waiting) begin
      stall_cycles <= stall_cycles + 1;
      if (!waiting_d_p) stall_events <= stall_events + 1;
    end
    if (i_dut.i_rope.accept) rot_accepts <= rot_accepts + 1;
    waiting_d_p <= i_dut.i_core.wb_stage_i.xif_waiting;
  end
  task automatic dump_stalls();
    $display("STALLPROBE accepts=%0d stall_events=%0d stall_cycles=%0d",
             rot_accepts, stall_events, stall_cycles);
  endtask
```

and call `dump_stalls();` immediately before the final
`$display("PASS: C test passed on the integrated core");`. Then:

```bash
make m7 2>&1 | grep STALLPROBE
make m9-core 2>&1 | grep STALLPROBE
```

```
3-stage:  accepts=200  stall_events=200  stall_cycles=200
2-stage:  accepts=200  stall_events=0    stall_cycles=0
```

**Remember to remove the probe afterwards.** It reaches into the DUT hierarchy and will
break if the wrapper or the core's WB stage is renamed.

The two numbers reconcile: 200 stall cycles removed, 73 resync bubbles added for the
rotations that do not walk sequentially (`test_rope_rot` uses random `(m, i)`,
`test_dependent_chain` repeats one `i`), and `200 - 73 = 127`.

### 9.7 Logic depth, for the timing argument

```bash
yosys -p "read_verilog -sv rtl/rope_sin_rom.sv;   synth -top rope_sin_rom;   ltp"
yosys -p "read_verilog -sv rtl/rope_theta_rom.sv; synth -top rope_theta_rom; ltp"
```

```
Longest topological path in rope_sin_rom   (length=20)
Longest topological path in rope_theta_rom (length=6)
```

These two are the only numbers in the stage-depth argument that are **measured**. The
multiply and the FPU are estimates, because Yosys cannot parse `rope_phase_gen`,
`rope_sin_lut` or anything containing `fpnew_fma` -- see section 6.

### 9.8 Everything, in one go

```bash
make lint && make lint-integration && make model && make test \
  && make m4-fma && make m9 && make m7 && make m9-core \
  && ! make m9-tier1 && echo "ALL AS EXPECTED"
```

Note the `!` before `m9-tier1`: that target is *supposed* to fail, so the chain inverts it.
