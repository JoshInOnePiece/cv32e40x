# Verification onramp

**Audience:** you are about to change the CV32E40X core or the RoPE coprocessor RTL by
hand, and you want to know whether it still works before and after. This is the operational
guide: where to stand, what to type, what it does, and what a healthy run prints.

Everything in this file was executed on 2026-09-20 against commit `fe84f3c8`. The expected
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
| Yosys | nothing that currently works — see §6 | 0.33 installed, **cannot read this RTL** |

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
| m5 | `tb_rope_xif_coproc` | CORE-V-XIF decode, writeback, kill, backpressure, plus 7 SVA properties |

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
tb_rope_core: exit_code=0x00000000 after 13993 cycles
PASS: C test passed on the integrated core
```

Reading these:

- **`exit_code=0x00000000`** is the pass condition. Anything else is the C program's own
  failure code, and the preceding `test_*` lines tell you which subtest stopped printing.
- **`after 13993 cycles`** is your regression canary. A correctness-neutral core change that
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
    M7  C test on the core   PASS in 13993 cycles
  ...
  Wall-clock for this run: 7s

  ALL CHECKS PASSED
```

`ALL CHECKS PASSED` is the line to look for.

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

Three points worth internalising:

- **Record the m7 cycle count before you start.** It is the only number in the suite that
  detects a change that is correct but slower.
- **`make test` does not compile the core at all.** Passing it proves nothing about a core
  edit. `lint-integration` and `m7` are your core coverage.
- **Verilator caches aggressively per directory** (`build/obj_m1/`, `build/obj_m7/`, …).
  It tracks source dependencies correctly, but if you see a result that cannot be explained
  by your change, `make clean` and re-run before debugging further.

---

## 5. When something fails

| Symptom | Where to look |
|---|---|
| `%Error` from Verilator during `lint-integration` | Real elaboration failure. The message names file and line. Do **not** add a waiver. |
| `errors=N` with N > 0 in m1–m5 | Testbench prints the first mismatching vector and its index. Reproduce with `make m4 NVEC=2000` and the vector file in `build/`. |
| m7 `exit_code` non-zero | Read the `test_*` lines above it — the last one printed is the subtest that failed. `sw/test_rope.c` names them. |
| m7 cycle count moved a lot | Pipeline behaviour changed. `test_dependent_chain` stresses stall/forwarding. |
| `sw verify` fails | GCC emitted an encoding `rope_pkg` does not decode. Compare `sw/test_rope.dis` against the decode constants in `rtl/rope_pkg.sv`. |
| Simulation hangs | `tb_rope_core.sv` has a cycle-limit watchdog; a hang usually means the coprocessor never asserted `result_valid`. |
| Result makes no sense | `make clean && make test NVEC=2000` |

Full logs from a target land in `build/` (`prep_m1.log`, …) when run through `demo-prep`.
Run a target directly and the output goes to your terminal.

---

## 6. Synthesis — read this before trying Yosys

**Plain Yosys cannot read this RTL.** Yosys 0.33 is installed, and its built-in Verilog
frontend fails on the first vendored file. These are the actual errors, reproduced from
`build/syn/*.log`:

```
rtl/vendor/common_cells/src/cf_math_pkg.sv:24: ERROR: syntax error, unexpected TOK_AUTOMATIC
rtl/rope_pkg.sv:156: ERROR: syntax error, unexpected TOK_ID
```

The cause is structural, not a flag you are missing. The design uses SystemVerilog packages,
`function automatic`, and — for anything involving the core — `cv32e40x_if_xif`, a
SystemVerilog **interface**. Yosys's Verilog-2005 reader elaborates none of that. Confirm
for yourself:

```bash
yosys -p "plugin -i slang"
# ERROR: Can't load module `./slang': /usr/lib/yosys/plugins/slang.so: cannot open shared object file
```

There is no `syn`/`yosys` target in the Makefile, and there should not be one until a real
elaborator is available. Three ways forward, in order of effort:

1. **Verilator lint as the structural check.** `make lint` and `make lint-integration`
   already elaborate the full design with a real SystemVerilog frontend. This will not give
   you area or timing, but it catches everything a synthesis elaboration would catch about
   the source. For an iterative RTL edit loop, this is the right tool.
2. **Yosys with the slang plugin** (`yosys-slang`). This is what actually works on this
   design — it is exactly what the ORFS flow was built around, with
   `SYNTH_HDL_FRONTEND = slang`. It requires building the plugin; the system Yosys does not
   have it.
3. **`syn/synth.tcl`** is a **Cadence Genus** script, not a Yosys one. Its own header says
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

They are split across two stashes, which must be restored **together** — one holds the
modified `rope-bf16/Makefile` that defines `m8`/`compare`/`bench`, the other holds the source
files those targets reference:

```bash
wsl -d Ubuntu-24.04 -- bash -lc "cd ~/cv32e40x && git stash list"
```

Restoring only one of the two leaves the Makefile referencing files that do not exist, or
files with no targets that build them.

Also not covered by anything here: the core's own upstream verification (`tb/`, the UVM
environment, RVFI checks). This suite verifies the coprocessor and its integration, not the
CV32E40X's baseline RISC-V compliance. A core change that breaks something unrelated to the
XIF path will pass every target in this document.
