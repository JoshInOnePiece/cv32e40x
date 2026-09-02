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

    path = sys.argv[1]
    with open(path) as f:
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
            print(
                f"FAIL {addr}: word=0x{word:08x} opcode=0x{d['opcode']:02x} "
                f"funct3={d['funct3']} funct7={d['funct7']}  ({text})"
            )

    sample = found[0]
    d = decode(sample[1])
    print(f"Found {len(found)} ROPE.ROT instruction(s) with opcode 0x{OPCODE:02X}.")
    print(
        f"  example @{sample[0]}: word=0x{sample[1]:08x} "
        f"rd=x{d['rd']} rs1=x{d['rs1']} rs2=x{d['rs2']} "
        f"funct3={d['funct3']} funct7={d['funct7']}"
    )
    print(f"  raw text: {sample[2]}")

    if bad:
        print(f"FAIL: {bad} instruction(s) with the wrong funct3/funct7.")
        return 1

    print("PASS: every custom-0 instruction matches rope_pkg's decode "
          "(opcode 0x0B, funct3 0, funct7 0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
