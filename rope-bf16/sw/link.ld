/* ---------------------------------------------------------------------------
 * link.ld -- memory map for the bare-metal RoPE test.
 *
 * Matches tb_rope_core.sv:
 *   0x00000000  RAM (instructions + data), 256 KB
 *   0x10000000  putchar   (write-only)
 *   0x20000000  exit code (write-only)
 *
 * The core boots at boot_addr_i, which the testbench drives to 0x00000080 (the reset
 * vector must be 128-bit aligned per the CV32E40X user manual, and low addresses are
 * left free for the vector table).
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
