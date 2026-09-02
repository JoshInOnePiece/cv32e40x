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
