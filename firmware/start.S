# start.S - reset code: stack, clear .bss, call main, report the exit code
    .section .text.start
    .globl _start
_start:
    la   sp, _stack_top
    la   t0, _bss_start
    la   t1, _bss_end
1:  bgeu t0, t1, 2f
    sw   zero, 0(t0)
    addi t0, t0, 4
    j    1b
2:  call main
    li   t0, 0x10000020        # MMIO_EXIT: tell the testbench we are done
    sw   a0, 0(t0)
3:  j    3b
