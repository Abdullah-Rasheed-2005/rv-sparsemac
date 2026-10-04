/* link.ld - everything lives in the first 64 KB of RAM, the CPU starts at 0 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 0x00010000
}

SECTIONS
{
    .text   : { *(.text.start) *(.text .text.*) } > RAM
    .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > RAM
    .data   : { *(.data .data.* .sdata .sdata.*) } > RAM
    .bss    : {
        . = ALIGN(4);
        _bss_start = .;
        *(.bss .bss.* .sbss .sbss.* COMMON)
        . = ALIGN(4);
        _bss_end = .;
    } > RAM
    _stack_top = ORIGIN(RAM) + LENGTH(RAM);
}
