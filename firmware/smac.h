/* smac.h - The five custom instructions of the sparse accelerator
 *
 * They are encoded with the assembler directive .insn (R-type):
 *     .insn r  opcode, funct3, funct7, rd, rs1, rs2
 * opcode 0x0B = RISC-V "custom-0". See hardware/rtl/sparsemac_pcpi.v for the
 * meaning of every instruction.
 */
#ifndef SMAC_H
#define SMAC_H
#include <stdint.h>

/* SMAC.LDW: write 4 pixels. pixel_addr = 0,4,8,...,780 in this order. */
static inline void smac_ldw(uint32_t pixel_addr, uint32_t four_pixels)
{
    asm volatile (".insn r 0x0b, 0, 0, zero, %0, %1" :: "r"(pixel_addr), "r"(four_pixels));
}

/* SMAC.RUN: run the whole inference, return the predicted digit. */
static inline uint32_t smac_run(void)
{
    uint32_t pred;
    asm volatile (".insn r 0x0b, 1, 0, %0, zero, zero" : "=r"(pred));
    return pred;
}

/* SMAC.LOGIT: logit number c (0..9) of the last run. */
static inline int32_t smac_logit(uint32_t c)
{
    int32_t v;
    asm volatile (".insn r 0x0b, 2, 0, %0, %1, zero" : "=r"(v) : "r"(c));
    return v;
}

/* SMAC.CYC: clock cycles the accelerator needed for the last run. */
static inline uint32_t smac_cycles(void)
{
    uint32_t c;
    asm volatile (".insn r 0x0b, 3, 0, %0, zero, zero" : "=r"(c));
    return c;
}

/* SMAC.CFG: bound the run time. budget = layer-1 cycle budget (0 = no limit),
 * const_time = 1 makes every SMAC.RUN take exactly the same number of cycles. */
static inline void smac_cfg(uint32_t budget, uint32_t const_time)
{
    asm volatile (".insn r 0x0b, 4, 0, zero, %0, %1" :: "r"(budget), "r"(const_time));
}

/* rdcycle written as a raw CSR read (csrrs rd, cycle, x0) so that it assembles
 * with every binutils version, with or without the Zicsr extension flag. */
static inline uint32_t rdcycle(void)
{
    uint32_t c;
    asm volatile (".insn i 0x73, 2, %0, zero, -1024" : "=r"(c));
    return c;
}

#endif
