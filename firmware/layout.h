/* layout.h - Memory map of the simulated system (firmware and testbench)
 *
 * The testbench (hardware/tb/tb_soc.v) copies the data files into RAM at these
 * addresses before the CPU starts. The Makefile turns this header into
 * build/layout.vh so the testbench always uses the same numbers.
 * Only plain "#define NAME 0xHEX" lines are allowed here (a script reads them).
 */
#ifndef LAYOUT_H
#define LAYOUT_H

/* ---- RAM: 256 KB at address 0 ---- */
#define RAM_SIZE      0x00040000
#define STACK_TOP     0x00010000   /* firmware: code + data + stack live below 64 KB */

/* ---- data copied into RAM by the testbench ---- */
#define W1_BASE       0x00010000   /* fc1 weights, int8, 64 x 784 (neuron after neuron) */
#define W2_BASE       0x0001C400   /* fc2 weights, int8, 10 x 64 */
#define B1_BASE       0x0001C800   /* fc1 bias, int32 x 64 */
#define B2_BASE       0x0001C900   /* fc2 bias, int32 x 10 */
#define SHIFT_ADDR    0x0001CA00   /* hidden right-shift, 1 byte */
#define LABELS_BASE   0x0001CB00   /* true digits, 1 byte per image */
#define IMAGES_BASE   0x00020000   /* test images, 784 bytes each */

/* ---- memory-mapped "devices" of the testbench ---- */
#define MMIO_CONSOLE  0x10000000   /* write a byte: it is printed */
#define MMIO_RESULT   0x10000010   /* write a word: the testbench collects it */
#define MMIO_EXIT     0x10000020   /* write the exit code: the simulation ends */

#endif
