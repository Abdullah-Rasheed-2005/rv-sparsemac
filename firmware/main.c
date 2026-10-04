/* main.c - MNIST inference on PicoRV32, with and without the accelerator
 *
 * Phase 1 (hardware): for NHW test images
 *     load the pixels with SMAC.LDW, start with SMAC.RUN, read the logits with
 *     SMAC.LOGIT. The accuracy and the cycle counts are printed.
 * Phase 2 (software baseline): run the SAME integer network in plain C on the
 *     CPU, in three flavours:
 *       - dense:      every pixel times every weight (the obvious program),
 *       - skip zeros: skips zero pixels (a smarter program, still dense weights),
 *       - CSC:        the accelerator's own algorithm (weights stored by column,
 *                     zero inputs and zero weights are both skipped). This is
 *                     the fair comparison: same work, CPU instead of hardware.
 *     The logits must be identical to the accelerator's, the cycles are compared.
 *
 * The network is the one of software/export_int8.py:
 *     hidden = min( max( pixels * W1 + b1, 0 ) >> shift, 127 )
 *     logits = hidden * W2 + b2          prediction = first maximum
 *
 * No division and no library function is used (the core has no divider and
 * there is no C library), so the printing code below counts by subtraction.
 */
#include <stdint.h>
#include "layout.h"
#include "smac.h"

#ifndef NHW
#define NHW 100          /* images run on the accelerator (max 100) */
#endif
#ifndef NSW
#define NSW 2            /* images run by the "skip zero pixels" software */
#endif
#ifndef NSW_DENSE
#define NSW_DENSE 1      /* images run by the plain dense software (slow!) */
#endif
#ifndef NSW_CSC
#define NSW_CSC 100      /* images run by the CSC software (the accelerator's algorithm) */
#endif
#define NCMP 100         /* logits kept from the accelerator for comparison */

/* ------------------------------------------------------------------ output */
static void put_char(char c) { *(volatile uint32_t *)MMIO_CONSOLE = (uint32_t)c; }

static void put_str(const char *s) { while (*s) put_char(*s++); }

static void put_dec(uint32_t v)
{
    static const uint32_t pow10[10] = { 1000000000u, 100000000u, 10000000u, 1000000u,
                                        100000u, 10000u, 1000u, 100u, 10u, 1u };
    int started = 0;
    for (int i = 0; i < 10; i++) {
        uint32_t d = 0;
        while (v >= pow10[i]) { v -= pow10[i]; d++; }
        if (d || started || i == 9) { put_char((char)('0' + d)); started = 1; }
    }
}

static void put_line(const char *name, uint32_t value)
{
    put_str(name);
    put_dec(value);
    put_char('\n');
}

static void send_result(uint32_t w) { *(volatile uint32_t *)MMIO_RESULT = w; }

/* ------------------------------------------------------- software inference */
/* Plain dense C, exactly what a first attempt would look like. */
static uint32_t sw_dense(const uint8_t *img, int32_t *logit)
{
    const int8_t  *w1 = (const int8_t  *)W1_BASE;
    const int32_t *b1 = (const int32_t *)B1_BASE;
    const int8_t  *w2 = (const int8_t  *)W2_BASE;
    const int32_t *b2 = (const int32_t *)B2_BASE;
    const uint32_t shift = *(const volatile uint8_t *)SHIFT_ADDR;
    int32_t hid[64];

    for (int n = 0; n < 64; n++) {
        const int8_t *w = w1 + n * 784;
        int32_t acc = b1[n];
        for (int j = 0; j < 784; j++)
            acc += (int32_t)img[j] * w[j];
        if (acc < 0) acc = 0;
        acc >>= shift;
        hid[n] = acc > 127 ? 127 : acc;
    }
    uint32_t best = 0;
    for (int o = 0; o < 10; o++) {
        const int8_t *w = w2 + o * 64;
        int32_t acc = b2[o];
        for (int j = 0; j < 64; j++)
            acc += hid[j] * w[j];
        logit[o] = acc;
        if (o == 0 || acc > logit[best]) best = (uint32_t)o;
    }
    return best;
}

/* Same network, but zero pixels (about 80 %) and zero hidden values are skipped.
 * The loops are swapped (pixel outside, neuron inside), so a weight is read with
 * a stride of 784 bytes. */
static uint32_t sw_skip(const uint8_t *img, int32_t *logit)
{
    const int8_t  *w1 = (const int8_t  *)W1_BASE;
    const int32_t *b1 = (const int32_t *)B1_BASE;
    const int8_t  *w2 = (const int8_t  *)W2_BASE;
    const int32_t *b2 = (const int32_t *)B2_BASE;
    const uint32_t shift = *(const volatile uint8_t *)SHIFT_ADDR;
    int32_t acc[64];
    int32_t hid[64];

    for (int n = 0; n < 64; n++) acc[n] = b1[n];
    for (int j = 0; j < 784; j++) {
        int32_t x = img[j];
        if (x == 0) continue;
        const int8_t *w = w1 + j;
        for (int n = 0; n < 64; n++, w += 784)
            acc[n] += x * *w;
    }
    for (int n = 0; n < 64; n++) {
        int32_t a = acc[n];
        if (a < 0) a = 0;
        a >>= shift;
        hid[n] = a > 127 ? 127 : a;
    }

    int32_t out[10];
    for (int o = 0; o < 10; o++) out[o] = b2[o];
    for (int j = 0; j < 64; j++) {
        int32_t x = hid[j];
        if (x == 0) continue;
        const int8_t *w = w2 + j;
        for (int o = 0; o < 10; o++, w += 64)
            out[o] += x * *w;
    }
    uint32_t best = 0;
    for (int o = 0; o < 10; o++) {
        logit[o] = out[o];
        if (o == 0 || out[o] > out[best]) best = (uint32_t)o;
    }
    return best;
}

/* One layer of the accelerator's algorithm (see hardware/rtl/sparse_mlp_zs.v):
 *     for every NONZERO input j:  for every NONZERO weight in column j:
 *         acc[row] += x[j] * weight
 * Column j owns the entries ptr[j] .. ptr[j+1]-1 (software/export_sparse.py). */
static void layer_csc(const uint8_t *x, int n_in, const uint16_t *ptr,
                      const uint8_t *row, const int8_t *val, int32_t *acc)
{
    for (int j = 0; j < n_in; j++) {
        int32_t a = x[j];
        if (a == 0) continue;
        uint32_t end = ptr[j + 1];
        for (uint32_t e = ptr[j]; e < end; e++)
            acc[row[e]] += a * val[e];
    }
}

/* Same network, same algorithm and same weight format as the accelerator. */
static uint32_t sw_csc(const uint8_t *img, int32_t *logit)
{
    const int32_t *b1 = (const int32_t *)B1_BASE;
    const int32_t *b2 = (const int32_t *)B2_BASE;
    const uint32_t shift = *(const volatile uint8_t *)SHIFT_ADDR;
    int32_t acc[64];
    uint8_t hid[64];

    for (int n = 0; n < 64; n++) acc[n] = b1[n];
    layer_csc(img, 784, (const uint16_t *)C1PTR_BASE, (const uint8_t *)C1ROW_BASE,
              (const int8_t *)C1VAL_BASE, acc);
    for (int n = 0; n < 64; n++) {
        int32_t a = acc[n];
        if (a < 0) a = 0;
        a >>= shift;
        hid[n] = (uint8_t)(a > 127 ? 127 : a);
    }

    for (int o = 0; o < 10; o++) logit[o] = b2[o];
    layer_csc(hid, 64, (const uint16_t *)C2PTR_BASE, (const uint8_t *)C2ROW_BASE,
              (const int8_t *)C2VAL_BASE, logit);
    uint32_t best = 0;
    for (int o = 1; o < 10; o++)
        if (logit[o] > logit[best]) best = (uint32_t)o;
    return best;
}

/* -------------------------------------------------------------------- main */
int main(void)
{
    const uint8_t *images = (const uint8_t *)IMAGES_BASE;
    const uint8_t *labels = (const uint8_t *)LABELS_BASE;

    static int32_t hw_logit[NCMP][10];
    static uint32_t hw_pred[NCMP];

    uint32_t t_load = 0, t_run = 0, t_read = 0, t_acc = 0, correct = 0;

    put_str("sparse-mac on PicoRV32\n");

    /* ------------------------------------------------ phase 1: accelerator */
    for (uint32_t i = 0; i < NHW; i++) {
        const uint32_t *px = (const uint32_t *)(images + i * 784);

        uint32_t c0 = rdcycle();
        #pragma GCC unroll 4
        for (uint32_t w = 0; w < 196; w++)
            smac_ldw(w * 4, px[w]);
        uint32_t c1 = rdcycle();

        uint32_t pred = smac_run();
        uint32_t c2 = rdcycle();

        for (uint32_t c = 0; c < 10; c++) {
            int32_t v = smac_logit(c);
            send_result((uint32_t)v);
            if (i < NCMP) hw_logit[i][c] = v;
        }
        send_result(pred);
        uint32_t c3 = rdcycle();

        t_load += c1 - c0;
        t_run  += c2 - c1;
        t_read += c3 - c2;
        t_acc  += smac_cycles();
        if (pred == labels[i]) correct++;
        if (i < NCMP) hw_pred[i] = pred;
    }

    put_str("== accelerator (CPU + SMAC instructions) ==\n");
    put_line("images:                      ", NHW);
    put_line("correct:                     ", correct);
    put_line("cycles, load pixels (total): ", t_load);
    put_line("cycles, SMAC.RUN    (total): ", t_run);
    put_line("cycles, read logits (total): ", t_read);
    put_line("cycles, accelerator (total): ", t_acc);
    put_line("cycles, all         (total): ", t_load + t_run + t_read);

    /* ------------------------------------------------ phase 2: software    */
    uint32_t mismatches = 0;

    uint32_t s_cyc = 0;
    for (uint32_t i = 0; i < NSW && i < NCMP; i++) {
        int32_t lg[10];
        uint32_t c0 = rdcycle();
        uint32_t pred = sw_skip(images + i * 784, lg);
        s_cyc += rdcycle() - c0;
        if (pred != hw_pred[i]) mismatches++;
        for (int c = 0; c < 10; c++)
            if (lg[c] != hw_logit[i][c]) mismatches++;
    }
    uint32_t d_cyc = 0;
    for (uint32_t i = 0; i < NSW_DENSE && i < NCMP; i++) {
        int32_t lg[10];
        uint32_t c0 = rdcycle();
        uint32_t pred = sw_dense(images + i * 784, lg);
        d_cyc += rdcycle() - c0;
        if (pred != hw_pred[i]) mismatches++;
        for (int c = 0; c < 10; c++)
            if (lg[c] != hw_logit[i][c]) mismatches++;
    }

    uint32_t c_cyc = 0;
    for (uint32_t i = 0; i < NSW_CSC && i < NCMP; i++) {
        int32_t lg[10];
        uint32_t c0 = rdcycle();
        uint32_t pred = sw_csc(images + i * 784, lg);
        c_cyc += rdcycle() - c0;
        if (pred != hw_pred[i]) mismatches++;
        for (int c = 0; c < 10; c++)
            if (lg[c] != hw_logit[i][c]) mismatches++;
    }

    put_str("== software on the CPU only ==\n");
    put_line("csc-sparse images:           ", NSW_CSC);
    put_line("csc-sparse cycles (total):   ", c_cyc);
    put_line("skip-zeros images:           ", NSW);
    put_line("skip-zeros cycles (total):   ", s_cyc);
    put_line("dense images:                ", NSW_DENSE);
    put_line("dense cycles (total):        ", d_cyc);
    put_line("software/accelerator mismatches: ", mismatches);

    put_str(mismatches == 0 ? "FIRMWARE: OK\n" : "FIRMWARE: MISMATCH\n");
    return mismatches == 0 ? 0 : 1;
}
