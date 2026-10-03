# Architecture

## 1. The neural network

```
input (784 pixels, 0..255) -> fc1 (784 x 64) -> ReLU -> fc2 (64 x 10) -> argmax
```

- fc1 has 50,176 weights (98.7% of the model). This is where pruning matters.
- fc2 has 640 weights. It is small and sensitive, so it stays dense.

## 2. Integer-only inference

This is the exact computation the hardware must do. It is implemented in
`software/export_int8.py` and checked in `software/verify_export.py`.

```
acc1   = sum over 784 inputs of (pixel * w1) + bias1      # 32-bit signed
hidden = min( max(acc1, 0) >> SHIFT , 127 )               # ReLU, shift, saturate
acc2   = sum over 64 inputs of (hidden * w2) + bias2      # 32-bit signed
class  = index of the largest acc2
```

| Item | Type | Range |
|---|---|---|
| pixel | unsigned 8-bit | 0..255 |
| hidden value | unsigned 7-bit | 0..127 |
| weight | signed 8-bit | -127..127 |
| accumulator | signed 32-bit | worst case needs 26 bits |

`SHIFT` is chosen once by looking at the largest hidden value (calibration).
For the 80% pruned model it is 13. A right shift replaces a division, which is
cheap in hardware.

Because the activations are **unsigned** and the weights are **signed**, the
MAC unit multiplies unsigned by signed. `mac.v` adds a `0` bit in front of the
activation to make it a positive signed number.

## 3. Where the zeros are

| Source of zeros | Share | Known when |
|---|---|---|
| fc1 weights (after pruning) | 80% | before running (offline) |
| input pixels | ~81% | per image (at run time) |
| hidden values after ReLU | ~42% | per image (at run time) |

Weight zeros can be removed ahead of time by storing only the nonzero weights
with their positions. Activation zeros must be detected while running.

## 4. Planned hardware

```
+--------------------+   custom instruction   +-------------------------+
|   PicoRV32 core    | ---------------------> |  Sparse MAC accelerator |
|  (RV32I, PCPI port)| <--------------------- |  (PCPI co-processor)    |
+--------------------+       result           +-------------------------+
                                                 |  MAC unit (mac.v)
                                                 |  sparse dot-product engine
                                                 |  weight memory (hex files)
```

The core is [PicoRV32](https://github.com/YosysHQ/picorv32). It has a
co-processor port called PCPI. When the core sees an instruction it does not
know, it sends it out on the PCPI signals (`pcpi_valid`, `pcpi_insn`,
`pcpi_rs1`, `pcpi_rs2`). The accelerator answers with `pcpi_wr`, `pcpi_rd`,
`pcpi_wait` and `pcpi_ready`.

The planned custom instruction is `SPARSEMAC rd, rs1, rs2` in the RISC-V
custom-0 opcode space.

Status: only the MAC unit exists today. Everything else in this section is a
plan and may change.

## 5. Verification method

1. Python runs the integer network and writes `golden_logits.txt`.
2. `verify_export.py` re-reads the hex files and checks they reproduce the
   golden values exactly.
3. The Verilog testbench loads the same hex files with `$readmemh` and must
   produce the same values.
