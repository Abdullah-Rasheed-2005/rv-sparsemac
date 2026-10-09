# Synthesis

Written by `scripts/synth_report.py` from the reports of `make synth` (Yosys 0.33, `synth_ecp5`).
Target: Lattice ECP5. These are cell counts after synthesis, before place and route.
The weight memories hold the normal model (`hardware/mem`); their size does not depend on the weights.

| Cell | accelerator, budget switched off | accelerator with budget + constant time | whole co-processor: accelerator + custom instructions + image loader |
|---|---|---|---|
| LUT4 | 1,172 | 1,236 | 2,287 |
| flip-flops | 380 | 398 | 709 |
| carry cells | 137 | 175 | 236 |
| block RAM (18 kbit) | 19 | 19 | 21 |
| distributed RAM cells (16 x 4 bit) | 32 | 32 | 40 |
| 18 x 18 multipliers | 1 | 1 | 1 |
| PFUMX (mux inside a slice) | 362 | 382 | 551 |
| L6MUX21 | 137 | 147 | 192 |

- The run-time budget and the constant-time mode cost +64 LUT4 and +18 flip-flops (+5.5% LUT4) and no memory.
- The custom-instruction wrapper with the image loader adds +1,051 LUT4, +311 flip-flops and +2 block RAM (image buffer and order table).
- The whole co-processor uses 2,287 LUT4, 709 flip-flops, 21 block RAMs and 1 multiplier. For comparison, an LFE5U-25F has 24,288 LUT4 and 56 block RAMs.
- Most block RAMs hold the fc1 weights: the list is sized for 16,384 nonzero weights; the 90 % pruned
  models use about 5,000, so half of that memory would be enough for them.
- The 64 accumulators are built from distributed RAM, not block RAM, because they are read and
  written in the same clock cycle.

## Clock frequency

Place and route with nextpnr-ecp5 (`make synth-timing`, LFE5U-85F, speed grade 6, pins placed
automatically): maximum clock frequency 38.07 MHz for the whole co-processor.
This is one run of a heuristic tool without timing constraints on the pins; treat it as an estimate.
