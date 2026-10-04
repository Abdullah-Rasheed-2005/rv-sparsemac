# Results

All numbers were measured on the 10,000 MNIST test images unless noted.

## Weight statistics (baseline model)

| Layer | Weights | Range | int8 scale (max/127) |
|---|---|---|---|
| fc1 | 50,176 | -0.576 .. 0.688 | 0.00542 |
| fc2 | 640 | -0.910 .. 0.513 | 0.00716 |

## Pruning without retraining (int8 weights)

Pruning both layers by the same fraction (`sweep_prune_both_layers.py`):

| Pruned | Accuracy |
|---|---|
| 0% | 96.83% |
| 30% | 96.69% |
| 50% | 95.02% |
| 70% | 77.24% |
| 90% | 26.07% |

Pruning only fc1 and keeping fc2 dense (`sweep_prune_fc1.py`):

| fc1 pruned | Accuracy |
|---|---|
| 30% | 96.78% |
| 50% | 96.77% |
| 60% | 96.32% |
| 70% | 94.72% |
| 80% | 89.79% |
| 90% | 59.63% |

Lesson: fc2 is small and sensitive. Keep it dense and prune fc1.

## Pruning with fine-tuning (`finetune_pruned.py`)

5 epochs of training after pruning, with the mask applied after every step.

| fc1 pruned | After pruning | After fine-tune | Fine-tune + int8 |
|---|---|---|---|
| 70% | 94.81% | 97.13% | 97.11% |
| 80% | 89.78% | 96.87% | 96.88% |
| 90% | 59.73% | 95.67% | 95.73% |

Note: the 70% model scores higher than the 96.82% baseline. Part of this comes
from the extra 5 epochs of training, not from pruning. A fair comparison would
also fine-tune the baseline for 5 epochs. This is on the to-do list.

## Integer-only pipeline (80% pruned model)

| Item | Value |
|---|---|
| Float accuracy | 96.87% |
| Pure integer accuracy | 96.89% |
| Hidden shift | 13 |
| fc1 nonzero weights | 10,033 / 50,176 |
| fc2 nonzero weights | 633 / 640 |
| Zero input pixels | 80.7% |
| Zero hidden values | 41.8% |
| Golden logits exact match | 100 / 100 |

## MAC operations per image (100 test images)

| Method | MACs | Reduction |
|---|---|---|
| Dense | 50,816 | 1.0x |
| Skip zero weights | 10,666 | 4.8x |
| Skip zero inputs | 9,347 | 5.4x |
| Skip both | 2,077 | 24.5x |

These are operation counts, not cycle counts. Hardware speedup will be
measured in simulation, see the sections below.

## Sparse dot-product engine (simulation)

`hardware/rtl/sparse_dot.v` skips zero weights (zero inputs are not skipped yet).
Measured with `make sim-sparse` on the first 10 test images, one MAC per cycle:

| Layer | Cycles per image | Dense (no skipping, calculated) |
|---|---|---|
| fc1 | 10,219 | 50,368 |
| fc2 | 663 | 670 |
| Total | 10,882 | 51,038 |

About 4.7x fewer cycles from skipping zero weights. The dense column is
calculated as (inputs + 3) cycles per neuron for the same engine, not simulated.
These numbers cover only the dot-product engine. Bias, ReLU, shift, memory
loading and the control FSM are not included yet.

One fc1 neuron has no nonzero weights left after pruning (count = 0).

## Full inference (simulation)

`hardware/rtl/sparse_mlp.v` runs the whole network (784 -> 64 -> 10) on its own:
fc1 neurons, bias, ReLU, shift, saturate at 127, hidden RAM, fc2 neurons and argmax.
Measured with `make sim-mlp` on the first 100 test images:

- 1000 of 1000 logits match `golden_logits` bit for bit.
- Predictions match, accuracy 98% on these 100 images.
- 11,030 cycles per image (start to done), about 148 cycles more than the
  dot-product engine alone (10,882), so the control FSM overhead is small.

Zero pixels and zero hidden values are not skipped yet, only zero weights.

## Hardware and RISC-V results (simulation)

All numbers in this section were produced by `make report` from the simulation
logs; none is typed by hand. Clock cycles of the RTL simulation, not FPGA timing.

### Accelerator alone (start to done, 100 golden images)

| Design | Cycles per image | Notes |
|---|---|---|
| `sparse_mlp.v` fed every weight (dense reference) | 51,186 | `make sim-dense`, same engine, nothing skipped |
| `sparse_mlp.v`, skips zero weights | 11,030 | `make sim-mlp`, includes bias/ReLU/shift/argmax |
| `sparse_mlp_zs.v`, skips zero weights and zero activations | 2,163 | `make sim-zs`, min 1,034, max 3,627 |

- Useful MACs per image (nonzero pixel x nonzero weight, both layers): 2,077.
  `sparse_mlp_zs.v` needs 86 cycles more than that for pipeline start-up and the two finish sweeps.
- Speedup of `sparse_mlp_zs.v` over `sparse_mlp.v`: 5.10x. Over the dense reference: 23.7x.
- Speedup of `sparse_mlp.v` over the dense reference: 4.64x.
- Data dependent: an all-zero image takes 280 cycles, an all-255 image 10,722 cycles (nothing to skip in the pixels).
- Accuracy on these 100 images: 98%.
- Correctness: 100 golden images bit-exact, plus 24 stress images against a dense
  reference model, plus an artificial network with saturation, ties and empty
  columns (`make sim-zs-edge`, 2,233 cycles per image there).

### PicoRV32 + accelerator (`make sim-soc`, `make sim-soc-ws`)

Per image, measured by the firmware with `rdcycle`. The CPU memory has one wait
state per access. Software figures are averages over 1 (dense), 2 (skip zero pixels)
and 100 (CSC) images.

| System | Cycles per image | Relative to dense software |
|---|---|---|
| CPU only, plain dense C | 2,187,920 | 1.0x |
| CPU only, C that skips zero pixels | 462,416 | 4.7x |
| CPU only, C with the accelerator's algorithm (CSC: skips zero weights and activations) | 164,210 | 13.3x |
| CPU + `sparse_mlp.v` (zero weights), whole loop | 17,232 | 127x |
| CPU + `sparse_mlp_zs.v` (zero weights and activations), whole loop | 8,366 | 262x |

The accelerator rows include loading the image through `SMAC.LDW`, running, and
reading the ten logits. Split for `sparse_mlp_zs.v`:

| Part | Cycles per image |
|---|---|
| Load pixels (196 x `SMAC.LDW`) | 5,836 |
| `SMAC.RUN` (waiting for the accelerator) | 2,174 |
| Read 10 logits (`SMAC.LOGIT`) | 356 |
| Accelerator compute only (`SMAC.CYC`) | 2,164 |

- Fair comparison (same algorithm and weight format in C on the CPU): the whole
  accelerated loop is 19.6x faster than the CSC software, the compute alone 75.9x.
- The larger ratios against the dense and skip-zero-pixels programs mostly measure a
  better algorithm, not the hardware: the CSC software alone is already 13.3x faster than dense C.
- Loading the image costs 70% of the accelerated loop. That is now the bottleneck, not the MACs.
- Accuracy on 100 images through the CPU: 98/100. Software and accelerator logits
  were compared by the firmware: 0 mismatches. The testbench also compared all 1000 logits with `golden_logits`.
- Handshake tests of the custom instructions without a CPU: `make sim-pcpi` passes
  (including instructions that must be ignored).
