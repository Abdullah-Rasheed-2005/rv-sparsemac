# Roadmap

## Phase 1 - Software pipeline (done)
- [x] Train the float32 baseline (96.82%)
- [x] Measure weight ranges and near-zero statistics
- [x] Prune and quantize without retraining (find the limits)
- [x] Prune fc1 and fine-tune with a mask (80%: 96.87%)
- [x] Integer-only inference (96.89%)
- [x] Export hex files and golden vectors
- [x] Verify the export (100/100 exact match)

## Phase 2 - Hardware building blocks
- [x] MAC unit: unsigned x signed, 32-bit accumulator (`mac.v`)
- [x] Sparse dot-product engine: process only nonzero weights for one neuron
- [x] Control FSM: load, compute, ReLU + shift, store
- [x] Full fc1 + fc2 inference in simulation, matching the golden logits
- [x] Zero skipping on the activation side too (`sparse_mlp_zs.v`, column-wise weights)

## Phase 3 - RISC-V integration
- [x] Clone PicoRV32 as a git submodule in `hardware/third_party/`
- [x] Study `picorv32_pcpi_mul` as a reference PCPI co-processor
- [x] Wrap the accelerator as a PCPI module (`SMAC.LDW`, `SMAC.RUN`, `SMAC.LOGIT`, `SMAC.CYC`)
- [x] Test program (C) that runs MNIST inference on the core (`firmware/main.c`)
- [ ] Run the system on an FPGA (block RAM instead of `$readmemh`, UART output)
- [ ] Let the accelerator read the image from RAM itself (bus master) instead of `SMAC.LDW`

## Phase 3b - Bounded run time
- [x] Software study: accuracy with a hard layer-1 budget, fixed input order, budget-aware fine-tuning
- [x] Budget and constant-time mode in `sparse_mlp_zs.v`, bit-exact against the software rule
- [x] `SMAC.CFG` instruction, SoC simulation with the budget set by the CPU
- [ ] Bus-master loader that applies the input order (bounds the whole loop, not only the accelerator)
- [ ] Repeat with 3 random seeds; compare with simply pruning more at the same average cost
- [ ] Second dataset (Fashion-MNIST)
- [ ] Timing-leak and slow-input measurements with and without the budget

## Phase 4 - Benchmark and report
- [x] Cycle count: software loop vs weight-skipping vs zero-skipping accelerator (`make report`)
- [x] Dense reference in simulation (`make sim-dense`: the same engine fed every weight)
- [ ] Plots of cycles and accuracy vs sparsity
- [ ] Fair baseline: fine-tune the dense model for the same 5 epochs
- [ ] Final write-up in `docs/`
