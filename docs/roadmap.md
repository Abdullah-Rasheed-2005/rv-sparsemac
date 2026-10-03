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

## Phase 3 - RISC-V integration
- [ ] Clone PicoRV32 as a git submodule in `hardware/third_party/`
- [ ] Study `picorv32_pcpi_mul` as a reference PCPI co-processor
- [ ] Wrap the accelerator as a PCPI module (`SPARSEMAC rd, rs1, rs2`)
- [ ] Test program (assembly or C) that runs MNIST inference on the core

## Phase 4 - Benchmark and report
- [ ] Cycle count: software loop vs dense accelerator vs sparse accelerator
- [ ] Plots of cycles and accuracy vs sparsity
- [ ] Fair baseline: fine-tune the dense model for the same 5 epochs
- [ ] Final write-up in `docs/`
