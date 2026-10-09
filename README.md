# rv-sparsemac

A sparse multiply-accumulate (MAC) accelerator for a small neural network,
attached to a RISC-V core as a custom instruction.

The idea: after pruning, most weights of a neural network are zero, and most
MNIST pixels are zero too. A MAC unit that **skips the zeros** does much less
work. This project builds the full chain: train a model, prune and quantize it,
export it to hardware, build the Verilog accelerator and run it on a RISC-V
core (in simulation).

## Status

| Part | State |
|---|---|
| Train, prune, fine-tune, quantize to int8 | Done |
| Integer-only inference + hex export + golden vectors | Done |
| MAC unit (`hardware/rtl/mac.v`) + testbench | Done |
| Sparse dot-product engine (skips zero weights) + testbench | Done |
| Control FSM + full inference in simulation | Done |
| Zero-skipping accelerator (skips zero pixels and zero hidden values too) + stress tests | Done |
| RISC-V integration (PicoRV32 + PCPI custom instructions) + C firmware | Done |
| Cycle benchmark: CPU software vs accelerator (simulation) | Done |
| Dense reference in simulation (same engine, nothing skipped) | Done |
| Bounded run time: hard cycle budget + constant-time mode (`SMAC.CFG`), budget-aware model | Done |
| Accelerator loads the image itself (second bus master, `SMAC.RUNM`) | Done |
| Synthesis for a Lattice ECP5 FPGA (cell counts, cost of the budget and the loader) | Done |
| Running on an FPGA board | Planned |

See [docs/roadmap.md](docs/roadmap.md) for the detailed plan.

## Software results (MNIST, 10,000 test images)

Network: 784 -> 64 -> 10, weights quantized to int8.

| Model | Accuracy |
|---|---|
| Float32 baseline | 96.82% |
| fc1 pruned 80% + fine-tuned, float | 96.87% |
| fc1 pruned 80% + fine-tuned, **pure integer pipeline** | **96.89%** |
| fc1 pruned 90% + fine-tuned, int8 | 95.73% |

Average multiply-accumulates per image (measured on 100 test images):

| Method | MACs per image | Reduction |
|---|---|---|
| Dense (no skipping) | 50,816 | 1.0x |
| Skip zero weights | 10,666 | 4.8x |
| Skip zero inputs | 9,347 | 5.4x |
| Skip both | 2,077 | 24.5x |

**Important:** the table above counts MAC operations. The measured hardware
result (RTL simulation, 100 test images) is in
[docs/results.md](docs/results.md): the zero-skipping accelerator needs about
2,163 cycles per image, against 11,030 for the weight-skipping-only design
and 51,186 for the same engine with nothing skipped (`make sim-dense`). The remaining cycles over the
2,077 useful MACs are pipeline start-up and the finish sweeps.

Details and notes: [docs/results.md](docs/results.md).

## Repository layout

```
rv-sparsemac/
├── README.md
├── LICENSE
├── Makefile                  shortcuts for every step (run `make help`)
├── requirements.txt          Python packages
├── docs/
│   ├── architecture.md       network, integer pipeline, hardware overview
│   ├── hardware.md           the accelerator and the RISC-V system in detail
│   ├── results.md            all measured numbers
│   └── roadmap.md            plan and progress
├── software/                 Python: train, prune, quantize, export
│   ├── paths.py              all file paths in one place
│   ├── model.py              the 784-64-10 network
│   ├── train_baseline.py
│   ├── evaluate_baseline.py
│   ├── inspect_weights.py
│   ├── sweep_prune_both_layers.py
│   ├── sweep_prune_fc1.py
│   ├── finetune_pruned.py
│   ├── export_int8.py
│   ├── verify_export.py
│   ├── export_sparse.py      sparse (CSR and CSC) weight lists + golden vectors
│   ├── export_dense_lists.py dense lists in the sparse format (for sim-dense)
│   ├── make_edge_model.py    artificial network for corner cases (sim-zs-edge)
│   ├── budget_lib.py         the run-time budget rule (exact integer model)
│   ├── budget_experiment.py  accuracy with a budget + budget-aware fine-tuning
│   ├── budget_study.py       seeds, alternatives with the same worst case, timing leak
│   ├── budget_pareto.py      pruning and the budget together (picks the hardware model)
│   ├── fashion_pareto.py     the same study on Fashion-MNIST
│   ├── export_budget.py      export a budget-aware model (build/budget_mem or build/fashion_mem)
│   └── learning/             small exercises used while learning
├── models/                   trained PyTorch weights (.pth)
├── firmware/                 C program for the RISC-V core + custom-instruction header
├── scripts/                  write_results.py, synth_report.py: write the measured numbers into docs/
└── hardware/
    ├── rtl/                  Verilog design files
    ├── tb/                   Verilog testbenches
    ├── mem/                  hex weights + golden vectors for $readmemh
    └── third_party/          external cores (PicoRV32, as a git submodule)
```

## Quick start

All commands run from the repository root.

```bash
# 1. Python environment
make venv
source venv/bin/activate

# 2. Software pipeline
make train        # train the float32 baseline
make evaluate     # baseline accuracy
make finetune     # prune fc1 (70/80/90%) and fine-tune
make export       # integer inference + hex files in hardware/mem/
make verify       # check hex files against golden vectors

# 3. Hardware (needs Icarus Verilog: sudo apt install iverilog)
make sim-mac      # simulate the MAC unit
make export-sparse  # sparse (index, value) lists
make sim-sparse   # simulate the sparse dot-product engine
make sim-mlp      # whole network, skips zero weights
make sim-zs       # whole network, skips zero weights AND zero activations
make sim-dense    # same engine with nothing skipped (dense baseline)
make sim-zs-budget  # bounded run time: hard cycle budget and constant time

# 4. RISC-V system. One-time setup: the PicoRV32 submodule and the RISC-V compiler
#    git submodule update --init      (or clone with --recurse-submodules)
#    sudo apt install gcc-riscv64-unknown-elf
make sim-pcpi     # custom-instruction wrapper alone
make sim-soc      # PicoRV32 + accelerator run the firmware on 100 images
make sim-soc-budget  # same with the run-time budget set by the CPU (SMAC.CFG)
make sim-zs-fashion  # the accelerator with the Fashion-MNIST model (same hardware)
make sim-soc-fashion # PicoRV32 + accelerator with the Fashion-MNIST model
make report       # run everything and write the numbers into docs/results.md

# 5. Synthesis (needs: sudo apt install yosys)
make synth        # cell counts for a Lattice ECP5, written to docs/synthesis.md
```

The trained models are already in `models/` and the hex files are already in
`hardware/mem/`, so you can run `make verify` and `make sim-mac` without
training anything.

Note: `train_baseline.py` has no fixed random seed, so retraining gives a slightly different baseline than the 96.82% reported here. All reported numbers come from the committed models in `models/`.

## Bounded run time

A zero-skipping accelerator is fast on average, but its run time depends on the
image: an image with every pixel set is about five times slower than an average
digit. With `SMAC.CFG` the CPU gives layer 1 a hard cycle budget. The inputs are
visited most-useful-first and the model is fine-tuned to cope with a cut image.

Measured on the 10,000 test images (layer-1 cycles, exact hardware arithmetic):

| Model | Accuracy | Average cycles | Guaranteed worst case |
|---|---|---|---|
| 80 % pruned, no budget | 97.07% | 1,857 | 10,036 |
| 90 % pruned, no budget | 96.85% | 923 | 5,483 |
| **90 % pruned, budget 1,000** | **96.81%** | **857** | **1,000** |
| pruning alone until the worst case is 1,000 (605 weights left) | 87.70% | - | 1,000 |

The same hardware also runs a Fashion-MNIST model (only half of the pixels are
zero there, and the task is harder): 86.27 % without a budget, 86.02 % with a
budget of 1,850 cycles instead of a worst case of 5,755
([docs/fashion_pareto.md](docs/fashion_pareto.md)).

In constant-time mode every image takes exactly the same number of cycles. With
`SMAC.RUNM` the accelerator also fetches the image from memory itself, in a fixed
number of cycles, so the bound and the constant time hold for the whole inference
as the CPU sees it.
Why this works and what it does not cover: [docs/hardware.md](docs/hardware.md)
section 6. All numbers: [docs/results.md](docs/results.md),
[docs/budget_study.md](docs/budget_study.md), [docs/budget_pareto.md](docs/budget_pareto.md).

## How it works (short version)

1. **Train** a small fully connected network on MNIST.
2. **Prune** the smallest 80% of the fc1 weights, then **fine-tune** with a
   mask so the removed weights stay zero. Accuracy comes back to baseline.
3. **Quantize** weights to int8 with a symmetric scale (`scale = max|w| / 127`).
4. Run inference with **integers only** (pixels 0..255, int8 weights, 32-bit
   accumulators, a right shift instead of a divide). This is what the hardware
   will do.
5. **Export** the weights as hex files together with golden outputs. The
   Verilog testbench must reproduce the golden outputs bit for bit.

More in [docs/architecture.md](docs/architecture.md) (the network) and
[docs/hardware.md](docs/hardware.md) (the accelerator and the RISC-V system).

## License

MIT. See [LICENSE](LICENSE).
