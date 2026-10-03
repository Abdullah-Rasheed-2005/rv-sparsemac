# rv-sparsemac

A sparse multiply-accumulate (MAC) accelerator for a small neural network,
attached to a RISC-V core as a custom instruction.

The idea: after pruning, most weights of a neural network are zero, and most
MNIST pixels are zero too. A MAC unit that **skips the zeros** does much less
work. This project builds the full chain: train a model, prune and quantize it,
export it to hardware, and (in progress) build the Verilog accelerator.

## Status

| Part | State |
|---|---|
| Train, prune, fine-tune, quantize to int8 | Done |
| Integer-only inference + hex export + golden vectors | Done |
| MAC unit (`hardware/rtl/mac.v`) + testbench | Done |
| Sparse dot-product engine | Next |
| RISC-V integration (PicoRV32 + PCPI custom instruction) | Planned |
| Cycle benchmark: dense vs sparse | Planned |

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

**Important:** these are counts of MAC operations, not hardware speedup.
Finding the nonzero values and reading their indices also costs cycles. The
real speedup will be measured in simulation and reported here.

Details and notes: [docs/results.md](docs/results.md).

## Repository layout

```
rv-sparsemac/
├── README.md
├── LICENSE
├── Makefile                  shortcuts for every step (run `make help`)
├── requirements.txt          Python packages
├── docs/
│   ├── architecture.md       network, integer pipeline, planned hardware
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
│   └── learning/             small exercises used while learning
├── models/                   trained PyTorch weights (.pth)
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
```

The trained models are already in `models/` and the hex files are already in
`hardware/mem/`, so you can run `make verify` and `make sim-mac` without
training anything.

Note: `train_baseline.py` has no fixed random seed, so retraining gives a slightly different baseline than the 96.82% reported here. All reported numbers come from the committed models in `models/`.

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

More in [docs/architecture.md](docs/architecture.md).

## License

MIT. See [LICENSE](LICENSE).
