# Hardware

| Folder | Content |
|---|---|
| `rtl/` | Verilog design files |
| `tb/` | Verilog testbenches (one per module, named `tb_<module>.v`) |
| `mem/` | Hex weights and golden vectors, written by `software/export_int8.py` |
| `third_party/` | External cores, e.g. PicoRV32 as a git submodule |

## Simulating

From the repository root:

```bash
make sim-mac
```

Needs Icarus Verilog (`sudo apt install iverilog`). Expected last line:

```
PASS: all MAC tests passed (acc=...)
```

The sparse dot-product engine needs its sparse files first (once):

```bash
make export-sparse
make sim-sparse
```

Expected last line:

```
PASS: sparse_dot matched the golden values (640 fc1 + 100 fc2 neurons)
```

The whole network (control FSM + sparse engine, all 100 golden images):

```bash
make sim-mlp
```

Expected last line:

```
PASS: sparse_mlp matched golden logits and predictions (100 images, 1000 logits)
```

The zero-skipping version (skips zero pixels and zero hidden values too, plus
stress tests against a dense reference model):

```bash
make sim-zs
```

Expected last line:

```
PASS: sparse_mlp_zs matched golden logits and the reference model (100 + 24 images)
```

An artificial network with saturating hidden values, tied logits, a fully pruned
neuron and empty columns (the real model never produces these):

```bash
make sim-zs-edge
```

## RISC-V system (PicoRV32 + accelerator)

One-time setup (PicoRV32 is a git submodule, the compiler turns C into RISC-V code):

```bash
git submodule update --init
sudo apt install gcc-riscv64-unknown-elf
```

Then, after `make export-sparse`:

```bash
make sim-pcpi    # the PCPI wrapper alone: handshake, foreign instructions, results
make sim-soc     # CPU + accelerator run firmware/main.c on 100 images
make sim-soc-ws  # same with the weight-skipping-only accelerator (comparison)
```

Expected last line of `make sim-soc`:

```
PASS: CPU + accelerator matched golden logits and predictions (100 images, 1000 logits)
```

| File | Content |
|---|---|
| `rtl/sparse_mlp_zs.v` | accelerator, skips zero weights and zero activations |
| `rtl/sparsemac_pcpi.v` | PCPI wrapper: the custom instructions `SMAC.LDW/RUN/LOGIT/CYC/CFG` |
| `tb/tb_zs_budget.v` | run-time budget and constant time of the accelerator (`make sim-zs-budget`) |
| `tb/tb_sparse_mlp_zs.v` | golden images + stress tests of the accelerator |
| `tb/tb_sparsemac_pcpi.v` | PCPI wrapper tested without a CPU |
| `tb/tb_soc.v` | PicoRV32 + RAM + wrapper running the firmware |
| `../firmware/` | C firmware, linker script, custom-instruction header |

## Naming rules

- Design file `foo.v` contains module `foo`.
- Its testbench is `tb/tb_foo.v` with module `tb_foo`.
