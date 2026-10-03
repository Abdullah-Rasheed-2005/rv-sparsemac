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

## Adding PicoRV32 (later)

```bash
git submodule add https://github.com/YosysHQ/picorv32.git hardware/third_party/picorv32
```

## Naming rules

- Design file `foo.v` contains module `foo`.
- Its testbench is `tb/tb_foo.v` with module `tb_foo`.
