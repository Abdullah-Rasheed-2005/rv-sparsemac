#!/usr/bin/env python3
"""
phase3_docs.py - Keep the documentation in step with the simulation results
===========================================================================
Two jobs, both safe to run more than once:

  python3 scripts/phase3_docs.py edit
      Updates README.md and docs/roadmap.md: status table, quick start,
      repository layout, roadmap check boxes. Changes that are already there
      are skipped. Prints what it did.

  python3 scripts/phase3_docs.py results
      Reads the simulation logs in build/logs/ (written by `make report`) and
      writes the section "## Hardware and RISC-V results (simulation)" at the
      end of docs/results.md. Every number in that section comes from YOUR logs,
      nothing is typed by hand. If the section exists it is replaced.

Needs only the Python standard library. Run from the repository root.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'build' / 'logs'
MARK = '## Hardware and RISC-V results (simulation)'


# ----------------------------------------------------------------- edit mode
def replace_once(text, old, new, label, notes):
    if new in text:
        notes.append(f'  already done : {label}')
        return text
    if old not in text:
        notes.append(f'  NOT FOUND    : {label}  (edit this one by hand)')
        return text
    notes.append(f'  changed      : {label}')
    return text.replace(old, new, 1)


def edit_docs():
    notes = []

    # ---------------- README.md ----------------
    p = ROOT / 'README.md'
    s = p.read_text()

    s = replace_once(
        s,
        '| Control FSM + full inference in simulation | Done |\n'
        '| RISC-V integration (PicoRV32 + PCPI custom instruction) | Planned |\n'
        '| Cycle benchmark: dense vs sparse | Planned |\n',
        '| Control FSM + full inference in simulation | Done |\n'
        '| Zero-skipping accelerator (skips zero pixels and zero hidden values too) + stress tests | Done |\n'
        '| RISC-V integration (PicoRV32 + PCPI custom instructions) + C firmware | Done |\n'
        '| Cycle benchmark: CPU software vs accelerator (simulation) | Done |\n'
        '| Dense accelerator in simulation, FPGA build | Planned |\n',
        'README status table', notes)

    s = replace_once(
        s,
        'make sim-sparse   # simulate the sparse dot-product engine\n',
        'make sim-sparse   # simulate the sparse dot-product engine\n'
        'make sim-mlp      # whole network, skips zero weights\n'
        'make sim-zs       # whole network, skips zero weights AND zero activations\n'
        '\n'
        '# 4. RISC-V system (one-time: git submodule add ... and sudo apt install gcc-riscv64-unknown-elf,\n'
        '#    see hardware/README.md)\n'
        'make sim-pcpi     # custom-instruction wrapper alone\n'
        'make sim-soc      # PicoRV32 + accelerator run the firmware on 100 images\n'
        'make report       # run everything and write the numbers into docs/results.md\n',
        'README quick start', notes)

    s = replace_once(
        s,
        '│   └── third_party/          external cores (PicoRV32, as a git submodule)\n',
        '│   └── third_party/          external cores (PicoRV32, as a git submodule)\n'
        '├── firmware/                 C program for the RISC-V core + custom-instruction header\n',
        'README repository layout', notes)

    s = replace_once(
        s,
        'More in [docs/architecture.md](docs/architecture.md).',
        'More in [docs/architecture.md](docs/architecture.md) (the network) and\n'
        '[docs/hardware.md](docs/hardware.md) (the accelerator and the RISC-V system).',
        'README links', notes)
    p.write_text(s)

    # ---------------- docs/roadmap.md ----------------
    p = ROOT / 'docs' / 'roadmap.md'
    s = p.read_text()
    pairs = [
        ('- [ ] Sparse dot-product engine:', '- [x] Sparse dot-product engine:'),
        ('- [ ] Control FSM:', '- [x] Control FSM:'),
        ('- [ ] Full fc1 + fc2 inference in simulation, matching the golden logits',
         '- [x] Full fc1 + fc2 inference in simulation, matching the golden logits'),
        ('- [x] Full fc1 + fc2 inference in simulation, matching the golden logits\n',
         '- [x] Full fc1 + fc2 inference in simulation, matching the golden logits\n'
         '- [x] Zero skipping on the activation side too (`sparse_mlp_zs.v`, column-wise weights)\n'),
        ('- [ ] Clone PicoRV32 as a git submodule', '- [x] Clone PicoRV32 as a git submodule'),
        ('- [ ] Study `picorv32_pcpi_mul`', '- [x] Study `picorv32_pcpi_mul`'),
        ('- [ ] Wrap the accelerator as a PCPI module (`SPARSEMAC rd, rs1, rs2`)',
         '- [x] Wrap the accelerator as a PCPI module (`SMAC.LDW`, `SMAC.RUN`, `SMAC.LOGIT`, `SMAC.CYC`)'),
        ('- [ ] Test program (assembly or C) that runs MNIST inference on the core',
         '- [x] Test program (C) that runs MNIST inference on the core (`firmware/main.c`)\n'
         '- [ ] Run the system on an FPGA (block RAM instead of `$readmemh`, UART output)\n'
         '- [ ] Let the accelerator read the image from RAM itself (bus master) instead of `SMAC.LDW`'),
        ('- [ ] Cycle count: software loop vs dense accelerator vs sparse accelerator',
         '- [x] Cycle count: software loop vs weight-skipping vs zero-skipping accelerator (`make report`)\n'
         '- [ ] Dense accelerator in simulation (the dense number is calculated so far)'),
    ]
    for old, new in pairs:
        s = replace_once(s, old, new, 'roadmap: ' + old[6:50], notes)
    p.write_text(s)

    print('\n'.join(notes))


# --------------------------------------------------------------- results mode
def read_log(name):
    f = LOGS / name
    if not f.exists():
        sys.exit(f'missing {f} - run `make report` first')
    return f.read_text()


def find(pattern, text, label, cast=int):
    m = re.search(pattern, text)
    if not m:
        sys.exit(f'could not find "{label}" in a log (pattern {pattern!r})')
    return cast(m.group(1).replace(',', ''))


def soc_numbers(text):
    d = {}
    d['n'] = find(r'images:\s+(\d+)\n', text, 'images')
    d['correct'] = find(r'correct:\s+(\d+)', text, 'correct')
    d['load'] = find(r'load pixels \(total\):\s+(\d+)', text, 'load')
    d['run'] = find(r'SMAC\.RUN\s+\(total\):\s+(\d+)', text, 'run')
    d['read'] = find(r'read logits \(total\):\s+(\d+)', text, 'read')
    d['acc'] = find(r'accelerator \(total\):\s+(\d+)', text, 'accelerator')
    d['all'] = find(r'all\s+\(total\):\s+(\d+)', text, 'all')
    d['s_n'] = find(r'skip-zeros images:\s+(\d+)', text, 'skip images')
    d['s_cyc'] = find(r'skip-zeros cycles \(total\):\s+(\d+)', text, 'skip cycles')
    d['d_n'] = find(r'dense images:\s+(\d+)', text, 'dense images')
    d['d_cyc'] = find(r'dense cycles \(total\):\s+(\d+)', text, 'dense cycles')
    d['mism'] = find(r'mismatches:\s+(\d+)', text, 'mismatches')
    if 'PASS: CPU + accelerator' not in text:
        sys.exit('a SoC log does not contain the PASS line - fix the failure first')
    return d


def results():
    mlp = read_log('sim-mlp.log')
    zs = read_log('sim-zs.log')
    edge = read_log('sim-zs-edge.log')
    pcpi = read_log('sim-pcpi.log')
    soc = read_log('sim-soc.log')
    socws = read_log('sim-soc-ws.log')
    for name, t, tag in (('sim-mlp', mlp, 'PASS: sparse_mlp matched'),
                         ('sim-zs', zs, 'PASS: sparse_mlp_zs matched'),
                         ('sim-zs-edge', edge, 'PASS: sparse_mlp_zs matched'),
                         ('sim-pcpi', pcpi, 'PASS: sparsemac_pcpi')):
        if tag not in t:
            sys.exit(f'{name} did not pass - fix the failure first')

    ws_cyc = find(r'cycles per image \(start to done\):\s+(\d+)', mlp, 'sim-mlp cycles')
    zs_avg = find(r'average (\d+), min', zs, 'zs average')
    zs_min = find(r'min (\d+), max', zs, 'zs min')
    zs_max = find(r'max (\d+)\n', zs, 'zs max')
    useful = find(r'useful MACs per image.*?average (\d+)', zs, 'useful MACs')
    over = find(r'overhead on top of the useful MACs: (\d+)', zs, 'overhead')
    acc100 = find(r'accuracy on these \d+ images: (\d+)%', zs, 'accuracy')
    all0 = find(r'all-zero image: (\d+) cycles', zs, 'all-zero')
    all255 = find(r'all-255 image:\s+(\d+) cycles', zs, 'all-255')
    edge_avg = find(r'average (\d+), min', edge, 'edge average')
    stress_n = find(r'(\d+) images checked against the dense reference', zs, 'stress images')
    dense_calc = 64 * (784 + 3) + 10 * (64 + 3)

    Z, W = soc_numbers(soc), soc_numbers(socws)
    n = Z['n']
    z_all, z_cmp = Z['all'] / n, Z['acc'] / n
    w_all, w_cmp = W['all'] / n, W['acc'] / n
    sw_dense = Z['d_cyc'] / Z['d_n']
    sw_skip = Z['s_cyc'] / Z['s_n']

    def f(x):
        return f'{x:,.0f}'

    out = f'''{MARK}

All numbers in this section were produced by `make report` from the simulation
logs; none is typed by hand. Clock cycles of the RTL simulation, not FPGA timing.

### Accelerator alone (start to done, 100 golden images)

| Design | Cycles per image | Notes |
|---|---|---|
| Dense dot-product engine | {f(dense_calc)} | calculated as (inputs + 3) cycles per neuron, not simulated |
| `sparse_mlp.v`, skips zero weights | {f(ws_cyc)} | `make sim-mlp`, includes bias/ReLU/shift/argmax |
| `sparse_mlp_zs.v`, skips zero weights and zero activations | {f(zs_avg)} | `make sim-zs`, min {f(zs_min)}, max {f(zs_max)} |

- Useful MACs per image (nonzero pixel x nonzero weight, both layers): {f(useful)}.
  `sparse_mlp_zs.v` needs {over} cycles more than that for pipeline start-up and the two finish sweeps.
- Speedup of `sparse_mlp_zs.v` over `sparse_mlp.v`: {ws_cyc / zs_avg:.2f}x. Over the calculated dense engine: {dense_calc / zs_avg:.1f}x.
- Data dependent: an all-zero image takes {f(all0)} cycles, an all-255 image {f(all255)} cycles (nothing to skip in the pixels).
- Accuracy on these 100 images: {acc100}%.
- Correctness: 100 golden images bit-exact, plus {stress_n} stress images against a dense
  reference model, plus an artificial network with saturation, ties and empty
  columns (`make sim-zs-edge`, {f(edge_avg)} cycles per image there).

### PicoRV32 + accelerator (`make sim-soc`, `make sim-soc-ws`)

Per image, measured by the firmware with `rdcycle`. The CPU memory has one wait
state per access. Software figures are averages over only {Z['d_n']} (dense) and {Z['s_n']} (skip zeros) images.

| System | Cycles per image | Relative to dense software |
|---|---|---|
| CPU only, plain dense C | {f(sw_dense)} | 1.0x |
| CPU only, C that skips zero pixels | {f(sw_skip)} | {sw_dense / sw_skip:.1f}x |
| CPU + `sparse_mlp.v` (zero weights), whole loop | {f(w_all)} | {sw_dense / w_all:.0f}x |
| CPU + `sparse_mlp_zs.v` (zero weights and activations), whole loop | {f(z_all)} | {sw_dense / z_all:.0f}x |

The accelerator rows include loading the image through `SMAC.LDW`, running, and
reading the ten logits. Split for `sparse_mlp_zs.v`:

| Part | Cycles per image |
|---|---|
| Load pixels (196 x `SMAC.LDW`) | {f(Z['load'] / n)} |
| `SMAC.RUN` (waiting for the accelerator) | {f(Z['run'] / n)} |
| Read 10 logits (`SMAC.LOGIT`) | {f(Z['read'] / n)} |
| Accelerator compute only (`SMAC.CYC`) | {f(z_cmp)} |

- Speedup of the whole accelerated loop over the best software tried (skip zeros): {sw_skip / z_all:.1f}x.
- Speedup of the compute alone over that software: {sw_skip / z_cmp:.0f}x.
- Loading the image costs {Z['load'] / Z['all'] * 100:.0f}% of the accelerated loop. That is now the bottleneck, not the MACs.
- Accuracy on 100 images through the CPU: {Z['correct']}/{n}. Software and accelerator logits
  were compared by the firmware: {Z['mism']} mismatches. The testbench also compared all {n * 10} logits with `golden_logits`.
- Handshake tests of the custom instructions without a CPU: `make sim-pcpi` passes
  (including instructions that must be ignored).
'''
    p = ROOT / 'docs' / 'results.md'
    s = p.read_text()
    if MARK in s:
        s = s[:s.index(MARK)].rstrip('\n') + '\n\n'
    else:
        s = s.rstrip('\n') + '\n\n'
    p.write_text(s + out)
    print(f'wrote the "{MARK[3:]}" section to {p}')
    print(out)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('edit', 'results'):
        sys.exit(__doc__)
    edit_docs() if sys.argv[1] == 'edit' else results()
