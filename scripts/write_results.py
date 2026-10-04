#!/usr/bin/env python3
"""
write_results.py - Write the measured simulation numbers into docs/results.md
=============================================================================
Reads the simulation logs in build/logs/ (written by `make report`) and writes
the section "## Hardware and RISC-V results (simulation)" at the end of
docs/results.md. Every number in that section comes from the logs, nothing is
typed by hand. If the section already exists it is replaced.

Needs only the Python standard library. `make report` runs it.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'build' / 'logs'
MARK = '## Hardware and RISC-V results (simulation)'


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
    dense = read_log('sim-dense.log')
    pcpi = read_log('sim-pcpi.log')
    soc = read_log('sim-soc.log')
    socws = read_log('sim-soc-ws.log')
    for name, t, tag in (('sim-mlp', mlp, 'PASS: sparse_mlp matched'),
                         ('sim-zs', zs, 'PASS: sparse_mlp_zs matched'),
                         ('sim-zs-edge', edge, 'PASS: sparse_mlp_zs matched'),
                         ('sim-dense', dense, 'PASS: dense reference matched'),
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
    dense_cyc = find(r'cycles per image \(start to done\):\s+(\d+)', dense, 'sim-dense cycles')

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
| `sparse_mlp.v` fed every weight (dense reference) | {f(dense_cyc)} | `make sim-dense`, same engine, nothing skipped |
| `sparse_mlp.v`, skips zero weights | {f(ws_cyc)} | `make sim-mlp`, includes bias/ReLU/shift/argmax |
| `sparse_mlp_zs.v`, skips zero weights and zero activations | {f(zs_avg)} | `make sim-zs`, min {f(zs_min)}, max {f(zs_max)} |

- Useful MACs per image (nonzero pixel x nonzero weight, both layers): {f(useful)}.
  `sparse_mlp_zs.v` needs {over} cycles more than that for pipeline start-up and the two finish sweeps.
- Speedup of `sparse_mlp_zs.v` over `sparse_mlp.v`: {ws_cyc / zs_avg:.2f}x. Over the dense reference: {dense_cyc / zs_avg:.1f}x.
- Speedup of `sparse_mlp.v` over the dense reference: {dense_cyc / ws_cyc:.2f}x.
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
    results()
