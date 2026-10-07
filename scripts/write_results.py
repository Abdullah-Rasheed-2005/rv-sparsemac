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
    d['c_n'] = find(r'csc-sparse images:\s+(\d+)', text, 'csc images')
    d['c_cyc'] = find(r'csc-sparse cycles \(total\):\s+(\d+)', text, 'csc cycles')
    d['d_n'] = find(r'dense images:\s+(\d+)', text, 'dense images')
    d['d_cyc'] = find(r'dense cycles \(total\):\s+(\d+)', text, 'dense cycles')
    d['mism'] = find(r'mismatches:\s+(\d+)', text, 'mismatches')
    if 'PASS: CPU + accelerator' not in text:
        sys.exit('a SoC log does not contain the PASS line - fix the failure first')
    return d


def budget_pass(text, title):
    """The seven numbers the firmware prints for one extra pass (see put_pass in firmware/main.c)."""
    m = re.search(r'\n' + title + r':\n\s+correct:\s+(\d+)\n\s+accelerator cycles, total:\s+(\d+)\n'
                  r'\s+accelerator cycles, min:\s+(\d+)\n\s+accelerator cycles, max:\s+(\d+)\n'
                  r'\s+whole loop cycles, total:\s+(\d+)\n\s+whole loop cycles, min:\s+(\d+)\n'
                  r'\s+whole loop cycles, max:\s+(\d+)', text)
    if not m:
        sys.exit(f'could not find the block "{title}" in a SoC log')
    return dict(zip(('correct', 'acc', 'min', 'max', 'all', 'all_min', 'all_max'), map(int, m.groups())))


def budget_section(tag='budget'):
    """The section about the run-time budget, from sim-zs-<tag>.log and sim-soc-<tag>.log.
    tag = 'budget' (MNIST) or 'fashion' (Fashion-MNIST, same hardware)."""
    zb = read_log(f'sim-zs-{tag}.log')
    sb = read_log(f'sim-soc-{tag}.log')
    if 'PASS: sparse_mlp_zs budget matched' not in zb:
        sys.exit(f'sim-zs-{tag} did not pass - fix the failure first')
    if tag == 'fashion':
        title = 'Bounded run time on Fashion-MNIST (`make sim-zs-fashion`, `make sim-soc-fashion`)'
        intro = ('The same hardware, only the weights, the order table and the images are different. About half of the\n'
                 'pixels are zero here (MNIST: about 81 %), so every image costs more cycles.\n')
        why = '[fashion_pareto.md](fashion_pareto.md)'
    else:
        title = 'Bounded run time (`make sim-zs-budget`, `make sim-soc-budget`)'
        intro = ''
        why = '[budget_pareto.md](budget_pareto.md) and [budget_study.md](budget_study.md)'
    S = soc_numbers(sb)                      # pass without a budget (also checks the PASS line)
    npass = find(r'passes checked: (\d+)', sb, 'passes checked')

    B = find(r'layer-1 budget = (\d+) cycles', zb, 'budget')
    model = find(r'model: (\S+) ', zb, 'model file', str)
    sp = find(r'\((\d+) % of fc1 pruned\)', zb, 'sparsity')
    acc_free = find(r'accuracy, no budget\s+: ([\d.]+)%', zb, 'accuracy no budget', float)
    acc_bud = find(r'accuracy, with budget : ([\d.]+)%', zb, 'accuracy with budget', float)
    l1_avg = find(r'accuracy, no budget.*?average (\d+)', zb, 'layer-1 average')
    l1_max = find(r'accuracy, no budget.*?maximum (\d+)', zb, 'layer-1 maximum')
    l1_worst = find(r'worst possible image (\d+)', zb, 'layer-1 worst')
    l1_bavg = find(r'accuracy, with budget.*?average (\d+)', zb, 'layer-1 average with budget')
    f_avg = find(r'--- no budget ---\ncycles per image \(start to done\): average (\d+)', zb, 'free average')
    f_max = find(r'--- no budget ---\ncycles per image \(start to done\): average \d+, max (\d+)', zb, 'free max')
    b_avg = find(r'cycles ---\ncycles per image \(start to done\): average (\d+)', zb, 'budget average')
    b_max = find(r'cycles ---\ncycles per image \(start to done\): average \d+, max (\d+)', zb, 'budget max')
    bound = find(r'bound (\d+)', zb, 'bound')
    cut = find(r'changed by the budget: (\d+) of', zb, 'cut images')
    w_bud = find(r'all-255 image with the budget: (\d+)', zb, 'all-255 budget')
    w_free = find(r'all-255 image without a budget: (\d+)', zb, 'all-255 free')
    ct = find(r'all zero, all 255\): (\d+) cycles', zb, 'constant time')

    pb = budget_pass(sb, 'with budget')
    pc = budget_pass(sb, 'with budget, constant time')
    db = budget_pass(sb, 'direct load, with budget')
    dc = budget_pass(sb, 'direct load, with budget, constant time')
    d_bud = find(r'all-255 image, direct load, budget:\s+(\d+)', sb, 'cpu all-255 direct budget')
    d_ct = find(r'all-255 image, direct load, constant time:\s+(\d+)', sb, 'cpu all-255 direct constant time')
    ld_reads = find(r'loader: (\d+) bus reads', sb, 'loader reads')
    ld_wait = find(r'waited (\d+) cycles for the CPU', sb, 'loader waits')
    c_free = find(r'all-255 image, no budget:\s+(\d+)', sb, 'cpu all-255 free')
    c_bud = find(r'all-255 image, budget:\s+(\d+)', sb, 'cpu all-255 budget')
    c_ct = find(r'all-255 image, constant time:\s+(\d+)', sb, 'cpu all-255 constant time')
    n = S['n']
    fixed = (S['load'] + S['read']) / n      # loading and reading do not depend on the budget
    load_m = (db['acc'] - pb['acc']) / n     # cycles SMAC.RUNM spends on fetching and ordering

    def f(x):
        return f'{x:,.0f}'

    return f'''
### {title}

{intro}Model: the {sp} % pruned network after budget-aware fine-tuning (`models/{model}`),
inputs stored most-useful-first. Layer-1 budget: {f(B)} cycles. Why this model and this
budget: {why}.
The accuracy lines are computed by `software/export_budget.py` on all 10,000 test images with
the same rule as the hardware; the cycle counts are RTL simulation of 100 test images.

| | No budget | Budget {f(B)} | Budget {f(B)} + constant time |
|---|---|---|---|
| Accuracy, 10,000 images | {acc_free:.2f}% | {acc_bud:.2f}% | {acc_bud:.2f}% |
| Accelerator cycles per image, average | {f(f_avg)} | {f(b_avg)} | {f(ct)} |
| Accelerator cycles, slowest of the 100 test images | {f(f_max)} | {f(b_max)} | {f(ct)} |
| Accelerator cycles, all-255 image (slowest possible input) | {f(w_free)} | {f(w_bud)} | {f(ct)} |
| Whole loop on PicoRV32, pixels sent with `SMAC.LDW` | {f(S['all'] / n)} | {f(pb['all'] / n)} | {f(pc['all'] / n)} |
| Whole loop on PicoRV32, image fetched by the accelerator (`SMAC.RUNM`) | - | {f(db['all'] / n)} | {f(dc['all'] / n)} |

- The budget costs {acc_free - acc_bud:.2f} points of accuracy. It cuts the slowest possible input from
  {f(w_free)} to {f(w_bud)} cycles ({w_free / w_bud:.1f}x) and the average from {f(f_avg)} to {f(b_avg)} cycles.
  Every run with the budget stayed below the bound of {f(bound)} cycles checked by the testbench.
- Layer 1 alone, on all 10,000 images (no simulation): average {f(l1_avg)} cycles and maximum {f(l1_max)} without a
  budget, {f(l1_worst)} for the slowest possible input; average {f(l1_bavg)} and maximum {f(B)} with the budget.
- The budget changes the logits of {cut} of the 100 test images. All logits match the golden values of
  `software/budget_lib.py` bit for bit, so the accuracy claim and the hardware use the same rule.
- Constant time: all images, an all-zero and an all-255 image finish after exactly {f(ct)} accelerator cycles.
  Seen from the CPU (`SMAC.CYC`, {n} images): minimum {f(pc['min'])}, maximum {f(pc['max'])};
  with the budget only: minimum {f(pb['min'])}, maximum {f(pb['max'])}.
  Constant time costs {100 * (ct / b_avg - 1):.0f}% more cycles than the budget alone.
- All-255 image through the CPU (`SMAC.CYC`): {f(c_free)} without a budget, {f(c_bud)} with the budget,
  {f(c_ct)} in constant time. The {npass} passes of the SoC simulation were checked against the golden values.
- With `SMAC.LDW` the CPU spends {f(fixed)} cycles per image on sending pixels and reading logits, whatever the
  budget is, and the images in RAM must already be stored most-useful-first.
- With `SMAC.RUNM` the accelerator reads the image from RAM itself (normal pixel order) and applies the
  order: {f(load_m)} cycles for every image, the same for all of them. The whole loop then takes
  {f(db['all'] / n)} cycles on average (minimum {f(db['all_min'])}, maximum {f(db['all_max'])}) with the budget.
  In constant time the CPU measures {f(dc['all_min'])} to {f(dc['all_max'])} cycles for the whole loop over {n} images,
  and the all-255 image takes {f(d_ct)} instruction cycles ({f(d_bud)} with the budget only).
  So the bound and the constant time cover the whole inference, not only the accelerator.
- The loader made {f(ld_reads)} bus reads and waited {f(ld_wait)} cycles in total for the CPU (its one prefetch per instruction).
- Limits: `SMAC.RUNM` spends about as many cycles on fetching and ordering as on computing for this small network.
'''


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
    sw_csc = Z['c_cyc'] / Z['c_n']
    D = budget_pass(soc, 'direct load')      # SMAC.RUNM pass of the normal SoC simulation
    d_all, d_acc = D['all'] / n, D['acc'] / n

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
state per access. Software figures are averages over {Z['d_n']} (dense), {Z['s_n']} (skip zero pixels)
and {Z['c_n']} (CSC) images.

| System | Cycles per image | Relative to dense software |
|---|---|---|
| CPU only, plain dense C | {f(sw_dense)} | 1.0x |
| CPU only, C that skips zero pixels | {f(sw_skip)} | {sw_dense / sw_skip:.1f}x |
| CPU only, C with the accelerator's algorithm (CSC: skips zero weights and activations) | {f(sw_csc)} | {sw_dense / sw_csc:.1f}x |
| CPU + `sparse_mlp.v` (zero weights), whole loop | {f(w_all)} | {sw_dense / w_all:.0f}x |
| CPU + `sparse_mlp_zs.v` (zero weights and activations), whole loop | {f(z_all)} | {sw_dense / z_all:.0f}x |
| CPU + `sparse_mlp_zs.v`, image fetched by the accelerator (`SMAC.RUNM`), whole loop | {f(d_all)} | {sw_dense / d_all:.0f}x |

The accelerator rows include loading the image, running, and reading the ten
logits. Split for `sparse_mlp_zs.v` with `SMAC.LDW`:

| Part | Cycles per image |
|---|---|
| Load pixels (196 x `SMAC.LDW`) | {f(Z['load'] / n)} |
| `SMAC.RUN` (waiting for the accelerator) | {f(Z['run'] / n)} |
| Read 10 logits (`SMAC.LOGIT`) | {f(Z['read'] / n)} |
| Accelerator compute only (`SMAC.CYC`) | {f(z_cmp)} |

- Fair comparison (same algorithm and weight format in C on the CPU): the whole
  accelerated loop is {sw_csc / z_all:.1f}x faster than the CSC software, the compute alone {sw_csc / z_cmp:.1f}x.
- The larger ratios against the dense and skip-zero-pixels programs mostly measure a
  better algorithm, not the hardware: the CSC software alone is already {sw_dense / sw_csc:.1f}x faster than dense C.
- Loading the image costs {Z['load'] / Z['all'] * 100:.0f}% of the accelerated loop with `SMAC.LDW`.
- With `SMAC.RUNM` the accelerator reads the image from RAM itself: {f(d_all)} cycles for the whole loop,
  {z_all / d_all:.1f}x faster than with `SMAC.LDW` and {sw_csc / d_all:.1f}x faster than the CSC software. The instruction
  itself takes {f(d_acc)} cycles, of which {f(d_acc - z_cmp)} are fetching and ordering the pixels.
- Accuracy on 100 images through the CPU: {Z['correct']}/{n}. Software and accelerator logits
  were compared by the firmware: {Z['mism']} mismatches. The testbench also compared all {n * 10} logits with `golden_logits`.
- Handshake tests of the custom instructions without a CPU: `make sim-pcpi` passes
  (including instructions that must be ignored).
'''
    out += budget_section('budget')
    out += budget_section('fashion')

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
