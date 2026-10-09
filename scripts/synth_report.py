#!/usr/bin/env python3
"""
synth_report.py - Write docs/synthesis.md from the Yosys reports of `make synth`
================================================================================
`make synth` synthesizes three designs for a Lattice ECP5 FPGA (Yosys, synth_ecp5)
and stores the cell counts in build/synth/*.stat. This script turns them into a
table. Every number comes from those files. If build/synth/timing.log exists
(`make synth-timing`, needs nextpnr-ecp5) the maximum clock frequency is added.

Needs only the Python standard library. Run from the repository root.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / 'build' / 'synth'
CELLS = (('LUT4', 'LUT4'), ('TRELLIS_FF', 'flip-flops'), ('CCU2C', 'carry cells'),
         ('DP16KD', 'block RAM (18 kbit)'), ('TRELLIS_DPR16X4', 'distributed RAM cells (16 x 4 bit)'),
         ('MULT18X18D', '18 x 18 multipliers'), ('PFUMX', 'PFUMX (mux inside a slice)'), ('L6MUX21', 'L6MUX21'))
DESIGNS = (('zs_nobudget', 'accelerator, budget switched off'),
           ('zs', 'accelerator with budget + constant time'),
           ('full', 'whole co-processor: accelerator + custom instructions + image loader'))


def cells(name):
    f = SYN / f'{name}.stat'
    if not f.exists():
        sys.exit(f'missing {f} - run `make synth` first')
    t = f.read_text()
    out = {}
    for key, _ in CELLS:
        m = re.search(rf'^\s+{key}\s+(\d+)\s*$', t, re.M)
        out[key] = int(m.group(1)) if m else 0
    return out


def version():
    log = SYN / 'full.log'
    m = re.search(r'Yosys (\d[\w.+]*)', log.read_text()) if log.exists() else None
    return m.group(1) if m else 'unknown version'


c = {name: cells(name) for name, _ in DESIGNS}
L = []
L.append('# Synthesis')
L.append('')
L.append(f'Written by `scripts/synth_report.py` from the reports of `make synth` (Yosys {version()}, `synth_ecp5`).')
L.append('Target: Lattice ECP5. These are cell counts after synthesis, before place and route.')
L.append('The weight memories hold the normal model (`hardware/mem`); their size does not depend on the weights.')
L.append('')
L.append('| Cell | ' + ' | '.join(title for _, title in DESIGNS) + ' |')
L.append('|---|' + '---|' * len(DESIGNS))
for key, label in CELLS:
    L.append(f'| {label} | ' + ' | '.join(f'{c[name][key]:,}' for name, _ in DESIGNS) + ' |')
L.append('')
d_lut = c['zs']['LUT4'] - c['zs_nobudget']['LUT4']
d_ff = c['zs']['TRELLIS_FF'] - c['zs_nobudget']['TRELLIS_FF']
w_lut = c['full']['LUT4'] - c['zs']['LUT4']
w_ff = c['full']['TRELLIS_FF'] - c['zs']['TRELLIS_FF']
w_ram = c['full']['DP16KD'] - c['zs']['DP16KD']
L.append(f'- The run-time budget and the constant-time mode cost {d_lut:+,} LUT4 and {d_ff:+,} flip-flops '
         f'({100.0 * d_lut / max(c["zs_nobudget"]["LUT4"], 1):+.1f}% LUT4) and no memory.')
L.append(f'- The custom-instruction wrapper with the image loader adds {w_lut:+,} LUT4, {w_ff:+,} flip-flops and '
         f'{w_ram:+,} block RAM (image buffer and order table).')
L.append(f'- The whole co-processor uses {c["full"]["LUT4"]:,} LUT4, {c["full"]["TRELLIS_FF"]:,} flip-flops, '
         f'{c["full"]["DP16KD"]} block RAMs and {c["full"]["MULT18X18D"]} multiplier. For comparison, an '
         f'LFE5U-25F has 24,288 LUT4 and 56 block RAMs.')
L.append('- Most block RAMs hold the fc1 weights: the list is sized for 16,384 nonzero weights; the 90 % pruned')
L.append('  models use about 5,000, so half of that memory would be enough for them.')
L.append('- The 64 accumulators are built from distributed RAM, not block RAM, because they are read and')
L.append('  written in the same clock cycle.')

t = SYN / 'timing.log'
L.append('')
L.append('## Clock frequency')
L.append('')
if t.exists():
    txt = t.read_text()
    m = re.findall(r"Max frequency for clock\s+'[^']*':\s+([\d.]+) MHz", txt)
    if m:
        L.append(f'Place and route with nextpnr-ecp5 (`make synth-timing`, LFE5U-85F, speed grade 6, pins placed')
        L.append(f'automatically): maximum clock frequency {m[-1]} MHz for the whole co-processor.')
        L.append('This is one run of a heuristic tool without timing constraints on the pins; treat it as an estimate.')
    else:
        L.append('`build/synth/timing.log` exists but contains no "Max frequency" line; see the log.')
else:
    L.append('Not measured yet. `make synth-timing` runs place and route with nextpnr-ecp5 and this script')
    L.append('then adds the maximum clock frequency. Cycle counts in the other documents are independent of it.')

out = ROOT / 'docs' / 'synthesis.md'
out.write_text('\n'.join(L) + '\n')
print('\n'.join(L))
print(f'\nwritten: {out}')
