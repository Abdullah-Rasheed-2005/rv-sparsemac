"""
export_dense_lists.py - Write the DENSE weight lists in the sparse file format
===============================================================================
The sparse engine reads, for every neuron, a list of (index, value) pairs.
If that list contains EVERY weight of the neuron (zeros included), the very
same engine does a dense dot product with no skipping at all. That gives a
measured dense baseline for the cycle comparison, built from the same
hardware, so the only difference between the designs is what they skip.

Files are written into build/dense_mem/ (not committed, regenerated each run):

  fc1_nz_ptr.hex  fc1_nz_idx.hex  fc1_nz_val.hex   (same names and formats
  fc2_nz_ptr.hex  fc2_nz_idx.hex  fc2_nz_val.hex    as the sparse files)

Run from the repository root (needs only numpy):
    python3 software/export_dense_lists.py
"""

import sys

import numpy as np

from paths import MEM_DIR, ROOT

OUT_DIR = ROOT / 'build' / 'dense_mem'


def load_hex(path, bits):
    """Read a hex file of signed two's complement values."""
    vals = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            v = int(s, 16)
            if v >= (1 << (bits - 1)):
                v -= (1 << bits)
            vals.append(v)
    return np.array(vals, dtype=np.int64)


def write_hex(path, values, digits):
    mask = (1 << (4 * digits)) - 1
    with open(path, 'w') as f:
        for v in values:
            f.write(f'{int(v) & mask:0{digits}x}\n')


OUT_DIR.mkdir(parents=True, exist_ok=True)

for name, n_out, n_in in (('fc1', 64, 784), ('fc2', 10, 64)):
    W = load_hex(MEM_DIR / f'{name}_weights.hex', 8).reshape(n_out, n_in)
    total = n_out * n_in
    if total >= 1 << 16:
        sys.exit(f'{name}: {total} entries do not fit in a 16-bit list address')

    idx = np.tile(np.arange(n_in), n_out)          # 0..n_in-1 for every neuron
    val = W.reshape(-1)                            # every weight, zeros included
    ptr = np.arange(0, total + 1, n_in)            # neuron n owns n*n_in .. (n+1)*n_in-1

    write_hex(OUT_DIR / f'{name}_nz_idx.hex', idx, 4)
    write_hex(OUT_DIR / f'{name}_nz_val.hex', val, 2)
    write_hex(OUT_DIR / f'{name}_nz_ptr.hex', ptr, 4)
    print(f'{name}: {total} entries ({int(np.count_nonzero(W))} nonzero, '
          f'{total - int(np.count_nonzero(W))} zero weights kept)')

print(f'Dense lists written -> {OUT_DIR}')
