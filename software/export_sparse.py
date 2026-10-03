"""
export_sparse.py - Turn the dense weight files into sparse lists
=================================================================
The dense files (fc1_weights.hex, fc2_weights.hex) contain every weight,
zeros included. The sparse hardware wants only the nonzero weights, each
with the input position it belongs to.

For each layer this script writes three files into hardware/mem/:

  fcN_nz_idx.hex   input index of each nonzero weight   (4 hex digits per line)
  fcN_nz_val.hex   the int8 weight itself                (2 hex digits per line)
  fcN_nz_ptr.hex   start of each neuron in the two lists (4 hex digits per line)

Neuron n owns the list entries ptr[n] .. ptr[n+1]-1. So ptr has one more
entry than the layer has neurons (65 for fc1, 11 for fc2), and the last
entry is the total number of nonzero weights.

It also writes golden values for the first N_TEST images, so the Verilog
testbench can check the raw dot products (bias NOT included):

  golden_fc1_acc.hex   W1 @ pixels, 64 values per image   (8 hex digits)
  golden_hidden.hex    hidden values, 64 per image        (2 hex digits)
  golden_fc2_acc.hex   W2 @ hidden, 10 values per image   (8 hex digits)

Run from the repository root (after export_int8.py). Needs only numpy:
    python3 software/export_sparse.py
"""

import sys

import numpy as np

from paths import MEM_DIR

N_TEST = 10   # number of images for the golden values (tb_sparse_dot.v uses 10)


def load_hex(path, bits, signed=True):
    """Read a hex file and turn each line into an integer."""
    vals = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            v = int(s, 16)
            if signed and v >= (1 << (bits - 1)):
                v -= (1 << bits)   # two's complement: big values are negative
            vals.append(v)
    return np.array(vals, dtype=np.int64)


def write_hex(path, values, digits):
    """One value per line, 'digits' hex digits, two's complement."""
    mask = (1 << (4 * digits)) - 1
    with open(path, 'w') as f:
        for v in values:
            f.write(f'{int(v) & mask:0{digits}x}\n')


def to_sparse(W):
    """Dense matrix (neurons x inputs) -> (index list, value list, pointers)."""
    idx, val, ptr = [], [], [0]
    for row in W:
        nz = np.nonzero(row)[0]            # positions of the nonzero weights
        idx.extend(nz.tolist())
        val.extend(row[nz].tolist())
        ptr.append(len(idx))
    return idx, val, ptr


def to_dense(idx, val, ptr, n_in):
    """Rebuild the dense matrix from the sparse lists (used as a self-check)."""
    W = np.zeros((len(ptr) - 1, n_in), dtype=np.int64)
    for n in range(len(ptr) - 1):
        for k in range(ptr[n], ptr[n + 1]):
            W[n, idx[k]] = val[k]
    return W


# ---------------- Read the dense files ----------------
W1 = load_hex(MEM_DIR / 'fc1_weights.hex', 8).reshape(64, 784)
B1 = load_hex(MEM_DIR / 'fc1_bias.hex', 32)
W2 = load_hex(MEM_DIR / 'fc2_weights.hex', 8).reshape(10, 64)
B2 = load_hex(MEM_DIR / 'fc2_bias.hex', 32)
X = load_hex(MEM_DIR / 'test_images.hex', 8, signed=False).reshape(-1, 784)

params = {}
with open(MEM_DIR / 'params.txt') as f:
    for line in f:
        k, v = line.strip().split('=')
        params[k] = v
shift = int(params['shift'])

golden_logits = np.loadtxt(MEM_DIR / 'golden_logits.txt', dtype=np.int64)

# ---------------- Sparse lists ----------------
for name, W, n_in in (('fc1', W1, 784), ('fc2', W2, 64)):
    idx, val, ptr = to_sparse(W)

    # Limits of sparse_dot.v: 16-bit list address, 10-bit input index
    if len(idx) >= 1 << 16:
        sys.exit(f'{name}: {len(idx)} nonzero weights do not fit in a 16-bit address')
    if n_in > 1 << 10:
        sys.exit(f'{name}: {n_in} inputs do not fit in a 10-bit index')
    if not np.array_equal(to_dense(idx, val, ptr, n_in), W):
        sys.exit(f'{name}: sparse lists do not rebuild the dense matrix')

    write_hex(MEM_DIR / f'{name}_nz_idx.hex', idx, 4)
    write_hex(MEM_DIR / f'{name}_nz_val.hex', val, 2)
    write_hex(MEM_DIR / f'{name}_nz_ptr.hex', ptr, 4)

    per_neuron = np.diff(ptr)
    print(f'{name}: {len(idx)} nonzero of {W.size} weights, '
          f'per neuron min={per_neuron.min()} max={per_neuron.max()}')

# ---------------- Golden dot products (no bias) ----------------
Xn = X[:N_TEST]
acc1 = Xn @ W1.T                                          # (N, 64)
hidden = np.minimum(np.maximum(acc1 + B1, 0) >> shift, 127)
acc2 = hidden @ W2.T                                      # (N, 10)

# Cross-check against the golden logits written by export_int8.py
if not np.array_equal(acc2 + B2, golden_logits[:N_TEST]):
    sys.exit('cross-check failed: dot products + bias do not match golden_logits.txt')
print(f'cross-check against golden_logits.txt: OK ({N_TEST} images)')

write_hex(MEM_DIR / 'golden_fc1_acc.hex', acc1.flatten(), 8)
write_hex(MEM_DIR / 'golden_hidden.hex', hidden.flatten(), 2)
write_hex(MEM_DIR / 'golden_fc2_acc.hex', acc2.flatten(), 8)

print(f'Export done -> {MEM_DIR}')
