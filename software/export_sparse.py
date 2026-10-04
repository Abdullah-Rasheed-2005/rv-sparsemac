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

For the whole-network testbench (tb_sparse_mlp.v) it also writes:

  hidden_shift.hex     the hidden right-shift from params.txt (2 hex digits)
  golden_logits.hex    the golden logits of all images, 10 per image (8 hex digits)

For the zero-skipping design (sparse_mlp_zs.v) the weights are also written
BY COLUMN (CSC format): column j holds the nonzero weights that multiply
input j. Column j owns the entries ptr[j] .. ptr[j+1]-1, ordered by row.

  fcN_csc_ptr.hex  start of each column  (4 hex digits; 785 lines for fc1, 65 for fc2)
  fcN_csc_row.hex  output neuron of each entry   (2 hex digits)
  fcN_csc_val.hex  the int8 weight of each entry (2 hex digits)

Run from the repository root (after export_int8.py). Needs only numpy:
    python3 software/export_sparse.py
(An optional folder argument makes it work on another set of files, see make_edge_model.py.)
"""

import sys

import numpy as np

from pathlib import Path

from paths import MEM_DIR

# Optional: python3 software/export_sparse.py <folder>  (used by make_edge_model.py)
if len(sys.argv) > 1:
    MEM_DIR = Path(sys.argv[1])

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


def to_csc(W):
    """Dense matrix (neurons x inputs) -> column lists (row list, value list, pointers).
    Column j = all nonzero weights that multiply input j, in row order."""
    row, val, ptr = [], [], [0]
    for j in range(W.shape[1]):
        nz = np.nonzero(W[:, j])[0]
        row.extend(nz.tolist())
        val.extend(W[nz, j].tolist())
        ptr.append(len(row))
    return row, val, ptr


def csc_to_dense(row, val, ptr, n_out):
    """Rebuild the dense matrix from the column lists (self-check)."""
    W = np.zeros((n_out, len(ptr) - 1), dtype=np.int64)
    for j in range(len(ptr) - 1):
        for k in range(ptr[j], ptr[j + 1]):
            W[row[k], j] = val[k]
    return W


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

    # the same weights, stored by column (for sparse_mlp_zs.v)
    crow, cval, cptr = to_csc(W)
    if not np.array_equal(csc_to_dense(crow, cval, cptr, W.shape[0]), W):
        sys.exit(f'{name}: column lists do not rebuild the dense matrix')
    write_hex(MEM_DIR / f'{name}_csc_ptr.hex', cptr, 4)
    write_hex(MEM_DIR / f'{name}_csc_row.hex', crow, 2)
    write_hex(MEM_DIR / f'{name}_csc_val.hex', cval, 2)
    print(f'{name}: column lists written (longest column = {np.diff(cptr).max()} entries)')

# ---------------- Golden dot products (no bias) ----------------
Xn = X[:N_TEST]
acc1 = Xn @ W1.T                                          # (N, 64)
hidden = np.minimum(np.maximum(acc1 + B1, 0) >> shift, 127)
acc2 = hidden @ W2.T                                      # (N, 10)

# Cross-check against the golden logits written by export_int8.py
if not np.array_equal(acc2 + B2, golden_logits[:N_TEST]):
    sys.exit('cross-check failed: dot products + bias do not match golden_logits.txt')
print(f'cross-check against golden_logits.txt: OK ({N_TEST} images)')

# Same check on ALL exported images, and the number of useful MACs when both
# kinds of zeros are skipped (this is what sparse_mlp_zs.v needs in cycles).
acc1_all = X @ W1.T
h_all = np.minimum(np.maximum(acc1_all + B1, 0) >> shift, 127)
if not np.array_equal(h_all @ W2.T + B2, golden_logits):
    sys.exit('cross-check failed on the full image set')
nzW1, nzW2 = (W1 != 0).astype(np.int64), (W2 != 0).astype(np.int64)
pairs1 = ((X != 0).astype(np.int64) @ nzW1.T).sum(axis=1)       # per image, fc1
pairs2 = ((h_all != 0).astype(np.int64) @ nzW2.T).sum(axis=1)   # per image, fc2
print(f'useful MACs per image when skipping both zeros ({len(X)} images): '
      f'fc1 = {pairs1.mean():.0f}, fc2 = {pairs2.mean():.0f}, total = {(pairs1 + pairs2).mean():.0f}')

write_hex(MEM_DIR / 'golden_fc1_acc.hex', acc1.flatten(), 8)
write_hex(MEM_DIR / 'golden_hidden.hex', hidden.flatten(), 2)
write_hex(MEM_DIR / 'golden_fc2_acc.hex', acc2.flatten(), 8)

# Whole-network files: the shift and the golden logits in hex form
write_hex(MEM_DIR / 'hidden_shift.hex', [shift], 2)
write_hex(MEM_DIR / 'golden_logits.hex', golden_logits.flatten(), 8)

print(f'Export done -> {MEM_DIR}')
