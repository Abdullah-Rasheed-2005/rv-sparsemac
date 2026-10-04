"""
make_edge_model.py - A small artificial network built to break the hardware
==========================================================================
The real MNIST model is "too nice" for testing: its logits never tie, and no
hidden value ever reaches the saturation limit 127 on the test images. A
design that gets those two wrong would still pass every golden test.

This script writes a second network, in exactly the same file format as
export_int8.py + export_sparse.py, into build/edge_mem/. It contains on
purpose:

  * hidden values that saturate at 127          (neuron 3 has a huge bias, neuron 1 huge weights)
  * a hidden neuron that is always 0 after ReLU (neuron 2: negative weights only)
  * a completely pruned fc1 neuron               (neuron 0: no nonzero weight, count = 0)
  * input columns without any weight             (pixels 5 and 700, hidden input 10)
  * two classes with exactly the same logit      (classes 3 and 7 -> the first one must win)
  * images: all zero, all 255, and random ones with 2..100 % nonzero pixels

Run from the repository root (needs only numpy):
    python3 software/make_edge_model.py
Then simulate with `make sim-zs-edge`.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

from paths import ROOT

OUT = ROOT / 'build' / 'edge_mem'
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2024)
SHIFT = 9
NIMG = 100


def sparse_rows(n_out, n_in, density, vmax=127):
    W = rng.integers(-vmax, vmax + 1, size=(n_out, n_in))
    W[rng.random((n_out, n_in)) >= density] = 0
    return W


# ---------------- fc1 (64 x 784) ----------------
W1 = sparse_rows(64, 784, 0.10)
B1 = rng.integers(-3000, 3000, size=64)
W1[0, :] = 0                                    # neuron 0: fully pruned
B1[0] = 70000                                   # its hidden value comes from the bias only
W1[1, :] = 0
W1[1, rng.choice(784, 200, replace=False)] = 127   # neuron 1: huge positive sum -> saturates
W1[2, :] = -np.abs(rng.integers(1, 128, size=784))  # neuron 2: only negative weights
B1[2] = 0
B1[3] = 1_000_000                               # neuron 3: always saturated
W1[:, 5] = 0                                    # input columns that nothing uses
W1[:, 700] = 0
W1[1:, 783] = rng.integers(-127, 128, size=63)  # the last pixel is used by everybody except neuron 0

# ---------------- fc2 (10 x 64) ----------------
W2 = rng.integers(-127, 128, size=(10, 64))
W2[:, 10] = 0                                   # hidden input 10 unused
W2[7, :] = W2[3, :]                             # classes 3 and 7 are twins ...
B2 = rng.integers(-2000, 2000, size=10)
B2[3] = B2[7] = 5_000_000                       # ... and both beat every other class

# ---------------- images ----------------
X = np.zeros((NIMG, 784), dtype=np.int64)
X[1, :] = 255
dens = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
for i in range(2, NIMG):
    d = dens[i % len(dens)]
    X[i] = rng.integers(1, 256, size=784) * (rng.random(784) < d)
X[3, 0] = 255                                   # first and last pixel set
X[3, 783] = 255

# ---------------- integer forward pass (same as export_int8.py) ----------------
acc1 = X @ W1.T + B1
H = np.minimum(np.maximum(acc1, 0) >> SHIFT, 127)
acc2 = H @ W2.T + B2
pred = acc2.argmax(axis=1)                      # numpy: the first maximum wins
print(f'hidden values at 127: {(H == 127).sum()}   zero: {(H == 0).mean() * 100:.0f}%   '
      f'tie 3/7 on {(acc2[:, 3] == acc2[:, 7]).sum()} images, pred 3: {(pred == 3).sum()}')
assert (acc2[:, 3] == acc2[:, 7]).all() and (pred == 3).all()
assert np.abs(acc1).max() < 2 ** 31 and np.abs(acc2).max() < 2 ** 31


def w8(name, t):
    (OUT / name).write_text(''.join(f'{int(v) & 0xFF:02x}\n' for v in np.ravel(t)))


def w32(name, t):
    (OUT / name).write_text(''.join(f'{int(v) & 0xFFFFFFFF:08x}\n' for v in np.ravel(t)))


w8('fc1_weights.hex', W1)
w32('fc1_bias.hex', B1)
w8('fc2_weights.hex', W2)
w32('fc2_bias.hex', B2)
w8('test_images.hex', X)
(OUT / 'test_labels.txt').write_text('\n'.join(str(int(v) % 10) for v in pred) + '\n')
(OUT / 'golden_logits.txt').write_text(''.join(' '.join(map(str, r)) + '\n' for r in acc2))
(OUT / 'golden_pred.txt').write_text('\n'.join(map(str, pred)) + '\n')
(OUT / 'params.txt').write_text(f'sparsity=90\nshift={SHIFT}\ns1=1\ns2=1\n')

# sparse + column files, golden hidden values, hidden_shift.hex: reuse the normal exporter
subprocess.run([sys.executable, str(Path(__file__).with_name('export_sparse.py')), str(OUT)], check=True)
print(f'Edge model written to {OUT}')
