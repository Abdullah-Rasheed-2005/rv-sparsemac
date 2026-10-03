"""
verify_export.py - Check the exported hex files
===============================================
Reads the .hex files back the same way Verilog will read them,
runs the integer network, and checks that:
  - the final scores match golden_logits.txt exactly,
  - the predictions match golden_pred.txt.
It also counts how many multiply-accumulates (MACs) are needed with
different zero-skipping methods.

Run from the repository root (after export_int8.py):
    python3 software/verify_export.py
"""

import numpy as np

from paths import MEM_DIR


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


W1 = load_hex(MEM_DIR / 'fc1_weights.hex', 8).reshape(64, 784)
B1 = load_hex(MEM_DIR / 'fc1_bias.hex', 32)
W2 = load_hex(MEM_DIR / 'fc2_weights.hex', 8).reshape(10, 64)
B2 = load_hex(MEM_DIR / 'fc2_bias.hex', 32)
X = load_hex(MEM_DIR / 'test_images.hex', 8, signed=False).reshape(-1, 784)

# Read the settings (we need the shift value)
params = {}
with open(MEM_DIR / 'params.txt') as f:
    for line in f:
        k, v = line.strip().split('=')
        params[k] = v
shift = int(params['shift'])

golden_logits = np.loadtxt(MEM_DIR / 'golden_logits.txt', dtype=np.int64)
golden_pred = np.loadtxt(MEM_DIR / 'golden_pred.txt', dtype=np.int64)
labels = np.loadtxt(MEM_DIR / 'test_labels.txt', dtype=np.int64)
N = X.shape[0]

exact = pred_match = correct = 0
macs = {'dense': [], 'weight': [], 'input': [], 'both': []}

for n in range(N):
    x = X[n]
    acc1 = W1 @ x + B1
    h = np.minimum(np.maximum(acc1, 0) >> shift, 127)   # ReLU + shift + saturate
    acc2 = W2 @ h + B2

    if np.array_equal(acc2, golden_logits[n]):
        exact += 1
    p = int(acc2.argmax())
    if p == golden_pred[n]:
        pred_match += 1
    if p == labels[n]:
        correct += 1

    xn = x != 0          # which pixels are nonzero
    hn = h != 0          # which hidden values are nonzero
    macs['dense'].append(64 * 784 + 10 * 64)
    macs['weight'].append(np.count_nonzero(W1) + np.count_nonzero(W2))
    macs['input'].append(64 * int(xn.sum()) + 10 * int(hn.sum()))
    macs['both'].append(np.count_nonzero(W1[:, xn]) + np.count_nonzero(W2[:, hn]))

print(f'Images checked             : {N}')
print(f'Golden logits exact match  : {exact}/{N}')
print(f'Golden prediction match    : {pred_match}/{N}')
print(f'Accuracy on these {N} imgs : {100 * correct / N:.1f}%')
print()
print('Average MACs per image (fc1 + fc2):')
base = np.mean(macs['dense'])
for name, label in [('dense', 'Dense (no skipping)'),
                    ('weight', 'Skip zero weights only'),
                    ('input', 'Skip zero inputs only'),
                    ('both', 'Skip weights + inputs')]:
    m = np.mean(macs[name])
    print(f'  {label:26s}: {m:9.0f}   ({base / m:5.1f}x fewer)')
