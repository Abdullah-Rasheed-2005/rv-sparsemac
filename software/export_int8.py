"""
export_int8.py - Pure integer inference and hex export for Verilog
==================================================================
This script does three things:
  1. Quantizes the pruned model to int8 and runs inference using only
     integer maths (this is what the hardware will do).
  2. Prints the accuracy and the sparsity numbers.
  3. Writes .hex files (for $readmemh in Verilog) and "golden" test
     vectors into hardware/mem/. The Verilog testbench output must
     match the golden values exactly.

Integer pipeline:
  acc1   = pixels(0..255) * int8 weights + bias        (32-bit)
  hidden = min(max(acc1, 0) >> SHIFT, 127)             (ReLU + shift)
  acc2   = hidden * int8 weights + bias                (32-bit)
  answer = index of the largest acc2                   (argmax)

Run from the repository root (needs models/pruned_fc1_80.pth):
    python3 software/export_int8.py
"""

import torch
from torchvision import datasets

from paths import DATA_DIR, MEM_DIR, pruned_model_path

SPARSITY = 80   # which pruned model to export

state = torch.load(pruned_model_path(SPARSITY))
W1, b1 = state['fc1.weight'], state['fc1.bias']
W2, b2 = state['fc2.weight'], state['fc2.bias']

train = datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
test = datasets.MNIST(root=str(DATA_DIR), train=False, download=True)
Xtr = train.data.view(-1, 784).long()      # raw pixels, 0..255
Xte = test.data.view(-1, 784).long()
Yte = test.targets


def imatmul(a, b):
    """Integer matrix multiply. Done in float64, which is exact here
    because all the numbers are small."""
    return (a.double() @ b.double()).round().long()


def quantize(w):
    """Symmetric int8: scale = max|w| / 127. Returns (int weights, scale)."""
    s = w.abs().max().item() / 127
    q = torch.clamp(torch.round(w / s), -127, 127).long()
    return q, s


q1, s1 = quantize(W1)
q2, s2 = quantize(W2)
sx = 1.0 / 255.0                            # scale of one pixel step

# Put the bias on the same scale as the accumulator
b1_int = torch.round(b1 / (sx * s1)).long()

# Calibration: look at the biggest hidden value, then pick a right-shift
# so the hidden values fit into 0..127.
acc1_cal = imatmul(Xtr[:10000], q1.T) + b1_int
max_act = acc1_cal.clamp(min=0).max().item()
shift = 0
while (max_act >> shift) > 127:
    shift += 1

# Layer 2 bias must match the scale of acc2
b2_int = torch.round(b2 / (sx * s1 * (2 ** shift) * s2)).long()


def int_forward(X):
    """Integer-only forward pass. Returns hidden values and final scores."""
    acc1 = imatmul(X, q1.T) + b1_int
    h = (acc1.clamp(min=0) >> shift).clamp(max=127)   # ReLU + shift + saturate
    acc2 = imatmul(h, q2.T) + b2_int
    return h, acc2


h, acc2 = int_forward(Xte)
pred = acc2.argmax(dim=1)
acc = (pred == Yte).float().mean().item() * 100

# Float version of the same pruned model, for comparison
hf = torch.relu((Xte.float() / 255.0) @ W1.T + b1)
accf = (((hf @ W2.T + b2).argmax(dim=1) == Yte).float().mean().item()) * 100

print(f'Float (pruned {SPARSITY}%)      : {accf:.2f}%')
print(f'Pure integer pipeline    : {acc:.2f}%')
print(f'Hidden shift             : {shift}  (calibration max acc1 = {max_act})')
print(f'fc1 nonzero weights      : {(q1 != 0).sum().item()} / {q1.numel()}')
print(f'fc2 nonzero weights      : {(q2 != 0).sum().item()} / {q2.numel()}')
print(f'Input pixels zero        : {(Xte == 0).float().mean().item()*100:.1f}%')
print(f'Hidden activations zero  : {(h == 0).float().mean().item()*100:.1f}%')

# ---------------- Export ----------------
MEM_DIR.mkdir(parents=True, exist_ok=True)


def w8(path, t):
    """Write one 8-bit value per line as 2 hex digits (two's complement)."""
    with open(path, 'w') as f:
        for v in t.flatten().tolist():
            f.write(f'{v & 0xFF:02x}\n')


def w32(path, t):
    """Write one 32-bit value per line as 8 hex digits (two's complement)."""
    with open(path, 'w') as f:
        for v in t.flatten().tolist():
            f.write(f'{v & 0xFFFFFFFF:08x}\n')


w8(MEM_DIR / 'fc1_weights.hex', q1)         # 64 x 784, one neuron after another
w32(MEM_DIR / 'fc1_bias.hex', b1_int)
w8(MEM_DIR / 'fc2_weights.hex', q2)         # 10 x 64
w32(MEM_DIR / 'fc2_bias.hex', b2_int)

N = 100                                     # number of golden test vectors
w8(MEM_DIR / 'test_images.hex', Xte[:N])
with open(MEM_DIR / 'test_labels.txt', 'w') as f:
    f.write('\n'.join(str(v) for v in Yte[:N].tolist()) + '\n')
with open(MEM_DIR / 'golden_logits.txt', 'w') as f:
    for row in acc2[:N].tolist():
        f.write(' '.join(str(v) for v in row) + '\n')
with open(MEM_DIR / 'golden_pred.txt', 'w') as f:
    f.write('\n'.join(str(v) for v in pred[:N].tolist()) + '\n')
with open(MEM_DIR / 'params.txt', 'w') as f:
    f.write(f'sparsity={SPARSITY}\nshift={shift}\ns1={s1}\ns2={s2}\n')

print(f'Export done -> {MEM_DIR}')
