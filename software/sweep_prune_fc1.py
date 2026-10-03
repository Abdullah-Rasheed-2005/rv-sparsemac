"""
sweep_prune_fc1.py - Prune only fc1, keep fc2 dense
===================================================
fc1 holds about 98.7% of all weights, so it is the layer that matters
for sparse hardware. fc2 is small and sensitive, so it stays dense
(it is only quantized to int8).

Run from the repository root:   python3 software/sweep_prune_fc1.py
"""

import torch
from torchvision import datasets, transforms

from paths import DATA_DIR, BASELINE_MODEL

state = torch.load(BASELINE_MODEL)
W1, b1 = state['fc1.weight'], state['fc1.bias']
W2, b2 = state['fc2.weight'], state['fc2.bias']

test = datasets.MNIST(root=str(DATA_DIR), train=False, download=True,
                      transform=transforms.ToTensor())
X = test.data.float().view(-1, 784) / 255.0
Y = test.targets


def quantize(w):
    """Symmetric int8: scale = max|w| / 127"""
    scale = w.abs().max().item() / 127
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q, scale


def prune(w, fraction):
    """Set the smallest |w| values to zero."""
    k = int(w.numel() * fraction)
    if k == 0:
        return w
    thr = w.abs().flatten().kthvalue(k).values
    return torch.where(w.abs() > thr, w, torch.zeros_like(w))


def accuracy(w1, w2):
    h = torch.relu(X @ w1.T + b1)
    out = h @ w2.T + b2
    return (out.argmax(dim=1) == Y).float().mean().item() * 100


# fc2 always stays dense (only int8 quantized)
q2, s2 = quantize(W2)
W2_deq = q2.float() * s2

print(f'Float baseline: {accuracy(W1, W2):.2f}%')
print('fc2 dense, only fc1 pruned:')

for frac in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
    q1, s1 = quantize(prune(W1, frac))
    acc = accuracy(q1.float() * s1, W2_deq)
    zeros = (q1 == 0).sum().item() / q1.numel() * 100
    print(f'  fc1 prune {int(frac*100):2d}% : acc={acc:.2f}%  fc1 zeros={zeros:.1f}%')
