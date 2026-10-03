"""
sweep_prune_both_layers.py - Prune BOTH layers, then quantize to int8
=====================================================================
For each pruning level this script:
  1. prunes fc1 and fc2 by the same fraction,
  2. quantizes the weights to int8,
  3. turns them back into floats ("fake quantization"),
  4. measures the test accuracy.

Result: pruning both layers hurts accuracy a lot, because the small
fc2 layer is sensitive. See sweep_prune_fc1.py for the better approach.

Run from the repository root:   python3 software/sweep_prune_both_layers.py
"""

import torch
from torchvision import datasets, transforms

from paths import DATA_DIR, BASELINE_MODEL

state = torch.load(BASELINE_MODEL)
W1, b1 = state['fc1.weight'], state['fc1.bias']
W2, b2 = state['fc2.weight'], state['fc2.bias']

# Put all test images into one tensor (this makes the test fast)
test = datasets.MNIST(root=str(DATA_DIR), train=False, download=True,
                      transform=transforms.ToTensor())
X = test.data.float().view(-1, 784) / 255.0   # pixels scaled to 0..1
Y = test.targets


def quantize(w):
    """Symmetric int8: scale = max|w| / 127"""
    scale = w.abs().max().item() / 127
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q, scale


def prune(w, fraction):
    """Set the smallest |w| values to zero. 'fraction' is how many."""
    k = int(w.numel() * fraction)
    if k == 0:
        return w
    thr = w.abs().flatten().kthvalue(k).values   # the k-th smallest size
    return torch.where(w.abs() > thr, w, torch.zeros_like(w))


def accuracy(w1, w2):
    """Run the two-layer network and return the accuracy in percent."""
    h = torch.relu(X @ w1.T + b1)
    out = h @ w2.T + b2
    return (out.argmax(dim=1) == Y).float().mean().item() * 100


print(f'Float baseline        : {accuracy(W1, W2):.2f}%')

for frac in [0.0, 0.3, 0.5, 0.7, 0.9]:
    q1, s1 = quantize(prune(W1, frac))
    q2, s2 = quantize(prune(W2, frac))
    acc = accuracy(q1.float() * s1, q2.float() * s2)   # q * scale = float again
    zeros = ((q1 == 0).sum() + (q2 == 0).sum()).item()
    total = q1.numel() + q2.numel()
    print(f'prune {int(frac*100):2d}% + int8 : acc={acc:.2f}%  zero weights={100*zeros/total:.1f}%')
