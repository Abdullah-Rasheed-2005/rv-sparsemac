"""
inspect_weights.py - Look at the weights inside the baseline model
==================================================================
Prints the shape, smallest value and largest value of every tensor.
For weight tensors it also prints how many weights are close to zero.
This helps us choose the quantization scale and the pruning level.

Run from the repository root:   python3 software/inspect_weights.py
"""

import torch

from paths import BASELINE_MODEL

state = torch.load(BASELINE_MODEL)

for name, w in state.items():
    print(f'{name}: shape={tuple(w.shape)} min={w.min().item():.4f} max={w.max().item():.4f}')
    if 'weight' in name:
        # What percent of the weights are smaller than each threshold?
        for t in [0.001, 0.01, 0.05]:
            pct = (w.abs() < t).sum().item() / w.numel() * 100
            print(f'   |w| < {t}: {pct:.1f}% of weights')
