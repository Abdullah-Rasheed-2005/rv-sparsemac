"""
budget_experiment_lib.py - Float weights -> integer network (shared helper)
===========================================================================
The same conversion as software/export_int8.py, as a function, so that
budget_experiment.py and export_budget.py produce identical integer networks.
"""

import numpy as np
import torch


def to_int(state, calib_images):
    """state: PyTorch state dict (fc1/fc2 weight + bias).
    calib_images: (N, 784) raw training pixels 0..255, used to choose the hidden shift.
    Returns (W1, B1, W2, B2, shift) as integer numpy arrays."""
    def quantize(w):
        s = w.abs().max().item() / 127
        return torch.clamp(torch.round(w / s), -127, 127).long().numpy().astype(np.int64), s

    q1, s1 = quantize(state['fc1.weight'])
    q2, s2 = quantize(state['fc2.weight'])
    sx = 1.0 / 255.0
    b1 = torch.round(state['fc1.bias'] / (sx * s1)).long().numpy().astype(np.int64)
    max_act = int(np.maximum(np.asarray(calib_images, dtype=np.int64) @ q1.T + b1, 0).max())
    shift = 0
    while (max_act >> shift) > 127:
        shift += 1
    b2 = torch.round(state['fc2.bias'] / (sx * s1 * (2 ** shift) * s2)).long().numpy().astype(np.int64)
    return q1, b1, q2, b2, shift
