"""
budget_lib.py - Integer inference with a hard MAC budget on layer 1
===================================================================
The zero-skipping accelerator (hardware/rtl/sparse_mlp_zs.v) needs one clock
per (nonzero pixel, nonzero weight) pair. Its run time therefore depends on the
image. This file models a BUDGETED version of layer 1:

    visit the pixels in a given ORDER
    a nonzero pixel j costs nnz[j] MACs (the nonzero weights of its column)
    a column is processed only if it still fits into the budget;
    the first column that does not fit ends layer 1 (the rest is dropped)

Layer 2 is small (at most 640 MACs) and is always run completely.

Everything here is plain numpy and exact integer arithmetic, so the numbers are
the ones a hardware implementation of the same rule would produce.
"""

import numpy as np


def column_cost(W1):
    """MACs needed for each input pixel when it is nonzero (nonzero weights per column)."""
    return (W1 != 0).sum(axis=0).astype(np.int64)


def column_cycles(W1):
    """Clock cycles the accelerator needs for each input pixel when it is nonzero.
    A column of n nonzero weights takes n cycles, but never less than 3, because
    the front end of sparse_mlp_zs.v needs 3 cycles to prepare a column. This is
    the cost the hardware budget counts, so a budget in these units bounds the
    run time of layer 1 in clock cycles (plus a few cycles of start-up)."""
    return np.maximum(column_cost(W1), 3)


def order_raster(W1):
    """The order used today: pixel 0, 1, 2, ..., 783."""
    return np.arange(W1.shape[1])


def order_l1(W1):
    """Fixed order: columns with the largest sum of |weight| first."""
    return np.argsort(-np.abs(W1).sum(axis=0), kind='stable')


def order_l1_per_mac(W1):
    """Fixed order: columns with the largest |weight| sum PER MAC first."""
    l1 = np.abs(W1).sum(axis=0)
    return np.argsort(-(l1 / np.maximum(column_cost(W1), 1)), kind='stable')


def truncate(X, order, cost, budget):
    """Apply the budget rule to a batch of images.

    X      : (N, 784) integer pixels
    order  : (784,) the SAME order for every image, or (N, 784) one order per image
    cost   : (784,) MACs per nonzero pixel
    budget : one number, or (N,) one budget per image
    Returns (X_kept, used): the images with the dropped pixels set to 0, and the
    MACs really used per image.
    """
    X = np.asarray(X, dtype=np.int64)
    N = X.shape[0]
    order = np.broadcast_to(order, (N, X.shape[1]))
    Xo = np.take_along_axis(X, order, axis=1)
    co = cost[order] * (Xo != 0)                    # cost of each visited pixel (0 if the pixel is zero)
    cum = np.cumsum(co, axis=1)
    b = np.broadcast_to(np.asarray(budget, dtype=np.int64).reshape(-1, 1), (N, 1))
    keep = cum <= b                                 # cum never decreases, so this is "stop at the first misfit"
    used = (co * keep).sum(axis=1)
    Xk = np.zeros_like(X)
    np.put_along_axis(Xk, order, Xo * keep, axis=1)
    return Xk, used


def forward(X, W1, B1, W2, B2, shift):
    """The integer network of export_int8.py. Returns the 10 logits per image."""
    acc1 = np.asarray(X, dtype=np.int64) @ W1.T + B1
    h = np.minimum(np.maximum(acc1, 0) >> shift, 127)
    return h @ W2.T + B2


def dynamic_order(X, W1):
    """Per-image order: largest pixel * column |weight| sum first.
    This needs sorting inside the hardware, so it is only an upper reference."""
    l1 = np.abs(W1).sum(axis=0)
    return np.argsort(-(np.asarray(X, dtype=np.int64) * l1), axis=1, kind='stable')


def accuracy_at(X, Y, net, order, budget, cost_fn=column_cost):
    """Accuracy (percent) and average cost used with a hard budget.
    cost_fn = column_cost counts MACs, cost_fn = column_cycles counts clock cycles."""
    W1, B1, W2, B2, shift = net
    Xk, used = truncate(X, order, cost_fn(W1), budget)
    pred = forward(Xk, W1, B1, W2, B2, shift).argmax(axis=1)
    return 100.0 * (pred == Y).mean(), used.mean()
