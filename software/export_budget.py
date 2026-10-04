"""
export_budget.py - Export the budget-aware model for the bounded-latency hardware
=================================================================================
Input (written by software/budget_pareto.py into build/, then copied into models/):
    models/budget_fc1_<S>.pth    the S % pruned model after budget-aware fine-tuning
    models/budget_order_<S>.txt  its fixed pixel order (most useful pixel first)
(For S = 80 the older file name models/budget_order.txt is also accepted.)

The hardware visits its inputs in the order 0, 1, 2, ... So the order is built
into the exported files: input k of the exported network is pixel order[k] of
the picture. The weight columns AND the test images are stored in that order.
Whoever loads a picture into the accelerator must send pixel order[k] as
input number k (hardware/mem_budget style files already are in that order).

Output folder: build/budget_mem/ (same file names and formats as hardware/mem/,
written by the same code path as the edge model), plus

    budget.hex            the layer-1 cycle budget used for the golden values (4 hex digits)
    golden_logits_b.hex   logits of the 100 test images WITH that budget (8 hex digits)
    golden_pred_b.txt     predicted digit WITH that budget
    order.hex             the pixel order (4 hex digits per line), for firmware / loaders

Run from the repository root:
    python3 software/export_budget.py            (budget 1000, 90 % pruned model)
    python3 software/export_budget.py 1700 80    (another budget, another model)
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

import budget_lib as L
from paths import MODELS_DIR, ROOT

OUT = ROOT / 'build' / 'budget_mem'
NIMG = 100


def write_dir(out, net, order, X, Y, budget, sparsity=80):
    """Write every file the simulations need. Plain numpy, no PyTorch.

    net   : (W1, B1, W2, B2, shift) integer network in NORMAL pixel order
    order : (784,) pixel order, most useful first
    X, Y  : test images (N, 784) in normal pixel order, and their labels
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    W1, B1, W2, B2, shift = net
    W1p = W1[:, order]                       # input k of the hardware = pixel order[k]
    Xp = np.asarray(X, dtype=np.int64)[:, order]
    netp = (W1p, B1, W2, B2, shift)
    natural = np.arange(W1.shape[1])         # the exported network is visited 0, 1, 2, ...
    cyc = L.column_cycles(W1p)

    def w8(name, t):
        (out / name).write_text(''.join(f'{int(v) & 0xFF:02x}\n' for v in np.ravel(t)))

    def w16(name, t):
        (out / name).write_text(''.join(f'{int(v) & 0xFFFF:04x}\n' for v in np.ravel(t)))

    def w32(name, t):
        (out / name).write_text(''.join(f'{int(v) & 0xFFFFFFFF:08x}\n' for v in np.ravel(t)))

    logits = L.forward(Xp, *netp)                                   # no budget
    Xk, used = L.truncate(Xp, natural, cyc, budget)
    logits_b = L.forward(Xk, *netp)                                 # with the budget

    w8('fc1_weights.hex', W1p)
    w32('fc1_bias.hex', B1)
    w8('fc2_weights.hex', W2)
    w32('fc2_bias.hex', B2)
    w8('test_images.hex', Xp)
    (out / 'test_labels.txt').write_text('\n'.join(str(int(v)) for v in Y) + '\n')
    (out / 'golden_logits.txt').write_text(''.join(' '.join(str(int(v)) for v in r) + '\n' for r in logits))
    (out / 'golden_pred.txt').write_text('\n'.join(str(int(v)) for v in logits.argmax(axis=1)) + '\n')
    (out / 'params.txt').write_text(f'sparsity={sparsity}\nshift={shift}\ns1=1\ns2=1\n')

    w16('budget.hex', [budget])
    w32('golden_logits_b.hex', logits_b)
    (out / 'golden_pred_b.txt').write_text('\n'.join(str(int(v)) for v in logits_b.argmax(axis=1)) + '\n')
    w16('order.hex', order)

    # sparse + column (CSC) files, golden_logits.hex, hidden_shift.hex: the normal exporter
    subprocess.run([sys.executable, str(Path(__file__).with_name('export_sparse.py')), str(out)], check=True)
    return used


if __name__ == '__main__':
    import torch
    from torchvision import datasets

    from budget_experiment_lib import to_int
    from paths import DATA_DIR

    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    sparsity = int(sys.argv[2]) if len(sys.argv) > 2 else 90
    MODEL = MODELS_DIR / f'budget_fc1_{sparsity}.pth'
    ORDER = MODELS_DIR / f'budget_order_{sparsity}.txt'
    if sparsity == 80 and not ORDER.exists():
        ORDER = MODELS_DIR / 'budget_order.txt'
    if not MODEL.exists() or not ORDER.exists():
        sys.exit(f'missing {MODEL.name} / {ORDER.name} in models/: run software/budget_pareto.py, '
                 f'then copy build/budget_fc1_{sparsity}.pth and build/budget_order_{sparsity}.txt into models/')

    train = datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
    test = datasets.MNIST(root=str(DATA_DIR), train=False, download=True)
    Xtr = train.data.view(-1, 784).numpy().astype(np.int64)
    Xte = test.data.view(-1, 784).numpy().astype(np.int64)
    Yte = test.targets.numpy()

    net = to_int(torch.load(MODEL), Xtr[:10000])
    order = np.loadtxt(ORDER, dtype=np.int64)
    if sorted(order.tolist()) != list(range(784)):
        sys.exit('budget_order.txt is not a permutation of 0..783')

    # the numbers behind the hardware claim, on all 10,000 test images, in CLOCK-CYCLE cost
    cyc = L.column_cycles(net[0])
    full, _ = L.accuracy_at(Xte, Yte, net, order, 10 ** 9, L.column_cycles)
    acc, used = L.accuracy_at(Xte, Yte, net, order, budget, L.column_cycles)
    _, free = L.truncate(Xte, order, cyc, 10 ** 9)
    print(f'model: {MODEL.name} ({sparsity} % of fc1 pruned), layer-1 budget = {budget} cycles')
    print(f'accuracy, no budget   : {full:.2f}%  (layer-1 cycles: average {free.mean():.0f}, maximum {free.max()}, '
          f'worst possible image {cyc.sum()})')
    print(f'accuracy, with budget : {acc:.2f}%  (layer-1 cycles: average {used:.0f}, maximum {budget})')

    used100 = write_dir(OUT, net, order, Xte[:NIMG], Yte[:NIMG], budget, sparsity)
    print(f'golden values for {NIMG} images written, {int((used100 < L.truncate(Xte[:NIMG], order, cyc, 10 ** 9)[1]).sum())} '
          f'of them are cut by the budget')
    print(f'Budget model written to {OUT}')
