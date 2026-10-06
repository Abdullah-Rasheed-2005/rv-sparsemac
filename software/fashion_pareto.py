"""
fashion_pareto.py - The same study on a second dataset: Fashion-MNIST
=====================================================================
MNIST is an easy case for a zero-skipping accelerator: about 80 % of the pixels
are zero. Fashion-MNIST has the same image size (so the hardware does not
change) but only about half of its pixels are zero and it is a harder task.
This script repeats the whole software pipeline on it:

  1. train the dense 784 -> 64 -> 10 network (fixed random seed)
  2. for each sparsity (80 / 90 / 95 %): prune fc1, fine-tune, choose the fixed
     pixel order, budget-aware fine-tuning, then accuracy for several budgets
  3. for the 90 % model: two other ways to guarantee the same worst case
     ("prune more" and "fixed pixel subset"), trained for the same number of epochs

Everything is evaluated on the 10,000 test images with the exact integer
arithmetic and the cycle cost of the hardware (budget_lib.py).

Run from the repository root (about 20-30 minutes on a CPU; downloads the
dataset on the first run):
    python3 software/fashion_pareto.py
Writes docs/fashion_pareto.md and the models + orders into build/.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets

import budget_lib as L
from budget_experiment_lib import to_int
from model import SimpleNN
from paths import DATA_DIR, ROOT

SPARSITIES = (80, 90, 95)
FRACTIONS = (0.4, 0.6, 0.8, 1.0, 1.25)   # budgets as a share of the model's own average cost
ALT_SPARSITY = 90                        # model used for the "other ways" comparison
ALT_FRACTIONS = (0.6, 1.0)
DENSE_EPOCHS = 10
EPOCHS = 8
FULL_SHARE = 0.3
NOCAP = 10 ** 9
OUT = ROOT / 'docs' / 'fashion_pareto.md'

train = datasets.FashionMNIST(root=str(DATA_DIR), train=True, download=True)
test = datasets.FashionMNIST(root=str(DATA_DIR), train=False, download=True)
Xtr = train.data.view(-1, 784).numpy().astype(np.int64)
Ytr = train.targets.numpy()
Xte = test.data.view(-1, 784).numpy().astype(np.int64)
Yte = test.targets.numpy()
Xf = torch.from_numpy(Xtr.astype(np.float32)) / 255.0
Yt = torch.from_numpy(Ytr)
RASTER = np.arange(784)


def fit(model, mask, seed, epochs, lr, cut=None):
    """Train 'model' in place. mask (or None) keeps pruned fc1 weights at zero.
    cut = (perm, cost_o, lo, hi): inputs are cut at random budgets between lo and hi."""
    torch.manual_seed(seed)
    if mask is not None:
        mask = torch.from_numpy(np.asarray(mask, dtype=np.float32))
        model.fc1.weight.data *= mask
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        shuffle = torch.randperm(Xf.size(0))
        for i in range(0, Xf.size(0), 128):
            idx = shuffle[i:i + 128]
            xb = Xf[idx]
            if cut is not None:
                perm, cost_o, lo, hi = cut
                xo = xb[:, perm]
                cum = torch.cumsum((xo > 0).float() * cost_o, dim=1)
                budget = torch.empty(xo.size(0), 1).uniform_(lo, hi)
                budget[torch.rand(xo.size(0), 1) < FULL_SHARE] = float(NOCAP)
                xk = torch.zeros_like(xo)
                xk[:, perm] = xo * (cum <= budget).float()
                xb = xk
            opt.zero_grad()
            loss = loss_fn(model(xb), Yt[idx])
            loss.backward()
            opt.step()
            if mask is not None:
                model.fc1.weight.data *= mask
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def finetune(state, mask, epochs, cut=None):
    model = SimpleNN()
    model.load_state_dict(state)
    return fit(model, mask, 1, epochs, 5e-4, cut)


def int_acc(net, X=None):
    X = Xte if X is None else X
    return 100.0 * (L.forward(X, *net).argmax(axis=1) == Yte).mean()


def free_cost(W1):
    """Cycles per nonzero pixel when an empty column costs nothing (used for the alternatives)."""
    n = (W1 != 0).sum(axis=0)
    return np.where(n == 0, 0, np.maximum(n, 3)).astype(np.int64)


def mask_cost(m):
    n = m.sum(axis=0)
    return int(np.where(n == 0, 0, np.maximum(n, 3)).sum())


lines = []


def say(text=''):
    print(text, flush=True)
    lines.append(text)


# ------------------------------------------------------------------ 1. dense network
print('... training the dense network', flush=True)
torch.manual_seed(0)                                 # fixed start weights
dense = fit(SimpleNN(), None, 0, DENSE_EPOCHS, 1e-3)
net_d = to_int(dense, Xtr[:10000])
w_abs = dense['fc1.weight'].abs().numpy()
vals = np.sort(w_abs.reshape(-1))[::-1]              # magnitudes, largest first

say('# The same study on Fashion-MNIST')
say()
say('Written by `software/fashion_pareto.py`. Same network size as for MNIST (784 -> 64 -> 10), so the')
say('hardware is unchanged. 10,000 test images, exact integer arithmetic and cycle cost of the hardware')
say('(layer-1 cycles). One random seed.')
say()
say(f'- zero pixels in the test images: {100.0 * (Xte == 0).mean():.1f}% (MNIST: about 81%)')
say(f'- dense network after {DENSE_EPOCHS} epochs, integer arithmetic: {int_acc(net_d):.2f}%')

# ------------------------------------------------------------------ 2. pruning + budget
rows, kept = [], {}
for sp in SPARSITIES:
    keep = int(round(w_abs.size * (1 - sp / 100)))
    mask = w_abs >= vals[keep - 1]

    print(f'... {sp} %: plain fine-tuning', flush=True)
    plain = finetune(dense, mask, EPOCHS)
    net_p = to_int(plain, Xtr[:10000])
    order = L.order_l1_per_mac(net_p[0])
    cyc_p = L.column_cycles(net_p[0])
    _, used_tr = L.truncate(Xtr[:10000], order, cyc_p, NOCAP)
    lo, hi = 0.3 * used_tr.mean(), 1.4 * used_tr.mean()

    print(f'... {sp} %: budget-aware fine-tuning (budgets {lo:.0f} .. {hi:.0f})', flush=True)
    cut = (torch.from_numpy(order.copy()), torch.from_numpy(cyc_p[order].astype(np.float32)), lo, hi)
    state = finetune(plain, mask, EPOCHS, cut)
    net = to_int(state, Xtr[:10000])

    print(f'... {sp} %: plain fine-tuning for the same total number of epochs (reference)', flush=True)
    ref = to_int(finetune(plain, mask, EPOCHS), Xtr[:10000])

    cyc = L.column_cycles(net[0])
    _, free = L.truncate(Xte, order, cyc, NOCAP)
    detail = []
    for fr in FRACTIONS:
        b = int(round(fr * free.mean() / 50.0)) * 50
        a, u = L.accuracy_at(Xte, Yte, net, order, b, L.column_cycles)
        detail.append((fr, b, a, u))
    rows.append((sp, int((net[0] != 0).sum()), int_acc(ref), int_acc(net), free.mean(), int(free.max()), int(cyc.sum()), detail))
    kept[sp] = (plain, mask, net, order, free.mean())

    out = ROOT / 'build'
    out.mkdir(exist_ok=True)
    torch.save(state, out / f'fashion_fc1_{sp}.pth')
    np.savetxt(out / f'fashion_order_{sp}.txt', order, fmt='%d')

say()
say('## Cost without a budget')
say()
say(f'"plain" = pruned and fine-tuned for {2 * EPOCHS} epochs without any budget; "budget-aware" = {EPOCHS} plain epochs,')
say(f'then {EPOCHS} epochs with inputs cut at random budgets. Both evaluated here WITHOUT a budget.')
say()
say('| fc1 pruned | weights left | accuracy, plain | accuracy, budget-aware | average cycles | slowest test image | slowest possible image |')
say('|---|---|---|---|---|---|---|')
for sp, nz, a_ref, a_free, mean, mx, worst, _ in rows:
    say(f'| {sp} % | {nz:,} | {a_ref:.2f}% | {a_free:.2f}% | {mean:,.0f} | {mx:,} | {worst:,} |')

say()
say('## Accuracy with a hard layer-1 budget')
say()
say('The budget is given as a share of the average cost of the same model without a budget.')
say()
say('| fc1 pruned | budget = guaranteed worst case | share of the average cost | accuracy | loss against no budget | average cycles used |')
say('|---|---|---|---|---|---|')
for sp, _, _, a_free, mean, _, worst, detail in rows:
    say(f'| {sp} % | none ({worst:,}) | - | {a_free:.2f}% | - | {mean:,.0f} |')
    for fr, b, a, u in detail:
        say(f'| {sp} % | {b:,} | {fr:.2f} | {a:.2f}% | {a - a_free:+.2f} | {u:,.0f} |')

# ------------------------------------------------------------------ 3. other ways to guarantee the same worst case
plain, mask, net, order, mean = kept[ALT_SPARSITY]
net_p = to_int(plain, Xtr[:10000])
cyc_p = L.column_cycles(net_p[0])
wp = plain['fc1.weight'].abs().numpy()
vp = np.sort(wp[wp > 0])[::-1]
say()
say(f'## Other ways to guarantee the same worst case ({ALT_SPARSITY} % model)')
say()
say(f'Each alternative starts from the same {ALT_SPARSITY} % pruned model and is fine-tuned for {EPOCHS} more epochs,')
say('the same as the budget-aware model. A column without any weight costs the alternatives nothing.')
say()
say('| Guaranteed worst case (cycles) | Budget | Prune more (weights left) | Fixed pixel subset (pixels kept) |')
say('|---|---|---|---|')
for fr in ALT_FRACTIONS:
    W = int(round(fr * mean / 50.0)) * 50
    a_b, _ = L.accuracy_at(Xte, Yte, net, order, W, L.column_cycles)

    lo_k, hi_k = 0, len(vp)
    while lo_k < hi_k:
        k = (lo_k + hi_k + 1) // 2
        if mask_cost(wp >= vp[k - 1]) <= W:
            lo_k = k
        else:
            hi_k = k - 1
    m_prune = (wp >= vp[lo_k - 1]) if lo_k > 0 else np.zeros_like(mask)
    print(f'... W = {W}: prune more ({int(m_prune.sum())} weights)', flush=True)
    n1 = to_int(finetune(plain, m_prune, EPOCHS), Xtr[:10000])

    ncol = int((np.cumsum(cyc_p[order]) <= W).sum())
    colmask = np.zeros(784, dtype=bool)
    colmask[order[:ncol]] = True
    print(f'... W = {W}: fixed subset ({ncol} pixels)', flush=True)
    n2 = to_int(finetune(plain, mask & colmask[None, :], EPOCHS), Xtr[:10000])

    say(f'| {W:,} | {a_b:.2f}% | {int_acc(n1):.2f}% ({int((n1[0] != 0).sum()):,}, worst {int(free_cost(n1[0]).sum()):,}) '
        f'| {int_acc(n2):.2f}% ({ncol}, worst {int(free_cost(n2[0]).sum()):,}) |')

OUT.write_text('\n'.join(lines) + '\n')
print(f'\nwritten: {OUT}')
