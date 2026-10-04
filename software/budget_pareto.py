"""
budget_pareto.py - Pruning more AND a run-time budget together
==============================================================
budget_study.py showed two things: a budget gives a much better guaranteed
worst case than pruning, but pruning gives a lower average cost. They are not
alternatives. This script combines them:

    for each sparsity (80 / 85 / 90 / 95 %):
        1. prune fc1 to that sparsity, fine-tune (plain)
        2. choose the fixed pixel order from the pruned weights
        3. budget-aware fine-tuning (inputs cut at random budgets)
        4. accuracy and cycles for several budgets, all 10,000 test images,
           exact integer arithmetic and the cycle cost of the hardware

Run from the repository root (about 10-15 minutes on a CPU):
    python3 software/budget_pareto.py
Writes docs/budget_pareto.md and, for every sparsity, the model and its order into build/.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets

import budget_lib as L
from budget_experiment_lib import to_int
from model import SimpleNN
from paths import DATA_DIR, ROOT, pruned_model_path

SPARSITIES = (80, 85, 90, 95)
BUDGETS = (400, 600, 800, 1000, 1300, 1700, 2200)
EPOCHS = 8
FULL_SHARE = 0.3
NOCAP = 10 ** 9
OUT = ROOT / 'docs' / 'budget_pareto.md'

train = datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
test = datasets.MNIST(root=str(DATA_DIR), train=False, download=True)
Xtr = train.data.view(-1, 784).numpy().astype(np.int64)
Ytr = train.targets.numpy()
Xte = test.data.view(-1, 784).numpy().astype(np.int64)
Yte = test.targets.numpy()
Xf = torch.from_numpy(Xtr.astype(np.float32)) / 255.0
Yt = torch.from_numpy(Ytr)

state0 = torch.load(pruned_model_path(80))
w_abs = state0['fc1.weight'].abs().numpy()
vals = np.sort(w_abs[w_abs > 0])[::-1]               # kept magnitudes, largest first


def finetune(state, mask, seed, cut=None):
    """EPOCHS of training from 'state' with the fc1 mask kept.
    cut = (perm, cost_o, lo, hi): inputs are cut at random budgets between lo and hi."""
    torch.manual_seed(seed)
    model = SimpleNN()
    model.load_state_dict(state)
    mask = torch.from_numpy(np.asarray(mask, dtype=np.float32))
    model.fc1.weight.data *= mask
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
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
            model.fc1.weight.data *= mask
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


lines = []


def say(text=''):
    print(text, flush=True)
    lines.append(text)


say('# Pruning and the run-time budget together')
say()
say('Written by `software/budget_pareto.py`. 10,000 MNIST test images, exact integer arithmetic and')
say('cycle cost of the hardware (layer-1 cycles). One random seed per sparsity.')
say(f'Each model: prune `models/pruned_fc1_80.pth` further, {EPOCHS} plain epochs, then {EPOCHS} budget-aware epochs.')
say()
say('Accuracy (%) for each layer-1 budget. "-" = the budget is above the slowest test image, so it never cuts.')
say()
say('| fc1 pruned | weights left | no budget | ' + ' | '.join(f'{b:,}' for b in BUDGETS) + ' |')
say('|---|---|---|' + '---|' * len(BUDGETS))

rows = []
for sp in SPARSITIES:
    keep = int(round(w_abs.size * (1 - sp / 100)))
    mask = (w_abs >= vals[keep - 1]) if sp > 80 else (w_abs > 0)

    print(f'... {sp} %: plain fine-tuning', flush=True)
    plain = finetune(state0, mask, 1)
    net_p = to_int(plain, Xtr[:10000])
    order = L.order_l1_per_mac(net_p[0])                       # fixed order from the pruned weights
    cyc_p = L.column_cycles(net_p[0])
    _, used_tr = L.truncate(Xtr[:10000], order, cyc_p, NOCAP)  # cost of this model on training images
    lo, hi = 0.3 * used_tr.mean(), 1.4 * used_tr.mean()

    print(f'... {sp} %: budget-aware fine-tuning (budgets {lo:.0f} .. {hi:.0f})', flush=True)
    cut = (torch.from_numpy(order.copy()), torch.from_numpy(cyc_p[order].astype(np.float32)), lo, hi)
    state = finetune(plain, mask, 1, cut)
    net = to_int(state, Xtr[:10000])
    cyc = L.column_cycles(net[0])

    _, free = L.truncate(Xte, order, cyc, NOCAP)
    acc_free = 100.0 * (L.forward(Xte, *net).argmax(axis=1) == Yte).mean()
    acc_plain = 100.0 * (L.forward(Xte, *net_p).argmax(axis=1) == Yte).mean()
    cells, detail = [], []
    for b in BUDGETS:
        if b >= free.max():
            cells.append('-')
            continue
        a, u = L.accuracy_at(Xte, Yte, net, order, b, L.column_cycles)
        cells.append(f'{a:.2f}')
        detail.append((b, a, u))
    say(f'| {sp} % | {int((net[0] != 0).sum()):,} | {acc_free:.2f} | ' + ' | '.join(cells) + ' |')
    rows.append((sp, acc_plain, acc_free, free.mean(), int(free.max()), int(cyc.sum()), detail))

    out = ROOT / 'build'
    out.mkdir(exist_ok=True)
    torch.save(state, out / f'budget_fc1_{sp}.pth')
    np.savetxt(out / f'budget_order_{sp}.txt', order, fmt='%d')

say()
say('## Cost without a budget')
say()
say('| fc1 pruned | accuracy before / after budget-aware training | average cycles | slowest test image | slowest possible image |')
say('|---|---|---|---|---|')
for sp, ap, af, mean, mx, worst, _ in rows:
    say(f'| {sp} % | {ap:.2f}% / {af:.2f}% | {mean:,.0f} | {mx:,} | {worst:,} |')

say()
say('## Every measured point (accuracy, average cycles, guaranteed worst case)')
say()
say('| fc1 pruned | budget = guaranteed worst case | accuracy | average cycles used |')
say('|---|---|---|---|')
for sp, _, af, mean, _, worst, detail in rows:
    say(f'| {sp} % | none ({worst:,}) | {af:.2f}% | {mean:,.0f} |')
    for b, a, u in detail:
        say(f'| {sp} % | {b:,} | {a:.2f}% | {u:,.0f} |')

OUT.write_text('\n'.join(lines) + '\n')
print(f'\nwritten: {OUT}')
