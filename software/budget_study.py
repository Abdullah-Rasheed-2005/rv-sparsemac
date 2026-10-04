"""
budget_study.py - Is the run-time budget better than the obvious alternatives?
==============================================================================
Three questions, all answered on the 10,000 MNIST test images with the exact
integer arithmetic of the hardware (budget_lib.py):

  A. Is the result stable?  The budget-aware fine-tuning is repeated with
     several random seeds.
  B. What else gives a GUARANTEED worst case of W cycles for layer 1?
       - "prune more":    remove weights (smallest first) until even an image
                          with every pixel set needs at most W cycles;
       - "fixed subset":  keep only the first columns of the fixed order whose
                          cycles add up to W, drop the other pixels for good.
     Both are then fine-tuned for the same number of epochs as the budget model.
  C. What does the same AVERAGE cost buy when you simply prune more
     (85 / 90 / 95 %), without any guarantee?

It also measures how much the cycle count alone tells about the digit
(timing leak) without a budget, with the budget, and in constant time.

Cost rules. Budget model: the hardware rule, max(nonzero weights, 3) cycles per
nonzero pixel, also for an empty column. Alternatives: the same, but a column
without any weight costs nothing (its pixel would simply never be loaded) -
this favours the alternatives, not the budget.

Run from the repository root (about 15-25 minutes on a CPU):
    python3 software/budget_study.py
Writes docs/budget_study.md. Needs models/pruned_fc1_80.pth and models/budget_order.txt.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets

import budget_lib as L
from budget_experiment_lib import to_int
from model import SimpleNN
from paths import DATA_DIR, MODELS_DIR, ROOT, pruned_model_path

BUDGETS = (1000, 1300, 1700, 2200)
SEEDS = (1, 2, 3)
EPOCHS = 8
TRAIN_LO, TRAIN_HI = 500, 2500
FULL_SHARE = 0.3
NOCAP = 10 ** 9
OUT = ROOT / 'docs' / 'budget_study.md'

train = datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
test = datasets.MNIST(root=str(DATA_DIR), train=False, download=True)
Xtr = train.data.view(-1, 784).numpy().astype(np.int64)
Ytr = train.targets.numpy()
Xte = test.data.view(-1, 784).numpy().astype(np.int64)
Yte = test.targets.numpy()
Xf = torch.from_numpy(Xtr.astype(np.float32)) / 255.0
Yt = torch.from_numpy(Ytr)
RASTER = np.arange(784)

state0 = torch.load(pruned_model_path(80))
net0 = to_int(state0, Xtr[:10000])
order = np.loadtxt(MODELS_DIR / 'budget_order.txt', dtype=np.int64)
cyc0 = L.column_cycles(net0[0])                      # hardware cost of every pixel, model of today
w_abs = state0['fc1.weight'].abs().numpy()           # (64, 784) float magnitudes
mask0 = (w_abs > 0)
vals = np.sort(w_abs[mask0])[::-1]                   # kept magnitudes, largest first


# ------------------------------------------------------------------ helpers
def finetune(state, mask, seed, cut=None):
    """EPOCHS of training from 'state' with the fc1 mask kept.
    cut = (perm, cost_o): budget-aware training, inputs are cut at random budgets."""
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
                perm, cost_o = cut
                xo = xb[:, perm]
                cum = torch.cumsum((xo > 0).float() * cost_o, dim=1)
                budget = torch.empty(xo.size(0), 1).uniform_(TRAIN_LO, TRAIN_HI)
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


def free_cost(W1):
    """Cycles per nonzero pixel for the alternatives: an empty column costs nothing."""
    n = (W1 != 0).sum(axis=0)
    return np.where(n == 0, 0, np.maximum(n, 3)).astype(np.int64)


def mask_cost(m):
    """Worst-case layer-1 cycles of a weight mask (every pixel nonzero)."""
    n = m.sum(axis=0)
    return int(np.where(n == 0, 0, np.maximum(n, 3)).sum())


def measure_plain(state):
    """A model without a run-time budget: accuracy, average / test maximum / guaranteed worst cycles."""
    net = to_int(state, Xtr[:10000])
    cost = free_cost(net[0])
    _, used = L.truncate(Xte, RASTER, cost, NOCAP)
    acc = 100.0 * (L.forward(Xte, *net).argmax(axis=1) == Yte).mean()
    return acc, used.mean(), int(used.max()), int(cost.sum()), int((net[0] != 0).sum())


def measure_budget(net, budget):
    """The budget model at one budget: accuracy, average cycles, per-image cycles."""
    Xk, used = L.truncate(Xte, order, L.column_cycles(net[0]), budget)
    acc = 100.0 * (L.forward(Xk, *net).argmax(axis=1) == Yte).mean()
    return acc, used.mean(), used


def leak(used_fit, y_fit, used_eval, y_eval, width=50):
    """Guess the digit from the cycle count alone (majority digit per bin of 'width' cycles)."""
    table = {}
    for b in np.unique(used_fit // width):
        table[b] = np.bincount(y_fit[used_fit // width == b], minlength=10).argmax()
    default = np.bincount(y_fit).argmax()
    return 100.0 * np.mean([table.get(b, default) == y for b, y in zip(used_eval // width, y_eval)])


lines = []


def say(text=''):
    print(text, flush=True)
    lines.append(text)


# ------------------------------------------------------------------ A. seeds
say('# Budget study')
say()
say('Written by `software/budget_study.py`. 10,000 MNIST test images, exact integer arithmetic of the')
say(f'hardware. Every model below starts from `models/pruned_fc1_80.pth` and is fine-tuned for {EPOCHS} more epochs.')
say('Cycles are layer-1 cycles; layer 2 and the finish sweeps add a fixed amount to every design.')
say()
say('## A. Budget-aware model, several random seeds')
say()
cut = (torch.from_numpy(order.copy()), torch.from_numpy(cyc0[order].astype(np.float32)))
nets = []
for s in SEEDS:
    print(f'... budget-aware fine-tuning, seed {s}', flush=True)
    nets.append(to_int(finetune(state0, mask0, s, cut), Xtr[:10000]))

say('| Layer-1 budget | ' + ' | '.join(f'seed {s}' for s in SEEDS) + ' | mean | spread (max - min) | average cycles used |')
say('|---|' + '---|' * (len(SEEDS) + 3))
ours = {}
for b in (NOCAP,) + BUDGETS:
    r = [measure_budget(n, b) for n in nets]
    a = np.array([x[0] for x in r])
    ours[b] = (a.mean(), np.mean([x[1] for x in r]))
    name = 'none' if b == NOCAP else f'{b:,}'
    say(f'| {name} | ' + ' | '.join(f'{v:.2f}%' for v in a) + f' | {a.mean():.2f}% | {a.max() - a.min():.2f} | {ours[b][1]:,.0f} |')
worst_free = int(np.mean([L.column_cycles(n[0]).sum() for n in nets]))
say()
say(f'Without a budget the slowest possible image needs about {worst_free:,} layer-1 cycles; with a budget it needs the budget.')

# ------------------------------------------------------------------ B. same guaranteed worst case
say()
say('## B. Other ways to guarantee the same worst case')
say()
say('| Guaranteed worst case (cycles) | Budget (mean of seeds) | Prune more (weights left) | Fixed pixel subset (pixels kept) |')
say('|---|---|---|---|')
for W in BUDGETS:
    # prune more: keep the k largest weights such that the worst case fits into W
    lo, hi = 0, len(vals)
    while lo < hi:
        k = (lo + hi + 1) // 2
        if mask_cost(w_abs >= vals[k - 1]) <= W:
            lo = k
        else:
            hi = k - 1
    m_prune = (w_abs >= vals[lo - 1]) if lo > 0 else np.zeros_like(mask0)
    print(f'... W = {W}: prune more ({int(m_prune.sum())} weights)', flush=True)
    p = measure_plain(finetune(state0, m_prune, 1))

    # fixed subset: the first columns of the fixed order whose cycles add up to W
    ncol = int((np.cumsum(cyc0[order]) <= W).sum())
    colmask = np.zeros(784, dtype=bool)
    colmask[order[:ncol]] = True
    print(f'... W = {W}: fixed subset ({ncol} pixels)', flush=True)
    q = measure_plain(finetune(state0, mask0 & colmask[None, :], 1))

    say(f'| {W:,} | {ours[W][0]:.2f}% | {p[0]:.2f}% ({p[4]:,}, worst {p[3]:,}) | {q[0]:.2f}% ({ncol}, worst {q[3]:,}) |')

# ------------------------------------------------------------------ C. same average cost, no guarantee
say()
say('## C. Simply pruning more, without a guarantee')
say()
say('| Model | Accuracy | Average cycles | Slowest test image | Slowest possible image |')
say('|---|---|---|---|---|')
print('... control: 80 % with the same extra training', flush=True)
c = measure_plain(finetune(state0, mask0, 1))
say(f'| 80 % pruned, same extra training, no budget | {c[0]:.2f}% | {c[1]:,.0f} | {c[2]:,} | {c[3]:,} |')
for sp in (85, 90, 95):
    keep = int(round(w_abs.size * (1 - sp / 100)))
    print(f'... {sp} % pruned', flush=True)
    r = measure_plain(finetune(state0, w_abs >= vals[keep - 1], 1))
    say(f'| {sp} % pruned | {r[0]:.2f}% | {r[1]:,.0f} | {r[2]:,} | {r[3]:,} |')
for b in BUDGETS:
    say(f'| budget {b:,} (mean of seeds) | {ours[b][0]:.2f}% | {ours[b][1]:,.0f} | {b:,} | {b:,} |')

# ------------------------------------------------------------------ timing leak
say()
say('## Timing leak')
say()
say('How often the digit is guessed correctly from the layer-1 cycle count alone')
say('(guess learned on 10,000 training images, tested on the test images; first seed).')
say()
net = nets[0]
cyc = L.column_cycles(net[0])
_, tr_free = L.truncate(Xtr[:10000], order, cyc, NOCAP)
_, te_free = L.truncate(Xte, order, cyc, NOCAP)
chance = 100.0 * (Yte == np.bincount(Ytr[:10000]).argmax()).mean()
say('| Mode | Digit guessed correctly |')
say('|---|---|')
say(f'| no budget | {leak(tr_free, Ytr[:10000], te_free, Yte):.1f}% |')
for b in BUDGETS:
    _, tr_b = L.truncate(Xtr[:10000], order, cyc, b)
    _, te_b = L.truncate(Xte, order, cyc, b)
    say(f'| budget {b:,} | {leak(tr_b, Ytr[:10000], te_b, Yte):.1f}% |')
say(f'| constant time (every image the same count) | {chance:.1f}% (always guessing the most common digit) |')

OUT.write_text('\n'.join(lines) + '\n')
print(f'\nwritten: {OUT}')
