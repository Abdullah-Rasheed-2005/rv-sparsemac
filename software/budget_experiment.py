"""
budget_experiment.py - Does a hard MAC budget on layer 1 keep the accuracy?
===========================================================================
Go / no-go experiment for a "bounded latency" version of the accelerator.
Nothing in the hardware is changed here; this is the software study only.

It prints four things, all on the full 10,000 MNIST test images and with the
exact integer arithmetic of the hardware (see budget_lib.py):

  1. How many layer-1 MACs the images need today (average, 95 %, maximum, and
     the worst possible image).
  2. How much the MAC count alone tells about the digit (timing leak).
  3. Accuracy when layer 1 is stopped at a hard budget, for several pixel orders.
  4. The same after "budget-aware" fine-tuning: the pruned model is trained a
     little more while its inputs are cut at random budgets, so it learns to
     work with a cut image. The pruning mask is kept.

Run from the repository root (needs models/pruned_fc1_80.pth):
    python3 software/budget_experiment.py
Takes a few minutes on a CPU. The trained model is written to build/ only.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets

import budget_lib as L
from model import SimpleNN
from paths import DATA_DIR, ROOT, pruned_model_path

torch.manual_seed(0)
np.random.seed(0)

SPARSITY = 80
BUDGETS = (700, 1000, 1300, 1700, 2200, 3000)
TRAIN_LO, TRAIN_HI = 500, 2500      # budgets drawn while fine-tuning
FULL_SHARE = 0.3                    # share of training images that are NOT cut
EPOCHS = 8

train = datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
test = datasets.MNIST(root=str(DATA_DIR), train=False, download=True)
Xtr = train.data.view(-1, 784).numpy().astype(np.int64)     # raw pixels 0..255
Ytr = train.targets.numpy()
Xte = test.data.view(-1, 784).numpy().astype(np.int64)
Yte = test.targets.numpy()


def to_int(state):
    """Float weights -> the integer network, exactly as software/export_int8.py does it."""
    def quantize(w):
        s = w.abs().max().item() / 127
        return torch.clamp(torch.round(w / s), -127, 127).long().numpy().astype(np.int64), s

    q1, s1 = quantize(state['fc1.weight'])
    q2, s2 = quantize(state['fc2.weight'])
    sx = 1.0 / 255.0
    b1 = torch.round(state['fc1.bias'] / (sx * s1)).long().numpy().astype(np.int64)
    max_act = int(np.maximum(Xtr[:10000] @ q1.T + b1, 0).max())      # calibration on training images
    shift = 0
    while (max_act >> shift) > 127:
        shift += 1
    b2 = torch.round(state['fc2.bias'] / (sx * s1 * (2 ** shift) * s2)).long().numpy().astype(np.int64)
    return q1, b1, q2, b2, shift


def table_row(name, net, order_fn):
    """One line: accuracy at every budget. order_fn(X) returns the order for these images."""
    cells = ''.join(f'{L.accuracy_at(Xte, Yte, net, order_fn(Xte), b)[0]:8.2f}' for b in BUDGETS)
    print(f'  {name:44s}{cells}')


def header():
    print(f'  {"pixel order":44s}' + ''.join(f'{b:8d}' for b in BUDGETS))


# ============================================================ the model of today
state0 = torch.load(pruned_model_path(SPARSITY))
net0 = to_int(state0)
W1 = net0[0]
cost = L.column_cost(W1)
NOCAP = 10 ** 9

full_acc, _ = L.accuracy_at(Xte, Yte, net0, L.order_raster(W1), NOCAP)
_, used_te = L.truncate(Xte, L.order_raster(W1), cost, NOCAP)
_, used_tr = L.truncate(Xtr[:10000], L.order_raster(W1), cost, NOCAP)

print('=== 1. Layer-1 MACs per image today (no budget) ===')
print(f'  accuracy (integer, 10,000 images) : {full_acc:.2f}%')
print(f'  average / 95% / maximum           : {used_te.mean():.0f} / {np.percentile(used_te, 95):.0f} / {used_te.max()}')
print(f'  worst possible image (all pixels) : {cost.sum()}   ({cost.sum() / used_te.mean():.1f}x the average)')

print('\n=== 2. Timing leak: what the MAC count alone says about the digit ===')
print('  average MACs per digit: ' + '  '.join(f'{d}:{used_te[Yte == d].mean():.0f}' for d in range(10)))
BIN = 50                                                    # guess the digit from the MAC count only
guess = {}
for b in np.unique(used_tr // BIN):
    guess[b] = np.bincount(Ytr[:10000][used_tr // BIN == b], minlength=10).argmax()
default = np.bincount(Ytr[:10000]).argmax()
leak = np.mean([guess.get(b, default) == y for b, y in zip(used_te // BIN, Yte)]) * 100
print(f'  digit guessed from the MAC count only: {leak:.1f}% correct (pure chance would be about 10%)')

print('\n=== 3. Accuracy (%) with a hard layer-1 budget, model of today ===')
header()
table_row('raster 0..783 (today)', net0, lambda X: L.order_raster(W1))
table_row('fixed: largest column |w| sum first', net0, lambda X: L.order_l1(W1))
table_row('fixed: largest |w| sum per MAC first', net0, lambda X: L.order_l1_per_mac(W1))
table_row('per image: pixel * |w| sum (needs sorting)', net0, lambda X: L.dynamic_order(X, W1))

# choose the fixed order on TRAINING images (never on the test set)
Xval, Yval = Xtr[10000:20000], Ytr[10000:20000]
cands = {'largest column |w| sum first': L.order_l1(W1),
         'largest |w| sum per MAC first': L.order_l1_per_mac(W1)}
best_name = max(cands, key=lambda k: L.accuracy_at(Xval, Yval, net0, cands[k], 1300)[0])
order = cands[best_name]
print(f'\n  fixed order chosen on training images: "{best_name}"')

# ============================================================ budget-aware fine-tuning
print(f'\n=== 4. Budget-aware fine-tuning ({EPOCHS} epochs, mask kept, order fixed) ===')
model = SimpleNN()
model.load_state_dict(state0)
mask = (model.fc1.weight.data != 0).float()
perm = torch.from_numpy(order.copy())
cost_o = torch.from_numpy(cost[order].astype(np.float32))    # cost of each pixel, in visiting order
Xf = torch.from_numpy(Xtr.astype(np.float32)) / 255.0
Yt = torch.from_numpy(Ytr)

opt = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    model.train()
    shuffle = torch.randperm(Xf.size(0))
    total = 0.0
    for i in range(0, Xf.size(0), 128):
        idx = shuffle[i:i + 128]
        xo = Xf[idx][:, perm]                                # pixels in visiting order
        cum = torch.cumsum((xo > 0).float() * cost_o, dim=1)
        budget = torch.empty(xo.size(0), 1).uniform_(TRAIN_LO, TRAIN_HI)
        budget[torch.rand(xo.size(0), 1) < FULL_SHARE] = float(NOCAP)
        xk = torch.zeros_like(xo)
        xk[:, perm] = xo * (cum <= budget).float()           # cut image, back in normal pixel order
        opt.zero_grad()
        loss = loss_fn(model(xk), Yt[idx])
        loss.backward()
        opt.step()
        model.fc1.weight.data *= mask                        # pruned weights stay zero
        total += loss.item()
    print(f'  epoch {epoch + 1}/{EPOCHS}  loss {total / (Xf.size(0) / 128):.4f}')

state1 = {k: v.detach().clone() for k, v in model.state_dict().items()}
net1 = to_int(state1)
full1, _ = L.accuracy_at(Xte, Yte, net1, order, NOCAP)
print(f'\n  accuracy without a budget: before {full_acc:.2f}%, after {full1:.2f}%')
print(f'  nonzero fc1 weights      : before {int((W1 != 0).sum())}, after {int((net1[0] != 0).sum())}')
header()
table_row('before training, fixed order', net0, lambda X: order)
table_row('AFTER budget-aware training, same fixed order', net1, lambda X: order)
table_row('before training, per-image order (reference)', net0, lambda X: L.dynamic_order(X, W1))

out = ROOT / 'build'
out.mkdir(exist_ok=True)
torch.save(state1, out / f'budget_fc1_{SPARSITY}.pth')
np.savetxt(out / 'budget_order.txt', order, fmt='%d')

print('\n=== Verdict ===')
ref = max(full_acc, full1)
for b in BUDGETS:
    a, u = L.accuracy_at(Xte, Yte, net1, order, b)
    print(f'  budget {b:5d}: accuracy {a:.2f}% ({a - ref:+.2f} vs no budget), average MACs used {u:.0f}, '
          f'worst case cut from {cost.sum()} to {b}')
print('  GO if some budget near the average of today loses less than about 0.5%.')
