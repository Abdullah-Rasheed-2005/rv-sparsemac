"""
finetune_pruned.py - Prune fc1, then train again to win the accuracy back
=========================================================================
Steps for each sparsity level (70%, 80%, 90%):
  A. Build a mask (1 = keep the weight, 0 = remove it) and apply it.
  B. Train for a few epochs. After every optimizer step the mask is
     applied again, so the removed weights stay exactly zero.
  C. Measure the accuracy (float, and also with int8 weights).
  D. Save the model as models/pruned_fc1_<percent>.pth

Run from the repository root:   python3 software/finetune_pruned.py
"""

import copy
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from model import SimpleNN
from paths import DATA_DIR, MODELS_DIR, BASELINE_MODEL, pruned_model_path

torch.manual_seed(0)   # same random numbers every run

# Load all data into tensors (this is fast)
tf = transforms.ToTensor()
train = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=tf)
test = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=tf)
Xtr = train.data.float().view(-1, 784) / 255.0
Ytr = train.targets
Xte = test.data.float().view(-1, 784) / 255.0
Yte = test.targets


def test_acc(model):
    """Test accuracy in percent."""
    model.eval()
    with torch.no_grad():
        return (model(Xte).argmax(dim=1) == Yte).float().mean().item() * 100


def int8_acc(model):
    """Quantize the weights to int8, then measure the accuracy."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        for layer in (m.fc1, m.fc2):
            w = layer.weight
            s = w.abs().max() / 127
            layer.weight.copy_(torch.clamp(torch.round(w / s), -127, 127) * s)
    return test_acc(m)


def make_mask(w, frac):
    """Mask is 1 for weights we keep, 0 for the smallest 'frac' of weights."""
    k = int(w.numel() * frac)
    thr = w.abs().flatten().kthvalue(k).values
    return (w.abs() > thr).float()


MODELS_DIR.mkdir(exist_ok=True)

for frac in [0.7, 0.8, 0.9]:
    pct = int(frac * 100)
    model = SimpleNN()
    model.load_state_dict(torch.load(BASELINE_MODEL))

    # Step A: build the mask and prune fc1
    mask = make_mask(model.fc1.weight.data, frac)
    model.fc1.weight.data *= mask
    before = test_acc(model)

    # Step B: fine-tune (the mask is applied again after every step)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(5):
        model.train()
        perm = torch.randperm(Xtr.size(0))   # shuffle the training images
        for i in range(0, Xtr.size(0), 128):   # batches of 128
            idx = perm[i:i + 128]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
            model.fc1.weight.data *= mask   # keep the pruned weights at zero

    # Step C: measure
    after = test_acc(model)
    after_int8 = int8_acc(model)
    zeros = (model.fc1.weight == 0).float().mean().item() * 100
    print(f'fc1 prune {pct}%: after pruning={before:.2f}%  '
          f'after fine-tune={after:.2f}%  fine-tune+int8={after_int8:.2f}%  '
          f'fc1 zeros={zeros:.1f}%')

    # Step D: save
    torch.save(model.state_dict(), pruned_model_path(pct))
