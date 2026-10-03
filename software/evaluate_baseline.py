"""
evaluate_baseline.py - Measure the accuracy of the baseline model
=================================================================
Loads models/baseline_fp32.pth and tests it on the 10,000 MNIST test
images. The number printed here is the reference that every pruned or
quantized version is compared against.

Run from the repository root:   python3 software/evaluate_baseline.py
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import SimpleNN
from paths import DATA_DIR, BASELINE_MODEL

model = SimpleNN()
model.load_state_dict(torch.load(BASELINE_MODEL))
model.eval()   # evaluation mode (we are only testing)

# Only ToTensor(), no Normalize. The model was trained this way.
test_data = datasets.MNIST(root=str(DATA_DIR), train=False, download=True,
                           transform=transforms.ToTensor())
loader = DataLoader(test_data, batch_size=64, shuffle=False)

correct = total = 0
with torch.no_grad():
    for images, labels in loader:
        predicted = model(images).argmax(dim=1)   # highest score wins
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Baseline Test Accuracy: {100 * correct / total:.2f}%')
