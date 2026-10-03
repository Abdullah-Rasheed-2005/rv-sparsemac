"""
model.py - The neural network used in this project
==================================================
A .pth file stores only the weight numbers, not the shape of the
network. So every script that loads a model imports this class to
build the same shape first.

Shape: 784 -> 64 -> 10
    784 inputs  = a 28x28 MNIST image, flattened
    64 hidden   = fully connected layer 1 (fc1) + ReLU
    10 outputs  = fully connected layer 2 (fc2), one score per digit
"""

import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        # fc1: 784 inputs -> 64 outputs (this layer holds ~98.7% of all weights)
        self.fc1 = nn.Linear(784, 64)
        # ReLU(x) = max(0, x)
        self.relu = nn.ReLU()
        # fc2: 64 inputs -> 10 outputs (one per digit 0-9)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # Flatten [batch, 1, 28, 28] (or [batch, 784]) into [batch, 784]
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))   # layer 1: matrix multiply + bias, then ReLU
        return self.fc2(x)           # layer 2: raw scores (logits)
