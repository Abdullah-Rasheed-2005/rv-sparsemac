"""
train_baseline.py - Train the float32 baseline model on MNIST
=============================================================
Trains the 784 -> 64 -> 10 network (see model.py) for 5 epochs and
saves the weights to models/baseline_fp32.pth.

This model is the "golden" reference. Pruned and quantized versions
are compared against its accuracy.

Run from the repository root:   python3 software/train_baseline.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import SimpleNN
from paths import DATA_DIR, MODELS_DIR, BASELINE_MODEL

# ---------------------------------------------------------
# STEP 1: Choose the device (CPU is enough, no GPU needed)
# ---------------------------------------------------------
device = torch.device("cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------
# STEP 2: Load the MNIST dataset
# ---------------------------------------------------------
# ToTensor() turns an image into a tensor and scales the
# pixel values from 0-255 down to 0-1.
transform = transforms.Compose([
    transforms.ToTensor()
])

# train=True  -> training set (60,000 images)
# train=False -> test set (10,000 images)
train_dataset = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=transform)

# A DataLoader gives the data in small batches
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ---------------------------------------------------------
# STEP 3: Create the model
# ---------------------------------------------------------
model = SimpleNN().to(device)
print(model)

# ---------------------------------------------------------
# STEP 4: Loss function and optimizer
# ---------------------------------------------------------
criterion = nn.CrossEntropyLoss()                        # standard loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)    # updates the weights

# ---------------------------------------------------------
# STEP 5: Training loop
# ---------------------------------------------------------
epochs = 5

for epoch in range(epochs):
    model.train()   # training mode
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass: get the predictions
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass: compute gradients and update the weights
        optimizer.zero_grad()   # clear old gradients
        loss.backward()          # compute new gradients (backpropagation)
        optimizer.step()         # update the weights

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# ---------------------------------------------------------
# STEP 6: Check the test accuracy
# ---------------------------------------------------------
model.eval()   # evaluation mode
correct = 0
total = 0

with torch.no_grad():   # no gradients needed while testing
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)   # highest score wins
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")

# ---------------------------------------------------------
# STEP 7: Save the model
# ---------------------------------------------------------
MODELS_DIR.mkdir(exist_ok=True)
torch.save(model.state_dict(), BASELINE_MODEL)
print(f"Model saved as {BASELINE_MODEL}")
