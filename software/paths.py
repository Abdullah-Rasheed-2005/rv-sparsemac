"""
paths.py - One place for every file and folder path in the project
==================================================================
All scripts import their paths from here. Because the paths are built
from the location of this file, the scripts work no matter which folder
you start them from.

Repository layout used here:
    <repo>/data/                 MNIST dataset (downloaded automatically)
    <repo>/models/               trained PyTorch weights (.pth)
    <repo>/hardware/mem/         hex files + golden vectors for Verilog
"""

from pathlib import Path

# <repo>/software/paths.py  ->  parents[1] is the repository root
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
MEM_DIR = ROOT / 'hardware' / 'mem'

BASELINE_MODEL = MODELS_DIR / 'baseline_fp32.pth'


def pruned_model_path(percent):
    """Path of the pruned + fine-tuned model, e.g. pruned_fc1_80.pth"""
    return MODELS_DIR / f'pruned_fc1_{percent}.pth'
