# Models

PyTorch weight files (about 200 KB each).

| File | Content |
|---|---|
| `baseline_fp32.pth` | Float32 baseline, 96.82% test accuracy |
| `pruned_fc1_70.pth` | fc1 pruned 70% + fine-tuned |
| `pruned_fc1_80.pth` | fc1 pruned 80% + fine-tuned (used for the hardware export) |
| `pruned_fc1_90.pth` | fc1 pruned 90% + fine-tuned |

Produced by `software/train_baseline.py` and `software/finetune_pruned.py`.
