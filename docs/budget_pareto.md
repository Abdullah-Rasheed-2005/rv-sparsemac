# Pruning and the run-time budget together

Written by `software/budget_pareto.py`. 10,000 MNIST test images, exact integer arithmetic and
cycle cost of the hardware (layer-1 cycles). One random seed per sparsity.
Each model: prune `models/pruned_fc1_80.pth` further, 8 plain epochs, then 8 budget-aware epochs.

Accuracy (%) for each layer-1 budget. "-" = the budget is above the slowest test image, so it never cuts.

| fc1 pruned | weights left | no budget | 400 | 600 | 800 | 1,000 | 1,300 | 1,700 | 2,200 |
|---|---|---|---|---|---|---|---|---|---|
| 80 % | 10,017 | 97.28 | 75.90 | 88.08 | 92.38 | 94.75 | 96.12 | 96.92 | 97.14 |
| 85 % | 7,519 | 97.13 | 86.59 | 93.12 | 95.10 | 96.11 | 96.85 | 97.03 | 97.12 |
| 90 % | 5,013 | 96.85 | 92.75 | 95.36 | 96.41 | 96.81 | 96.88 | 96.85 | - |
| 95 % | 2,507 | 96.15 | 95.18 | 96.04 | 96.13 | 96.15 | - | - | - |

## Cost without a budget

| fc1 pruned | accuracy before / after budget-aware training | average cycles | slowest test image | slowest possible image |
|---|---|---|---|---|
| 80 % | 97.07% / 97.28% | 1,857 | 3,782 | 10,327 |
| 85 % | 97.05% / 97.13% | 1,351 | 2,754 | 7,855 |
| 90 % | 96.80% / 96.85% | 923 | 1,893 | 5,483 |
| 95 % | 95.64% / 96.15% | 563 | 1,169 | 3,411 |

## Every measured point (accuracy, average cycles, guaranteed worst case)

| fc1 pruned | budget = guaranteed worst case | accuracy | average cycles used |
|---|---|---|---|
| 80 % | none (10,327) | 97.28% | 1,857 |
| 80 % | 400 | 75.90% | 393 |
| 80 % | 600 | 88.08% | 593 |
| 80 % | 800 | 92.38% | 793 |
| 80 % | 1,000 | 94.75% | 988 |
| 80 % | 1,300 | 96.12% | 1,266 |
| 80 % | 1,700 | 96.92% | 1,573 |
| 80 % | 2,200 | 97.14% | 1,784 |
| 85 % | none (7,855) | 97.13% | 1,351 |
| 85 % | 400 | 86.59% | 395 |
| 85 % | 600 | 93.12% | 595 |
| 85 % | 800 | 95.10% | 788 |
| 85 % | 1,000 | 96.11% | 967 |
| 85 % | 1,300 | 96.85% | 1,180 |
| 85 % | 1,700 | 97.03% | 1,316 |
| 85 % | 2,200 | 97.12% | 1,349 |
| 90 % | none (5,483) | 96.85% | 923 |
| 90 % | 400 | 92.75% | 396 |
| 90 % | 600 | 95.36% | 588 |
| 90 % | 800 | 96.41% | 750 |
| 90 % | 1,000 | 96.81% | 857 |
| 90 % | 1,300 | 96.88% | 914 |
| 90 % | 1,700 | 96.85% | 923 |
| 95 % | none (3,411) | 96.15% | 563 |
| 95 % | 400 | 95.18% | 387 |
| 95 % | 600 | 96.04% | 517 |
| 95 % | 800 | 96.13% | 557 |
| 95 % | 1,000 | 96.15% | 562 |
