# Budget study

Written by `software/budget_study.py`. 10,000 MNIST test images, exact integer arithmetic of the
hardware. Every model below starts from `models/pruned_fc1_80.pth` and is fine-tuned for 8 more epochs.
Cycles are layer-1 cycles; layer 2 and the finish sweeps add a fixed amount to every design.

## A. Budget-aware model, several random seeds

| Layer-1 budget | seed 1 | seed 2 | seed 3 | mean | spread (max - min) | average cycles used |
|---|---|---|---|---|---|---|
| none | 96.99% | 96.98% | 96.98% | 96.98% | 0.01 | 1,856 |
| 1,000 | 95.37% | 95.35% | 95.40% | 95.37% | 0.05 | 989 |
| 1,300 | 96.03% | 96.04% | 96.07% | 96.05% | 0.04 | 1,267 |
| 1,700 | 96.71% | 96.67% | 96.68% | 96.69% | 0.04 | 1,573 |
| 2,200 | 96.96% | 96.85% | 96.90% | 96.90% | 0.11 | 1,783 |

Without a budget the slowest possible image needs about 10,324 layer-1 cycles; with a budget it needs the budget.

## B. Other ways to guarantee the same worst case

| Guaranteed worst case (cycles) | Budget (mean of seeds) | Prune more (weights left) | Fixed pixel subset (pixels kept) |
|---|---|---|---|
| 1,000 | 95.37% | 87.70% (605, worst 1,000) | 69.55% (48, worst 981) |
| 1,300 | 96.05% | 91.34% (875, worst 1,299) | 72.25% (65, worst 1,295) |
| 1,700 | 96.69% | 93.11% (1,298, worst 1,700) | 79.70% (85, worst 1,680) |
| 2,200 | 96.90% | 94.42% (1,831, worst 2,199) | 83.91% (118, worst 2,175) |

## C. Simply pruning more, without a guarantee

| Model | Accuracy | Average cycles | Slowest test image | Slowest possible image |
|---|---|---|---|---|
| 80 % pruned, same extra training, no budget | 97.07% | 1,857 | 3,780 | 10,036 |
| 85 % pruned | 97.05% | 1,355 | 2,757 | 7,556 |
| 90 % pruned | 96.80% | 924 | 1,897 | 5,106 |
| 95 % pruned | 95.64% | 519 | 1,077 | 2,794 |
| budget 1,000 (mean of seeds) | 95.37% | 989 | 1,000 | 1,000 |
| budget 1,300 (mean of seeds) | 96.05% | 1,267 | 1,300 | 1,300 |
| budget 1,700 (mean of seeds) | 96.69% | 1,573 | 1,700 | 1,700 |
| budget 2,200 (mean of seeds) | 96.90% | 1,783 | 2,200 | 2,200 |

## Timing leak

How often the digit is guessed correctly from the layer-1 cycle count alone
(guess learned on 10,000 training images, tested on the test images; first seed).

| Mode | Digit guessed correctly |
|---|---|
| no budget | 22.8% |
| budget 1,000 | 12.7% |
| budget 1,300 | 17.0% |
| budget 1,700 | 21.1% |
| budget 2,200 | 22.8% |
| constant time (every image the same count) | 11.3% (always guessing the most common digit) |
