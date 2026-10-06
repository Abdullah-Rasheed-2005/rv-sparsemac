# The same study on Fashion-MNIST

Written by `software/fashion_pareto.py`. Same network size as for MNIST (784 -> 64 -> 10), so the
hardware is unchanged. 10,000 test images, exact integer arithmetic and cycle cost of the hardware
(layer-1 cycles). One random seed.

- zero pixels in the test images: 50.0% (MNIST: about 81%)
- dense network after 10 epochs, integer arithmetic: 87.28%

## Cost without a budget

"plain" = pruned and fine-tuned for 16 epochs without any budget; "budget-aware" = 8 plain epochs,
then 8 epochs with inputs cut at random budgets. Both evaluated here WITHOUT a budget.

| fc1 pruned | weights left | accuracy, plain | accuracy, budget-aware | average cycles | slowest test image | slowest possible image |
|---|---|---|---|---|---|---|
| 80 % | 9,993 | 87.34% | 87.41% | 3,404 | 9,067 | 10,143 |
| 90 % | 4,996 | 86.46% | 86.27% | 1,837 | 5,048 | 5,755 |
| 95 % | 2,497 | 84.18% | 84.11% | 1,393 | 3,361 | 3,743 |

## Accuracy with a hard layer-1 budget

The budget is given as a share of the average cost of the same model without a budget.

| fc1 pruned | budget = guaranteed worst case | share of the average cost | accuracy | loss against no budget | average cycles used |
|---|---|---|---|---|---|
| 80 % | none (10,143) | - | 87.41% | - | 3,404 |
| 80 % | 1,350 | 0.40 | 83.32% | -4.09 | 1,339 |
| 80 % | 2,050 | 0.60 | 86.04% | -1.37 | 1,989 |
| 80 % | 2,700 | 0.80 | 86.96% | -0.45 | 2,463 |
| 80 % | 3,400 | 1.00 | 87.30% | -0.11 | 2,863 |
| 80 % | 4,250 | 1.25 | 87.32% | -0.09 | 3,200 |
| 90 % | none (5,755) | - | 86.27% | - | 1,837 |
| 90 % | 750 | 0.40 | 82.07% | -4.20 | 743 |
| 90 % | 1,100 | 0.60 | 85.11% | -1.16 | 1,076 |
| 90 % | 1,450 | 0.80 | 85.74% | -0.53 | 1,342 |
| 90 % | 1,850 | 1.00 | 86.02% | -0.25 | 1,571 |
| 90 % | 2,300 | 1.25 | 86.27% | +0.00 | 1,740 |
| 95 % | none (3,743) | - | 84.11% | - | 1,393 |
| 95 % | 550 | 0.40 | 82.57% | -1.54 | 546 |
| 95 % | 850 | 0.60 | 83.91% | -0.20 | 834 |
| 95 % | 1,100 | 0.80 | 83.92% | -0.19 | 1,030 |
| 95 % | 1,400 | 1.00 | 84.08% | -0.03 | 1,208 |
| 95 % | 1,750 | 1.25 | 84.08% | -0.03 | 1,341 |

## Other ways to guarantee the same worst case (90 % model)

Each alternative starts from the same 90 % pruned model and is fine-tuned for 8 more epochs,
the same as the budget-aware model. A column without any weight costs the alternatives nothing.

| Guaranteed worst case (cycles) | Budget | Prune more (weights left) | Fixed pixel subset (pixels kept) |
|---|---|---|---|
| 1,100 | 85.11% | 81.91% (690, worst 1,099) | 80.30% (160, worst 1,098) |
| 1,850 | 86.02% | 84.60% (1,408, worst 1,850) | 81.64% (225, worst 1,841) |
