# Results

All numbers were measured on the 10,000 MNIST test images unless noted.

## Weight statistics (baseline model)

| Layer | Weights | Range | int8 scale (max/127) |
|---|---|---|---|
| fc1 | 50,176 | -0.576 .. 0.688 | 0.00542 |
| fc2 | 640 | -0.910 .. 0.513 | 0.00716 |

## Pruning without retraining (int8 weights)

Pruning both layers by the same fraction (`sweep_prune_both_layers.py`):

| Pruned | Accuracy |
|---|---|
| 0% | 96.83% |
| 30% | 96.69% |
| 50% | 95.02% |
| 70% | 77.24% |
| 90% | 26.07% |

Pruning only fc1 and keeping fc2 dense (`sweep_prune_fc1.py`):

| fc1 pruned | Accuracy |
|---|---|
| 30% | 96.78% |
| 50% | 96.77% |
| 60% | 96.32% |
| 70% | 94.72% |
| 80% | 89.79% |
| 90% | 59.63% |

Lesson: fc2 is small and sensitive. Keep it dense and prune fc1.

## Pruning with fine-tuning (`finetune_pruned.py`)

5 epochs of training after pruning, with the mask applied after every step.

| fc1 pruned | After pruning | After fine-tune | Fine-tune + int8 |
|---|---|---|---|
| 70% | 94.81% | 97.13% | 97.11% |
| 80% | 89.78% | 96.87% | 96.88% |
| 90% | 59.73% | 95.67% | 95.73% |

Note: the 70% model scores higher than the 96.82% baseline. Part of this comes
from the extra 5 epochs of training, not from pruning. A fair comparison would
also fine-tune the baseline for 5 epochs. This is on the to-do list.

## Integer-only pipeline (80% pruned model)

| Item | Value |
|---|---|
| Float accuracy | 96.87% |
| Pure integer accuracy | 96.89% |
| Hidden shift | 13 |
| fc1 nonzero weights | 10,033 / 50,176 |
| fc2 nonzero weights | 633 / 640 |
| Zero input pixels | 80.7% |
| Zero hidden values | 41.8% |
| Golden logits exact match | 100 / 100 |

## MAC operations per image (100 test images)

| Method | MACs | Reduction |
|---|---|---|
| Dense | 50,816 | 1.0x |
| Skip zero weights | 10,666 | 4.8x |
| Skip zero inputs | 9,347 | 5.4x |
| Skip both | 2,077 | 24.5x |

These are operation counts, not cycle counts. Hardware speedup will be
measured in simulation and added here.
