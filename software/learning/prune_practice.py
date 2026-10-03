# ============================================================
# prune_practice.py - Learning exercise: pruning
# Sets small weights to zero and measures the sparsity.
# IMPORTANT: Pruning should run BEFORE quantization.
# ============================================================

def prune_weights(weights, threshold):
    """
    Look at the size (absolute value) of each weight and set the
    small ones to 0. The sign does not matter, only the size.
    """
    pruned = []
    for w in weights:
        if abs(w) < threshold:
            pruned.append(0.0)
        else:
            pruned.append(w)
    return pruned


def calculate_sparsity(weights):
    """What percent of the weights are zero."""
    zero_count = sum(1 for w in weights if w == 0.0)
    return (zero_count / len(weights)) * 100


def estimate_speedup(weights, cycles_per_mac=1):
    """
    Estimate how many cycles hardware would save if it skips zeros.
    The returned percent is the reduction in cycles.
    """
    total = len(weights)
    zero_count = sum(1 for w in weights if w == 0.0)
    cycles_without_skip = total * cycles_per_mac
    cycles_with_skip = (total - zero_count) * cycles_per_mac
    speedup_percent = ((cycles_without_skip - cycles_with_skip) / cycles_without_skip) * 100
    return cycles_without_skip, cycles_with_skip, speedup_percent


# ============================================================
# Exercise: the same weights used in quantize_practice.py
# ============================================================

if __name__ == "__main__":
    sample_weights = [0.85, 0.003, -0.72, 0.001, 0.45, -0.0008, 0.91, -0.15]

    threshold = 0.01
    pruned = prune_weights(sample_weights, threshold)

    print(f"Threshold: {threshold}")
    print(f"{'Original':>10} {'Pruned':>10} {'Status':>12}")
    for orig, p in zip(sample_weights, pruned):
        status = "PRUNED" if orig != p else "KEPT"
        print(f"{orig:>10.4f} {p:>10.4f} {status:>12}")

    sparsity = calculate_sparsity(pruned)
    total, with_skip, speedup = estimate_speedup(pruned)

    print()
    print(f"Sparsity: {sparsity:.1f}%")
    print(f"Cycles without zero-skip: {total}, with zero-skip: {with_skip}")
    print(f"Cycle reduction: {speedup:.1f}%")
