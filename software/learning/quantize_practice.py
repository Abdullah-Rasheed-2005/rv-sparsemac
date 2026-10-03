# ============================================================
# quantize_practice.py - Learning exercise: quantization
# Turns a small list of float weights into int8 numbers.
#
# NOTE: This exercise uses the "min to max" (asymmetric) formula
# without a zero-point, so it clips the biggest values (see the
# output: 0.85 and 0.91 both become 127). The real project uses
# the symmetric formula scale = max(|w|) / 127.
# See software/export_int8.py for the real version.
# ============================================================

def calculate_scale_factor(weights, bits=8):
    """
    Look at the list of weights and work out the scale factor.
    Formula: scale_factor = float_range / int_range
    """
    float_min = min(weights)
    float_max = max(weights)

    int_max = 2**(bits-1) - 1   # for 8 bits this is 127
    int_min = -2**(bits-1)      # for 8 bits this is -128

    scale_factor = (float_max - float_min) / (int_max - int_min)
    return scale_factor, float_min, float_max


def quantize_weights(weights, scale_factor, bits=8):
    """
    Quantize every weight with this formula:
    quantized = round(float_value / scale_factor)
    Then CLIP the result (clip, do not wrap around).
    """
    int_max = 2**(bits-1) - 1
    int_min = -2**(bits-1)

    quantized = []
    for w in weights:
        q = round(w / scale_factor)
        # Clipping: if the value is out of range, stop at the limit
        q = max(int_min, min(int_max, q))
        quantized.append(q)

    return quantized


def dequantize_weights(quantized, scale_factor):
    """Convert back to float so we can look at the error."""
    return [q * scale_factor for q in quantized]


# ============================================================
# Exercise: test on a small list (easy to check by hand)
# ============================================================

if __name__ == "__main__":
    sample_weights = [0.85, 0.003, -0.72, 0.001, 0.45, -0.0008, 0.91, -0.15]

    scale, fmin, fmax = calculate_scale_factor(sample_weights)
    print(f"Float range: {fmin} to {fmax}")
    print(f"Scale factor: {scale:.6f}")
    print()

    quantized = quantize_weights(sample_weights, scale)
    dequantized = dequantize_weights(quantized, scale)

    print(f"{'Original':>10} {'Quantized':>10} {'Dequantized':>12} {'Error':>10}")
    for orig, q, dq in zip(sample_weights, quantized, dequantized):
        error = abs(orig - dq)
        print(f"{orig:>10.4f} {q:>10d} {dq:>12.4f} {error:>10.5f}")
