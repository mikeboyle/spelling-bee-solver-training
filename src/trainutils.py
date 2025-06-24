import numpy as np

def evaluate_thresholds(log_freq, labels, low_range, high_range, min_bin_size=0.1):
    best_score = float('inf')
    best_thresholds = (None, None)
    n = len(log_freq)

    for low in low_range:
        for high in high_range:
            if low >= high:
                continue

            low_mask = log_freq < low
            high_mask = log_freq > high
            med_mask = (log_freq >= low) & (log_freq <= high) # new

            if low_mask.mean() < min_bin_size or high_mask.mean() < min_bin_size:
                continue

            low_labels = labels[low_mask]
            high_labels = labels[high_mask]
            med_labels = labels[med_mask] # new

            pos_rate_low = (low_labels == 1).mean()
            neg_rate_high = (high_labels == 0).mean()
            pos_rate_med = (med_labels == 1).mean()

            score = pos_rate_low + neg_rate_high + np.abs(0.5 - pos_rate_med)  # still minimizing

            if score < best_score:
                best_score = score
                best_thresholds = (low, high)

    return best_thresholds, best_score