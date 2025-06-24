from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, FloatType
import numpy as np

def convert_df_to_features_v1(df: DataFrame) -> DataFrame:
    """
    Converts a spark dataframe from the silver.word_states
    table to a dataframe with columns `words`, `features`, and `label`.
    
    The `features` column concatenates the word frequency and embedding into
    a single vector, with the frequency as the first component.
    
    The `labels` column is the classification (`1.0` or `0.0`).
    
    The returned df is appropriate for training and inference for the v1 model.

    Args:
        df: a DataFrame of rows from silver.word_states where label is not null
    
    Returns:
        final_df: DataFrame with columns `words`, `features`, `label`
    """
    # Define UDF to concatenate frequency with embedding
    def concat_features(frequency, embedding):
        return [float(frequency)] + embedding

    concat_udf = F.udf(concat_features, ArrayType(FloatType()))

    # Apply the transformation
    training_df = df.withColumn(
        "features", 
        concat_udf(F.col("log_frequency"), F.col("embedding"))
    )

    # Select features and label
    final_df = training_df.select("features", "label")
    return final_df

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