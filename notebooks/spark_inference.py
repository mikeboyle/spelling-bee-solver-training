from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
import pandas as pd
import numpy as np

"""
TODO: use this function in context
- Does silver.word_states have a probability column or a label column?
- Should input df be filtered only to the words that need prediction? label = null or last_seen_on = null (should be the same thing)
- Is this ever done incrementally or only as a boostrap/update? (I think the latter, check the plan)
- In this case do we write as a single new gold table (check the plan)
- First rename label column probability, then filter on probability = null to get df_model and filter on probability NOT null to get df_truth
- Drop "probability" (or select only word, frequency, embedding) to get df_model2
- Unpickle svd and clf, invoke df_model3 = inference(df_model2, svd, clf)
- final_df = df_truth.union(df_model2)

- If no table exists, create and write table for first time, partitioned by source?
- If a table does exist, is this a merge into??

"""


def inference(df, svd_model, classifier):
    """
    Apply inference to a Spark DataFrame with word, frequency, and embedding columns.
    
    Args:
        df: Spark DataFrame with columns [word, frequency, embedding]
        svd_model: Pre-trained SVD model with transform() method
        classifier: Trained classifier with predict_proba() method
    
    Returns:
        Spark DataFrame with added 'probability' column
    """
    
    # Define a pandas UDF for batch processing
    @F.pandas_udf(returnType=FloatType())
    def predict_probability(frequencies: pd.Series, embeddings: pd.Series) -> pd.Series:
        # Convert embeddings from list format to numpy array
        embedding_matrix = np.array(embeddings.tolist())
        
        # Apply SVD to reduce dimensions from 768 to 50
        reduced_embeddings = svd_model.transform(embedding_matrix)
        
        # Transform frequencies with log10(frequency + 1) and reshape
        freq_array = np.log10(frequencies.values + 1).reshape(-1, 1)
        
        # Concatenate frequency with reduced embeddings
        features = np.concatenate([freq_array, reduced_embeddings], axis=1)
        
        # Get predictions (probability of positive class)
        probabilities = classifier.predict_proba(features)[:, 1]
        
        return pd.Series(probabilities)
    
    # Apply the UDF to get probabilities
    result_df = df.withColumn(
        "probability", 
        predict_probability(F.col("frequency"), F.col("embedding"))
    )
    
    return result_df

# Usage example:
# result_df = inference(your_df, your_svd_model, your_classifier)