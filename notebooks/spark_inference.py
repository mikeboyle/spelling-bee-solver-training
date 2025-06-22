from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
import pandas as pd
import numpy as np

def inference(df, svd_model, classifier):
    """
    Apply inference to a Spark DataFrame with word embeddings.
    
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