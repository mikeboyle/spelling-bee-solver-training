from typing import Any
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
import pyspark.sql.functions as F

WORD_DECISIONS_PARTITIONS = ["year", "month"]

word_decisions_schema: StructType = StructType(
    [
        StructField("word", StringType(), False),
        StructField("center_letter", StringType(), False),
        StructField("outer_letters", StringType(), False),
        StructField("puzzle_date", DateType(), False),
        StructField("accepted", FloatType(), False)
    ]
)

words_schema: StructType = StructType([
    StructField("word", StringType(), False),
    StructField("letter_set", StringType(), False),
    StructField("version", IntegerType(), False),
    StructField("date_added", DateType(), True), 
    StructField("frequency", DoubleType(), False),
    StructField("embedding", ArrayType(FloatType(), True), False)
])


def rows_to_word_decisions_df(rows: list[Any], spark: SparkSession) -> DataFrame:
    """
    Writes rows to dataframe
    """
    df = spark.createDataFrame(rows, schema=word_decisions_schema)

    # Add derived year/month/day columns for partitioning
    df = (
        df.withColumn("year", F.year("puzzle_date"))
        .withColumn("month", F.month("puzzle_date"))
        .withColumn("day", F.dayofmonth("puzzle_date"))
    )

    return df

def validate_embeddings(df, column='embedding', expected_length=768, expected_dtype='float64'):
    """Validate that embeddings meet expected criteria"""
    results = {
        'total_rows': len(df),
        'valid_embeddings': 0,
        'issues': []
    }
    
    for idx, embedding in enumerate(df[column]):
        try:
            # Convert to numpy array
            arr = np.array(embedding)
            
            # Check length
            if len(arr) != expected_length:
                results['issues'].append(f"Row {idx}: Wrong length {len(arr)}, expected {expected_length}")
                continue
                
            # Check dtype
            if arr.dtype != expected_dtype:
                results['issues'].append(f"Row {idx}: Wrong dtype {arr.dtype}, expected {expected_dtype}")
                continue
                
            results['valid_embeddings'] += 1
            
        except Exception as e:
            results['issues'].append(f"Row {idx}: Error - {str(e)}")

    if results['issues']:
        print("Issues found:")
        for issue in results['issues'][:10]:  # Show first 10 issues
            print(f"  {issue}")
        raise ValueError("Embeddings are not valid; see issues printed to output.")
    
    return results
