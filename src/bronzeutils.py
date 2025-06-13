from typing import Any, TypedDict
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    BooleanType,
    DateType,
    IntegerType,
)
import pyspark.sql.functions as F

bronze_schema: StructType = StructType(
    [
        StructField("word", StringType(), False),
        StructField("accepted", BooleanType(), False),
        StructField("was_in_wordlist", BooleanType(), False),
        StructField("puzzle_date", DateType(), False),
        StructField("center_letter", StringType(), False),
        StructField("outer_letters", StringType(), False),
        StructField("wordlist_version", IntegerType(), False),
    ]
)


def rows_to_bronze_df(rows: list[Any], spark: SparkSession) -> DataFrame:
    """
    Writes rows to dataframe and (TODO) saves to table
    """
    df = spark.createDataFrame(rows, schema=bronze_schema)

    # Add derived year/month/day columns for partitioning
    df = (
        df.withColumn("year", F.year("puzzle_date"))
        .withColumn("month", F.month("puzzle_date"))
        .withColumn("day", F.dayofmonth("puzzle_date"))
    )

    return df
