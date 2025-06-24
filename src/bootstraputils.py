import os
import csv
from pathlib import Path
import time
from typing import Any, Callable
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    DataType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from src.constants import BATCH_LOGS_PATH
from src.fileutils import get_local_path, last_line_csv
from src.embeddingutils import get_word_embeddings
from src.ngramsutils import get_word_frequencies_threaded
from src.sparkdbutils import write_to_table_replace_where


def collect_word_data_batched(
    spark: SparkSession,
    words_list: list[str],
    collect_fn: Callable[[list[str]], dict[str, Any]],
    db_name: str,
    table_name: str,
    data_col: str,
    data_type: DataType,
    batch_size: int = 100,
    resume_job: bool = False,
) -> None:
    """
    Collect data for a list of words, saving each batch to to the table `{db_name}.{table_name}`.
    Maintains a log of batches processed so that interrupted jobs can be resumed from the latest batch.

    Args:
    - spark: SparkSession
    - words_list: the list of words to be processed
    - collect_fn: function that performs the processing on the words, returning a dictionary of [word]: [data]
    - db_name: the name of the database to write to
    - table_name: the name of the table to write to
    - data_col: the column name of the data collected by collect_fn
    - data_type: the data type of the data colelcted by collect_fn
    - batch_size: the number of words per batch
    - resum_job: if True, finds the last completed batch in the logfile and resumes from there.
    """
    # Make logfile path
    log_file_name = f"{data_col}_batch_log.log"
    logfile_path = f"{BATCH_LOGS_PATH}/{log_file_name}"
    logfile_dir = Path(get_local_path(logfile_path)).parent
    logfile_dir.mkdir(parents=True, exist_ok=True)


    # Initialize or find resume point
    start_batch = 0

    if resume_job:
        if os.path.exists(get_local_path(logfile_path)):
            last_row = last_line_csv(get_local_path(logfile_path))
            last_batch = int(last_row[0])
            end_idx = int(last_row[2])
            start_batch = last_batch + 1
            print(
                f"Resuming from batch {start_batch} with {end_idx + 1} words already processed"
            )
        else:
            raise Exception(f"Cannot resume because file at {get_local_path(logfile_path)} does not exist.")

    total_batches = (len(words_list) + batch_size - 1) // batch_size

    try:
        # Process remaining batches
        for batch_idx in range(start_batch, total_batches):
            batch_start = time.time()

            # Get the words for this batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(words_list))
            batch_words = words_list[start_idx:end_idx]

            # Get the data for the words for this batch
            batch_collect_start = time.time()
            
            result_dict = collect_fn(batch_words)
            
            batch_collect_time = time.time() - batch_collect_start

            # Get the dataframe rows, including column for batch_id
            rows = [(word, data, batch_idx) for word, data in result_dict.items()]
            schema = StructType(
                [
                    StructField("word", StringType(), False),
                    StructField(data_col, data_type, False),
                    StructField("batch_id", IntegerType(), False),
                ]
            )

            df = spark.createDataFrame(rows, schema)

            batch_save_start = time.time()
            write_to_table_replace_where(
                spark, df, db_name, table_name, {"batch_id": batch_idx}
            )

            batch_save_time = time.time() - batch_save_start
            full_batch_time = time.time() - batch_start

            # Append to log
            with open(get_local_path(logfile_path), "a") as f_log:
                writer = csv.writer(f_log)
                writer.writerow(
                    (
                        batch_idx,
                        start_idx,
                        end_idx,
                        batch_words[0],
                        batch_words[-1],
                        f"{batch_collect_time:.2f}",
                        f"{batch_save_time:.2f}",
                        f"{full_batch_time:.2f}",
                    )
                )

            print(f"⌛️ collected batch data in {batch_collect_time:.2f}s")
            print(f"⌛️ saved batch in {batch_save_time:.2f}s")
            print(
                f"✅ Completed batch {batch_idx + 1}/{total_batches} in {full_batch_time:.2f}s\n"
            )

    except Exception as err:
        print(f"Exception occurred! Resume job from batch {batch_idx}")
        raise err

    print(f"✅ Processing complete! Logfile: {logfile_path}")


def collect_frequencies_batched(
    spark: SparkSession,
    words_list: list[str],
    db_name: str,
    table_name: str,
    batch_size: int = 100,
    resume_job: bool = False,
) -> None:
    return collect_word_data_batched(
        spark=spark,
        words_list=words_list,
        collect_fn=get_word_frequencies_threaded,
        db_name=db_name,
        table_name=table_name,
        data_col="frequency",
        data_type=FloatType(),
        batch_size=batch_size,
        resume_job=resume_job,
    )


def collect_embeddings_batched(
    spark: SparkSession,
    words_list: list[str],
    db_name: str,
    table_name: str,
    batch_size: int = 100,
    resume_job: bool = False,
) -> None:
    return collect_word_data_batched(
        spark=spark,
        words_list=words_list,
        collect_fn=get_word_embeddings,
        db_name=db_name,
        table_name=table_name,
        data_col="embedding",
        data_type=ArrayType(FloatType(), False),
        batch_size=batch_size,
        resume_job=resume_job,
    )
