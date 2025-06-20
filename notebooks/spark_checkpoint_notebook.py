# Separated approach: Data collection + Data application
import csv
import os
import pickle
from pyspark.sql import functions as F
from pyspark.sql.types import *

_TARGET_DB_NAME = "bronze"
_TARGET_TABLE_NAME = "words"
VERSION = 1

# Step 1: Filter wordlist and create temp CSV (same as before)
wordlist = filter_wordlist(word_file_to_set(f"{WORDLIST_PATH}/{RAW_WORDLIST_FILENAME}"))
print(len(wordlist), "words")

rows = [(word, get_letter_set(word), VERSION) for word in wordlist]
temp_path = get_local_path(f"{WORDLIST_PATH}/{WORDLIST_TEMP_CSV_FILENAME}")
with open(temp_path, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["word", "letter_set", "version"])
    writer.writerows(rows)

# Step 2: Read CSV directly into Spark
initial_schema = StructType([
    StructField("word", StringType(), True),
    StructField("letter_set", StringType(), True),
    StructField("version", IntegerType(), True)
])

spark_df = spark.read.csv(temp_path, header=True, schema=initial_schema)

# STAGE 1: Collect frequency and embedding data with checkpointing
def collect_word_data_with_checkpoint(words_list: list, 
                                     output_path: str,
                                     batch_size: int = 100,
                                     resume_job: bool = False):
    """
    Collect frequency and embedding data for a list of words with checkpointing.
    Saves final results as a pickle file that can be reused for multiple purposes.
    
    Args:
        words_list: List of words to process
        output_path: Path to save final pickle file with all data
        batch_size: Batch size for API calls (default 100 for API limits)
        resume_job: Whether to resume from existing checkpoint
    
    Returns:
        dict: Contains 'freq_dict' and 'embeddings_dict'
    """
    checkpoint_path = output_path.replace('.pkl', '_checkpoint.pkl')
    
    # Check for existing final output
    if os.path.exists(output_path) and not resume_job:
        print(f"Final output already exists at {output_path}")
        with open(output_path, 'rb') as f:
            return pickle.load(f)
    
    # Initialize or load checkpoint
    freq_dict = {}
    embeddings_dict = {}
    start_batch = 0
    
    if resume_job and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
            freq_dict = checkpoint_data.get('freq_dict', {})
            embeddings_dict = checkpoint_data.get('embeddings_dict', {})
            start_batch = checkpoint_data.get('last_batch', 0) + 1
        print(f"Resuming from batch {start_batch} with {len(freq_dict)} words already processed")
    
    total_batches = (len(words_list) + batch_size - 1) // batch_size
    
    try:
        # Process remaining batches
        for batch_idx in range(start_batch, total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(words_list))
            batch_words = words_list[start_idx:end_idx]
            
            print(f"Processing batch {batch_idx + 1}/{total_batches}: words {start_idx}-{end_idx-1}")
            
            # Get frequencies for this batch
            batch_freq_dict = get_word_frequencies_threaded(batch_words, max_workers=10)
            freq_dict.update(batch_freq_dict)
            
            # Get embeddings for this batch
            batch_embeddings_dict = get_word_embeddings(batch_words)
            embeddings_dict.update(batch_embeddings_dict)
            
            # Save checkpoint after each batch
            checkpoint_data = {
                'freq_dict': freq_dict,
                'embeddings_dict': embeddings_dict,
                'last_batch': batch_idx,
                'total_batches': total_batches
            }
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            print(f"✅ Completed batch {batch_idx + 1}/{total_batches}")
    
    except Exception as err:
        print(f"Exception occurred! Resume job from batch {batch_idx}")
        print(f"Current progress: {len(freq_dict)} words processed")
        raise err
    
    # Save final output
    final_data = {
        'freq_dict': freq_dict,
        'embeddings_dict': embeddings_dict,
        'processed_words': list(freq_dict.keys()),
        'total_words': len(freq_dict),
        'metadata': {
            'batch_size': batch_size,
            'total_batches': total_batches
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(final_data, f)
    
    print(f"✅ Processing complete! Saved {len(freq_dict)} words to {output_path}")
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("✅ Cleaned up checkpoint file")
    
    return final_data

# STAGE 2: Apply frequency and embedding data to Spark DataFrame
def apply_word_data_to_spark_df(df, word_data_path: str):
    """
    Apply frequency and embedding data to a Spark DataFrame using broadcast variables.
    
    Args:
        df: Spark DataFrame with a 'word' column
        word_data_path: Path to pickle file created by collect_word_data_with_checkpoint
    
    Returns:
        Spark DataFrame with 'frequency' and 'embedding' columns added
    """
    # Load the word data
    with open(word_data_path, 'rb') as f:
        word_data = pickle.load(f)
    
    freq_dict = word_data['freq_dict']
    embeddings_dict = word_data['embeddings_dict']
    
    print(f"Loaded data for {len(freq_dict)} words from {word_data_path}")
    
    # Create broadcast variables for efficient lookups
    freq_broadcast = spark.sparkContext.broadcast(freq_dict)
    embeddings_broadcast = spark.sparkContext.broadcast(embeddings_dict)
    
    # Define UDFs to add frequency and embeddings
    def get_frequency_udf(word):
        return freq_broadcast.value.get(word, 0)
    
    def get_embedding_udf(word):
        return embeddings_broadcast.value.get(word, [])
    
    # Register UDFs (adjust types based on your actual data)
    frequency_udf = F.udf(get_frequency_udf, IntegerType())
    embedding_udf = F.udf(get_embedding_udf, ArrayType(FloatType()))
    
    # Add the new columns
    result_df = df.withColumn("frequency", frequency_udf(F.col("word"))) \
                 .withColumn("embedding", embedding_udf(F.col("word")))
    
    return result_df

# Execute Stage 1: Collect word data
words_list = [row.word for row in spark_df.select("word").collect()]
word_data_path = get_local_path(f"{WORDLIST_PATH}/word_data.pkl")

# First run - set resume_job=False
word_data = collect_word_data_with_checkpoint(
    words_list, 
    word_data_path,
    batch_size=100,
    resume_job=False
)

# To resume if it crashes - set resume_job=True
# word_data = collect_word_data_with_checkpoint(
#     words_list, 
#     word_data_path,
#     batch_size=100,
#     resume_job=True
# )

# Execute Stage 2: Apply word data to DataFrame
enriched_df = apply_word_data_to_spark_df(spark_df, word_data_path)

# Step 3: Validate embeddings (no nulls)
null_elements_df = enriched_df.filter(F.expr("exists(embedding, x -> x IS NULL)"))
count_null_elements = null_elements_df.count()

if count_null_elements > 0:
    raise Exception("Source data has null values in its embeddings")
else:
    print("✅ No null values in any embeddings")

# Step 4: Apply final schema
def apply_schema(df, new_schema):
    current_schema = {field.name: field.dataType for field in df.schema.fields}
    exprs = []
    for field in new_schema.fields:
        if field.name in current_schema:
            current_type = current_schema[field.name]
            target_type = field.dataType

            if current_type == target_type:
                exprs.append(F.col(field.name).alias(field.name))
            else:
                exprs.append(F.col(field.name).cast(target_type).alias(field.name))
        else:
            exprs.append(F.lit(None).cast(field.dataType).alias(field.name))
    
    return df.select(*exprs)

final_df = apply_schema(enriched_df, words_schema)

# Step 5: Create the table
create_unpartitioned_table(spark, final_df, _TARGET_TABLE_NAME, _TARGET_DB_NAME)

print(f"✅ Successfully created bootstrapped words table {_TARGET_DB_NAME}.{_TARGET_TABLE_NAME}")

# Step 6: Cleanup temp files
if os.path.exists(temp_path):
    os.remove(temp_path)
    print("✅ Cleaned up temporary CSV file")

# Note: Keeping word_data.pkl for potential reuse in other projects