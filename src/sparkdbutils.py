from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from typing import Any, Optional

from src.constants import MOUNT_POINT
from src.envutils import is_databricks_env


def get_spark_db_location_path(relative_path: str) -> str:
    """
    Get the correct path for the current environment to
    set the `LOCATION` in a `CREATE DATABASE` statement.

    Takes a relative path (where you want the db location
    to be relative to the root of your local data lake or
    relative to the Databricks mount point).

    On local, the relative path is fine as is. Spark will interpret
    that as a path relative to the spark warehouse dir.

    On databricks, this must be an absolute path from the
    mount point. (Not from `/dbfs/mnt/...` but from `/mnt/...`)
    """
    if is_databricks_env():
        new_path = Path(MOUNT_POINT) / relative_path.lstrip("/")
        return str(new_path)

    return relative_path


def get_or_create_db(spark: SparkSession, db_name: str) -> Optional[DataFrame]:
    """
    If db does not already exists, creates it at the root
    of the data lake / storage location
    """
    db_location = get_spark_db_location_path(db_name)

    if not spark.catalog.databaseExists(db_name):
        print(f"creating database {db_name}...")
        return spark.sql(f"CREATE DATABASE {db_name} LOCATION '{db_location}'")


def write_to_table(
    spark: SparkSession,
    df: DataFrame,
    db_name: str,
    table_name: str,
    replace_where_dict: dict[str, Any],
    partitions: list[str],
) -> None:

    filters = [f"{column} = {value}" for column, value in replace_where_dict.items()]
    where_clause = " AND ".join(filters)

    if not spark.catalog.tableExists(f"{db_name}.{table_name}"):
        # Write table for first time; ok and necesssary to set the partitions
        # No point to using replaceWhere because we the table doesn't exist yet
        df.write.format("delta").mode("overwrite").partitionBy(partitions).saveAsTable(
            f"{db_name}.{table_name}"
        )
    else:
        # Write to existing table
        # Use replaceWhere to insert to table; DO NOT parition again (will overwrite entire table!)
        df.write.format("delta").mode("overwrite").option(
            "replaceWhere", where_clause
        ).saveAsTable(f"{db_name}.{table_name}")
