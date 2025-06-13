import os

def is_databricks_env():
    """Check if we're running in Databricks"""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ