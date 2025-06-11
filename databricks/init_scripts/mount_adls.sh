#!/bin/bash
/databricks/python/bin/python - <<EOF
from pyspark.dbutils import DBUtils
dbutils = DBUtils(spark)

STORAGE_ACCOUNT_NAME = dbutils.secrets.get(scope="spelling-bee-scope", key="storage-account-name")
STORAGE_ACCOUNT_KEY = dbutils.secrets.get(scope="spelling-bee-scope", key="storage-account-key")
BLOB_CONTAINER = "spelling-bee"
MOUNT_POINT = "/mnt/spelling-bee"

configs = {
  f"fs.azure.account.key.{STORAGE_ACCOUNT_NAME}.blob.core.windows.net": STORAGE_ACCOUNT_KEY
}

if not any(mount.mountPoint == MOUNT_POINT for mount in dbutils.fs.mounts()):
    dbutils.fs.mount(
        source = f"wasbs://{BLOB_CONTAINER}@{STORAGE_ACCOUNT_NAME}.blob.core.windows.net/",
        mount_point = "/mnt/spelling-bee",
        extra_configs = configs
    )
EOF