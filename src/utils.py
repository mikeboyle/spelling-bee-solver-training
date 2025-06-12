import json
import os
from pathlib import Path
from typing import Any

from src.constants import MOUNT_POINT

def is_databricks_env():
    """Check if we're running in Databricks"""
    return 'DATABRICKS_RUNTIME_VERSION' in os.environ

def get_local_path(filepath: str) -> str:
    """Convert paths to DBFS local paths when in Databricks"""
    if is_databricks_env():
        filepath = Path("/dbfs") / MOUNT_POINT.lstrip("/") / filepath.lstrip("/")
        return filepath
    
    else:
        # Not in Databricks, return with path to local data storage
        filepath = os.path.join("data", filepath)
        return filepath

def dump_json_to_file(data: Any,
                      filepath: str,
                      sort_keys: bool = True,
                      indent: int = 4) -> None:
    """
    Creates a new file, overwrites existing file with same name
    """
    fp = get_local_path(filepath)
    with open(fp, 'w') as f:
        json_data = json.dumps(data, sort_keys=sort_keys, indent=indent)
        f.write(json_data)
        f.write("\n") # add newline at end

def dump_word_list_to_file(words: list[str], filepath: str) -> None:
    """
    Creates a new file, overwrites existing file with same name
    """
    with open(filepath, 'w') as f:
        for word in words[:-1]:
            f.write(f"{word}\n")
        
        f.write(words[-1])

def word_file_to_set(filepath: str) -> set[str]:
    """
    Reads in a .txt file (one word per line)
    and returns a set of the words in the file
    """
    output = set()
    with open(filepath) as f:
        for line in f.readlines():
            line = line.strip()
            # Strip quotation marks around words.
            # Wordnik's list puts each word in double quotation marks,
            # which become part of the string when read in from the
            # source .txt file.
            start = 0
            end = len(line)
            if line[0] == "\"" or line[0] == "'":
                start += 1
            if line[-1] == "\"" or line[-1] == "'":
                end -= 1
            
            word = line[start:end]
            output.add(word)
    
    return output
