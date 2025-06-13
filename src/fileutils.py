import json
import os
from glob import glob
from pathlib import Path
from typing import Any

from src.constants import MOUNT_POINT, LOCAL_DATA_LAKE

def is_databricks_env():
    """Check if we're running in Databricks"""
    return 'DATABRICKS_RUNTIME_VERSION' in os.environ

def get_local_path(filepath: str) -> str:
    """Convert paths to DBFS local paths when in Databricks"""
    if is_databricks_env():
        new_path = Path("/dbfs") / MOUNT_POINT.lstrip("/") / filepath.lstrip("/")
        return str(new_path)
    
    else:
        # Not in Databricks, return with path to local data storage
        new_path = os.path.join(LOCAL_DATA_LAKE, filepath)
        return new_path


def get_puzzle_paths(year: int, month: int) -> list[str]:
    """Get all paths to puzzles for a given year and month"""
    input_path = get_local_path(f"raw/solutions/year={year}/month={month:02}")
    return glob(os.path.join(input_path, "*.json"))

def get_puzzle_path(date_str: str) -> str:
    """Takes the YYYY-MM-DD date of a puzzle and returns
    the path to the puzzle solution"""
    filename = f"{date_str}.json"
    year, month, _ = date_str.split("-")
    file_path = f"raw/solutions/year={year}/month={month}/{filename}"

    return get_local_path(file_path)

def get_puzzle_by_date(puzzle_date: str) -> dict[str, Any]:
    puzzle_path = get_puzzle_path(puzzle_date)
    with open(puzzle_path) as f:
        puzzle = json.load(f)
    
    return puzzle

def get_puzzle_by_path(puzzle_path: str) -> dict[str, Any]:
    fp = get_local_path(puzzle_path)
    with open(fp) as f:
        puzzle = json.load(f)
    
    return puzzle

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
    fp = get_local_path(filepath)
    with open(fp, 'w') as f:
        for word in words[:-1]:
            f.write(f"{word}\n")
        
        f.write(words[-1])

def word_file_to_set(filepath: str) -> set[str]:
    """
    Reads in a .txt file (one word per line)
    and returns a set of the words in the file
    """
    output = set()
    fp = get_local_path(filepath)
    with open(fp) as f:
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
