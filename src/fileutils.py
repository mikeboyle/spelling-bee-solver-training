import json
import os
from glob import glob
from pathlib import Path
from typing import Any, Optional

from src.constants import (
    DATABRICKS_PATH_PREFIX,
    LOCAL_DATA_LAKE_PATH,
    MOUNT_POINT,
    RAW_SOLUTIONS_PATH,
    WORDLIST_PATH,
)
from src.envutils import is_databricks_env


def get_local_path(filepath: str) -> str:
    """
    Convert path to DBFS local path when in Databricks environment,
    so that we can use local file system methods such as `open` in both local
    and Databricks environments.

    Prepend path to local data lake when in local (non Databricks) env,
    and prepend the mount point in the Databricks environment.

    This allows calling code in either environment to simply use
    relative paths, e.g., "raw/solutions/year=2024".
    """
    if is_databricks_env():
        if filepath.startswith(DATABRICKS_PATH_PREFIX):
            # already converted; just return it
            return filepath

        new_path = (
            Path(DATABRICKS_PATH_PREFIX)
            / MOUNT_POINT.lstrip("/")
            / filepath.lstrip("/")
        )
        return str(new_path)

    else:
        # Not in Databricks, assume we are in local env.
        # Return with path to local data storage

        if filepath.startswith(LOCAL_DATA_LAKE_PATH):
            # already converted; just return it
            return filepath

        new_path = os.path.join(LOCAL_DATA_LAKE_PATH, filepath)
        return new_path


def get_all_files(
    dirpath: str, exts: Optional[list[str]] = None, recursive: bool = True
) -> list[str]:
    """
    Get all files in the given dirpath.
    If `exts` arg is provided, filters file list by given extensions.
    """
    local_dirpath = get_local_path(dirpath)
    file_list = glob(os.path.join(local_dirpath, "**/*"), recursive=recursive)
    if exts is not None:
        exts_set = set()
        for ext in exts:
            if ext.startswith("."):
                exts_set.add(ext)
            else:
                exts_set.add(f".{ext}")

        file_list = [f for f in file_list if "".join(Path(f).suffixes) in exts_set]

    return file_list


def get_puzzle_paths(year: int, month: int) -> list[str]:
    """Get all paths to puzzles for a given year and month"""
    input_path = f"{RAW_SOLUTIONS_PATH}/year={year}/month={month:02}"
    return get_all_files(input_path, [".json"])


def get_puzzle_path(date_str: str) -> str:
    """Takes the YYYY-MM-DD date of a puzzle and returns
    the path to the puzzle solution"""
    filename = f"{date_str}.json"
    year, month, _ = date_str.split("-")
    file_path = f"{RAW_SOLUTIONS_PATH}/year={year}/month={month}/{filename}"

    return get_local_path(file_path)


def get_puzzle_by_date(puzzle_date: str) -> dict[str, Any]:
    """
    Attempts to get a puzzle by date. Will error if no puzzle exists for date
    """
    puzzle_path = get_puzzle_path(puzzle_date)
    with open(puzzle_path, "r") as f:
        puzzle = json.load(f)

    return puzzle


def get_puzzle_by_path(puzzle_path: str) -> dict[str, Any]:
    """
    Attempts to get a puzzle by its path. Will error if path or filename doesn't exist.
    """
    fp = get_local_path(puzzle_path)
    with open(fp, "r") as f:
        puzzle = json.load(f)

    return puzzle


def dump_json_to_file(
    data: Any, filepath: str, sort_keys: bool = True, indent: int = 4
) -> None:
    """
    Creates a new file, overwrites existing file with same name
    """
    fp = get_local_path(filepath)
    with open(fp, "w") as f:
        json_data = json.dumps(data, sort_keys=sort_keys, indent=indent)
        f.write(json_data)
        f.write("\n")  # add newline at end


def dump_word_list_to_file(words: list[str], filepath: str) -> None:
    """
    Creates a new file, overwrites existing file with same name
    """
    fp = get_local_path(filepath)
    with open(fp, "w") as f:
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
            if line[0] == '"' or line[0] == "'":
                start += 1
            if line[-1] == '"' or line[-1] == "'":
                end -= 1

            word = line[start:end]
            output.add(word)

    return output


def get_wordlist_version(wordlist: str) -> int:
    """
    Parse the wordlist filename for the version number.
    Expects filenames to have the format `{filename}_vN.{ext}`
    where N is the version number
    """
    filename = Path(wordlist).stem
    version = filename.split("_")[-1]
    version_num = int(version[1:])

    return version_num


def get_latest_wordlist() -> tuple[str, int]:
    """
    Return the most recent word list and its version number
    """
    wordlists = get_all_files(WORDLIST_PATH, ["txt"])
    wordlist_versions = [
        (wordlist, get_wordlist_version(wordlist)) for wordlist in wordlists
    ]

    return max(wordlist_versions, key=lambda x: x[1])
