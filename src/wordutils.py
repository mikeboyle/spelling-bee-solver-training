from typing import Any
from collections import defaultdict
from datetime import datetime
import itertools

from src.fileutils import get_puzzle_by_date, get_puzzle_by_path
from src.constants import DATE_FORMAT


def get_word_from_row(row: str | tuple[str, Any]) -> str:
    """Extracts the word from a row in a wordlist.
    Wordlists could be a simple list of strings,
    or a more complex list of tuples"""
    if isinstance(row, str):
        # Assume each "row" is just a word
        # (row from a larger list of words)
        return row
    else:
        # Assume that the first element is the word
        # Tuple from a list of tuples such as
        # (word, probability, source)
        return row[0]


def get_letter_set(word: str) -> str:
    """
    Return the distinct letters of `word` as a sorted and uppercased string
    """
    distinct_letters = set(word)
    return "".join(sorted(distinct_letters)).upper()


def filter_wordlist(words: set[str]) -> set[str]:
    """
    Remove words from list that cannot be Spelling Beee solutions.
    - less than 4 chars long
    - formed by more than 7 distinct letters
    """
    output = set()
    for word in words:
        if len(word) < 4:
            continue

        letter_set = get_letter_set(word)
        if len(letter_set) > 7:
            continue

        output.add(word)

    return output


def get_letter_set_map(wordlist: set[str]) -> dict[str, list[Any]]:
    """Takes a wordlist and groups them by the distinct letter sets
    they are formed from"""
    letter_set_map = defaultdict(list)
    for row in wordlist:
        word = get_word_from_row(row)
        if len(word) < 4:
            continue

        letter_set = get_letter_set(word)
        if len(letter_set) > 7:
            continue

        letter_set_map[letter_set].append(row)

    return letter_set_map


def get_matching_words(
    center_letter: str, outer_letters: str, letter_set_map: dict[str, list[str]]
) -> list[str]:
    """
    Return all words in the word map that can be formed with the given
    center letter and outer letters
    """
    matching_words = []

    for i in range(1, 7):  # number of outer letters to include
        for combination in itertools.combinations(outer_letters, i):
            letter_set = [center_letter] + list(combination)
            letter_set_key = "".join(sorted(letter_set))
            if letter_set_key in letter_set_map:
                matching_words.extend(letter_set_map[letter_set_key])

    return matching_words


def _transform_puzzle_to_word_decisions(
    puzzle: dict[str, Any],
    letter_set_map: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """
    For the given puzzle, wordlist, and letter_set_map:
    - get the date, center / outer letters, and official solution from puzzle
    - find all words in wordlist which _could_ be formed by center / outer letters
    - return row for each word to record the explicit or implicit decision to include
    or exclude the word from the official solution
    """

    puzzle_date = puzzle["printDate"]
    center_letter = puzzle["centerLetter"].upper()
    outer_letters = "".join([letter.upper() for letter in puzzle["outerLetters"]])

    official_solution = set(puzzle["answers"])

    predicted_solution = set(
        get_matching_words(center_letter, outer_letters, letter_set_map)
    )

    all_words = official_solution | predicted_solution

    rows = []
    for word in all_words:
        rows.append(
            {
                "word": word,
                "center_letter": center_letter,
                "outer_letters": outer_letters,
                "puzzle_date": datetime.strptime(puzzle_date, DATE_FORMAT),
                "accepted": float(word in official_solution)
            }
        )

    return rows


def transform_puzzle_to_word_decisions_by_path(
    puzzle_path: str,
    letter_set_map: dict[str, list[Any]],
) -> list[dict[str, Any]]:

    puzzle = get_puzzle_by_path(puzzle_path)

    return _transform_puzzle_to_word_decisions(puzzle, letter_set_map)


def transform_puzzle_to_word_decisions_by_date(
    puzzle_date: str,
    letter_set_map: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """
    Read in puzzle for the given date, finds all possible words in wordlist that
    could be formed from the puzzle's center and outer letters, then returns a
    `word_decision` record for each possible word.
    """
    puzzle = get_puzzle_by_date(puzzle_date)

    return _transform_puzzle_to_word_decisions(puzzle, letter_set_map)
