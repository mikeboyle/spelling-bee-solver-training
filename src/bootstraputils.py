from src.wordutils import get_letter_set, filter_wordlist
from src.fileutils import word_file_to_set
from src.constants import WORDLIST_PATH, RAW_WORDLIST_FILENAME

# filter the wordlist
wordlist = filter_wordlist(word_file_to_set(f"{WORDLIST_PATH}/{RAW_WORDLIST_FILENAME}"))
print(len(wordlist),"words")