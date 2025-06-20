import requests
import concurrent.futures
import requests
from time import sleep
from src.constants import NGRAMS_API_BASE

def query_one(word: str) -> requests.Response:
    url = f"{NGRAMS_API_BASE}/search?query={word}&flags=cs"

    return requests.get(url)

def query_batch(words: list[str]) -> requests.Response:
    url = f"{NGRAMS_API_BASE}/batch"
    json_data = {
        "flags": "cs",
        "queries": words
    }

    return requests.post(url, json=json_data)

def get_ngram(ngram_id: str) -> requests.Response:
    url = f"{NGRAMS_API_BASE}/{ngram_id}"

    return requests.get(url)

def get_ngram_latest_frequency(ngram_id: str) -> float:
    """Fetch the latest year frequency of a single word"""
    res = get_ngram(ngram_id)
    body = res.json()
    stats = body['stats']
    latest = stats[-1]
    # print(".", end="") # to show progress
    return float(latest['absMatchCount'])

def get_ngram_latest_frequency_threaded(word_id_tuple: tuple[str, str]) -> tuple[str, float]:
    word, ngram_id = word_id_tuple
    # Add small delay to be respectful to API rate limits
    sleep(0.1) 
    return word, get_ngram_latest_frequency(ngram_id)

def get_ngram_stats_frequencies(ngram_id: str) -> list[float]:
    res = get_ngram(ngram_id)
    body = res.json()
    stats = body['stats']
    return [float(stat["absMatchCount"]) for stat in stats]

def get_word_frequencies(words: list[str]) -> dict[str, float]:
    """Fetches the latest word frequency from ngrams API for a list of words.
    Returns as a dictionary because we may not be able to guarantee that API
    call results are returned in the order of the words in the input list."""
    print(f"get_word_frequencies(), words = {','.join(words[:10])}...")
    frequencies_dict = {}

    res = query_batch(words)
    body = res.json()
    results = body['results']
    print(f"get_word_frequencies(), parsing {len(results)} results...") 
    for result in results:
        word = result['query']
        ngrams = result['ngrams']
        frequencies_dict[word] = 0.0
        if ngrams:
            ngram = ngrams[0]
            ngram_id = ngram['id']
            frequencies_dict[word] = get_ngram_latest_frequency(ngram_id)
        print(".", end="") # to show progress
    print("\n")

    return frequencies_dict

def get_word_frequencies_threaded(words: list[str], max_workers=5):
    """Threaded version of get_word_frequencies"""
    print(f"get_word_frequencies_threaded(), words = {', '.join(words[:10])}...")
    frequencies_dict = {}

    res = query_batch(words)
    body = res.json()
    results = body['results']
    word_id_tuples = []
    for result in results:
        word = result['query']
        ngrams = result['ngrams']
        frequencies_dict[word] = 0.0
        if ngrams:
            ngram = ngrams[0]
            ngram_id = ngram['id']
            word_id_tuples.append((word, ngram_id))
            # frequencies_dict[word] = get_ngram_latest_frequency(ngram_id)
    print(f"get_word_frequencies_threaded(), parsing {len(results)} results...") 

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_word = {executor.submit(get_ngram_latest_frequency_threaded, word_id_tuple): word_id_tuple 
                         for word_id_tuple in word_id_tuples}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_word):
            word, frequency = future.result()
            frequencies_dict[word] = frequency
    
    # print("\n")
    return frequencies_dict



