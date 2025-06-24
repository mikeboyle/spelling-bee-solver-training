from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
import torch
import concurrent.futures
from typing import Any
import time


def get_word_embedding(args: tuple[str, Any, Any]) -> list[float]:
    word, model, tokenizer = args
    inputs = tokenizer(word, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state
    # Use mean embedding
    embedding = last_hidden_state[0].mean(dim=0)

    # Compose into a single feature vector
    embedding_np = embedding.cpu().detach().numpy()
    return embedding_np.tolist()


def get_word_embeddings_threaded(
    words: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_workers: int = 5,
) -> dict[str, list[float]]:
    """Threaded version of get_word_embeddings"""
    start = time.time()
    print(f"get_word_embeddings_threaded(), words = {', '.join(words[:10])}...")

    embeddings_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_word = {}
        for word in words:
            args = (word, model, tokenizer)
            future = executor.submit(get_word_embedding, args)
            future_to_word[future] = word

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_word):
            word = future_to_word[future]
            embedding = future.result()
            embeddings_dict[word] = embedding
    
    print(f"get_word_embeddings_threaded() time: {time.time() - start:.2f}s")

    return embeddings_dict


def get_word_embeddings(words: list[str]) -> dict[str, list[float]]:
    """Gets the BERT pretrained embeddings for a list of words"""
    print(f"getting word embeddings for {', '.join(words[:10])}...")
    # Load pre-trained model and tokenizer
    model = AutoModel.from_pretrained("bert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    embeddings_dict = {}

    for word in words:
        # Get the mean embedding of all subtoken embeddings
        inputs = tokenizer(word, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_state = outputs.last_hidden_state
        # Use mean embedding
        embedding = last_hidden_state[0].mean(dim=0)

        # Compose into a single feature vector
        embedding_np = embedding.cpu().detach().numpy()
        embeddings_dict[word] = embedding_np.tolist()

    return embeddings_dict
