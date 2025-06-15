from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

def get_word_embeddings(words: list[str]) -> dict[str, list[float]]:
    """Gets the BERT pretrained embeddings for a list of words"""
    print(f"getting word embedings for {','.join(words[:10])}...")
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
        embeddings_dict[word] = embedding_np

    return embeddings_dict