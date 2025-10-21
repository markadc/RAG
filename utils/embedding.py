import ollama

def to_embedding(texts):
    if isinstance(texts, str):
        response = ollama.embeddings(model="bge-m3", prompt=texts)
        return response["embedding"]
    else:
        embeddings = []
        for text in texts:
            response = ollama.embeddings(model="bge-m3", prompt=text)
            embeddings.append(response["embedding"])
        return embeddings