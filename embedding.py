import numpy as np

from fastembed import TextEmbedding


class Embedder:
    def __init__(self):
        self.embedding_model = TextEmbedding()
        print("The model BAAI/bge-small-en-v1.5 is ready to use.")

    def embed(self, text: str) -> np.ndarray:
        embedding_generator = self.embedding_model.embed([text])
        return np.array(next(embedding_generator), dtype=np.float32)
