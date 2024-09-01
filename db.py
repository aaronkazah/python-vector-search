import os
import pickle
import faiss
import numpy as np
from typing import List


class Vector:
    def __init__(self, id: str, text: str, embeddings: np.ndarray, **attributes):
        self.id = id
        self.text = text
        self.embeddings = embeddings
        self.attributes = attributes

    def __repr__(self):
        return f"Vector(id={self.id}, text={self.text[:20]}..., attributes={self.attributes})"


class VectorDB:
    """Faiss implementation of a vector database"""

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "Flat",
        collections_name: str = "default",
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = self._create_index()
        self.vectors = []
        self.id_map = {}
        self.collections_name = collections_name
        self.index_file = f".bin/{collections_name}_index.bin"
        self.data_file = f".bin/{collections_name}_data.pkl"
        os.makedirs(".bin", exist_ok=True)  # Ensure .bin directory exists

    def _create_index(self):
        if self.index_type == "Flat":
            return faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "IVF":
            nlist = 10
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            return faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        elif self.index_type == "HNSW":
            return faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def create(self, id: str, text: str, embeddings: np.ndarray, **attributes) -> None:
        if self.index_type in ["IVF", "HNSW"]:
            self.index.train(embeddings[np.newaxis, :])  # Train the index if needed
        self.index.add(embeddings[np.newaxis, :])  # Add a single vector to the index
        vector = Vector(id=id, text=text, embeddings=embeddings, **attributes)
        self.vectors.append(vector)
        self.id_map[len(self.vectors) - 1] = vector

    def search(
        self, query_vector: np.ndarray, top_k: int = 5, log_time: bool = False
    ) -> List[Vector]:
        if log_time:
            import time

            start_time = time.time()

        distances, indices = self.index.search(query_vector[np.newaxis, :], top_k)

        if log_time:
            elapsed_time = time.time() - start_time
            print(f"Search time: {elapsed_time:.4f} seconds")

        return [self.id_map[i] for i in indices[0] if i != -1]

    def save(self) -> None:
        faiss.write_index(self.index, self.index_file)
        with open(self.data_file, "wb") as f:
            pickle.dump(self.vectors, f)

    def load(self) -> None:
        self.index = faiss.read_index(self.index_file)
        with open(self.data_file, "rb") as f:
            self.vectors = pickle.load(f)
        self.id_map = {i: v for i, v in enumerate(self.vectors)}
