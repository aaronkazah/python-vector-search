import time
import unittest
import faiss
import numpy as np
from db import Vector, VectorDB
from embedding import Embedder


class TestVectorDB(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize the embedder once
        cls.embedder = Embedder()

    def setUp(self):
        # Initialize VectorDB with embedding dimension of 384
        self.db = VectorDB(
            embedding_dim=384, index_type="Flat", name="test_db"
        )

        # Example phrases and their embeddings
        self.phrases = [
            "Artificial intelligence is transforming the world.",
            "Machine learning algorithms are powerful tools.",
            "Natural language processing enables computers to understand text.",
            "Deep learning models can handle large datasets.",
            "Reinforcement learning optimizes decision-making through rewards.",
        ]
        self.embeddings = [self.embed_text(phrase) for phrase in self.phrases]

        # Add phrases to the database
        for i, phrase in enumerate(self.phrases):
            self.db.create(id=f"phrase_{i}", text=phrase, embeddings=self.embeddings[i])

    def embed_text(self, text: str) -> np.ndarray:
        return self.__class__.embedder.embed(text)

    def test_create_and_search(self):
        query_phrase = "Machine learning algorithms are tools."
        query_embedding = self.embed_text(query_phrase)
        results = self.db.search(query_vector=query_embedding, top_k=3)

        self.assertGreater(len(results), 0, "No results found for the query.")
        self.assertIsInstance(
            results[0], Vector, "Search results should be instances of Vector."
        )

    def test_save_and_load(self):
        self.db.save()

        # Initialize a new database instance
        new_db = VectorDB(
            embedding_dim=384, index_type="Flat", name="test_db"
        )
        new_db.load()

        # Test that the data was loaded correctly
        self.assertEqual(
            len(new_db.vectors),
            len(self.phrases),
            "The number of vectors loaded does not match the number of saved vectors.",
        )
        self.assertEqual(
            len(new_db.id_map),
            len(self.phrases),
            "The id_map size does not match the number of saved vectors.",
        )

    def test_empty_db(self):
        empty_db = VectorDB(
            embedding_dim=384, index_type="Flat", name="test_db"
        )
        query_phrase = "Some random text."
        query_embedding = self.embed_text(query_phrase)
        results = empty_db.search(query_vector=query_embedding, top_k=3)

        self.assertEqual(
            len(results), 0, "Search results should be empty for an empty database."
        )

    def test_invalid_query_embedding(self):
        db_with_data = VectorDB(
            embedding_dim=384, index_type="Flat", name="test_db"
        )
        valid_embedding = self.embed_text("Test phrase")
        db_with_data.create(id="test", text="Test phrase", embeddings=valid_embedding)

        # Query with incorrect dimension
        invalid_query_embedding = np.random.rand(128).astype(np.float32)

        with self.assertRaises(Exception):
            db_with_data.search(query_vector=invalid_query_embedding, top_k=3)

    def test_index_type(self):
        db_flat = VectorDB(
            embedding_dim=384, index_type="Flat", name="flat_test"
        )
        db_ivf = VectorDB(
            embedding_dim=384, index_type="IVF", name="ivf_test"
        )
        db_hnsw = VectorDB(
            embedding_dim=384, index_type="HNSW", name="hnsw_test"
        )

        self.assertIsInstance(
            db_flat.index,
            faiss.IndexFlatL2,
            "Index type should be IndexFlatL2 for 'Flat'.",
        )
        self.assertIsInstance(
            db_ivf.index,
            faiss.IndexIVFFlat,
            "Index type should be IndexIVFFlat for 'IVF'.",
        )
        self.assertIsInstance(
            db_hnsw.index,
            faiss.IndexHNSWFlat,
            "Index type should be IndexHNSWFlat for 'HNSW'.",
        )

    def test_search_performance(self):
        """Test search performance with varying numbers of embeddings and report average search time per iteration"""
        scales = [
            1_000,
            10_000,
            100_000,
            1_000_000,
        ]  # Number of embeddings to add in each test
        num_searches = 1_000  # Number of searches to perform for each scale

        for scale in scales:

            # Reinitialize the database for each scale
            self.db = VectorDB(
                embedding_dim=384,
                index_type="Flat",
                name=f"performance_test_{scale}",
            )

            print(f"Generating with {scale} embeddings...")
            upload_time_start = time.time()
            # Generate and add embeddings
            for i in range(scale):
                text = f"Sample text {i}"
                embedding = np.random.rand(384).astype(np.float32)
                self.db.create(id=f"id_{i}", text=text, embeddings=embedding)
            upload_time = time.time() - upload_time_start
            print(f"Upload time for {scale} embeddings: {upload_time * 1000:.2f} ms")
            print(f"Searching with {scale} embeddings...")
            # Generate query embedding
            query_embedding = self.embed_text("Sample query text.")

            # Perform searches and measure time
            search_times = []
            for _ in range(num_searches):
                start_time = time.time()
                self.db.search(query_vector=query_embedding, top_k=5)
                search_time = time.time() - start_time
                search_times.append(search_time)

            # Calculate and print average search time
            average_search_time = np.mean(search_times)
            print(
                f"Scale {scale}: Average search time for {num_searches} searches: {average_search_time * 1000:.2f} ms"
            )

            # Check that average search time is within an acceptable range (e.g., less than 50ms)
            self.assertLess(
                average_search_time,
                0.05,
                f"Average search time exceeds 50ms for scale {scale}.",
            )


if __name__ == "__main__":
    unittest.main()
