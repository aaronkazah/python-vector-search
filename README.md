# Python Vector Search

A lightweight, high-performance vector database implementation using the Faiss library. Offers comparable or better speed than leading vector database providers, with less overhead and fewer dependencies. Includes built-in embedding capabilities using FastEmbed.

## Key Features

- **High-Performance Vector Search**: Utilizes Faiss for fast indexing and search.
- **Built-in Embedding**: Includes FastEmbed for generating text embeddings. 
- **Minimal Overhead**: Designed to be lightweight with fewer dependencies.
- **Scalable**: Suitable for handling around 1 million embeddings effectively.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aaronkazah/python-vector-search.git
   cd python-vector-search
   ```

2. Create a virtual environment (optional but recommended):
   ```bash  
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Here's how to use the `VectorDB` class:

```python
from db import VectorDB
from embedding import Embedder

# Initialize the embedder and vector database
embedder = Embedder()
db = VectorDB(embedding_dim=384, index_type='Flat', name='example_db')

# Create vectors 
text = "Sample text for embedding."
embedding = embedder.embed(text)
db.create(id="sample_id", text=text, embeddings=embedding)

# Search for similar vectors
query_embedding = embedder.embed("Sample query text.")
results = db.search(query_vector=query_embedding, top_k=5)

for result in results:
    print(result)
```

## Performance Testing

The vector database has been tested for performance with varying numbers of embeddings. Here are the results of average search times for 1,000 searches:

| Number of Embeddings | Number of Searches | Average Search Time (s) |
|----------------------|--------------------|--------------------------|
| 1,000                | 1,000              | 0.00003  # 0.03 ms       |
| 10,000               | 1,000              | 0.00036  # 0.36 ms       |
| 100,000              | 1,000              | 0.00354  # 3.54 ms       |
| 1,000,000            | 1,000              | 0.03421  # 34.21 ms      |


These results demonstrate the database's ability to handle large-scale searches efficiently.

Upload time for 1,000,000 embeddings is approximately 4.5 seconds (4446.84 ms)

These tests were ran on a Macbook Pro M1 with 32GB of RAM

### Implementation Note

This implementation serves as an example of how you might build a vector database. While FAISS is used here to provide efficient indexing and search capabilities, you can achieve similar performance using just NumPy directly. FAISS is included for convenience and to illustrate how such a system can be integrated, but the core concept remains the same.

### Running Tests
```bash
python -m unittest tests.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.