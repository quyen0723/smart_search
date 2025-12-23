"""Performance tests for embedding module."""

import pytest
import time
import numpy as np


@pytest.mark.slow
class TestEmbeddingResultPerformance:
    """Performance tests for embedding result operations."""

    def test_embedding_creation_performance(self):
        """Test creating many embeddings."""
        start = time.perf_counter()

        embeddings = []
        for i in range(1000):
            embedding = {
                "id": f"text_{i}",
                "vector": np.random.rand(768).tolist(),
                "model": "test-model",
            }
            embeddings.append(embedding)

        elapsed = time.perf_counter() - start

        assert len(embeddings) == 1000
        assert elapsed < 1.0, f"Embedding creation took {elapsed:.3f}s"

    def test_embedding_similarity_performance(self):
        """Test computing embedding similarities."""
        # Create embeddings
        embeddings = [np.random.rand(768) for _ in range(100)]

        start = time.perf_counter()

        # Compute pairwise similarities
        similarities = []
        for i in range(100):
            for j in range(i + 1, 100):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)

        elapsed = time.perf_counter() - start

        # 4950 similarity computations
        assert len(similarities) == 4950
        assert elapsed < 0.5, f"Similarity computation took {elapsed:.3f}s"

    def test_embedding_normalization_performance(self):
        """Test normalizing many embeddings."""
        embeddings = [np.random.rand(768) for _ in range(1000)]

        start = time.perf_counter()

        normalized = []
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            if norm > 0:
                normalized.append(emb / norm)
            else:
                normalized.append(emb)

        elapsed = time.perf_counter() - start

        assert len(normalized) == 1000
        assert elapsed < 0.5, f"Normalization took {elapsed:.3f}s"

    def test_embedding_search_performance(self):
        """Test searching embeddings by similarity."""
        # Create database of embeddings
        database = [np.random.rand(768) for _ in range(1000)]
        # Normalize
        database = [e / np.linalg.norm(e) for e in database]

        # Query embedding
        query = np.random.rand(768)
        query = query / np.linalg.norm(query)

        start = time.perf_counter()

        # Find top-10 similar
        scores = []
        for i, emb in enumerate(database):
            score = np.dot(query, emb)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_10 = scores[:10]

        elapsed = time.perf_counter() - start

        assert len(top_10) == 10
        assert elapsed < 0.1, f"Similarity search took {elapsed:.3f}s"


@pytest.mark.slow
class TestVectorOperationsPerformance:
    """Performance tests for vector operations."""

    def test_batch_dot_product(self):
        """Test batch dot product performance."""
        # Query and database
        query = np.random.rand(768)
        database = np.random.rand(1000, 768)

        start = time.perf_counter()

        # Batch dot product using matrix multiplication
        scores = database @ query

        elapsed = time.perf_counter() - start

        assert len(scores) == 1000
        assert elapsed < 0.1, f"Batch dot product took {elapsed:.3f}s"

    def test_cosine_similarity_batch(self):
        """Test batch cosine similarity."""
        query = np.random.rand(768)
        database = np.random.rand(500, 768)

        # Normalize
        query_norm = query / np.linalg.norm(query)
        db_norms = np.linalg.norm(database, axis=1, keepdims=True)
        database_norm = database / db_norms

        start = time.perf_counter()

        scores = database_norm @ query_norm

        elapsed = time.perf_counter() - start

        assert len(scores) == 500
        assert elapsed < 0.1, f"Cosine similarity took {elapsed:.3f}s"

    def test_top_k_selection(self):
        """Test top-k selection performance."""
        scores = np.random.rand(10000)

        start = time.perf_counter()

        # Get indices of top 100
        top_k_indices = np.argpartition(scores, -100)[-100:]
        top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]

        elapsed = time.perf_counter() - start

        assert len(top_k_indices) == 100
        assert elapsed < 0.01, f"Top-k selection took {elapsed:.3f}s"
