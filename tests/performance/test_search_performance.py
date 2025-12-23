"""Performance tests for search module."""

import pytest
import time
from pathlib import Path
from dataclasses import dataclass


@dataclass
class MockSearchHit:
    """Mock search hit for performance testing."""
    id: str
    score: float
    name: str
    file_path: str
    code_type: str
    language: str
    line_start: int
    line_end: int


@pytest.mark.slow
class TestSearchResultProcessing:
    """Performance tests for search result processing."""

    def test_result_ranking_performance(self):
        """Test result ranking performance."""
        # Create many hits
        hits = [
            MockSearchHit(
                id=f"hit_{i}",
                score=0.5 + (i % 50) * 0.01,
                file_path=f"/src/file_{i}.py",
                name=f"item_{i}",
                code_type="function",
                language="python",
                line_start=i,
                line_end=i + 10,
            )
            for i in range(1000)
        ]

        start = time.perf_counter()

        # Sort by score
        sorted_hits = sorted(hits, key=lambda h: h.score, reverse=True)

        # Take top 50
        top_hits = sorted_hits[:50]

        elapsed = time.perf_counter() - start

        assert len(top_hits) == 50
        assert elapsed < 0.1, f"Ranking took {elapsed:.3f}s"

    def test_result_deduplication_performance(self):
        """Test result deduplication performance."""
        # Create hits with some duplicates
        hits = []
        for i in range(500):
            # Create duplicates for every 10th item
            dup_id = i if i % 10 != 0 else i // 10
            hits.append(
                MockSearchHit(
                    id=f"hit_{dup_id}",
                    score=0.9 - i * 0.001,
                    file_path=f"/src/file_{dup_id}.py",
                    name=f"item_{dup_id}",
                    code_type="function",
                    language="python",
                    line_start=dup_id,
                    line_end=dup_id + 5,
                )
            )

        start = time.perf_counter()

        # Deduplicate by ID
        seen = set()
        unique_hits = []
        for hit in hits:
            if hit.id not in seen:
                seen.add(hit.id)
                unique_hits.append(hit)

        elapsed = time.perf_counter() - start

        assert len(unique_hits) < len(hits)
        assert elapsed < 0.1, f"Deduplication took {elapsed:.3f}s"

    def test_result_aggregation_performance(self):
        """Test aggregating results from multiple sources."""
        # Create results from different sources
        keyword_hits = [
            MockSearchHit(
                id=f"kw_{i}",
                score=0.8 + i * 0.01,
                file_path=f"/src/kw_{i}.py",
                name=f"kw_func_{i}",
                code_type="function",
                language="python",
                line_start=i,
                line_end=i + 5,
            )
            for i in range(100)
        ]

        semantic_hits = [
            MockSearchHit(
                id=f"sem_{i}",
                score=0.85 + i * 0.005,
                file_path=f"/src/sem_{i}.py",
                name=f"sem_func_{i}",
                code_type="function",
                language="python",
                line_start=i,
                line_end=i + 5,
            )
            for i in range(100)
        ]

        start = time.perf_counter()

        # Merge and re-rank
        all_hits = keyword_hits + semantic_hits
        ranked = sorted(all_hits, key=lambda h: h.score, reverse=True)[:50]

        elapsed = time.perf_counter() - start

        assert len(ranked) == 50
        assert elapsed < 0.1, f"Aggregation took {elapsed:.3f}s"

    def test_large_result_set_processing(self):
        """Test processing large result sets."""
        # Create large result set
        hits = [
            MockSearchHit(
                id=f"hit_{i}",
                score=0.99 - i * 0.0001,
                file_path=f"/src/file_{i % 100}.py",
                name=f"func_{i}",
                code_type="function" if i % 2 == 0 else "class",
                language="python",
                line_start=i % 1000,
                line_end=(i % 1000) + 10,
            )
            for i in range(10000)
        ]

        start = time.perf_counter()

        # Filter by code type
        functions = [h for h in hits if h.code_type == "function"]

        # Group by file
        by_file = {}
        for hit in functions:
            if hit.file_path not in by_file:
                by_file[hit.file_path] = []
            by_file[hit.file_path].append(hit)

        # Take top 3 from each file
        final_results = []
        for file_hits in by_file.values():
            sorted_file_hits = sorted(file_hits, key=lambda h: h.score, reverse=True)
            final_results.extend(sorted_file_hits[:3])

        elapsed = time.perf_counter() - start

        assert len(final_results) > 0
        assert elapsed < 0.5, f"Large result processing took {elapsed:.3f}s"


@pytest.mark.slow
class TestSearchIndexOperations:
    """Performance tests for search index operations."""

    def test_document_preparation_performance(self):
        """Test document preparation for indexing."""
        start = time.perf_counter()

        documents = []
        for i in range(1000):
            doc = {
                "id": f"doc_{i}",
                "name": f"func_{i}",
                "qualified_name": f"module.submodule.func_{i}",
                "code_type": "function",
                "file_path": f"/src/module_{i // 100}/file_{i % 100}.py",
                "line_start": i * 10,
                "line_end": i * 10 + 9,
                "content": f"def func_{i}(x, y):\n    return x + y",
                "language": "python",
                "docstring": f"Function {i} documentation.",
            }
            documents.append(doc)

        elapsed = time.perf_counter() - start

        assert len(documents) == 1000
        assert elapsed < 0.5, f"Document preparation took {elapsed:.3f}s"

    def test_batch_splitting_performance(self):
        """Test splitting documents into batches."""
        documents = [{"id": f"doc_{i}"} for i in range(5000)]

        start = time.perf_counter()

        batch_size = 100
        batches = []
        for i in range(0, len(documents), batch_size):
            batches.append(documents[i:i + batch_size])

        elapsed = time.perf_counter() - start

        assert len(batches) == 50
        assert elapsed < 0.1, f"Batch splitting took {elapsed:.3f}s"

    def test_document_filtering_performance(self):
        """Test filtering documents by criteria."""
        documents = [
            {
                "id": f"doc_{i}",
                "language": "python" if i % 3 == 0 else "javascript" if i % 3 == 1 else "go",
                "code_type": "function" if i % 2 == 0 else "class",
                "score": i / 1000,
            }
            for i in range(10000)
        ]

        start = time.perf_counter()

        # Filter by language and type
        filtered = [
            doc for doc in documents
            if doc["language"] == "python" and doc["code_type"] == "function"
        ]

        # Sort by score
        sorted_docs = sorted(filtered, key=lambda d: d["score"], reverse=True)

        # Take top 100
        top_docs = sorted_docs[:100]

        elapsed = time.perf_counter() - start

        assert len(top_docs) <= 100
        assert elapsed < 0.2, f"Document filtering took {elapsed:.3f}s"
