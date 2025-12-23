"""Search indexer for code units.

Handles indexing of code units into the search engine.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smart_search.embedding.jina_embedder import BaseEmbedder
from smart_search.embedding.pipeline import EmbeddingPipeline
from smart_search.parsing.models import CodeUnit
from smart_search.search.meilisearch_client import MeilisearchClient
from smart_search.search.schemas import IndexDocument, IndexStats
from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IndexingConfig:
    """Configuration for indexing.

    Attributes:
        batch_size: Number of documents per batch.
        include_embeddings: Whether to include embeddings.
        max_content_length: Maximum content length to index.
        parallel_embedding: Whether to parallelize embedding.
    """

    batch_size: int = 50
    include_embeddings: bool = True
    max_content_length: int = 10000
    parallel_embedding: bool = True


@dataclass
class IndexingResult:
    """Result of indexing operation.

    Attributes:
        total_indexed: Total documents indexed.
        total_updated: Total documents updated.
        total_deleted: Total documents deleted.
        total_skipped: Total documents skipped.
        errors: List of errors encountered.
        duration_ms: Duration in milliseconds.
    """

    total_indexed: int = 0
    total_updated: int = 0
    total_deleted: int = 0
    total_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        """Whether indexing was successful."""
        return len(self.errors) == 0


class SearchIndexer:
    """Indexes code units into search engine."""

    def __init__(
        self,
        client: MeilisearchClient,
        embedder: BaseEmbedder | None = None,
        config: IndexingConfig | None = None,
    ) -> None:
        """Initialize indexer.

        Args:
            client: Meilisearch client.
            embedder: Optional embedder for vector search.
            config: Indexing configuration.
        """
        self.client = client
        self.embedder = embedder
        self.config = config or IndexingConfig()
        self._pipeline: EmbeddingPipeline | None = None

        if embedder:
            self._pipeline = EmbeddingPipeline(embedder)

    async def index_units(
        self,
        units: list[CodeUnit],
        update: bool = False,
    ) -> IndexingResult:
        """Index code units.

        Args:
            units: Code units to index.
            update: Whether to update existing documents.

        Returns:
            IndexingResult.
        """
        import time

        start = time.time()
        result = IndexingResult()

        if not units:
            return result

        try:
            # Convert to index documents
            documents = await self._prepare_documents(units)

            # Index in batches
            for i in range(0, len(documents), self.config.batch_size):
                batch = documents[i : i + self.config.batch_size]

                if update:
                    count = await self.client.update_documents(batch)
                    result.total_updated += count
                else:
                    count = await self.client.add_documents(batch)
                    result.total_indexed += count

                logger.debug(
                    "Indexed batch",
                    batch_size=len(batch),
                    total=result.total_indexed + result.total_updated,
                )

        except Exception as e:
            error_msg = f"Indexing failed: {str(e)}"
            result.errors.append(error_msg)
            logger.error(error_msg, exc_info=e)

        result.duration_ms = (time.time() - start) * 1000

        logger.info(
            "Indexing complete",
            indexed=result.total_indexed,
            updated=result.total_updated,
            errors=len(result.errors),
            duration_ms=result.duration_ms,
        )

        return result

    async def _prepare_documents(
        self,
        units: list[CodeUnit],
    ) -> list[IndexDocument]:
        """Prepare documents for indexing.

        Args:
            units: Code units.

        Returns:
            List of IndexDocuments.
        """
        documents = []
        embeddings: dict[str, list[float]] = {}

        # Generate embeddings if configured
        if self.config.include_embeddings and self._pipeline:
            embeddings = await self._generate_embeddings(units)

        for unit in units:
            # Truncate content if needed
            content = unit.content
            if len(content) > self.config.max_content_length:
                content = content[: self.config.max_content_length] + "..."

            doc = IndexDocument(
                id=unit.id,
                name=unit.name,
                qualified_name=unit.qualified_name,
                code_type=unit.type.value,
                file_path=str(unit.file_path),
                line_start=unit.span.start.line,
                line_end=unit.span.end.line,
                content=content,
                language=unit.language.value,
                signature=unit.signature or "",
                docstring=unit.docstring or "",
                embedding=embeddings.get(unit.id),
                metadata={
                    "decorators": unit.decorators,
                    "base_classes": unit.base_classes,
                    "parent_id": unit.parent_id,
                },
            )
            documents.append(doc)

        return documents

    async def _generate_embeddings(
        self,
        units: list[CodeUnit],
    ) -> dict[str, list[float]]:
        """Generate embeddings for units.

        Args:
            units: Code units.

        Returns:
            Dict mapping unit ID to embedding.
        """
        if not self._pipeline:
            return {}

        embeddings: dict[str, list[float]] = {}

        try:
            # Prepare texts for embedding
            texts = []
            ids = []
            for unit in units:
                # Create embedding text with context
                text_parts = []
                if unit.signature:
                    text_parts.append(unit.signature)
                if unit.docstring:
                    text_parts.append(unit.docstring)
                text_parts.append(unit.content[:2000])  # Limit content

                texts.append(" ".join(text_parts))
                ids.append(unit.id)

            # Generate embeddings
            results = await self._pipeline.embed_texts(texts)

            for unit_id, result in zip(ids, results):
                embeddings[unit_id] = result.embedding

        except Exception as e:
            logger.warning("Embedding generation failed", error=str(e))

        return embeddings

    async def delete_by_file(self, file_path: Path) -> int:
        """Delete all documents for a file.

        Args:
            file_path: File path.

        Returns:
            Number of deleted documents.
        """
        try:
            filter_str = f'file_path = "{str(file_path)}"'
            await self.client.delete_by_filter(filter_str)
            logger.info("Deleted documents for file", file_path=str(file_path))
            return 1  # Meilisearch doesn't return count
        except Exception as e:
            logger.error("Delete failed", file_path=str(file_path), error=str(e))
            return 0

    async def delete_by_ids(self, ids: list[str]) -> int:
        """Delete documents by ID.

        Args:
            ids: Document IDs.

        Returns:
            Number of deleted documents.
        """
        return await self.client.delete_documents(ids)

    async def reindex_file(
        self,
        file_path: Path,
        units: list[CodeUnit],
    ) -> IndexingResult:
        """Reindex a file by deleting and re-adding.

        Args:
            file_path: File path.
            units: New code units.

        Returns:
            IndexingResult.
        """
        # Delete existing documents
        await self.delete_by_file(file_path)

        # Index new units
        return await self.index_units(units)

    async def get_stats(self) -> IndexStats:
        """Get indexing statistics.

        Returns:
            IndexStats.
        """
        return await self.client.get_stats()

    async def clear_index(self) -> None:
        """Clear all documents from index."""
        await self.client.initialize_index(recreate=True)
        logger.info("Index cleared")

    async def close(self) -> None:
        """Close resources."""
        if self._pipeline:
            await self._pipeline.close()
        await self.client.close()


class BatchIndexer:
    """Batch indexer for large codebases.

    Handles indexing of many files with progress tracking.
    """

    def __init__(
        self,
        indexer: SearchIndexer,
        batch_size: int = 100,
    ) -> None:
        """Initialize batch indexer.

        Args:
            indexer: Search indexer.
            batch_size: Files per batch.
        """
        self.indexer = indexer
        self.batch_size = batch_size

    async def index_all(
        self,
        units_by_file: dict[Path, list[CodeUnit]],
        on_progress: Any = None,
    ) -> IndexingResult:
        """Index all files.

        Args:
            units_by_file: Dict mapping file path to units.
            on_progress: Optional progress callback.

        Returns:
            Combined IndexingResult.
        """
        import time

        start = time.time()
        total_result = IndexingResult()
        files = list(units_by_file.keys())
        total_files = len(files)

        for i in range(0, total_files, self.batch_size):
            batch_files = files[i : i + self.batch_size]
            batch_units = []

            for file_path in batch_files:
                batch_units.extend(units_by_file[file_path])

            result = await self.indexer.index_units(batch_units)

            total_result.total_indexed += result.total_indexed
            total_result.total_updated += result.total_updated
            total_result.errors.extend(result.errors)

            if on_progress:
                progress = min(i + self.batch_size, total_files) / total_files
                on_progress(progress, total_result)

            logger.info(
                "Batch complete",
                progress=f"{min(i + self.batch_size, total_files)}/{total_files}",
            )

        total_result.duration_ms = (time.time() - start) * 1000
        return total_result

    async def reindex_changed(
        self,
        changed_files: dict[Path, list[CodeUnit]],
        deleted_files: list[Path],
    ) -> IndexingResult:
        """Reindex changed and deleted files.

        Args:
            changed_files: Changed files with new units.
            deleted_files: Files that were deleted.

        Returns:
            IndexingResult.
        """
        import time

        start = time.time()
        result = IndexingResult()

        # Delete removed files
        for file_path in deleted_files:
            await self.indexer.delete_by_file(file_path)
            result.total_deleted += 1

        # Reindex changed files
        for file_path, units in changed_files.items():
            file_result = await self.indexer.reindex_file(file_path, units)
            result.total_indexed += file_result.total_indexed
            result.errors.extend(file_result.errors)

        result.duration_ms = (time.time() - start) * 1000

        logger.info(
            "Incremental reindex complete",
            indexed=result.total_indexed,
            deleted=result.total_deleted,
            duration_ms=result.duration_ms,
        )

        return result
