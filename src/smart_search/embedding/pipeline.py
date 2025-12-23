"""Embedding pipeline for processing code chunks.

Orchestrates the embedding of code units and chunks.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any

from smart_search.embedding.jina_embedder import BaseEmbedder
from smart_search.embedding.models import (
    ChunkEmbedding,
    EmbeddingConfig,
    EmbeddingResult,
)
from smart_search.parsing.models import CodeUnit
from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for code chunking.

    Attributes:
        max_chunk_size: Maximum characters per chunk.
        min_chunk_size: Minimum characters per chunk.
        overlap: Character overlap between chunks.
        include_context: Include surrounding context.
    """

    max_chunk_size: int = 2000
    min_chunk_size: int = 100
    overlap: int = 200
    include_context: bool = True


@dataclass
class CodeChunk:
    """A chunk of code for embedding.

    Attributes:
        id: Unique chunk identifier.
        unit_id: Parent code unit ID.
        content: Chunk content.
        start_line: Starting line number.
        end_line: Ending line number.
        context: Optional surrounding context.
        metadata: Additional metadata.
    """

    id: str
    unit_id: str
    content: str
    start_line: int
    end_line: int
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """Get content hash for cache invalidation."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]

    @property
    def full_content(self) -> str:
        """Get full content including context."""
        if self.context:
            return f"{self.context}\n\n{self.content}"
        return self.content


@dataclass
class PipelineResult:
    """Result of embedding pipeline execution.

    Attributes:
        embeddings: List of chunk embeddings.
        total_chunks: Total chunks processed.
        total_tokens: Total tokens embedded.
        skipped: Number of chunks skipped (cached).
        errors: List of errors encountered.
    """

    embeddings: list[ChunkEmbedding]
    total_chunks: int
    total_tokens: int
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


class CodeChunker:
    """Chunks code units into smaller pieces for embedding."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        """Initialize chunker.

        Args:
            config: Chunking configuration.
        """
        self.config = config or ChunkingConfig()

    def chunk_unit(self, unit: CodeUnit) -> list[CodeChunk]:
        """Chunk a code unit.

        Args:
            unit: Code unit to chunk.

        Returns:
            List of code chunks.
        """
        content = unit.content

        # If small enough, return as single chunk
        if len(content) <= self.config.max_chunk_size:
            return [
                CodeChunk(
                    id=f"{unit.id}::chunk_0",
                    unit_id=unit.id,
                    content=content,
                    start_line=unit.span.start.line,
                    end_line=unit.span.end.line,
                    context=self._build_context(unit),
                    metadata={
                        "name": unit.name,
                        "type": unit.type.value,
                        "file": str(unit.file_path),
                    },
                )
            ]

        # Split into chunks
        chunks = []
        lines = content.split("\n")
        current_chunk: list[str] = []
        current_size = 0
        chunk_start_line = unit.span.start.line
        chunk_index = 0

        for i, line in enumerate(lines):
            line_size = len(line) + 1  # +1 for newline

            # Check if adding this line exceeds max size
            if current_size + line_size > self.config.max_chunk_size and current_chunk:
                # Save current chunk
                chunk_content = "\n".join(current_chunk)
                chunks.append(
                    CodeChunk(
                        id=f"{unit.id}::chunk_{chunk_index}",
                        unit_id=unit.id,
                        content=chunk_content,
                        start_line=chunk_start_line,
                        end_line=unit.span.start.line + i - 1,
                        context=self._build_context(unit),
                        metadata={
                            "name": unit.name,
                            "type": unit.type.value,
                            "file": str(unit.file_path),
                            "chunk_index": chunk_index,
                        },
                    )
                )
                chunk_index += 1

                # Start new chunk with overlap
                overlap_lines = self._get_overlap_lines(current_chunk)
                current_chunk = overlap_lines + [line]
                current_size = sum(len(l) + 1 for l in current_chunk)
                chunk_start_line = unit.span.start.line + i - len(overlap_lines)
            else:
                current_chunk.append(line)
                current_size += line_size

        # Save remaining chunk
        if current_chunk:
            chunk_content = "\n".join(current_chunk)
            if len(chunk_content) >= self.config.min_chunk_size:
                chunks.append(
                    CodeChunk(
                        id=f"{unit.id}::chunk_{chunk_index}",
                        unit_id=unit.id,
                        content=chunk_content,
                        start_line=chunk_start_line,
                        end_line=unit.span.end.line,
                        context=self._build_context(unit),
                        metadata={
                            "name": unit.name,
                            "type": unit.type.value,
                            "file": str(unit.file_path),
                            "chunk_index": chunk_index,
                        },
                    )
                )
            elif chunks:
                # Append to previous chunk if too small
                last_chunk = chunks[-1]
                chunks[-1] = CodeChunk(
                    id=last_chunk.id,
                    unit_id=last_chunk.unit_id,
                    content=last_chunk.content + "\n" + chunk_content,
                    start_line=last_chunk.start_line,
                    end_line=unit.span.end.line,
                    context=last_chunk.context,
                    metadata=last_chunk.metadata,
                )

        return chunks if chunks else [
            CodeChunk(
                id=f"{unit.id}::chunk_0",
                unit_id=unit.id,
                content=content,
                start_line=unit.span.start.line,
                end_line=unit.span.end.line,
                context=self._build_context(unit),
                metadata={
                    "name": unit.name,
                    "type": unit.type.value,
                    "file": str(unit.file_path),
                },
            )
        ]

    def chunk_units(self, units: list[CodeUnit]) -> list[CodeChunk]:
        """Chunk multiple code units.

        Args:
            units: Code units to chunk.

        Returns:
            List of all code chunks.
        """
        chunks = []
        for unit in units:
            chunks.extend(self.chunk_unit(unit))
        return chunks

    def _build_context(self, unit: CodeUnit) -> str:
        """Build context string for a code unit.

        Args:
            unit: Code unit.

        Returns:
            Context string.
        """
        if not self.config.include_context:
            return ""

        parts = []

        # Add signature if available
        if unit.signature:
            parts.append(f"Signature: {unit.signature}")

        # Add qualified name
        if unit.qualified_name:
            parts.append(f"Name: {unit.qualified_name}")

        # Add type info
        parts.append(f"Type: {unit.type.value}")

        # Add docstring preview if available
        if unit.docstring:
            doc_preview = unit.docstring[:200]
            if len(unit.docstring) > 200:
                doc_preview += "..."
            parts.append(f"Doc: {doc_preview}")

        return " | ".join(parts)

    def _get_overlap_lines(self, lines: list[str]) -> list[str]:
        """Get overlap lines from current chunk.

        Args:
            lines: Current chunk lines.

        Returns:
            Lines to include in next chunk.
        """
        if not self.config.overlap:
            return []

        overlap_size = 0
        overlap_lines = []

        for line in reversed(lines):
            line_size = len(line) + 1
            if overlap_size + line_size > self.config.overlap:
                break
            overlap_lines.insert(0, line)
            overlap_size += line_size

        return overlap_lines


class EmbeddingPipeline:
    """Pipeline for embedding code chunks."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        chunker: CodeChunker | None = None,
        config: EmbeddingConfig | None = None,
    ) -> None:
        """Initialize pipeline.

        Args:
            embedder: Embedder to use.
            chunker: Code chunker (optional).
            config: Embedding configuration.
        """
        self.embedder = embedder
        self.chunker = chunker or CodeChunker()
        self.config = config or EmbeddingConfig()

    async def embed_units(
        self,
        units: list[CodeUnit],
        cached_hashes: set[str] | None = None,
    ) -> PipelineResult:
        """Embed code units.

        Args:
            units: Code units to embed.
            cached_hashes: Set of content hashes already cached.

        Returns:
            PipelineResult with embeddings.
        """
        cached_hashes = cached_hashes or set()

        # Chunk all units
        all_chunks = self.chunker.chunk_units(units)
        logger.info(
            "Chunked units",
            unit_count=len(units),
            chunk_count=len(all_chunks),
        )

        return await self.embed_chunks(all_chunks, cached_hashes)

    async def embed_chunks(
        self,
        chunks: list[CodeChunk],
        cached_hashes: set[str] | None = None,
    ) -> PipelineResult:
        """Embed code chunks.

        Args:
            chunks: Code chunks to embed.
            cached_hashes: Set of content hashes already cached.

        Returns:
            PipelineResult with embeddings.
        """
        cached_hashes = cached_hashes or set()
        embeddings: list[ChunkEmbedding] = []
        errors: list[str] = []
        skipped = 0
        total_tokens = 0

        # Filter out cached chunks
        chunks_to_embed = []
        for chunk in chunks:
            if chunk.content_hash in cached_hashes:
                skipped += 1
            else:
                chunks_to_embed.append(chunk)

        if not chunks_to_embed:
            return PipelineResult(
                embeddings=[],
                total_chunks=len(chunks),
                total_tokens=0,
                skipped=skipped,
            )

        # Prepare texts for embedding
        texts = [chunk.full_content for chunk in chunks_to_embed]

        try:
            # Embed all texts
            result = await self.embedder.embed_batch(texts)
            total_tokens = result.total_tokens

            # Create chunk embeddings
            for chunk, embed_result in zip(chunks_to_embed, result.results):
                embeddings.append(
                    ChunkEmbedding(
                        chunk_id=chunk.id,
                        unit_id=chunk.unit_id,
                        embedding=embed_result.embedding,
                        content_hash=chunk.content_hash,
                        metadata={
                            **chunk.metadata,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "token_count": embed_result.token_count,
                        },
                    )
                )

        except Exception as e:
            error_msg = f"Embedding failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg, exc_info=e)

        logger.info(
            "Embedding complete",
            total_chunks=len(chunks),
            embedded=len(embeddings),
            skipped=skipped,
            errors=len(errors),
        )

        return PipelineResult(
            embeddings=embeddings,
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            skipped=skipped,
            errors=errors,
        )

    async def embed_text(self, text: str) -> EmbeddingResult:
        """Embed arbitrary text.

        Args:
            text: Text to embed.

        Returns:
            EmbeddingResult.
        """
        return await self.embedder.embed(text)

    async def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed multiple texts.

        Args:
            texts: Texts to embed.

        Returns:
            List of EmbeddingResults.
        """
        result = await self.embedder.embed_batch(texts)
        return result.results

    async def close(self) -> None:
        """Clean up resources."""
        await self.embedder.close()
