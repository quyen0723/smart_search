"""Code chunker for splitting large code units.

Splits large code units into smaller chunks suitable for embedding models
while preserving semantic context.
"""

import hashlib
from typing import Iterator

from smart_search.parsing.models import CodeChunk, CodeUnit
from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


class CodeChunker:
    """Splits code units into chunks for embedding.

    Handles the splitting of large code units that exceed the
    embedding model's context window, while preserving context.
    """

    def __init__(
        self,
        max_chunk_size: int = 6000,
        overlap_size: int = 200,
        min_chunk_size: int = 100,
    ) -> None:
        """Initialize the chunker.

        Args:
            max_chunk_size: Maximum characters per chunk.
            overlap_size: Overlap between consecutive chunks.
            min_chunk_size: Minimum chunk size (don't split if smaller).
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size

    def chunk_unit(self, unit: CodeUnit) -> list[CodeChunk]:
        """Split a code unit into chunks.

        Args:
            unit: The code unit to chunk.

        Returns:
            List of CodeChunk objects.
        """
        content = unit.content

        # If content fits in one chunk, return single chunk
        if len(content) <= self.max_chunk_size:
            return [
                CodeChunk(
                    id=f"{unit.id}::0",
                    unit_id=unit.id,
                    content=content,
                    chunk_index=0,
                    total_chunks=1,
                    context_before=self._build_context(unit),
                    file_path=str(unit.file_path),
                    language=unit.language.value,
                    unit_type=unit.type.value,
                    unit_name=unit.qualified_name,
                )
            ]

        # Split into multiple chunks
        chunks = self._split_content(content, unit)
        return chunks

    def chunk_units(self, units: list[CodeUnit]) -> list[CodeChunk]:
        """Chunk multiple code units.

        Args:
            units: List of code units.

        Returns:
            List of all chunks from all units.
        """
        all_chunks: list[CodeChunk] = []
        for unit in units:
            chunks = self.chunk_unit(unit)
            all_chunks.extend(chunks)

        logger.debug(
            "Chunked units",
            input_units=len(units),
            output_chunks=len(all_chunks),
        )
        return all_chunks

    def _split_content(self, content: str, unit: CodeUnit) -> list[CodeChunk]:
        """Split content into multiple chunks.

        Tries to split at natural boundaries (empty lines, function definitions).

        Args:
            content: The content to split.
            unit: The parent code unit.

        Returns:
            List of chunks.
        """
        chunks: list[CodeChunk] = []
        lines = content.split("\n")

        current_chunk_lines: list[str] = []
        current_size = 0
        chunk_index = 0

        # Pre-calculate context
        base_context = self._build_context(unit)

        for i, line in enumerate(lines):
            line_size = len(line) + 1  # +1 for newline

            # Check if adding this line would exceed max size
            if current_size + line_size > self.max_chunk_size and current_chunk_lines:
                # Create chunk from accumulated lines
                chunk_content = "\n".join(current_chunk_lines)

                # Only create chunk if it meets minimum size
                if len(chunk_content) >= self.min_chunk_size:
                    chunks.append(
                        self._create_chunk(
                            content=chunk_content,
                            unit=unit,
                            chunk_index=chunk_index,
                            context=base_context,
                        )
                    )
                    chunk_index += 1

                # Start new chunk with overlap
                overlap_lines = self._get_overlap_lines(current_chunk_lines)
                current_chunk_lines = overlap_lines + [line]
                current_size = sum(len(l) + 1 for l in current_chunk_lines)
            else:
                current_chunk_lines.append(line)
                current_size += line_size

        # Handle remaining lines
        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            if len(chunk_content) >= self.min_chunk_size or chunk_index == 0:
                chunks.append(
                    self._create_chunk(
                        content=chunk_content,
                        unit=unit,
                        chunk_index=chunk_index,
                        context=base_context,
                    )
                )

        # Update total_chunks count
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        return chunks

    def _create_chunk(
        self,
        content: str,
        unit: CodeUnit,
        chunk_index: int,
        context: str,
    ) -> CodeChunk:
        """Create a CodeChunk object.

        Args:
            content: The chunk content.
            unit: Parent code unit.
            chunk_index: Index of this chunk.
            context: Context string.

        Returns:
            CodeChunk object.
        """
        return CodeChunk(
            id=f"{unit.id}::{chunk_index}",
            unit_id=unit.id,
            content=content,
            chunk_index=chunk_index,
            total_chunks=0,  # Will be updated later
            context_before=context,
            file_path=str(unit.file_path),
            language=unit.language.value,
            unit_type=unit.type.value,
            unit_name=unit.qualified_name,
        )

    def _build_context(self, unit: CodeUnit) -> str:
        """Build context string for a code unit.

        Includes signature, docstring summary, and location info.

        Args:
            unit: The code unit.

        Returns:
            Context string.
        """
        parts: list[str] = []

        # Add location info
        parts.append(f"# File: {unit.file_path}")
        parts.append(f"# {unit.type.value}: {unit.qualified_name}")

        # Add signature for functions/methods
        if unit.signature:
            parts.append(f"# Signature: {unit.signature}")

        # Add docstring summary (first line)
        if unit.docstring:
            first_line = unit.docstring.split("\n")[0].strip()
            if first_line:
                parts.append(f"# {first_line}")

        return "\n".join(parts)

    def _get_overlap_lines(self, lines: list[str]) -> list[str]:
        """Get lines for overlap with next chunk.

        Args:
            lines: Lines from current chunk.

        Returns:
            Lines to include at start of next chunk.
        """
        if not lines:
            return []

        # Calculate how many lines to include based on overlap_size
        total_size = 0
        overlap_lines: list[str] = []

        for line in reversed(lines):
            line_size = len(line) + 1
            if total_size + line_size > self.overlap_size:
                break
            overlap_lines.insert(0, line)
            total_size += line_size

        return overlap_lines


def estimate_token_count(text: str) -> int:
    """Estimate token count for text.

    A rough estimate based on character count.
    Actual tokenization depends on the specific tokenizer.

    Args:
        text: Text to estimate.

    Returns:
        Estimated token count.
    """
    # Rough estimate: ~4 characters per token for code
    return len(text) // 4


class SmartChunker(CodeChunker):
    """Smart chunker that tries to split at semantic boundaries.

    Extends CodeChunker to prefer splitting at:
    - Empty lines
    - Function/method boundaries
    - Class boundaries
    - Block boundaries (try/except, if/else, etc.)
    """

    # Patterns that indicate good split points
    SPLIT_PATTERNS = [
        "\n\n",  # Empty line
        "\ndef ",  # Function definition
        "\nasync def ",  # Async function
        "\nclass ",  # Class definition
        "\n    def ",  # Method definition
        "\n    async def ",  # Async method
    ]

    def _split_content(self, content: str, unit: CodeUnit) -> list[CodeChunk]:
        """Split content at semantic boundaries.

        Args:
            content: The content to split.
            unit: The parent code unit.

        Returns:
            List of chunks.
        """
        # First try to find natural split points
        split_points = self._find_split_points(content)

        if not split_points:
            # Fall back to basic line-based splitting
            return super()._split_content(content, unit)

        chunks: list[CodeChunk] = []
        base_context = self._build_context(unit)
        chunk_index = 0
        start = 0

        for split_point in split_points:
            if split_point - start >= self.min_chunk_size:
                chunk_content = content[start:split_point].strip()
                if chunk_content:
                    chunks.append(
                        self._create_chunk(
                            content=chunk_content,
                            unit=unit,
                            chunk_index=chunk_index,
                            context=base_context,
                        )
                    )
                    chunk_index += 1
                start = split_point

        # Handle remaining content
        if start < len(content):
            remaining = content[start:].strip()
            if remaining and len(remaining) >= self.min_chunk_size:
                chunks.append(
                    self._create_chunk(
                        content=remaining,
                        unit=unit,
                        chunk_index=chunk_index,
                        context=base_context,
                    )
                )

        # Update total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        return chunks if chunks else super()._split_content(content, unit)

    def _find_split_points(self, content: str) -> list[int]:
        """Find good positions to split content.

        Args:
            content: The content to analyze.

        Returns:
            List of character positions for splitting.
        """
        split_points: list[int] = []
        current_size = 0
        last_good_split = 0

        for i, char in enumerate(content):
            current_size += 1

            # Check if we're at a potential split point
            for pattern in self.SPLIT_PATTERNS:
                if content[i:].startswith(pattern):
                    # Only use this split point if we've accumulated enough content
                    if current_size >= self.min_chunk_size:
                        # Check if this would make a reasonable chunk
                        if current_size <= self.max_chunk_size:
                            split_points.append(i)
                            current_size = 0
                            last_good_split = i
                    break

            # Force split if we're approaching max size
            if current_size >= self.max_chunk_size - self.overlap_size:
                # Find nearest newline for cleaner split
                next_newline = content.find("\n", i)
                if next_newline != -1 and next_newline - i < 200:
                    split_points.append(next_newline + 1)
                    current_size = 0
                elif last_good_split > split_points[-1] if split_points else 0:
                    split_points.append(last_good_split)
                    current_size = i - last_good_split

        return split_points
