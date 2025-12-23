"""Content hashing for change detection.

Provides efficient file and content hashing for incremental indexing.
"""

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FileHash:
    """Hash information for a file.

    Attributes:
        file_path: Path to the file.
        content_hash: SHA-256 hash of file content.
        mtime: File modification time.
        size: File size in bytes.
    """

    file_path: Path
    content_hash: str
    mtime: float
    size: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": str(self.file_path),
            "content_hash": self.content_hash,
            "mtime": self.mtime,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileHash":
        """Create from dictionary."""
        return cls(
            file_path=Path(data["file_path"]),
            content_hash=data["content_hash"],
            mtime=data["mtime"],
            size=data["size"],
        )


@dataclass
class HashComparison:
    """Result of comparing file hashes.

    Attributes:
        added: Files that are new.
        modified: Files that have changed.
        deleted: Files that were removed.
        unchanged: Files with no changes.
    """

    added: list[Path] = field(default_factory=list)
    modified: list[Path] = field(default_factory=list)
    deleted: list[Path] = field(default_factory=list)
    unchanged: list[Path] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Whether any changes were detected."""
        return bool(self.added or self.modified or self.deleted)

    @property
    def change_count(self) -> int:
        """Total number of changes."""
        return len(self.added) + len(self.modified) + len(self.deleted)


class ContentHasher:
    """Computes content hashes for files."""

    CHUNK_SIZE = 65536  # 64KB chunks for hashing

    def __init__(self, algorithm: str = "sha256") -> None:
        """Initialize hasher.

        Args:
            algorithm: Hash algorithm to use.
        """
        self.algorithm = algorithm

    def hash_content(self, content: str | bytes) -> str:
        """Hash string or bytes content.

        Args:
            content: Content to hash.

        Returns:
            Hex digest of hash.
        """
        if isinstance(content, str):
            content = content.encode("utf-8")

        hasher = hashlib.new(self.algorithm)
        hasher.update(content)
        return hasher.hexdigest()

    def hash_file(self, file_path: Path) -> FileHash:
        """Hash a file.

        Args:
            file_path: Path to file.

        Returns:
            FileHash with content hash and metadata.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        stat = file_path.stat()
        hasher = hashlib.new(self.algorithm)

        with open(file_path, "rb") as f:
            while chunk := f.read(self.CHUNK_SIZE):
                hasher.update(chunk)

        return FileHash(
            file_path=file_path,
            content_hash=hasher.hexdigest(),
            mtime=stat.st_mtime,
            size=stat.st_size,
        )

    def hash_files(self, file_paths: list[Path]) -> dict[Path, FileHash]:
        """Hash multiple files.

        Args:
            file_paths: List of file paths.

        Returns:
            Dict mapping path to FileHash.
        """
        hashes = {}
        for path in file_paths:
            try:
                hashes[path] = self.hash_file(path)
            except (FileNotFoundError, PermissionError) as e:
                logger.warning("Failed to hash file", path=str(path), error=str(e))
        return hashes

    def quick_hash(self, file_path: Path) -> str:
        """Quick hash using mtime and size (no content read).

        Args:
            file_path: Path to file.

        Returns:
            Hash string based on metadata.
        """
        stat = file_path.stat()
        data = f"{file_path}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.md5(data.encode()).hexdigest()


class HashStore:
    """Stores and compares file hashes."""

    def __init__(self) -> None:
        """Initialize hash store."""
        self._hashes: dict[Path, FileHash] = {}

    def update(self, file_hash: FileHash) -> None:
        """Update hash for a file.

        Args:
            file_hash: FileHash to store.
        """
        self._hashes[file_hash.file_path] = file_hash

    def update_many(self, hashes: dict[Path, FileHash]) -> None:
        """Update multiple hashes.

        Args:
            hashes: Dict of path to FileHash.
        """
        self._hashes.update(hashes)

    def get(self, file_path: Path) -> FileHash | None:
        """Get hash for a file.

        Args:
            file_path: Path to file.

        Returns:
            FileHash or None if not found.
        """
        return self._hashes.get(file_path)

    def remove(self, file_path: Path) -> bool:
        """Remove hash for a file.

        Args:
            file_path: Path to file.

        Returns:
            True if removed.
        """
        if file_path in self._hashes:
            del self._hashes[file_path]
            return True
        return False

    def has(self, file_path: Path) -> bool:
        """Check if hash exists for file.

        Args:
            file_path: Path to file.

        Returns:
            True if hash exists.
        """
        return file_path in self._hashes

    def clear(self) -> None:
        """Clear all hashes."""
        self._hashes.clear()

    def compare(
        self,
        current_hashes: dict[Path, FileHash],
    ) -> HashComparison:
        """Compare current hashes with stored hashes.

        Args:
            current_hashes: Current file hashes.

        Returns:
            HashComparison with changes.
        """
        comparison = HashComparison()
        current_paths = set(current_hashes.keys())
        stored_paths = set(self._hashes.keys())

        # New files
        for path in current_paths - stored_paths:
            comparison.added.append(path)

        # Deleted files
        for path in stored_paths - current_paths:
            comparison.deleted.append(path)

        # Check for modifications
        for path in current_paths & stored_paths:
            current = current_hashes[path]
            stored = self._hashes[path]

            if current.content_hash != stored.content_hash:
                comparison.modified.append(path)
            else:
                comparison.unchanged.append(path)

        return comparison

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "hashes": {
                str(path): fh.to_dict() for path, fh in self._hashes.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HashStore":
        """Create from dictionary.

        Args:
            data: Dictionary data.

        Returns:
            HashStore instance.
        """
        store = cls()
        for path_str, hash_data in data.get("hashes", {}).items():
            file_hash = FileHash.from_dict(hash_data)
            store._hashes[Path(path_str)] = file_hash
        return store

    @property
    def count(self) -> int:
        """Number of stored hashes."""
        return len(self._hashes)

    @property
    def paths(self) -> list[Path]:
        """List of all stored paths."""
        return list(self._hashes.keys())


def compute_directory_hash(
    directory: Path,
    patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> dict[Path, FileHash]:
    """Compute hashes for all files in a directory.

    Args:
        directory: Directory to hash.
        patterns: File patterns to include (e.g., ["*.py"]).
        exclude_patterns: Patterns to exclude.

    Returns:
        Dict mapping path to FileHash.
    """
    hasher = ContentHasher()
    hashes = {}

    patterns = patterns or ["*"]
    exclude_patterns = exclude_patterns or []

    # Default excludes
    default_excludes = [
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        ".pytest_cache",
        "*.pyc",
        "*.pyo",
    ]
    exclude_patterns.extend(default_excludes)

    for pattern in patterns:
        for file_path in directory.rglob(pattern):
            if not file_path.is_file():
                continue

            # Check excludes
            skip = False
            for exclude in exclude_patterns:
                if exclude in str(file_path) or file_path.match(exclude):
                    skip = True
                    break

            if skip:
                continue

            try:
                hashes[file_path] = hasher.hash_file(file_path)
            except (PermissionError, OSError) as e:
                logger.warning(
                    "Failed to hash file",
                    path=str(file_path),
                    error=str(e),
                )

    return hashes
