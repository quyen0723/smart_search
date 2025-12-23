"""Incremental indexing for efficient updates.

Provides incremental indexing that only processes changed files.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smart_search.indexing.hasher import (
    ContentHasher,
    FileHash,
    HashComparison,
    HashStore,
)
from smart_search.indexing.watcher import ChangeType, FileChange
from smart_search.parsing.models import CodeUnit, ParsedFile
from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IndexState:
    """State of the index for a directory.

    Attributes:
        directory: Root directory being indexed.
        hash_store: Store of file hashes.
        indexed_files: Set of indexed file paths.
        last_index_time: Timestamp of last indexing.
        version: State version for compatibility.
    """

    directory: Path
    hash_store: HashStore = field(default_factory=HashStore)
    indexed_files: set[Path] = field(default_factory=set)
    last_index_time: float = 0.0
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "directory": str(self.directory),
            "hash_store": self.hash_store.to_dict(),
            "indexed_files": [str(p) for p in self.indexed_files],
            "last_index_time": self.last_index_time,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexState":
        """Create from dictionary."""
        return cls(
            directory=Path(data["directory"]),
            hash_store=HashStore.from_dict(data.get("hash_store", {})),
            indexed_files={Path(p) for p in data.get("indexed_files", [])},
            last_index_time=data.get("last_index_time", 0.0),
            version=data.get("version", 1),
        )

    def save(self, state_path: Path) -> None:
        """Save state to file.

        Args:
            state_path: Path to save state.
        """
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.debug("Index state saved", path=str(state_path))

    @classmethod
    def load(cls, state_path: Path) -> "IndexState | None":
        """Load state from file.

        Args:
            state_path: Path to state file.

        Returns:
            IndexState or None if not found.
        """
        if not state_path.exists():
            return None

        try:
            with open(state_path) as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load index state", error=str(e))
            return None


@dataclass
class IncrementalUpdate:
    """Result of incremental update calculation.

    Attributes:
        files_to_index: Files that need indexing.
        files_to_remove: Files to remove from index.
        files_unchanged: Files with no changes.
    """

    files_to_index: list[Path] = field(default_factory=list)
    files_to_remove: list[Path] = field(default_factory=list)
    files_unchanged: list[Path] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Whether there are any changes."""
        return bool(self.files_to_index or self.files_to_remove)

    @property
    def total_changes(self) -> int:
        """Total number of changes."""
        return len(self.files_to_index) + len(self.files_to_remove)


class IncrementalIndexer:
    """Handles incremental indexing of codebases."""

    def __init__(
        self,
        directory: Path,
        state_path: Path | None = None,
        patterns: list[str] | None = None,
    ) -> None:
        """Initialize incremental indexer.

        Args:
            directory: Directory to index.
            state_path: Path to store index state.
            patterns: File patterns to index.
        """
        self.directory = directory
        self.state_path = state_path or directory / ".smart_search" / "index_state.json"
        self.patterns = patterns or ["*.py", "*.js", "*.ts", "*.java", "*.go", "*.rs"]
        self._hasher = ContentHasher()
        self._state: IndexState | None = None

    def load_state(self) -> IndexState:
        """Load or create index state.

        Returns:
            Current index state.
        """
        if self._state is None:
            self._state = IndexState.load(self.state_path)
            if self._state is None:
                self._state = IndexState(directory=self.directory)
        return self._state

    def save_state(self) -> None:
        """Save current state."""
        if self._state:
            self._state.save(self.state_path)

    def calculate_update(self) -> IncrementalUpdate:
        """Calculate what needs to be updated.

        Returns:
            IncrementalUpdate with files to process.
        """
        state = self.load_state()
        update = IncrementalUpdate()

        # Get current file hashes
        current_hashes: dict[Path, FileHash] = {}
        for pattern in self.patterns:
            for file_path in self.directory.rglob(pattern):
                if not file_path.is_file():
                    continue
                if self._should_skip(file_path):
                    continue
                try:
                    current_hashes[file_path] = self._hasher.hash_file(file_path)
                except (OSError, PermissionError):
                    pass

        # Compare with stored hashes
        comparison = state.hash_store.compare(current_hashes)

        update.files_to_index = comparison.added + comparison.modified
        update.files_to_remove = comparison.deleted
        update.files_unchanged = comparison.unchanged

        logger.info(
            "Calculated incremental update",
            to_index=len(update.files_to_index),
            to_remove=len(update.files_to_remove),
            unchanged=len(update.files_unchanged),
        )

        return update

    def _should_skip(self, file_path: Path) -> bool:
        """Check if a file should be skipped.

        Args:
            file_path: Path to check.

        Returns:
            True if should skip.
        """
        skip_patterns = [
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "node_modules",
            ".pytest_cache",
            ".mypy_cache",
        ]
        return any(pattern in str(file_path) for pattern in skip_patterns)

    def mark_indexed(
        self,
        file_paths: list[Path],
        file_hashes: dict[Path, FileHash] | None = None,
    ) -> None:
        """Mark files as indexed.

        Args:
            file_paths: Paths that were indexed.
            file_hashes: Optional pre-computed hashes.
        """
        state = self.load_state()

        for path in file_paths:
            state.indexed_files.add(path)

            # Update hash
            if file_hashes and path in file_hashes:
                state.hash_store.update(file_hashes[path])
            else:
                try:
                    file_hash = self._hasher.hash_file(path)
                    state.hash_store.update(file_hash)
                except (OSError, PermissionError):
                    pass

        import time

        state.last_index_time = time.time()

    def mark_removed(self, file_paths: list[Path]) -> None:
        """Mark files as removed from index.

        Args:
            file_paths: Paths that were removed.
        """
        state = self.load_state()

        for path in file_paths:
            state.indexed_files.discard(path)
            state.hash_store.remove(path)

    def process_changes(
        self,
        changes: list[FileChange],
    ) -> IncrementalUpdate:
        """Process file changes into an update.

        Args:
            changes: List of file changes.

        Returns:
            IncrementalUpdate based on changes.
        """
        update = IncrementalUpdate()

        for change in changes:
            if change.change_type == ChangeType.DELETED:
                update.files_to_remove.append(change.path)
            elif change.change_type in (ChangeType.CREATED, ChangeType.MODIFIED):
                # Check if file matches patterns
                if any(change.path.match(p) for p in self.patterns):
                    update.files_to_index.append(change.path)

        return update

    def reset(self) -> None:
        """Reset index state (force full reindex)."""
        self._state = IndexState(directory=self.directory)
        if self.state_path.exists():
            self.state_path.unlink()
        logger.info("Index state reset")

    @property
    def indexed_count(self) -> int:
        """Number of indexed files."""
        state = self.load_state()
        return len(state.indexed_files)

    @property
    def is_initialized(self) -> bool:
        """Whether index has been initialized."""
        return self.state_path.exists()


class DeltaBuilder:
    """Builds deltas between index states.

    Used for efficient syncing and updates.
    """

    def __init__(self) -> None:
        """Initialize delta builder."""
        self._units_added: list[CodeUnit] = []
        self._units_removed: list[str] = []  # Unit IDs
        self._units_modified: list[CodeUnit] = []

    def add_unit(self, unit: CodeUnit) -> None:
        """Add a new unit.

        Args:
            unit: Unit to add.
        """
        self._units_added.append(unit)

    def remove_unit(self, unit_id: str) -> None:
        """Remove a unit.

        Args:
            unit_id: ID of unit to remove.
        """
        self._units_removed.append(unit_id)

    def modify_unit(self, unit: CodeUnit) -> None:
        """Mark a unit as modified.

        Args:
            unit: Modified unit.
        """
        self._units_modified.append(unit)

    def add_parsed_file(self, parsed_file: ParsedFile) -> None:
        """Add all units from a parsed file.

        Args:
            parsed_file: Parsed file with units.
        """
        for unit in parsed_file.units:
            self.add_unit(unit)

    def remove_file_units(self, file_path: Path, unit_ids: list[str]) -> None:
        """Remove all units from a file.

        Args:
            file_path: File path.
            unit_ids: IDs of units to remove.
        """
        self._units_removed.extend(unit_ids)

    @property
    def has_changes(self) -> bool:
        """Whether there are any changes."""
        return bool(
            self._units_added or self._units_removed or self._units_modified
        )

    @property
    def added_count(self) -> int:
        """Number of added units."""
        return len(self._units_added)

    @property
    def removed_count(self) -> int:
        """Number of removed units."""
        return len(self._units_removed)

    @property
    def modified_count(self) -> int:
        """Number of modified units."""
        return len(self._units_modified)

    def get_added_units(self) -> list[CodeUnit]:
        """Get added units."""
        return self._units_added

    def get_removed_ids(self) -> list[str]:
        """Get IDs of removed units."""
        return self._units_removed

    def get_modified_units(self) -> list[CodeUnit]:
        """Get modified units."""
        return self._units_modified

    def clear(self) -> None:
        """Clear all changes."""
        self._units_added.clear()
        self._units_removed.clear()
        self._units_modified.clear()
