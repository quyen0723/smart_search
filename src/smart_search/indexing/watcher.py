"""File system watcher for detecting changes.

Monitors directories for file changes and triggers callbacks.
"""

import asyncio
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


class ChangeType(str, Enum):
    """Type of file change."""

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class FileChange:
    """Represents a file change event.

    Attributes:
        path: Path to the changed file.
        change_type: Type of change.
        timestamp: When the change was detected.
    """

    path: Path
    change_type: ChangeType
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "change_type": self.change_type.value,
            "timestamp": self.timestamp,
        }


@dataclass
class WatcherConfig:
    """Configuration for file watcher.

    Attributes:
        patterns: File patterns to watch (e.g., ["*.py"]).
        exclude_patterns: Patterns to exclude.
        poll_interval: Interval between polls in seconds.
        debounce_delay: Delay before processing changes.
        recursive: Whether to watch subdirectories.
    """

    patterns: list[str] = field(default_factory=lambda: ["*.py"])
    exclude_patterns: list[str] = field(
        default_factory=lambda: ["__pycache__", ".git", ".venv", "node_modules"]
    )
    poll_interval: float = 1.0
    debounce_delay: float = 0.5
    recursive: bool = True


class FileWatcher:
    """Watches directories for file changes using polling.

    Simple polling-based watcher that works across all platforms.
    """

    def __init__(
        self,
        directory: Path,
        config: WatcherConfig | None = None,
    ) -> None:
        """Initialize watcher.

        Args:
            directory: Directory to watch.
            config: Watcher configuration.
        """
        self.directory = directory
        self.config = config or WatcherConfig()
        self._running = False
        self._file_mtimes: dict[Path, float] = {}
        self._callbacks: list[Callable[[list[FileChange]], Any]] = []
        self._pending_changes: list[FileChange] = []
        self._last_process_time = 0.0

    def add_callback(
        self,
        callback: Callable[[list[FileChange]], Any],
    ) -> None:
        """Add a callback for file changes.

        Args:
            callback: Function to call with list of changes.
        """
        self._callbacks.append(callback)

    def remove_callback(
        self,
        callback: Callable[[list[FileChange]], Any],
    ) -> bool:
        """Remove a callback.

        Args:
            callback: Callback to remove.

        Returns:
            True if removed.
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            return True
        return False

    def _should_watch(self, path: Path) -> bool:
        """Check if a path should be watched.

        Args:
            path: Path to check.

        Returns:
            True if should watch.
        """
        # Check excludes
        for exclude in self.config.exclude_patterns:
            if exclude in str(path):
                return False

        # Check patterns
        for pattern in self.config.patterns:
            if path.match(pattern):
                return True

        return False

    def _scan_directory(self) -> dict[Path, float]:
        """Scan directory for files and their mtimes.

        Returns:
            Dict mapping path to mtime.
        """
        mtimes = {}

        if self.config.recursive:
            iterator = self.directory.rglob("*")
        else:
            iterator = self.directory.glob("*")

        for path in iterator:
            if not path.is_file():
                continue

            if not self._should_watch(path):
                continue

            try:
                mtimes[path] = path.stat().st_mtime
            except (OSError, PermissionError):
                pass

        return mtimes

    def _detect_changes(
        self,
        current_mtimes: dict[Path, float],
    ) -> list[FileChange]:
        """Detect changes between scans.

        Args:
            current_mtimes: Current file mtimes.

        Returns:
            List of file changes.
        """
        import time

        changes = []
        now = time.time()
        current_paths = set(current_mtimes.keys())
        previous_paths = set(self._file_mtimes.keys())

        # New files
        for path in current_paths - previous_paths:
            changes.append(
                FileChange(
                    path=path,
                    change_type=ChangeType.CREATED,
                    timestamp=now,
                )
            )

        # Deleted files
        for path in previous_paths - current_paths:
            changes.append(
                FileChange(
                    path=path,
                    change_type=ChangeType.DELETED,
                    timestamp=now,
                )
            )

        # Modified files
        for path in current_paths & previous_paths:
            if current_mtimes[path] != self._file_mtimes[path]:
                changes.append(
                    FileChange(
                        path=path,
                        change_type=ChangeType.MODIFIED,
                        timestamp=now,
                    )
                )

        return changes

    async def _process_changes(self, changes: list[FileChange]) -> None:
        """Process and dispatch changes to callbacks.

        Args:
            changes: List of changes.
        """
        if not changes:
            return

        for callback in self._callbacks:
            try:
                result = callback(changes)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(
                    "Callback error",
                    error=str(e),
                    callback=str(callback),
                )

    async def start(self) -> None:
        """Start watching for changes."""
        self._running = True
        self._file_mtimes = self._scan_directory()

        logger.info(
            "File watcher started",
            directory=str(self.directory),
            file_count=len(self._file_mtimes),
        )

        while self._running:
            await asyncio.sleep(self.config.poll_interval)

            current_mtimes = self._scan_directory()
            changes = self._detect_changes(current_mtimes)

            if changes:
                self._pending_changes.extend(changes)
                self._file_mtimes = current_mtimes

                # Debounce
                import time

                current_time = time.time()
                if current_time - self._last_process_time >= self.config.debounce_delay:
                    await self._process_changes(self._pending_changes)
                    self._pending_changes = []
                    self._last_process_time = current_time

    def stop(self) -> None:
        """Stop watching."""
        self._running = False
        logger.info("File watcher stopped")

    @property
    def is_running(self) -> bool:
        """Whether watcher is running."""
        return self._running

    @property
    def watched_files(self) -> list[Path]:
        """List of currently watched files."""
        return list(self._file_mtimes.keys())


class ChangeCollector:
    """Collects and batches file changes.

    Useful for collecting changes over a period before processing.
    """

    def __init__(self, batch_delay: float = 1.0) -> None:
        """Initialize collector.

        Args:
            batch_delay: Delay before processing batch.
        """
        self.batch_delay = batch_delay
        self._changes: list[FileChange] = []
        self._lock = asyncio.Lock()

    async def add(self, change: FileChange) -> None:
        """Add a change.

        Args:
            change: File change to add.
        """
        async with self._lock:
            # Deduplicate - keep latest change per path
            self._changes = [c for c in self._changes if c.path != change.path]
            self._changes.append(change)

    async def add_many(self, changes: list[FileChange]) -> None:
        """Add multiple changes.

        Args:
            changes: List of changes.
        """
        for change in changes:
            await self.add(change)

    async def get_and_clear(self) -> list[FileChange]:
        """Get all changes and clear.

        Returns:
            List of collected changes.
        """
        async with self._lock:
            changes = self._changes
            self._changes = []
            return changes

    @property
    def count(self) -> int:
        """Number of pending changes."""
        return len(self._changes)

    def get_by_type(self, change_type: ChangeType) -> list[FileChange]:
        """Get changes by type.

        Args:
            change_type: Type of change.

        Returns:
            List of matching changes.
        """
        return [c for c in self._changes if c.change_type == change_type]
