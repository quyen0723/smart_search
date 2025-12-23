"""Tests for file watcher."""

import asyncio
import time
from pathlib import Path

import pytest

from smart_search.indexing.watcher import (
    ChangeCollector,
    ChangeType,
    FileChange,
    FileWatcher,
    WatcherConfig,
)


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_values(self) -> None:
        """Test enum values."""
        assert ChangeType.CREATED.value == "created"
        assert ChangeType.MODIFIED.value == "modified"
        assert ChangeType.DELETED.value == "deleted"


class TestFileChange:
    """Tests for FileChange dataclass."""

    def test_creation(self) -> None:
        """Test creating a file change."""
        change = FileChange(
            path=Path("/test.py"),
            change_type=ChangeType.CREATED,
            timestamp=1234567890.0,
        )

        assert change.path == Path("/test.py")
        assert change.change_type == ChangeType.CREATED
        assert change.timestamp == 1234567890.0

    def test_to_dict(self) -> None:
        """Test conversion to dict."""
        change = FileChange(
            path=Path("/test.py"),
            change_type=ChangeType.MODIFIED,
            timestamp=1234567890.0,
        )

        d = change.to_dict()

        assert d["path"] == "/test.py"
        assert d["change_type"] == "modified"
        assert d["timestamp"] == 1234567890.0


class TestWatcherConfig:
    """Tests for WatcherConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = WatcherConfig()

        assert config.patterns == ["*.py"]
        assert "__pycache__" in config.exclude_patterns
        assert config.poll_interval == 1.0
        assert config.debounce_delay == 0.5
        assert config.recursive is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = WatcherConfig(
            patterns=["*.js", "*.ts"],
            exclude_patterns=["node_modules"],
            poll_interval=2.0,
            debounce_delay=1.0,
            recursive=False,
        )

        assert config.patterns == ["*.js", "*.ts"]
        assert config.poll_interval == 2.0


class TestFileWatcher:
    """Tests for FileWatcher."""

    @pytest.fixture
    def watcher(self, tmp_path: Path) -> FileWatcher:
        """Create file watcher."""
        config = WatcherConfig(
            patterns=["*.py"],
            poll_interval=0.1,
            debounce_delay=0.05,
        )
        return FileWatcher(tmp_path, config)

    def test_initialization(self, watcher: FileWatcher, tmp_path: Path) -> None:
        """Test watcher initialization."""
        assert watcher.directory == tmp_path
        assert watcher.config is not None
        assert not watcher.is_running

    def test_add_callback(self, watcher: FileWatcher) -> None:
        """Test adding callback."""
        changes = []

        def callback(c: list[FileChange]) -> None:
            changes.extend(c)

        watcher.add_callback(callback)

        assert len(watcher._callbacks) == 1

    def test_remove_callback(self, watcher: FileWatcher) -> None:
        """Test removing callback."""

        def callback(c: list[FileChange]) -> None:
            pass

        watcher.add_callback(callback)
        result = watcher.remove_callback(callback)

        assert result is True
        assert len(watcher._callbacks) == 0

    def test_remove_callback_not_found(self, watcher: FileWatcher) -> None:
        """Test removing non-existent callback."""

        def callback(c: list[FileChange]) -> None:
            pass

        result = watcher.remove_callback(callback)
        assert result is False

    def test_should_watch_matches_pattern(
        self, watcher: FileWatcher, tmp_path: Path
    ) -> None:
        """Test _should_watch matches pattern."""
        assert watcher._should_watch(tmp_path / "test.py")
        assert not watcher._should_watch(tmp_path / "test.txt")

    def test_should_watch_excludes_pattern(
        self, watcher: FileWatcher, tmp_path: Path
    ) -> None:
        """Test _should_watch excludes pattern."""
        assert not watcher._should_watch(tmp_path / "__pycache__" / "cache.py")
        assert not watcher._should_watch(tmp_path / ".git" / "config.py")

    def test_scan_directory(self, watcher: FileWatcher, tmp_path: Path) -> None:
        """Test scanning directory."""
        (tmp_path / "file1.py").write_text("content 1")
        (tmp_path / "file2.py").write_text("content 2")
        (tmp_path / "file.txt").write_text("should be ignored")

        mtimes = watcher._scan_directory()

        assert len(mtimes) == 2
        assert tmp_path / "file1.py" in mtimes
        assert tmp_path / "file2.py" in mtimes

    def test_scan_directory_recursive(self, tmp_path: Path) -> None:
        """Test recursive directory scanning."""
        config = WatcherConfig(patterns=["*.py"], recursive=True)
        watcher = FileWatcher(tmp_path, config)

        (tmp_path / "file.py").write_text("content")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "nested.py").write_text("nested")

        mtimes = watcher._scan_directory()

        assert len(mtimes) == 2

    def test_scan_directory_non_recursive(self, tmp_path: Path) -> None:
        """Test non-recursive directory scanning."""
        config = WatcherConfig(patterns=["*.py"], recursive=False)
        watcher = FileWatcher(tmp_path, config)

        (tmp_path / "file.py").write_text("content")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "nested.py").write_text("nested")

        mtimes = watcher._scan_directory()

        assert len(mtimes) == 1
        assert tmp_path / "file.py" in mtimes

    def test_detect_changes_new_file(self, watcher: FileWatcher, tmp_path: Path) -> None:
        """Test detecting new files."""
        watcher._file_mtimes = {}

        current = {tmp_path / "new.py": time.time()}
        changes = watcher._detect_changes(current)

        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.CREATED

    def test_detect_changes_deleted_file(
        self, watcher: FileWatcher, tmp_path: Path
    ) -> None:
        """Test detecting deleted files."""
        watcher._file_mtimes = {tmp_path / "deleted.py": time.time()}

        changes = watcher._detect_changes({})

        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.DELETED

    def test_detect_changes_modified_file(
        self, watcher: FileWatcher, tmp_path: Path
    ) -> None:
        """Test detecting modified files."""
        path = tmp_path / "modified.py"
        watcher._file_mtimes = {path: 1000.0}

        current = {path: 2000.0}
        changes = watcher._detect_changes(current)

        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.MODIFIED

    def test_detect_changes_unchanged_file(
        self, watcher: FileWatcher, tmp_path: Path
    ) -> None:
        """Test unchanged files are not reported."""
        path = tmp_path / "unchanged.py"
        mtime = time.time()
        watcher._file_mtimes = {path: mtime}

        current = {path: mtime}
        changes = watcher._detect_changes(current)

        assert len(changes) == 0

    @pytest.mark.asyncio
    async def test_start_and_stop(self, watcher: FileWatcher) -> None:
        """Test starting and stopping watcher."""
        # Start in background
        task = asyncio.create_task(watcher.start())

        await asyncio.sleep(0.05)
        assert watcher.is_running

        watcher.stop()
        await asyncio.sleep(0.05)
        assert not watcher.is_running

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_watched_files_property(
        self, watcher: FileWatcher, tmp_path: Path
    ) -> None:
        """Test watched_files property."""
        (tmp_path / "file.py").write_text("content")

        task = asyncio.create_task(watcher.start())
        await asyncio.sleep(0.05)

        assert len(watcher.watched_files) == 1

        watcher.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_callback_invoked_on_change(self, tmp_path: Path) -> None:
        """Test callback is invoked on file change."""
        config = WatcherConfig(
            patterns=["*.py"],
            poll_interval=0.05,
            debounce_delay=0.02,
        )
        watcher = FileWatcher(tmp_path, config)

        changes_received: list[FileChange] = []

        def callback(changes: list[FileChange]) -> None:
            changes_received.extend(changes)

        watcher.add_callback(callback)

        # Create initial file
        test_file = tmp_path / "test.py"
        test_file.write_text("initial")

        task = asyncio.create_task(watcher.start())
        await asyncio.sleep(0.1)

        # Modify file
        test_file.write_text("modified")
        await asyncio.sleep(0.2)

        watcher.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have detected the modification
        assert any(c.change_type == ChangeType.MODIFIED for c in changes_received)

    @pytest.mark.asyncio
    async def test_async_callback(self, tmp_path: Path) -> None:
        """Test async callback support."""
        config = WatcherConfig(
            patterns=["*.py"],
            poll_interval=0.05,
            debounce_delay=0.02,
        )
        watcher = FileWatcher(tmp_path, config)

        changes_received: list[FileChange] = []

        async def async_callback(changes: list[FileChange]) -> None:
            await asyncio.sleep(0.01)
            changes_received.extend(changes)

        watcher.add_callback(async_callback)

        test_file = tmp_path / "test.py"
        test_file.write_text("initial")

        task = asyncio.create_task(watcher.start())
        await asyncio.sleep(0.1)

        test_file.write_text("modified")
        await asyncio.sleep(0.2)

        watcher.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


class TestChangeCollector:
    """Tests for ChangeCollector."""

    @pytest.fixture
    def collector(self) -> ChangeCollector:
        """Create change collector."""
        return ChangeCollector(batch_delay=0.1)

    @pytest.mark.asyncio
    async def test_add_change(self, collector: ChangeCollector) -> None:
        """Test adding a change."""
        change = FileChange(
            path=Path("/test.py"),
            change_type=ChangeType.CREATED,
            timestamp=time.time(),
        )

        await collector.add(change)

        assert collector.count == 1

    @pytest.mark.asyncio
    async def test_add_many_changes(self, collector: ChangeCollector) -> None:
        """Test adding multiple changes."""
        changes = [
            FileChange(Path(f"/file{i}.py"), ChangeType.CREATED, time.time())
            for i in range(5)
        ]

        await collector.add_many(changes)

        assert collector.count == 5

    @pytest.mark.asyncio
    async def test_deduplication(self, collector: ChangeCollector) -> None:
        """Test deduplication of changes for same path."""
        path = Path("/test.py")

        await collector.add(
            FileChange(path, ChangeType.CREATED, timestamp=1.0)
        )
        await collector.add(
            FileChange(path, ChangeType.MODIFIED, timestamp=2.0)
        )

        # Should keep only the latest
        assert collector.count == 1

    @pytest.mark.asyncio
    async def test_get_and_clear(self, collector: ChangeCollector) -> None:
        """Test get_and_clear method."""
        await collector.add(
            FileChange(Path("/test.py"), ChangeType.CREATED, time.time())
        )

        changes = await collector.get_and_clear()

        assert len(changes) == 1
        assert collector.count == 0

    def test_get_by_type(self, collector: ChangeCollector) -> None:
        """Test get_by_type method."""
        collector._changes = [
            FileChange(Path("/new.py"), ChangeType.CREATED, time.time()),
            FileChange(Path("/modified.py"), ChangeType.MODIFIED, time.time()),
            FileChange(Path("/deleted.py"), ChangeType.DELETED, time.time()),
        ]

        created = collector.get_by_type(ChangeType.CREATED)
        modified = collector.get_by_type(ChangeType.MODIFIED)
        deleted = collector.get_by_type(ChangeType.DELETED)

        assert len(created) == 1
        assert len(modified) == 1
        assert len(deleted) == 1
