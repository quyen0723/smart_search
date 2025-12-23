"""Tests for incremental indexing."""

import json
import time
from pathlib import Path

import pytest

from smart_search.indexing.hasher import FileHash
from smart_search.indexing.incremental import (
    DeltaBuilder,
    IncrementalIndexer,
    IncrementalUpdate,
    IndexState,
)
from smart_search.indexing.watcher import ChangeType, FileChange
from smart_search.parsing.models import CodeUnit, CodeUnitType, Language, Position, Span


class TestIndexState:
    """Tests for IndexState."""

    @pytest.fixture
    def state(self, tmp_path: Path) -> IndexState:
        """Create index state."""
        return IndexState(directory=tmp_path)

    def test_initialization(self, state: IndexState, tmp_path: Path) -> None:
        """Test state initialization."""
        assert state.directory == tmp_path
        assert state.indexed_files == set()
        assert state.last_index_time == 0.0
        assert state.version == 1

    def test_to_dict(self, state: IndexState, tmp_path: Path) -> None:
        """Test serialization."""
        state.indexed_files.add(tmp_path / "file.py")
        state.last_index_time = 1234567890.0

        d = state.to_dict()

        assert d["directory"] == str(tmp_path)
        assert len(d["indexed_files"]) == 1
        assert d["last_index_time"] == 1234567890.0
        assert d["version"] == 1

    def test_from_dict(self, tmp_path: Path) -> None:
        """Test deserialization."""
        data = {
            "directory": str(tmp_path),
            "hash_store": {"hashes": {}},
            "indexed_files": [str(tmp_path / "file.py")],
            "last_index_time": 1234567890.0,
            "version": 1,
        }

        state = IndexState.from_dict(data)

        assert state.directory == tmp_path
        assert tmp_path / "file.py" in state.indexed_files
        assert state.last_index_time == 1234567890.0

    def test_from_dict_defaults(self, tmp_path: Path) -> None:
        """Test deserialization with defaults."""
        data = {"directory": str(tmp_path)}

        state = IndexState.from_dict(data)

        assert state.indexed_files == set()
        assert state.last_index_time == 0.0

    def test_save_and_load(self, state: IndexState, tmp_path: Path) -> None:
        """Test save and load."""
        state.indexed_files.add(tmp_path / "file.py")
        state.last_index_time = time.time()

        state_path = tmp_path / ".smart_search" / "state.json"
        state.save(state_path)

        loaded = IndexState.load(state_path)

        assert loaded is not None
        assert loaded.directory == state.directory
        assert loaded.indexed_files == state.indexed_files

    def test_load_not_found(self, tmp_path: Path) -> None:
        """Test loading non-existent state."""
        result = IndexState.load(tmp_path / "nonexistent.json")
        assert result is None

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        """Test loading invalid JSON."""
        state_path = tmp_path / "invalid.json"
        state_path.write_text("not json")

        result = IndexState.load(state_path)
        assert result is None


class TestIncrementalUpdate:
    """Tests for IncrementalUpdate."""

    def test_empty_update(self) -> None:
        """Test empty update."""
        update = IncrementalUpdate()

        assert update.files_to_index == []
        assert update.files_to_remove == []
        assert update.files_unchanged == []
        assert not update.has_changes
        assert update.total_changes == 0

    def test_has_changes_with_files_to_index(self) -> None:
        """Test has_changes with files to index."""
        update = IncrementalUpdate(files_to_index=[Path("/new.py")])
        assert update.has_changes

    def test_has_changes_with_files_to_remove(self) -> None:
        """Test has_changes with files to remove."""
        update = IncrementalUpdate(files_to_remove=[Path("/old.py")])
        assert update.has_changes

    def test_total_changes(self) -> None:
        """Test total_changes calculation."""
        update = IncrementalUpdate(
            files_to_index=[Path("/a.py"), Path("/b.py")],
            files_to_remove=[Path("/c.py")],
        )
        assert update.total_changes == 3


class TestIncrementalIndexer:
    """Tests for IncrementalIndexer."""

    @pytest.fixture
    def indexer(self, tmp_path: Path) -> IncrementalIndexer:
        """Create incremental indexer."""
        return IncrementalIndexer(
            directory=tmp_path,
            patterns=["*.py"],
        )

    def test_initialization(self, indexer: IncrementalIndexer, tmp_path: Path) -> None:
        """Test indexer initialization."""
        assert indexer.directory == tmp_path
        assert indexer.patterns == ["*.py"]
        assert not indexer.is_initialized

    def test_load_state_creates_new(self, indexer: IncrementalIndexer) -> None:
        """Test load_state creates new state if none exists."""
        state = indexer.load_state()

        assert state is not None
        assert state.directory == indexer.directory

    def test_load_state_caches(self, indexer: IncrementalIndexer) -> None:
        """Test load_state caches the state."""
        state1 = indexer.load_state()
        state2 = indexer.load_state()

        assert state1 is state2

    def test_save_state(self, indexer: IncrementalIndexer, tmp_path: Path) -> None:
        """Test saving state."""
        state = indexer.load_state()
        state.indexed_files.add(tmp_path / "file.py")

        indexer.save_state()

        assert indexer.state_path.exists()

    def test_calculate_update_no_files(self, indexer: IncrementalIndexer) -> None:
        """Test calculate_update with no files."""
        update = indexer.calculate_update()

        assert not update.has_changes

    def test_calculate_update_new_files(
        self, indexer: IncrementalIndexer, tmp_path: Path
    ) -> None:
        """Test calculate_update with new files."""
        (tmp_path / "new.py").write_text("content")

        update = indexer.calculate_update()

        assert len(update.files_to_index) == 1
        assert tmp_path / "new.py" in update.files_to_index

    def test_calculate_update_modified_files(
        self, indexer: IncrementalIndexer, tmp_path: Path
    ) -> None:
        """Test calculate_update with modified files."""
        file_path = tmp_path / "file.py"
        file_path.write_text("original")

        # Index initially
        indexer.load_state()
        indexer.mark_indexed([file_path])

        # Modify file
        file_path.write_text("modified content")

        update = indexer.calculate_update()

        assert file_path in update.files_to_index

    def test_calculate_update_deleted_files(
        self, indexer: IncrementalIndexer, tmp_path: Path
    ) -> None:
        """Test calculate_update with deleted files."""
        file_path = tmp_path / "file.py"
        file_path.write_text("content")

        # Index initially
        indexer.load_state()
        indexer.mark_indexed([file_path])

        # Delete file
        file_path.unlink()

        update = indexer.calculate_update()

        assert file_path in update.files_to_remove

    def test_should_skip(self, indexer: IncrementalIndexer, tmp_path: Path) -> None:
        """Test _should_skip method."""
        assert indexer._should_skip(tmp_path / "__pycache__" / "cache.pyc")
        assert indexer._should_skip(tmp_path / ".git" / "config")
        assert indexer._should_skip(tmp_path / ".venv" / "lib" / "module.py")
        assert indexer._should_skip(tmp_path / "node_modules" / "package.js")
        assert not indexer._should_skip(tmp_path / "src" / "main.py")

    def test_mark_indexed(
        self, indexer: IncrementalIndexer, tmp_path: Path
    ) -> None:
        """Test mark_indexed method."""
        file_path = tmp_path / "file.py"
        file_path.write_text("content")

        indexer.mark_indexed([file_path])

        state = indexer.load_state()
        assert file_path in state.indexed_files
        assert state.hash_store.has(file_path)
        assert state.last_index_time > 0

    def test_mark_indexed_with_precomputed_hashes(
        self, indexer: IncrementalIndexer, tmp_path: Path
    ) -> None:
        """Test mark_indexed with pre-computed hashes."""
        file_path = tmp_path / "file.py"
        file_path.write_text("content")

        file_hashes = {
            file_path: FileHash(file_path, "precomputed_hash", 1.0, 10)
        }

        indexer.mark_indexed([file_path], file_hashes=file_hashes)

        state = indexer.load_state()
        stored = state.hash_store.get(file_path)
        assert stored is not None
        assert stored.content_hash == "precomputed_hash"

    def test_mark_removed(
        self, indexer: IncrementalIndexer, tmp_path: Path
    ) -> None:
        """Test mark_removed method."""
        file_path = tmp_path / "file.py"
        file_path.write_text("content")

        # First index
        indexer.mark_indexed([file_path])

        # Then remove
        indexer.mark_removed([file_path])

        state = indexer.load_state()
        assert file_path not in state.indexed_files
        assert not state.hash_store.has(file_path)

    def test_process_changes(self, indexer: IncrementalIndexer, tmp_path: Path) -> None:
        """Test process_changes method."""
        changes = [
            FileChange(
                path=tmp_path / "new.py",
                change_type=ChangeType.CREATED,
                timestamp=time.time(),
            ),
            FileChange(
                path=tmp_path / "modified.py",
                change_type=ChangeType.MODIFIED,
                timestamp=time.time(),
            ),
            FileChange(
                path=tmp_path / "deleted.py",
                change_type=ChangeType.DELETED,
                timestamp=time.time(),
            ),
            FileChange(
                path=tmp_path / "other.txt",  # Doesn't match pattern
                change_type=ChangeType.CREATED,
                timestamp=time.time(),
            ),
        ]

        update = indexer.process_changes(changes)

        assert len(update.files_to_index) == 2  # new.py and modified.py
        assert len(update.files_to_remove) == 1  # deleted.py

    def test_reset(self, indexer: IncrementalIndexer, tmp_path: Path) -> None:
        """Test reset method."""
        file_path = tmp_path / "file.py"
        file_path.write_text("content")

        # Index
        indexer.mark_indexed([file_path])
        indexer.save_state()

        # Reset
        indexer.reset()

        assert not indexer.state_path.exists()
        state = indexer.load_state()
        assert len(state.indexed_files) == 0

    def test_indexed_count(
        self, indexer: IncrementalIndexer, tmp_path: Path
    ) -> None:
        """Test indexed_count property."""
        file_path = tmp_path / "file.py"
        file_path.write_text("content")

        assert indexer.indexed_count == 0

        indexer.mark_indexed([file_path])

        assert indexer.indexed_count == 1

    def test_is_initialized(
        self, indexer: IncrementalIndexer, tmp_path: Path
    ) -> None:
        """Test is_initialized property."""
        assert not indexer.is_initialized

        indexer.load_state()
        indexer.save_state()

        assert indexer.is_initialized


class TestDeltaBuilder:
    """Tests for DeltaBuilder."""

    @pytest.fixture
    def builder(self) -> DeltaBuilder:
        """Create delta builder."""
        return DeltaBuilder()

    @pytest.fixture
    def sample_unit(self, tmp_path: Path) -> CodeUnit:
        """Create sample code unit."""
        return CodeUnit(
            id="module::test_func",
            name="test_func",
            qualified_name="module.test_func",
            type=CodeUnitType.FUNCTION,
            file_path=tmp_path / "test.py",
            span=Span(
                start=Position(line=1, column=0),
                end=Position(line=10, column=0),
            ),
            language=Language.PYTHON,
            content="def test_func(): pass",
        )

    def test_add_unit(self, builder: DeltaBuilder, sample_unit: CodeUnit) -> None:
        """Test adding a unit."""
        builder.add_unit(sample_unit)

        assert builder.added_count == 1
        assert builder.has_changes

    def test_remove_unit(self, builder: DeltaBuilder) -> None:
        """Test removing a unit."""
        builder.remove_unit("unit_id_123")

        assert builder.removed_count == 1
        assert builder.has_changes

    def test_modify_unit(self, builder: DeltaBuilder, sample_unit: CodeUnit) -> None:
        """Test modifying a unit."""
        builder.modify_unit(sample_unit)

        assert builder.modified_count == 1
        assert builder.has_changes

    def test_no_changes(self, builder: DeltaBuilder) -> None:
        """Test builder with no changes."""
        assert not builder.has_changes
        assert builder.added_count == 0
        assert builder.removed_count == 0
        assert builder.modified_count == 0

    def test_get_added_units(
        self, builder: DeltaBuilder, sample_unit: CodeUnit
    ) -> None:
        """Test get_added_units method."""
        builder.add_unit(sample_unit)

        added = builder.get_added_units()

        assert len(added) == 1
        assert added[0] == sample_unit

    def test_get_removed_ids(self, builder: DeltaBuilder) -> None:
        """Test get_removed_ids method."""
        builder.remove_unit("id1")
        builder.remove_unit("id2")

        removed = builder.get_removed_ids()

        assert len(removed) == 2
        assert "id1" in removed
        assert "id2" in removed

    def test_get_modified_units(
        self, builder: DeltaBuilder, sample_unit: CodeUnit
    ) -> None:
        """Test get_modified_units method."""
        builder.modify_unit(sample_unit)

        modified = builder.get_modified_units()

        assert len(modified) == 1
        assert modified[0] == sample_unit

    def test_clear(self, builder: DeltaBuilder, sample_unit: CodeUnit) -> None:
        """Test clear method."""
        builder.add_unit(sample_unit)
        builder.remove_unit("id")
        builder.modify_unit(sample_unit)

        builder.clear()

        assert not builder.has_changes
        assert builder.added_count == 0
        assert builder.removed_count == 0
        assert builder.modified_count == 0

    def test_remove_file_units(self, builder: DeltaBuilder, tmp_path: Path) -> None:
        """Test remove_file_units method."""
        file_path = tmp_path / "file.py"
        unit_ids = ["unit1", "unit2", "unit3"]

        builder.remove_file_units(file_path, unit_ids)

        assert builder.removed_count == 3
        removed = builder.get_removed_ids()
        assert all(uid in removed for uid in unit_ids)
