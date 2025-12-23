"""Tests for content hasher."""

import tempfile
from pathlib import Path

import pytest

from smart_search.indexing.hasher import (
    ContentHasher,
    FileHash,
    HashComparison,
    HashStore,
    compute_directory_hash,
)


class TestContentHasher:
    """Tests for ContentHasher."""

    @pytest.fixture
    def hasher(self) -> ContentHasher:
        """Create content hasher."""
        return ContentHasher()

    def test_hash_content_string(self, hasher: ContentHasher) -> None:
        """Test hashing string content."""
        result = hasher.hash_content("hello world")
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex digest

    def test_hash_content_bytes(self, hasher: ContentHasher) -> None:
        """Test hashing bytes content."""
        result = hasher.hash_content(b"hello world")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_content_deterministic(self, hasher: ContentHasher) -> None:
        """Test hash is deterministic."""
        content = "test content"
        hash1 = hasher.hash_content(content)
        hash2 = hasher.hash_content(content)
        assert hash1 == hash2

    def test_hash_content_different_for_different_input(
        self, hasher: ContentHasher
    ) -> None:
        """Test different content produces different hash."""
        hash1 = hasher.hash_content("content 1")
        hash2 = hasher.hash_content("content 2")
        assert hash1 != hash2

    def test_hash_file(self, hasher: ContentHasher, tmp_path: Path) -> None:
        """Test hashing a file."""
        file_path = tmp_path / "test.py"
        file_path.write_text("def hello(): pass")

        file_hash = hasher.hash_file(file_path)

        assert isinstance(file_hash, FileHash)
        assert file_hash.file_path == file_path
        assert len(file_hash.content_hash) == 64
        assert file_hash.mtime > 0
        assert file_hash.size > 0

    def test_hash_file_not_found(self, hasher: ContentHasher) -> None:
        """Test hashing non-existent file."""
        with pytest.raises(FileNotFoundError):
            hasher.hash_file(Path("/nonexistent/file.py"))

    def test_hash_files_multiple(self, hasher: ContentHasher, tmp_path: Path) -> None:
        """Test hashing multiple files."""
        files = []
        for i in range(3):
            f = tmp_path / f"file{i}.py"
            f.write_text(f"content {i}")
            files.append(f)

        hashes = hasher.hash_files(files)

        assert len(hashes) == 3
        assert all(isinstance(h, FileHash) for h in hashes.values())

    def test_hash_files_skip_missing(
        self, hasher: ContentHasher, tmp_path: Path
    ) -> None:
        """Test hash_files skips missing files."""
        existing = tmp_path / "exists.py"
        existing.write_text("content")

        hashes = hasher.hash_files([existing, Path("/nonexistent.py")])

        assert len(hashes) == 1
        assert existing in hashes

    def test_quick_hash(self, hasher: ContentHasher, tmp_path: Path) -> None:
        """Test quick hash using metadata."""
        file_path = tmp_path / "test.py"
        file_path.write_text("content")

        quick = hasher.quick_hash(file_path)

        assert isinstance(quick, str)
        assert len(quick) == 32  # MD5 hex digest

    def test_custom_algorithm(self, tmp_path: Path) -> None:
        """Test using custom hash algorithm."""
        hasher = ContentHasher(algorithm="md5")
        result = hasher.hash_content("test")
        assert len(result) == 32  # MD5 hex digest


class TestFileHash:
    """Tests for FileHash dataclass."""

    def test_to_dict(self, tmp_path: Path) -> None:
        """Test conversion to dict."""
        file_hash = FileHash(
            file_path=tmp_path / "test.py",
            content_hash="abc123",
            mtime=1234567890.0,
            size=100,
        )

        d = file_hash.to_dict()

        assert d["content_hash"] == "abc123"
        assert d["mtime"] == 1234567890.0
        assert d["size"] == 100

    def test_from_dict(self) -> None:
        """Test creation from dict."""
        data = {
            "file_path": "/path/to/file.py",
            "content_hash": "abc123",
            "mtime": 1234567890.0,
            "size": 100,
        }

        file_hash = FileHash.from_dict(data)

        assert file_hash.file_path == Path("/path/to/file.py")
        assert file_hash.content_hash == "abc123"

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Test to_dict and from_dict roundtrip."""
        original = FileHash(
            file_path=tmp_path / "test.py",
            content_hash="hash123",
            mtime=1234567890.0,
            size=256,
        )

        d = original.to_dict()
        restored = FileHash.from_dict(d)

        assert restored.file_path == original.file_path
        assert restored.content_hash == original.content_hash
        assert restored.mtime == original.mtime
        assert restored.size == original.size


class TestHashComparison:
    """Tests for HashComparison."""

    def test_empty_comparison(self) -> None:
        """Test empty comparison."""
        comparison = HashComparison()

        assert comparison.added == []
        assert comparison.modified == []
        assert comparison.deleted == []
        assert comparison.unchanged == []
        assert not comparison.has_changes
        assert comparison.change_count == 0

    def test_has_changes(self) -> None:
        """Test has_changes property."""
        comparison = HashComparison(added=[Path("new.py")])
        assert comparison.has_changes
        assert comparison.change_count == 1

    def test_change_count(self) -> None:
        """Test change_count property."""
        comparison = HashComparison(
            added=[Path("a.py"), Path("b.py")],
            modified=[Path("c.py")],
            deleted=[Path("d.py")],
        )
        assert comparison.change_count == 4


class TestHashStore:
    """Tests for HashStore."""

    @pytest.fixture
    def store(self) -> HashStore:
        """Create hash store."""
        return HashStore()

    @pytest.fixture
    def sample_hash(self, tmp_path: Path) -> FileHash:
        """Create sample file hash."""
        return FileHash(
            file_path=tmp_path / "test.py",
            content_hash="abc123",
            mtime=1234567890.0,
            size=100,
        )

    def test_update_and_get(self, store: HashStore, sample_hash: FileHash) -> None:
        """Test update and get."""
        store.update(sample_hash)

        retrieved = store.get(sample_hash.file_path)

        assert retrieved is not None
        assert retrieved.content_hash == sample_hash.content_hash

    def test_get_not_found(self, store: HashStore) -> None:
        """Test get returns None for unknown path."""
        result = store.get(Path("/unknown.py"))
        assert result is None

    def test_has(self, store: HashStore, sample_hash: FileHash) -> None:
        """Test has method."""
        assert not store.has(sample_hash.file_path)

        store.update(sample_hash)

        assert store.has(sample_hash.file_path)

    def test_remove(self, store: HashStore, sample_hash: FileHash) -> None:
        """Test remove method."""
        store.update(sample_hash)

        assert store.remove(sample_hash.file_path)
        assert not store.has(sample_hash.file_path)
        assert not store.remove(sample_hash.file_path)  # Already removed

    def test_clear(self, store: HashStore, sample_hash: FileHash) -> None:
        """Test clear method."""
        store.update(sample_hash)

        store.clear()

        assert store.count == 0

    def test_update_many(self, store: HashStore, tmp_path: Path) -> None:
        """Test update_many method."""
        hashes = {
            tmp_path / "a.py": FileHash(tmp_path / "a.py", "hash_a", 1.0, 10),
            tmp_path / "b.py": FileHash(tmp_path / "b.py", "hash_b", 2.0, 20),
        }

        store.update_many(hashes)

        assert store.count == 2

    def test_compare_added(self, store: HashStore, tmp_path: Path) -> None:
        """Test compare detects added files."""
        new_hash = FileHash(tmp_path / "new.py", "hash", 1.0, 10)
        current = {new_hash.file_path: new_hash}

        comparison = store.compare(current)

        assert len(comparison.added) == 1
        assert new_hash.file_path in comparison.added

    def test_compare_deleted(self, store: HashStore, tmp_path: Path) -> None:
        """Test compare detects deleted files."""
        old_hash = FileHash(tmp_path / "old.py", "hash", 1.0, 10)
        store.update(old_hash)

        comparison = store.compare({})

        assert len(comparison.deleted) == 1
        assert old_hash.file_path in comparison.deleted

    def test_compare_modified(self, store: HashStore, tmp_path: Path) -> None:
        """Test compare detects modified files."""
        path = tmp_path / "file.py"
        old_hash = FileHash(path, "old_hash", 1.0, 10)
        new_hash = FileHash(path, "new_hash", 2.0, 15)

        store.update(old_hash)
        comparison = store.compare({path: new_hash})

        assert len(comparison.modified) == 1
        assert path in comparison.modified

    def test_compare_unchanged(self, store: HashStore, tmp_path: Path) -> None:
        """Test compare detects unchanged files."""
        path = tmp_path / "file.py"
        file_hash = FileHash(path, "same_hash", 1.0, 10)

        store.update(file_hash)
        comparison = store.compare({path: file_hash})

        assert len(comparison.unchanged) == 1
        assert path in comparison.unchanged

    def test_to_dict(self, store: HashStore, sample_hash: FileHash) -> None:
        """Test serialization."""
        store.update(sample_hash)

        d = store.to_dict()

        assert "hashes" in d
        assert str(sample_hash.file_path) in d["hashes"]

    def test_from_dict(self) -> None:
        """Test deserialization."""
        data = {
            "hashes": {
                "/path/to/file.py": {
                    "file_path": "/path/to/file.py",
                    "content_hash": "abc123",
                    "mtime": 1234567890.0,
                    "size": 100,
                }
            }
        }

        store = HashStore.from_dict(data)

        assert store.count == 1
        assert store.has(Path("/path/to/file.py"))

    def test_from_dict_empty(self) -> None:
        """Test deserialization of empty store."""
        store = HashStore.from_dict({})
        assert store.count == 0

    def test_paths_property(self, store: HashStore, tmp_path: Path) -> None:
        """Test paths property."""
        path1 = tmp_path / "a.py"
        path2 = tmp_path / "b.py"

        store.update(FileHash(path1, "hash1", 1.0, 10))
        store.update(FileHash(path2, "hash2", 2.0, 20))

        paths = store.paths

        assert len(paths) == 2
        assert path1 in paths
        assert path2 in paths


class TestComputeDirectoryHash:
    """Tests for compute_directory_hash function."""

    def test_compute_directory_hash(self, tmp_path: Path) -> None:
        """Test computing hashes for a directory."""
        # Create some files
        (tmp_path / "file1.py").write_text("content 1")
        (tmp_path / "file2.py").write_text("content 2")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file3.py").write_text("content 3")

        hashes = compute_directory_hash(tmp_path, patterns=["*.py"])

        assert len(hashes) == 3

    def test_exclude_patterns(self, tmp_path: Path) -> None:
        """Test excluding patterns."""
        (tmp_path / "keep.py").write_text("keep")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cache.pyc").write_bytes(b"cache")

        hashes = compute_directory_hash(tmp_path, patterns=["*.py", "*.pyc"])

        # Only keep.py should be included, __pycache__ excluded
        assert len(hashes) == 1

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Test empty directory."""
        hashes = compute_directory_hash(tmp_path)
        assert hashes == {}
