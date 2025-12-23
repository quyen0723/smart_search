"""Unit tests for configuration module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from smart_search.config import (
    APISettings,
    AppSettings,
    CacheSettings,
    EmbeddingSettings,
    GraphSettings,
    IndexingSettings,
    MeilisearchSettings,
    PerformanceSettings,
    Settings,
    get_settings,
)


class TestAppSettings:
    """Tests for AppSettings."""

    def test_default_values(self) -> None:
        """Test default settings values."""
        settings = AppSettings()
        assert settings.name == "smart-search"
        assert settings.env == "development"
        assert settings.debug is True
        assert settings.log_level == "DEBUG"

    def test_is_production_property(self) -> None:
        """Test is_production property."""
        dev_settings = AppSettings(env="development")
        assert dev_settings.is_production is False

        prod_settings = AppSettings(env="production")
        assert prod_settings.is_production is True

    def test_env_override(self) -> None:
        """Test environment variable override."""
        with patch.dict(os.environ, {"APP_ENV": "production", "APP_DEBUG": "false"}):
            settings = AppSettings()
            assert settings.env == "production"
            assert settings.debug is False


class TestAPISettings:
    """Tests for APISettings."""

    def test_default_values(self) -> None:
        """Test default API settings."""
        settings = APISettings()
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.workers == 1
        assert settings.reload is True

    def test_env_override(self) -> None:
        """Test environment variable override."""
        with patch.dict(os.environ, {"API_PORT": "9000", "API_WORKERS": "4"}):
            settings = APISettings()
            assert settings.port == 9000
            assert settings.workers == 4


class TestMeilisearchSettings:
    """Tests for MeilisearchSettings."""

    def test_default_values(self) -> None:
        """Test default Meilisearch settings."""
        settings = MeilisearchSettings()
        assert settings.host == "http://localhost"
        assert settings.port == 7700
        assert settings.index_code_units == "code_units"
        assert settings.batch_size == 1000

    def test_url_property(self) -> None:
        """Test URL construction."""
        settings = MeilisearchSettings(host="http://meilisearch", port=7701)
        assert settings.url == "http://meilisearch:7701"

    def test_batch_size_validation(self) -> None:
        """Test batch size bounds validation."""
        # Valid batch size
        settings = MeilisearchSettings(batch_size=5000)
        assert settings.batch_size == 5000

        # Minimum bound
        with pytest.raises(ValueError):
            MeilisearchSettings(batch_size=0)

        # Maximum bound
        with pytest.raises(ValueError):
            MeilisearchSettings(batch_size=20000)


class TestEmbeddingSettings:
    """Tests for EmbeddingSettings."""

    def test_default_values(self) -> None:
        """Test default embedding settings."""
        settings = EmbeddingSettings()
        assert settings.model == "jinaai/jina-embeddings-v2-base-code"
        assert settings.dimensions == 768
        assert settings.max_seq_length == 8192
        assert settings.batch_size == 32
        assert settings.device == "cpu"

    def test_device_validation(self) -> None:
        """Test device validation."""
        for device in ["cpu", "cuda", "mps"]:
            settings = EmbeddingSettings(device=device)
            assert settings.device == device


class TestGraphSettings:
    """Tests for GraphSettings."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Test default graph settings."""
        with patch.dict(os.environ, {"GRAPH_STORAGE_PATH": str(tmp_path / "graphs")}):
            settings = GraphSettings()
            assert settings.max_depth == 10

    def test_directory_creation(self, tmp_path: Path) -> None:
        """Test automatic directory creation."""
        graph_path = tmp_path / "new_graphs"
        with patch.dict(os.environ, {"GRAPH_STORAGE_PATH": str(graph_path)}):
            settings = GraphSettings()
            assert settings.storage_path.exists()


class TestIndexingSettings:
    """Tests for IndexingSettings."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Test default indexing settings."""
        with patch.dict(
            os.environ,
            {
                "INDEX_STORAGE_PATH": str(tmp_path / "indices"),
                "EMBEDDING_CACHE_PATH": str(tmp_path / "embeddings"),
            },
        ):
            settings = IndexingSettings()
            assert settings.file_hash_algorithm == "xxhash"
            assert settings.watch_debounce_seconds == 2.0

    def test_ignore_patterns_list(self, tmp_path: Path) -> None:
        """Test ignore patterns parsing."""
        with patch.dict(
            os.environ,
            {
                "INDEX_STORAGE_PATH": str(tmp_path / "indices"),
                "EMBEDDING_CACHE_PATH": str(tmp_path / "embeddings"),
                "WATCH_IGNORE_PATTERNS": "*.pyc,__pycache__,.git",
            },
        ):
            settings = IndexingSettings()
            patterns = settings.ignore_patterns_list
            assert "*.pyc" in patterns
            assert "__pycache__" in patterns
            assert ".git" in patterns


class TestCacheSettings:
    """Tests for CacheSettings."""

    def test_default_values(self) -> None:
        """Test default cache settings."""
        settings = CacheSettings()
        assert settings.ttl_seconds == 3600
        assert settings.max_size_mb == 512

    def test_ttl_validation(self) -> None:
        """Test TTL minimum validation."""
        with pytest.raises(ValueError):
            CacheSettings(ttl_seconds=30)  # Below minimum of 60


class TestPerformanceSettings:
    """Tests for PerformanceSettings."""

    def test_default_values(self) -> None:
        """Test default performance settings."""
        settings = PerformanceSettings()
        assert settings.search_timeout_ms == 5000
        assert settings.graph_max_nodes == 10000


class TestSettings:
    """Tests for main Settings container."""

    def test_aggregates_all_settings(self, tmp_path: Path) -> None:
        """Test that Settings contains all subsettings."""
        with patch.dict(
            os.environ,
            {
                "GRAPH_STORAGE_PATH": str(tmp_path / "graphs"),
                "INDEX_STORAGE_PATH": str(tmp_path / "indices"),
                "EMBEDDING_CACHE_PATH": str(tmp_path / "embeddings"),
            },
        ):
            get_settings.cache_clear()
            settings = Settings()

            assert hasattr(settings, "app")
            assert hasattr(settings, "api")
            assert hasattr(settings, "meilisearch")
            assert hasattr(settings, "embedding")
            assert hasattr(settings, "graph")
            assert hasattr(settings, "indexing")
            assert hasattr(settings, "cache")
            assert hasattr(settings, "performance")


class TestGetSettings:
    """Tests for get_settings function."""

    def test_caching(self, tmp_path: Path) -> None:
        """Test that settings are cached."""
        with patch.dict(
            os.environ,
            {
                "GRAPH_STORAGE_PATH": str(tmp_path / "graphs"),
                "INDEX_STORAGE_PATH": str(tmp_path / "indices"),
                "EMBEDDING_CACHE_PATH": str(tmp_path / "embeddings"),
            },
        ):
            get_settings.cache_clear()
            settings1 = get_settings()
            settings2 = get_settings()
            assert settings1 is settings2
            get_settings.cache_clear()
