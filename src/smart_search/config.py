"""Configuration management using Pydantic Settings.

This module provides centralized, type-safe configuration for the Smart Search system.
Configuration is loaded from environment variables with sensible defaults.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Application-level settings."""

    model_config = SettingsConfigDict(env_prefix="APP_")

    name: str = "smart-search"
    env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.env == "production"


class APISettings(BaseSettings):
    """API server settings."""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = True


class MeilisearchSettings(BaseSettings):
    """Meilisearch connection and index settings."""

    model_config = SettingsConfigDict(env_prefix="MEILISEARCH_")

    host: str = "http://localhost"
    port: int = 7700
    master_key: str = "smart_search_dev_key"
    env: Literal["development", "production"] = "development"
    log_level: str = "INFO"

    # Index settings
    index_code_units: str = "code_units"
    batch_size: int = Field(default=1000, ge=1, le=10000)

    @property
    def url(self) -> str:
        """Get full Meilisearch URL."""
        return f"{self.host}:{self.port}"


class EmbeddingSettings(BaseSettings):
    """Embedding model settings."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    model: str = "jinaai/jina-embeddings-v2-base-code"
    dimensions: int = 768
    max_seq_length: int = 8192
    batch_size: int = Field(default=32, ge=1, le=256)
    device: Literal["cpu", "cuda", "mps"] = "cpu"

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device is available."""
        # In production, this could check if CUDA/MPS is actually available
        return v


class GraphSettings(BaseSettings):
    """Graph engine settings."""

    model_config = SettingsConfigDict(env_prefix="GRAPH_")

    storage_path: Path = Path("./data/graphs")
    max_depth: int = Field(default=10, ge=1, le=100)

    def __init__(self, **data):  # type: ignore[no-untyped-def]
        super().__init__(**data)
        self.storage_path.mkdir(parents=True, exist_ok=True)


class IndexingSettings(BaseSettings):
    """Indexing and file watching settings."""

    model_config = SettingsConfigDict(env_prefix="")

    index_storage_path: Path = Path("./data/indices")
    embedding_cache_path: Path = Path("./data/embeddings_cache")
    file_hash_algorithm: Literal["xxhash", "md5", "sha256"] = "xxhash"

    # File watching
    watch_debounce_seconds: float = Field(default=2.0, ge=0.1, le=60.0)
    watch_ignore_patterns: str = "__pycache__,*.pyc,.git,node_modules,.venv,venv"

    def __init__(self, **data):  # type: ignore[no-untyped-def]
        super().__init__(**data)
        self.index_storage_path.mkdir(parents=True, exist_ok=True)
        self.embedding_cache_path.mkdir(parents=True, exist_ok=True)

    @property
    def ignore_patterns_list(self) -> list[str]:
        """Get ignore patterns as a list."""
        return [p.strip() for p in self.watch_ignore_patterns.split(",")]


class CacheSettings(BaseSettings):
    """Cache settings."""

    model_config = SettingsConfigDict(env_prefix="CACHE_")

    ttl_seconds: int = Field(default=3600, ge=60)
    max_size_mb: int = Field(default=512, ge=64)


class PerformanceSettings(BaseSettings):
    """Performance tuning settings."""

    model_config = SettingsConfigDict(env_prefix="")

    search_timeout_ms: int = Field(default=5000, ge=100, le=60000)
    graph_max_nodes: int = Field(default=10000, ge=100)


class FeatureFlags(BaseSettings):
    """Feature flags for gradual rollout and A/B testing.

    All flags default to False for safe rollout.
    Enable via environment variables: FF_USE_MEILISEARCH_SEARCH=true
    """

    model_config = SettingsConfigDict(env_prefix="FF_")

    # Search routing
    use_meilisearch_search: bool = Field(
        default=False,
        description="Route search queries to Meilisearch instead of brute-force"
    )

    # Reference index
    use_reference_index: bool = Field(
        default=False,
        description="Use inverted index for find_references instead of file scanning"
    )

    # Caching
    use_file_cache: bool = Field(
        default=True,
        description="Enable LRU cache for file content"
    )

    # Async I/O
    use_async_io: bool = Field(
        default=True,
        description="Use aiofiles for non-blocking file I/O"
    )

    # HybridSearcher
    use_hybrid_searcher: bool = Field(
        default=False,
        description="Use HybridSearcher instead of SimpleIndexer"
    )


class Settings(BaseSettings):
    """Main settings container aggregating all configuration sections."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app: AppSettings = Field(default_factory=AppSettings)
    api: APISettings = Field(default_factory=APISettings)
    meilisearch: MeilisearchSettings = Field(default_factory=MeilisearchSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    indexing: IndexingSettings = Field(default_factory=IndexingSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Uses LRU cache to ensure settings are only loaded once.

    Returns:
        Settings: The application settings instance.
    """
    return Settings()
