"""Pytest configuration and shared fixtures.

This module provides fixtures used across all test modules.
"""

import os
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

# Set test environment before importing app modules
os.environ["APP_ENV"] = "development"
os.environ["APP_DEBUG"] = "true"
os.environ["MEILISEARCH_MASTER_KEY"] = "test_key"


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get the test data directory path."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory for tests.

    Args:
        tmp_path: Pytest's temporary path fixture.

    Returns:
        Path: Temporary directory for test data.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "graphs").mkdir()
    (data_dir / "indices").mkdir()
    (data_dir / "embeddings_cache").mkdir()
    return data_dir


@pytest.fixture
def mock_settings(temp_data_dir: Path) -> Generator[Any, None, None]:
    """Provide mocked settings for testing.

    Args:
        temp_data_dir: Temporary data directory.

    Yields:
        Mocked settings instance.
    """
    with patch.dict(
        os.environ,
        {
            "APP_ENV": "development",
            "APP_DEBUG": "true",
            "GRAPH_STORAGE_PATH": str(temp_data_dir / "graphs"),
            "INDEX_STORAGE_PATH": str(temp_data_dir / "indices"),
            "EMBEDDING_CACHE_PATH": str(temp_data_dir / "embeddings_cache"),
        },
    ):
        # Clear cached settings
        from smart_search.config import get_settings

        get_settings.cache_clear()
        yield get_settings()
        get_settings.cache_clear()


@pytest.fixture
def app(mock_settings: Any) -> Any:
    """Create a test application instance.

    Args:
        mock_settings: Mocked settings fixture.

    Returns:
        FastAPI application instance.
    """
    from smart_search.main import create_app

    return create_app()


@pytest.fixture
def client(app: Any) -> Generator[TestClient, None, None]:
    """Create a synchronous test client.

    Args:
        app: FastAPI application instance.

    Yields:
        TestClient instance.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client(app: Any) -> AsyncGenerator[AsyncClient, None]:
    """Create an asynchronous test client.

    Args:
        app: FastAPI application instance.

    Yields:
        AsyncClient instance.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def sample_python_code() -> str:
    """Provide sample Python code for testing."""
    return '''
def hello_world():
    """Say hello to the world."""
    print("Hello, World!")


class Calculator:
    """A simple calculator class."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b
'''


@pytest.fixture
def sample_code_unit() -> dict[str, Any]:
    """Provide a sample code unit document for testing."""
    return {
        "id": "test_module::hello_world",
        "name": "hello_world",
        "type": "function",
        "content": 'def hello_world():\n    """Say hello."""\n    print("Hello")',
        "file_path": "/test/module.py",
        "start_line": 1,
        "end_line": 3,
        "language": "python",
        "docstring": "Say hello.",
        "callers": [],
        "callees": ["print"],
        "imports": [],
        "content_hash": "abc123",
    }


# Markers for test categories
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow tests")
