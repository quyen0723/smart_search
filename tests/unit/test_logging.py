"""Unit tests for logging module."""

import logging
from unittest.mock import patch

import pytest
import structlog

from smart_search.utils.logging import (
    LogContext,
    bind_context,
    clear_context,
    get_logger,
    setup_logging,
    unbind_context,
)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_configures_structlog(self) -> None:
        """Test that setup_logging configures structlog."""
        with patch("smart_search.utils.logging.get_settings") as mock_settings:
            mock_settings.return_value.app.env = "development"
            mock_settings.return_value.app.log_level = "DEBUG"

            setup_logging()

            # Structlog should be configured
            logger = structlog.get_logger()
            assert logger is not None

    def test_setup_logging_development_mode(self) -> None:
        """Test logging configuration in development mode."""
        with patch("smart_search.utils.logging.get_settings") as mock_settings:
            mock_settings.return_value.app.env = "development"
            mock_settings.return_value.app.log_level = "DEBUG"

            setup_logging()

            # Should use console renderer in development
            config = structlog.get_config()
            processors = config.get("processors", [])
            assert len(processors) > 0

    def test_setup_logging_production_mode(self) -> None:
        """Test logging configuration in production mode."""
        with patch("smart_search.utils.logging.get_settings") as mock_settings:
            mock_settings.return_value.app.env = "production"
            mock_settings.return_value.app.log_level = "INFO"

            setup_logging()

            # Should use JSON renderer in production
            config = structlog.get_config()
            processors = config.get("processors", [])
            assert len(processors) > 0

    def test_setup_logging_sets_log_level(self) -> None:
        """Test that log level is set correctly."""
        with patch("smart_search.utils.logging.get_settings") as mock_settings:
            mock_settings.return_value.app.env = "development"
            mock_settings.return_value.app.log_level = "WARNING"

            setup_logging()

            root_logger = logging.getLogger()
            assert root_logger.level == logging.WARNING


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_bound_logger(self) -> None:
        """Test that get_logger returns a bound logger."""
        logger = get_logger("test_module")
        assert logger is not None

    def test_get_logger_with_name(self) -> None:
        """Test get_logger with specific name."""
        logger = get_logger("my.module")
        assert logger is not None

    def test_get_logger_without_name(self) -> None:
        """Test get_logger without name."""
        logger = get_logger()
        assert logger is not None


class TestLogContext:
    """Tests for LogContext context manager."""

    def test_log_context_binds_variables(self) -> None:
        """Test that LogContext binds context variables."""
        clear_context()

        with LogContext(request_id="abc123", user_id="user1"):
            # Context should be bound during the block
            ctx = structlog.contextvars.get_contextvars()
            assert ctx.get("request_id") == "abc123"
            assert ctx.get("user_id") == "user1"

        # Context should be cleared after the block
        ctx = structlog.contextvars.get_contextvars()
        assert "request_id" not in ctx
        assert "user_id" not in ctx

    def test_log_context_nesting(self) -> None:
        """Test nested LogContext."""
        clear_context()

        with LogContext(outer="value1"):
            ctx = structlog.contextvars.get_contextvars()
            assert ctx.get("outer") == "value1"

            with LogContext(inner="value2"):
                ctx = structlog.contextvars.get_contextvars()
                assert ctx.get("outer") == "value1"
                assert ctx.get("inner") == "value2"

            # Inner context removed
            ctx = structlog.contextvars.get_contextvars()
            assert "inner" not in ctx
            assert ctx.get("outer") == "value1"


class TestBindContext:
    """Tests for bind_context function."""

    def test_bind_context_adds_variables(self) -> None:
        """Test that bind_context adds variables."""
        clear_context()

        bind_context(key1="value1", key2="value2")

        ctx = structlog.contextvars.get_contextvars()
        assert ctx.get("key1") == "value1"
        assert ctx.get("key2") == "value2"

        clear_context()

    def test_bind_context_overwrites(self) -> None:
        """Test that bind_context overwrites existing values."""
        clear_context()

        bind_context(key="original")
        bind_context(key="updated")

        ctx = structlog.contextvars.get_contextvars()
        assert ctx.get("key") == "updated"

        clear_context()


class TestUnbindContext:
    """Tests for unbind_context function."""

    def test_unbind_context_removes_variables(self) -> None:
        """Test that unbind_context removes variables."""
        clear_context()

        bind_context(key1="value1", key2="value2")
        unbind_context("key1")

        ctx = structlog.contextvars.get_contextvars()
        assert "key1" not in ctx
        assert ctx.get("key2") == "value2"

        clear_context()

    def test_unbind_context_multiple_keys(self) -> None:
        """Test unbind_context with multiple keys."""
        clear_context()

        bind_context(key1="value1", key2="value2", key3="value3")
        unbind_context("key1", "key2")

        ctx = structlog.contextvars.get_contextvars()
        assert "key1" not in ctx
        assert "key2" not in ctx
        assert ctx.get("key3") == "value3"

        clear_context()


class TestClearContext:
    """Tests for clear_context function."""

    def test_clear_context_removes_all(self) -> None:
        """Test that clear_context removes all variables."""
        bind_context(key1="value1", key2="value2")
        clear_context()

        ctx = structlog.contextvars.get_contextvars()
        assert len(ctx) == 0


class TestLoggerUsage:
    """Tests for actual logger usage patterns."""

    def test_logger_info(self) -> None:
        """Test logger.info works."""
        logger = get_logger("test")
        # Should not raise
        logger.info("Test message", key="value")

    def test_logger_warning(self) -> None:
        """Test logger.warning works."""
        logger = get_logger("test")
        logger.warning("Warning message")

    def test_logger_error(self) -> None:
        """Test logger.error works."""
        logger = get_logger("test")
        logger.error("Error message", error_code=500)

    def test_logger_with_context(self) -> None:
        """Test logger with context variables."""
        clear_context()
        logger = get_logger("test")

        with LogContext(request_id="123"):
            logger.info("Request started")
            # Context should be included in log

        clear_context()
