"""Structured logging configuration for Smart Search.

This module sets up structlog for consistent, structured logging
throughout the application with JSON output for production and
colored console output for development.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from smart_search.config import get_settings


def setup_logging() -> None:
    """Configure structured logging for the application.

    Sets up structlog with appropriate processors based on environment:
    - Development: Colored console output with timestamps
    - Production: JSON output for log aggregation

    Also configures standard library logging to use structlog.
    """
    settings = get_settings()
    is_dev = settings.app.env == "development"

    # Shared processors for all environments
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if is_dev:
        # Development: colored console output
        processors: list[Processor] = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # Production: JSON output
        processors = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    log_level = getattr(logging, settings.app.log_level)

    # Root logger configuration
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Set levels for noisy libraries
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name. If None, uses the calling module's name.

    Returns:
        A bound structlog logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started", file_count=10)
    """
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding temporary context to logs.

    Example:
        >>> with LogContext(request_id="abc123", user_id="user1"):
        ...     logger.info("Processing request")  # includes request_id and user_id
    """

    def __init__(self, **context: Any) -> None:
        self.context = context
        self._token: Any = None

    def __enter__(self) -> "LogContext":
        self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        structlog.contextvars.unbind_contextvars(*self.context.keys())


def bind_context(**context: Any) -> None:
    """Bind context variables that will be included in all subsequent logs.

    Args:
        **context: Key-value pairs to include in log context.

    Example:
        >>> bind_context(request_id="abc123")
        >>> logger.info("Start")  # includes request_id
        >>> logger.info("End")    # also includes request_id
    """
    structlog.contextvars.bind_contextvars(**context)


def unbind_context(*keys: str) -> None:
    """Remove context variables from logging context.

    Args:
        *keys: Keys to remove from context.
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all context variables."""
    structlog.contextvars.clear_contextvars()
