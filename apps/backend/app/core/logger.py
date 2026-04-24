"""
Application-wide logging configuration.
"""

from __future__ import annotations

import sys
import logging

from app.core.config import settings


class InterceptHandler(logging.Handler):
    """
    Intercept logging messages and log them to the console.
    Future implementation: Send logs to Sentry.
    """
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            sys.stdout.write(msg + "\n")
        except Exception:
            self.handleError(record)


class ColorFormatter(logging.Formatter):
    TIME_COLOR = "\x1b[35m"
    NAME_COLOR = "\x1b[36m"
    LEVEL_COLORS = {
        "DEBUG": "\x1b[36m",
        "INFO": "\x1b[32m",
        "WARNING": "\x1b[33m",
        "ERROR": "\x1b[31m",
        "CRITICAL": "\x1b[1;31m",
    }
    RESET = "\x1b[0m"

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        color = self.LEVEL_COLORS.get(levelname, "")
        if color:
            record.levelname = f"{color}{levelname}{self.RESET}"
        try:
            base = super().format(record)
            timestamp = self.formatTime(record, self.datefmt)
            colored_timestamp = f"{self.TIME_COLOR}{timestamp}{self.RESET}"
            colored_name = f"{self.NAME_COLOR}{record.name}{self.RESET}"
            base = base.replace(timestamp, colored_timestamp, 1)
            return base.replace(record.name, colored_name, 1)
        finally:
            record.levelname = levelname


def _build_formatter() -> logging.Formatter:
    """
    Build a formatter with ISO-8601 timestamps suitable for production logs.
    Future implementation: Use JSON formatter for structured logging in Kubernetes.
    """
    return ColorFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _configure_app_logger() -> logging.Logger:
    """
    Configure the root application logger once.
    """
    # Get `app` logger.
    logger = logging.getLogger("app")

    # If handlers are already configured, return the logger.
    if logger.handlers:
        return logger

    # Set level.
    logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)

    # Configure Stdout handler.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_build_formatter())
    logger.addHandler(handler)

    # Prevent double logging with uvicorn/fastapi.
    logger.propagate = False  

    return logger


def setup_logging() -> None:
    """
    Universal logging setup for the application.
    
    This function should be called only once, early in the application startup.
    Converts all logs to the configured level.
    """
    app_logger = _configure_app_logger()

    # Force uvicorn to use our interceptor.
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        log = logging.getLogger(logger_name)
        # Remove existing handlers.
        log.handlers.clear()
        log.setLevel(app_logger.level)
        # Use our interceptor.
        handler = InterceptHandler()
        handler.setFormatter(_build_formatter())
        log.addHandler(handler)
        log.propagate = False


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Return a configured logger.
    If name is None, returns the `app` root logger.
    If name is provided, return a child of `app` (e.g. "app.service.auth")
    """
    # Ensure `app` logger is configured.
    app_logger = _configure_app_logger()

    if name:
        return app_logger.getChild(name)

    return app_logger


# Default logger instance.
logger = get_logger()

__all__ = ["get_logger", "logger", "setup_logging"]
