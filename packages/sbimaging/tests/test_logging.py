"""Tests for the logging system."""

import logging
import tempfile
from pathlib import Path

from sbimaging.logging import LoggerFactory, configure_logging, get_logger


def test_get_logger_returns_logger():
    """get_logger should return a logging.Logger instance."""
    logger = get_logger("test")
    assert isinstance(logger, logging.Logger)


def test_logger_has_handlers():
    """Logger should have at least a terminal handler."""
    logger = get_logger("test_handlers")
    assert len(logger.handlers) > 0


def test_logger_writes_to_file():
    """Logger should write to log file when configured."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)

        LoggerFactory._initialized = False
        LoggerFactory._file_handler = None
        configure_logging(log_dir=log_dir, level=logging.DEBUG)

        logger = get_logger("test_file_write")
        logger.info("Test message")

        log_files = list(log_dir.glob("sbi_*.log"))
        assert len(log_files) == 1

        content = log_files[0].read_text()
        assert "Test message" in content


def test_configure_logging_sets_level():
    """configure_logging should set the log level."""
    LoggerFactory._initialized = False
    LoggerFactory._file_handler = None
    configure_logging(level=logging.WARNING, log_to_file=False)

    assert LoggerFactory._log_level == logging.WARNING


def test_logger_name_preserved():
    """Logger should preserve the module name."""
    logger = get_logger("my.module.name")
    assert logger.name == "my.module.name"
