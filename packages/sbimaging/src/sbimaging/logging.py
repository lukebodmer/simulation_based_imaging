"""Unified logging system for sbimaging.

Provides consistent logging to both terminal and log files across all components.

Usage:
    from sbimaging.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Simulation started")
    logger.debug("Detailed debug info")
    logger.error("Something went wrong")
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


class LoggerFactory:
    """Factory for creating configured loggers.

    Ensures all loggers share consistent formatting and output destinations.
    """

    _initialized = False
    _log_dir: Path | None = None
    _log_level = logging.INFO
    _file_handler: logging.FileHandler | None = None

    @classmethod
    def configure(
        cls,
        log_dir: Path | str | None = None,
        level: int = logging.INFO,
        log_to_file: bool = True,
    ) -> None:
        """Configure the logging system.

        Args:
            log_dir: Directory for log files. Defaults to data/logs/.
            level: Logging level (e.g., logging.DEBUG, logging.INFO).
            log_to_file: Whether to write logs to file.
        """
        cls._log_level = level

        if log_to_file:
            if log_dir is None:
                log_dir = Path("data/logs")
            cls._log_dir = Path(log_dir)
            cls._log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = cls._log_dir / f"sbi_{timestamp}.log"

            cls._file_handler = logging.FileHandler(log_file)
            cls._file_handler.setLevel(level)
            cls._file_handler.setFormatter(cls._create_file_formatter())

        cls._initialized = True

    @classmethod
    def _create_terminal_formatter(cls) -> logging.Formatter:
        """Create formatter for terminal output."""
        return logging.Formatter(
            fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
            datefmt="%H:%M:%S",
        )

    @classmethod
    def _create_file_formatter(cls) -> logging.Formatter:
        """Create formatter for file output."""
        return logging.Formatter(
            fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(filename)s:%(lineno)d │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a configured logger instance.

        Args:
            name: Logger name, typically __name__ of the calling module.

        Returns:
            Configured logger instance.
        """
        if not cls._initialized:
            cls.configure()

        logger = logging.getLogger(name)
        logger.setLevel(cls._log_level)

        if not logger.handlers:
            terminal_handler = logging.StreamHandler(sys.stdout)
            terminal_handler.setLevel(cls._log_level)
            terminal_handler.setFormatter(cls._create_terminal_formatter())
            logger.addHandler(terminal_handler)

            if cls._file_handler is not None:
                logger.addHandler(cls._file_handler)

        logger.propagate = False

        return logger


def configure_logging(
    log_dir: Path | str | None = None,
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> None:
    """Configure the global logging system.

    Call this once at application startup to set logging preferences.

    Args:
        log_dir: Directory for log files. Defaults to data/logs/.
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
        log_to_file: Whether to write logs to file.
    """
    LoggerFactory.configure(log_dir=log_dir, level=level, log_to_file=log_to_file)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logger instance.

    Example:
        from sbimaging.logging import get_logger

        logger = get_logger(__name__)
        logger.info("Starting simulation")
    """
    return LoggerFactory.get_logger(name)
