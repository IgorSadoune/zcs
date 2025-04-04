import logging
import os
import sys
from typing import Optional


def setup_logger(
    name: str = "zero_cost_search",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
) -> logging.Logger:
    """Set up a logger with file and/or console output.

    Args:
        name: Name of the logger.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to log file. If None, no file logging is used.
        console_output: Whether to output logs to console.

    Returns:
        Configured logger instance.
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    logger.handlers = []  # Clear existing handlers

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add file handler if log_file is provided
    if log_file is not None:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
