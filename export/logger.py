"""
Logging utilities for Semantic Ising Simulator.

This module provides structured logging with error tracking and context
management for the simulation pipeline.
"""

import logging
from typing import Optional


def init_logger(log_path: str, level: str = "INFO") -> logging.Logger:
    """
    Initialize structured logger with file handler.
    
    Args:
        log_path: Path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("semantic_ising")
    
    # Set log level
    try:
        logger.setLevel(getattr(logging, level.upper()))
    except AttributeError:
        # Default to INFO for invalid levels
        logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def log_event(logger: logging.Logger, message: str, level: str = "INFO") -> None:
    """
    Log an event with specified level.
    
    Args:
        logger: Logger instance
        message: Event message
        level: Logging level
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.log(log_level, message)


def log_exception(logger: logging.Logger, exception: Exception, context: str = "") -> None:
    """
    Log an exception with context and full traceback.
    
    Args:
        logger: Logger instance
        exception: Exception to log
        context: Context string for the exception
    """
    context_str = f"Exception in {context}" if context else "Exception"
    logger.error(f"{context_str}: {str(exception)}", exc_info=True) 