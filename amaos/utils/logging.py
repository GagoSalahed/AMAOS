"""Logging utilities for AMAOS.

This module provides utilities for configuring logging in the AMAOS system.
"""

import logging
import os
import sys
from typing import Dict, Optional, Union

from pydantic import BaseModel


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file_path: Optional[str] = None
    console_output: bool = True
    log_to_file: bool = False


def configure_logging(config: Optional[LoggingConfig] = None) -> logging.Logger:
    """Configure logging for the AMAOS system.
    
    Args:
        config: Configuration for logging.
        
    Returns:
        Root logger.
    """
    config = config or LoggingConfig()
    
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Set the log level
    level = getattr(logging, config.level.upper(), logging.INFO)
    root_logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(config.format, config.date_format)
    
    # Add console handler if enabled
    if config.console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
    # Add file handler if enabled
    if config.log_to_file and config.file_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(config.file_path)), exist_ok=True)
        
        file_handler = logging.FileHandler(config.file_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
    return root_logger


def get_logger(name: str, level: Optional[Union[str, int]] = None) -> logging.Logger:
    """Get a logger with the specified name and level.
    
    Args:
        name: Name of the logger.
        level: Log level.
        
    Returns:
        Logger.
    """
    logger = logging.getLogger(name)
    
    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(level)
        
    return logger
