"""
Logger module for Viren Cloud

This module provides standardized logging functionality for all Viren components.
"""

import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional, Dict, Any, Union

# Configure default logging format
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log levels
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

class VirenLogger:
    """
    Standardized logger for Viren components.
    
    This logger provides consistent formatting and behavior across all Viren
    components, with support for file and console logging, as well as
    structured logging for cloud integration.
    """
    
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        log_file: Optional[str] = None,
        console: bool = True,
        format_str: str = DEFAULT_FORMAT,
        date_format: str = DEFAULT_DATE_FORMAT
    ):
        """
        Initialize the Viren logger.
        
        Args:
            name: Logger name (typically the module name)
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (if None, file logging is disabled)
            console: Whether to log to console
            format_str: Log format string
            date_format: Date format string
        """
        self.name = name
        self.level = LOG_LEVELS.get(level.upper(), logging.INFO)
        self.log_file = log_file
        self.console = console
        self.format_str = format_str
        self.date_format = date_format
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.logger.propagate = False
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(format_str, date_format)
        
        # Add console handler if requested
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Add file handler if log_file is provided
        if log_file:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log a debug message."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log an info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log a warning message."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log an error message."""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log a critical message."""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """
        Internal logging method with support for structured logging.
        
        Args:
            level: Log level
            message: Log message
            **kwargs: Additional structured data to include in the log
        """
        if kwargs:
            # If additional data is provided, include it in the log message
            structured_data = " | " + " | ".join(f"{k}={v}" for k, v in kwargs.items())
            self.logger.log(level, message + structured_data)
        else:
            self.logger.log(level, message)

# Default logger instance
default_logger = VirenLogger(
    name="viren",
    level=os.environ.get("VIREN_LOG_LEVEL", "INFO"),
    log_file=os.environ.get("VIREN_LOG_FILE", "C:/Viren/logs/viren.log"),
    console=True
)

# Convenience functions using the default logger
def debug(message: str, **kwargs):
    """Log a debug message using the default logger."""
    default_logger.debug(message, **kwargs)

def info(message: str, **kwargs):
    """Log an info message using the default logger."""
    default_logger.info(message, **kwargs)

def warning(message: str, **kwargs):
    """Log a warning message using the default logger."""
    default_logger.warning(message, **kwargs)

def error(message: str, **kwargs):
    """Log an error message using the default logger."""
    default_logger.error(message, **kwargs)

def critical(message: str, **kwargs):
    """Log a critical message using the default logger."""
    default_logger.critical(message, **kwargs)

def get_logger(name: str, **kwargs) -> VirenLogger:
    """
    Get a new logger with the specified name.
    
    Args:
        name: Logger name
        **kwargs: Additional configuration options for the logger
        
    Returns:
        A new VirenLogger instance
    """
    return VirenLogger(name=name, **kwargs)