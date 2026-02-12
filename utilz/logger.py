"""
Logging configuration module for HDAAGT.
Provides a centralized logging setup with both console and file handlers.
"""
import logging
import os
from datetime import datetime


def setup_logger(name='HDAAGT', log_dir='logs', log_filename=None, level=logging.INFO):
    """
    Set up a logger with both console and file handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        log_filename: Name of the log file (if None, uses timestamp)
        level: Logging level (default: INFO)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Generate log filename if not provided
    if log_filename is None:
        # Use timestamp format matching the existing convention in main.py
        # Format: MM-DD-HH-MM (month-day-hour-minute)
        timestamp = datetime.now().strftime("%m-%d-%H-%M")
        log_filename = f"log-{timestamp}.txt"
    
    log_path = os.path.join(log_dir, log_filename)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatters
    # Using simple message-only format to match the existing log format
    # This maintains backward compatibility with existing log parsing tools
    file_formatter = logging.Formatter('%(message)s')
    console_formatter = logging.Formatter('%(message)s')
    
    # Create and configure file handler
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Create and configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name='HDAAGT'):
    """
    Get an existing logger instance.
    
    Args:
        name: Logger name
    
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)
