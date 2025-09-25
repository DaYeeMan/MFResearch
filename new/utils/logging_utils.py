"""
Logging utilities for SABR MDA-CNN project.
Provides structured logging with different levels and output formats.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime
import json


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'experiment_name'):
            log_entry['experiment_name'] = record.experiment_name
        if hasattr(record, 'epoch'):
            log_entry['epoch'] = record.epoch
        if hasattr(record, 'batch'):
            log_entry['batch'] = record.batch
        if hasattr(record, 'metric_name'):
            log_entry['metric_name'] = record.metric_name
        if hasattr(record, 'metric_value'):
            log_entry['metric_value'] = record.metric_value
            
        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{level_color}{record.levelname}{self.COLORS['RESET']}"
        
        # Format the message
        formatted = super().format(record)
        return formatted


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Union[str, Path]] = None,
    experiment_name: Optional[str] = None,
    console_output: bool = True,
    json_format: bool = False
) -> logging.Logger:
    """
    Set up logging configuration for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to save log files (optional)
        experiment_name: Name of the experiment for log file naming
        console_output: Whether to output logs to console
        json_format: Whether to use JSON format for file logs
        
    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create root logger
    logger = logging.getLogger('sabr_mda_cnn')
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        # Use colored formatter for console
        console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        console_formatter = ColoredFormatter(console_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            log_filename = f"{experiment_name}_{timestamp}.log"
        else:
            log_filename = f"sabr_mda_cnn_{timestamp}.log"
        
        log_file = log_dir / log_filename
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        
        # Choose formatter based on json_format flag
        if json_format:
            file_formatter = JSONFormatter()
        else:
            file_format = '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
            file_formatter = logging.Formatter(file_format)
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(f'sabr_mda_cnn.{name}')


class ExperimentLogger:
    """Specialized logger for experiment tracking."""
    
    def __init__(self, experiment_name: str, log_dir: Optional[Union[str, Path]] = None):
        self.experiment_name = experiment_name
        self.logger = get_logger('experiment')
        
        if log_dir:
            # Set up experiment-specific file handler
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{experiment_name}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            formatter = JSONFormatter()
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log_experiment_start(self, config: dict):
        """Log experiment start with configuration."""
        self.logger.info(
            "Experiment started",
            extra={
                'experiment_name': self.experiment_name,
                'config': config
            }
        )
    
    def log_experiment_end(self, results: dict):
        """Log experiment end with results."""
        self.logger.info(
            "Experiment completed",
            extra={
                'experiment_name': self.experiment_name,
                'results': results
            }
        )
    
    def log_epoch(self, epoch: int, metrics: dict):
        """Log epoch metrics."""
        for metric_name, metric_value in metrics.items():
            self.logger.info(
                f"Epoch {epoch} - {metric_name}: {metric_value}",
                extra={
                    'experiment_name': self.experiment_name,
                    'epoch': epoch,
                    'metric_name': metric_name,
                    'metric_value': metric_value
                }
            )
    
    def log_batch(self, epoch: int, batch: int, metrics: dict):
        """Log batch metrics."""
        for metric_name, metric_value in metrics.items():
            self.logger.debug(
                f"Epoch {epoch}, Batch {batch} - {metric_name}: {metric_value}",
                extra={
                    'experiment_name': self.experiment_name,
                    'epoch': epoch,
                    'batch': batch,
                    'metric_name': metric_name,
                    'metric_value': metric_value
                }
            )
    
    def log_model_checkpoint(self, epoch: int, checkpoint_path: str, metrics: dict):
        """Log model checkpoint save."""
        self.logger.info(
            f"Model checkpoint saved at epoch {epoch}",
            extra={
                'experiment_name': self.experiment_name,
                'epoch': epoch,
                'checkpoint_path': checkpoint_path,
                'metrics': metrics
            }
        )
    
    def log_error(self, error_msg: str, exception: Optional[Exception] = None):
        """Log experiment error."""
        if exception:
            self.logger.error(
                error_msg,
                extra={'experiment_name': self.experiment_name},
                exc_info=True
            )
        else:
            self.logger.error(
                error_msg,
                extra={'experiment_name': self.experiment_name}
            )


def log_function_call(func):
    """Decorator to log function calls with arguments and execution time."""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function start
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {execution_time:.4f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f}s: {e}")
            raise
    
    return wrapper


# Initialize default logger
default_logger = setup_logging()