"""
Logging utilities for NGIS.

Provides setup for rich logging with multiple output handlers
and integration with TensorBoard and Weights & Biases.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# Install rich traceback handler
install()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    use_rich: bool = True
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level.
        log_file: Optional log file path.
        log_dir: Directory for log files.
        use_rich: Whether to use rich formatting.
        
    Returns:
        Configured logger.
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure logging level
    if isinstance(level, int):
        numeric_level = level
    else:
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level}")
    
    # Create logger
    logger = logging.getLogger("ngis")
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    if use_rich:
        # Rich console handler
        console = Console()
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True
        )
        rich_handler.setLevel(numeric_level)
        logger.addHandler(rich_handler)
    else:
        # Standard console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(numeric_level)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = log_path / "ngis.log"
    
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(numeric_level)
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = "ngis") -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name.
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


def log_config(config, logger: Optional[logging.Logger] = None):
    """
    Log configuration parameters.
    
    Args:
        config: Configuration object.
        logger: Logger instance.
    """
    if logger is None:
        logger = get_logger()
    
    logger.info("Configuration:")
    for section_name, section in config.__dict__.items():
        if hasattr(section, '__dict__'):
            logger.info(f"  {section_name}:")
            for key, value in section.__dict__.items():
                logger.info(f"    {key}: {value}")
        else:
            logger.info(f"  {section_name}: {section}")


def log_model_info(model, logger: Optional[logging.Logger] = None):
    """
    Log model information.
    
    Args:
        model: PyTorch model.
        logger: Logger instance.
    """
    if logger is None:
        logger = get_logger()
    
    param_counts = model.count_parameters()
    logger.info(f"Model parameters: {param_counts['trainable_parameters']} trainable, {param_counts['total_parameters']} total")
    
    if hasattr(model, 'get_model_info'):
        model_info = model.get_model_info()
        logger.info("Model architecture:")
        for key, value in model_info.items():
            if key != 'parameters':
                logger.info(f"  {key}: {value}")


def log_training_start(config, model, logger: Optional[logging.Logger] = None):
    """
    Log training start information.
    
    Args:
        config: Configuration object.
        model: PyTorch model.
        logger: Logger instance.
    """
    if logger is None:
        logger = get_logger()
    
    logger.info("=" * 50)
    logger.info("Starting NGIS Training")
    logger.info("=" * 50)
    
    log_config(config, logger)
    log_model_info(model, logger)
    
    logger.info("=" * 50)


def log_epoch(epoch: int, train_loss: float, val_loss: float, lr: float, logger: Optional[logging.Logger] = None):
    """
    Log epoch information.
    
    Args:
        epoch: Current epoch.
        train_loss: Training loss.
        val_loss: Validation loss.
        lr: Learning rate.
        logger: Logger instance.
    """
    if logger is None:
        logger = get_logger()
    
    logger.info(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.6f}")


def log_batch(step: int, loss: float, logger: Optional[logging.Logger] = None):
    """
    Log batch information.
    
    Args:
        step: Current step.
        loss: Batch loss.
        logger: Logger instance.
    """
    if logger is None:
        logger = get_logger()
    
    logger.debug(f"Step {step:6d} | Loss: {loss:.4f}")


def log_metrics(metrics: dict, logger: Optional[logging.Logger] = None):
    """
    Log metrics dictionary.
    
    Args:
        metrics: Dictionary of metrics.
        logger: Logger instance.
    """
    if logger is None:
        logger = get_logger()
    
    logger.info("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")


def log_error(error: Exception, logger: Optional[logging.Logger] = None):
    """
    Log error with traceback.
    
    Args:
        error: Exception to log.
        logger: Logger instance.
    """
    if logger is None:
        logger = get_logger()
    
    logger.error(f"Error occurred: {error}")
    logger.exception("Full traceback:") 