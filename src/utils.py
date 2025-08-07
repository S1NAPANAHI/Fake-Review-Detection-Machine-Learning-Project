"""
Utility functions for the Fake Review Detection System.

This module provides helper functions for:
- Path resolution and project structure
- Data saving/loading (joblib, parquet)
- Random seed management
- Logging configuration
- Timing decorator
- Input validation
- Prometheus metrics stubs
- Configuration loading
"""

import os
import sys
import time
import random
import logging
import functools
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Callable
from datetime import datetime

import yaml
import joblib
import pandas as pd
import numpy as np
from logging.handlers import RotatingFileHandler


# =============================================================================
# Path Resolution and Project Structure
# =============================================================================

def get_project_root() -> Path:
    """
    Get the project root directory by looking for specific marker files.
    
    Returns:
        Path: Path to the project root directory
        
    Raises:
        RuntimeError: If project root cannot be determined
    """
    # Start from current file's directory
    current_path = Path(__file__).resolve()
    
    # Look for marker files that indicate project root
    marker_files = [
        'requirements.txt',
        'README.md',
        'config/settings.yaml',
        '.gitignore',
        'Dockerfile'
    ]
    
    # Search up the directory tree
    for path in [current_path] + list(current_path.parents):
        if any((path / marker).exists() for marker in marker_files):
            return path
    
    # Fallback: assume parent of src directory is root
    if current_path.parent.name == 'src':
        return current_path.parent.parent
    
    raise RuntimeError(
        "Could not determine project root. "
        "Make sure you're running from within the project directory."
    )


def resolve_path(path: Union[str, Path], base_path: Optional[Path] = None) -> Path:
    """
    Resolve a path relative to the project root or a specified base path.
    
    Args:
        path: Path to resolve (can be relative or absolute)
        base_path: Base path to resolve relative paths against (defaults to project root)
        
    Returns:
        Path: Resolved absolute path
    """
    path = Path(path)
    
    # If already absolute, return as-is
    if path.is_absolute():
        return path
    
    # Use project root as base if not specified
    if base_path is None:
        base_path = get_project_root()
    
    return (base_path / path).resolve()


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path: The directory path
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# Configuration Loading
# =============================================================================

_SETTINGS_CACHE = None


def load_settings(config_path: Optional[Union[str, Path]] = None, 
                 force_reload: bool = False) -> Dict[str, Any]:
    """
    Load configuration settings from YAML file.
    
    Args:
        config_path: Path to settings file (defaults to config/settings.yaml)
        force_reload: Force reload even if settings are cached
        
    Returns:
        Dict: Configuration settings
        
    Raises:
        FileNotFoundError: If settings file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    global _SETTINGS_CACHE
    
    # Return cached settings if available and not forcing reload
    if _SETTINGS_CACHE is not None and not force_reload:
        return _SETTINGS_CACHE
    
    # Default to config/settings.yaml in project root
    if config_path is None:
        config_path = get_project_root() / "config" / "settings.yaml"
    else:
        config_path = resolve_path(config_path)
    
    # Check if file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Settings file not found: {config_path}")
    
    # Load YAML configuration
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            _SETTINGS_CACHE = yaml.safe_load(f)
            return _SETTINGS_CACHE
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing settings file {config_path}: {e}")


def get_setting(key: str, default: Any = None) -> Any:
    """
    Get a specific setting value using dot notation.
    
    Args:
        key: Setting key in dot notation (e.g., 'ml.model.classifier.n_estimators')
        default: Default value if key not found
        
    Returns:
        Any: Setting value or default
    """
    settings = load_settings()
    
    # Navigate through nested dictionary using dot notation
    value = settings
    for key_part in key.split('.'):
        if isinstance(value, dict) and key_part in value:
            value = value[key_part]
        else:
            return default
    
    return value


# =============================================================================
# Random Seed Management
# =============================================================================

def set_random_seed(seed: Optional[int] = None) -> int:
    """
    Set random seeds for reproducibility across different libraries.
    
    Args:
        seed: Random seed value (uses setting from config if None)
        
    Returns:
        int: The seed value that was set
    """
    if seed is None:
        seed = get_setting('environment.random_seed', 42)
    
    # Set seeds for different libraries
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set seeds for ML libraries if available
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    return seed


# =============================================================================
# Data Saving and Loading Helpers
# =============================================================================

def save_joblib(obj: Any, filepath: Union[str, Path], 
                create_dirs: bool = True) -> Path:
    """
    Save object using joblib with error handling.
    
    Args:
        obj: Object to save
        filepath: Path to save file
        create_dirs: Create parent directories if they don't exist
        
    Returns:
        Path: Path where object was saved
    """
    filepath = resolve_path(filepath)
    
    if create_dirs:
        ensure_dir(filepath.parent)
    
    try:
        joblib.dump(obj, filepath)
        return filepath
    except Exception as e:
        raise IOError(f"Failed to save joblib file {filepath}: {e}")


def load_joblib(filepath: Union[str, Path]) -> Any:
    """
    Load object using joblib with error handling.
    
    Args:
        filepath: Path to load file from
        
    Returns:
        Any: Loaded object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If loading fails
    """
    filepath = resolve_path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Joblib file not found: {filepath}")
    
    try:
        return joblib.load(filepath)
    except Exception as e:
        raise IOError(f"Failed to load joblib file {filepath}: {e}")


def save_parquet(df: pd.DataFrame, filepath: Union[str, Path], 
                create_dirs: bool = True, **kwargs) -> Path:
    """
    Save DataFrame as parquet file with error handling.
    
    Args:
        df: DataFrame to save
        filepath: Path to save file
        create_dirs: Create parent directories if they don't exist
        **kwargs: Additional arguments for to_parquet()
        
    Returns:
        Path: Path where DataFrame was saved
    """
    filepath = resolve_path(filepath)
    
    if create_dirs:
        ensure_dir(filepath.parent)
    
    try:
        # Ensure .parquet extension
        if not filepath.suffix == '.parquet':
            filepath = filepath.with_suffix('.parquet')
        
        df.to_parquet(filepath, **kwargs)
        return filepath
    except Exception as e:
        raise IOError(f"Failed to save parquet file {filepath}: {e}")


def load_parquet(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load DataFrame from parquet file with error handling.
    
    Args:
        filepath: Path to load file from
        **kwargs: Additional arguments for read_parquet()
        
    Returns:
        pd.DataFrame: Loaded DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If loading fails
    """
    filepath = resolve_path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Parquet file not found: {filepath}")
    
    try:
        return pd.read_parquet(filepath, **kwargs)
    except Exception as e:
        raise IOError(f"Failed to load parquet file {filepath}: {e}")


def save_json(obj: Any, filepath: Union[str, Path], 
              create_dirs: bool = True, **kwargs) -> Path:
    """
    Save object as JSON file with error handling.
    
    Args:
        obj: Object to save (must be JSON serializable)
        filepath: Path to save file
        create_dirs: Create parent directories if they don't exist
        **kwargs: Additional arguments for json.dump()
        
    Returns:
        Path: Path where object was saved
    """
    import json
    
    filepath = resolve_path(filepath)
    
    if create_dirs:
        ensure_dir(filepath.parent)
    
    try:
        # Ensure .json extension
        if not filepath.suffix == '.json':
            filepath = filepath.with_suffix('.json')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, **kwargs)
        
        return filepath
    except Exception as e:
        raise IOError(f"Failed to save JSON file {filepath}: {e}")


def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load object from JSON file with error handling.
    
    Args:
        filepath: Path to load file from
        
    Returns:
        Any: Loaded object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If loading fails
    """
    import json
    
    filepath = resolve_path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise IOError(f"Failed to load JSON file {filepath}: {e}")


# =============================================================================
# Logging Configuration
# =============================================================================

def setup_logging(name: Optional[str] = None, 
                 level: Optional[str] = None,
                 log_file: Optional[Union[str, Path]] = None) -> logging.Logger:
    """
    Set up logging configuration based on settings.
    
    Args:
        name: Logger name (defaults to __name__)
        level: Log level (uses setting from config if None)
        log_file: Log file path (uses setting from config if None)
        
    Returns:
        logging.Logger: Configured logger
    """
    if name is None:
        name = __name__
    
    # Get configuration from settings
    if level is None:
        level = get_setting('logging.level', 'INFO')
    
    log_format = get_setting(
        'logging.format', 
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified or configured)
    if log_file is None:
        logs_dir = get_setting('paths.logs', 'logs')
        logs_path = resolve_path(logs_dir)
        ensure_dir(logs_path)
        log_file = logs_path / f"{name}.log"
    else:
        log_file = resolve_path(log_file)
        ensure_dir(log_file.parent)
    
    # Rotating file handler
    if get_setting('logging.file_rotation', True):
        max_bytes = parse_size(get_setting('logging.max_file_size', '10MB'))
        backup_count = get_setting('logging.backup_count', 5)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
    else:
        file_handler = logging.FileHandler(log_file)
    
    file_handler.setLevel(getattr(logging, level.upper()))
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def parse_size(size_str: str) -> int:
    """
    Parse size string like '10MB' to bytes.
    
    Args:
        size_str: Size string with unit
        
    Returns:
        int: Size in bytes
    """
    size_str = str(size_str).upper().strip()
    
    # Handle various unit formats
    units = {
        'B': 1,
        'KB': 1024,
        'K': 1024,  # Handle 'K' as KB
        'MB': 1024**2,
        'M': 1024**2,  # Handle 'M' as MB
        'GB': 1024**3,
        'G': 1024**3,  # Handle 'G' as GB
        'TB': 1024**4,
        'T': 1024**4   # Handle 'T' as TB
    }
    
    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            number_str = size_str[:-len(unit)].strip()
            try:
                return int(float(number_str) * multiplier)
            except ValueError:
                # If we can't parse the number, try the next unit
                continue
    
    # Try to parse as plain number (assume bytes)
    try:
        return int(float(size_str))
    except ValueError:
        # Default fallback
        return 10 * 1024 * 1024  # 10MB default


# =============================================================================
# Timing Decorator
# =============================================================================

def timing_decorator(func: Optional[Callable] = None, *, 
                    logger: Optional[logging.Logger] = None,
                    log_level: str = 'INFO') -> Callable:
    """
    Decorator to measure and log function execution time.
    
    Args:
        func: Function to decorate
        logger: Logger to use (creates one if None)
        log_level: Log level for timing messages
        
    Returns:
        Callable: Decorated function
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Get or create logger
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(f.__module__)
            
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
                execution_time = time.time() - start_time
                
                log_func = getattr(logger, log_level.lower())
                log_func(f"Function '{f.__name__}' executed in {execution_time:.4f} seconds")
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Function '{f.__name__}' failed after {execution_time:.4f} seconds: {e}")
                raise
        return wrapper
    
    # Handle both @timing_decorator and @timing_decorator() usage
    if func is None:
        return decorator
    else:
        return decorator(func)


# =============================================================================
# Input Validation Helpers
# =============================================================================

def validate_dataframe(df: pd.DataFrame, 
                      required_columns: Optional[List[str]] = None,
                      min_rows: int = 1) -> None:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows, got {len(df)}")
    
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")


def validate_file_path(filepath: Union[str, Path], 
                      must_exist: bool = False,
                      allowed_extensions: Optional[List[str]] = None) -> Path:
    """
    Validate file path.
    
    Args:
        filepath: Path to validate
        must_exist: Whether file must already exist
        allowed_extensions: List of allowed file extensions
        
    Returns:
        Path: Validated path
        
    Raises:
        ValueError: If validation fails
    """
    filepath = Path(filepath)
    
    if must_exist and not filepath.exists():
        raise ValueError(f"File does not exist: {filepath}")
    
    if allowed_extensions:
        extension = filepath.suffix.lower()
        if extension not in [ext.lower() for ext in allowed_extensions]:
            raise ValueError(
                f"File extension '{extension}' not allowed. "
                f"Allowed extensions: {allowed_extensions}"
            )
    
    return filepath


def validate_config_keys(config: Dict[str, Any], 
                        required_keys: List[str],
                        config_name: str = "configuration") -> None:
    """
    Validate that required configuration keys are present.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys (supports dot notation)
        config_name: Name of configuration for error messages
        
    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = []
    
    for key in required_keys:
        value = config
        for key_part in key.split('.'):
            if isinstance(value, dict) and key_part in value:
                value = value[key_part]
            else:
                missing_keys.append(key)
                break
    
    if missing_keys:
        raise ValueError(
            f"Missing required keys in {config_name}: {missing_keys}"
        )


# =============================================================================
# Prometheus Metrics Stubs
# =============================================================================

class MetricsCollector:
    """
    Stub implementation of metrics collection for monitoring.
    Can be extended to integrate with actual Prometheus client.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._counters = {}
        self._gauges = {}
        self._histograms = {}
        
    def increment_counter(self, name: str, value: float = 1.0, 
                         labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        if not self.enabled:
            return
        
        key = (name, tuple(sorted((labels or {}).items())))
        self._counters[key] = self._counters.get(key, 0) + value
        
    def set_gauge(self, name: str, value: float, 
                  labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value."""
        if not self.enabled:
            return
        
        key = (name, tuple(sorted((labels or {}).items())))
        self._gauges[key] = value
        
    def observe_histogram(self, name: str, value: float, 
                         labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value for histogram metric."""
        if not self.enabled:
            return
        
        key = (name, tuple(sorted((labels or {}).items())))
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        return {
            'counters': dict(self._counters),
            'gauges': dict(self._gauges),
            'histograms': {
                key: {
                    'count': len(values),
                    'sum': sum(values),
                    'avg': sum(values) / len(values) if values else 0,
                    'min': min(values) if values else 0,
                    'max': max(values) if values else 0
                }
                for key, values in self._histograms.items()
            }
        }
        
    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


# Global metrics collector instance
metrics = MetricsCollector()


def metrics_timer(metric_name: str, 
                 labels: Optional[Dict[str, str]] = None) -> Callable:
    """
    Decorator to time function execution and record as histogram metric.
    
    Args:
        metric_name: Name of the metric
        labels: Optional metric labels
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                metrics.observe_histogram(metric_name, execution_time, labels)
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                error_labels = (labels or {}).copy()
                error_labels['status'] = 'error'
                metrics.observe_histogram(metric_name, execution_time, error_labels)
                metrics.increment_counter(f"{metric_name}_errors", labels=labels)
                raise
        return wrapper
    return decorator


# =============================================================================
# Additional Utility Functions
# =============================================================================

def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp as formatted string.
    
    Args:
        format_str: Timestamp format string
        
    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime(format_str)


def safe_division(numerator: float, denominator: float, 
                 default: float = 0.0) -> float:
    """
    Perform division with protection against division by zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division by zero
        
    Returns:
        float: Division result or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.
    
    Returns:
        Dict: Memory usage statistics in MB
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}


def suppress_warnings(category=None):
    """
    Context manager or decorator to suppress warnings.
    
    Args:
        category: Warning category to suppress (None for all)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category)
                return func(*args, **kwargs)
        return wrapper
    
    # Can be used as context manager
    if category is None:
        return warnings.catch_warnings()
    else:
        class WarningsSuppressor:
            def __enter__(self):
                warnings.simplefilter("ignore", category)
                return self
            def __exit__(self, *args):
                warnings.resetwarnings()
        return WarningsSuppressor()


# Initialize logging for this module (use basic logging to avoid circular dependency)
try:
    logger = setup_logging(__name__)
except:
    # Fallback to basic logging if settings are not available
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

# Set random seed on import if configured (with fallback)
try:
    if get_setting('environment.reproducible', True):
        seed = set_random_seed()
        logger.info(f"Random seed set to {seed} for reproducibility")
except:
    # Silently continue if settings are not available
    pass
