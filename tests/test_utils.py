"""
Unit tests for the utils module.

This module tests core utility functions including path resolution,
configuration loading, data I/O, logging setup, validation, and metrics collection.
Uses small synthetic fixtures and validates function outputs and shapes.
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
import numpy as np
import yaml
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    get_project_root, resolve_path, ensure_dir, load_settings, get_setting,
    set_random_seed, save_joblib, load_joblib, save_parquet, load_parquet,
    save_json, load_json, setup_logging, parse_size, timing_decorator,
    validate_dataframe, validate_file_path, validate_config_keys,
    MetricsCollector, metrics_timer, get_timestamp, safe_division,
    memory_usage, suppress_warnings
)


class TestPathResolution(unittest.TestCase):
    """Test path resolution and project structure utilities."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Clean up test directory
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_get_project_root(self):
        """Test project root detection."""
        # This should return a valid path
        root = get_project_root()
        self.assertIsInstance(root, Path)
        self.assertTrue(root.exists())
        
    def test_resolve_path(self):
        """Test path resolution."""
        # Test absolute path
        abs_path = resolve_path("/test/path")
        self.assertTrue(abs_path.is_absolute())
        
        # Test relative path
        rel_path = resolve_path("test_file.txt")
        self.assertTrue(rel_path.is_absolute())
        
        # Test with base path
        base = Path(self.test_dir)
        resolved = resolve_path("test.txt", base_path=base)
        expected = base / "test.txt"
        self.assertEqual(resolved, expected.resolve())
    
    def test_ensure_dir(self):
        """Test directory creation."""
        test_path = Path(self.test_dir) / "new_dir" / "nested"
        
        # Directory should not exist initially
        self.assertFalse(test_path.exists())
        
        # Create directory
        result = ensure_dir(test_path)
        
        # Should return the path and directory should exist
        self.assertEqual(result, test_path)
        self.assertTrue(test_path.exists())
        self.assertTrue(test_path.is_dir())


class TestConfigurationLoading(unittest.TestCase):
    """Test configuration loading and settings management."""
    
    def setUp(self):
        self.test_settings = {
            'environment': {
                'random_seed': 42,
                'reproducible': True
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'ml': {
                'model': {
                    'classifier': {
                        'n_estimators': 100
                    }
                }
            }
        }
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    def test_load_settings_success(self, mock_yaml_load, mock_file):
        """Test successful settings loading."""
        mock_yaml_load.return_value = self.test_settings
        
        # Mock file existence
        with patch('pathlib.Path.exists', return_value=True):
            settings = load_settings("test_config.yaml")
        
        self.assertEqual(settings, self.test_settings)
        mock_file.assert_called_once()
        mock_yaml_load.assert_called_once()
    
    @patch('pathlib.Path.exists', return_value=False)
    def test_load_settings_file_not_found(self, mock_exists):
        """Test settings loading when file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            load_settings("nonexistent.yaml")
    
    def test_get_setting_nested(self):
        """Test getting nested settings with dot notation."""
        with patch('src.utils.load_settings', return_value=self.test_settings):
            # Test nested access
            value = get_setting('ml.model.classifier.n_estimators')
            self.assertEqual(value, 100)
            
            # Test with default
            default_value = get_setting('nonexistent.key', 'default')
            self.assertEqual(default_value, 'default')
            
            # Test top-level key
            env = get_setting('environment')
            self.assertEqual(env, self.test_settings['environment'])


class TestDataIO(unittest.TestCase):
    """Test data I/O utilities."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_data = {
            'numbers': [1, 2, 3, 4, 5],
            'strings': ['a', 'b', 'c'],
            'nested': {'key': 'value'}
        }
        self.test_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_save_load_joblib(self):
        """Test joblib save and load functionality."""
        test_path = Path(self.test_dir) / "test_object.joblib"
        
        # Save object
        saved_path = save_joblib(self.test_data, test_path)
        
        # Verify file was created
        self.assertEqual(saved_path, test_path)
        self.assertTrue(test_path.exists())
        
        # Load object
        loaded_data = load_joblib(test_path)
        
        # Verify data integrity
        self.assertEqual(loaded_data, self.test_data)
    
    def test_save_load_parquet(self):
        """Test parquet save and load functionality."""
        test_path = Path(self.test_dir) / "test_data.parquet"
        
        # Save DataFrame
        saved_path = save_parquet(self.test_df, test_path)
        
        # Verify file was created
        self.assertTrue(str(saved_path).endswith('.parquet'))
        self.assertTrue(saved_path.exists())
        
        # Load DataFrame
        loaded_df = load_parquet(saved_path)
        
        # Verify data integrity and shape
        self.assertEqual(loaded_df.shape, self.test_df.shape)
        pd.testing.assert_frame_equal(loaded_df, self.test_df)
    
    def test_save_load_json(self):
        """Test JSON save and load functionality."""
        test_path = Path(self.test_dir) / "test_data.json"
        
        # Save JSON
        saved_path = save_json(self.test_data, test_path)
        
        # Verify file was created
        self.assertTrue(str(saved_path).endswith('.json'))
        self.assertTrue(saved_path.exists())
        
        # Load JSON
        loaded_data = load_json(saved_path)
        
        # Verify data integrity
        self.assertEqual(loaded_data, self.test_data)
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent files raises appropriate errors."""
        nonexistent_path = Path(self.test_dir) / "nonexistent.joblib"
        
        with self.assertRaises(FileNotFoundError):
            load_joblib(nonexistent_path)
        
        with self.assertRaises(FileNotFoundError):
            load_parquet(nonexistent_path)
        
        with self.assertRaises(FileNotFoundError):
            load_json(nonexistent_path)


class TestRandomSeed(unittest.TestCase):
    """Test random seed management."""
    
    def test_set_random_seed(self):
        """Test setting random seed."""
        # Test with explicit seed
        seed = set_random_seed(123)
        self.assertEqual(seed, 123)
        
        # Test numpy random state
        np.random.seed(123)
        expected_value = np.random.random()
        
        set_random_seed(123)
        actual_value = np.random.random()
        self.assertEqual(actual_value, expected_value)
    
    @patch('src.utils.get_setting', return_value=42)
    def test_set_random_seed_from_config(self, mock_get_setting):
        """Test setting random seed from configuration."""
        seed = set_random_seed(None)
        self.assertEqual(seed, 42)
        mock_get_setting.assert_called_once_with('environment.random_seed', 42)


class TestLogging(unittest.TestCase):
    """Test logging configuration."""
    
    @patch('src.utils.get_setting')
    def test_setup_logging(self, mock_get_setting):
        """Test logging setup."""
        # Mock settings
        mock_get_setting.side_effect = lambda key, default: {
            'logging.level': 'INFO',
            'logging.format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'paths.logs': 'logs'
        }.get(key, default)
        
        # Setup logging
        logger = setup_logging("test_logger")
        
        # Verify logger properties
        self.assertEqual(logger.name, "test_logger")
        self.assertTrue(len(logger.handlers) > 0)
    
    def test_parse_size(self):
        """Test size string parsing."""
        # Test various formats
        self.assertEqual(parse_size("10MB"), 10 * 1024 * 1024)
        self.assertEqual(parse_size("5KB"), 5 * 1024)
        self.assertEqual(parse_size("2GB"), 2 * 1024 * 1024 * 1024)
        self.assertEqual(parse_size("1024B"), 1024)
        
        # Test shorthand formats
        self.assertEqual(parse_size("10M"), 10 * 1024 * 1024)
        self.assertEqual(parse_size("5K"), 5 * 1024)
        
        # Test invalid format (should return default)
        default_size = 10 * 1024 * 1024  # 10MB
        self.assertEqual(parse_size("invalid"), default_size)


class TestDecoratorsAndTiming(unittest.TestCase):
    """Test decorators and timing utilities."""
    
    def test_timing_decorator(self):
        """Test timing decorator functionality."""
        # Define a test function
        @timing_decorator
        def test_function(duration=0.01):
            import time
            time.sleep(duration)
            return "completed"
        
        # Test the decorated function
        result = test_function(0.01)
        self.assertEqual(result, "completed")
    
    def test_get_timestamp(self):
        """Test timestamp generation."""
        # Test default format
        timestamp = get_timestamp()
        self.assertIsInstance(timestamp, str)
        self.assertEqual(len(timestamp), 15)  # YYYYMMDD_HHMMSS format
        
        # Test custom format
        custom_timestamp = get_timestamp("%Y-%m-%d")
        self.assertIsInstance(custom_timestamp, str)
        self.assertEqual(len(custom_timestamp), 10)  # YYYY-MM-DD format


class TestValidation(unittest.TestCase):
    """Test validation utilities."""
    
    def setUp(self):
        self.valid_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'required_col': [10, 20, 30]
        })
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_validate_dataframe_valid(self):
        """Test DataFrame validation with valid data."""
        # Should not raise any exception
        validate_dataframe(self.valid_df)
        validate_dataframe(self.valid_df, required_columns=['col1', 'col2'])
        validate_dataframe(self.valid_df, min_rows=2)
    
    def test_validate_dataframe_invalid(self):
        """Test DataFrame validation with invalid data."""
        # Test non-DataFrame input
        with self.assertRaises(ValueError):
            validate_dataframe("not a dataframe")
        
        # Test insufficient rows
        with self.assertRaises(ValueError):
            validate_dataframe(self.valid_df, min_rows=5)
        
        # Test missing required columns
        with self.assertRaises(ValueError):
            validate_dataframe(self.valid_df, required_columns=['nonexistent_col'])
    
    def test_validate_file_path(self):
        """Test file path validation."""
        # Create a test file
        test_file = Path(self.test_dir) / "test.txt"
        test_file.write_text("test content")
        
        # Test existing file validation
        result = validate_file_path(test_file, must_exist=True)
        self.assertEqual(result, test_file)
        
        # Test non-existing file requirement
        nonexistent = Path(self.test_dir) / "nonexistent.txt"
        with self.assertRaises(ValueError):
            validate_file_path(nonexistent, must_exist=True)
        
        # Test extension validation
        validate_file_path(test_file, allowed_extensions=['.txt'])
        
        with self.assertRaises(ValueError):
            validate_file_path(test_file, allowed_extensions=['.pdf'])
    
    def test_validate_config_keys(self):
        """Test configuration key validation."""
        config = {
            'level1': {
                'level2': {
                    'key': 'value'
                }
            },
            'simple_key': 'simple_value'
        }
        
        # Test valid keys
        validate_config_keys(config, ['level1.level2.key', 'simple_key'])
        
        # Test missing keys
        with self.assertRaises(ValueError):
            validate_config_keys(config, ['nonexistent.key'])


class TestMetricsCollector(unittest.TestCase):
    """Test metrics collection utilities."""
    
    def setUp(self):
        self.metrics = MetricsCollector(enabled=True)
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization."""
        self.assertTrue(self.metrics.enabled)
        self.assertEqual(len(self.metrics._counters), 0)
        self.assertEqual(len(self.metrics._gauges), 0)
        self.assertEqual(len(self.metrics._histograms), 0)
    
    def test_increment_counter(self):
        """Test counter increment functionality."""
        # Increment counter without labels
        self.metrics.increment_counter('test_counter', 5)
        
        # Increment counter with labels
        self.metrics.increment_counter('test_counter', 3, {'status': 'success'})
        
        # Verify counters were recorded
        self.assertTrue(len(self.metrics._counters) > 0)
    
    def test_set_gauge(self):
        """Test gauge setting functionality."""
        self.metrics.set_gauge('test_gauge', 42.5)
        self.metrics.set_gauge('test_gauge', 50.0, {'type': 'memory'})
        
        # Verify gauges were recorded
        self.assertTrue(len(self.metrics._gauges) > 0)
    
    def test_observe_histogram(self):
        """Test histogram observation functionality."""
        values = [1.2, 2.3, 3.4, 4.5, 5.6]
        for value in values:
            self.metrics.observe_histogram('test_histogram', value)
        
        # Verify histogram was recorded
        self.assertTrue(len(self.metrics._histograms) > 0)
    
    def test_get_metrics_summary(self):
        """Test metrics summary generation."""
        # Add some test metrics
        self.metrics.increment_counter('requests', 10)
        self.metrics.set_gauge('memory_usage', 75.5)
        self.metrics.observe_histogram('response_time', 0.5)
        self.metrics.observe_histogram('response_time', 1.2)
        
        # Get summary
        summary = self.metrics.get_metrics_summary()
        
        # Verify summary structure
        self.assertIn('counters', summary)
        self.assertIn('gauges', summary)
        self.assertIn('histograms', summary)
        
        # Verify histogram calculations
        histogram_data = list(summary['histograms'].values())[0]
        self.assertEqual(histogram_data['count'], 2)
        self.assertEqual(histogram_data['sum'], 1.7)
        self.assertEqual(histogram_data['avg'], 0.85)
    
    def test_reset_metrics(self):
        """Test metrics reset functionality."""
        # Add some metrics
        self.metrics.increment_counter('test', 1)
        self.metrics.set_gauge('test', 1)
        self.metrics.observe_histogram('test', 1)
        
        # Verify metrics exist
        self.assertTrue(len(self.metrics._counters) > 0)
        
        # Reset metrics
        self.metrics.reset_metrics()
        
        # Verify metrics were cleared
        self.assertEqual(len(self.metrics._counters), 0)
        self.assertEqual(len(self.metrics._gauges), 0)
        self.assertEqual(len(self.metrics._histograms), 0)
    
    def test_disabled_metrics_collector(self):
        """Test metrics collector when disabled."""
        disabled_metrics = MetricsCollector(enabled=False)
        
        # Operations should not record anything
        disabled_metrics.increment_counter('test', 1)
        disabled_metrics.set_gauge('test', 1)
        disabled_metrics.observe_histogram('test', 1)
        
        # Verify nothing was recorded
        self.assertEqual(len(disabled_metrics._counters), 0)
        self.assertEqual(len(disabled_metrics._gauges), 0)
        self.assertEqual(len(disabled_metrics._histograms), 0)


class TestUtilityFunctions(unittest.TestCase):
    """Test miscellaneous utility functions."""
    
    def test_safe_division(self):
        """Test safe division utility."""
        # Test normal division
        self.assertEqual(safe_division(10, 2), 5.0)
        self.assertEqual(safe_division(7, 3), 7/3)
        
        # Test division by zero
        self.assertEqual(safe_division(10, 0), 0.0)  # default
        self.assertEqual(safe_division(10, 0, -1), -1)  # custom default
        
        # Test invalid inputs
        self.assertEqual(safe_division(None, 1), 0.0)
        self.assertEqual(safe_division(1, None), 0.0)
    
    def test_memory_usage(self):
        """Test memory usage utility."""
        memory_info = memory_usage()
        
        # Should return a dictionary with expected keys
        self.assertIsInstance(memory_info, dict)
        self.assertIn('rss_mb', memory_info)
        self.assertIn('vms_mb', memory_info)
        self.assertIn('percent', memory_info)
        
        # Values should be non-negative numbers
        for key, value in memory_info.items():
            self.assertGreaterEqual(value, 0)
    
    def test_suppress_warnings(self):
        """Test warning suppression utility."""
        import warnings
        
        # Test as decorator
        @suppress_warnings(UserWarning)
        def function_with_warning():
            warnings.warn("Test warning", UserWarning)
            return "success"
        
        # Should not raise warning
        result = function_with_warning()
        self.assertEqual(result, "success")


class TestMetricsTimer(unittest.TestCase):
    """Test metrics timer decorator."""
    
    def setUp(self):
        # Reset global metrics
        from src.utils import metrics
        metrics.reset_metrics()
    
    def test_metrics_timer_success(self):
        """Test metrics timer for successful execution."""
        @metrics_timer('test_operation')
        def test_function():
            import time
            time.sleep(0.01)
            return "success"
        
        result = test_function()
        self.assertEqual(result, "success")
        
        # Check that metrics were recorded
        from src.utils import metrics
        summary = metrics.get_metrics_summary()
        self.assertTrue(len(summary['histograms']) > 0)
    
    def test_metrics_timer_exception(self):
        """Test metrics timer for failed execution."""
        @metrics_timer('test_operation_fail', {'component': 'test'})
        def failing_function():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            failing_function()
        
        # Check that error metrics were recorded
        from src.utils import metrics
        summary = metrics.get_metrics_summary()
        self.assertTrue(len(summary['counters']) > 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
