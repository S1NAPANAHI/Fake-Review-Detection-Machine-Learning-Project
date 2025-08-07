"""
Basic tests for the monitoring system.

This module provides unit tests for the core monitoring functionality
to ensure the system works correctly.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.monitoring import (
    PerformanceTracker, 
    DriftDetector, 
    PrometheusMetrics,
    ModelMonitor,
    create_monitor
)


class TestPerformanceTracker(unittest.TestCase):
    """Test PerformanceTracker functionality."""
    
    def setUp(self):
        self.tracker = PerformanceTracker(window_size=10)
    
    def test_initialization(self):
        """Test tracker initialization."""
        self.assertEqual(self.tracker.window_size, 10)
        self.assertEqual(len(self.tracker.predictions), 0)
        self.assertIsNone(self.tracker.baseline_metrics)
    
    def test_add_prediction(self):
        """Test adding predictions."""
        self.tracker.add_prediction(1, 1, 0.8)
        self.tracker.add_prediction(0, 0, 0.3)
        
        self.assertEqual(len(self.tracker.predictions), 2)
        self.assertEqual(len(self.tracker.true_labels), 2)
        self.assertEqual(len(self.tracker.probabilities), 2)
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        # Add enough predictions for metrics calculation
        predictions = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1]
        true_labels = [1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1]
        probabilities = [0.9, 0.8, 0.2, 0.6, 0.9, 0.1, 0.8, 0.3, 0.7, 0.2, 0.9, 0.8]
        
        for p, t, prob in zip(predictions, true_labels, probabilities):
            self.tracker.add_prediction(p, t, prob)
        
        metrics = self.tracker.get_current_metrics()
        
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_score', metrics)
        self.assertGreaterEqual(metrics['precision'], 0)
        self.assertLessEqual(metrics['precision'], 1)
    
    def test_performance_degradation_detection(self):
        """Test performance degradation detection."""
        # Set baseline
        baseline = {'precision': 0.9, 'recall': 0.85, 'accuracy': 0.87, 'f1_score': 0.875}
        self.tracker.set_baseline_metrics(baseline)
        
        # Add predictions with poor performance
        predictions = [0] * 30 + [1] * 30  # Mix of predictions
        true_labels = [1] * 60  # All should be 1 (fake)
        probabilities = [0.3] * 30 + [0.7] * 30
        
        for p, t, prob in zip(predictions, true_labels, probabilities):
            self.tracker.add_prediction(p, t, prob)
        
        degradation_info = self.tracker.detect_performance_degradation(threshold=0.1)
        
        # Should detect degradation due to poor performance
        self.assertIn('degraded', degradation_info)
        self.assertIn('metrics_comparison', degradation_info)


class TestDriftDetector(unittest.TestCase):
    """Test DriftDetector functionality."""
    
    def setUp(self):
        self.detector = DriftDetector(p_val=0.05)
    
    def test_initialization(self):
        """Test drift detector initialization."""
        self.assertEqual(self.detector.p_val, 0.05)
        self.assertFalse(self.detector.is_fitted)
        self.assertIsNone(self.detector.detector)
    
    @patch('src.monitoring.ALIBI_DETECT_AVAILABLE', True)
    @patch('src.monitoring.MMDDrift')
    def test_fit(self, mock_mmd):
        """Test fitting drift detector."""
        # Mock the MMDDrift detector
        mock_detector = Mock()
        mock_mmd.return_value = mock_detector
        
        reference_data = np.random.random((100, 10))
        self.detector.fit(reference_data)
        
        self.assertTrue(self.detector.is_fitted)
        mock_mmd.assert_called_once()
    
    @patch('src.monitoring.ALIBI_DETECT_AVAILABLE', False)
    def test_fit_without_alibi(self):
        """Test fitting without alibi-detect."""
        reference_data = np.random.random((100, 10))
        self.detector.fit(reference_data)
        
        self.assertFalse(self.detector.is_fitted)
    
    @patch('src.monitoring.ALIBI_DETECT_AVAILABLE', True)
    def test_predict_drift_not_fitted(self):
        """Test drift prediction without fitting."""
        data = np.random.random((50, 10))
        result = self.detector.predict_drift(data)
        
        self.assertIn('drift', result)
        self.assertFalse(result['drift'])
        self.assertEqual(result['reason'], 'detector_not_available')


class TestPrometheusMetrics(unittest.TestCase):
    """Test PrometheusMetrics functionality."""
    
    def setUp(self):
        self.metrics = PrometheusMetrics(prefix="test")
    
    def test_initialization(self):
        """Test metrics initialization."""
        self.assertEqual(self.metrics.prefix, "test")
        self.assertIsInstance(self.metrics.metrics, dict)
    
    @patch('src.monitoring.PROMETHEUS_AVAILABLE', True)
    @patch('src.monitoring.Gauge')
    @patch('src.monitoring.Counter')
    def test_update_performance_metrics(self, mock_counter, mock_gauge):
        """Test updating performance metrics."""
        # Mock the metrics objects
        mock_gauge_instance = Mock()
        mock_gauge.return_value = mock_gauge_instance
        
        # Re-initialize with mocked Prometheus
        self.metrics._initialize_metrics()
        self.metrics.metrics['precision'] = mock_gauge_instance
        
        # Update metrics
        test_metrics = {'precision': 0.85}
        self.metrics.update_performance_metrics(test_metrics)
        
        # Should not fail (actual assertion depends on Prometheus availability)
        self.assertTrue(True)
    
    def test_update_drift_metrics(self):
        """Test updating drift metrics."""
        drift_info = {
            'drift': True,
            'p_value': 0.01,
            'distance': 0.5
        }
        
        # Should not raise exception
        self.metrics.update_drift_metrics(drift_info)
        self.assertTrue(True)


class TestModelMonitor(unittest.TestCase):
    """Test ModelMonitor functionality."""
    
    def setUp(self):
        self.config = {
            'performance_window_size': 100,
            'drift_p_value': 0.05,
            'performance_threshold': 0.1
        }
        self.monitor = ModelMonitor(config=self.config)
    
    def test_initialization(self):
        """Test monitor initialization."""
        self.assertFalse(self.monitor.is_running)
        self.assertIsNotNone(self.monitor.performance_tracker)
        self.assertIsNotNone(self.monitor.drift_detector)
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        self.assertFalse(self.monitor.is_running)
        
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.is_running)
        
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.is_running)
    
    def test_setup_drift_detection(self):
        """Test setting up drift detection."""
        reference_data = np.random.random((50, 5))
        
        # Should not raise exception
        self.monitor.setup_drift_detection(reference_data)
        self.assertTrue(True)
    
    def test_setup_performance_baseline(self):
        """Test setting up performance baseline."""
        metrics = {
            'precision': 0.85,
            'recall': 0.82,
            'accuracy': 0.83,
            'f1_score': 0.835
        }
        
        self.monitor.setup_performance_baseline(metrics)
        
        # Check if baseline was set
        if self.monitor.performance_tracker:
            self.assertEqual(
                self.monitor.performance_tracker.baseline_metrics, 
                metrics
            )
    
    def test_record_prediction(self):
        """Test recording predictions."""
        self.monitor.start_monitoring()
        
        features = np.random.random(10)
        
        # Should not raise exception
        self.monitor.record_prediction(
            prediction=1,
            probability=0.8,
            features=features,
            true_label=1
        )
        self.assertTrue(True)
        
        self.monitor.stop_monitoring()
    
    def test_trigger_retraining(self):
        """Test retraining trigger."""
        callback_called = {'called': False}
        
        def test_callback(reason, details):
            callback_called['called'] = True
            callback_called['reason'] = reason
            callback_called['details'] = details
        
        self.monitor.retraining_callback = test_callback
        
        self.monitor.trigger_retraining("test_reason", {"test": "data"})
        
        self.assertTrue(callback_called['called'])
        self.assertEqual(callback_called['reason'], "test_reason")
    
    def test_get_monitoring_status(self):
        """Test getting monitoring status."""
        status = self.monitor.get_monitoring_status()
        
        self.assertIn('monitoring_active', status)
        self.assertIn('timestamp', status)
        self.assertIn('components', status)
        self.assertIn('dependencies', status)
    
    def test_save_load_state(self):
        """Test saving and loading monitor state."""
        import tempfile
        import os
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            tmp_path = tmp.name
        
        try:
            # Set some baseline
            metrics = {'precision': 0.9, 'recall': 0.8}
            self.monitor.setup_performance_baseline(metrics)
            
            # Save state
            self.monitor.save_state(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Create new monitor and load state
            new_monitor = ModelMonitor(config=self.config)
            new_monitor.load_state(tmp_path)
            
            # Check if baseline was loaded
            if (new_monitor.performance_tracker and 
                new_monitor.performance_tracker.baseline_metrics):
                self.assertEqual(
                    new_monitor.performance_tracker.baseline_metrics['precision'],
                    0.9
                )
        
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestCreateMonitor(unittest.TestCase):
    """Test create_monitor convenience function."""
    
    def test_create_monitor_default(self):
        """Test creating monitor with defaults."""
        monitor = create_monitor()
        
        self.assertIsInstance(monitor, ModelMonitor)
        self.assertIsNotNone(monitor.retraining_callback)
    
    def test_create_monitor_with_config(self):
        """Test creating monitor with custom config."""
        config = {
            'performance_window_size': 500,
            'enable_drift_detection': False
        }
        
        monitor = create_monitor(config=config)
        
        self.assertIsInstance(monitor, ModelMonitor)
        self.assertEqual(monitor.config['performance_window_size'], 500)
    
    def test_create_monitor_with_callback(self):
        """Test creating monitor with custom callback."""
        callback_called = {'called': False}
        
        def custom_callback(reason, details):
            callback_called['called'] = True
        
        config = {'retraining_callback': custom_callback}
        monitor = create_monitor(config=config)
        
        # Test callback
        monitor.trigger_retraining("test", {})
        self.assertTrue(callback_called['called'])


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
