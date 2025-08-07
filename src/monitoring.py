"""
Monitoring module for Fake Review Detection System.

This module provides comprehensive monitoring capabilities including:
- Drift detection using alibi-detect (MMDDrift)
- Performance tracking with rolling windows for precision/recall
- Prometheus gauge updates for metrics
- Automatic retraining triggers based on performance degradation

The module integrates with the existing FakeReviewDetector and provides
real-time monitoring of model performance and data distribution changes.
"""

import os
import time
import logging
import threading
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.base import BaseEstimator

# Drift detection
try:
    from alibi_detect import MMDDrift
    from alibi_detect.utils.saving import save_detector, load_detector
    ALIBI_DETECT_AVAILABLE = True
except ImportError:
    MMDDrift = None
    save_detector = None
    load_detector = None
    ALIBI_DETECT_AVAILABLE = False

# Prometheus metrics
try:
    from prometheus_client import Gauge, Counter, Histogram, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    Gauge = Counter = Histogram = Info = None
    PROMETHEUS_AVAILABLE = False

# Import utilities
from . import utils


class PerformanceTracker:
    """
    Rolling window performance tracker for model monitoring.
    
    Tracks precision, recall, accuracy, and F1-score over a rolling window
    to detect performance degradation.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize performance tracker.
        
        Args:
            window_size: Size of rolling window for metrics calculation
        """
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.true_labels = deque(maxlen=window_size)
        self.probabilities = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Current metrics
        self.current_metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'accuracy': 0.0,
            'f1_score': 0.0,
            'count': 0
        }
        
        # Baseline metrics for comparison
        self.baseline_metrics = None
        
        self.logger = logging.getLogger(__name__)
    
    def add_prediction(self, 
                      prediction: Union[int, bool], 
                      true_label: Union[int, bool],
                      probability: float,
                      timestamp: Optional[datetime] = None) -> None:
        """
        Add a new prediction to the rolling window.
        
        Args:
            prediction: Binary prediction (0/1 or False/True)
            true_label: True label (0/1 or False/True)
            probability: Prediction probability
            timestamp: Prediction timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Convert to int for consistency
        prediction = int(prediction)
        true_label = int(true_label)
        
        # Add to rolling window
        self.predictions.append(prediction)
        self.true_labels.append(true_label)
        self.probabilities.append(probability)
        self.timestamps.append(timestamp)
        
        # Update metrics if we have enough data
        if len(self.predictions) >= 10:  # Minimum samples for meaningful metrics
            self._update_metrics()
    
    def _update_metrics(self) -> None:
        """Update current performance metrics."""
        if len(self.predictions) == 0:
            return
            
        try:
            y_true = np.array(list(self.true_labels))
            y_pred = np.array(list(self.predictions))
            
            self.current_metrics = {
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'accuracy': accuracy_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'count': len(self.predictions)
            }
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return self.current_metrics.copy()
    
    def set_baseline_metrics(self, metrics: Dict[str, float]) -> None:
        """Set baseline metrics for performance comparison."""
        self.baseline_metrics = metrics.copy()
    
    def detect_performance_degradation(self, threshold: float = 0.05) -> Dict[str, Any]:
        """
        Detect if current performance has degraded compared to baseline.
        
        Args:
            threshold: Degradation threshold (e.g., 0.05 for 5% drop)
            
        Returns:
            Dict with degradation status and details
        """
        if self.baseline_metrics is None or len(self.predictions) < 50:
            return {"degraded": False, "reason": "insufficient_data"}
        
        degradation_info = {
            "degraded": False,
            "metrics_comparison": {},
            "degraded_metrics": []
        }
        
        for metric in ['precision', 'recall', 'accuracy', 'f1_score']:
            baseline_value = self.baseline_metrics.get(metric, 0)
            current_value = self.current_metrics.get(metric, 0)
            
            if baseline_value > 0:
                degradation = (baseline_value - current_value) / baseline_value
                degradation_info["metrics_comparison"][metric] = {
                    "baseline": baseline_value,
                    "current": current_value,
                    "degradation": degradation
                }
                
                if degradation > threshold:
                    degradation_info["degraded"] = True
                    degradation_info["degraded_metrics"].append(metric)
        
        return degradation_info


class DriftDetector:
    """
    Data drift detector using Maximum Mean Discrepancy (MMD).
    
    Monitors input data distribution changes that might affect model performance.
    """
    
    def __init__(self, 
                 p_val: float = 0.05,
                 update_X_ref: Optional[Dict[str, int]] = None,
                 preprocess_fn: Optional[Callable] = None):
        """
        Initialize drift detector.
        
        Args:
            p_val: p-value threshold for drift detection
            update_X_ref: Parameters for updating reference data
            preprocess_fn: Optional preprocessing function for input data
        """
        self.p_val = p_val
        self.update_X_ref = update_X_ref or {'last': 1000, 'sigma': 1.0}
        self.preprocess_fn = preprocess_fn
        
        self.detector = None
        self.reference_data = None
        self.is_fitted = False
        
        self.logger = logging.getLogger(__name__)
    
    def fit(self, reference_data: np.ndarray) -> None:
        """
        Fit drift detector on reference data.
        
        Args:
            reference_data: Reference dataset to compare against
        """
        if not ALIBI_DETECT_AVAILABLE:
            self.logger.warning("alibi-detect not available. Drift detection disabled.")
            return
            
        try:
            # Preprocess if needed
            if self.preprocess_fn is not None:
                reference_data = self.preprocess_fn(reference_data)
            
            # Initialize MMD drift detector
            self.detector = MMDDrift(
                x_ref=reference_data,
                p_val=self.p_val,
                update_x_ref=self.update_X_ref
            )
            
            self.reference_data = reference_data.copy()
            self.is_fitted = True
            
            self.logger.info(f"Drift detector fitted on {len(reference_data)} reference samples")
            
        except Exception as e:
            self.logger.error(f"Error fitting drift detector: {e}")
            self.is_fitted = False
    
    def predict_drift(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Predict if drift has occurred in the data.
        
        Args:
            data: New data to check for drift
            
        Returns:
            Dict with drift prediction results
        """
        if not self.is_fitted or not ALIBI_DETECT_AVAILABLE:
            return {"drift": False, "reason": "detector_not_available"}
        
        try:
            # Preprocess if needed
            if self.preprocess_fn is not None:
                data = self.preprocess_fn(data)
            
            # Predict drift
            drift_pred = self.detector.predict(data)
            
            return {
                "drift": bool(drift_pred["data"]["is_drift"]),
                "p_value": float(drift_pred["data"]["p_val"]),
                "threshold": self.p_val,
                "distance": float(drift_pred["data"]["distance"])
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting drift: {e}")
            return {"drift": False, "error": str(e)}
    
    def save_detector(self, filepath: Union[str, Path]) -> None:
        """Save fitted detector to file."""
        if not self.is_fitted or not ALIBI_DETECT_AVAILABLE:
            return
            
        try:
            save_detector(self.detector, str(filepath))
            self.logger.info(f"Drift detector saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving drift detector: {e}")
    
    def load_detector(self, filepath: Union[str, Path]) -> None:
        """Load detector from file."""
        if not ALIBI_DETECT_AVAILABLE:
            return
            
        try:
            self.detector = load_detector(str(filepath))
            self.is_fitted = True
            self.logger.info(f"Drift detector loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading drift detector: {e}")


class PrometheusMetrics:
    """
    Prometheus metrics integration for monitoring.
    
    Provides gauges and counters for real-time monitoring dashboards.
    """
    
    def __init__(self, prefix: str = "fake_review_detection"):
        """
        Initialize Prometheus metrics.
        
        Args:
            prefix: Metric name prefix
        """
        self.prefix = prefix
        self.metrics = {}
        
        if PROMETHEUS_AVAILABLE:
            self._initialize_metrics()
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_metrics(self) -> None:
        """Initialize Prometheus metric objects."""
        # Performance metrics
        self.metrics['precision'] = Gauge(
            f'{self.prefix}_precision',
            'Model precision score'
        )
        self.metrics['recall'] = Gauge(
            f'{self.prefix}_recall',
            'Model recall score'
        )
        self.metrics['accuracy'] = Gauge(
            f'{self.prefix}_accuracy',
            'Model accuracy score'
        )
        self.metrics['f1_score'] = Gauge(
            f'{self.prefix}_f1_score',
            'Model F1 score'
        )
        
        # Drift metrics
        self.metrics['drift_detected'] = Gauge(
            f'{self.prefix}_drift_detected',
            'Whether data drift was detected (1=drift, 0=no drift)'
        )
        self.metrics['drift_p_value'] = Gauge(
            f'{self.prefix}_drift_p_value',
            'P-value from drift detection test'
        )
        self.metrics['drift_distance'] = Gauge(
            f'{self.prefix}_drift_distance',
            'Distance metric from drift detection'
        )
        
        # Monitoring status
        self.metrics['monitoring_status'] = Gauge(
            f'{self.prefix}_monitoring_status',
            'Monitoring system status (1=healthy, 0=unhealthy)'
        )
        self.metrics['performance_degraded'] = Gauge(
            f'{self.prefix}_performance_degraded',
            'Whether performance degradation detected (1=degraded, 0=normal)'
        )
        self.metrics['retraining_triggered'] = Counter(
            f'{self.prefix}_retraining_triggered_total',
            'Number of times retraining was triggered'
        )
        
        # Data quality metrics
        self.metrics['prediction_count'] = Counter(
            f'{self.prefix}_predictions_total',
            'Total number of predictions made'
        )
        self.metrics['feedback_count'] = Counter(
            f'{self.prefix}_feedback_total',
            'Total number of feedback samples received'
        )
    
    def update_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Update performance metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        for metric_name in ['precision', 'recall', 'accuracy', 'f1_score']:
            if metric_name in metrics and metric_name in self.metrics:
                self.metrics[metric_name].set(metrics[metric_name])
    
    def update_drift_metrics(self, drift_info: Dict[str, Any]) -> None:
        """Update drift detection metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        self.metrics['drift_detected'].set(1 if drift_info.get('drift', False) else 0)
        
        if 'p_value' in drift_info:
            self.metrics['drift_p_value'].set(drift_info['p_value'])
        if 'distance' in drift_info:
            self.metrics['drift_distance'].set(drift_info['distance'])
    
    def increment_prediction_count(self) -> None:
        """Increment prediction counter."""
        if PROMETHEUS_AVAILABLE and 'prediction_count' in self.metrics:
            self.metrics['prediction_count'].inc()
    
    def increment_feedback_count(self) -> None:
        """Increment feedback counter."""
        if PROMETHEUS_AVAILABLE and 'feedback_count' in self.metrics:
            self.metrics['feedback_count'].inc()
    
    def trigger_retraining(self) -> None:
        """Record retraining trigger."""
        if PROMETHEUS_AVAILABLE and 'retraining_triggered' in self.metrics:
            self.metrics['retraining_triggered'].inc()
    
    def set_monitoring_status(self, healthy: bool) -> None:
        """Set monitoring system status."""
        if PROMETHEUS_AVAILABLE and 'monitoring_status' in self.metrics:
            self.metrics['monitoring_status'].set(1 if healthy else 0)
    
    def set_performance_degraded(self, degraded: bool) -> None:
        """Set performance degradation status."""
        if PROMETHEUS_AVAILABLE and 'performance_degraded' in self.metrics:
            self.metrics['performance_degraded'].set(1 if degraded else 0)


class ModelMonitor:
    """
    Main monitoring class that coordinates all monitoring components.
    
    Integrates drift detection, performance tracking, and retraining triggers.
    """
    
    def __init__(self,
                 drift_detection_enabled: bool = True,
                 performance_tracking_enabled: bool = True,
                 prometheus_enabled: bool = True,
                 retraining_callback: Optional[Callable] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize model monitor.
        
        Args:
            drift_detection_enabled: Enable drift detection
            performance_tracking_enabled: Enable performance tracking
            prometheus_enabled: Enable Prometheus metrics
            retraining_callback: Callback function for retraining trigger
            config: Configuration parameters
        """
        self.config = config or {}
        self.retraining_callback = retraining_callback
        
        # Initialize components
        self.performance_tracker = PerformanceTracker(
            window_size=self.config.get('performance_window_size', 1000)
        ) if performance_tracking_enabled else None
        
        self.drift_detector = DriftDetector(
            p_val=self.config.get('drift_p_value', 0.05),
            update_X_ref=self.config.get('drift_update_ref', {'last': 1000, 'sigma': 1.0})
        ) if drift_detection_enabled else None
        
        self.prometheus_metrics = PrometheusMetrics(
            prefix=self.config.get('metrics_prefix', 'fake_review_detection')
        ) if prometheus_enabled else None
        
        # Monitoring state
        self.is_running = False
        self.last_drift_check = datetime.now()
        self.last_performance_check = datetime.now()
        
        # Thresholds
        self.performance_degradation_threshold = self.config.get('performance_threshold', 0.05)
        self.drift_check_interval = timedelta(minutes=self.config.get('drift_check_minutes', 30))
        self.performance_check_interval = timedelta(minutes=self.config.get('performance_check_minutes', 10))
        
        self.logger = logging.getLogger(__name__)
        
        # Thread lock for concurrent access
        self._lock = threading.Lock()
    
    def start_monitoring(self) -> None:
        """Start the monitoring system."""
        with self._lock:
            if self.is_running:
                return
                
            self.is_running = True
            
            if self.prometheus_metrics:
                self.prometheus_metrics.set_monitoring_status(True)
            
            self.logger.info("Model monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        with self._lock:
            if not self.is_running:
                return
                
            self.is_running = False
            
            if self.prometheus_metrics:
                self.prometheus_metrics.set_monitoring_status(False)
            
            self.logger.info("Model monitoring stopped")
    
    def setup_drift_detection(self, reference_data: np.ndarray) -> None:
        """
        Setup drift detection with reference data.
        
        Args:
            reference_data: Reference dataset for drift detection
        """
        if self.drift_detector:
            self.drift_detector.fit(reference_data)
            self.logger.info("Drift detection setup completed")
    
    def setup_performance_baseline(self, metrics: Dict[str, float]) -> None:
        """
        Setup performance baseline for degradation detection.
        
        Args:
            metrics: Baseline performance metrics
        """
        if self.performance_tracker:
            self.performance_tracker.set_baseline_metrics(metrics)
            self.logger.info(f"Performance baseline set: {metrics}")
    
    def record_prediction(self,
                         prediction: Union[int, bool],
                         probability: float,
                         features: Optional[np.ndarray] = None,
                         true_label: Optional[Union[int, bool]] = None,
                         timestamp: Optional[datetime] = None) -> None:
        """
        Record a prediction for monitoring.
        
        Args:
            prediction: Binary prediction
            probability: Prediction probability
            features: Feature vector (for drift detection)
            true_label: True label (if available for performance tracking)
            timestamp: Prediction timestamp
        """
        if not self.is_running:
            return
            
        with self._lock:
            # Record in performance tracker
            if self.performance_tracker and true_label is not None:
                self.performance_tracker.add_prediction(
                    prediction, true_label, probability, timestamp
                )
                
                # Update Prometheus metrics
                if self.prometheus_metrics:
                    current_metrics = self.performance_tracker.get_current_metrics()
                    self.prometheus_metrics.update_performance_metrics(current_metrics)
            
            # Increment prediction counter
            if self.prometheus_metrics:
                self.prometheus_metrics.increment_prediction_count()
            
            # Check for performance degradation
            self._check_performance_degradation()
            
            # Check for data drift (with features)
            if features is not None:
                self._check_data_drift(features)
    
    def record_feedback(self,
                       prediction_id: str,
                       true_label: Union[int, bool],
                       features: Optional[np.ndarray] = None) -> None:
        """
        Record feedback for a previous prediction.
        
        Args:
            prediction_id: ID of the original prediction
            true_label: True label provided as feedback
            features: Feature vector (optional)
        """
        if not self.is_running:
            return
            
        with self._lock:
            # Increment feedback counter
            if self.prometheus_metrics:
                self.prometheus_metrics.increment_feedback_count()
            
            self.logger.debug(f"Feedback recorded for prediction {prediction_id}")
    
    def _check_performance_degradation(self) -> None:
        """Check if performance has degraded and trigger retraining if needed."""
        if not self.performance_tracker:
            return
            
        now = datetime.now()
        if (now - self.last_performance_check) < self.performance_check_interval:
            return
            
        self.last_performance_check = now
        
        # Check for degradation
        degradation_info = self.performance_tracker.detect_performance_degradation(
            self.performance_degradation_threshold
        )
        
        if degradation_info["degraded"]:
            self.logger.warning(
                f"Performance degradation detected: {degradation_info['degraded_metrics']}"
            )
            
            # Update Prometheus metrics
            if self.prometheus_metrics:
                self.prometheus_metrics.set_performance_degraded(True)
            
            # Trigger retraining
            self.trigger_retraining("performance_degradation", degradation_info)
        else:
            if self.prometheus_metrics:
                self.prometheus_metrics.set_performance_degraded(False)
    
    def _check_data_drift(self, features: np.ndarray) -> None:
        """Check for data drift."""
        if not self.drift_detector or not self.drift_detector.is_fitted:
            return
            
        now = datetime.now()
        if (now - self.last_drift_check) < self.drift_check_interval:
            return
            
        self.last_drift_check = now
        
        # Check for drift
        drift_info = self.drift_detector.predict_drift(features.reshape(1, -1))
        
        # Update Prometheus metrics
        if self.prometheus_metrics:
            self.prometheus_metrics.update_drift_metrics(drift_info)
        
        if drift_info.get("drift", False):
            self.logger.warning(f"Data drift detected: p-value={drift_info.get('p_value', 'N/A')}")
            
            # Trigger retraining
            self.trigger_retraining("data_drift", drift_info)
    
    def trigger_retraining(self, reason: str, details: Dict[str, Any]) -> None:
        """
        Trigger model retraining.
        
        Args:
            reason: Reason for retraining trigger
            details: Additional details about the trigger
        """
        self.logger.info(f"Triggering retraining: {reason}")
        
        # Update Prometheus metrics
        if self.prometheus_metrics:
            self.prometheus_metrics.trigger_retraining()
        
        # Call retraining callback if provided
        if self.retraining_callback:
            try:
                self.retraining_callback(reason, details)
            except Exception as e:
                self.logger.error(f"Error in retraining callback: {e}")
        else:
            self.logger.warning("No retraining callback provided")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring status.
        
        Returns:
            Dict with current monitoring status
        """
        status = {
            "monitoring_active": self.is_running,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "performance_tracking": self.performance_tracker is not None,
                "drift_detection": self.drift_detector is not None and self.drift_detector.is_fitted,
                "prometheus_metrics": self.prometheus_metrics is not None
            }
        }
        
        # Add performance metrics
        if self.performance_tracker:
            status["performance"] = self.performance_tracker.get_current_metrics()
        
        # Add component availability
        status["dependencies"] = {
            "alibi_detect": ALIBI_DETECT_AVAILABLE,
            "prometheus_client": PROMETHEUS_AVAILABLE
        }
        
        return status
    
    def save_state(self, filepath: Union[str, Path]) -> None:
        """Save monitoring state to file."""
        state = {
            "config": self.config,
            "last_drift_check": self.last_drift_check,
            "last_performance_check": self.last_performance_check,
            "performance_baseline": getattr(self.performance_tracker, 'baseline_metrics', None) 
                                   if self.performance_tracker else None
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            self.logger.info(f"Monitoring state saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving monitoring state: {e}")
    
    def load_state(self, filepath: Union[str, Path]) -> None:
        """Load monitoring state from file."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.last_drift_check = state.get("last_drift_check", datetime.now())
            self.last_performance_check = state.get("last_performance_check", datetime.now())
            
            # Restore performance baseline
            if (self.performance_tracker and 
                "performance_baseline" in state and 
                state["performance_baseline"]):
                self.performance_tracker.set_baseline_metrics(state["performance_baseline"])
            
            self.logger.info(f"Monitoring state loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading monitoring state: {e}")


# Convenience function for easy integration
def create_monitor(detector_instance: Optional[Any] = None,
                  config: Optional[Dict[str, Any]] = None) -> ModelMonitor:
    """
    Create a ModelMonitor instance with sensible defaults.
    
    Args:
        detector_instance: FakeReviewDetector instance (optional)
        config: Configuration parameters
        
    Returns:
        Configured ModelMonitor instance
    """
    config = config or {}
    
    # Default retraining callback
    def default_retraining_callback(reason: str, details: Dict[str, Any]) -> None:
        logger = logging.getLogger(__name__)
        logger.info(f"Retraining triggered - Reason: {reason}, Details: {details}")
        
        # Here you could integrate with your training pipeline
        # For example:
        # - Send a message to a queue
        # - Create a training job
        # - Update a database flag
        # - Send an alert/notification
        
        # Placeholder for actual retraining logic
        print(f"RETRAINING TRIGGER: {reason} - {details}")
    
    # Use provided callback or default
    retraining_callback = config.get('retraining_callback', default_retraining_callback)
    
    monitor = ModelMonitor(
        drift_detection_enabled=config.get('enable_drift_detection', True),
        performance_tracking_enabled=config.get('enable_performance_tracking', True),
        prometheus_enabled=config.get('enable_prometheus', True),
        retraining_callback=retraining_callback,
        config=config
    )
    
    return monitor


# Example usage and integration helper
if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create monitor
    config = {
        'performance_window_size': 500,
        'drift_p_value': 0.05,
        'performance_threshold': 0.1,
        'drift_check_minutes': 15,
        'performance_check_minutes': 5
    }
    
    monitor = create_monitor(config=config)
    monitor.start_monitoring()
    
    # Setup with dummy data
    reference_data = np.random.random((100, 10))
    monitor.setup_drift_detection(reference_data)
    
    baseline_metrics = {
        'precision': 0.85,
        'recall': 0.82,
        'accuracy': 0.83,
        'f1_score': 0.835
    }
    monitor.setup_performance_baseline(baseline_metrics)
    
    # Simulate some predictions
    for i in range(50):
        features = np.random.random(10)
        prediction = np.random.choice([0, 1])
        probability = np.random.random()
        true_label = np.random.choice([0, 1])
        
        monitor.record_prediction(
            prediction=prediction,
            probability=probability,
            features=features,
            true_label=true_label
        )
    
    # Check status
    status = monitor.get_monitoring_status()
    print("Monitoring Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    monitor.stop_monitoring()
