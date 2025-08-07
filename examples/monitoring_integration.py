"""
Integration example for model monitoring system.

This example demonstrates how to integrate the monitoring system
with the FakeReviewDetector for comprehensive ML monitoring.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from deployment import initialize_detector, get_detector
from src.monitoring import create_monitor
from src import utils


def setup_monitoring_system(config: Dict[str, Any] = None) -> tuple:
    """
    Setup the complete monitoring system with detector integration.
    
    Args:
        config: Configuration parameters for monitoring
        
    Returns:
        Tuple of (detector, monitor) instances
    """
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load detector
    logger.info("Initializing FakeReviewDetector...")
    detector = get_detector()
    
    if not detector.is_loaded:
        logger.warning("Detector not loaded - attempting to load with defaults")
        try:
            detector.load_model()
        except Exception as e:
            logger.error(f"Failed to load detector: {e}")
            return None, None
    
    # Custom retraining callback that integrates with your training pipeline
    def retraining_callback(reason: str, details: Dict[str, Any]) -> None:
        logger.critical(f"🚨 RETRAINING TRIGGERED - {reason.upper()}")
        logger.info(f"Details: {details}")
        
        # Here you could implement actual retraining logic:
        # 1. Collect new training data
        # 2. Trigger training pipeline (MLflow, Airflow, etc.)
        # 3. Send alerts (email, Slack, PagerDuty)
        # 4. Update model registry
        # 5. Schedule model deployment
        
        # Example integrations:
        # - Send to message queue (Redis, RabbitMQ, Kafka)
        # - Create training job in Kubernetes
        # - Update database flags
        # - Call webhook/API endpoint
        
        # Placeholder implementation
        print(f"📊 Would trigger retraining pipeline for: {reason}")
        print(f"📈 Monitoring details: {details}")
        
        # Example: Save trigger info to file for later processing
        trigger_info = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'reason': reason,
            'details': details,
            'model_info': detector.get_model_info() if detector.is_loaded else None
        }
        
        # Save to monitoring directory
        monitoring_dir = Path("artifacts/monitoring")
        monitoring_dir.mkdir(exist_ok=True, parents=True)
        
        trigger_file = monitoring_dir / f"retraining_trigger_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        utils.save_json(trigger_info, trigger_file)
        logger.info(f"Retraining trigger saved to: {trigger_file}")
    
    # Configure monitoring system
    monitoring_config = {
        'performance_window_size': 1000,
        'drift_p_value': 0.05,
        'performance_threshold': 0.05,  # 5% degradation threshold
        'drift_check_minutes': 30,
        'performance_check_minutes': 10,
        'metrics_prefix': 'fake_review_detection',
        'enable_drift_detection': True,
        'enable_performance_tracking': True,
        'enable_prometheus': True,
        'retraining_callback': retraining_callback
    }
    
    # Override with provided config
    if config:
        monitoring_config.update(config)
    
    # Create monitor
    logger.info("Creating model monitor...")
    monitor = create_monitor(detector_instance=detector, config=monitoring_config)
    
    logger.info("✅ Monitoring system setup complete!")
    return detector, monitor


def setup_baseline_data(monitor, detector):
    """
    Setup baseline data for monitoring (drift detection and performance baseline).
    
    Args:
        monitor: ModelMonitor instance
        detector: FakeReviewDetector instance
    """
    logger = logging.getLogger(__name__)
    
    # Generate dummy reference data for drift detection
    # In practice, this would be your training/validation data
    logger.info("Setting up drift detection baseline...")
    
    # Simulate feature extraction from reference reviews
    reference_reviews = [
        "This product is amazing! Highly recommend to everyone.",
        "Great quality, fast shipping, excellent customer service.",
        "Not bad, decent product for the price point.",
        "Could be better, had some issues with durability.",
        "Terrible product, waste of money. Don't buy this!",
        "Outstanding quality and value. Will buy again.",
        "Good product but delivery was delayed significantly.",
        "Excellent build quality, works as expected.",
        "Average product, nothing special but does the job.",
        "Perfect! Exactly what I was looking for."
    ]
    
    # Extract features from reference data
    reference_features = []
    for review in reference_reviews:
        try:
            # Use the detector's preprocessing to extract features
            input_data = detector._prepare_input(review, None, None, None)
            features = detector._preprocess_input(input_data)
            reference_features.append(features.flatten())
        except Exception as e:
            logger.warning(f"Failed to extract features from reference review: {e}")
            # Use dummy features as fallback
            reference_features.append(np.random.random(10))
    
    reference_data = np.array(reference_features)
    monitor.setup_drift_detection(reference_data)
    
    # Setup performance baseline
    logger.info("Setting up performance baseline...")
    baseline_metrics = {
        'precision': detector.model_metadata.get('best_score', 0.85),
        'recall': 0.82,
        'accuracy': 0.83,
        'f1_score': 0.835
    }
    monitor.setup_performance_baseline(baseline_metrics)
    
    logger.info("✅ Baseline data setup complete!")


def simulate_production_usage(detector, monitor, num_predictions: int = 100):
    """
    Simulate production usage with monitoring.
    
    Args:
        detector: FakeReviewDetector instance
        monitor: ModelMonitor instance
        num_predictions: Number of predictions to simulate
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Simulating {num_predictions} predictions with monitoring...")
    
    # Sample reviews for simulation
    sample_reviews = [
        "This product is absolutely fantastic! I love it so much!",
        "Great quality product, would recommend to friends.",
        "Not satisfied with the purchase. Poor quality.",
        "Amazing! Best product ever! Buy it now! 5 stars!",
        "Decent product for the price. Nothing special.",
        "Terrible! Worst purchase ever! Don't waste your money!",
        "Good product, fast delivery, happy with purchase.",
        "Outstanding quality and customer service experience.",
        "Average product, works as described in listing.",
        "Perfect! Exactly what I needed for my project."
    ]
    
    # Simulate predictions with some ground truth feedback
    for i in range(num_predictions):
        # Select random review
        review = np.random.choice(sample_reviews)
        
        try:
            # Make prediction
            probability = detector.predict(review)
            prediction = int(probability > 0.5)
            
            # Extract features for drift monitoring
            input_data = detector._prepare_input(review, None, None, None)
            features = detector._preprocess_input(input_data).flatten()
            
            # Simulate ground truth (in practice, this comes from user feedback)
            # Add some noise to make it realistic
            if np.random.random() < 0.1:  # 10% chance of having ground truth
                # Simulate ground truth with some correlation to prediction
                true_label = prediction if np.random.random() < 0.8 else 1 - prediction
                
                # Record prediction with ground truth
                monitor.record_prediction(
                    prediction=prediction,
                    probability=probability,
                    features=features,
                    true_label=true_label
                )
                
                logger.debug(f"Prediction {i}: prob={probability:.3f}, pred={prediction}, true={true_label}")
            else:
                # Record prediction without ground truth
                monitor.record_prediction(
                    prediction=prediction,
                    probability=probability,
                    features=features
                )
                
                logger.debug(f"Prediction {i}: prob={probability:.3f}, pred={prediction}")
                
        except Exception as e:
            logger.error(f"Error in prediction {i}: {e}")
    
    logger.info("✅ Production simulation complete!")


def demonstrate_monitoring_features(monitor):
    """
    Demonstrate various monitoring features.
    
    Args:
        monitor: ModelMonitor instance
    """
    logger = logging.getLogger(__name__)
    logger.info("Demonstrating monitoring features...")
    
    # Get monitoring status
    status = monitor.get_monitoring_status()
    logger.info(f"Monitoring Status: {status}")
    
    # Save monitoring state
    state_file = Path("artifacts/monitoring/monitor_state.pkl")
    state_file.parent.mkdir(exist_ok=True, parents=True)
    monitor.save_state(state_file)
    logger.info(f"Monitor state saved to: {state_file}")
    
    # Demonstrate manual feedback recording
    logger.info("Recording sample feedback...")
    monitor.record_feedback("pred_123", true_label=1)
    monitor.record_feedback("pred_124", true_label=0)
    
    logger.info("✅ Monitoring features demonstration complete!")


def main():
    """Main execution function."""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Monitoring System Integration Demo")
    
    # Setup monitoring system
    detector, monitor = setup_monitoring_system()
    
    if detector is None or monitor is None:
        logger.error("Failed to setup monitoring system")
        return
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Setup baseline data
        setup_baseline_data(monitor, detector)
        
        # Simulate production usage
        simulate_production_usage(detector, monitor, num_predictions=50)
        
        # Demonstrate monitoring features
        demonstrate_monitoring_features(monitor)
        
        # Final status check
        final_status = monitor.get_monitoring_status()
        logger.info("📊 Final Monitoring Status:")
        for key, value in final_status.items():
            logger.info(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
        logger.info("🛑 Monitoring system stopped")
    
    logger.info("✅ Demo completed successfully!")


if __name__ == "__main__":
    main()
