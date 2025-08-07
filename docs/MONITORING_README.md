# Model Monitoring System

This document describes the comprehensive model monitoring system implemented for the Fake Review Detection project. The system provides real-time monitoring of model performance, data drift detection, and automatic retraining triggers.

## Overview

The monitoring system consists of several key components:

- **Performance Tracker**: Rolling window tracking of precision, recall, accuracy, and F1-score
- **Drift Detector**: MMD-based drift detection using alibi-detect
- **Prometheus Metrics**: Real-time metrics for monitoring dashboards
- **Retraining Triggers**: Automatic alerts when performance degrades or drift is detected

## Features

### 🎯 Performance Tracking
- Rolling window metrics calculation
- Performance degradation detection
- Baseline comparison
- Configurable thresholds and window sizes

### 🔍 Drift Detection
- Maximum Mean Discrepancy (MMD) based drift detection
- Configurable p-value thresholds
- Reference data management
- Feature-level monitoring

### 📊 Prometheus Integration
- Real-time metrics export
- Dashboard-ready gauges and counters
- Performance and drift metrics
- System health indicators

### 🔄 Retraining Automation
- Configurable retraining triggers
- Custom callback functions
- Integration with training pipelines
- Alert and notification support

## Installation

### Dependencies

The monitoring system requires the following additional packages:

```bash
# Core monitoring dependencies
pip install alibi-detect==0.11.4
pip install prometheus_client==0.17.1

# Already included in main requirements.txt
pip install scikit-learn numpy pandas
```

### Import and Setup

```python
from src.monitoring import create_monitor, ModelMonitor
from deployment import get_detector

# Create monitoring system
config = {
    'performance_window_size': 1000,
    'drift_p_value': 0.05,
    'performance_threshold': 0.05
}

monitor = create_monitor(config=config)
```

## Usage

### Basic Setup

```python
import numpy as np
from src.monitoring import create_monitor
from deployment import get_detector

# 1. Initialize detector and monitor
detector = get_detector()
monitor = create_monitor()

# 2. Start monitoring
monitor.start_monitoring()

# 3. Setup drift detection with reference data
reference_data = np.random.random((100, 10))  # Your training features
monitor.setup_drift_detection(reference_data)

# 4. Set performance baseline
baseline_metrics = {
    'precision': 0.85,
    'recall': 0.82,
    'accuracy': 0.83,
    'f1_score': 0.835
}
monitor.setup_performance_baseline(baseline_metrics)
```

### Recording Predictions

```python
# Record prediction for monitoring
prediction = detector.predict("This is a test review")
features = detector._preprocess_input(input_data).flatten()

monitor.record_prediction(
    prediction=int(prediction > 0.5),
    probability=prediction,
    features=features,
    true_label=1  # If ground truth is available
)
```

### Custom Retraining Callback

```python
def custom_retraining_callback(reason: str, details: dict):
    """Custom callback for retraining triggers."""
    print(f"Retraining triggered: {reason}")
    
    # Your retraining logic here:
    # - Send message to queue
    # - Trigger training pipeline
    # - Send alerts
    # - Update model registry

config = {
    'retraining_callback': custom_retraining_callback
}
monitor = create_monitor(config=config)
```

## Configuration

### Available Configuration Options

```python
config = {
    # Performance tracking
    'performance_window_size': 1000,           # Rolling window size
    'performance_threshold': 0.05,             # 5% degradation threshold
    'performance_check_minutes': 10,           # Check interval

    # Drift detection
    'drift_p_value': 0.05,                     # p-value threshold
    'drift_check_minutes': 30,                 # Check interval
    'drift_update_ref': {                      # Reference data update
        'last': 1000, 
        'sigma': 1.0
    },

    # System settings
    'metrics_prefix': 'fake_review_detection', # Prometheus prefix
    'enable_drift_detection': True,
    'enable_performance_tracking': True,
    'enable_prometheus': True,

    # Callbacks
    'retraining_callback': your_callback_function
}
```

## Integration Examples

### With FastAPI

```python
from fastapi import FastAPI
from src.monitoring import create_monitor
from deployment import get_detector

app = FastAPI()
detector = get_detector()
monitor = create_monitor()

@app.on_event("startup")
async def startup_event():
    monitor.start_monitoring()
    # Setup baseline data...

@app.post("/predict")
async def predict(request: ReviewRequest):
    # Make prediction
    probability = detector.predict(request.text)
    
    # Record for monitoring
    monitor.record_prediction(
        prediction=int(probability > 0.5),
        probability=probability,
        features=extracted_features
    )
    
    return {"probability": probability}

@app.get("/monitoring/status")
async def monitoring_status():
    return monitor.get_monitoring_status()
```

### With MLflow

```python
import mlflow

def mlflow_retraining_callback(reason: str, details: dict):
    """Trigger MLflow training run."""
    
    # Create MLflow experiment
    experiment_name = "fake_review_retraining"
    mlflow.set_experiment(experiment_name)
    
    # Start training run
    with mlflow.start_run():
        mlflow.log_param("trigger_reason", reason)
        mlflow.log_params(details)
        
        # Your training code here...
        # model = train_model()
        # mlflow.sklearn.log_model(model, "model")

config = {'retraining_callback': mlflow_retraining_callback}
monitor = create_monitor(config=config)
```

### With Airflow

```python
from airflow.models import DagBag

def airflow_retraining_callback(reason: str, details: dict):
    """Trigger Airflow DAG for retraining."""
    
    dag_id = "fake_review_retraining_dag"
    
    # Trigger DAG run
    dagbag = DagBag()
    dag = dagbag.get_dag(dag_id)
    
    if dag:
        dag.create_dagrun(
            run_id=f"monitoring_trigger_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            execution_date=datetime.now(),
            state="running",
            conf={
                'trigger_reason': reason,
                'details': details
            }
        )
```

## Prometheus Metrics

### Available Metrics

The system exports the following Prometheus metrics:

#### Performance Metrics
- `fake_review_detection_precision`: Current model precision
- `fake_review_detection_recall`: Current model recall  
- `fake_review_detection_accuracy`: Current model accuracy
- `fake_review_detection_f1_score`: Current model F1 score

#### Drift Metrics
- `fake_review_detection_drift_detected`: Drift detection flag (0/1)
- `fake_review_detection_drift_p_value`: Current p-value from drift test
- `fake_review_detection_drift_distance`: Distance metric from drift detection

#### System Metrics
- `fake_review_detection_monitoring_status`: Monitoring system health (0/1)
- `fake_review_detection_performance_degraded`: Performance degradation flag (0/1)
- `fake_review_detection_retraining_triggered_total`: Total retraining triggers
- `fake_review_detection_predictions_total`: Total predictions made
- `fake_review_detection_feedback_total`: Total feedback received

### Grafana Dashboard Example

```json
{
  "dashboard": {
    "title": "ML Model Monitoring",
    "panels": [
      {
        "title": "Model Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "fake_review_detection_precision"
          },
          {
            "expr": "fake_review_detection_recall"
          }
        ]
      },
      {
        "title": "Drift Detection",
        "type": "singlestat",
        "targets": [
          {
            "expr": "fake_review_detection_drift_detected"
          }
        ]
      }
    ]
  }
}
```

## API Integration

### Monitoring Endpoints

Add these endpoints to your FastAPI application:

```python
@app.get("/monitoring/status")
async def get_monitoring_status():
    """Get comprehensive monitoring status."""
    return monitor.get_monitoring_status()

@app.post("/monitoring/feedback")
async def submit_feedback(prediction_id: str, true_label: int):
    """Submit feedback for a prediction."""
    monitor.record_feedback(prediction_id, true_label)
    return {"status": "feedback_recorded"}

@app.get("/monitoring/metrics")
async def get_prometheus_metrics():
    """Get Prometheus metrics."""
    if monitor.prometheus_metrics:
        from prometheus_client import generate_latest
        return Response(
            content=generate_latest(),
            media_type="text/plain"
        )
    return {"error": "Prometheus not available"}

@app.post("/monitoring/trigger-retraining")
async def manual_retraining_trigger():
    """Manually trigger retraining."""
    monitor.trigger_retraining("manual", {"triggered_by": "api"})
    return {"status": "retraining_triggered"}
```

## Troubleshooting

### Common Issues

1. **alibi-detect not available**
   ```bash
   pip install alibi-detect==0.11.4
   ```

2. **Prometheus metrics not working**
   ```bash
   pip install prometheus_client==0.17.1
   ```

3. **Memory issues with large reference datasets**
   ```python
   # Reduce reference data size
   config['drift_update_ref'] = {'last': 500, 'sigma': 1.0}
   ```

4. **Performance degradation false positives**
   ```python
   # Increase threshold or window size
   config['performance_threshold'] = 0.1  # 10% instead of 5%
   config['performance_window_size'] = 2000  # Larger window
   ```

### Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor-specific logging
logger = logging.getLogger('src.monitoring')
logger.setLevel(logging.DEBUG)
```

## Best Practices

### 1. Reference Data Management
- Use representative training/validation data
- Update reference data periodically
- Monitor reference data quality

### 2. Threshold Tuning
- Start with conservative thresholds
- Monitor false positive rates
- Adjust based on business requirements

### 3. Performance Monitoring
- Ensure sufficient ground truth data
- Use appropriate window sizes
- Consider seasonal patterns

### 4. Alerting Strategy
- Implement proper alert routing
- Set up escalation procedures
- Balance sensitivity vs. noise

### 5. Retraining Pipeline
- Automate data collection
- Validate new models before deployment
- Maintain model versioning

## Advanced Usage

### Custom Drift Detection

```python
from src.monitoring import DriftDetector

# Custom preprocessing function
def preprocess_features(data):
    # Your custom preprocessing
    return processed_data

detector = DriftDetector(
    p_val=0.01,  # More sensitive
    preprocess_fn=preprocess_features
)
```

### Multiple Model Monitoring

```python
# Monitor multiple models
monitors = {}
for model_name in ['model_a', 'model_b']:
    config = {
        'metrics_prefix': f'model_{model_name}',
        'performance_window_size': 500
    }
    monitors[model_name] = create_monitor(config=config)
```

### State Persistence

```python
# Save monitoring state
monitor.save_state("artifacts/monitoring/state.pkl")

# Load monitoring state
monitor.load_state("artifacts/monitoring/state.pkl")
```

## Example Run

See `examples/monitoring_integration.py` for a complete working example:

```bash
cd examples
python monitoring_integration.py
```

This will demonstrate:
- Setting up the monitoring system
- Configuring drift detection and performance baselines
- Simulating production predictions
- Triggering retraining scenarios
- Saving and loading monitoring state

## Support

For issues or questions about the monitoring system:

1. Check the troubleshooting section
2. Review the example integration code
3. Enable debug logging for detailed information
4. Consult the inline documentation in `src/monitoring.py`
