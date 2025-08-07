# Fake Review Detection - Deployment & API

This document describes how to deploy and use the Fake Review Detection system's REST API.

## Overview

The deployment consists of:
- **FakeReviewDetector**: Core prediction class that loads models and provides predictions
- **FastAPI Service**: REST API with endpoints for predictions, health checks, and metrics
- **Configuration Management**: YAML-based settings with environment overrides
- **Monitoring**: Integrated Prometheus metrics and logging
- **Rate Limiting**: Built-in protection against abuse

## Quick Start

### 1. Install Dependencies

```bash
# Install API-specific dependencies
pip install -r requirements-api.txt

# Or install core dependencies if not already done
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
# Basic startup (uses settings from config/settings.yaml)
python run_api.py

# With debug mode
python run_api.py --debug

# With custom host/port
python run_api.py --host 0.0.0.0 --port 8080

# With specific model path
python run_api.py --model-path artifacts/models/my_model.joblib
```

### 3. Test the Deployment

```bash
# Run deployment tests
python test_deployment.py

# Test API endpoints (requires running server)
python test_deployment.py --api
```

## API Endpoints

### Core Endpoints

#### `GET /` - Service Information
Returns basic service information and available endpoints.

```bash
curl http://localhost:8000/
```

#### `POST /predict` - Single Prediction
Predict if a single review is fake.

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This product is amazing! I love it so much!",
    "user_id": "user123",
    "rating": 5,
    "timestamp": "2023-12-01T12:00:00Z"
  }'
```

Response:
```json
{
  "probability": 0.75,
  "prediction": "fake",
  "confidence": "high",
  "processing_time": 0.045,
  "model_version": "RandomForestClassifier"
}
```

#### `POST /predict/batch` - Batch Prediction
Predict multiple reviews at once (up to 100 reviews per request).

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Great product, highly recommend!",
      "Terrible quality, waste of money"
    ],
    "ratings": [5, 1]
  }'
```

### Monitoring Endpoints

#### `GET /health` - Health Check
Check service and model health.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2023-12-01T12:00:00Z",
  "model_loaded": true,
  "model_health": "passed",
  "uptime": 3600.5,
  "memory_usage": {
    "rss_mb": 256.7,
    "vms_mb": 512.3,
    "percent": 12.5
  }
}
```

#### `GET /metrics` - Prometheus Metrics
Get Prometheus-format metrics for monitoring.

```bash
curl http://localhost:8000/metrics
```

#### `GET /model/info` - Model Information
Get detailed information about the loaded model.

```bash
curl http://localhost:8000/model/info
```

## Configuration

### Settings File (config/settings.yaml)

Key configuration sections:

```yaml
# API Configuration
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  prediction_threshold: 0.5
  cors:
    origins: ["*"]
  rate_limit:
    enabled: true
    requests_per_minute: 60

# Deployment Configuration
deployment:
  model_path: null  # Auto-detect if null
  preprocessor_path: null
  load_on_startup: true

# Monitoring Configuration
monitoring:
  prometheus:
    enabled: true
  logging:
    api_requests: true
    predictions: true
```

### Environment Variables

Override settings using environment variables:
- `FAKE_REVIEW_MODEL_PATH`: Path to model file
- `FAKE_REVIEW_PREPROCESSOR_PATH`: Path to preprocessor file
- `FAKE_REVIEW_API_HOST`: API host
- `FAKE_REVIEW_API_PORT`: API port

## FakeReviewDetector Usage

### Basic Usage

```python
from deployment import FakeReviewDetector

# Initialize detector (auto-detects latest model)
detector = FakeReviewDetector()

# Make prediction
probability = detector.predict("This product is amazing!")
print(f"Fake probability: {probability:.3f}")

# Prediction with additional features
probability = detector.predict(
    text="Great product!",
    user_id="user123",
    timestamp="2023-12-01T12:00:00",
    additional_features={"rating": 5}
)
```

### Batch Predictions

```python
texts = [
    "Great product, highly recommend!",
    "Terrible quality, waste of money",
    "Average product, okay value"
]

probabilities = detector.predict_batch(texts)
for text, prob in zip(texts, probabilities):
    print(f"{text}: {prob:.3f}")
```

### Model Information

```python
# Get model information
info = detector.get_model_info()
print(f"Model type: {info['model_type']}")
print(f"Performance: {info.get('performance', {}).get('best_score', 'N/A')}")

# Health check
health = detector.health_check()
print(f"Status: {health['status']}")
```

## Rate Limiting

The API includes built-in rate limiting:
- **Single predictions**: 60 requests/minute per IP
- **Batch predictions**: 10 requests/minute per IP
- **Other endpoints**: No limits

Rate limits can be configured in `config/settings.yaml` or disabled by setting `api.rate_limit.enabled: false`.

## Monitoring & Metrics

### Prometheus Metrics

Available metrics:
- `http_requests_total`: Total HTTP requests by method, endpoint, status
- `http_request_duration_seconds`: Request duration histogram
- `predictions_total`: Total predictions made
- `prediction_duration_seconds`: Prediction processing time
- `prediction_probability`: Distribution of prediction probabilities
- `model_health_status`: Model health (1=healthy, 0=unhealthy)
- `active_requests`: Currently active requests

### Logging

Logs are written to:
- Console (structured format)
- Files in `logs/` directory (rotated)

Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

## Deployment Options

### Development

```bash
# Debug mode with auto-reload
python run_api.py --debug --reload
```

### Production

```bash
# Production with multiple workers
python run_api.py --workers 4 --host 0.0.0.0 --port 8000

# Using Gunicorn (alternative)
pip install gunicorn
gunicorn api.app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker

Create a Dockerfile:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY . .

EXPOSE 8000
CMD ["python", "run_api.py", "--workers", "4"]
```

Build and run:
```bash
docker build -t fake-review-api .
docker run -p 8000:8000 fake-review-api
```

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid input data
- **429 Too Many Requests**: Rate limit exceeded
- **503 Service Unavailable**: Model not loaded or unhealthy
- **500 Internal Server Error**: Unexpected server errors

Example error response:
```json
{
  "detail": "Invalid input: Text cannot be empty"
}
```

## Security Considerations

1. **Rate Limiting**: Prevents abuse and DoS attacks
2. **Input Validation**: Strict validation of all inputs
3. **CORS**: Configurable cross-origin policies
4. **Logging**: Comprehensive audit trail
5. **Health Checks**: Monitor service availability

For production deployment:
- Use HTTPS/TLS
- Implement authentication if needed
- Monitor resource usage
- Set up proper backup and recovery
- Use environment variables for secrets

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure model files exist in `artifacts/models/`
2. **Import errors**: Check Python path and dependencies
3. **Port conflicts**: Use different port with `--port` flag
4. **Memory issues**: Adjust worker count or model size

### Debug Mode

Enable debug logging:
```bash
python run_api.py --debug --log-level DEBUG
```

### Health Monitoring

Check service health:
```bash
curl http://localhost:8000/health
```

Check model status:
```bash
curl http://localhost:8000/model/info
```

## Performance Tuning

### API Performance
- Adjust worker count based on CPU cores
- Use async endpoints for I/O-bound operations
- Enable response compression
- Monitor memory usage

### Model Performance
- Pre-load models on startup
- Use model caching
- Consider model quantization
- Batch predictions when possible

### Monitoring
- Set up Prometheus + Grafana
- Monitor key metrics (latency, throughput, errors)
- Set up alerting for critical issues
- Log analysis for performance insights
