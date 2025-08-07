# API Documentation

The Fake Review Detection System provides a comprehensive REST API for detecting fraudulent reviews. This document covers all available endpoints, request/response formats, and usage examples.

## Base URL

```
http://localhost:8000  # Development
https://your-domain.com  # Production
```

## Authentication

Currently, the API does not require authentication. For production deployment, consider implementing API key authentication or OAuth2.

## Rate Limiting

- **Single predictions**: 60 requests per minute per IP
- **Batch predictions**: 10 requests per minute per IP

## Content Type

All requests should use `Content-Type: application/json`.

## Endpoints

### 1. Root Endpoint

**GET** `/`

Get basic API information and available endpoints.

#### Response

```json
{
  "service": "Fake Review Detection API",
  "version": "1.0.0",
  "status": "running",
  "timestamp": "2024-01-15T10:30:00.123Z",
  "endpoints": {
    "predict": "/predict",
    "predict_batch": "/predict/batch",
    "health": "/health",
    "metrics": "/metrics"
  }
}
```

### 2. Single Review Prediction

**POST** `/predict`

Predict whether a single review is fake or legitimate.

#### Request Body

```json
{
  "text": "This product is amazing! Best purchase ever!",
  "user_id": "user123",
  "timestamp": "2024-01-15T10:30:00Z",
  "rating": 5,
  "additional_features": {
    "verified_purchase": true,
    "review_length": 45
  }
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Review text (1-10,000 characters) |
| `user_id` | string | No | User identifier (max 100 chars) |
| `timestamp` | string | No | ISO format timestamp |
| `rating` | number | No | Rating score (1-5) |
| `additional_features` | object | No | Additional feature data |

#### Response

```json
{
  "probability": 0.15,
  "prediction": "legitimate",
  "confidence": "high",
  "processing_time": 0.045,
  "model_version": "random_forest"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `probability` | number | Probability of being fake (0-1) |
| `prediction` | string | Binary prediction ("fake" or "legitimate") |
| `confidence` | string | Confidence level ("low", "medium", "high") |
| `processing_time` | number | Processing time in seconds |
| `model_version` | string | Model type used for prediction |

#### Confidence Levels

- **High**: Probability < 0.3 or > 0.7
- **Medium**: 0.4 ≤ Probability ≤ 0.6
- **Low**: 0.3 ≤ Probability < 0.4 or 0.6 < Probability ≤ 0.7

#### Example cURL Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Great product, highly recommend!",
       "rating": 5,
       "user_id": "user123"
     }'
```

#### Example Python Request

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "text": "Great product, highly recommend!",
        "rating": 5,
        "user_id": "user123"
    }
)
print(response.json())
```

### 3. Batch Review Prediction

**POST** `/predict/batch`

Predict multiple reviews in a single request.

#### Request Body

```json
{
  "texts": [
    "This product is amazing!",
    "Terrible quality, don't buy",
    "Average product, nothing special"
  ],
  "user_ids": ["user1", "user2", "user3"],
  "timestamps": [
    "2024-01-15T10:00:00Z",
    "2024-01-15T11:00:00Z",
    "2024-01-15T12:00:00Z"
  ],
  "ratings": [5, 1, 3],
  "additional_features": [
    {"verified_purchase": true},
    {"verified_purchase": false},
    {"verified_purchase": true}
  ]
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `texts` | array[string] | Yes | List of review texts (1-100 items) |
| `user_ids` | array[string] | No | List of user identifiers |
| `timestamps` | array[string] | No | List of ISO timestamps |
| `ratings` | array[number] | No | List of ratings (1-5) |
| `additional_features` | array[object] | No | List of additional features |

#### Response

```json
{
  "predictions": [
    {
      "probability": 0.25,
      "prediction": "legitimate",
      "confidence": "high",
      "processing_time": 0.015,
      "model_version": "random_forest"
    },
    {
      "probability": 0.85,
      "prediction": "fake",
      "confidence": "high",
      "processing_time": 0.015,
      "model_version": "random_forest"
    },
    {
      "probability": 0.45,
      "prediction": "legitimate",
      "confidence": "low",
      "processing_time": 0.015,
      "model_version": "random_forest"
    }
  ],
  "total_processed": 3,
  "processing_time": 0.045
}
```

### 4. Health Check

**GET** `/health`

Check the health status of the API and model.

#### Response

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.123Z",
  "model_loaded": true,
  "model_health": "healthy",
  "uptime": 86400.5,
  "memory_usage": {
    "rss_mb": 512.3,
    "vms_mb": 1024.6,
    "percent": 12.5
  }
}
```

#### Status Codes

- **200**: Service is healthy
- **503**: Service is unhealthy

### 5. Metrics

**GET** `/metrics`

Get Prometheus metrics (if enabled) or basic JSON metrics.

#### Prometheus Response (Content-Type: text/plain)

```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="POST",endpoint="/predict",status_code="200"} 1523

# HELP prediction_duration_seconds Prediction processing time
# TYPE prediction_duration_seconds histogram
prediction_duration_seconds_bucket{le="0.1"} 1200
prediction_duration_seconds_bucket{le="0.5"} 1500
prediction_duration_seconds_bucket{le="+Inf"} 1523
```

#### JSON Response (fallback)

```json
{
  "requests_total": 1523,
  "predictions_total": 1450,
  "average_response_time": 0.045,
  "model_health": 1.0,
  "uptime": 86400.5
}
```

### 6. Model Information

**GET** `/model/info`

Get information about the loaded model.

#### Response

```json
{
  "model_type": "random_forest",
  "version": "1.0.0",
  "training_date": "2024-01-10T15:30:00Z",
  "features": [
    "text_features",
    "behavioral_features",
    "network_features"
  ],
  "performance_metrics": {
    "accuracy": 0.94,
    "precision": 0.93,
    "recall": 0.95,
    "f1_score": 0.94
  },
  "training_data_size": 50000,
  "preprocessing_pipeline": "standard"
}
```

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message description"
}
```

### Common Error Codes

| Status Code | Error | Description |
|-------------|-------|-------------|
| 400 | Bad Request | Invalid request parameters |
| 422 | Validation Error | Request validation failed |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server processing error |
| 503 | Service Unavailable | Model not loaded |

### Example Error Response

```json
{
  "detail": "Invalid input: Text cannot be empty"
}
```

## WebSocket Support (Future)

WebSocket endpoints for real-time predictions will be available in future versions:

```
ws://localhost:8000/ws/predict
```

## SDK and Client Libraries

### Python SDK

```python
from fake_review_detector import FakeReviewClient

client = FakeReviewClient("http://localhost:8000")
result = client.predict("Great product!")
print(f"Prediction: {result.prediction}")
```

### JavaScript SDK

```javascript
import { FakeReviewClient } from 'fake-review-detector-js';

const client = new FakeReviewClient('http://localhost:8000');
const result = await client.predict('Great product!');
console.log(`Prediction: ${result.prediction}`);
```

## Best Practices

### 1. Batch Processing

For multiple predictions, always use the batch endpoint for better performance:

```python
# Good - Batch processing
texts = ["review1", "review2", "review3"]
results = client.predict_batch(texts)

# Avoid - Multiple single requests
results = [client.predict(text) for text in texts]
```

### 2. Error Handling

Always implement proper error handling:

```python
import requests

try:
    response = requests.post(
        "http://localhost:8000/predict",
        json={"text": "review text"},
        timeout=10
    )
    response.raise_for_status()
    result = response.json()
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

### 3. Rate Limiting

Respect rate limits and implement exponential backoff:

```python
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)
```

### 4. Monitoring

Monitor API usage and performance:

```python
import requests
import time

start_time = time.time()
response = requests.post(url, json=data)
response_time = time.time() - start_time

print(f"Response time: {response_time:.3f}s")
print(f"Status: {response.status_code}")
```

## Performance Considerations

- **Single predictions**: ~50ms average response time
- **Batch predictions**: ~5ms per review in batch
- **Throughput**: ~1000 requests per minute per instance
- **Memory usage**: ~500MB baseline + ~1MB per concurrent request

## Security Considerations

- Always use HTTPS in production
- Implement rate limiting to prevent abuse
- Validate and sanitize all input data
- Monitor for unusual usage patterns
- Consider implementing authentication for sensitive use cases

---

For more information, see the [Interactive API Documentation](http://localhost:8000/docs) available when running the service.
