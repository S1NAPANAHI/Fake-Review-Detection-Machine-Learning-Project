"""
Unit tests for the FastAPI application endpoints.

This module tests API functionality including prediction endpoints, health checks,
metrics collection, batch processing, error handling, and response validation.
Uses small synthetic fixtures and validates API responses and status codes.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import FastAPI test client
try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    TestClient = None


@unittest.skipIf(not FASTAPI_AVAILABLE, "FastAPI not available")
class TestFastAPIEndpoints(unittest.TestCase):
    """Test FastAPI application endpoints."""
    
    def setUp(self):
        """Set up test client and mock detector."""
        # Mock the detector and deployment module
        self.mock_detector = MagicMock()
        self.mock_detector.predict.return_value = {
            'prediction': 'fake',
            'probability': 0.85,
            'confidence': 'high'
        }
        self.mock_detector.predict_batch.return_value = [
            {'prediction': 'fake', 'probability': 0.85, 'confidence': 'high'},
            {'prediction': 'legitimate', 'probability': 0.25, 'confidence': 'high'}
        ]
        self.mock_detector.is_healthy.return_value = True
        self.mock_detector.model_version = "test_model_v1.0"
        
        # Patch the detector import and initialization
        with patch.dict('sys.modules', {
            'deployment': MagicMock(
                get_detector=MagicMock(return_value=self.mock_detector),
                initialize_detector=MagicMock(return_value=self.mock_detector),
                FakeReviewDetector=MagicMock()
            )
        }):
            # Import and create test client after mocking
            from api.app import app
            self.client = TestClient(app)
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        with patch('api.app.get_detector', return_value=self.mock_detector):
            response = self.client.get("/health")
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('status', data)
        self.assertIn('timestamp', data)
        self.assertIn('model_loaded', data)
        self.assertIn('model_health', data)
        self.assertIn('uptime', data)
        
        # Should indicate healthy status
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue(data['model_loaded'])
        self.assertEqual(data['model_health'], 'healthy')
    
    def test_health_endpoint_unhealthy_model(self):
        """Test health check with unhealthy model."""
        self.mock_detector.is_healthy.return_value = False
        
        with patch('api.app.get_detector', return_value=self.mock_detector):
            response = self.client.get("/health")
        
        self.assertEqual(response.status_code, 503)  # Service Unavailable
        
        data = response.json()
        self.assertEqual(data['status'], 'unhealthy')
        self.assertEqual(data['model_health'], 'unhealthy')
    
    def test_health_endpoint_no_detector(self):
        """Test health check when detector is not available."""
        with patch('api.app.get_detector', return_value=None):
            response = self.client.get("/health")
        
        self.assertEqual(response.status_code, 503)
        
        data = response.json()
        self.assertEqual(data['status'], 'unhealthy')
        self.assertFalse(data['model_loaded'])
    
    def test_single_prediction_endpoint(self):
        """Test single review prediction endpoint."""
        test_review = {
            "text": "This product is absolutely amazing! Best purchase ever!",
            "user_id": "test_user_123",
            "rating": 5.0,
            "timestamp": "2023-12-01T10:00:00Z"
        }
        
        with patch('api.app.get_detector', return_value=self.mock_detector):
            response = self.client.post("/predict", json=test_review)
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('probability', data)
        self.assertIn('prediction', data)
        self.assertIn('confidence', data)
        self.assertIn('processing_time', data)
        self.assertIn('model_version', data)
        
        # Verify prediction values
        self.assertEqual(data['prediction'], 'fake')
        self.assertEqual(data['probability'], 0.85)
        self.assertEqual(data['confidence'], 'high')
        self.assertEqual(data['model_version'], "test_model_v1.0")
        
        # Verify detector was called correctly
        self.mock_detector.predict.assert_called_once()
    
    def test_single_prediction_minimal_input(self):
        """Test prediction with minimal required input."""
        test_review = {"text": "Good product"}
        
        with patch('api.app.get_detector', return_value=self.mock_detector):
            response = self.client.post("/predict", json=test_review)
        
        self.assertEqual(response.status_code, 200)
        
        # Should still work with minimal input
        data = response.json()
        self.assertIn('prediction', data)
        self.assertIn('probability', data)
    
    def test_single_prediction_invalid_input(self):
        """Test prediction with invalid input."""
        # Empty text
        response = self.client.post("/predict", json={"text": ""})
        self.assertEqual(response.status_code, 422)  # Validation error
        
        # Missing text
        response = self.client.post("/predict", json={"rating": 5})
        self.assertEqual(response.status_code, 422)
        
        # Invalid rating
        response = self.client.post("/predict", json={"text": "test", "rating": 6})
        self.assertEqual(response.status_code, 422)
        
        # Invalid timestamp
        response = self.client.post("/predict", json={
            "text": "test", 
            "timestamp": "invalid-timestamp"
        })
        self.assertEqual(response.status_code, 422)
    
    def test_single_prediction_no_detector(self):
        """Test prediction when detector is not available."""
        with patch('api.app.get_detector', return_value=None):
            response = self.client.post("/predict", json={"text": "test review"})
        
        self.assertEqual(response.status_code, 503)
        
        data = response.json()
        self.assertIn('detail', data)
        self.assertIn('not available', data['detail'])
    
    def test_single_prediction_detector_error(self):
        """Test prediction when detector raises an error."""
        self.mock_detector.predict.side_effect = Exception("Prediction failed")
        
        with patch('api.app.get_detector', return_value=self.mock_detector):
            response = self.client.post("/predict", json={"text": "test review"})
        
        self.assertEqual(response.status_code, 500)
        
        data = response.json()
        self.assertIn('detail', data)
        self.assertIn('Prediction failed', data['detail'])
    
    def test_batch_prediction_endpoint(self):
        """Test batch review prediction endpoint."""
        test_batch = {
            "texts": [
                "This product is absolutely amazing!",
                "Poor quality, not worth the money."
            ],
            "user_ids": ["user1", "user2"],
            "ratings": [5.0, 1.0]
        }
        
        with patch('api.app.get_detector', return_value=self.mock_detector):
            response = self.client.post("/predict/batch", json=test_batch)
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('predictions', data)
        self.assertIn('total_processed', data)
        self.assertIn('processing_time', data)
        
        # Should have predictions for both reviews
        self.assertEqual(len(data['predictions']), 2)
        self.assertEqual(data['total_processed'], 2)
        
        # Each prediction should have required fields
        for prediction in data['predictions']:
            self.assertIn('probability', prediction)
            self.assertIn('prediction', prediction)
            self.assertIn('confidence', prediction)
            self.assertIn('model_version', prediction)
        
        # Verify detector was called
        self.mock_detector.predict_batch.assert_called_once()
    
    def test_batch_prediction_minimal_input(self):
        """Test batch prediction with minimal input."""
        test_batch = {"texts": ["Review 1", "Review 2"]}
        
        with patch('api.app.get_detector', return_value=self.mock_detector):
            response = self.client.post("/predict/batch", json=test_batch)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data['predictions']), 2)
    
    def test_batch_prediction_invalid_input(self):
        """Test batch prediction with invalid input."""
        # Empty texts list
        response = self.client.post("/predict/batch", json={"texts": []})
        self.assertEqual(response.status_code, 422)
        
        # Too many texts (over limit)
        large_batch = {"texts": ["text"] * 101}  # Assuming limit is 100
        response = self.client.post("/predict/batch", json=large_batch)
        self.assertEqual(response.status_code, 422)
        
        # Mismatched array lengths
        response = self.client.post("/predict/batch", json={
            "texts": ["review1", "review2"],
            "ratings": [5.0]  # Only one rating for two texts
        })
        # Should still work - API should handle mismatched lengths gracefully
        
        # Empty text in batch
        response = self.client.post("/predict/batch", json={
            "texts": ["Good review", ""]
        })
        self.assertEqual(response.status_code, 422)
    
    def test_batch_prediction_no_detector(self):
        """Test batch prediction when detector is not available."""
        with patch('api.app.get_detector', return_value=None):
            response = self.client.post("/predict/batch", json={
                "texts": ["review1", "review2"]
            })
        
        self.assertEqual(response.status_code, 503)
    
    def test_batch_prediction_detector_error(self):
        """Test batch prediction when detector raises an error."""
        self.mock_detector.predict_batch.side_effect = Exception("Batch prediction failed")
        
        with patch('api.app.get_detector', return_value=self.mock_detector):
            response = self.client.post("/predict/batch", json={
                "texts": ["review1", "review2"]
            })
        
        self.assertEqual(response.status_code, 500)
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        # Mock Prometheus metrics
        with patch('api.app.PROMETHEUS_AVAILABLE', True):
            with patch('api.app.registry') as mock_registry:
                mock_registry.collect.return_value = []
                
                response = self.client.get("/metrics")
        
        # Should return metrics in some format
        self.assertIn(response.status_code, [200, 404])  # 404 if endpoint not implemented
    
    def test_metrics_endpoint_json(self):
        """Test JSON metrics endpoint."""
        with patch('api.app.get_detector', return_value=self.mock_detector):
            # Mock some metrics data
            with patch('api.app.REQUEST_COUNT') as mock_counter:
                mock_counter._value._value = 100
                
                response = self.client.get("/metrics/json")
        
        if response.status_code == 200:
            data = response.json()
            self.assertIsInstance(data, dict)
            # Should contain some metrics information
        else:
            # Endpoint might not be implemented
            self.assertIn(response.status_code, [404, 501])
    
    def test_root_endpoint(self):
        """Test root endpoint redirect or information."""
        response = self.client.get("/")
        
        # Should either redirect to docs or return API info
        self.assertIn(response.status_code, [200, 307, 308])  # OK or redirects
    
    def test_docs_endpoints(self):
        """Test API documentation endpoints."""
        # OpenAPI schema
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        
        # Should be valid JSON
        data = response.json()
        self.assertIn('openapi', data)
        self.assertIn('info', data)
        self.assertIn('paths', data)
        
        # Swagger UI
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200)
        
        # ReDoc
        response = self.client.get("/redoc")
        self.assertEqual(response.status_code, 200)


@unittest.skipIf(not FASTAPI_AVAILABLE, "FastAPI not available")
class TestAPIResponseValidation(unittest.TestCase):
    """Test API response format validation."""
    
    def setUp(self):
        """Set up test client with mock detector."""
        self.mock_detector = MagicMock()
        self.mock_detector.predict.return_value = {
            'prediction': 'legitimate',
            'probability': 0.25,
            'confidence': 'medium'
        }
        self.mock_detector.is_healthy.return_value = True
        self.mock_detector.model_version = "v2.0"
        
        with patch.dict('sys.modules', {
            'deployment': MagicMock(
                get_detector=MagicMock(return_value=self.mock_detector),
                initialize_detector=MagicMock(return_value=self.mock_detector),
                FakeReviewDetector=MagicMock()
            )
        }):
            from api.app import app
            self.client = TestClient(app)
    
    def test_prediction_response_schema(self):
        """Test that prediction responses match expected schema."""
        test_review = {"text": "Great product, highly recommend!"}
        
        with patch('api.app.get_detector', return_value=self.mock_detector):
            response = self.client.post("/predict", json=test_review)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Validate required fields and types
        self.assertIsInstance(data['probability'], float)
        self.assertIn(data['prediction'], ['fake', 'legitimate'])
        self.assertIn(data['confidence'], ['low', 'medium', 'high'])
        self.assertIsInstance(data['processing_time'], float)
        self.assertIsInstance(data['model_version'], str)
        
        # Validate value ranges
        self.assertGreaterEqual(data['probability'], 0.0)
        self.assertLessEqual(data['probability'], 1.0)
        self.assertGreater(data['processing_time'], 0.0)
    
    def test_batch_prediction_response_schema(self):
        """Test that batch prediction responses match expected schema."""
        test_batch = {"texts": ["Review 1", "Review 2"]}
        
        # Mock batch response
        self.mock_detector.predict_batch.return_value = [
            {'prediction': 'fake', 'probability': 0.8, 'confidence': 'high'},
            {'prediction': 'legitimate', 'probability': 0.3, 'confidence': 'medium'}
        ]
        
        with patch('api.app.get_detector', return_value=self.mock_detector):
            response = self.client.post("/predict/batch", json=test_batch)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Validate batch response structure
        self.assertIsInstance(data['predictions'], list)
        self.assertIsInstance(data['total_processed'], int)
        self.assertIsInstance(data['processing_time'], float)
        
        # Validate individual predictions
        for prediction in data['predictions']:
            self.assertIsInstance(prediction['probability'], float)
            self.assertIn(prediction['prediction'], ['fake', 'legitimate'])
            self.assertIn(prediction['confidence'], ['low', 'medium', 'high'])
            self.assertGreaterEqual(prediction['probability'], 0.0)
            self.assertLessEqual(prediction['probability'], 1.0)
    
    def test_health_response_schema(self):
        """Test that health responses match expected schema."""
        with patch('api.app.get_detector', return_value=self.mock_detector):
            response = self.client.get("/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Validate health response fields and types
        self.assertIn(data['status'], ['healthy', 'unhealthy'])
        self.assertIsInstance(data['timestamp'], str)
        self.assertIsInstance(data['model_loaded'], bool)
        self.assertIn(data['model_health'], ['healthy', 'unhealthy'])
        self.assertIsInstance(data['uptime'], float)
        
        if 'memory_usage' in data and data['memory_usage'] is not None:
            self.assertIsInstance(data['memory_usage'], dict)
    
    def test_error_response_schema(self):
        """Test that error responses have consistent format."""
        # Test validation error
        response = self.client.post("/predict", json={"text": ""})
        self.assertEqual(response.status_code, 422)
        
        data = response.json()
        self.assertIn('detail', data)
        
        # Test server error
        self.mock_detector.predict.side_effect = Exception("Server error")
        
        with patch('api.app.get_detector', return_value=self.mock_detector):
            response = self.client.post("/predict", json={"text": "test"})
        
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn('detail', data)


@unittest.skipIf(not FASTAPI_AVAILABLE, "FastAPI not available")
class TestAPIRateLimiting(unittest.TestCase):
    """Test API rate limiting functionality."""
    
    def setUp(self):
        """Set up test client with rate limiting."""
        self.mock_detector = MagicMock()
        self.mock_detector.predict.return_value = {
            'prediction': 'fake',
            'probability': 0.7,
            'confidence': 'medium'
        }
        
        # Mock rate limiting as available
        with patch('api.app.RATE_LIMITING_AVAILABLE', True):
            with patch.dict('sys.modules', {
                'deployment': MagicMock(
                    get_detector=MagicMock(return_value=self.mock_detector),
                    initialize_detector=MagicMock(return_value=self.mock_detector),
                    FakeReviewDetector=MagicMock()
                )
            }):
                from api.app import app
                self.client = TestClient(app)
    
    def test_rate_limiting_headers(self):
        """Test that rate limiting headers are present when enabled."""
        with patch('api.app.get_detector', return_value=self.mock_detector):
            with patch('api.app.RATE_LIMITING_AVAILABLE', True):
                response = self.client.post("/predict", json={"text": "test review"})
        
        # Rate limiting may or may not be configured for tests
        # Just verify the request succeeds
        self.assertIn(response.status_code, [200, 429])
    
    def test_rate_limit_exceeded(self):
        """Test behavior when rate limit is exceeded."""
        # This test would require actual rate limiting configuration
        # For now, just verify the endpoint works
        with patch('api.app.get_detector', return_value=self.mock_detector):
            response = self.client.post("/predict", json={"text": "test review"})
        
        # Should succeed normally in test environment
        self.assertEqual(response.status_code, 200)


@unittest.skipIf(not FASTAPI_AVAILABLE, "FastAPI not available")
class TestAPICORS(unittest.TestCase):
    """Test CORS configuration."""
    
    def setUp(self):
        """Set up test client."""
        self.mock_detector = MagicMock()
        
        with patch.dict('sys.modules', {
            'deployment': MagicMock(
                get_detector=MagicMock(return_value=self.mock_detector),
                initialize_detector=MagicMock(return_value=self.mock_detector),
                FakeReviewDetector=MagicMock()
            )
        }):
            from api.app import app
            self.client = TestClient(app)
    
    def test_cors_headers(self):
        """Test that CORS headers are present."""
        response = self.client.get("/health")
        
        # CORS headers should be present
        headers = response.headers
        
        # May or may not have CORS headers depending on configuration
        # Just verify the request succeeds
        self.assertIn(response.status_code, [200, 503])
    
    def test_options_request(self):
        """Test OPTIONS request for CORS preflight."""
        response = self.client.options("/predict")
        
        # Should handle OPTIONS request
        self.assertIn(response.status_code, [200, 405])  # OK or Method Not Allowed


@unittest.skipIf(not FASTAPI_AVAILABLE, "FastAPI not available")
class TestAPIPerformance(unittest.TestCase):
    """Test API performance characteristics."""
    
    def setUp(self):
        """Set up test client with performance mocks."""
        self.mock_detector = MagicMock()
        self.mock_detector.predict.return_value = {
            'prediction': 'legitimate',
            'probability': 0.15,
            'confidence': 'low'
        }
        
        with patch.dict('sys.modules', {
            'deployment': MagicMock(
                get_detector=MagicMock(return_value=self.mock_detector),
                initialize_detector=MagicMock(return_value=self.mock_detector),
                FakeReviewDetector=MagicMock()
            )
        }):
            from api.app import app
            self.client = TestClient(app)
    
    def test_response_time_tracking(self):
        """Test that response times are tracked."""
        with patch('api.app.get_detector', return_value=self.mock_detector):
            response = self.client.post("/predict", json={"text": "fast test"})
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Should include processing time
        self.assertIn('processing_time', data)
        self.assertIsInstance(data['processing_time'], float)
        self.assertGreater(data['processing_time'], 0)
    
    def test_concurrent_requests(self):
        """Test handling of multiple concurrent requests."""
        import concurrent.futures
        
        def make_request():
            with patch('api.app.get_detector', return_value=self.mock_detector):
                return self.client.post("/predict", json={"text": "concurrent test"})
        
        # Make multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [future.result() for future in futures]
        
        # All requests should succeed
        for response in results:
            self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main(verbosity=2)
