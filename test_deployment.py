#!/usr/bin/env python3
"""
Test script for Fake Review Detection deployment and API.

This script tests the FakeReviewDetector class and API endpoints
to ensure everything is working correctly.
"""

import sys
import asyncio
import time
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import our modules
from deployment import FakeReviewDetector, initialize_detector
from src.preprocessing import TextPreprocessor
from src import utils


def create_mock_model_and_data():
    """Create mock model and data for testing."""
    print("Creating mock model and test data...")
    
    # Create sample data
    sample_data = {
        'review_text': [
            "This product is amazing! I love it so much!",
            "Terrible quality. Would not recommend.",
            "Good value for money. Works as expected.",
            "Best purchase ever! 5 stars! Amazing quality!",
            "Poor customer service and bad product quality.",
            "Excellent product, fast shipping, great value",
            "Disappointing. Not as described. Waste of money.",
            "Perfect! Exactly what I needed. Highly recommended.",
            "Cheap quality, broke after one day of use",
            "Outstanding service and product quality is superb"
        ],
        'rating': [5, 1, 4, 5, 2, 5, 1, 5, 1, 5],
        'user_id': [f'user_{i}' for i in range(10)],
        'product_id': [f'prod_{i%3}' for i in range(10)],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                 '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10'],
        'is_fake': [1, 0, 0, 1, 0, 0, 0, 0, 0, 1]  # Mock labels
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create and fit preprocessor
    preprocessor = TextPreprocessor(
        feature_engineering=True,
        temporal_check=False,
        use_smote=False
    )
    
    processed_df = preprocessor.fit_transform(df)
    
    # Extract features for training
    feature_columns = processed_df.select_dtypes(include=[np.number]).columns
    X = processed_df[feature_columns].fillna(0)
    y = df['is_fake']
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    return model, preprocessor, df

def test_fake_review_detector():
    """Test the FakeReviewDetector class."""
    print("\n" + "="*60)
    print("Testing FakeReviewDetector")
    print("="*60)
    
    # Create mock model and data
    model, preprocessor, test_data = create_mock_model_and_data()
    
    # Save model temporarily
    models_dir = PROJECT_ROOT / 'artifacts' / 'models'
    utils.ensure_dir(models_dir)
    
    model_path = models_dir / 'test_model.joblib'
    preprocessor_path = models_dir / 'test_preprocessor.joblib'
    
    # Save in the expected format
    model_data = {
        'model': model,
        'trainer_config': {'test': True},
        'best_score': 0.85,
        'best_params': {'n_estimators': 10},
        'timestamp': '2023-12-01 12:00:00'
    }
    
    utils.save_joblib(model_data, model_path)
    utils.save_joblib(preprocessor, preprocessor_path)
    
    try:
        # Test detector initialization
        print("1. Testing detector initialization...")
        detector = FakeReviewDetector(
            model_path=model_path,
            preprocessor_path=preprocessor_path
        )
        print("✓ Detector initialized successfully")
        
        # Test single prediction
        print("\n2. Testing single prediction...")
        test_text = "This product is absolutely amazing! Best purchase ever!"
        probability = detector.predict(test_text)
        print(f"✓ Prediction: {probability:.3f}")
        
        # Test prediction with additional features
        print("\n3. Testing prediction with additional features...")
        probability = detector.predict(
            text=test_text,
            user_id="test_user",
            timestamp="2023-12-01T12:00:00",
            additional_features={'rating': 5}
        )
        print(f"✓ Prediction with features: {probability:.3f}")
        
        # Test batch prediction
        print("\n4. Testing batch prediction...")
        test_texts = [
            "Great product, highly recommend!",
            "Terrible quality, complete waste of money",
            "Average product, okay value for money"
        ]
        probabilities = detector.predict_batch(test_texts)
        print(f"✓ Batch predictions: {[f'{p:.3f}' for p in probabilities]}")
        
        # Test model info
        print("\n5. Testing model info...")
        info = detector.get_model_info()
        print(f"✓ Model info: {info['model_type']}, Status: {info['status']}")
        
        # Test health check
        print("\n6. Testing health check...")
        health = detector.health_check()
        print(f"✓ Health check: {health['status']}")
        
        print("\n✓ All FakeReviewDetector tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ FakeReviewDetector test failed: {e}")
        return False
    
    finally:
        # Clean up test files
        if model_path.exists():
            model_path.unlink()
        if preprocessor_path.exists():
            preprocessor_path.unlink()

def test_api_endpoints():
    """Test API endpoints (requires the API to be running)."""
    print("\n" + "="*60)
    print("Testing API Endpoints")
    print("="*60)
    
    try:
        import httpx
        
        base_url = "http://localhost:8000"
        
        # Test root endpoint
        print("1. Testing root endpoint...")
        with httpx.Client() as client:
            response = client.get(f"{base_url}/")
            if response.status_code == 200:
                print("✓ Root endpoint working")
            else:
                print(f"✗ Root endpoint failed: {response.status_code}")
                return False
        
        # Test health endpoint
        print("\n2. Testing health endpoint...")
        with httpx.Client() as client:
            response = client.get(f"{base_url}/health")
            if response.status_code in [200, 503]:  # 503 is acceptable if model not loaded
                health_data = response.json()
                print(f"✓ Health endpoint: {health_data['status']}")
            else:
                print(f"✗ Health endpoint failed: {response.status_code}")
                return False
        
        # Test prediction endpoint (if model is loaded)
        print("\n3. Testing prediction endpoint...")
        prediction_data = {
            "text": "This product is amazing! I love it so much!",
            "user_id": "test_user",
            "rating": 5
        }
        
        with httpx.Client() as client:
            response = client.post(f"{base_url}/predict", json=prediction_data)
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Prediction endpoint: probability={result['probability']:.3f}")
            elif response.status_code == 503:
                print("⚠ Prediction endpoint: Model not loaded (expected if no trained model)")
            else:
                print(f"✗ Prediction endpoint failed: {response.status_code}")
                return False
        
        # Test metrics endpoint
        print("\n4. Testing metrics endpoint...")
        with httpx.Client() as client:
            response = client.get(f"{base_url}/metrics")
            if response.status_code in [200, 501]:  # 501 if Prometheus not available
                print("✓ Metrics endpoint working")
            else:
                print(f"✗ Metrics endpoint failed: {response.status_code}")
                return False
        
        print("\n✓ All API endpoint tests passed!")
        return True
        
    except ImportError:
        print("⚠ httpx not available, skipping API tests")
        return True
    except Exception as e:
        print(f"✗ API test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\n" + "="*60)
    print("Testing Configuration")
    print("="*60)
    
    try:
        # Test settings loading
        print("1. Testing settings loading...")
        settings = utils.load_settings()
        print(f"✓ Settings loaded: {len(settings)} sections")
        
        # Test specific settings
        print("\n2. Testing specific settings...")
        api_host = utils.get_setting('api.host', 'default')
        api_port = utils.get_setting('api.port', 8000)
        print(f"✓ API settings: {api_host}:{api_port}")
        
        model_path_setting = utils.get_setting('deployment.model_path')
        print(f"✓ Model path setting: {model_path_setting}")
        
        print("\n✓ All configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Starting Fake Review Detection Deployment Tests")
    print("="*60)
    
    test_results = []
    
    # Test configuration
    test_results.append(test_configuration())
    
    # Test FakeReviewDetector
    test_results.append(test_fake_review_detector())
    
    # Test API endpoints (optional - requires running server)
    print(f"\nNote: To test API endpoints, run the server first:")
    print(f"  python run_api.py --debug")
    print(f"Then run this test script again with --api flag")
    
    if "--api" in sys.argv:
        test_results.append(test_api_endpoints())
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Deployment is ready.")
        return 0
    else:
        print("✗ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
