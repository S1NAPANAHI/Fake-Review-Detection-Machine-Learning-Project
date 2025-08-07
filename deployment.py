"""
Deployment module for Fake Review Detection System.

This module provides the FakeReviewDetector class for model serving,
which loads the trained model and preprocessing pipeline and exposes
a predict method for inference.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

# Import utilities
from src import utils
from src.preprocessing import TextPreprocessor
from src.monitoring import create_monitor, ModelMonitor


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass


class PredictionError(Exception):
    """Exception raised when prediction fails."""
    pass


class FakeReviewDetector:
    """
    Fake Review Detection model wrapper for serving predictions.
    
    This class encapsulates the trained model and preprocessing pipeline,
    providing a clean interface for making predictions on review data.
    """
    
    def __init__(self, 
                 model_path: Optional[Union[str, Path]] = None,
                 preprocessor_path: Optional[Union[str, Path]] = None,
                 load_on_init: bool = True):
        """
        Initialize the FakeReviewDetector.
        
        Args:
            model_path: Path to the trained model file
            preprocessor_path: Path to the preprocessing pipeline
            load_on_init: Whether to load model immediately on initialization
        """
        self.logger = utils.setup_logging(__name__)
        
        # Initialize paths
        self.model_path = self._resolve_model_path(model_path)
        self.preprocessor_path = self._resolve_preprocessor_path(preprocessor_path)
        
        # Initialize components
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        self.model_metadata = {}
        
        self.logger.info("FakeReviewDetector initialized")
        
        if load_on_init:
            self.load_model()
    
    def _resolve_model_path(self, model_path: Optional[Union[str, Path]]) -> Path:
        """
        Resolve the model path from settings or provided path.
        
        Args:
            model_path: Provided model path
            
        Returns:
            Resolved path to model file
        """
        if model_path is not None:
            return utils.resolve_path(model_path)
        
        # Try to get from settings
        model_path_setting = utils.get_setting('deployment.model_path')
        if model_path_setting:
            return utils.resolve_path(model_path_setting)
        
        # Fallback to models directory
        models_dir = utils.resolve_path(utils.get_setting('paths.models', 'artifacts/models'))
        
        # Look for the most recent model file
        if models_dir.exists():
            model_files = list(models_dir.glob('*.joblib'))
            if model_files:
                # Sort by modification time and get the most recent
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                self.logger.info(f"Auto-detected latest model: {latest_model}")
                return latest_model
        
        raise ModelLoadError(f"No model file found. Please specify model_path or place a model in {models_dir}")
    
    def _resolve_preprocessor_path(self, preprocessor_path: Optional[Union[str, Path]]) -> Optional[Path]:
        """
        Resolve the preprocessor path from settings or provided path.
        
        Args:
            preprocessor_path: Provided preprocessor path
            
        Returns:
            Resolved path to preprocessor file or None if not found
        """
        if preprocessor_path is not None:
            return utils.resolve_path(preprocessor_path)
        
        # Try to get from settings
        preprocessor_path_setting = utils.get_setting('deployment.preprocessor_path')
        if preprocessor_path_setting:
            return utils.resolve_path(preprocessor_path_setting)
        
        # Try to find preprocessor in same directory as model
        model_dir = self.model_path.parent if hasattr(self, 'model_path') else Path('.')
        preprocessor_files = list(model_dir.glob('*preprocessor*.joblib'))
        
        if preprocessor_files:
            latest_preprocessor = max(preprocessor_files, key=lambda x: x.stat().st_mtime)
            self.logger.info(f"Auto-detected preprocessor: {latest_preprocessor}")
            return latest_preprocessor
        
        self.logger.warning("No preprocessor found. Will initialize a default one.")
        return None
    
    @utils.timing_decorator
    def load_model(self) -> None:
        """
        Load the trained model and preprocessing pipeline.
        
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            self.logger.info(f"Loading model from: {self.model_path}")
            
            # Load model data
            model_data = utils.load_joblib(self.model_path)
            
            # Handle different model file formats
            if isinstance(model_data, dict):
                # New format with metadata
                self.model = model_data['model']
                self.model_metadata = {
                    'trainer_config': model_data.get('trainer_config', {}),
                    'cv_results': model_data.get('cv_results', {}),
                    'best_score': model_data.get('best_score'),
                    'best_params': model_data.get('best_params', {}),
                    'timestamp': model_data.get('timestamp'),
                    'model_type': type(self.model).__name__
                }
            else:
                # Legacy format (direct model object)
                self.model = model_data
                self.model_metadata = {
                    'model_type': type(self.model).__name__,
                    'loaded_at': datetime.now().isoformat()
                }
            
            # Validate model
            if not hasattr(self.model, 'predict') or not hasattr(self.model, 'predict_proba'):
                raise ModelLoadError("Loaded object is not a valid classifier")
            
            # Load or initialize preprocessor
            self._load_preprocessor()
            
            self.is_loaded = True
            self.logger.info(f"Model loaded successfully: {self.model_metadata.get('model_type')}")
            
            # Log model metadata
            if self.model_metadata.get('best_score'):
                self.logger.info(f"Model performance - Best score: {self.model_metadata['best_score']:.4f}")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}")
    
    def _load_preprocessor(self) -> None:
        """Load or initialize the text preprocessor."""
        if self.preprocessor_path and self.preprocessor_path.exists():
            try:
                self.logger.info(f"Loading preprocessor from: {self.preprocessor_path}")
                self.preprocessor = utils.load_joblib(self.preprocessor_path)
                
                # Validate preprocessor
                if not hasattr(self.preprocessor, 'transform'):
                    raise ValueError("Loaded preprocessor is not valid")
                    
            except Exception as e:
                self.logger.warning(f"Failed to load preprocessor: {e}")
                self._initialize_default_preprocessor()
        else:
            self._initialize_default_preprocessor()
    
    def _initialize_default_preprocessor(self) -> None:
        """Initialize a default preprocessor."""
        self.logger.info("Initializing default preprocessor")
        
        self.preprocessor = TextPreprocessor(
            text_column='text',
            lowercase=True,
            remove_html=True,
            remove_urls=True,
            remove_emails=True,
            remove_phone_numbers=True,
            remove_punctuation=True,
            remove_extra_whitespace=True,
            remove_stopwords=True,
            lemmatize=True,
            feature_engineering=True,
            temporal_check=False,  # Disable for single predictions
            use_smote=False  # Disable for prediction
        )
    
    def predict(self, 
                text: str, 
                user_id: Optional[str] = None, 
                timestamp: Optional[Union[str, datetime]] = None,
                additional_features: Optional[Dict[str, Any]] = None) -> float:
        """
        Predict the probability that a review is fake.
        
        Args:
            text: Review text to analyze
            user_id: User ID (optional, for behavioral features)
            timestamp: Review timestamp (optional, for temporal features)
            additional_features: Additional features as key-value pairs
            
        Returns:
            Probability that the review is fake (0-1)
            
        Raises:
            PredictionError: If prediction fails
        """
        if not self.is_loaded:
            raise PredictionError("Model not loaded. Call load_model() first.")
        
        try:
            # Record prediction metrics
            utils.metrics.increment_counter('prediction_requests')
            
            with utils.metrics_timer('prediction_duration'):
                # Prepare input data
                input_data = self._prepare_input(text, user_id, timestamp, additional_features)
                
                # Preprocess the input
                processed_data = self._preprocess_input(input_data)
                
                # Make prediction
                prediction_proba = self.model.predict_proba(processed_data)
                
                # Extract probability for fake class (assuming binary classification)
                if prediction_proba.shape[1] == 2:
                    # Binary classification: return probability of positive class (fake)
                    fake_probability = prediction_proba[0, 1]
                else:
                    # Multi-class: return max probability
                    fake_probability = prediction_proba[0].max()
                
                # Record prediction metrics
                utils.metrics.observe_histogram('prediction_probability', fake_probability)
                utils.metrics.increment_counter('predictions_completed')
                
                self.logger.debug(f"Prediction completed: {fake_probability:.4f}")
                
                return float(fake_probability)
                
        except Exception as e:
            utils.metrics.increment_counter('prediction_errors')
            raise PredictionError(f"Prediction failed: {str(e)}")
    
    def _prepare_input(self, 
                      text: str, 
                      user_id: Optional[str], 
                      timestamp: Optional[Union[str, datetime]],
                      additional_features: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """
        Prepare input data for preprocessing and prediction.
        
        Args:
            text: Review text
            user_id: User ID
            timestamp: Review timestamp
            additional_features: Additional features
            
        Returns:
            DataFrame with prepared input data
        """
        # Create base input data
        input_dict = {
            'text': text,
            'review_text': text,  # Alternative column name
        }
        
        # Add optional fields
        if user_id is not None:
            input_dict['user_id'] = user_id
        
        if timestamp is not None:
            if isinstance(timestamp, str):
                try:
                    timestamp = pd.to_datetime(timestamp)
                except:
                    self.logger.warning(f"Failed to parse timestamp: {timestamp}")
                    timestamp = datetime.now()
            input_dict['date'] = timestamp
            input_dict['timestamp'] = timestamp
        
        # Add additional features
        if additional_features:
            for key, value in additional_features.items():
                if key not in input_dict:  # Don't override existing keys
                    input_dict[key] = value
        
        # Create DataFrame
        df = pd.DataFrame([input_dict])
        
        return df
    
    def _preprocess_input(self, input_data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess input data using the loaded preprocessor.
        
        Args:
            input_data: Raw input DataFrame
            
        Returns:
            Preprocessed feature array
        """
        try:
            # Use preprocessor if available
            if self.preprocessor and hasattr(self.preprocessor, 'transform'):
                # Ensure preprocessor is fitted
                if not getattr(self.preprocessor, 'is_fitted', False):
                    self.preprocessor.fit(input_data)
                
                processed_data = self.preprocessor.transform(input_data)
                
                # Extract numerical columns for model input
                numerical_columns = processed_data.select_dtypes(include=[np.number]).columns
                feature_array = processed_data[numerical_columns].values
                
            else:
                # Fallback to basic preprocessing
                feature_array = self._basic_preprocessing(input_data)
            
            # Ensure we have a 2D array
            if feature_array.ndim == 1:
                feature_array = feature_array.reshape(1, -1)
            
            return feature_array
            
        except Exception as e:
            raise PredictionError(f"Preprocessing failed: {str(e)}")
    
    def _basic_preprocessing(self, input_data: pd.DataFrame) -> np.ndarray:
        """
        Basic preprocessing fallback when no preprocessor is available.
        
        Args:
            input_data: Raw input DataFrame
            
        Returns:
            Basic feature array
        """
        features = []
        
        # Text length features
        text_col = 'text' if 'text' in input_data.columns else 'review_text'
        if text_col in input_data.columns:
            text = str(input_data[text_col].iloc[0])
            features.extend([
                len(text),  # text length
                len(text.split()),  # word count
                text.count('.') + text.count('!') + text.count('?'),  # sentence count
                len([c for c in text if c.isupper()]) / (len(text) + 1),  # uppercase ratio
                len([c for c in text if c.isdigit()]) / (len(text) + 1),  # digit ratio
            ])
        else:
            features.extend([0, 0, 0, 0, 0])  # Default values
        
        # Rating feature (if available)
        if 'rating' in input_data.columns:
            rating = input_data['rating'].iloc[0]
            features.append(float(rating) if pd.notna(rating) else 3.0)
        else:
            features.append(3.0)  # Default rating
        
        return np.array(features).reshape(1, -1)
    
    def predict_batch(self, 
                     texts: list, 
                     user_ids: Optional[list] = None,
                     timestamps: Optional[list] = None,
                     additional_features_list: Optional[list] = None) -> list:
        """
        Predict probabilities for a batch of reviews.
        
        Args:
            texts: List of review texts
            user_ids: List of user IDs (optional)
            timestamps: List of timestamps (optional)
            additional_features_list: List of additional feature dictionaries (optional)
            
        Returns:
            List of fake probabilities
        """
        if not self.is_loaded:
            raise PredictionError("Model not loaded. Call load_model() first.")
        
        predictions = []
        
        for i, text in enumerate(texts):
            user_id = user_ids[i] if user_ids else None
            timestamp = timestamps[i] if timestamps else None
            additional_features = additional_features_list[i] if additional_features_list else None
            
            try:
                prob = self.predict(text, user_id, timestamp, additional_features)
                predictions.append(prob)
            except Exception as e:
                self.logger.error(f"Failed to predict for item {i}: {e}")
                predictions.append(0.5)  # Default probability
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        info = {
            "status": "loaded",
            "model_path": str(self.model_path),
            "preprocessor_path": str(self.preprocessor_path) if self.preprocessor_path else None,
            "model_type": self.model_metadata.get('model_type', 'unknown'),
            "loaded_at": datetime.now().isoformat(),
        }
        
        # Add performance metrics if available
        if self.model_metadata.get('best_score'):
            info['performance'] = {
                'best_score': self.model_metadata['best_score'],
                'best_params': self.model_metadata.get('best_params', {}),
                'cv_results': self.model_metadata.get('cv_results', {})
            }
        
        return info
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the model.
        
        Returns:
            Dictionary with health status
        """
        status = {
            "status": "healthy" if self.is_loaded else "unhealthy",
            "model_loaded": self.is_loaded,
            "preprocessor_loaded": self.preprocessor is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Test prediction if model is loaded
        if self.is_loaded:
            try:
                test_prob = self.predict("This is a test review")
                status["test_prediction"] = test_prob
                status["prediction_test"] = "passed"
            except Exception as e:
                status["status"] = "unhealthy"
                status["prediction_test"] = "failed"
                status["error"] = str(e)
        
        return status


# Global detector instance for API
detector = None


def get_detector() -> FakeReviewDetector:
    """
    Get or create the global detector instance.
    
    Returns:
        FakeReviewDetector instance
    """
    global detector
    if detector is None:
        detector = FakeReviewDetector()
    return detector


def initialize_detector(model_path: Optional[str] = None, 
                       preprocessor_path: Optional[str] = None) -> FakeReviewDetector:
    """
    Initialize the global detector instance.
    
    Args:
        model_path: Path to model file
        preprocessor_path: Path to preprocessor file
        
    Returns:
        Initialized FakeReviewDetector instance
    """
    global detector
    detector = FakeReviewDetector(model_path, preprocessor_path)
    return detector
