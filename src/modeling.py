"""
Machine Learning Model Training and Selection Module.

This module provides the ModelTrainer class for:
- Training multiple ML algorithms (LogisticRegression, RandomForest, SVM, XGBoost)
- Handling class imbalance through sample weighting
- Hyperparameter optimization using Bayesian and Grid Search
- Model selection and cross-validation
- Model persistence using utils
"""

import warnings
import logging
from typing import Dict, Any, Tuple, Optional, List, Union
from pathlib import Path
import numpy as np
import pandas as pd

# Scikit-learn imports
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator

# XGBoost
import xgboost as xgb

# Bayesian optimization
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    warnings.warn("scikit-optimize not available. Falling back to GridSearchCV only.")

# Import utilities
from . import utils

logger = utils.setup_logging(__name__)


class ModelTrainer:
    """
    Comprehensive model training and selection class.
    
    Supports training of multiple ML algorithms with hyperparameter optimization,
    class imbalance handling, and automated model selection based on cross-validation.
    """
    
    def __init__(self, 
                 random_state: Optional[int] = None,
                 cv_folds: int = 5,
                 scoring: str = 'f1',
                 handle_imbalance: bool = True,
                 use_bayesian_search: bool = True,
                 n_iter: int = 50,
                 n_jobs: int = -1,
                 verbose: bool = True):
        """
        Initialize the ModelTrainer.
        
        Args:
            random_state: Random state for reproducibility
            cv_folds: Number of cross-validation folds
            scoring: Primary scoring metric for model selection
            handle_imbalance: Whether to handle class imbalance with sample weights
            use_bayesian_search: Use BayesianSearchCV if available, else GridSearchCV
            n_iter: Number of iterations for Bayesian search
            n_jobs: Number of parallel jobs (-1 for all cores)
            verbose: Enable verbose logging
        """
        self.random_state = random_state or utils.get_setting('environment.random_seed', 42)
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.handle_imbalance = handle_imbalance
        self.use_bayesian_search = use_bayesian_search and BAYESIAN_AVAILABLE
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Initialize model configurations
        self._initialize_models()
        self._initialize_param_grids()
        
        # Results storage
        self.results_ = {}
        self.best_model_ = None
        self.best_score_ = None
        self.best_params_ = None
        
        if self.verbose:
            search_method = "BayesianSearchCV" if self.use_bayesian_search else "GridSearchCV"
            logger.info(f"ModelTrainer initialized with {search_method}")
    
    def _initialize_models(self) -> None:
        """Initialize base models with default configurations."""
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='liblinear'
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100
            ),
            'svm': SVC(
                random_state=self.random_state,
                probability=True  # Enable probability estimates for ROC-AUC
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                verbosity=0
            )
        }
    
    def _initialize_param_grids(self) -> None:
        """Initialize hyperparameter grids for each model."""
        if self.use_bayesian_search:
            self._initialize_bayesian_spaces()
        else:
            self._initialize_grid_spaces()
    
    def _initialize_bayesian_spaces(self) -> None:
        """Initialize Bayesian search spaces."""
        self.param_spaces = {
            'logistic_regression': {
                'C': Real(0.01, 100.0, prior='log-uniform'),
                'penalty': Categorical(['l1', 'l2']),
                'solver': Categorical(['liblinear', 'saga'])
            },
            'random_forest': {
                'n_estimators': Integer(50, 300),
                'max_depth': Integer(5, 20),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 10),
                'max_features': Categorical(['sqrt', 'log2', None])
            },
            'svm': {
                'C': Real(0.01, 100.0, prior='log-uniform'),
                'gamma': Real(0.001, 1.0, prior='log-uniform'),
                'kernel': Categorical(['rbf', 'linear', 'poly'])
            },
            'xgboost': {
                'n_estimators': Integer(50, 300),
                'max_depth': Integer(3, 10),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'subsample': Real(0.6, 1.0),
                'colsample_bytree': Real(0.6, 1.0),
                'reg_alpha': Real(0.0, 10.0),
                'reg_lambda': Real(1.0, 10.0)
            }
        }
    
    def _initialize_grid_spaces(self) -> None:
        """Initialize grid search spaces."""
        self.param_grids = {
            'logistic_regression': {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'svm': {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                'kernel': ['rbf', 'linear', 'poly']
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        }
    
    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for handling imbalanced datasets.
        
        Args:
            y: Target labels
            
        Returns:
            Dict mapping class labels to weights
        """
        if not self.handle_imbalance:
            return None
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        
        if self.verbose:
            logger.info(f"Computed class weights: {class_weights}")
        
        return class_weights
    
    def _get_scoring_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive scoring metrics for cross-validation.
        
        Returns:
            Dictionary of scoring functions
        """
        return {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall': make_scorer(recall_score, average='weighted', zero_division=0),
            'f1': make_scorer(f1_score, average='weighted', zero_division=0),
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='weighted')
        }
    
    def _train_single_model(self, 
                           model_name: str, 
                           X: np.ndarray, 
                           y: np.ndarray) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """
        Train a single model with hyperparameter optimization.
        
        Args:
            model_name: Name of the model to train
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (best_estimator, cv_results)
        """
        if self.verbose:
            logger.info(f"Training {model_name}...")
        
        base_model = self.models[model_name].set_params(random_state=self.random_state)
        
        # Handle class weights for models that support it
        class_weights = self._compute_class_weights(y)
        if class_weights and model_name in ['logistic_regression', 'random_forest', 'svm', 'xgboost']:
            if model_name == 'xgboost':
                # XGBoost uses scale_pos_weight instead of class_weight
                if len(class_weights) == 2:
                    scale_pos_weight = class_weights[0] / class_weights[1]
                    base_model.set_params(scale_pos_weight=scale_pos_weight)
            else:
                base_model.set_params(class_weight=class_weights)
        
        # Set up cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Choose search strategy
        if self.use_bayesian_search:
            search = BayesSearchCV(
                estimator=base_model,
                search_spaces=self.param_spaces[model_name],
                n_iter=self.n_iter,
                cv=cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=0
            )
        else:
            search = GridSearchCV(
                estimator=base_model,
                param_grid=self.param_grids[model_name],
                cv=cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=0
            )
        
        # Fit the search
        with utils.suppress_warnings():
            search.fit(X, y)
        
        # Get comprehensive cross-validation results
        scoring_metrics = self._get_scoring_metrics()
        cv_results = cross_validate(
            search.best_estimator_,
            X, y,
            cv=cv,
            scoring=scoring_metrics,
            n_jobs=self.n_jobs,
            return_train_score=True
        )
        
        # Compute summary statistics
        cv_summary = {}
        for metric in scoring_metrics.keys():
            cv_summary[f'test_{metric}_mean'] = cv_results[f'test_{metric}'].mean()
            cv_summary[f'test_{metric}_std'] = cv_results[f'test_{metric}'].std()
            cv_summary[f'train_{metric}_mean'] = cv_results[f'train_{metric}'].mean()
            cv_summary[f'train_{metric}_std'] = cv_results[f'train_{metric}'].std()
        
        cv_summary['best_score'] = search.best_score_
        cv_summary['best_params'] = search.best_params_
        cv_summary['fit_time_mean'] = cv_results['fit_time'].mean()
        cv_summary['score_time_mean'] = cv_results['score_time'].mean()
        
        if self.verbose:
            logger.info(f"{model_name} - Best {self.scoring}: {search.best_score_:.4f}")
            logger.info(f"{model_name} - Best params: {search.best_params_}")
        
        return search.best_estimator_, cv_summary
    
    def select_best(self, X: np.ndarray, y: np.ndarray) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """
        Train all models and select the best performer.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (best_fitted_model, all_cv_results)
        """
        if self.verbose:
            logger.info("Starting model selection process...")
            logger.info(f"Training data shape: {X.shape}")
            logger.info(f"Target distribution: {np.bincount(y)}")
        
        # Validate inputs
        utils.validate_dataframe(pd.DataFrame(X), min_rows=self.cv_folds)
        
        # Train all models
        trained_models = {}
        all_results = {}
        
        for model_name in self.models.keys():
            try:
                with utils.metrics_timer(f"model_training_{model_name}"):
                    model, results = self._train_single_model(model_name, X, y)
                    trained_models[model_name] = model
                    all_results[model_name] = results
                    
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                continue
        
        if not trained_models:
            raise ValueError("No models were successfully trained")
        
        # Select best model based on primary scoring metric
        best_model_name = max(
            all_results.keys(),
            key=lambda name: all_results[name]['best_score']
        )
        
        self.best_model_ = trained_models[best_model_name]
        self.best_score_ = all_results[best_model_name]['best_score']
        self.best_params_ = all_results[best_model_name]['best_params']
        self.results_ = all_results
        
        if self.verbose:
            logger.info(f"Best model: {best_model_name}")
            logger.info(f"Best {self.scoring} score: {self.best_score_:.4f}")
            
            # Log comparison of all models
            logger.info("Model comparison:")
            for name, results in all_results.items():
                score = results['best_score']
                logger.info(f"  {name}: {score:.4f}")
        
        # Record metrics
        utils.metrics.set_gauge('best_model_score', self.best_score_)
        utils.metrics.increment_counter('model_selection_completed')
        
        return self.best_model_, self.results_
    
    def save_model(self, 
                   model: Optional[BaseEstimator] = None,
                   filepath: Optional[Union[str, Path]] = None,
                   include_results: bool = True) -> Path:
        """
        Save trained model using utils.
        
        Args:
            model: Model to save (uses best model if None)
            filepath: Path to save model (auto-generated if None)
            include_results: Whether to save CV results alongside model
            
        Returns:
            Path where model was saved
        """
        # Use best model if none specified
        if model is None:
            if self.best_model_ is None:
                raise ValueError("No model to save. Train a model first using select_best()")
            model = self.best_model_
        
        # Generate filepath if not provided
        if filepath is None:
            timestamp = utils.get_timestamp()
            models_dir = utils.resolve_path("models")
            utils.ensure_dir(models_dir)
            filepath = models_dir / f"best_model_{timestamp}.joblib"
        else:
            filepath = utils.resolve_path(filepath)
        
        # Prepare data to save
        model_data = {
            'model': model,
            'trainer_config': {
                'random_state': self.random_state,
                'cv_folds': self.cv_folds,
                'scoring': self.scoring,
                'handle_imbalance': self.handle_imbalance,
                'use_bayesian_search': self.use_bayesian_search
            },
            'timestamp': utils.get_timestamp("%Y-%m-%d %H:%M:%S")
        }
        
        if include_results and hasattr(self, 'results_'):
            model_data['cv_results'] = self.results_
            model_data['best_score'] = self.best_score_
            model_data['best_params'] = self.best_params_
        
        # Save using utils
        saved_path = utils.save_joblib(model_data, filepath)
        
        if self.verbose:
            logger.info(f"Model saved to: {saved_path}")
        
        utils.metrics.increment_counter('models_saved')
        return saved_path
    
    def load_model(self, filepath: Union[str, Path]) -> BaseEstimator:
        """
        Load a previously saved model.
        
        Args:
            filepath: Path to saved model file
            
        Returns:
            Loaded model
        """
        model_data = utils.load_joblib(filepath)
        
        if isinstance(model_data, dict) and 'model' in model_data:
            self.best_model_ = model_data['model']
            
            # Restore other attributes if available
            if 'cv_results' in model_data:
                self.results_ = model_data['cv_results']
            if 'best_score' in model_data:
                self.best_score_ = model_data['best_score']
            if 'best_params' in model_data:
                self.best_params_ = model_data['best_params']
                
            if self.verbose:
                logger.info(f"Model loaded from: {filepath}")
                
            return self.best_model_
        else:
            # Handle legacy format (direct model object)
            self.best_model_ = model_data
            return self.best_model_
    
    def get_feature_importance(self, 
                              model: Optional[BaseEstimator] = None,
                              feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get feature importance for the trained model.
        
        Args:
            model: Model to get importance from (uses best model if None)
            feature_names: Names of features (uses indices if None)
            
        Returns:
            DataFrame with feature importance
        """
        if model is None:
            if self.best_model_ is None:
                raise ValueError("No model available. Train a model first.")
            model = self.best_model_
        
        # Extract feature importance based on model type
        importance = None
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models (RandomForest, XGBoost)
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models (LogisticRegression, SVM with linear kernel)
            importance = np.abs(model.coef_).flatten()
        else:
            # For models without direct feature importance
            logger.warning(f"Feature importance not available for {type(model).__name__}")
            return pd.DataFrame()
        
        # Create DataFrame
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of training results.
        
        Returns:
            Dictionary with model performance summary
        """
        if not hasattr(self, 'results_') or not self.results_:
            return {"error": "No training results available"}
        
        summary = {
            'best_model': type(self.best_model_).__name__ if self.best_model_ else None,
            'best_score': self.best_score_,
            'best_params': self.best_params_,
            'training_config': {
                'cv_folds': self.cv_folds,
                'scoring': self.scoring,
                'handle_imbalance': self.handle_imbalance,
                'search_method': 'BayesianSearchCV' if self.use_bayesian_search else 'GridSearchCV'
            },
            'all_models': {}
        }
        
        # Add results for all trained models
        for model_name, results in self.results_.items():
            summary['all_models'][model_name] = {
                'score': results['best_score'],
                'test_f1_mean': results.get('test_f1_mean', None),
                'test_accuracy_mean': results.get('test_accuracy_mean', None),
                'fit_time_mean': results.get('fit_time_mean', None)
            }
        
        return summary
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Train all available models and return results.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with results for each model
        """
        return self.select_best(X, y)[1]  # Return the results dictionary
    
    def get_best_model(self) -> Tuple[BaseEstimator, str]:
        """
        Get the best trained model and its name.
        
        Returns:
            Tuple of (best_model, best_model_name)
        """
        if self.best_model_ is None:
            raise ValueError("No model has been trained yet. Call train_all_models() first.")
        
        # Find the best model name by looking at results
        best_model_name = None
        if hasattr(self, 'results_') and self.results_:
            best_model_name = max(
                self.results_.keys(),
                key=lambda name: self.results_[name]['best_score']
            )
        else:
            best_model_name = "unknown_model"
        
        return self.best_model_, best_model_name
