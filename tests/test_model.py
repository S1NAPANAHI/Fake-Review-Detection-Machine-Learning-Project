"""
Unit tests for the machine learning model training and selection module.

This module tests model training functionality including multiple algorithm support,
hyperparameter optimization, class imbalance handling, cross-validation,
model selection, and model persistence.
Uses small synthetic fixtures and validates model performance metrics.
"""

import unittest
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modeling import ModelTrainer


class TestModelTrainerInitialization(unittest.TestCase):
    """Test ModelTrainer initialization and configuration."""
    
    def test_default_initialization(self):
        """Test ModelTrainer initialization with default parameters."""
        trainer = ModelTrainer()
        
        # Verify default settings
        self.assertEqual(trainer.cv_folds, 5)
        self.assertEqual(trainer.scoring, 'f1')
        self.assertTrue(trainer.handle_imbalance)
        self.assertEqual(trainer.n_jobs, -1)
        self.assertTrue(trainer.verbose)
        self.assertIsNone(trainer.best_model_)
        self.assertEqual(len(trainer.results_), 0)
    
    def test_custom_initialization(self):
        """Test ModelTrainer initialization with custom parameters."""
        trainer = ModelTrainer(
            random_state=123,
            cv_folds=3,
            scoring='accuracy',
            handle_imbalance=False,
            use_bayesian_search=False,
            n_iter=25,
            verbose=False
        )
        
        # Verify custom settings
        self.assertEqual(trainer.random_state, 123)
        self.assertEqual(trainer.cv_folds, 3)
        self.assertEqual(trainer.scoring, 'accuracy')
        self.assertFalse(trainer.handle_imbalance)
        self.assertFalse(trainer.use_bayesian_search)
        self.assertEqual(trainer.n_iter, 25)
        self.assertFalse(trainer.verbose)
    
    def test_model_initialization(self):
        """Test initialization of base models."""
        trainer = ModelTrainer(random_state=42)
        
        # Should have initialized models
        expected_models = ['logistic_regression', 'random_forest', 'svm', 'xgboost']
        for model_name in expected_models:
            self.assertIn(model_name, trainer.models)
            self.assertIsInstance(trainer.models[model_name], BaseEstimator)
        
        # All models should have consistent random state
        self.assertEqual(trainer.models['logistic_regression'].random_state, 42)
        self.assertEqual(trainer.models['random_forest'].random_state, 42)


class TestHyperparameterGrids(unittest.TestCase):
    """Test hyperparameter grid initialization."""
    
    def test_bayesian_search_spaces(self):
        """Test Bayesian search space initialization."""
        with patch('src.modeling.BAYESIAN_AVAILABLE', True):
            trainer = ModelTrainer(use_bayesian_search=True)
            
            # Should have Bayesian spaces
            self.assertTrue(hasattr(trainer, 'param_spaces'))
            self.assertIn('logistic_regression', trainer.param_spaces)
            self.assertIn('random_forest', trainer.param_spaces)
            self.assertIn('svm', trainer.param_spaces)
            self.assertIn('xgboost', trainer.param_spaces)
    
    def test_grid_search_spaces(self):
        """Test grid search space initialization."""
        trainer = ModelTrainer(use_bayesian_search=False)
        
        # Should have grid spaces
        self.assertTrue(hasattr(trainer, 'param_grids'))
        self.assertIn('logistic_regression', trainer.param_grids)
        self.assertIn('random_forest', trainer.param_grids)
        
        # Grid should contain lists of values
        lr_grid = trainer.param_grids['logistic_regression']
        self.assertIsInstance(lr_grid['C'], list)
        self.assertTrue(len(lr_grid['C']) > 1)
    
    def test_bayesian_fallback(self):
        """Test fallback to grid search when Bayesian not available."""
        with patch('src.modeling.BAYESIAN_AVAILABLE', False):
            trainer = ModelTrainer(use_bayesian_search=True)
            
            # Should fall back to grid search
            self.assertFalse(trainer.use_bayesian_search)
            self.assertTrue(hasattr(trainer, 'param_grids'))


class TestClassImbalanceHandling(unittest.TestCase):
    """Test class imbalance handling functionality."""
    
    def setUp(self):
        self.trainer = ModelTrainer(handle_imbalance=True)
        
        # Create imbalanced dataset
        self.imbalanced_y = np.array([0] * 80 + [1] * 20)  # 80-20 split
        self.balanced_y = np.array([0] * 50 + [1] * 50)   # 50-50 split
    
    def test_compute_class_weights_imbalanced(self):
        """Test class weight computation for imbalanced data."""
        weights = self.trainer._compute_class_weights(self.imbalanced_y)
        
        # Should return dictionary with class weights
        self.assertIsInstance(weights, dict)
        self.assertIn(0, weights)
        self.assertIn(1, weights)
        
        # Minority class should have higher weight
        self.assertGreater(weights[1], weights[0])
    
    def test_compute_class_weights_balanced(self):
        """Test class weight computation for balanced data."""
        weights = self.trainer._compute_class_weights(self.balanced_y)
        
        # Weights should be roughly equal for balanced data
        self.assertAlmostEqual(weights[0], weights[1], places=1)
    
    def test_compute_class_weights_disabled(self):
        """Test class weights when imbalance handling is disabled."""
        trainer = ModelTrainer(handle_imbalance=False)
        weights = trainer._compute_class_weights(self.imbalanced_y)
        
        # Should return None when disabled
        self.assertIsNone(weights)


class TestScoringMetrics(unittest.TestCase):
    """Test scoring metrics configuration."""
    
    def setUp(self):
        self.trainer = ModelTrainer()
    
    def test_get_scoring_metrics(self):
        """Test scoring metrics configuration."""
        metrics = self.trainer._get_scoring_metrics()
        
        # Should contain standard classification metrics
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # All metrics should be callable
        for metric_name, metric_func in metrics.items():
            self.assertTrue(callable(metric_func))


class TestSingleModelTraining(unittest.TestCase):
    """Test training individual models."""
    
    def setUp(self):
        self.trainer = ModelTrainer(random_state=42, verbose=False)
        
        # Create synthetic dataset
        np.random.seed(42)
        self.X = np.random.randn(100, 10)
        self.y = np.random.choice([0, 1], size=100, p=[0.7, 0.3])
    
    @patch('sklearn.model_selection.cross_validate')
    @patch('sklearn.model_selection.GridSearchCV.fit')
    @patch('sklearn.model_selection.GridSearchCV.best_estimator_')
    @patch('sklearn.model_selection.GridSearchCV.best_score_', 0.85)
    @patch('sklearn.model_selection.GridSearchCV.best_params_', {'C': 1.0})
    def test_train_single_model_success(self, mock_best_estimator, mock_fit, mock_cv):
        """Test successful single model training."""
        # Mock the fitted estimator
        mock_estimator = MagicMock()
        mock_best_estimator = mock_estimator
        
        # Mock cross_validate results
        mock_cv_results = {
            'test_accuracy': np.array([0.8, 0.85, 0.9]),
            'test_precision': np.array([0.75, 0.8, 0.85]),
            'test_recall': np.array([0.7, 0.75, 0.8]),
            'test_f1': np.array([0.72, 0.77, 0.82]),
            'test_roc_auc': np.array([0.82, 0.87, 0.92]),
            'train_accuracy': np.array([0.9, 0.92, 0.95]),
            'train_precision': np.array([0.85, 0.87, 0.9]),
            'train_recall': np.array([0.8, 0.82, 0.85]),
            'train_f1': np.array([0.82, 0.84, 0.87]),
            'train_roc_auc': np.array([0.92, 0.94, 0.97]),
            'fit_time': np.array([0.1, 0.12, 0.15]),
            'score_time': np.array([0.01, 0.02, 0.02])
        }
        mock_cv.return_value = mock_cv_results
        
        # Train model
        best_model, cv_summary = self.trainer._train_single_model('logistic_regression', self.X, self.y)
        
        # Verify results
        self.assertEqual(best_model, mock_estimator)
        self.assertIsInstance(cv_summary, dict)
        self.assertIn('best_score', cv_summary)
        self.assertIn('best_params', cv_summary)
        self.assertIn('test_f1_mean', cv_summary)
        self.assertEqual(cv_summary['best_score'], 0.85)
    
    def test_train_single_model_with_class_weights(self):
        """Test single model training with class weight handling."""
        trainer = ModelTrainer(handle_imbalance=True, verbose=False)
        
        # Create imbalanced dataset
        imbalanced_y = np.array([0] * 80 + [1] * 20)
        
        with patch('sklearn.model_selection.GridSearchCV') as mock_grid_search:
            mock_search_instance = MagicMock()
            mock_search_instance.best_estimator_ = MagicMock()
            mock_search_instance.best_score_ = 0.8
            mock_search_instance.best_params_ = {}
            mock_grid_search.return_value = mock_search_instance
            
            with patch('sklearn.model_selection.cross_validate') as mock_cv:
                mock_cv.return_value = {
                    'test_f1': np.array([0.7, 0.8, 0.75]),
                    'train_f1': np.array([0.8, 0.85, 0.82]),
                    'test_accuracy': np.array([0.75, 0.8, 0.78]),
                    'train_accuracy': np.array([0.85, 0.9, 0.87]),
                    'test_precision': np.array([0.7, 0.75, 0.72]),
                    'train_precision': np.array([0.8, 0.82, 0.78]),
                    'test_recall': np.array([0.65, 0.7, 0.68]),
                    'train_recall': np.array([0.75, 0.78, 0.72]),
                    'test_roc_auc': np.array([0.8, 0.85, 0.82]),
                    'train_roc_auc': np.array([0.9, 0.92, 0.88]),
                    'fit_time': np.array([0.1, 0.12, 0.11]),
                    'score_time': np.array([0.01, 0.02, 0.01])
                }
                
                # Should train successfully with class weights
                best_model, cv_summary = trainer._train_single_model('logistic_regression', self.X, imbalanced_y)
                
                self.assertIsNotNone(best_model)
                self.assertIsInstance(cv_summary, dict)


class TestModelSelection(unittest.TestCase):
    """Test model selection and comparison."""
    
    def setUp(self):
        self.trainer = ModelTrainer(random_state=42, verbose=False)
        
        # Create synthetic dataset
        np.random.seed(42)
        self.X = np.random.randn(50, 5)  # Smaller dataset for faster testing
        self.y = np.random.choice([0, 1], size=50, p=[0.6, 0.4])
    
    @patch('src.modeling.ModelTrainer._train_single_model')
    def test_select_best_model(self, mock_train_single):
        """Test model selection process."""
        # Mock different model performance
        mock_results = {
            'logistic_regression': (MagicMock(), {'best_score': 0.85, 'best_params': {'C': 1.0}}),
            'random_forest': (MagicMock(), {'best_score': 0.90, 'best_params': {'n_estimators': 100}}),
            'svm': (MagicMock(), {'best_score': 0.82, 'best_params': {'C': 10.0}}),
            'xgboost': (MagicMock(), {'best_score': 0.88, 'best_params': {'n_estimators': 50}})
        }
        
        # Configure mock to return different results for each model
        def side_effect(model_name, X, y):
            return mock_results[model_name]
        
        mock_train_single.side_effect = side_effect
        
        # Run model selection
        best_model, all_results = self.trainer.select_best(self.X, self.y)
        
        # Best model should be random forest (highest score: 0.90)
        self.assertEqual(self.trainer.best_score_, 0.90)
        self.assertEqual(self.trainer.best_params_, {'n_estimators': 100})
        self.assertIsNotNone(self.trainer.best_model_)
        
        # All models should have been trained
        self.assertEqual(mock_train_single.call_count, 4)
        self.assertEqual(len(all_results), 4)
    
    @patch('src.modeling.ModelTrainer._train_single_model')
    def test_select_best_with_failure(self, mock_train_single):
        """Test model selection when some models fail."""
        # Mock one model failing
        def side_effect(model_name, X, y):
            if model_name == 'svm':
                raise Exception("SVM training failed")
            return MagicMock(), {'best_score': 0.80, 'best_params': {}}
        
        mock_train_single.side_effect = side_effect
        
        # Should handle failure gracefully
        best_model, all_results = self.trainer.select_best(self.X, self.y)
        
        # Should still have results from other models
        self.assertIsNotNone(best_model)
        self.assertGreater(len(all_results), 0)
        self.assertNotIn('svm', all_results)
    
    def test_select_best_all_fail(self):
        """Test model selection when all models fail."""
        with patch('src.modeling.ModelTrainer._train_single_model', side_effect=Exception("Training failed")):
            with self.assertRaises(ValueError) as context:
                self.trainer.select_best(self.X, self.y)
            
            self.assertIn("No models were successfully trained", str(context.exception))
    
    def test_select_best_input_validation(self):
        """Test input validation for model selection."""
        # Test with insufficient data
        tiny_X = np.random.randn(2, 5)
        tiny_y = np.array([0, 1])
        
        with self.assertRaises(ValueError):
            self.trainer.select_best(tiny_X, tiny_y)


class TestModelPersistence(unittest.TestCase):
    """Test model saving and loading functionality."""
    
    def setUp(self):
        self.trainer = ModelTrainer(random_state=42)
        self.test_dir = tempfile.mkdtemp()
        
        # Mock a trained model
        self.mock_model = MagicMock()
        self.trainer.best_model_ = self.mock_model
        self.trainer.best_score_ = 0.85
        self.trainer.best_params_ = {'C': 1.0}
        self.trainer.results_ = {'logistic_regression': {'best_score': 0.85}}
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('src.modeling.utils.save_joblib')
    @patch('src.modeling.utils.resolve_path')
    @patch('src.modeling.utils.ensure_dir')
    @patch('src.modeling.utils.get_timestamp')
    def test_save_model_default_path(self, mock_timestamp, mock_ensure_dir, mock_resolve_path, mock_save_joblib):
        """Test saving model with default path generation."""
        # Mock utilities
        mock_timestamp.return_value = "20231201_120000"
        mock_resolve_path.side_effect = lambda x: Path(self.test_dir) / x if isinstance(x, str) else x
        mock_ensure_dir.return_value = Path(self.test_dir)
        mock_save_joblib.return_value = Path(self.test_dir) / "best_model_20231201_120000.joblib"
        
        # Save model
        saved_path = self.trainer.save_model()
        
        # Verify save was called with correct data
        mock_save_joblib.assert_called_once()
        call_args = mock_save_joblib.call_args[0]
        model_data = call_args[0]
        
        self.assertIn('model', model_data)
        self.assertIn('trainer_config', model_data)
        self.assertIn('cv_results', model_data)
        self.assertEqual(model_data['model'], self.mock_model)
    
    @patch('src.modeling.utils.save_joblib')
    def test_save_model_custom_path(self, mock_save_joblib):
        """Test saving model with custom path."""
        custom_path = Path(self.test_dir) / "custom_model.joblib"
        mock_save_joblib.return_value = custom_path
        
        saved_path = self.trainer.save_model(filepath=custom_path)
        
        mock_save_joblib.assert_called_once()
        self.assertEqual(saved_path, custom_path)
    
    def test_save_model_no_model(self):
        """Test saving when no model is trained."""
        trainer = ModelTrainer()
        
        with self.assertRaises(ValueError) as context:
            trainer.save_model()
        
        self.assertIn("No model to save", str(context.exception))
    
    @patch('src.modeling.utils.load_joblib')
    def test_load_model_success(self, mock_load_joblib):
        """Test successful model loading."""
        # Mock loaded data
        mock_model_data = {
            'model': self.mock_model,
            'cv_results': {'logistic_regression': {'best_score': 0.85}},
            'best_score': 0.85,
            'best_params': {'C': 1.0}
        }
        mock_load_joblib.return_value = mock_model_data
        
        loaded_model = self.trainer.load_model("test_model.joblib")
        
        # Verify model and attributes were restored
        self.assertEqual(loaded_model, self.mock_model)
        self.assertEqual(self.trainer.best_model_, self.mock_model)
        self.assertEqual(self.trainer.best_score_, 0.85)
        self.assertEqual(self.trainer.best_params_, {'C': 1.0})
    
    @patch('src.modeling.utils.load_joblib')
    def test_load_model_legacy_format(self, mock_load_joblib):
        """Test loading model in legacy format (direct model object)."""
        # Mock legacy format (direct model object)
        mock_load_joblib.return_value = self.mock_model
        
        loaded_model = self.trainer.load_model("legacy_model.joblib")
        
        # Should still load the model
        self.assertEqual(loaded_model, self.mock_model)
        self.assertEqual(self.trainer.best_model_, self.mock_model)


class TestFeatureImportance(unittest.TestCase):
    """Test feature importance extraction."""
    
    def setUp(self):
        self.trainer = ModelTrainer()
    
    def test_feature_importance_tree_model(self):
        """Test feature importance for tree-based models."""
        # Mock tree-based model with feature_importances_
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.3, 0.5, 0.2])
        
        importance_df = self.trainer.get_feature_importance(
            mock_model, 
            feature_names=['feature1', 'feature2', 'feature3']
        )
        
        # Verify DataFrame structure
        self.assertIsInstance(importance_df, pd.DataFrame)
        self.assertEqual(len(importance_df), 3)
        self.assertIn('feature', importance_df.columns)
        self.assertIn('importance', importance_df.columns)
        
        # Should be sorted by importance (descending)
        self.assertTrue((importance_df['importance'].diff().dropna() <= 0).all())
        
        # Highest importance should be feature2
        self.assertEqual(importance_df.iloc[0]['feature'], 'feature2')
    
    def test_feature_importance_linear_model(self):
        """Test feature importance for linear models."""
        # Mock linear model with coef_
        mock_model = MagicMock()
        mock_model.coef_ = np.array([[0.1, -0.3, 0.2]])  # 2D array for classifier
        mock_model.feature_importances_ = None
        
        # Remove feature_importances_ attribute
        del mock_model.feature_importances_
        
        importance_df = self.trainer.get_feature_importance(mock_model, ['f1', 'f2', 'f3'])
        
        # Should use absolute values of coefficients
        self.assertEqual(len(importance_df), 3)
        # Highest importance should be f2 (abs(-0.3) = 0.3)
        self.assertEqual(importance_df.iloc[0]['feature'], 'f2')
    
    def test_feature_importance_no_support(self):
        """Test feature importance for models without support."""
        # Mock model without feature_importances_ or coef_
        mock_model = MagicMock()
        del mock_model.feature_importances_
        del mock_model.coef_
        
        importance_df = self.trainer.get_feature_importance(mock_model)
        
        # Should return empty DataFrame
        self.assertIsInstance(importance_df, pd.DataFrame)
        self.assertEqual(len(importance_df), 0)
    
    def test_feature_importance_default_names(self):
        """Test feature importance with default feature names."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.4, 0.6])
        
        importance_df = self.trainer.get_feature_importance(mock_model)
        
        # Should use default names
        expected_names = ['feature_0', 'feature_1']
        actual_names = importance_df['feature'].tolist()
        for expected in expected_names:
            self.assertIn(expected, actual_names)
    
    def test_feature_importance_no_model(self):
        """Test feature importance when no model is available."""
        trainer = ModelTrainer()
        
        with self.assertRaises(ValueError) as context:
            trainer.get_feature_importance()
        
        self.assertIn("No model available", str(context.exception))


class TestModelSummary(unittest.TestCase):
    """Test model training summary functionality."""
    
    def setUp(self):
        self.trainer = ModelTrainer()
        
        # Mock training results
        self.trainer.best_model_ = MagicMock()
        self.trainer.best_score_ = 0.90
        self.trainer.best_params_ = {'n_estimators': 100}
        self.trainer.results_ = {
            'logistic_regression': {
                'best_score': 0.85,
                'test_f1_mean': 0.83,
                'test_accuracy_mean': 0.86,
                'fit_time_mean': 0.15
            },
            'random_forest': {
                'best_score': 0.90,
                'test_f1_mean': 0.88,
                'test_accuracy_mean': 0.91,
                'fit_time_mean': 0.45
            }
        }
    
    def test_get_model_summary(self):
        """Test getting comprehensive model summary."""
        summary = self.trainer.get_model_summary()
        
        # Verify summary structure
        self.assertIsInstance(summary, dict)
        self.assertIn('best_model', summary)
        self.assertIn('best_score', summary)
        self.assertIn('best_params', summary)
        self.assertIn('training_config', summary)
        self.assertIn('all_models', summary)
        
        # Verify values
        self.assertEqual(summary['best_score'], 0.90)
        self.assertEqual(summary['best_params'], {'n_estimators': 100})
        
        # Verify all models are included
        self.assertIn('logistic_regression', summary['all_models'])
        self.assertIn('random_forest', summary['all_models'])
        
        # Verify model details
        lr_details = summary['all_models']['logistic_regression']
        self.assertEqual(lr_details['score'], 0.85)
        self.assertEqual(lr_details['test_f1_mean'], 0.83)
    
    def test_get_model_summary_no_results(self):
        """Test getting summary when no training results exist."""
        trainer = ModelTrainer()
        
        summary = trainer.get_model_summary()
        
        self.assertIn('error', summary)
        self.assertEqual(summary['error'], 'No training results available')
    
    def test_get_best_model(self):
        """Test getting best trained model."""
        best_model, best_name = self.trainer.get_best_model()
        
        self.assertEqual(best_model, self.trainer.best_model_)
        self.assertEqual(best_name, 'random_forest')  # Highest score
    
    def test_get_best_model_no_training(self):
        """Test getting best model when no training has occurred."""
        trainer = ModelTrainer()
        
        with self.assertRaises(ValueError) as context:
            trainer.get_best_model()
        
        self.assertIn("No model has been trained", str(context.exception))


class TestTrainAllModels(unittest.TestCase):
    """Test training all models functionality."""
    
    def setUp(self):
        self.trainer = ModelTrainer(verbose=False)
        
        # Create small synthetic dataset
        np.random.seed(42)
        self.X = np.random.randn(30, 3)
        self.y = np.random.choice([0, 1], size=30)
    
    @patch('src.modeling.ModelTrainer.select_best')
    def test_train_all_models(self, mock_select_best):
        """Test training all available models."""
        # Mock select_best return value
        mock_results = {
            'logistic_regression': {'best_score': 0.80},
            'random_forest': {'best_score': 0.85},
            'svm': {'best_score': 0.78}
        }
        mock_select_best.return_value = (MagicMock(), mock_results)
        
        results = self.trainer.train_all_models(self.X, self.y)
        
        # Should return results dictionary
        self.assertEqual(results, mock_results)
        mock_select_best.assert_called_once_with(self.X, self.y)


class TestModelTrainerEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_dataset(self):
        """Test training with empty dataset."""
        trainer = ModelTrainer(verbose=False)
        
        empty_X = np.array([]).reshape(0, 5)
        empty_y = np.array([])
        
        with self.assertRaises(ValueError):
            trainer.select_best(empty_X, empty_y)
    
    def test_single_class_dataset(self):
        """Test training with single class dataset."""
        trainer = ModelTrainer(verbose=False)
        
        X = np.random.randn(20, 5)
        y = np.ones(20)  # All same class
        
        # Should handle single class gracefully (though may not train well)
        with patch('src.modeling.ModelTrainer._train_single_model') as mock_train:
            mock_train.return_value = (MagicMock(), {'best_score': 0.5, 'best_params': {}})
            
            best_model, results = trainer.select_best(X, y)
            self.assertIsNotNone(best_model)
    
    def test_more_features_than_samples(self):
        """Test training with more features than samples."""
        trainer = ModelTrainer(verbose=False)
        
        # More features than samples
        X = np.random.randn(10, 50)
        y = np.random.choice([0, 1], size=10)
        
        # Should still attempt training (may adjust internally)
        with patch('src.modeling.ModelTrainer._train_single_model') as mock_train:
            mock_train.return_value = (MagicMock(), {'best_score': 0.6, 'best_params': {}})
            
            best_model, results = trainer.select_best(X, y)
            self.assertIsNotNone(best_model)
    
    @patch('src.modeling.utils.metrics.set_gauge')
    @patch('src.modeling.utils.metrics.increment_counter')
    def test_metrics_recording(self, mock_counter, mock_gauge):
        """Test that metrics are recorded during training."""
        trainer = ModelTrainer(verbose=False)
        
        with patch('src.modeling.ModelTrainer._train_single_model') as mock_train:
            mock_train.return_value = (MagicMock(), {'best_score': 0.85, 'best_params': {}})
            
            X = np.random.randn(20, 5)
            y = np.random.choice([0, 1], size=20)
            
            trainer.select_best(X, y)
            
            # Should record metrics
            mock_gauge.assert_called_with('best_model_score', 0.85)
            mock_counter.assert_called_with('model_selection_completed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
