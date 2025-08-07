"""
Unit tests for the feature engineering module.

This module tests feature engineering functionality including TF-IDF vectorization,
behavioral feature extraction, graph feature generation, sentiment analysis,
and comprehensive feature matrix construction.
Uses small synthetic fixtures and validates feature shapes and types.
"""

import unittest
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from scipy import sparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import (
    FeatureEngineer, FeatureEngineeringError
)


class TestFeatureEngineerInitialization(unittest.TestCase):
    """Test FeatureEngineer initialization and configuration."""
    
    def test_default_initialization(self):
        """Test FeatureEngineer initialization with default parameters."""
        engineer = FeatureEngineer()
        
        # Verify default settings
        self.assertEqual(engineer.text_column, 'review_text')
        self.assertEqual(engineer.tfidf_max_features, 5000)
        self.assertEqual(engineer.tfidf_ngram_range, (1, 3))
        self.assertEqual(engineer.node2vec_dimensions, 64)
        self.assertTrue(engineer.enable_sentiment)
        self.assertFalse(engineer.is_fitted)
    
    def test_custom_initialization(self):
        """Test FeatureEngineer initialization with custom parameters."""
        engineer = FeatureEngineer(
            text_column='custom_text',
            tfidf_max_features=1000,
            tfidf_ngram_range=(1, 2),
            enable_graph_features=False,
            enable_sentiment=False,
            feature_scaling='minmax'
        )
        
        # Verify custom settings
        self.assertEqual(engineer.text_column, 'custom_text')
        self.assertEqual(engineer.tfidf_max_features, 1000)
        self.assertEqual(engineer.tfidf_ngram_range, (1, 2))
        self.assertFalse(engineer.enable_graph_features)
        self.assertFalse(engineer.enable_sentiment)
        self.assertEqual(engineer.feature_scaling, 'minmax')


class TestTextFeatureEngineering(unittest.TestCase):
    """Test TF-IDF text feature engineering."""
    
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'review_text': [
                'This product is amazing and high quality',
                'Terrible quality, worst purchase ever',
                'Good value for money, decent product',
                'Excellent service and fast shipping',
                'Poor customer support, disappointed'
            ],
            'rating': [5, 1, 4, 5, 2],
            'user_id': ['user1', 'user2', 'user3', 'user4', 'user5'],
            'product_id': ['prod1', 'prod1', 'prod2', 'prod3', 'prod1']
        })
        
        self.engineer = FeatureEngineer(
            tfidf_max_features=100,  # Small for testing
            enable_graph_features=False,  # Disable for simpler testing
            enable_sentiment=False
        )
    
    def test_fit_text_features(self):
        """Test fitting TF-IDF vectorizer."""
        self.engineer.fit(self.sample_data)
        
        # Should have fitted TF-IDF vectorizer
        self.assertTrue(hasattr(self.engineer.tfidf_vectorizer, 'vocabulary_'))
        self.assertIsInstance(self.engineer.tfidf_vectorizer.vocabulary_, dict)
        self.assertTrue(len(self.engineer.tfidf_vectorizer.vocabulary_) > 0)
    
    def test_engineer_text_features(self):
        """Test text feature engineering with TF-IDF."""
        self.engineer.fit(self.sample_data)
        
        tfidf_matrix, feature_names = self.engineer.engineer_text_features(self.sample_data)
        
        # Verify output structure
        self.assertTrue(sparse.issparse(tfidf_matrix))
        self.assertEqual(tfidf_matrix.shape[0], len(self.sample_data))
        self.assertIsInstance(feature_names, list)
        self.assertTrue(len(feature_names) > 0)
        
        # Feature names should have tfidf_ prefix
        self.assertTrue(all(name.startswith('tfidf_') for name in feature_names))
    
    def test_text_features_missing_column(self):
        """Test text feature engineering with missing text column."""
        data_no_text = pd.DataFrame({'rating': [1, 2, 3]})
        self.engineer.fit(data_no_text)
        
        tfidf_matrix, feature_names = self.engineer.engineer_text_features(data_no_text)
        
        # Should return empty matrix
        self.assertEqual(tfidf_matrix.shape[1], 0)
        self.assertEqual(len(feature_names), 0)
    
    def test_tfidf_parameters(self):
        """Test TF-IDF vectorizer parameters."""
        engineer = FeatureEngineer(
            tfidf_max_features=50,
            tfidf_ngram_range=(1, 2),
            tfidf_min_df=1,
            tfidf_max_df=0.9
        )
        engineer.fit(self.sample_data)
        
        # Verify TF-IDF parameters
        vectorizer = engineer.tfidf_vectorizer
        self.assertEqual(vectorizer.max_features, 50)
        self.assertEqual(vectorizer.ngram_range, (1, 2))
        self.assertEqual(vectorizer.min_df, 1)
        self.assertEqual(vectorizer.max_df, 0.9)


class TestBehavioralFeatureEngineering(unittest.TestCase):
    """Test behavioral feature engineering."""
    
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'review_text': [
                'AMAZING PRODUCT!!! Best purchase ever!!!',
                'terrible quality, not worth it',
                'Good product, works as expected.',
                'Excellent value! Highly recommend!!!',
                'Poor service. Very disappointed.'
            ],
            'rating': [5, 1, 4, 5, 2],
            'user_id': ['user1', 'user2', 'user3', 'user1', 'user4'],
            'product_id': ['prod1', 'prod1', 'prod2', 'prod1', 'prod3'],
            'date': [
                '2023-01-01 10:00:00',
                '2023-01-02 15:30:00',
                '2023-01-03 09:15:00',
                '2023-01-04 14:20:00',
                '2023-01-05 16:45:00'
            ]
        })
        
        self.engineer = FeatureEngineer(
            enable_graph_features=False,
            enable_sentiment=False
        )
        self.engineer.is_fitted = True
    
    def test_basic_text_statistics(self):
        """Test basic text-based behavioral features."""
        behavioral_matrix, feature_names = self.engineer.engineer_behavioral_features(self.sample_data)
        
        # Verify output structure
        self.assertTrue(sparse.issparse(behavioral_matrix))
        self.assertEqual(behavioral_matrix.shape[0], len(self.sample_data))
        self.assertIsInstance(feature_names, list)
        
        # Should have basic text features
        expected_features = [
            'review_length', 'word_count', 'avg_word_length',
            'uppercase_ratio', 'digit_ratio', 'punctuation_ratio',
            'exclamation_count', 'question_count'
        ]
        
        for expected in expected_features:
            self.assertTrue(any(expected in name for name in feature_names))
    
    def test_user_behavioral_features(self):
        """Test user-level behavioral features."""
        behavioral_matrix, feature_names = self.engineer.engineer_behavioral_features(self.sample_data)
        
        # Should have user-level features
        user_features = [name for name in feature_names if 'user_' in name]
        self.assertTrue(len(user_features) > 0)
    
    def test_product_behavioral_features(self):
        """Test product-level behavioral features."""
        behavioral_matrix, feature_names = self.engineer.engineer_behavioral_features(self.sample_data)
        
        # Should have product-level features
        product_features = [name for name in feature_names if 'product_' in name]
        self.assertTrue(len(product_features) > 0)
    
    def test_temporal_behavioral_features(self):
        """Test temporal behavioral features."""
        behavioral_matrix, feature_names = self.engineer.engineer_behavioral_features(self.sample_data)
        
        # Should have temporal features
        temporal_features = [name for name in feature_names if any(t in name for t in ['hour', 'day', 'weekend', 'epoch'])]
        self.assertTrue(len(temporal_features) > 0)
    
    def test_rating_behavioral_features(self):
        """Test rating-based behavioral features."""
        behavioral_matrix, feature_names = self.engineer.engineer_behavioral_features(self.sample_data)
        
        # Should have rating features
        rating_features = [name for name in feature_names if 'rating' in name or 'extreme' in name]
        self.assertTrue(len(rating_features) > 0)
    
    def test_compute_avg_word_length(self):
        """Test average word length computation."""
        # Test normal text
        self.assertAlmostEqual(self.engineer._compute_avg_word_length("hello world"), 5.0)
        
        # Test single word
        self.assertAlmostEqual(self.engineer._compute_avg_word_length("test"), 4.0)
        
        # Test empty text
        self.assertEqual(self.engineer._compute_avg_word_length(""), 0.0)
        
        # Test mixed length words
        self.assertAlmostEqual(self.engineer._compute_avg_word_length("a bb ccc"), 2.0)


class TestGraphFeatureEngineering(unittest.TestCase):
    """Test graph-based feature engineering."""
    
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'review_text': ['Review 1', 'Review 2', 'Review 3', 'Review 4'],
            'user_id': ['user1', 'user2', 'user1', 'user3'],
            'product_id': ['prod1', 'prod1', 'prod2', 'prod1'],
            'rating': [5, 3, 4, 2]
        })
    
    @patch('src.feature_engineering.NETWORKX_AVAILABLE', True)
    @patch('src.feature_engineering.nx.Graph')
    def test_build_user_product_graph(self, mock_graph):
        """Test building user-product bipartite graph."""
        # Mock graph instance
        mock_graph_instance = MagicMock()
        mock_graph.return_value = mock_graph_instance
        
        engineer = FeatureEngineer(enable_graph_features=True)
        graph = engineer._build_user_product_graph(self.sample_data)
        
        # Should have called graph methods
        mock_graph_instance.add_node.assert_called()
        mock_graph_instance.add_edge.assert_called()
    
    @patch('src.feature_engineering.NETWORKX_AVAILABLE', False)
    def test_graph_features_disabled(self):
        """Test graph features when NetworkX is not available."""
        engineer = FeatureEngineer(enable_graph_features=True)
        engineer.is_fitted = True
        
        graph_matrix, feature_names = engineer.engineer_graph_features(self.sample_data)
        
        # Should return empty matrix
        self.assertEqual(graph_matrix.shape[1], 0)
        self.assertEqual(len(feature_names), 0)
    
    @patch('src.feature_engineering.NETWORKX_AVAILABLE', True)
    def test_extract_node_features(self):
        """Test extracting features for graph nodes."""
        engineer = FeatureEngineer(enable_graph_features=True)
        
        # Mock graph with some basic structure
        mock_graph = MagicMock()
        mock_graph.__contains__.return_value = True
        mock_graph.degree.return_value = 3
        
        with patch('src.feature_engineering.nx.clustering', return_value=0.5), \
             patch('src.feature_engineering.nx.betweenness_centrality', return_value={'test_node': 0.1}):
            
            features = engineer._extract_node_features(mock_graph, 'test_node')
            
            # Should return array with 3 features (degree, clustering, centrality)
            self.assertEqual(len(features), 3)
            self.assertIsInstance(features, np.ndarray)
    
    def test_get_node2vec_embedding_missing(self):
        """Test getting node2vec embedding for missing node."""
        engineer = FeatureEngineer(node2vec_dimensions=32)
        engineer.node2vec_model = None
        
        embedding = engineer._get_node2vec_embedding('missing_node')
        
        # Should return zero vector
        self.assertEqual(len(embedding), 32)
        self.assertTrue(np.all(embedding == 0))


class TestSentimentFeatureEngineering(unittest.TestCase):
    """Test sentiment analysis feature engineering."""
    
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'review_text': [
                'This product is absolutely amazing!',
                'Terrible quality, very disappointed.',
                'Neutral product, nothing special.',
                'Love it! Great value for money.',
                'Worst purchase ever, complete waste.'
            ]
        })
        
        self.engineer = FeatureEngineer(
            enable_sentiment=True,
            enable_graph_features=False
        )
        self.engineer.is_fitted = True
    
    @patch('src.feature_engineering.TEXTBLOB_AVAILABLE', True)
    @patch('src.feature_engineering.TextBlob')
    def test_textblob_sentiment_features(self, mock_textblob):
        """Test TextBlob sentiment feature extraction."""
        # Mock TextBlob sentiment
        mock_blob = MagicMock()
        mock_blob.sentiment.polarity = 0.8
        mock_blob.sentiment.subjectivity = 0.6
        mock_textblob.return_value = mock_blob
        
        sentiment_features = self.engineer._extract_textblob_sentiment(
            pd.Series(['Great product!'])
        )
        
        # Should return array with polarity and subjectivity
        self.assertEqual(sentiment_features.shape, (1, 2))
        self.assertEqual(sentiment_features[0, 0], 0.8)  # polarity
        self.assertEqual(sentiment_features[0, 1], 0.6)  # subjectivity
    
    @patch('src.feature_engineering.VADER_AVAILABLE', True)
    def test_vader_sentiment_features(self):
        """Test VADER sentiment feature extraction."""
        # Mock VADER analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.polarity_scores.return_value = {
            'compound': 0.8,
            'pos': 0.7,
            'neu': 0.2,
            'neg': 0.1
        }
        self.engineer.vader_analyzer = mock_analyzer
        
        sentiment_features = self.engineer._extract_vader_sentiment(
            pd.Series(['Great product!'])
        )
        
        # Should return array with compound, pos, neu, neg scores
        self.assertEqual(sentiment_features.shape, (1, 4))
        self.assertEqual(sentiment_features[0, 0], 0.8)  # compound
        self.assertEqual(sentiment_features[0, 1], 0.7)  # pos
    
    def test_sentiment_features_disabled(self):
        """Test sentiment features when disabled."""
        engineer = FeatureEngineer(enable_sentiment=False)
        engineer.is_fitted = True
        
        sentiment_matrix, feature_names = engineer.extract_sentiment(self.sample_data)
        
        # Should return empty matrix
        self.assertEqual(sentiment_matrix.shape[1], 0)
        self.assertEqual(len(feature_names), 0)
    
    @patch('src.feature_engineering.TEXTBLOB_AVAILABLE', False)
    @patch('src.feature_engineering.VADER_AVAILABLE', False)
    def test_sentiment_features_no_libraries(self):
        """Test sentiment features when libraries are unavailable."""
        engineer = FeatureEngineer(enable_sentiment=True)
        engineer.is_fitted = True
        
        sentiment_matrix, feature_names = engineer.extract_sentiment(self.sample_data)
        
        # Should return empty matrix
        self.assertEqual(sentiment_matrix.shape[1], 0)
        self.assertEqual(len(feature_names), 0)


class TestFeatureScaling(unittest.TestCase):
    """Test feature scaling functionality."""
    
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'review_text': ['Test review one', 'Test review two'],
            'rating': [5, 1]
        })
    
    def test_standard_scaling(self):
        """Test standard scaling of features."""
        engineer = FeatureEngineer(
            feature_scaling='standard',
            enable_graph_features=False,
            enable_sentiment=False
        )
        
        engineer.fit(self.sample_data)
        
        # Scaler should be StandardScaler
        from sklearn.preprocessing import StandardScaler
        self.assertIsInstance(engineer.scaler, StandardScaler)
    
    def test_minmax_scaling(self):
        """Test MinMax scaling of features."""
        engineer = FeatureEngineer(
            feature_scaling='minmax',
            enable_graph_features=False,
            enable_sentiment=False
        )
        
        engineer.fit(self.sample_data)
        
        # Scaler should be MinMaxScaler
        from sklearn.preprocessing import MinMaxScaler
        self.assertIsInstance(engineer.scaler, MinMaxScaler)
    
    def test_no_scaling(self):
        """Test disabled feature scaling."""
        engineer = FeatureEngineer(
            feature_scaling='none',
            enable_graph_features=False,
            enable_sentiment=False
        )
        
        engineer.fit(self.sample_data)
        
        # Scaler should be None
        self.assertIsNone(engineer.scaler)


class TestComprehensiveFeatureEngineering(unittest.TestCase):
    """Test comprehensive feature engineering pipeline."""
    
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'review_text': [
                'This product is absolutely fantastic! Highly recommend!',
                'Poor quality and terrible customer service.',
                'Good value for the price, works well.',
                'Amazing product! Will definitely buy again!',
                'Not impressed, would not purchase again.',
                'Excellent service and fast delivery.',
                'Worst product ever, complete waste of money.',
                'Great quality, very satisfied with purchase.'
            ],
            'rating': [5, 1, 4, 5, 2, 5, 1, 4],
            'user_id': ['user1', 'user2', 'user3', 'user1', 'user4', 'user2', 'user5', 'user3'],
            'product_id': ['prod1', 'prod1', 'prod2', 'prod3', 'prod1', 'prod2', 'prod4', 'prod1'],
            'date': [
                '2023-01-01 10:00:00', '2023-01-02 15:30:00',
                '2023-01-03 09:15:00', '2023-01-04 14:20:00',
                '2023-01-05 16:45:00', '2023-01-06 11:30:00',
                '2023-01-07 13:15:00', '2023-01-08 08:45:00'
            ]
        })
    
    def test_fit_transform_pipeline(self):
        """Test complete fit-transform pipeline."""
        engineer = FeatureEngineer(
            tfidf_max_features=100,  # Small for testing
            enable_graph_features=False,  # Disable for simpler testing
            enable_sentiment=False,
            feature_scaling='standard'
        )
        
        # Fit and transform
        feature_matrix, feature_names = engineer.fit_transform(self.sample_data)
        
        # Verify output structure
        self.assertTrue(sparse.issparse(feature_matrix))
        self.assertEqual(feature_matrix.shape[0], len(self.sample_data))
        self.assertIsInstance(feature_names, list)
        self.assertTrue(len(feature_names) > 0)
        
        # Should be fitted
        self.assertTrue(engineer.is_fitted)
        
        # Feature matrix should have reasonable number of features
        self.assertGreater(feature_matrix.shape[1], 10)
    
    def test_transform_only_after_fit(self):
        """Test transform-only after fitting."""
        engineer = FeatureEngineer(
            enable_graph_features=False,
            enable_sentiment=False
        )
        
        # Fit first
        engineer.fit(self.sample_data)
        
        # Then transform
        feature_matrix, feature_names = engineer.transform(self.sample_data)
        
        self.assertTrue(engineer.is_fitted)
        self.assertTrue(sparse.issparse(feature_matrix))
    
    def test_transform_without_fit(self):
        """Test transform automatically fits if not fitted."""
        engineer = FeatureEngineer(
            enable_graph_features=False,
            enable_sentiment=False
        )
        
        # Transform without explicit fit
        feature_matrix, feature_names = engineer.transform(self.sample_data)
        
        # Should automatically fit and then transform
        self.assertTrue(engineer.is_fitted)
        self.assertTrue(sparse.issparse(feature_matrix))
    
    def test_get_feature_names(self):
        """Test getting all feature names."""
        engineer = FeatureEngineer(
            enable_graph_features=False,
            enable_sentiment=False
        )
        
        engineer.fit_transform(self.sample_data)
        
        feature_names = engineer.get_feature_names()
        
        self.assertIsInstance(feature_names, list)
        self.assertTrue(len(feature_names) > 0)
        self.assertEqual(len(feature_names), len(engineer.feature_names_))
    
    def test_get_feature_names_by_type(self):
        """Test getting feature names grouped by type."""
        engineer = FeatureEngineer(
            tfidf_max_features=50,
            enable_graph_features=False,
            enable_sentiment=False
        )
        
        engineer.fit_transform(self.sample_data)
        
        feature_types = engineer.get_feature_names_by_type()
        
        self.assertIsInstance(feature_types, dict)
        self.assertIn('text', feature_types)
        self.assertIn('behavioral', feature_types)
        self.assertIn('graph', feature_types)
        self.assertIn('sentiment', feature_types)
        
        # Text features should be present
        self.assertTrue(len(feature_types['text']) > 0)
        self.assertTrue(len(feature_types['behavioral']) > 0)
    
    def test_feature_matrix_properties(self):
        """Test properties of the generated feature matrix."""
        engineer = FeatureEngineer(
            tfidf_max_features=50,
            enable_graph_features=False,
            enable_sentiment=False
        )
        
        feature_matrix, feature_names = engineer.fit_transform(self.sample_data)
        
        # Verify matrix properties
        self.assertTrue(sparse.issparse(feature_matrix))
        self.assertEqual(feature_matrix.shape[0], len(self.sample_data))
        self.assertEqual(feature_matrix.shape[1], len(feature_names))
        
        # Matrix should not be all zeros
        self.assertGreater(feature_matrix.nnz, 0)
        
        # Feature names should be unique
        self.assertEqual(len(feature_names), len(set(feature_names)))


class TestMissingValueHandling(unittest.TestCase):
    """Test missing value handling in feature engineering."""
    
    def setUp(self):
        self.data_with_missing = pd.DataFrame({
            'review_text': ['Great product!', None, '', 'Love it!'],
            'rating': [5, np.nan, 3, 4],
            'user_id': ['user1', 'user2', None, 'user4'],
            'product_id': [None, 'prod2', 'prod3', 'prod1']
        })
    
    def test_handle_missing_mean(self):
        """Test handling missing values with mean imputation."""
        engineer = FeatureEngineer(
            handle_missing='mean',
            enable_graph_features=False,
            enable_sentiment=False
        )
        
        feature_matrix, feature_names = engineer.fit_transform(self.data_with_missing)
        
        # Should process without errors
        self.assertTrue(sparse.issparse(feature_matrix))
        self.assertEqual(feature_matrix.shape[0], len(self.data_with_missing))
    
    def test_handle_missing_median(self):
        """Test handling missing values with median imputation."""
        engineer = FeatureEngineer(
            handle_missing='median',
            enable_graph_features=False,
            enable_sentiment=False
        )
        
        feature_matrix, feature_names = engineer.fit_transform(self.data_with_missing)
        
        # Should process without errors
        self.assertTrue(sparse.issparse(feature_matrix))
    
    def test_handle_missing_zero(self):
        """Test handling missing values with zero filling."""
        engineer = FeatureEngineer(
            handle_missing='zero',
            enable_graph_features=False,
            enable_sentiment=False
        )
        
        feature_matrix, feature_names = engineer.fit_transform(self.data_with_missing)
        
        # Should process without errors
        self.assertTrue(sparse.issparse(feature_matrix))
    
    def test_missing_value_handling_method(self):
        """Test the _handle_missing_values method directly."""
        engineer = FeatureEngineer(handle_missing='mean')
        
        # Create test matrix with missing values
        test_matrix = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, np.nan]])
        
        handled_matrix = engineer._handle_missing_values(test_matrix)
        
        # Should not contain NaN values
        self.assertFalse(np.any(np.isnan(handled_matrix)))


class TestFeatureEngineeringError(unittest.TestCase):
    """Test custom FeatureEngineeringError exception."""
    
    def test_feature_engineering_error(self):
        """Test FeatureEngineeringError exception handling."""
        with self.assertRaises(FeatureEngineeringError):
            raise FeatureEngineeringError("Test feature engineering error")
        
        try:
            raise FeatureEngineeringError("Test feature engineering error")
        except FeatureEngineeringError as e:
            self.assertEqual(str(e), "Test feature engineering error")


class TestFeatureEngineeringEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test feature engineering with empty DataFrame."""
        empty_df = pd.DataFrame()
        engineer = FeatureEngineer()
        
        # Should handle gracefully
        with self.assertRaises(ValueError):  # validate_dataframe should catch this
            engineer.fit_transform(empty_df)
    
    def test_single_row_dataframe(self):
        """Test feature engineering with single row."""
        single_row_df = pd.DataFrame({
            'review_text': ['Single review'],
            'rating': [5]
        })
        
        engineer = FeatureEngineer(
            enable_graph_features=False,
            enable_sentiment=False
        )
        
        feature_matrix, feature_names = engineer.fit_transform(single_row_df)
        
        # Should process single row
        self.assertEqual(feature_matrix.shape[0], 1)
        self.assertTrue(len(feature_names) > 0)
    
    def test_dataframe_without_text_column(self):
        """Test feature engineering without text column."""
        df_no_text = pd.DataFrame({
            'rating': [1, 2, 3],
            'user_id': ['u1', 'u2', 'u3']
        })
        
        engineer = FeatureEngineer(
            enable_graph_features=False,
            enable_sentiment=False
        )
        
        feature_matrix, feature_names = engineer.fit_transform(df_no_text)
        
        # Should still process other features
        self.assertEqual(feature_matrix.shape[0], 3)
        # Text features should not be present
        text_features = [name for name in feature_names if name.startswith('tfidf_')]
        self.assertEqual(len(text_features), 0)
    
    def test_very_short_text(self):
        """Test feature engineering with very short text."""
        short_text_df = pd.DataFrame({
            'review_text': ['a', 'b', 'c'],
            'rating': [1, 2, 3]
        })
        
        engineer = FeatureEngineer(
            tfidf_min_df=1,  # Allow single character words
            enable_graph_features=False,
            enable_sentiment=False
        )
        
        feature_matrix, feature_names = engineer.fit_transform(short_text_df)
        
        # Should handle short text
        self.assertEqual(feature_matrix.shape[0], 3)
        self.assertTrue(len(feature_names) > 0)
    
    def test_identical_text_rows(self):
        """Test feature engineering with identical text."""
        identical_df = pd.DataFrame({
            'review_text': ['same text', 'same text', 'same text'],
            'rating': [1, 2, 3],
            'user_id': ['u1', 'u2', 'u3']
        })
        
        engineer = FeatureEngineer(
            enable_graph_features=False,
            enable_sentiment=False
        )
        
        feature_matrix, feature_names = engineer.fit_transform(identical_df)
        
        # Should process identical text
        self.assertEqual(feature_matrix.shape[0], 3)
        # TF-IDF values might be identical for identical text
        self.assertTrue(len(feature_names) > 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
