"""
Unit tests for the text preprocessing module.

This module tests text preprocessing functionality including text cleaning,
tokenization, lemmatization, feature engineering, temporal validation,
SMOTE integration, and comprehensive preprocessing pipeline.
Uses small synthetic fixtures and validates preprocessing outputs and shapes.
"""

import unittest
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    TextPreprocessor, SMOTEWrapper, PreprocessingError
)


class TestTextPreprocessorInitialization(unittest.TestCase):
    """Test TextPreprocessor initialization and configuration."""
    
    def test_default_initialization(self):
        """Test TextPreprocessor initialization with default parameters."""
        preprocessor = TextPreprocessor()
        
        # Verify default settings
        self.assertEqual(preprocessor.text_column, 'review_text')
        self.assertEqual(preprocessor.target_column, 'is_fake')
        self.assertTrue(preprocessor.lowercase)
        self.assertTrue(preprocessor.remove_html)
        self.assertTrue(preprocessor.remove_urls)
        self.assertTrue(preprocessor.lemmatize)
        self.assertFalse(preprocessor.stem)
        self.assertFalse(preprocessor.is_fitted)
    
    def test_custom_initialization(self):
        """Test TextPreprocessor initialization with custom parameters."""
        preprocessor = TextPreprocessor(
            text_column='custom_text',
            lowercase=False,
            remove_stopwords=False,
            stem=True,
            feature_engineering=False
        )
        
        # Verify custom settings
        self.assertEqual(preprocessor.text_column, 'custom_text')
        self.assertFalse(preprocessor.lowercase)
        self.assertFalse(preprocessor.remove_stopwords)
        self.assertTrue(preprocessor.stem)
        self.assertFalse(preprocessor.feature_engineering)


class TestTextCleaning(unittest.TestCase):
    """Test text cleaning functionality."""
    
    def setUp(self):
        self.preprocessor = TextPreprocessor()
        self.preprocessor.is_fitted = True  # Skip fitting for unit tests
    
    def test_remove_html_tags(self):
        """Test HTML tag removal."""
        test_text = "<p>This is a <b>great</b> product!</p>"
        cleaned_text = self.preprocessor._remove_html_tags(test_text)
        self.assertEqual(cleaned_text, "This is a great product!")
    
    def test_remove_urls(self):
        """Test URL removal."""
        test_text = "Check out https://example.com for more info!"
        cleaned_text = self.preprocessor._remove_urls(test_text)
        self.assertEqual(cleaned_text, "Check out  for more info!")
        
        # Test www patterns
        test_text_www = "Visit www.example.com today!"
        cleaned_text_www = self.preprocessor._remove_urls(test_text_www)
        self.assertEqual(cleaned_text_www, "Visit  today!")
    
    def test_remove_emails(self):
        """Test email address removal."""
        test_text = "Contact us at support@example.com for help."
        cleaned_text = self.preprocessor._remove_emails(test_text)
        self.assertEqual(cleaned_text, "Contact us at  for help.")
    
    def test_remove_phone_numbers(self):
        """Test phone number removal."""
        test_cases = [
            "Call 123-456-7890 now!",
            "Phone: (123) 456-7890",
            "Number: 123 456 7890"
        ]
        
        for test_text in test_cases:
            cleaned_text = self.preprocessor._remove_phone_numbers(test_text)
            # Should not contain the phone number
            self.assertNotIn("123", cleaned_text.replace(" ", ""))
    
    def test_comprehensive_text_cleaning(self):
        """Test comprehensive text preprocessing pipeline."""
        # Complex test text with various elements
        test_text = """<p>This is an AMAZING product!!! 
                      Visit https://example.com or email support@test.com.
                      Call (555) 123-4567 for more info!</p>"""
        
        cleaned_text = self.preprocessor._preprocess_text(test_text)
        
        # Should be cleaned and processed
        self.assertIsInstance(cleaned_text, str)
        self.assertNotIn('<p>', cleaned_text)
        self.assertNotIn('https://', cleaned_text)
        self.assertNotIn('@', cleaned_text)
        
        # With lowercase enabled, should be lowercase
        if self.preprocessor.lowercase:
            self.assertEqual(cleaned_text, cleaned_text.lower())
    
    def test_empty_and_null_text_handling(self):
        """Test handling of empty and null text."""
        # Test empty string
        self.assertEqual(self.preprocessor._preprocess_text(""), "")
        
        # Test None
        self.assertEqual(self.preprocessor._preprocess_text(None), "")
        
        # Test pandas NaN
        self.assertEqual(self.preprocessor._preprocess_text(pd.NaType()), "")


class TestTokenizationAndNLP(unittest.TestCase):
    """Test tokenization and NLP processing functionality."""
    
    def setUp(self):
        # Create preprocessor with specific NLP settings
        self.preprocessor = TextPreprocessor(
            tokenize=True,
            lemmatize=True,
            stem=False,
            remove_stopwords=True,
            min_word_length=2,
            max_word_length=15
        )
        self.preprocessor.is_fitted = True
        self.preprocessor._initialize_nlp_components()
    
    @patch('src.preprocessing.nltk')
    def test_tokenize_and_process_with_nltk(self, mock_nltk):
        """Test tokenization and processing with NLTK available."""
        # Mock NLTK components
        mock_nltk.word_tokenize.return_value = ['great', 'product', 'amazing', 'quality']
        
        # Mock lemmatizer
        mock_lemmatizer = MagicMock()
        mock_lemmatizer.lemmatize.side_effect = lambda x: x  # Return as-is
        self.preprocessor.lemmatizer = mock_lemmatizer
        
        # Mock stopwords
        self.preprocessor.stopwords_set = {'the', 'is', 'a', 'an'}
        
        result = self.preprocessor._tokenize_and_process("Great product with amazing quality!")
        
        # Should return list of processed tokens
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
    
    def test_tokenize_and_process_fallback(self):
        """Test tokenization fallback when NLTK fails."""
        # Test with simple text splitting
        test_text = "great product amazing quality"
        
        # Force fallback by setting nltk to None
        with patch('src.preprocessing.nltk', None):
            result = self.preprocessor._tokenize_and_process(test_text)
        
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
    
    def test_word_length_filtering(self):
        """Test word length filtering."""
        # Create tokens with various lengths
        test_tokens = ['a', 'ok', 'good', 'excellent', 'supercalifragilisticexpialidocious']
        
        # Filter based on min/max length
        filtered_tokens = []
        for token in test_tokens:
            if (len(token) >= self.preprocessor.min_word_length and 
                len(token) <= self.preprocessor.max_word_length):
                filtered_tokens.append(token)
        
        # Should filter out 'a' (too short) and very long word
        self.assertNotIn('a', filtered_tokens)
        self.assertIn('ok', filtered_tokens)
        self.assertIn('good', filtered_tokens)
        self.assertNotIn('supercalifragilisticexpialidocious', filtered_tokens)


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functionality."""
    
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'review_text': [
                'This product is AMAZING!!! Great quality and fast shipping.',
                'terrible product, waste of money',
                'Good value for the price, works as expected.',
                'LOVE IT LOVE IT LOVE IT!!! Best purchase ever!!!',
                'Not bad, could be better though.'
            ],
            'rating': [5, 1, 4, 5, 3],
            'user_id': ['user1', 'user2', 'user3', 'user1', 'user4'],
            'product_id': ['prod1', 'prod1', 'prod2', 'prod3', 'prod1'],
            'date': [
                '2023-01-01 10:00:00',
                '2023-01-02 15:30:00',
                '2023-01-03 09:15:00',
                '2023-01-01 11:00:00',  # Same day as first review
                '2023-01-04 16:45:00'
            ]
        })
        
        self.preprocessor = TextPreprocessor(feature_engineering=True)
        self.preprocessor.is_fitted = True
    
    def test_basic_text_features(self):
        """Test basic text-based feature engineering."""
        result_df = self.preprocessor._engineer_features(self.sample_data.copy())
        
        # Check for expected text features
        expected_features = [
            'text_length', 'text_word_count', 'text_sentence_count',
            'text_avg_word_length', 'text_uppercase_ratio', 'text_digit_ratio',
            'text_punctuation_ratio', 'text_repeated_chars', 'text_repeated_words'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, result_df.columns, f"Missing feature: {feature}")
        
        # Verify feature values are reasonable
        self.assertTrue(all(result_df['text_length'] > 0))
        self.assertTrue(all(result_df['text_word_count'] > 0))
        self.assertTrue(all(result_df['text_uppercase_ratio'] >= 0))
        self.assertTrue(all(result_df['text_uppercase_ratio'] <= 1))
    
    def test_rating_features(self):
        """Test rating-based feature engineering."""
        result_df = self.preprocessor._engineer_features(self.sample_data.copy())
        
        # Check for rating features
        self.assertIn('rating_is_extreme', result_df.columns)
        self.assertIn('rating_numerical', result_df.columns)
        
        # Verify extreme rating detection
        extreme_ratings = result_df['rating_is_extreme'].sum()
        expected_extreme = sum(1 for r in self.sample_data['rating'] if r in [1, 5])
        self.assertEqual(extreme_ratings, expected_extreme)
    
    def test_user_behavior_features(self):
        """Test user behavior feature engineering."""
        result_df = self.preprocessor._engineer_features(self.sample_data.copy())
        
        # Should have user-level aggregated features
        user_feature_cols = [col for col in result_df.columns if 'user' in col.lower()]
        self.assertTrue(len(user_feature_cols) > 0)
        
        # Verify user1 has multiple reviews reflected in features
        user1_rows = result_df[result_df['user_id'] == 'user1']
        self.assertTrue(len(user1_rows) > 1)
    
    def test_temporal_features(self):
        """Test temporal feature engineering."""
        result_df = self.preprocessor._engineer_features(self.sample_data.copy())
        
        # Check for temporal features
        temporal_features = [
            'review_hour', 'review_day_of_week', 'review_month', 'review_is_weekend'
        ]
        
        for feature in temporal_features:
            self.assertIn(feature, result_df.columns)
        
        # Verify temporal values are in expected ranges
        self.assertTrue(all(result_df['review_hour'].between(0, 23)))
        self.assertTrue(all(result_df['review_day_of_week'].between(0, 6)))
        self.assertTrue(all(result_df['review_month'].between(1, 12)))
        self.assertTrue(all(result_df['review_is_weekend'].isin([0, 1])))
    
    def test_repeated_characters_counting(self):
        """Test repeated character counting."""
        # Test repeated chars
        self.assertEqual(self.preprocessor._count_repeated_chars("aaa"), 2)
        self.assertEqual(self.preprocessor._count_repeated_chars("hello"), 2)  # 'll'
        self.assertEqual(self.preprocessor._count_repeated_chars("test"), 0)
        self.assertEqual(self.preprocessor._count_repeated_chars(""), 0)
    
    def test_repeated_words_counting(self):
        """Test repeated word counting."""
        # Test repeated words
        self.assertEqual(self.preprocessor._count_repeated_words("great great product"), 1)
        self.assertEqual(self.preprocessor._count_repeated_words("this is a great product"), 0)
        self.assertEqual(self.preprocessor._count_repeated_words("love love love it"), 1)  # 'love' repeated
        self.assertEqual(self.preprocessor._count_repeated_words(""), 0)


class TestTemporalConsistency(unittest.TestCase):
    """Test temporal consistency validation."""
    
    def setUp(self):
        self.preprocessor = TextPreprocessor(
            temporal_check=True,
            temporal_threshold_days=1
        )
        self.preprocessor.is_fitted = True
        
        # Create test data with temporal anomalies
        self.temporal_data = pd.DataFrame({
            'review_text': ['Review 1', 'Review 2', 'Review 3', 'Review 4'],
            'user_id': ['user1', 'user1', 'user2', 'user1'],
            'date': [
                '2023-01-01 10:00:00',
                '2023-01-01 10:30:00',  # 30 minutes later - suspicious
                '2023-01-02 15:00:00',  # Different user, OK
                '2025-01-01 12:00:00'   # Future date - anomaly
            ]
        })
    
    def test_temporal_consistency_check(self):
        """Test temporal consistency validation."""
        result_df = self.preprocessor._check_temporal_consistency(self.temporal_data.copy())
        
        # Should have temporal anomaly columns
        self.assertIn('temporal_anomaly', result_df.columns)
        self.assertIn('temporal_anomaly_type', result_df.columns)
        
        # Should detect anomalies
        anomaly_count = result_df['temporal_anomaly'].sum()
        self.assertGreater(anomaly_count, 0)
        
        # Should detect future date anomaly
        future_anomalies = result_df[result_df['temporal_anomaly_type'] == 'future_date']
        self.assertTrue(len(future_anomalies) > 0)
    
    def test_temporal_check_missing_date_column(self):
        """Test temporal check with missing date column."""
        data_no_date = pd.DataFrame({
            'review_text': ['Review 1'],
            'user_id': ['user1']
        })
        
        # Should not crash and return data unchanged (except for added columns)
        result_df = self.preprocessor._check_temporal_consistency(data_no_date)
        self.assertEqual(len(result_df), len(data_no_date))
    
    def test_user_review_frequency_detection(self):
        """Test detection of users posting reviews too frequently."""
        # Create data with reviews posted within threshold
        frequent_data = pd.DataFrame({
            'review_text': ['Review 1', 'Review 2'],
            'user_id': ['user1', 'user1'],
            'date': [
                '2023-01-01 10:00:00',
                '2023-01-01 12:00:00'  # 2 hours later - within 1 day threshold
            ]
        })
        
        result_df = self.preprocessor._check_temporal_consistency(frequent_data)
        
        # Should detect frequency anomaly
        frequent_anomalies = result_df[result_df['temporal_anomaly_type'] == 'too_frequent']
        self.assertTrue(len(frequent_anomalies) > 0)


class TestMissingValueHandling(unittest.TestCase):
    """Test missing value handling functionality."""
    
    def setUp(self):
        self.data_with_missing = pd.DataFrame({
            'review_text': ['Great product!', None, '', 'Love it!'],
            'rating': [5, 3, np.nan, 4],
            'user_id': ['user1', 'user2', 'user3', 'user4']
        })
    
    def test_handle_missing_drop(self):
        """Test dropping rows with missing values."""
        preprocessor = TextPreprocessor(handle_missing='drop')
        preprocessor.is_fitted = True
        
        result_df = preprocessor._handle_missing_values(self.data_with_missing.copy())
        
        # Should drop rows with missing text
        self.assertLess(len(result_df), len(self.data_with_missing))
        self.assertTrue(all(result_df['review_text'].notna()))
    
    def test_handle_missing_fill(self):
        """Test filling missing values."""
        preprocessor = TextPreprocessor(
            handle_missing='fill',
            missing_fill_value='NO_TEXT'
        )
        preprocessor.is_fitted = True
        
        result_df = preprocessor._handle_missing_values(self.data_with_missing.copy())
        
        # Should fill missing text values
        self.assertEqual(len(result_df), len(self.data_with_missing))
        filled_values = result_df[result_df['review_text'] == 'NO_TEXT']
        self.assertTrue(len(filled_values) > 0)
    
    def test_handle_missing_keep(self):
        """Test keeping missing values as-is."""
        preprocessor = TextPreprocessor(handle_missing='keep')
        preprocessor.is_fitted = True
        
        result_df = preprocessor._handle_missing_values(self.data_with_missing.copy())
        
        # Should keep all rows
        self.assertEqual(len(result_df), len(self.data_with_missing))


class TestPreprocessorPipeline(unittest.TestCase):
    """Test complete preprocessing pipeline."""
    
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'review_text': [
                'This is an AMAZING product!!! Highly recommended.',
                'Poor quality, would not buy again.',
                '<p>Good value for money.</p>',
                'Best purchase ever! Contact support@test.com for questions.',
                'Not impressed with this product.'
            ],
            'rating': [5, 1, 4, 5, 2],
            'user_id': ['user1', 'user2', 'user3', 'user1', 'user4'],
            'product_id': ['prod1', 'prod1', 'prod2', 'prod3', 'prod1'],
            'date': [
                '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'
            ],
            'is_fake': [1, 0, 0, 1, 0]
        })
    
    def test_fit_transform_pipeline(self):
        """Test complete fit-transform pipeline."""
        preprocessor = TextPreprocessor(
            feature_engineering=True,
            temporal_check=True,
            use_smote=False  # Disable for small dataset
        )
        
        # Fit and transform
        result_df = preprocessor.fit_transform(self.sample_data)
        
        # Verify output structure
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertGreater(len(result_df.columns), len(self.sample_data.columns))
        
        # Should have cleaned text column
        self.assertIn('review_text_cleaned', result_df.columns)
        
        # Should have engineered features
        feature_cols = [col for col in result_df.columns if 'text_' in col]
        self.assertTrue(len(feature_cols) > 5)
    
    def test_transform_only_after_fit(self):
        """Test transform-only after fitting."""
        preprocessor = TextPreprocessor()
        
        # Fit first
        preprocessor.fit(self.sample_data)
        
        # Then transform
        result_df = preprocessor.transform(self.sample_data)
        
        self.assertTrue(preprocessor.is_fitted)
        self.assertIsInstance(result_df, pd.DataFrame)
    
    def test_transform_without_fit(self):
        """Test transform automatically fits if not fitted."""
        preprocessor = TextPreprocessor()
        
        # Transform without explicit fit
        result_df = preprocessor.transform(self.sample_data)
        
        # Should automatically fit and then transform
        self.assertTrue(preprocessor.is_fitted)
        self.assertIsInstance(result_df, pd.DataFrame)
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        preprocessor = TextPreprocessor(feature_engineering=True)
        preprocessor.fit(self.sample_data)
        
        feature_names = preprocessor.get_feature_names()
        
        self.assertIsInstance(feature_names, list)
        self.assertTrue(len(feature_names) > 0)
        
        # Should contain expected feature categories
        text_features = [name for name in feature_names if 'text_' in name]
        rating_features = [name for name in feature_names if 'rating_' in name]
        
        self.assertTrue(len(text_features) > 0)
        self.assertTrue(len(rating_features) > 0)
    
    def test_get_preprocessing_stats(self):
        """Test getting preprocessing statistics."""
        preprocessor = TextPreprocessor()
        result_df = preprocessor.fit_transform(self.sample_data)
        
        stats = preprocessor.get_preprocessing_stats(result_df)
        
        self.assertIsInstance(stats, dict)
        self.assertIn('original_text_stats', stats)
        self.assertIn('cleaned_text_stats', stats)
        
        # Verify stats structure
        orig_stats = stats['original_text_stats']
        self.assertIn('mean_length', orig_stats)
        self.assertIn('total_records', orig_stats)


class TestSMOTEWrapper(unittest.TestCase):
    """Test SMOTE integration wrapper."""
    
    def setUp(self):
        # Create imbalanced dataset
        self.imbalanced_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'text_feature': ['text'] * 10,  # Non-numeric feature
            'target': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  # Imbalanced: 7 vs 3
        })
        
        self.preprocessor = TextPreprocessor()
    
    @patch('src.preprocessing.SMOTE')
    def test_smote_wrapper_initialization(self, mock_smote):
        """Test SMOTEWrapper initialization."""
        wrapper = SMOTEWrapper(self.preprocessor)
        
        self.assertEqual(wrapper.preprocessor, self.preprocessor)
        self.assertIsInstance(wrapper.smote_params, dict)
        mock_smote.assert_called_once()
    
    @patch('src.preprocessing.SMOTE')
    def test_smote_fit_resample(self, mock_smote_class):
        """Test SMOTE fit_resample functionality."""
        # Mock SMOTE instance
        mock_smote_instance = MagicMock()
        mock_smote_class.return_value = mock_smote_instance
        
        # Mock fit_resample to return resampled data
        X_resampled = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
        y_resampled = np.array([0, 0, 1, 1])
        mock_smote_instance.fit_resample.return_value = (X_resampled, y_resampled)
        
        wrapper = SMOTEWrapper(self.preprocessor)
        
        X = self.imbalanced_data[['feature1', 'feature2']]
        y = self.imbalanced_data['target']
        
        X_result, y_result = wrapper.fit_resample(X, y)
        
        # Verify results
        self.assertIsInstance(X_result, pd.DataFrame)
        self.assertIsInstance(y_result, pd.Series)
        mock_smote_instance.fit_resample.assert_called_once()
    
    def test_smote_with_no_numerical_features(self):
        """Test SMOTE with no numerical features."""
        wrapper = SMOTEWrapper(self.preprocessor)
        
        # DataFrame with only non-numerical features
        X_non_numeric = pd.DataFrame({'text_col': ['text1', 'text2', 'text3']})
        y = pd.Series([0, 1, 0])
        
        X_result, y_result = wrapper.fit_resample(X_non_numeric, y)
        
        # Should return original data unchanged
        pd.testing.assert_frame_equal(X_result, X_non_numeric)
        pd.testing.assert_series_equal(y_result, y)


class TestPreprocessingError(unittest.TestCase):
    """Test custom PreprocessingError exception."""
    
    def test_preprocessing_error(self):
        """Test PreprocessingError exception handling."""
        with self.assertRaises(PreprocessingError):
            raise PreprocessingError("Test preprocessing error")
        
        try:
            raise PreprocessingError("Test preprocessing error")
        except PreprocessingError as e:
            self.assertEqual(str(e), "Test preprocessing error")


class TestPreprocessorEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test preprocessing with empty DataFrame."""
        empty_df = pd.DataFrame()
        preprocessor = TextPreprocessor()
        
        # Should handle gracefully
        with self.assertRaises(ValueError):  # validate_dataframe should catch this
            preprocessor.fit_transform(empty_df)
    
    def test_dataframe_without_text_column(self):
        """Test preprocessing without required text column."""
        df_no_text = pd.DataFrame({'rating': [1, 2, 3], 'user_id': ['u1', 'u2', 'u3']})
        preprocessor = TextPreprocessor(text_column='missing_column')
        
        result_df = preprocessor.fit_transform(df_no_text)
        
        # Should still process but not add cleaned text
        self.assertNotIn('missing_column_cleaned', result_df.columns)
    
    def test_very_large_text_values(self):
        """Test preprocessing with very large text values."""
        large_text_data = pd.DataFrame({
            'review_text': ['short', 'a' * 10000],  # Very long text
            'rating': [5, 1]
        })
        
        preprocessor = TextPreprocessor()
        result_df = preprocessor.fit_transform(large_text_data)
        
        # Should handle large text without crashing
        self.assertEqual(len(result_df), 2)
        self.assertIn('review_text_cleaned', result_df.columns)
    
    def test_special_character_handling(self):
        """Test handling of special characters and unicode."""
        special_data = pd.DataFrame({
            'review_text': [
                'Café is great! 😊',
                'Product works well ★★★★★',
                '这个产品很好',  # Chinese characters
                'Très bon produit!'  # French
            ]
        })
        
        preprocessor = TextPreprocessor()
        result_df = preprocessor.fit_transform(special_data)
        
        # Should process without errors
        self.assertEqual(len(result_df), 4)
        self.assertTrue(all(result_df['review_text_cleaned'].notna()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
