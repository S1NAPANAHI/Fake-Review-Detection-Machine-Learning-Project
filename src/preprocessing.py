"""
Text Preprocessing Module for Fake Review Detection System

This module implements the TextPreprocessor class that handles comprehensive text preprocessing:
- Lowercase conversion and regex-based cleaning
- NLTK stopword removal, tokenization, and lemmatization
- Optional stemming support
- Missing value handling and data quality checks
- SMOTE wrapper for handling class imbalance
- Temporal consistency validation
- Basic metadata feature engineering

The main interface is through the transform(df) method which returns cleaned text
and engineered metadata features.
"""

import re
import logging
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any, Set
from datetime import datetime, timedelta
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# NLTK imports with error handling
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.tree import Tree
    
    # Download required NLTK data
    nltk_downloads = [
        'punkt', 'stopwords', 'wordnet', 'omw-1.4', 
        'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'
    ]
    
    for item in nltk_downloads:
        try:
            nltk.data.find(f'tokenizers/{item}')
        except LookupError:
            try:
                nltk.download(item, quiet=True)
            except Exception:
                pass
                
except ImportError:
    print("Warning: NLTK not available. Text preprocessing will be limited.")
    nltk = None

# Standalone utility functions (avoiding utils import issues)
import logging
import functools
import time

def setup_logging(name):
    """Simple logging setup"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger(name)

def timing_decorator(func):
    """Simple timing decorator"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Function '{func.__name__}' executed in {time.time() - start:.4f} seconds")
        return result
    return wrapper

def validate_dataframe(df, required_columns=None, min_rows=1):
    """Simple dataframe validation"""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if len(df) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows")

def get_setting(key, default=None):
    """Simple settings getter (returns defaults)"""
    return default


class PreprocessingError(Exception):
    """Custom exception for preprocessing errors"""
    pass


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Comprehensive text preprocessor for fake review detection.
    
    Features:
    - Text cleaning (lowercase, regex, stopwords)
    - Tokenization and lemmatization
    - Optional stemming
    - Missing value handling
    - Basic metadata feature engineering
    - SMOTE integration for class balancing
    - Temporal consistency validation
    - Configurable preprocessing pipeline
    """
    
    def __init__(self, 
                 text_column: str = 'review_text',
                 target_column: Optional[str] = 'is_fake',
                 user_id_column: str = 'user_id',
                 product_id_column: str = 'product_id',
                 date_column: str = 'date',
                 rating_column: str = 'rating',
                 lowercase: bool = True,
                 remove_html: bool = True,
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 remove_phone_numbers: bool = True,
                 remove_numbers: bool = False,
                 remove_punctuation: bool = True,
                 remove_extra_whitespace: bool = True,
                 remove_stopwords: bool = True,
                 custom_stopwords: Optional[List[str]] = None,
                 language: str = 'english',
                 tokenize: bool = True,
                 lemmatize: bool = True,
                 stem: bool = False,
                 stemmer_type: str = 'porter',
                 min_word_length: int = 2,
                 max_word_length: int = 15,
                 handle_missing: str = 'fill',  # 'drop', 'fill', 'keep'
                 missing_fill_value: str = '',
                 use_smote: bool = False,
                 smote_random_state: int = 42,
                 temporal_check: bool = True,
                 temporal_threshold_days: int = 1,
                 feature_engineering: bool = True,
                 max_features_tfidf: int = 5000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 random_state: int = 42):
        """
        Initialize TextPreprocessor with comprehensive configuration.
        
        Args:
            text_column: Name of the text column to process
            target_column: Name of the target column (for SMOTE)
            user_id_column: Name of user ID column
            product_id_column: Name of product ID column  
            date_column: Name of date column
            rating_column: Name of rating column
            lowercase: Convert text to lowercase
            remove_html: Remove HTML tags
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            remove_phone_numbers: Remove phone numbers
            remove_numbers: Remove all numbers
            remove_punctuation: Remove punctuation
            remove_extra_whitespace: Remove extra whitespace
            remove_stopwords: Remove stopwords
            custom_stopwords: Additional stopwords to remove
            language: Language for stopwords
            tokenize: Tokenize text
            lemmatize: Apply lemmatization
            stem: Apply stemming
            stemmer_type: Type of stemmer ('porter', 'snowball')
            min_word_length: Minimum word length to keep
            max_word_length: Maximum word length to keep
            handle_missing: How to handle missing values
            missing_fill_value: Value to fill missing values with
            use_smote: Apply SMOTE for class balancing
            smote_random_state: Random state for SMOTE
            temporal_check: Check temporal consistency
            temporal_threshold_days: Threshold for temporal anomalies
            feature_engineering: Generate metadata features
            max_features_tfidf: Maximum features for TF-IDF
            ngram_range: N-gram range for TF-IDF
            random_state: Random state for reproducibility
        """
        # Column names
        self.text_column = text_column
        self.target_column = target_column
        self.user_id_column = user_id_column
        self.product_id_column = product_id_column
        self.date_column = date_column
        self.rating_column = rating_column
        
        # Text cleaning options
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_phone_numbers = remove_phone_numbers
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.remove_extra_whitespace = remove_extra_whitespace
        
        # NLP options
        self.remove_stopwords = remove_stopwords
        self.custom_stopwords = custom_stopwords or []
        self.language = language
        self.tokenize = tokenize
        self.lemmatize = lemmatize
        self.stem = stem
        self.stemmer_type = stemmer_type
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        
        # Missing value handling
        self.handle_missing = handle_missing
        self.missing_fill_value = missing_fill_value
        
        # SMOTE options
        self.use_smote = use_smote
        self.smote_random_state = smote_random_state
        
        # Temporal validation
        self.temporal_check = temporal_check
        self.temporal_threshold_days = temporal_threshold_days
        
        # Feature engineering
        self.feature_engineering = feature_engineering
        self.max_features_tfidf = max_features_tfidf
        self.ngram_range = ngram_range
        self.random_state = random_state
        
        # Initialize components
        self.logger = setup_logging(self.__class__.__name__)
        self._initialize_nlp_components()
        
        # Fitted components
        self.is_fitted = False
        self.stopwords_set = set()
        self.scaler = None
        self.smote = None
        self.tfidf_vectorizer = None
        
        self.logger.info("TextPreprocessor initialized")
    
    def _initialize_nlp_components(self):
        """Initialize NLP components (lemmatizer, stemmer, stopwords)"""
        # Initialize lemmatizer
        if self.lemmatize and nltk:
            try:
                self.lemmatizer = WordNetLemmatizer()
            except Exception as e:
                self.logger.warning(f"Failed to initialize lemmatizer: {e}")
                self.lemmatizer = None
        else:
            self.lemmatizer = None
        
        # Initialize stemmer
        if self.stem and nltk:
            try:
                if self.stemmer_type == 'porter':
                    self.stemmer = PorterStemmer()
                elif self.stemmer_type == 'snowball':
                    self.stemmer = SnowballStemmer(self.language)
                else:
                    self.logger.warning(f"Unknown stemmer type: {self.stemmer_type}")
                    self.stemmer = PorterStemmer()
            except Exception as e:
                self.logger.warning(f"Failed to initialize stemmer: {e}")
                self.stemmer = None
        else:
            self.stemmer = None
        
        # Initialize stopwords
        if self.remove_stopwords and nltk:
            try:
                self.stopwords_set = set(stopwords.words(self.language))
                if self.custom_stopwords:
                    self.stopwords_set.update(self.custom_stopwords)
            except Exception as e:
                self.logger.warning(f"Failed to load stopwords: {e}")
                self.stopwords_set = set(self.custom_stopwords) if self.custom_stopwords else set()
        else:
            self.stopwords_set = set(self.custom_stopwords) if self.custom_stopwords else set()
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TextPreprocessor':
        """
        Fit the preprocessor to the training data.
        
        Args:
            X: Input DataFrame
            y: Target values (optional, will be extracted from X if target_column exists)
            
        Returns:
            self
        """
        self.logger.info("Fitting TextPreprocessor")
        
        # Validate input
        validate_dataframe(X, min_rows=1)
        
        # Extract target if available
        if y is None and self.target_column and self.target_column in X.columns:
            y = X[self.target_column]
        
        # Initialize SMOTE if requested
        if self.use_smote and y is not None:
            self.smote = SMOTE(random_state=self.smote_random_state)
        
        # Initialize scaler for numerical features
        self.scaler = StandardScaler()
        
        # Initialize TF-IDF vectorizer if feature engineering is enabled
        if self.feature_engineering:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features_tfidf,
                ngram_range=self.ngram_range,
                stop_words='english' if self.language == 'english' else None,
                lowercase=True,
                strip_accents='unicode'
            )
        
        self.is_fitted = True
        self.logger.info("TextPreprocessor fitted successfully")
        
        return self
    
    @timing_decorator
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame with comprehensive preprocessing.
        
        Args:
            df: Input DataFrame to transform
            
        Returns:
            Transformed DataFrame with cleaned text and engineered features
        """
        if not self.is_fitted:
            self.logger.warning("Preprocessor not fitted. Fitting with current data.")
            self.fit(df)
        
        self.logger.info(f"Transforming {len(df)} records")
        
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Handle missing values first
        df_processed = self._handle_missing_values(df_processed)
        
        # Perform temporal consistency check
        if self.temporal_check:
            df_processed = self._check_temporal_consistency(df_processed)
        
        # Clean and preprocess text
        if self.text_column in df_processed.columns:
            df_processed[self.text_column + '_cleaned'] = df_processed[self.text_column].apply(
                self._preprocess_text
            )
        
        # Engineer metadata features
        if self.feature_engineering:
            df_processed = self._engineer_features(df_processed)
        
        # Apply SMOTE if configured and target is available
        if (self.use_smote and 
            self.target_column and 
            self.target_column in df_processed.columns and
            self.smote is not None):
            df_processed = self._apply_smote(df_processed)
        
        self.logger.info(f"Transformation completed. Output shape: {df_processed.shape}")
        
        return df_processed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit the preprocessor and transform the data in one step.
        
        Args:
            X: Input DataFrame
            y: Target values (optional)
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to the configured strategy"""
        if self.handle_missing == 'drop':
            # Drop rows with missing text
            if self.text_column in df.columns:
                df = df.dropna(subset=[self.text_column])
        elif self.handle_missing == 'fill':
            # Fill missing values
            if self.text_column in df.columns:
                df[self.text_column] = df[self.text_column].fillna(self.missing_fill_value)
        # 'keep' option does nothing
        
        return df
    
    def _check_temporal_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check and flag temporal inconsistencies in the data"""
        if self.date_column not in df.columns:
            self.logger.warning(f"Date column '{self.date_column}' not found. Skipping temporal check.")
            return df
        
        self.logger.info("Performing temporal consistency check")
        
        # Convert date column to datetime if needed
        try:
            if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
                df[self.date_column] = pd.to_datetime(df[self.date_column], errors='coerce')
        except Exception as e:
            self.logger.warning(f"Failed to convert date column: {e}")
            return df
        
        # Initialize temporal flags
        df['temporal_anomaly'] = False
        df['temporal_anomaly_type'] = None
        
        # Check for reviews too close in time from same user
        if self.user_id_column in df.columns:
            df_sorted = df.sort_values([self.user_id_column, self.date_column])
            
            # Calculate time differences between consecutive reviews from same user
            df_sorted['prev_review_date'] = df_sorted.groupby(self.user_id_column)[self.date_column].shift()
            df_sorted['time_diff_hours'] = (
                (df_sorted[self.date_column] - df_sorted['prev_review_date'])
                .dt.total_seconds() / 3600
            )
            
            # Flag reviews posted too quickly
            threshold_hours = self.temporal_threshold_days * 24
            suspicious_mask = (
                df_sorted['time_diff_hours'].notna() & 
                (df_sorted['time_diff_hours'] < threshold_hours)
            )
            
            df_sorted.loc[suspicious_mask, 'temporal_anomaly'] = True
            df_sorted.loc[suspicious_mask, 'temporal_anomaly_type'] = 'too_frequent'
            
            # Update original dataframe
            df.update(df_sorted[['temporal_anomaly', 'temporal_anomaly_type']])
        
        # Check for future dates
        current_date = datetime.now()
        future_mask = df[self.date_column] > current_date
        df.loc[future_mask, 'temporal_anomaly'] = True
        df.loc[future_mask, 'temporal_anomaly_type'] = 'future_date'
        
        anomaly_count = df['temporal_anomaly'].sum()
        if anomaly_count > 0:
            self.logger.info(f"Found {anomaly_count} temporal anomalies")
        
        return df
    
    def _preprocess_text(self, text: str) -> str:
        """
        Comprehensive text preprocessing pipeline.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned and processed text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string if not already
        text = str(text)
        
        # Apply cleaning steps in order
        if self.remove_html:
            text = self._remove_html_tags(text)
        
        if self.remove_urls:
            text = self._remove_urls(text)
        
        if self.remove_emails:
            text = self._remove_emails(text)
        
        if self.remove_phone_numbers:
            text = self._remove_phone_numbers(text)
        
        if self.lowercase:
            text = text.lower()
        
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        if self.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization and NLP processing
        if self.tokenize and nltk:
            tokens = self._tokenize_and_process(text)
            text = ' '.join(tokens)
        
        return text
    
    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text"""
        html_pattern = re.compile(r'<[^>]+>')
        return html_pattern.sub('', text)
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = url_pattern.sub('', text)
        # Also remove www. patterns
        www_pattern = re.compile(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        return www_pattern.sub('', text)
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text"""
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        return email_pattern.sub('', text)
    
    def _remove_phone_numbers(self, text: str) -> str:
        """Remove phone numbers from text"""
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US format
            r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b',  # (123) 456-7890
            r'\b\d{3}\s\d{3}\s\d{4}\b',  # 123 456 7890
        ]
        
        for pattern in phone_patterns:
            text = re.sub(pattern, '', text)
        
        return text
    
    def _tokenize_and_process(self, text: str) -> List[str]:
        """Tokenize text and apply NLP processing (stopwords, lemmatization, stemming)"""
        try:
            # Tokenize
            tokens = word_tokenize(text)
            
            # Filter tokens
            processed_tokens = []
            for token in tokens:
                # Skip if too short or too long
                if len(token) < self.min_word_length or len(token) > self.max_word_length:
                    continue
                
                # Skip stopwords
                if self.remove_stopwords and token.lower() in self.stopwords_set:
                    continue
                
                # Apply lemmatization
                if self.lemmatize and self.lemmatizer:
                    try:
                        token = self.lemmatizer.lemmatize(token.lower())
                    except Exception:
                        pass
                
                # Apply stemming
                if self.stem and self.stemmer:
                    try:
                        token = self.stemmer.stem(token)
                    except Exception:
                        pass
                
                processed_tokens.append(token)
            
            return processed_tokens
            
        except Exception as e:
            # Fallback to simple splitting if NLTK fails
            self.logger.warning(f"NLTK processing failed, using simple tokenization: {e}")
            tokens = text.split()
            
            # Apply basic filtering
            filtered_tokens = []
            for token in tokens:
                if (len(token) >= self.min_word_length and 
                    len(token) <= self.max_word_length and 
                    (not self.remove_stopwords or token.lower() not in self.stopwords_set)):
                    filtered_tokens.append(token)
            
            return filtered_tokens
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer basic metadata features from the text and other columns"""
        self.logger.info("Engineering metadata features")
        
        # Text-based features
        if self.text_column in df.columns:
            text_col = df[self.text_column].fillna('')
            cleaned_col = df.get(self.text_column + '_cleaned', text_col)
            
            # Basic text statistics
            df['text_length'] = text_col.str.len()
            df['text_word_count'] = text_col.str.split().str.len()
            df['text_sentence_count'] = text_col.str.count(r'[.!?]+') + 1
            df['text_avg_word_length'] = df.apply(
                lambda row: np.mean([len(word) for word in str(row[self.text_column]).split()]) 
                if pd.notna(row[self.text_column]) and row[self.text_column] else 0, axis=1
            )
            
            # Cleaned text statistics
            df['cleaned_text_length'] = cleaned_col.str.len()
            df['cleaned_word_count'] = cleaned_col.str.split().str.len()
            
            # Text complexity features
            df['text_uppercase_ratio'] = text_col.str.count(r'[A-Z]') / (text_col.str.len() + 1)
            df['text_digit_ratio'] = text_col.str.count(r'\d') / (text_col.str.len() + 1)
            df['text_punctuation_ratio'] = text_col.str.count(r'[^\w\s]') / (text_col.str.len() + 1)
            df['text_special_char_ratio'] = text_col.str.count(r'[!@#$%^&*()_+]') / (text_col.str.len() + 1)
            
            # Repetition features
            df['text_repeated_chars'] = text_col.apply(self._count_repeated_chars)
            df['text_repeated_words'] = text_col.apply(self._count_repeated_words)
            
        # Rating-based features (if available)
        if self.rating_column in df.columns:
            df['rating_is_extreme'] = df[self.rating_column].isin([1, 5]).astype(int)
            # Add rating as numerical feature
            df['rating_numerical'] = pd.to_numeric(df[self.rating_column], errors='coerce').fillna(0)
        
        # User behavior features (if available)
        if self.user_id_column in df.columns:
            user_stats = df.groupby(self.user_id_column).agg({
                'rating_numerical': ['count', 'mean', 'std'] if 'rating_numerical' in df.columns else 'count',
                'text_length': ['mean', 'std'] if 'text_length' in df.columns else 'count'
            }).reset_index()
            
            # Flatten column names
            user_stats.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                for col in user_stats.columns.values]
            user_stats = user_stats.rename(columns={f'{self.user_id_column}_': self.user_id_column})
            
            # Merge back to original dataframe
            df = df.merge(user_stats, on=self.user_id_column, how='left', suffixes=('', '_user_avg'))
        
        # Product-based features (if available)
        if self.product_id_column in df.columns:
            product_stats = df.groupby(self.product_id_column).agg({
                'rating_numerical': ['count', 'mean', 'std'] if 'rating_numerical' in df.columns else 'count',
                'text_length': ['mean', 'std'] if 'text_length' in df.columns else 'count'
            }).reset_index()
            
            # Flatten column names
            product_stats.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                   for col in product_stats.columns.values]
            product_stats = product_stats.rename(columns={f'{self.product_id_column}_': self.product_id_column})
            
            # Merge back to original dataframe
            df = df.merge(product_stats, on=self.product_id_column, how='left', suffixes=('', '_product_avg'))
        
        # Time-based features (if available)
        if self.date_column in df.columns:
            try:
                date_series = pd.to_datetime(df[self.date_column], errors='coerce')
                df['review_hour'] = date_series.dt.hour
                df['review_day_of_week'] = date_series.dt.dayofweek
                df['review_month'] = date_series.dt.month
                df['review_is_weekend'] = (date_series.dt.dayofweek >= 5).astype(int)
            except Exception as e:
                self.logger.warning(f"Failed to extract time features: {e}")
        
        # Fill NaN values in engineered features
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        self.logger.info(f"Engineered {len([col for col in df.columns if col not in df.columns[:10]])} new features")
        
        return df
    
    def _count_repeated_chars(self, text: str) -> int:
        """Count repeated characters in text"""
        if pd.isna(text):
            return 0
        
        repeated_count = 0
        for i in range(1, len(text)):
            if text[i] == text[i-1]:
                repeated_count += 1
        
        return repeated_count
    
    def _count_repeated_words(self, text: str) -> int:
        """Count repeated words in text"""
        if pd.isna(text):
            return 0
        
        words = str(text).lower().split()
        word_counts = Counter(words)
        return sum(1 for count in word_counts.values() if count > 1)
    
    def _apply_smote(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply SMOTE for handling class imbalance"""
        if self.target_column not in df.columns:
            self.logger.warning("Target column not found. Skipping SMOTE.")
            return df
        
        self.logger.info("Applying SMOTE for class balancing")
        
        try:
            # Prepare features and target
            feature_columns = [col for col in df.columns 
                             if col != self.target_column and df[col].dtype in ['int64', 'float64']]
            
            if len(feature_columns) == 0:
                self.logger.warning("No numerical features found for SMOTE. Skipping.")
                return df
            
            X = df[feature_columns]
            y = df[self.target_column]
            
            # Handle missing values in features
            X = X.fillna(0)
            
            # Apply SMOTE
            X_resampled, y_resampled = self.smote.fit_resample(X, y)
            
            # Create new dataframe with resampled data
            df_resampled = pd.DataFrame(X_resampled, columns=feature_columns)
            df_resampled[self.target_column] = y_resampled
            
            # Add back non-numerical columns (this is simplified - in practice you might want more sophisticated handling)
            non_numeric_columns = [col for col in df.columns if col not in feature_columns + [self.target_column]]
            
            # For non-numeric columns, we'll duplicate the original values based on the resampled indices
            # This is a simplification - for production, you might want more sophisticated handling
            original_indices = np.arange(len(df))
            resampled_indices = np.random.choice(original_indices, size=len(X_resampled), replace=True)
            
            for col in non_numeric_columns:
                df_resampled[col] = df[col].iloc[resampled_indices].values
            
            self.logger.info(f"SMOTE applied. Original shape: {df.shape}, Resampled shape: {df_resampled.shape}")
            
            return df_resampled
            
        except Exception as e:
            self.logger.error(f"SMOTE failed: {e}")
            return df
    
    def get_feature_names(self) -> List[str]:
        """Get names of engineered features"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        # This would be populated during fit/transform
        # For now, return a list of expected feature names
        base_features = [
            'text_length', 'text_word_count', 'text_sentence_count', 'text_avg_word_length',
            'cleaned_text_length', 'cleaned_word_count',
            'text_uppercase_ratio', 'text_digit_ratio', 'text_punctuation_ratio', 'text_special_char_ratio',
            'text_repeated_chars', 'text_repeated_words',
            'rating_is_extreme', 'rating_numerical',
            'review_hour', 'review_day_of_week', 'review_month', 'review_is_weekend',
            'temporal_anomaly'
        ]
        
        return base_features
    
    def get_preprocessing_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the preprocessing results"""
        stats = {}
        
        if self.text_column in df.columns:
            original_text = df[self.text_column].fillna('')
            cleaned_text = df.get(self.text_column + '_cleaned', original_text)
            
            stats['original_text_stats'] = {
                'mean_length': original_text.str.len().mean(),
                'median_length': original_text.str.len().median(),
                'empty_count': (original_text == '').sum(),
                'total_records': len(df)
            }
            
            stats['cleaned_text_stats'] = {
                'mean_length': cleaned_text.str.len().mean(),
                'median_length': cleaned_text.str.len().median(),
                'empty_count': (cleaned_text == '').sum(),
                'reduction_ratio': 1 - (cleaned_text.str.len().mean() / (original_text.str.len().mean() + 1))
            }
        
        if 'temporal_anomaly' in df.columns:
            stats['temporal_anomalies'] = {
                'total_anomalies': df['temporal_anomaly'].sum(),
                'anomaly_rate': df['temporal_anomaly'].mean()
            }
        
        return stats


class SMOTEWrapper:
    """
    Wrapper class for SMOTE integration with text preprocessing.
    
    This provides a clean interface for applying SMOTE after text preprocessing,
    handling the complexity of mixed data types (text and numerical features).
    """
    
    def __init__(self, 
                 preprocessor: TextPreprocessor,
                 smote_params: Optional[Dict[str, Any]] = None):
        """
        Initialize SMOTE wrapper.
        
        Args:
            preprocessor: Fitted TextPreprocessor instance
            smote_params: Parameters for SMOTE
        """
        self.preprocessor = preprocessor
        self.smote_params = smote_params or {'random_state': 42}
        self.smote = SMOTE(**self.smote_params)
        self.logger = setup_logging(self.__class__.__name__)
    
    def fit_resample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE resampling to preprocessed data.
        
        Args:
            X: Preprocessed features
            y: Target values
            
        Returns:
            Resampled features and targets
        """
        self.logger.info("Applying SMOTE resampling")
        
        # Select only numerical columns for SMOTE
        numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_columns) == 0:
            self.logger.warning("No numerical features found. Returning original data.")
            return X, y
        
        X_numerical = X[numerical_columns].fillna(0)
        
        # Apply SMOTE
        X_resampled, y_resampled = self.smote.fit_resample(X_numerical, y)
        
        # Convert back to DataFrame
        X_resampled_df = pd.DataFrame(X_resampled, columns=numerical_columns)
        
        # Handle non-numerical columns by sampling from original data
        non_numerical_columns = [col for col in X.columns if col not in numerical_columns]
        
        if non_numerical_columns:
            # Sample non-numerical data to match resampled size
            original_indices = np.arange(len(X))
            resampled_indices = np.random.choice(
                original_indices, 
                size=len(X_resampled), 
                replace=True
            )
            
            for col in non_numerical_columns:
                X_resampled_df[col] = X[col].iloc[resampled_indices].values
        
        self.logger.info(f"SMOTE completed. Shape: {X.shape} -> {X_resampled_df.shape}")
        
        return X_resampled_df, pd.Series(y_resampled)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'review_text': [
            "This product is AMAZING!!! I love it so much!!!",
            "Terrible quality. Would not recommend.",
            "Good value for money. Works as expected.",
            "Best purchase ever! 5 stars!",
            "Poor customer service and bad product quality."
        ],
        'rating': [5, 1, 4, 5, 2],
        'user_id': ['user1', 'user2', 'user3', 'user1', 'user4'],
        'product_id': ['prod1', 'prod1', 'prod2', 'prod3', 'prod1'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-04'],
        'is_fake': [1, 0, 0, 1, 0]
    })
    
    # Initialize and test preprocessor
    preprocessor = TextPreprocessor(
        feature_engineering=True,
        temporal_check=True,
        use_smote=False  # Disable for small sample
    )
    
    # Transform data
    processed_data = preprocessor.fit_transform(sample_data)
    
    print("Preprocessing completed successfully!")
    print(f"Original shape: {sample_data.shape}")
    print(f"Processed shape: {processed_data.shape}")
    print(f"New columns: {set(processed_data.columns) - set(sample_data.columns)}")
    
    # Get preprocessing statistics
    stats = preprocessor.get_preprocessing_stats(processed_data)
    print("\nPreprocessing Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
