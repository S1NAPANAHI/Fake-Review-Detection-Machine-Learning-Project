"""
Feature Engineering Module for Fake Review Detection System

This module implements the FeatureEngineer class that handles comprehensive feature extraction:
- Text features using TF-IDF vectorization (1-3-gram, max 5k features)
- Behavioral features (review length, frequency, user statistics)
- Graph features using NetworkX metrics and optional node2vec embeddings
- Sentiment features using TextBlob and VADER sentiment analysis

The main interface returns concatenated sparse/dense matrices and feature names
for downstream machine learning models.
"""

import logging
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# NetworkX for graph features
try:
    import networkx as nx
    from node2vec import Node2Vec
    NETWORKX_AVAILABLE = True
except ImportError:
    print("Warning: NetworkX or node2vec not available. Graph features will be limited.")
    NETWORKX_AVAILABLE = False
    nx = None
    Node2Vec = None

# Sentiment analysis libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    print("Warning: TextBlob not available. TextBlob sentiment features will be disabled.")
    TEXTBLOB_AVAILABLE = False
    TextBlob = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    print("Warning: VADER not available. VADER sentiment features will be disabled.")
    VADER_AVAILABLE = False
    SentimentIntensityAnalyzer = None

# Import from project modules - with fallbacks for standalone execution
try:
    from .utils import timing_decorator, setup_logging, validate_dataframe
except ImportError:
    try:
        from utils import timing_decorator, setup_logging, validate_dataframe
    except ImportError:
        # Fallback implementations
        import functools
        import time
        
        def timing_decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                print(f"Function '{func.__name__}' executed in {time.time() - start:.4f} seconds")
                return result
            return wrapper
            
        def setup_logging(name):
            import logging
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            return logging.getLogger(name)
            
        def validate_dataframe(df, required_columns=None, min_rows=1):
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
            if len(df) < min_rows:
                raise ValueError(f"DataFrame must have at least {min_rows} rows")


class FeatureEngineeringError(Exception):
    """Custom exception for feature engineering errors"""
    pass


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Comprehensive feature engineer for fake review detection.
    
    Features:
    - Text features using TF-IDF (1-3-gram, max 5k features)  
    - Behavioral features (review stats, user behavior patterns)
    - Graph features using NetworkX metrics and optional node2vec
    - Sentiment features using TextBlob and VADER
    
    Returns concatenated sparse/dense feature matrix and feature names.
    """
    
    def __init__(self,
                 text_column: str = 'review_text',
                 user_id_column: str = 'user_id',
                 product_id_column: str = 'product_id', 
                 date_column: str = 'date',
                 rating_column: str = 'rating',
                 
                 # Text feature parameters
                 tfidf_max_features: int = 5000,
                 tfidf_ngram_range: Tuple[int, int] = (1, 3),
                 tfidf_min_df: int = 2,
                 tfidf_max_df: float = 0.95,
                 
                 # Behavioral feature parameters
                 behavioral_time_window: int = 30,  # days
                 
                 # Graph feature parameters
                 enable_graph_features: bool = True,
                 node2vec_dimensions: int = 64,
                 node2vec_walk_length: int = 30,
                 node2vec_num_walks: int = 200,
                 node2vec_workers: int = 4,
                 
                 # Sentiment feature parameters
                 enable_sentiment: bool = True,
                 
                 # General parameters
                 feature_scaling: str = 'standard',  # 'standard', 'minmax', 'none'
                 handle_missing: str = 'median',  # 'mean', 'median', 'zero'
                 random_state: int = 42):
        """
        Initialize FeatureEngineer with comprehensive configuration.
        
        Args:
            text_column: Name of the text column
            user_id_column: Name of user ID column
            product_id_column: Name of product ID column
            date_column: Name of date column
            rating_column: Name of rating column
            tfidf_max_features: Maximum TF-IDF features
            tfidf_ngram_range: N-gram range for TF-IDF
            tfidf_min_df: Minimum document frequency for TF-IDF
            tfidf_max_df: Maximum document frequency for TF-IDF
            behavioral_time_window: Time window for behavioral features (days)
            enable_graph_features: Enable graph feature extraction
            node2vec_dimensions: Dimensions for node2vec embeddings
            node2vec_walk_length: Walk length for node2vec
            node2vec_num_walks: Number of walks for node2vec
            node2vec_workers: Number of workers for node2vec
            enable_sentiment: Enable sentiment feature extraction
            feature_scaling: Feature scaling method
            handle_missing: Missing value handling method
            random_state: Random state for reproducibility
        """
        # Column names
        self.text_column = text_column
        self.user_id_column = user_id_column
        self.product_id_column = product_id_column
        self.date_column = date_column
        self.rating_column = rating_column
        
        # Text feature parameters
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_df = tfidf_max_df
        
        # Behavioral parameters
        self.behavioral_time_window = behavioral_time_window
        
        # Graph parameters
        self.enable_graph_features = enable_graph_features and NETWORKX_AVAILABLE
        self.node2vec_dimensions = node2vec_dimensions
        self.node2vec_walk_length = node2vec_walk_length
        self.node2vec_num_walks = node2vec_num_walks
        self.node2vec_workers = node2vec_workers
        
        # Sentiment parameters  
        self.enable_sentiment = enable_sentiment
        
        # General parameters
        self.feature_scaling = feature_scaling
        self.handle_missing = handle_missing
        self.random_state = random_state
        
        # Initialize components
        self.logger = setup_logging(self.__class__.__name__)
        self._initialize_components()
        
        # Fitted components
        self.is_fitted = False
        self.feature_names_ = []
        self.text_feature_names_ = []
        self.behavioral_feature_names_ = []
        self.graph_feature_names_ = []
        self.sentiment_feature_names_ = []
        
        self.logger.info("FeatureEngineer initialized")
    
    def _initialize_components(self):
        """Initialize feature extraction components"""
        # TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=self.tfidf_ngram_range,
            min_df=self.tfidf_min_df,
            max_df=self.tfidf_max_df,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode',
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Only words with 2+ letters
        )
        
        # Scalers for numerical features
        if self.feature_scaling == 'standard':
            self.scaler = StandardScaler()
        elif self.feature_scaling == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
            
        # Sentiment analyzers
        if self.enable_sentiment:
            if VADER_AVAILABLE:
                self.vader_analyzer = SentimentIntensityAnalyzer()
            else:
                self.vader_analyzer = None
                
        # Graph-related components
        self.user_product_graph = None
        self.node2vec_model = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineer':
        """
        Fit the feature engineer to the training data.
        
        Args:
            X: Input DataFrame
            y: Target values (optional)
            
        Returns:
            self
        """
        self.logger.info("Fitting FeatureEngineer")
        
        # Validate input
        validate_dataframe(X, min_rows=1)
        
        # Fit text features
        if self.text_column in X.columns:
            self._fit_text_features(X)
            
        # Fit behavioral features (no fitting required, computed on-the-fly)
        
        # Fit graph features
        if self.enable_graph_features:
            self._fit_graph_features(X)
            
        # Fit sentiment features (no fitting required)
        
        # Fit scalers for numerical features (will be done after transform)
        
        self.is_fitted = True
        self.logger.info("FeatureEngineer fitted successfully")
        
        return self
    
    @timing_decorator
    def transform(self, df: pd.DataFrame) -> Tuple[sparse.csr_matrix, List[str]]:
        """
        Transform the input DataFrame into feature matrix.
        
        Args:
            df: Input DataFrame to transform
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if not self.is_fitted:
            self.logger.warning("FeatureEngineer not fitted. Fitting with current data.")
            self.fit(df)
            
        self.logger.info(f"Transforming {len(df)} records")
        
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Extract different types of features
        feature_matrices = []
        all_feature_names = []
        
        # 1. Text features
        if self.text_column in df_processed.columns:
            text_matrix, text_names = self.engineer_text_features(df_processed)
            feature_matrices.append(text_matrix)
            all_feature_names.extend(text_names)
            self.text_feature_names_ = text_names
            
        # 2. Behavioral features
        behavioral_matrix, behavioral_names = self.engineer_behavioral_features(df_processed)
        if behavioral_matrix.shape[1] > 0:
            feature_matrices.append(behavioral_matrix)
            all_feature_names.extend(behavioral_names)
            self.behavioral_feature_names_ = behavioral_names
            
        # 3. Graph features
        if self.enable_graph_features:
            graph_matrix, graph_names = self.engineer_graph_features(df_processed)
            if graph_matrix.shape[1] > 0:
                feature_matrices.append(graph_matrix)
                all_feature_names.extend(graph_names)
                self.graph_feature_names_ = graph_names
                
        # 4. Sentiment features
        if self.enable_sentiment and self.text_column in df_processed.columns:
            sentiment_matrix, sentiment_names = self.extract_sentiment(df_processed)
            if sentiment_matrix.shape[1] > 0:
                feature_matrices.append(sentiment_matrix)
                all_feature_names.extend(sentiment_names)
                self.sentiment_feature_names_ = sentiment_names
        
        # Concatenate all feature matrices
        if feature_matrices:
            combined_matrix = sparse.hstack(feature_matrices, format='csr')
        else:
            # Return empty matrix if no features
            combined_matrix = sparse.csr_matrix((len(df_processed), 0))
            
        # Store feature names
        self.feature_names_ = all_feature_names
        
        self.logger.info(f"Feature engineering completed. Output shape: {combined_matrix.shape}")
        
        return combined_matrix, all_feature_names
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[sparse.csr_matrix, List[str]]:
        """
        Fit the feature engineer and transform the data in one step.
        
        Args:
            X: Input DataFrame
            y: Target values (optional)
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        return self.fit(X, y).transform(X)
    
    def engineer_text_features(self, df: pd.DataFrame) -> Tuple[sparse.csr_matrix, List[str]]:
        """
        Engineer text features using TF-IDF vectorization.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        self.logger.info("Engineering text features with TF-IDF")
        
        if self.text_column not in df.columns:
            self.logger.warning(f"Text column '{self.text_column}' not found")
            return sparse.csr_matrix((len(df), 0)), []
            
        # Get text data
        texts = df[self.text_column].fillna('').astype(str)
        
        try:
            # Transform using fitted TF-IDF vectorizer
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            
            # Get feature names
            feature_names = [f'tfidf_{name}' for name in self.tfidf_vectorizer.get_feature_names_out()]
            
            self.logger.info(f"Generated {len(feature_names)} TF-IDF features")
            
            return tfidf_matrix, feature_names
            
        except Exception as e:
            self.logger.error(f"Text feature engineering failed: {e}")
            return sparse.csr_matrix((len(df), 0)), []
    
    def engineer_behavioral_features(self, df: pd.DataFrame) -> Tuple[sparse.csr_matrix, List[str]]:
        """
        Engineer behavioral features from user and review patterns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        self.logger.info("Engineering behavioral features")
        
        features = []
        feature_names = []
        
        # Basic text statistics
        if self.text_column in df.columns:
            texts = df[self.text_column].fillna('').astype(str)
            
            # Review length features
            review_lengths = texts.str.len()
            features.append(review_lengths.values.reshape(-1, 1))
            feature_names.append('review_length')
            
            word_counts = texts.str.split().str.len()
            features.append(word_counts.values.reshape(-1, 1))
            feature_names.append('word_count')
            
            # Average word length
            avg_word_lengths = texts.apply(self._compute_avg_word_length)
            features.append(avg_word_lengths.values.reshape(-1, 1))
            feature_names.append('avg_word_length')
            
            # Text complexity features
            uppercase_ratios = texts.str.count(r'[A-Z]') / (texts.str.len() + 1)
            features.append(uppercase_ratios.values.reshape(-1, 1))
            feature_names.append('uppercase_ratio')
            
            digit_ratios = texts.str.count(r'\d') / (texts.str.len() + 1)
            features.append(digit_ratios.values.reshape(-1, 1))
            feature_names.append('digit_ratio')
            
            punctuation_ratios = texts.str.count(r'[^\w\s]') / (texts.str.len() + 1)
            features.append(punctuation_ratios.values.reshape(-1, 1))
            feature_names.append('punctuation_ratio')
            
            # Exclamation and question marks
            exclamation_counts = texts.str.count('!')
            features.append(exclamation_counts.values.reshape(-1, 1))
            feature_names.append('exclamation_count')
            
            question_counts = texts.str.count('?')
            features.append(question_counts.values.reshape(-1, 1))
            feature_names.append('question_count')
        
        # User behavioral features
        if self.user_id_column in df.columns:
            user_features, user_feature_names = self._extract_user_features(df)
            features.extend(user_features)
            feature_names.extend(user_feature_names)
            
        # Product behavioral features
        if self.product_id_column in df.columns:
            product_features, product_feature_names = self._extract_product_features(df)
            features.extend(product_features)
            feature_names.extend(product_feature_names)
            
        # Temporal behavioral features
        if self.date_column in df.columns:
            temporal_features, temporal_feature_names = self._extract_temporal_features(df)
            features.extend(temporal_features)
            feature_names.extend(temporal_feature_names)
            
        # Rating features
        if self.rating_column in df.columns:
            rating_features, rating_feature_names = self._extract_rating_features(df)
            features.extend(rating_features)
            feature_names.extend(rating_feature_names)
        
        # Combine all features
        if features:
            feature_matrix = np.hstack(features)
            
            # Handle missing values
            feature_matrix = self._handle_missing_values(feature_matrix)
            
            # Apply scaling if configured
            if self.scaler is not None and hasattr(self, 'is_fitted') and self.is_fitted:
                try:
                    feature_matrix = self.scaler.transform(feature_matrix)
                except Exception as e:
                    self.logger.warning(f"Scaling failed: {e}")
            elif self.scaler is not None:
                try:
                    feature_matrix = self.scaler.fit_transform(feature_matrix)
                except Exception as e:
                    self.logger.warning(f"Scaling failed: {e}")
                    
            # Convert to sparse matrix
            feature_matrix = sparse.csr_matrix(feature_matrix)
            
            self.logger.info(f"Generated {len(feature_names)} behavioral features")
            
            return feature_matrix, feature_names
        else:
            self.logger.warning("No behavioral features could be generated")
            return sparse.csr_matrix((len(df), 0)), []
    
    def engineer_graph_features(self, df: pd.DataFrame) -> Tuple[sparse.csr_matrix, List[str]]:
        """
        Engineer graph features using NetworkX metrics and optional node2vec.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if not self.enable_graph_features or not NETWORKX_AVAILABLE:
            self.logger.warning("Graph features disabled or NetworkX not available")
            return sparse.csr_matrix((len(df), 0)), []
            
        self.logger.info("Engineering graph features")
        
        try:
            # Build user-product bipartite graph
            if self.user_product_graph is None:
                self.user_product_graph = self._build_user_product_graph(df)
                
            features = []
            feature_names = []
            
            # Extract graph-based features for each review
            for idx, row in df.iterrows():
                user_id = row.get(self.user_id_column)
                product_id = row.get(self.product_id_column)
                
                # User-based graph features
                user_features = self._extract_node_features(self.user_product_graph, user_id, prefix='user')
                features.append(user_features)
                
                # Product-based graph features  
                product_features = self._extract_node_features(self.user_product_graph, product_id, prefix='product')
                
                # Combine user and product features
                if idx == 0:  # Get feature names from first row
                    feature_names.extend([f'user_{name}' for name in ['degree', 'clustering', 'centrality']])
                    feature_names.extend([f'product_{name}' for name in ['degree', 'clustering', 'centrality']])
                    
                    # Add node2vec features if available
                    if hasattr(self, 'node2vec_model') and self.node2vec_model is not None:
                        feature_names.extend([f'user_n2v_{i}' for i in range(self.node2vec_dimensions)])
                        feature_names.extend([f'product_n2v_{i}' for i in range(self.node2vec_dimensions)])
                
                # Add node2vec embeddings if available
                if hasattr(self, 'node2vec_model') and self.node2vec_model is not None:
                    user_embedding = self._get_node2vec_embedding(user_id)
                    product_embedding = self._get_node2vec_embedding(product_id)
                    user_features = np.concatenate([user_features, user_embedding])
                    product_features = np.concatenate([product_features, product_embedding])
                
                combined_features = np.concatenate([user_features, product_features])
                features.append(combined_features)
            
            # Convert to matrix
            if features:
                feature_matrix = np.vstack(features)
                feature_matrix = self._handle_missing_values(feature_matrix)
                feature_matrix = sparse.csr_matrix(feature_matrix)
                
                self.logger.info(f"Generated {len(feature_names)} graph features")
                
                return feature_matrix, feature_names
            else:
                return sparse.csr_matrix((len(df), 0)), []
                
        except Exception as e:
            self.logger.error(f"Graph feature engineering failed: {e}")
            return sparse.csr_matrix((len(df), 0)), []
    
    def extract_sentiment(self, df: pd.DataFrame) -> Tuple[sparse.csr_matrix, List[str]]:
        """
        Extract sentiment features using TextBlob and VADER.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if not self.enable_sentiment:
            self.logger.warning("Sentiment features disabled")
            return sparse.csr_matrix((len(df), 0)), []
            
        self.logger.info("Extracting sentiment features")
        
        if self.text_column not in df.columns:
            self.logger.warning(f"Text column '{self.text_column}' not found")
            return sparse.csr_matrix((len(df), 0)), []
        
        texts = df[self.text_column].fillna('').astype(str)
        features = []
        feature_names = []
        
        # TextBlob sentiment features
        if TEXTBLOB_AVAILABLE:
            self.logger.info("Computing TextBlob sentiment features")
            textblob_features = self._extract_textblob_sentiment(texts)
            features.append(textblob_features)
            feature_names.extend(['textblob_polarity', 'textblob_subjectivity'])
            
        # VADER sentiment features  
        if VADER_AVAILABLE and self.vader_analyzer is not None:
            self.logger.info("Computing VADER sentiment features")
            vader_features = self._extract_vader_sentiment(texts)
            features.append(vader_features)
            feature_names.extend(['vader_compound', 'vader_pos', 'vader_neu', 'vader_neg'])
        
        # Combine sentiment features
        if features:
            feature_matrix = np.hstack(features)
            feature_matrix = self._handle_missing_values(feature_matrix)
            feature_matrix = sparse.csr_matrix(feature_matrix)
            
            self.logger.info(f"Generated {len(feature_names)} sentiment features")
            
            return feature_matrix, feature_names
        else:
            self.logger.warning("No sentiment features could be generated")
            return sparse.csr_matrix((len(df), 0)), []
    
    # Helper methods
    def _fit_text_features(self, df: pd.DataFrame):
        """Fit TF-IDF vectorizer"""
        if self.text_column in df.columns:
            texts = df[self.text_column].fillna('').astype(str)
            self.tfidf_vectorizer.fit(texts)
            self.logger.info(f"TF-IDF fitted with {len(self.tfidf_vectorizer.vocabulary_)} features")
    
    def _fit_graph_features(self, df: pd.DataFrame):
        """Fit graph-based features"""
        if not self.enable_graph_features:
            return
            
        try:
            # Build user-product graph
            self.user_product_graph = self._build_user_product_graph(df)
            
            # Train node2vec if configured
            if Node2Vec is not None and len(self.user_product_graph.nodes()) > 10:
                self.logger.info("Training node2vec model")
                node2vec = Node2Vec(
                    self.user_product_graph,
                    dimensions=self.node2vec_dimensions,
                    walk_length=self.node2vec_walk_length,
                    num_walks=self.node2vec_num_walks,
                    workers=self.node2vec_workers,
                    seed=self.random_state
                )
                self.node2vec_model = node2vec.fit(window=10, min_count=1, batch_words=4)
                
        except Exception as e:
            self.logger.error(f"Graph fitting failed: {e}")
            self.enable_graph_features = False
    
    def _build_user_product_graph(self, df: pd.DataFrame) -> 'nx.Graph':
        """Build bipartite user-product graph"""
        G = nx.Graph()
        
        for _, row in df.iterrows():
            user_id = f"user_{row[self.user_id_column]}"
            product_id = f"product_{row[self.product_id_column]}"
            
            # Add nodes with type attribute
            G.add_node(user_id, bipartite=0, node_type='user')
            G.add_node(product_id, bipartite=1, node_type='product')
            
            # Add edge (review relationship)
            if G.has_edge(user_id, product_id):
                G[user_id][product_id]['weight'] += 1
            else:
                G.add_edge(user_id, product_id, weight=1)
                
        self.logger.info(f"Built graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
        return G
    
    def _extract_node_features(self, graph: 'nx.Graph', node_id: str, prefix: str = '') -> np.ndarray:
        """Extract features for a specific node"""
        if prefix:
            node_id = f"{prefix}_{node_id}"
            
        features = np.zeros(3)  # degree, clustering, centrality
        
        if node_id in graph:
            # Degree
            features[0] = graph.degree(node_id)
            
            # Clustering coefficient
            try:
                features[1] = nx.clustering(graph, node_id)
            except:
                features[1] = 0
                
            # Betweenness centrality (approximate for large graphs)
            try:
                centrality = nx.betweenness_centrality(graph, k=min(100, len(graph.nodes())))
                features[2] = centrality.get(node_id, 0)
            except:
                features[2] = 0
                
        return features
    
    def _get_node2vec_embedding(self, node_id: str) -> np.ndarray:
        """Get node2vec embedding for a node"""
        try:
            if self.node2vec_model and node_id in self.node2vec_model.wv:
                return self.node2vec_model.wv[node_id]
        except:
            pass
        return np.zeros(self.node2vec_dimensions)
    
    def _extract_user_features(self, df: pd.DataFrame) -> Tuple[List[np.ndarray], List[str]]:
        """Extract user-level behavioral features"""
        features = []
        feature_names = []
        
        # Calculate user statistics
        user_stats = df.groupby(self.user_id_column).agg({
            self.text_column: ['count'] if self.text_column in df.columns else [],
            self.rating_column: ['mean', 'std'] if self.rating_column in df.columns else [],
            self.date_column: ['nunique'] if self.date_column in df.columns else []
        }).reset_index()
        
        # Flatten column names
        user_stats.columns = ['_'.join(col).strip() if col[1] else col[0] 
                             for col in user_stats.columns.values]
        user_stats = user_stats.rename(columns={f'{self.user_id_column}_': self.user_id_column})
        
        # Merge back to original dataframe
        df_with_user_stats = df.merge(user_stats, on=self.user_id_column, how='left')
        
        # Extract user feature columns
        user_feature_cols = [col for col in df_with_user_stats.columns 
                           if col not in df.columns]
        
        for col in user_feature_cols:
            values = df_with_user_stats[col].fillna(0).values
            features.append(values.reshape(-1, 1))
            feature_names.append(f'user_{col}')
            
        return features, feature_names
    
    def _extract_product_features(self, df: pd.DataFrame) -> Tuple[List[np.ndarray], List[str]]:
        """Extract product-level behavioral features"""
        features = []
        feature_names = []
        
        # Calculate product statistics
        product_stats = df.groupby(self.product_id_column).agg({
            self.text_column: ['count'] if self.text_column in df.columns else [],
            self.rating_column: ['mean', 'std'] if self.rating_column in df.columns else [],
            self.user_id_column: ['nunique'] if self.user_id_column in df.columns else []
        }).reset_index()
        
        # Flatten column names
        product_stats.columns = ['_'.join(col).strip() if col[1] else col[0] 
                               for col in product_stats.columns.values]
        product_stats = product_stats.rename(columns={f'{self.product_id_column}_': self.product_id_column})
        
        # Merge back to original dataframe
        df_with_product_stats = df.merge(product_stats, on=self.product_id_column, how='left')
        
        # Extract product feature columns
        product_feature_cols = [col for col in df_with_product_stats.columns 
                              if col not in df.columns]
        
        for col in product_feature_cols:
            values = df_with_product_stats[col].fillna(0).values
            features.append(values.reshape(-1, 1))
            feature_names.append(f'product_{col}')
            
        return features, feature_names
    
    def _extract_temporal_features(self, df: pd.DataFrame) -> Tuple[List[np.ndarray], List[str]]:
        """Extract temporal behavioral features"""
        features = []
        feature_names = []
        
        try:
            # Convert date column to datetime
            dates = pd.to_datetime(df[self.date_column], errors='coerce')
            
            # Hour of day
            hours = dates.dt.hour.fillna(12).values  # Default to noon
            features.append(hours.reshape(-1, 1))
            feature_names.append('review_hour')
            
            # Day of week
            day_of_week = dates.dt.dayofweek.fillna(0).values
            features.append(day_of_week.reshape(-1, 1))
            feature_names.append('review_day_of_week')
            
            # Is weekend
            is_weekend = (dates.dt.dayofweek >= 5).fillna(False).astype(int).values
            features.append(is_weekend.reshape(-1, 1))
            feature_names.append('is_weekend')
            
            # Days since epoch (for temporal trends)
            days_since_epoch = (dates - pd.Timestamp('2000-01-01')).dt.days.fillna(0).values
            features.append(days_since_epoch.reshape(-1, 1))
            feature_names.append('days_since_epoch')
            
        except Exception as e:
            self.logger.warning(f"Temporal feature extraction failed: {e}")
            
        return features, feature_names
    
    def _extract_rating_features(self, df: pd.DataFrame) -> Tuple[List[np.ndarray], List[str]]:
        """Extract rating-based features"""
        features = []
        feature_names = []
        
        ratings = pd.to_numeric(df[self.rating_column], errors='coerce').fillna(3)  # Default to 3
        
        # Rating value
        features.append(ratings.values.reshape(-1, 1))
        feature_names.append('rating_value')
        
        # Is extreme rating (1 or 5)
        is_extreme = ratings.isin([1, 5]).astype(int).values
        features.append(is_extreme.reshape(-1, 1))
        feature_names.append('is_extreme_rating')
        
        return features, feature_names
    
    def _extract_textblob_sentiment(self, texts: pd.Series) -> np.ndarray:
        """Extract TextBlob sentiment features"""
        polarities = []
        subjectivities = []
        
        for text in texts:
            try:
                blob = TextBlob(str(text))
                polarities.append(blob.sentiment.polarity)
                subjectivities.append(blob.sentiment.subjectivity)
            except:
                polarities.append(0.0)
                subjectivities.append(0.0)
                
        return np.column_stack([polarities, subjectivities])
    
    def _extract_vader_sentiment(self, texts: pd.Series) -> np.ndarray:
        """Extract VADER sentiment features"""
        compounds = []
        positives = []
        neutrals = []
        negatives = []
        
        for text in texts:
            try:
                scores = self.vader_analyzer.polarity_scores(str(text))
                compounds.append(scores['compound'])
                positives.append(scores['pos'])
                neutrals.append(scores['neu'])
                negatives.append(scores['neg'])
            except:
                compounds.append(0.0)
                positives.append(0.0)
                neutrals.append(0.0)
                negatives.append(0.0)
                
        return np.column_stack([compounds, positives, neutrals, negatives])
    
    def _compute_avg_word_length(self, text: str) -> float:
        """Compute average word length"""
        try:
            words = str(text).split()
            if len(words) == 0:
                return 0.0
            return sum(len(word) for word in words) / len(words)
        except:
            return 0.0
    
    def _handle_missing_values(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Handle missing values in feature matrix"""
        if self.handle_missing == 'mean':
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            return imputer.fit_transform(feature_matrix)
        elif self.handle_missing == 'median':
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            return imputer.fit_transform(feature_matrix)
        elif self.handle_missing == 'zero':
            return np.nan_to_num(feature_matrix, nan=0.0)
        else:
            return feature_matrix
    
    def get_feature_names(self) -> List[str]:
        """Get names of all engineered features"""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted first")
        return self.feature_names_.copy()
    
    def get_feature_names_by_type(self) -> Dict[str, List[str]]:
        """Get feature names grouped by type"""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted first")
            
        return {
            'text': self.text_feature_names_.copy(),
            'behavioral': self.behavioral_feature_names_.copy(),
            'graph': self.graph_feature_names_.copy(),
            'sentiment': self.sentiment_feature_names_.copy()
        }


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'review_text': [
            "This product is absolutely amazing! Love it so much!",
            "Terrible quality. Would not recommend to anyone.",
            "Good value for money. Works as expected, nothing special.",
            "Best purchase ever! 5 stars all the way!",
            "Poor customer service and bad product quality overall.",
            "Decent product, shipping was fast and packaging good.",
            "Not worth the money. Very disappointed with purchase.",
            "Exceeded my expectations! Will definitely buy again!"
        ],
        'rating': [5, 1, 4, 5, 2, 3, 1, 5],
        'user_id': ['user1', 'user2', 'user3', 'user1', 'user4', 'user2', 'user5', 'user3'],
        'product_id': ['prod1', 'prod1', 'prod2', 'prod3', 'prod1', 'prod2', 'prod4', 'prod3'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-04', 
                '2023-01-05', '2023-01-06', '2023-01-07']
    })
    
    # Initialize and test feature engineer
    feature_engineer = FeatureEngineer(
        enable_graph_features=True,
        enable_sentiment=True,
        feature_scaling='standard'
    )
    
    # Transform data
    try:
        feature_matrix, feature_names = feature_engineer.fit_transform(sample_data)
        
        print("Feature engineering completed successfully!")
        print(f"Feature matrix shape: {feature_matrix.shape}")
        print(f"Number of features: {len(feature_names)}")
        print(f"Feature matrix type: {type(feature_matrix)}")
        print(f"Is sparse: {sparse.issparse(feature_matrix)}")
        
        # Show feature names by type
        feature_types = feature_engineer.get_feature_names_by_type()
        for feature_type, names in feature_types.items():
            if names:
                print(f"\n{feature_type.title()} features ({len(names)}):")
                print(names[:5])  # Show first 5 features of each type
                if len(names) > 5:
                    print(f"... and {len(names) - 5} more")
                    
    except Exception as e:
        print(f"Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
