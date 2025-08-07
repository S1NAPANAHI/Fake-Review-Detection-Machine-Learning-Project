# Feature Engineering Documentation

This document describes the feature engineering pipeline used in the Fake Review Detection System, including text processing, behavioral analysis, and network-based features.

## Overview

The feature engineering pipeline transforms raw review data into meaningful numerical representations that machine learning models can use to detect fake reviews. Our approach combines multiple types of features:

1. **Text Features**: NLP-based features from review content
2. **Behavioral Features**: User and review behavior patterns  
3. **Network Features**: Graph-based relationships between users and reviews
4. **Temporal Features**: Time-based patterns and trends

## Text Features

### 1. TF-IDF Features

Term Frequency-Inverse Document Frequency captures the importance of words in reviews.

#### Configuration
```python
TfidfVectorizer(
    max_features=10000,      # Top 10k most important features
    ngram_range=(1, 2),      # Unigrams and bigrams
    min_df=2,                # Ignore terms in less than 2 documents
    max_df=0.95,             # Ignore terms in more than 95% of documents
    stop_words='english',    # Remove common English stop words
    lowercase=True,          # Convert to lowercase
    token_pattern=r'(?u)\b\w\w+\b'  # Only words with 2+ characters
)
```

#### Implementation
```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_tfidf_features(texts, vectorizer=None, fit=False):
    """Extract TF-IDF features from texts."""
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
    
    if fit:
        tfidf_matrix = vectorizer.fit_transform(texts)
    else:
        tfidf_matrix = vectorizer.transform(texts)
    
    return tfidf_matrix.toarray(), vectorizer
```

### 2. Word Embeddings

Pre-trained transformer models provide semantic representations of review text.

#### Sentence Transformers
```python
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingExtractor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def extract_embeddings(self, texts, batch_size=32):
        """Extract sentence embeddings."""
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def get_embedding_size(self):
        return self.model.get_sentence_embedding_dimension()
```

#### Word2Vec Features
```python
from gensim.models import Word2Vec
import numpy as np

def train_word2vec(texts, vector_size=100, window=5, min_count=2):
    """Train Word2Vec model on review texts."""
    sentences = [text.split() for text in texts]
    
    model = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=1  # Skip-gram
    )
    
    return model

def get_word2vec_features(text, model, vector_size=100):
    """Get average Word2Vec representation of text."""
    words = text.split()
    word_vectors = []
    
    for word in words:
        if word in model.wv:
            word_vectors.append(model.wv[word])
    
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(vector_size)
```

### 3. Sentiment Features

Sentiment analysis provides emotional context of reviews.

#### VADER Sentiment
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

def extract_vader_sentiment(texts):
    """Extract VADER sentiment scores."""
    analyzer = SentimentIntensityAnalyzer()
    
    results = []
    for text in texts:
        scores = analyzer.polarity_scores(text)
        results.append({
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        })
    
    return pd.DataFrame(results)
```

#### TextBlob Sentiment
```python
from textblob import TextBlob

def extract_textblob_sentiment(texts):
    """Extract TextBlob sentiment features."""
    results = []
    
    for text in texts:
        blob = TextBlob(text)
        results.append({
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        })
    
    return pd.DataFrame(results)
```

### 4. Linguistic Features

Various text-based linguistic patterns that may indicate fake reviews.

```python
import re
import string
from textstat import flesch_reading_ease, flesch_kincaid_grade

def extract_linguistic_features(texts):
    """Extract linguistic features from texts."""
    features = []
    
    for text in texts:
        feature_dict = {}
        
        # Basic statistics
        feature_dict['char_count'] = len(text)
        feature_dict['word_count'] = len(text.split())
        feature_dict['sentence_count'] = len(re.findall(r'[.!?]+', text))
        feature_dict['avg_word_length'] = np.mean([len(word) for word in text.split()])
        
        # Punctuation features
        feature_dict['exclamation_count'] = text.count('!')
        feature_dict['question_count'] = text.count('?')
        feature_dict['comma_count'] = text.count(',')
        feature_dict['period_count'] = text.count('.')
        
        # Capital letters
        feature_dict['capital_ratio'] = sum(c.isupper() for c in text) / len(text) if text else 0
        feature_dict['all_caps_words'] = len([w for w in text.split() if w.isupper() and len(w) > 1])
        
        # Readability
        try:
            feature_dict['flesch_reading_ease'] = flesch_reading_ease(text)
            feature_dict['flesch_kincaid_grade'] = flesch_kincaid_grade(text)
        except:
            feature_dict['flesch_reading_ease'] = 0
            feature_dict['flesch_kincaid_grade'] = 0
        
        # Special characters
        feature_dict['digit_ratio'] = sum(c.isdigit() for c in text) / len(text) if text else 0
        feature_dict['special_char_ratio'] = sum(c in string.punctuation for c in text) / len(text) if text else 0
        
        features.append(feature_dict)
    
    return pd.DataFrame(features)
```

## Behavioral Features

### 1. User-Level Features

Features that capture user behavior patterns across all their reviews.

```python
def extract_user_features(df):
    """Extract user-level behavioral features."""
    user_features = df.groupby('user_id').agg({
        'rating': ['count', 'mean', 'std', 'min', 'max'],
        'review_length': ['mean', 'std'],
        'timestamp': ['min', 'max'],
        'verified_purchase': 'mean',
        'helpful_votes': ['sum', 'mean']
    }).round(4)
    
    # Flatten column names
    user_features.columns = ['_'.join(col).strip() for col in user_features.columns]
    
    # Calculate additional features
    user_features['rating_range'] = user_features['rating_max'] - user_features['rating_min']
    user_features['account_age_days'] = (
        pd.to_datetime(user_features['timestamp_max']) - 
        pd.to_datetime(user_features['timestamp_min'])
    ).dt.days
    
    # Activity patterns
    user_features['reviews_per_day'] = (
        user_features['rating_count'] / (user_features['account_age_days'] + 1)
    )
    
    return user_features
```

### 2. Review-Level Features

Features specific to individual reviews.

```python
def extract_review_features(df):
    """Extract review-level features."""
    features = df.copy()
    
    # Rating patterns
    product_avg_rating = df.groupby('product_id')['rating'].mean()
    features['rating_deviation'] = abs(
        features['rating'] - features['product_id'].map(product_avg_rating)
    )
    
    # Review timing
    features['review_hour'] = pd.to_datetime(features['timestamp']).dt.hour
    features['review_day_of_week'] = pd.to_datetime(features['timestamp']).dt.dayofweek
    features['is_weekend'] = features['review_day_of_week'].isin([5, 6]).astype(int)
    
    # Review position (early vs late reviewer)
    product_review_dates = df.groupby('product_id')['timestamp'].min()
    features['days_since_first_review'] = (
        pd.to_datetime(features['timestamp']) - 
        features['product_id'].map(product_review_dates)
    ).dt.days
    
    # Review density
    features['reviews_same_day'] = df.groupby([
        'user_id', 
        pd.to_datetime(df['timestamp']).dt.date
    ])['review_id'].transform('count')
    
    return features
```

### 3. Temporal Features

Time-based patterns that may indicate coordinated fake review campaigns.

```python
def extract_temporal_features(df):
    """Extract temporal behavior features."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Time-based aggregations
    temporal_features = []
    
    for user_id, user_data in df.groupby('user_id'):
        user_data = user_data.sort_values('timestamp')
        
        feature_dict = {'user_id': user_id}
        
        # Review frequency patterns
        if len(user_data) > 1:
            time_diffs = user_data['timestamp'].diff().dt.total_seconds() / 3600  # hours
            feature_dict['avg_time_between_reviews'] = time_diffs.mean()
            feature_dict['std_time_between_reviews'] = time_diffs.std()
            feature_dict['min_time_between_reviews'] = time_diffs.min()
        else:
            feature_dict['avg_time_between_reviews'] = 0
            feature_dict['std_time_between_reviews'] = 0
            feature_dict['min_time_between_reviews'] = 0
        
        # Burst detection (multiple reviews in short time)
        feature_dict['reviews_in_1_hour'] = (time_diffs < 1).sum()
        feature_dict['reviews_in_24_hours'] = (time_diffs < 24).sum()
        
        # Activity patterns
        hours = user_data['timestamp'].dt.hour
        feature_dict['most_active_hour'] = hours.mode().iloc[0] if len(hours.mode()) > 0 else 0
        feature_dict['hour_diversity'] = hours.nunique()
        
        temporal_features.append(feature_dict)
    
    return pd.DataFrame(temporal_features)
```

## Network Features

### 1. User Similarity

Features based on similarities between users.

```python
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np

def calculate_user_similarity(df, feature_cols):
    """Calculate user similarity matrix."""
    user_features = df.groupby('user_id')[feature_cols].mean()
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(user_features)
    
    # Create similarity features
    similarity_features = []
    user_ids = user_features.index.tolist()
    
    for i, user_id in enumerate(user_ids):
        similarities = similarity_matrix[i]
        # Exclude self-similarity
        other_similarities = np.concatenate([similarities[:i], similarities[i+1:]])
        
        feature_dict = {
            'user_id': user_id,
            'max_similarity': other_similarities.max(),
            'avg_similarity': other_similarities.mean(),
            'similar_users_count': (other_similarities > 0.8).sum()
        }
        similarity_features.append(feature_dict)
    
    return pd.DataFrame(similarity_features)
```

### 2. Review Similarity

Features based on similarities between review texts.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_review_similarity(df, text_col='review_text'):
    """Calculate review similarity features."""
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df[text_col])
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    similarity_features = []
    
    for i, review_id in enumerate(df['review_id']):
        similarities = similarity_matrix[i]
        # Exclude self-similarity
        other_similarities = np.concatenate([similarities[:i], similarities[i+1:]])
        
        feature_dict = {
            'review_id': review_id,
            'max_text_similarity': other_similarities.max(),
            'avg_text_similarity': other_similarities.mean(),
            'similar_reviews_count': (other_similarities > 0.7).sum(),
            'highly_similar_reviews': (other_similarities > 0.9).sum()
        }
        similarity_features.append(feature_dict)
    
    return pd.DataFrame(similarity_features)
```

### 3. Graph Features

Network-based features using graph analysis.

```python
import networkx as nx

def create_user_product_graph(df):
    """Create bipartite graph of users and products."""
    G = nx.Graph()
    
    # Add user nodes
    users = df['user_id'].unique()
    G.add_nodes_from(users, bipartite=0)
    
    # Add product nodes
    products = df['product_id'].unique()
    G.add_nodes_from(products, bipartite=1)
    
    # Add edges (user reviewed product)
    edges = [(row['user_id'], row['product_id']) for _, row in df.iterrows()]
    G.add_edges_from(edges)
    
    return G

def extract_graph_features(df):
    """Extract graph-based features."""
    G = create_user_product_graph(df)
    
    graph_features = []
    
    for user_id in df['user_id'].unique():
        feature_dict = {'user_id': user_id}
        
        if user_id in G:
            # Node centrality measures
            feature_dict['degree_centrality'] = nx.degree_centrality(G)[user_id]
            feature_dict['betweenness_centrality'] = nx.betweenness_centrality(G)[user_id]
            feature_dict['closeness_centrality'] = nx.closeness_centrality(G)[user_id]
            
            # Local clustering
            feature_dict['clustering_coefficient'] = nx.clustering(G, user_id)
            
            # Neighborhood features
            neighbors = list(G.neighbors(user_id))
            feature_dict['neighbor_count'] = len(neighbors)
            
            if neighbors:
                # Average degree of neighbors
                neighbor_degrees = [G.degree(n) for n in neighbors]
                feature_dict['avg_neighbor_degree'] = np.mean(neighbor_degrees)
            else:
                feature_dict['avg_neighbor_degree'] = 0
        else:
            # Default values for isolated nodes
            for key in ['degree_centrality', 'betweenness_centrality', 
                       'closeness_centrality', 'clustering_coefficient']:
                feature_dict[key] = 0
            feature_dict['neighbor_count'] = 0
            feature_dict['avg_neighbor_degree'] = 0
        
        graph_features.append(feature_dict)
    
    return pd.DataFrame(graph_features)
```

## Feature Pipeline

### 1. Main Pipeline

Complete feature extraction pipeline that combines all feature types.

```python
class FeaturePipeline:
    def __init__(self, config):
        self.config = config
        self.fitted = False
        self.vectorizers = {}
        self.scalers = {}
        
    def fit_transform(self, df):
        """Fit feature extractors and transform data."""
        features = []
        
        # Text features
        if self.config.get('use_tfidf', True):
            tfidf_features, self.vectorizers['tfidf'] = extract_tfidf_features(
                df['review_text'], fit=True
            )
            features.append(tfidf_features)
        
        if self.config.get('use_embeddings', True):
            embedding_extractor = EmbeddingExtractor()
            embedding_features = embedding_extractor.extract_embeddings(df['review_text'])
            features.append(embedding_features)
        
        # Sentiment features
        if self.config.get('use_sentiment', True):
            sentiment_features = extract_vader_sentiment(df['review_text'])
            features.append(sentiment_features.values)
        
        # Linguistic features
        if self.config.get('use_linguistic', True):
            linguistic_features = extract_linguistic_features(df['review_text'])
            features.append(linguistic_features.values)
        
        # Behavioral features
        if self.config.get('use_behavioral', True):
            user_features = extract_user_features(df)
            review_features = extract_review_features(df)
            temporal_features = extract_temporal_features(df)
            
            # Merge behavioral features
            behavioral_df = df.merge(user_features, on='user_id', how='left')
            behavioral_df = behavioral_df.merge(temporal_features, on='user_id', how='left')
            
            behavioral_cols = (list(user_features.columns) + 
                             list(temporal_features.columns)[1:])  # Exclude user_id
            features.append(behavioral_df[behavioral_cols].fillna(0).values)
        
        # Network features
        if self.config.get('use_network', True):
            similarity_features = calculate_user_similarity(df, ['rating'])
            graph_features = extract_graph_features(df)
            
            network_df = df.merge(similarity_features, on='user_id', how='left')
            network_df = network_df.merge(graph_features, on='user_id', how='left')
            
            network_cols = (list(similarity_features.columns)[1:] + 
                           list(graph_features.columns)[1:])
            features.append(network_df[network_cols].fillna(0).values)
        
        # Combine all features
        combined_features = np.hstack(features)
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        self.scalers['main'] = StandardScaler()
        scaled_features = self.scalers['main'].fit_transform(combined_features)
        
        self.fitted = True
        return scaled_features
    
    def transform(self, df):
        """Transform new data using fitted extractors."""
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        # Similar to fit_transform but using fitted extractors
        features = []
        
        # Text features
        if self.config.get('use_tfidf', True):
            tfidf_features, _ = extract_tfidf_features(
                df['review_text'], 
                vectorizer=self.vectorizers['tfidf'], 
                fit=False
            )
            features.append(tfidf_features)
        
        # ... (similar pattern for other features)
        
        # Combine and scale
        combined_features = np.hstack(features)
        scaled_features = self.scalers['main'].transform(combined_features)
        
        return scaled_features
```

### 2. Feature Selection

Automated feature selection to improve model performance and reduce overfitting.

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

def select_features(X, y, method='univariate', k=1000):
    """Select top k features using specified method."""
    
    if method == 'univariate':
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        
    elif method == 'rfe':
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        selector = RFE(estimator, n_features_to_select=k)
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        
    elif method == 'importance':
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        selected_indices = np.argsort(importances)[-k:]
        X_selected = X[:, selected_indices]
    
    return X_selected, selected_indices, selector
```

### 3. Feature Engineering Configuration

YAML configuration for feature engineering pipeline.

```yaml
# Feature Engineering Configuration
features:
  # Text features
  text:
    use_tfidf: true
    tfidf_max_features: 10000
    tfidf_ngram_range: [1, 2]
    
    use_embeddings: true
    embedding_model: "all-MiniLM-L6-v2"
    
    use_sentiment: true
    sentiment_models: ["vader", "textblob"]
    
    use_linguistic: true
    linguistic_features: [
      "char_count", "word_count", "sentence_count",
      "exclamation_count", "capital_ratio", "digit_ratio"
    ]
  
  # Behavioral features
  behavioral:
    use_user_features: true
    use_review_features: true
    use_temporal_features: true
    
    user_aggregations: ["count", "mean", "std", "min", "max"]
    temporal_windows: [1, 24, 168]  # hours
  
  # Network features
  network:
    use_similarity: true
    similarity_threshold: 0.8
    
    use_graph: true
    graph_features: [
      "degree_centrality", "betweenness_centrality",
      "clustering_coefficient"
    ]
  
  # Feature selection
  selection:
    enabled: true
    method: "importance"  # univariate, rfe, importance
    k_features: 1000
  
  # Preprocessing
  preprocessing:
    scale_features: true
    handle_missing: "zero"  # zero, mean, median
    remove_outliers: false
```

## Feature Monitoring

### 1. Feature Drift Detection

Monitor feature distributions over time to detect data drift.

```python
from scipy import stats
import pandas as pd

def detect_feature_drift(reference_features, current_features, threshold=0.05):
    """Detect drift in feature distributions."""
    drift_results = []
    
    for i in range(reference_features.shape[1]):
        ref_feature = reference_features[:, i]
        cur_feature = current_features[:, i]
        
        # Kolmogorov-Smirnov test
        ks_statistic, ks_p_value = stats.ks_2samp(ref_feature, cur_feature)
        
        # Population Stability Index (PSI)
        psi = calculate_psi(ref_feature, cur_feature)
        
        drift_results.append({
            'feature_index': i,
            'ks_statistic': ks_statistic,
            'ks_p_value': ks_p_value,
            'psi': psi,
            'drift_detected': ks_p_value < threshold or psi > 0.25
        })
    
    return pd.DataFrame(drift_results)

def calculate_psi(expected, actual, buckets=10):
    """Calculate Population Stability Index."""
    # Create buckets based on expected distribution
    breakpoints = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    
    # Calculate distributions
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    # Normalize to percentages
    expected_pct = expected_counts / len(expected) + 1e-6
    actual_pct = actual_counts / len(actual) + 1e-6
    
    # Calculate PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return psi
```

### 2. Feature Quality Metrics

Monitor feature quality and importance over time.

```python
def calculate_feature_quality(X, y, feature_names):
    """Calculate quality metrics for features."""
    quality_metrics = []
    
    for i, feature_name in enumerate(feature_names):
        feature = X[:, i]
        
        metrics = {
            'feature_name': feature_name,
            'missing_rate': np.isnan(feature).mean(),
            'zero_rate': (feature == 0).mean(),
            'unique_values': len(np.unique(feature)),
            'std': np.std(feature),
            'skewness': stats.skew(feature),
            'kurtosis': stats.kurtosis(feature)
        }
        
        # Correlation with target
        if len(np.unique(y)) == 2:  # Binary classification
            correlation = np.corrcoef(feature, y)[0, 1]
            metrics['target_correlation'] = correlation
        
        quality_metrics.append(metrics)
    
    return pd.DataFrame(quality_metrics)
```

---

This feature engineering pipeline provides a comprehensive approach to extracting meaningful patterns from review data for fake review detection. The modular design allows for easy experimentation and customization based on specific requirements and data characteristics.
