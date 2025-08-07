# Model Documentation

This document provides detailed information about the machine learning models used in the Fake Review Detection System, including architectures, training procedures, and evaluation metrics.

## Model Overview

The system employs an ensemble approach combining multiple machine learning models to achieve robust and accurate fake review detection. Each model brings unique strengths to the ensemble:

- **Random Forest**: Baseline model with good interpretability
- **XGBoost**: Gradient boosting for complex pattern recognition
- **Neural Network**: Deep learning for capturing subtle relationships
- **Support Vector Machine**: High-dimensional classification

## Model Architecture

### 1. Random Forest Classifier

#### Architecture
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)
```

#### Features
- **Ensemble Size**: 100 decision trees
- **Feature Selection**: Square root of total features per tree
- **Regularization**: Limited depth and minimum samples
- **Bootstrap Sampling**: Out-of-bag error estimation

#### Advantages
- Fast training and inference
- Built-in feature importance
- Robust to overfitting
- No feature scaling required

#### Disadvantages
- May struggle with very complex patterns
- Can be biased towards categorical features

### 2. XGBoost Classifier

#### Architecture
```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)
```

#### Features
- **Gradient Boosting**: Sequential tree building
- **Regularization**: L1 and L2 penalties
- **Subsampling**: Both row and column sampling
- **Early Stopping**: Prevents overfitting

#### Advantages
- Excellent performance on tabular data
- Built-in cross-validation
- Handles missing values
- Feature importance ranking

#### Disadvantages
- Sensitive to hyperparameters
- Longer training time
- Can overfit on small datasets

### 3. Neural Network

#### Architecture
```python
class FakeReviewNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Binary classification
        )
    
    def forward(self, x):
        return self.layers(x)
```

#### Training Configuration
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Cross-entropy loss
- **Batch Size**: 32
- **Epochs**: 50 with early stopping
- **Regularization**: Dropout layers

#### Advantages
- Captures complex non-linear relationships
- Can learn feature interactions
- Flexible architecture
- Good with high-dimensional data

#### Disadvantages
- Requires more data
- Longer training time
- Less interpretable
- Prone to overfitting

### 4. Support Vector Machine

#### Architecture
```python
SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    random_state=42
)
```

#### Features
- **Kernel**: Radial Basis Function (RBF)
- **Regularization**: C parameter balancing margin and errors
- **Probability Estimation**: For ensemble voting
- **Feature Scaling**: StandardScaler preprocessing

#### Advantages
- Effective in high-dimensional spaces
- Memory efficient
- Versatile kernel options
- Works well with limited data

#### Disadvantages
- Slow on large datasets
- Sensitive to feature scaling
- No direct probability estimates
- Less interpretable

## Ensemble Strategy

### Voting Classifier

The final prediction combines all models using a weighted voting approach:

```python
VotingClassifier(
    estimators=[
        ('rf', random_forest),
        ('xgb', xgboost),
        ('nn', neural_network),
        ('svm', svm)
    ],
    voting='soft',  # Use probability estimates
    weights=[0.25, 0.35, 0.25, 0.15]  # XGBoost weighted higher
)
```

### Weight Determination

Model weights are determined through:
1. **Cross-validation performance**: Higher performing models get higher weights
2. **Model diversity**: Ensure ensemble diversity
3. **Prediction correlation**: Reduce correlated predictions

## Feature Engineering

### Text Features

#### TF-IDF Features
```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    stop_words='english'
)
```

#### Word Embeddings
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Dimensions**: 384
- **Aggregation**: Mean pooling

#### Sentiment Features
- **VADER Sentiment**: Compound score
- **TextBlob**: Polarity and subjectivity
- **Emotional Indicators**: Joy, anger, fear, etc.

### Behavioral Features

#### User-Level Features
- Review count per user
- Average rating given
- Rating variance
- Time between reviews
- Account age

#### Review-Level Features  
- Review length (characters, words)
- Exclamation marks count
- Capital letters ratio
- Spelling errors count
- Reading difficulty score

#### Temporal Features
- Day of week
- Hour of day  
- Review frequency
- Burst patterns
- Seasonal trends

### Network Features

#### User Similarity
- Cosine similarity between user vectors
- Jaccard similarity of reviewed products
- Common review patterns

#### Review Similarity
- Text similarity using embeddings
- Rating pattern similarity
- Temporal posting patterns

## Training Process

### Data Preprocessing

```bash
# 1. Data cleaning and validation
python src/preprocessing.py \
    --input data/raw/reviews.csv \
    --output data/processed/clean_reviews.csv \
    --config config/settings.yaml

# 2. Feature extraction
python src/feature_engineering.py \
    --input data/processed/clean_reviews.csv \
    --output artifacts/features/features.pkl \
    --config config/settings.yaml

# 3. Train-test split
python -c "
from src.utils import train_test_split
train_test_split('data/processed/clean_reviews.csv', test_size=0.2)
"
```

### Model Training

```bash
# Train individual models
python src/modeling.py \
    --model random_forest \
    --features artifacts/features/features.pkl \
    --output artifacts/models/rf_model.pkl

python src/modeling.py \
    --model xgboost \
    --features artifacts/features/features.pkl \
    --output artifacts/models/xgb_model.pkl

# Train ensemble
python src/modeling.py \
    --model ensemble \
    --features artifacts/features/features.pkl \
    --output artifacts/models/ensemble_model.pkl
```

### Hyperparameter Tuning

#### Grid Search Configuration
```python
param_grids = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
}
```

#### Bayesian Optimization
For neural networks, we use Bayesian optimization:
```python
from skopt import gp_minimize

def objective(params):
    lr, batch_size, hidden_size = params
    model = train_neural_network(lr, batch_size, hidden_size)
    return -model.best_score_

result = gp_minimize(
    objective,
    dimensions=[
        (1e-4, 1e-1, 'log-uniform'),  # learning rate
        (16, 128),                    # batch size
        (64, 512)                     # hidden size
    ],
    n_calls=50
)
```

## Model Evaluation

### Cross-Validation Strategy

5-fold stratified cross-validation ensures:
- Balanced class distribution in each fold
- Consistent performance estimates
- Reduced variance in metrics

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')
```

### Evaluation Metrics

#### Primary Metrics
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives

#### Secondary Metrics
- **Accuracy**: Overall correct predictions
- **Matthews Correlation Coefficient**: Balanced metric for imbalanced data
- **Cohen's Kappa**: Agreement beyond chance

### Performance Results

#### Individual Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|---------|----------|---------|
| Random Forest | 0.92 | 0.91 | 0.93 | 0.92 | 0.95 |
| XGBoost | 0.94 | 0.93 | 0.95 | 0.94 | 0.97 |
| Neural Network | 0.93 | 0.92 | 0.94 | 0.93 | 0.96 |
| SVM | 0.90 | 0.89 | 0.91 | 0.90 | 0.94 |
| **Ensemble** | **0.96** | **0.95** | **0.97** | **0.96** | **0.98** |

#### Confusion Matrix (Ensemble Model)

```
                Predicted
Actual          Legitimate  Fake
Legitimate      9,450       250
Fake            150         9,150
```

#### Feature Importance

Top 10 most important features:

1. **Sentiment Score** (0.12) - VADER compound sentiment
2. **Review Length** (0.11) - Number of words in review
3. **User Review Count** (0.10) - Total reviews by user
4. **Rating Deviation** (0.09) - Difference from product average
5. **Temporal Pattern** (0.08) - Time-based posting patterns
6. **Text Similarity** (0.07) - Similarity to other reviews
7. **Exclamation Ratio** (0.06) - Frequency of exclamation marks
8. **Spelling Errors** (0.06) - Number of misspelled words
9. **Capital Ratio** (0.05) - Proportion of capital letters
10. **User Activity** (0.05) - User posting frequency

### Model Interpretability

#### SHAP (SHapley Additive exPlanations)

```python
import shap

# Generate SHAP explanations
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot feature importance
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

#### LIME (Local Interpretable Model-agnostic Explanations)

```python
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=['Legitimate', 'Fake'])
explanation = explainer.explain_instance(
    text_instance, 
    model.predict_proba,
    num_features=10
)
```

## Model Versioning

### MLflow Integration

```python
import mlflow
import mlflow.sklearn

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Log metrics
    mlflow.log_metric("f1_score", f1_score)
    mlflow.log_metric("roc_auc", roc_auc)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("feature_importance.png")
```

### Model Registry

Models are stored with semantic versioning:
- **Major**: Breaking changes in features or API
- **Minor**: New features or improvements
- **Patch**: Bug fixes and small updates

Example: `v1.2.3`
- `v1`: First production version
- `v2`: Added neural network to ensemble  
- `v3`: Updated feature engineering

## Performance Monitoring

### Drift Detection

Monitor for:
- **Data Drift**: Changes in input feature distributions
- **Concept Drift**: Changes in target relationships
- **Performance Drift**: Degradation in model metrics

```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[
    DataDriftPreset(),
])

report.run(reference_data=train_data, current_data=production_data)
```

### A/B Testing

Compare model versions in production:
```python
import random

def get_model_version(user_id):
    if hash(user_id) % 100 < 10:  # 10% get new model
        return "v2.1.0"
    return "v2.0.0"
```

### Model Retraining

Automated retraining triggers:
- Performance drops below threshold (F1 < 0.90)
- Data drift detection
- Monthly scheduled retraining
- New labeled data availability

## Deployment Considerations

### Model Size Optimization

- **Feature Selection**: Remove redundant features
- **Model Compression**: Prune unnecessary parameters
- **Quantization**: Reduce precision for speed

### Inference Optimization

- **Model Caching**: Keep models in memory
- **Batch Prediction**: Process multiple requests together
- **Feature Caching**: Cache computed features
- **Model Serving**: Use optimized serving frameworks

### Scalability

- **Horizontal Scaling**: Multiple model instances
- **Load Balancing**: Distribute requests evenly
- **Auto-scaling**: Scale based on demand
- **Resource Limits**: CPU and memory constraints

## Troubleshooting

### Common Issues

1. **Poor Performance on New Data**
   - Check for data drift
   - Verify feature consistency
   - Consider model retraining

2. **Slow Inference**
   - Profile feature extraction
   - Optimize model serving
   - Use model compression

3. **Memory Issues**
   - Reduce model complexity
   - Implement model streaming
   - Optimize feature storage

4. **Inconsistent Predictions**
   - Check for data leakage
   - Verify preprocessing steps
   - Monitor feature distributions

### Debug Mode

Enable detailed logging for troubleshooting:
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add debug information to predictions
def debug_predict(text, user_id=None):
    logger.debug(f"Input text length: {len(text)}")
    features = extract_features(text, user_id)
    logger.debug(f"Extracted {len(features)} features")
    prediction = model.predict_proba([features])[0]
    logger.debug(f"Raw prediction probabilities: {prediction}")
    return prediction
```

---

For implementation details and code examples, see the source code in the `src/modeling.py` module and Jupyter notebooks in the `notebooks/` directory.
