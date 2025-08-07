# Evaluation Module Documentation

## Overview

The `src/evaluation.py` module provides a comprehensive `Evaluation` class for machine learning model evaluation. This class implements all the requested features for Step 8 of the Fake Review Detection System.

## Features Implemented

### ✅ Core Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted precision for multi-class support
- **Recall**: Weighted recall for multi-class support  
- **F1 Score**: Weighted F1 score for multi-class support
- **ROC-AUC**: Receiver Operating Characteristic Area Under Curve
- **Additional Binary Metrics**: Specificity, Sensitivity, Matthews Correlation Coefficient, PR-AUC

### ✅ Visualization & Reporting
- **Confusion Matrix Plot**: Heatmap with customizable normalization
- **Classification Report**: Detailed per-class metrics
- **ROC Curves**: For binary and multi-class classification
- **Precision-Recall Curves**: For binary classification

### ✅ Export Capabilities
- **Metrics JSON**: Structured metrics export with metadata
- **PNG Plots**: High-resolution plots (300 DPI) saved to `artifacts/reports`
- **Text Reports**: Classification reports saved as text files
- **Model Comparison**: CSV comparison tables for multiple models

### ✅ Prometheus Integration
- **Counters**: Track evaluation runs, metrics computed, plots generated, reports exported
- **Gauges**: Model performance metrics with labels
- **Error Tracking**: Failed evaluation attempts

## Class Structure

```python
class Evaluation:
    """Comprehensive model evaluation class."""
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 class_labels: Optional[List[str]] = None,
                 pos_label: Optional[Union[str, int]] = None,
                 reports_dir: Optional[Union[str, Path]] = None,
                 enable_prometheus: bool = True,
                 verbose: bool = True)
```

## Key Methods

### Core Evaluation
- `compute_metrics()` - Calculate all evaluation metrics
- `comprehensive_evaluation()` - Run complete evaluation pipeline
- `get_evaluation_summary()` - Get current evaluation state

### Visualization
- `generate_confusion_matrix()` - Create confusion matrix plot
- `plot_roc_curves()` - Generate ROC curve plots
- `plot_precision_recall_curves()` - Generate PR curve plots

### Reporting
- `generate_classification_report()` - Create detailed classification report
- `export_metrics_json()` - Export metrics to JSON
- `compare_models()` - Compare multiple models

## Usage Examples

### Basic Usage

```python
from src.evaluation import Evaluation

# Initialize evaluator
evaluator = Evaluation(
    model_name="RandomForest_FakeReview",
    class_labels=["Fake", "Real"],
    pos_label=1,
    verbose=True
)

# Perform comprehensive evaluation
results = evaluator.comprehensive_evaluation(
    y_true=y_test,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
    save_all=True
)

# Access metrics
print("Model Performance:")
for metric, value in results['metrics'].items():
    print(f"{metric}: {value:.4f}")
```

### Individual Components

```python
# Compute specific metrics
metrics = evaluator.compute_metrics(y_true, y_pred, y_pred_proba)

# Generate confusion matrix
cm = evaluator.generate_confusion_matrix(y_true, y_pred, normalize='true')

# Create classification report
report = evaluator.generate_classification_report(y_true, y_pred)

# Plot ROC curves
roc_data = evaluator.plot_roc_curves(y_true, y_pred_proba)

# Export to JSON
json_path = evaluator.export_metrics_json()
```

### Model Comparison

```python
# Compare multiple models
evaluation_results = {
    "RandomForest": rf_results,
    "XGBoost": xgb_results,
    "LogisticRegression": lr_results
}

comparison_df = evaluator.compare_models(
    evaluation_results,
    save_comparison=True
)
print(comparison_df)
```

## File Outputs

The evaluation class creates the following files in `artifacts/reports/`:

### Plots (PNG, 300 DPI)
- `confusion_matrix_{model_name}_{timestamp}.png`
- `roc_curves_{model_name}_{timestamp}.png`
- `pr_curves_{model_name}_{timestamp}.png`

### Reports 
- `classification_report_{model_name}_{timestamp}.txt`
- `metrics_{model_name}_{timestamp}.json`
- `model_comparison_{timestamp}.csv`

## JSON Export Format

```json
{
  "model_name": "RandomForest_Test",
  "metrics": {
    "accuracy": 0.965,
    "precision": 0.9650,
    "recall": 0.965,
    "f1_score": 0.9650,
    "roc_auc": 0.9927
  },
  "metadata": {
    "timestamp": "2025-01-08T15:03:45.123456",
    "evaluation_version": "1.0",
    "class_labels": ["Fake", "Real"],
    "pos_label": 1
  },
  "confusion_matrix": [[95, 4], [3, 98]]
}
```

## Prometheus Metrics

The class integrates with Prometheus through counters and gauges:

### Counters
- `evaluation_runs_total{model="model_name"}` - Total evaluation runs
- `evaluation_metrics_computed_total{model="model_name"}` - Metrics computed
- `evaluation_plots_generated_total{model="model_name"}` - Plots generated  
- `evaluation_reports_exported_total{model="model_name"}` - Reports exported
- `evaluation_errors_total{model="model_name"}` - Evaluation errors

### Gauges
- `model_accuracy{model="model_name"}` - Current model accuracy
- `model_precision{model="model_name"}` - Current model precision
- `model_recall{model="model_name"}` - Current model recall
- `model_f1_score{model="model_name"}` - Current model F1 score
- `model_roc_auc{model="model_name"}` - Current model ROC-AUC
- `evaluation_quality_score{model="model_name", metric="f1"}` - Overall quality

## Configuration

The class uses the existing `utils.py` module for:
- Path resolution and directory creation
- Timestamp generation
- Prometheus metrics collection
- Logging configuration

Configuration is read from `config/settings.yaml`:

```yaml
evaluation:
  metrics:
    classification:
      - "accuracy"
      - "precision" 
      - "recall"
      - "f1_score"
      - "roc_auc"
      - "confusion_matrix"
```

## Dependencies

Required packages (already in `requirements.txt`):
- `scikit-learn==1.3.0` - Core metrics and evaluation
- `matplotlib==3.7.1` - Plotting
- `seaborn==0.12.2` - Enhanced visualizations
- `pandas==2.0.3` - Data handling
- `numpy==1.24.3` - Numerical computations
- `prometheus_client==0.17.1` - Metrics collection

## Error Handling

The class includes comprehensive error handling:
- Graceful degradation when probabilities not available
- Warning messages for incompatible data shapes
- Exception handling with proper logging
- Prometheus error counters

## Testing

Run the test script to verify functionality:

```bash
python simple_test_evaluation.py
```

This will:
1. Generate synthetic classification data
2. Train a RandomForest model
3. Run comprehensive evaluation
4. Save all outputs to `artifacts/reports/`
5. Display results and file locations

## Integration with Existing Codebase

The `Evaluation` class integrates seamlessly with the existing codebase:

- Uses `utils.py` for common functionality
- Follows existing logging patterns  
- Respects configuration from `settings.yaml`
- Saves to standard `artifacts/reports/` directory
- Compatible with `ModelTrainer` class outputs

## Future Enhancements

Potential improvements for future versions:
- Custom metric plugins
- Interactive plots with Plotly
- Automatic threshold optimization
- Feature importance integration  
- Model explainability reports
- Performance benchmarking
- A/B testing support

## Conclusion

The `Evaluation` class provides a complete solution for Step 8 requirements:

✅ **Implemented all requested metrics**: accuracy, precision, recall, F1, ROC-AUC  
✅ **Generated visualizations**: confusion matrix plots and classification reports  
✅ **Export capabilities**: JSON metrics and PNG plots to `artifacts/reports`  
✅ **Prometheus integration**: counters and gauges for monitoring  
✅ **Comprehensive testing**: verified with synthetic data  
✅ **Production-ready**: error handling, logging, and documentation

The implementation is robust, well-documented, and ready for integration into the broader Fake Review Detection System.
