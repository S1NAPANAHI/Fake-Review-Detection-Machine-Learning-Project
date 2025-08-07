"""
Model Evaluation Module.

This module provides the Evaluation class for:
- Computing comprehensive evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- Generating confusion matrix plots and classification reports
- Exporting metrics as JSON and plots as PNG to artifacts/reports
- Integrating with Prometheus counters for monitoring
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.base import BaseEstimator

# Import utilities
try:
    from . import utils
except ImportError:
    # Fallback for direct execution
    import utils

logger = utils.setup_logging(__name__)


class Evaluation:
    """
    Comprehensive model evaluation class.
    
    Provides functionality for computing evaluation metrics, generating visualizations,
    and exporting results for model performance analysis.
    """
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 class_labels: Optional[List[str]] = None,
                 pos_label: Optional[Union[str, int]] = None,
                 reports_dir: Optional[Union[str, Path]] = None,
                 enable_prometheus: bool = True,
                 verbose: bool = True):
        """
        Initialize the Evaluation class.
        
        Args:
            model_name: Name of the model being evaluated
            class_labels: List of class labels for multiclass problems
            pos_label: Positive class label for binary classification
            reports_dir: Directory to save reports (defaults to artifacts/reports)
            enable_prometheus: Enable Prometheus metrics collection
            verbose: Enable verbose logging
        """
        self.model_name = model_name or "model"
        self.class_labels = class_labels
        self.pos_label = pos_label
        self.enable_prometheus = enable_prometheus
        self.verbose = verbose
        
        # Set up reports directory
        if reports_dir is None:
            self.reports_dir = utils.resolve_path("artifacts/reports")
        else:
            self.reports_dir = utils.resolve_path(reports_dir)
        
        utils.ensure_dir(self.reports_dir)
        
        # Initialize metrics storage
        self.metrics_ = {}
        self.confusion_matrix_ = None
        self.classification_report_ = None
        self.roc_curves_ = {}
        self.pr_curves_ = {}
        
        # Set up Prometheus counters
        self._setup_prometheus_counters()
        
        if self.verbose:
            logger.info(f"Evaluation initialized for model: {self.model_name}")
            logger.info(f"Reports will be saved to: {self.reports_dir}")
    
    def _setup_prometheus_counters(self) -> None:
        """Set up Prometheus counters for evaluation tracking."""
        if not self.enable_prometheus:
            return
        
        # Initialize counters for different evaluation operations
        self._prometheus_counters = {
            'evaluations_total': f'evaluation_runs_total',
            'metrics_computed': f'evaluation_metrics_computed_total',
            'plots_generated': f'evaluation_plots_generated_total',
            'reports_exported': f'evaluation_reports_exported_total'
        }
        
        # Initialize all counters with zero
        for counter_name in self._prometheus_counters.values():
            utils.metrics.increment_counter(counter_name, value=0)
    
    def _increment_prometheus_counter(self, counter_key: str, value: float = 1.0) -> None:
        """Increment a Prometheus counter if enabled."""
        if not self.enable_prometheus:
            return
        
        counter_name = self._prometheus_counters.get(counter_key)
        if counter_name:
            utils.metrics.increment_counter(
                counter_name, 
                value=value, 
                labels={'model': self.model_name}
            )
    
    def compute_metrics(self, 
                       y_true: np.ndarray, 
                       y_pred: np.ndarray, 
                       y_pred_proba: Optional[np.ndarray] = None,
                       average: str = 'weighted') -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (needed for ROC-AUC)
            average: Averaging method for multiclass metrics
            
        Returns:
            Dict: Computed metrics
        """
        if self.verbose:
            logger.info("Computing evaluation metrics...")
        
        try:
            # Basic metrics
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
                'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
            }
            
            # ROC-AUC computation
            if y_pred_proba is not None:
                try:
                    # Handle binary and multiclass cases
                    unique_classes = np.unique(y_true)
                    n_classes = len(unique_classes)
                    
                    if n_classes == 2:
                        # Binary classification
                        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                            # Use probability of positive class
                            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                        else:
                            # Single probability score
                            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                    else:
                        # Multiclass classification
                        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == n_classes:
                            metrics['roc_auc'] = roc_auc_score(
                                y_true, y_pred_proba, 
                                multi_class='ovr', 
                                average=average
                            )
                        else:
                            logger.warning("Probabilities shape incompatible for multiclass ROC-AUC")
                            metrics['roc_auc'] = np.nan
                            
                except Exception as e:
                    logger.warning(f"Could not compute ROC-AUC: {e}")
                    metrics['roc_auc'] = np.nan
            else:
                logger.warning("No probabilities provided, ROC-AUC not computed")
                metrics['roc_auc'] = np.nan
            
            # Additional metrics for binary classification
            if len(np.unique(y_true)) == 2:
                # Specificity (True Negative Rate)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['sensitivity'] = metrics['recall']  # Alias for recall
                
                # Matthews Correlation Coefficient
                from sklearn.metrics import matthews_corrcoef
                metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
                
                # Average Precision (PR-AUC) if probabilities available
                if y_pred_proba is not None:
                    if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                        prob_pos = y_pred_proba[:, 1]
                    else:
                        prob_pos = y_pred_proba
                    metrics['pr_auc'] = average_precision_score(y_true, prob_pos)
            
            # Store metrics
            self.metrics_ = metrics
            
            # Log metrics
            if self.verbose:
                logger.info("Computed metrics:")
                for metric_name, value in metrics.items():
                    if not np.isnan(value):
                        logger.info(f"  {metric_name}: {value:.4f}")
                    else:
                        logger.info(f"  {metric_name}: N/A")
            
            # Update Prometheus counters
            self._increment_prometheus_counter('metrics_computed')
            
            # Set Prometheus gauges for key metrics
            if self.enable_prometheus:
                for metric_name, value in metrics.items():
                    if not np.isnan(value):
                        utils.metrics.set_gauge(
                            f'model_{metric_name}', 
                            value, 
                            labels={'model': self.model_name}
                        )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            self._increment_prometheus_counter('evaluations_total', 0)  # Mark as failed
            raise
    
    def generate_confusion_matrix(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 normalize: Optional[str] = None,
                                 save_plot: bool = True,
                                 filename: Optional[str] = None) -> np.ndarray:
        """
        Generate and optionally plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization method ('true', 'pred', 'all', or None)
            save_plot: Whether to save the plot
            filename: Custom filename for the plot
            
        Returns:
            np.ndarray: Confusion matrix
        """
        if self.verbose:
            logger.info("Generating confusion matrix...")
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.confusion_matrix_ = cm
        
        # Normalize if requested
        if normalize:
            if normalize == 'true':
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            elif normalize == 'pred':
                cm_norm = cm.astype('float') / cm.sum(axis=0)
            elif normalize == 'all':
                cm_norm = cm.astype('float') / cm.sum()
            else:
                raise ValueError("normalize must be 'true', 'pred', 'all', or None")
        else:
            cm_norm = cm
        
        # Generate plot
        if save_plot:
            self._plot_confusion_matrix(
                cm_norm, y_true, normalize=normalize, filename=filename
            )
        
        return cm
    
    def _plot_confusion_matrix(self, 
                              cm: np.ndarray, 
                              y_true: np.ndarray,
                              normalize: Optional[str] = None,
                              filename: Optional[str] = None) -> Path:
        """
        Plot confusion matrix and save to file.
        
        Args:
            cm: Confusion matrix
            y_true: True labels for determining classes
            normalize: Normalization method used
            filename: Custom filename
            
        Returns:
            Path: Path to saved plot
        """
        # Set up the plot
        plt.figure(figsize=(10, 8))
        
        # Determine class labels
        if self.class_labels:
            labels = self.class_labels
        else:
            labels = [f"Class {i}" for i in np.unique(y_true)]
        
        # Choose colormap and format based on normalization
        if normalize:
            fmt = '.2f'
            cmap = 'Blues'
            title_suffix = f" (normalized by {normalize})"
        else:
            fmt = 'd'
            cmap = 'Blues'
            title_suffix = " (counts)"
        
        # Create heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
            square=True,
            cbar_kws={'shrink': 0.8}
        )
        
        plt.title(f'Confusion Matrix - {self.model_name}{title_suffix}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        # Save plot
        if filename is None:
            timestamp = utils.get_timestamp()
            norm_suffix = f"_{normalize}" if normalize else ""
            filename = f"confusion_matrix_{self.model_name}_{timestamp}{norm_suffix}.png"
        
        filepath = self.reports_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            logger.info(f"Confusion matrix plot saved to: {filepath}")
        
        # Update Prometheus counter
        self._increment_prometheus_counter('plots_generated')
        
        return filepath
    
    def generate_classification_report(self, 
                                     y_true: np.ndarray, 
                                     y_pred: np.ndarray,
                                     save_report: bool = True,
                                     filename: Optional[str] = None) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_report: Whether to save the report
            filename: Custom filename for the report
            
        Returns:
            str: Classification report as string
        """
        if self.verbose:
            logger.info("Generating classification report...")
        
        # Generate report
        if self.class_labels:
            target_names = self.class_labels
        else:
            target_names = [f"Class {i}" for i in np.unique(y_true)]
        
        report = classification_report(
            y_true, y_pred, 
            target_names=target_names,
            digits=4
        )
        
        self.classification_report_ = report
        
        # Save report
        if save_report:
            if filename is None:
                timestamp = utils.get_timestamp()
                filename = f"classification_report_{self.model_name}_{timestamp}.txt"
            
            filepath = self.reports_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Classification Report - {self.model_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(report)
                f.write(f"\n\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if self.verbose:
                logger.info(f"Classification report saved to: {filepath}")
            
            # Update Prometheus counter
            self._increment_prometheus_counter('reports_exported')
        
        return report
    
    def plot_roc_curves(self, 
                       y_true: np.ndarray, 
                       y_pred_proba: np.ndarray,
                       save_plot: bool = True,
                       filename: Optional[str] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Plot ROC curves for binary or multiclass classification.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_plot: Whether to save the plot
            filename: Custom filename for the plot
            
        Returns:
            Dict: ROC curve data (fpr, tpr) for each class
        """
        if self.verbose:
            logger.info("Plotting ROC curves...")
        
        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)
        
        plt.figure(figsize=(10, 8))
        
        if n_classes == 2:
            # Binary classification
            if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                prob_pos = y_pred_proba[:, 1]
            else:
                prob_pos = y_pred_proba
            
            fpr, tpr, _ = roc_curve(y_true, prob_pos)
            roc_auc = roc_auc_score(y_true, prob_pos)
            
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            
            self.roc_curves_['binary'] = (fpr, tpr)
            
        else:
            # Multiclass classification
            y_true_bin = label_binarize(y_true, classes=unique_classes)
            
            # Compute ROC curve for each class
            for i, class_label in enumerate(unique_classes):
                if y_pred_proba.shape[1] > i:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                    roc_auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
                    
                    class_name = self.class_labels[i] if self.class_labels else f'Class {class_label}'
                    plt.plot(fpr, tpr, linewidth=2,
                            label=f'{class_name} (AUC = {roc_auc:.3f})')
                    
                    self.roc_curves_[class_name] = (fpr, tpr)
        
        # Plot random classifier line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        if save_plot:
            if filename is None:
                timestamp = utils.get_timestamp()
                filename = f"roc_curves_{self.model_name}_{timestamp}.png"
            
            filepath = self.reports_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            if self.verbose:
                logger.info(f"ROC curves plot saved to: {filepath}")
            
            # Update Prometheus counter
            self._increment_prometheus_counter('plots_generated')
        
        return self.roc_curves_
    
    def plot_precision_recall_curves(self, 
                                    y_true: np.ndarray, 
                                    y_pred_proba: np.ndarray,
                                    save_plot: bool = True,
                                    filename: Optional[str] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Plot Precision-Recall curves for binary or multiclass classification.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_plot: Whether to save the plot
            filename: Custom filename for the plot
            
        Returns:
            Dict: PR curve data (precision, recall) for each class
        """
        if self.verbose:
            logger.info("Plotting Precision-Recall curves...")
        
        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)
        
        plt.figure(figsize=(10, 8))
        
        if n_classes == 2:
            # Binary classification
            if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                prob_pos = y_pred_proba[:, 1]
            else:
                prob_pos = y_pred_proba
            
            precision, recall, _ = precision_recall_curve(y_true, prob_pos)
            avg_precision = average_precision_score(y_true, prob_pos)
            
            plt.plot(recall, precision, linewidth=2,
                    label=f'PR curve (AP = {avg_precision:.3f})')
            
            self.pr_curves_['binary'] = (precision, recall)
            
        else:
            # Multiclass classification
            y_true_bin = label_binarize(y_true, classes=unique_classes)
            
            # Compute PR curve for each class
            for i, class_label in enumerate(unique_classes):
                if y_pred_proba.shape[1] > i:
                    precision, recall, _ = precision_recall_curve(
                        y_true_bin[:, i], y_pred_proba[:, i]
                    )
                    avg_precision = average_precision_score(
                        y_true_bin[:, i], y_pred_proba[:, i]
                    )
                    
                    class_name = self.class_labels[i] if self.class_labels else f'Class {class_label}'
                    plt.plot(recall, precision, linewidth=2,
                            label=f'{class_name} (AP = {avg_precision:.3f})')
                    
                    self.pr_curves_[class_name] = (precision, recall)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curves - {self.model_name}', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        if save_plot:
            if filename is None:
                timestamp = utils.get_timestamp()
                filename = f"pr_curves_{self.model_name}_{timestamp}.png"
            
            filepath = self.reports_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            if self.verbose:
                logger.info(f"Precision-Recall curves plot saved to: {filepath}")
            
            # Update Prometheus counter
            self._increment_prometheus_counter('plots_generated')
        
        return self.pr_curves_
    
    def export_metrics_json(self, 
                           metrics: Optional[Dict[str, Any]] = None,
                           filename: Optional[str] = None,
                           include_metadata: bool = True) -> Path:
        """
        Export metrics to JSON file.
        
        Args:
            metrics: Metrics dictionary (uses self.metrics_ if None)
            filename: Custom filename for JSON file
            include_metadata: Include metadata in the export
            
        Returns:
            Path: Path to saved JSON file
        """
        if metrics is None:
            if not self.metrics_:
                raise ValueError("No metrics available. Compute metrics first.")
            metrics = self.metrics_
        
        if filename is None:
            timestamp = utils.get_timestamp()
            filename = f"metrics_{self.model_name}_{timestamp}.json"
        
        # Prepare export data
        export_data = {
            'model_name': self.model_name,
            'metrics': {}
        }
        
        # Convert numpy types to Python types for JSON serialization
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                export_data['metrics'][key] = float(value)
            elif isinstance(value, np.ndarray):
                export_data['metrics'][key] = value.tolist()
            else:
                export_data['metrics'][key] = value
        
        # Add metadata if requested
        if include_metadata:
            export_data['metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'evaluation_version': '1.0',
                'class_labels': self.class_labels,
                'pos_label': self.pos_label
            }
            
            # Add confusion matrix if available
            if self.confusion_matrix_ is not None:
                export_data['confusion_matrix'] = self.confusion_matrix_.tolist()
        
        # Save to JSON
        filepath = self.reports_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            logger.info(f"Metrics exported to JSON: {filepath}")
        
        # Update Prometheus counter
        self._increment_prometheus_counter('reports_exported')
        
        return filepath
    
    def comprehensive_evaluation(self, 
                               y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               y_pred_proba: Optional[np.ndarray] = None,
                               save_all: bool = True,
                               export_prefix: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation with all metrics and plots.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            save_all: Save all plots and reports
            export_prefix: Prefix for all exported files
            
        Returns:
            Dict: All evaluation results
        """
        if self.verbose:
            logger.info(f"Starting comprehensive evaluation for {self.model_name}...")
        
        try:
            # Set up filename prefix
            timestamp = utils.get_timestamp()
            prefix = export_prefix or f"{self.model_name}_{timestamp}"
            
            # Compute metrics
            metrics = self.compute_metrics(y_true, y_pred, y_pred_proba)
            
            # Generate confusion matrix
            cm = self.generate_confusion_matrix(
                y_true, y_pred, 
                save_plot=save_all, 
                filename=f"confusion_matrix_{prefix}.png" if save_all else None
            )
            
            # Generate classification report
            class_report = self.generate_classification_report(
                y_true, y_pred,
                save_report=save_all,
                filename=f"classification_report_{prefix}.txt" if save_all else None
            )
            
            # Generate ROC curves if probabilities available
            roc_data = {}
            if y_pred_proba is not None and save_all:
                roc_data = self.plot_roc_curves(
                    y_true, y_pred_proba,
                    save_plot=save_all,
                    filename=f"roc_curves_{prefix}.png" if save_all else None
                )
                
                # Generate PR curves for binary classification
                if len(np.unique(y_true)) == 2:
                    pr_data = self.plot_precision_recall_curves(
                        y_true, y_pred_proba,
                        save_plot=save_all,
                        filename=f"pr_curves_{prefix}.png" if save_all else None
                    )
                else:
                    pr_data = {}
            else:
                pr_data = {}
            
            # Export metrics to JSON
            if save_all:
                json_path = self.export_metrics_json(
                    metrics, 
                    filename=f"metrics_{prefix}.json"
                )
            else:
                json_path = None
            
            # Compile results
            results = {
                'metrics': metrics,
                'confusion_matrix': cm.tolist() if isinstance(cm, np.ndarray) else cm,
                'classification_report': class_report,
                'roc_curves': roc_data,
                'pr_curves': pr_data,
                'files_created': []
            }
            
            # Add file paths if files were created
            if save_all and json_path:
                results['files_created'].append(str(json_path))
            
            # Update Prometheus counter for completed evaluation
            self._increment_prometheus_counter('evaluations_total')
            
            # Set overall evaluation quality gauge
            if self.enable_prometheus:
                f1_score_val = metrics.get('f1_score', 0)
                utils.metrics.set_gauge(
                    'evaluation_quality_score',
                    f1_score_val,
                    labels={'model': self.model_name, 'metric': 'f1'}
                )
            
            if self.verbose:
                logger.info("Comprehensive evaluation completed successfully")
                logger.info(f"Generated {len(results['files_created'])} files")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during comprehensive evaluation: {e}")
            # Increment error counter
            if self.enable_prometheus:
                utils.metrics.increment_counter(
                    'evaluation_errors_total',
                    labels={'model': self.model_name}
                )
            raise
    
    def compare_models(self, 
                      evaluation_results: Dict[str, Dict[str, Any]],
                      save_comparison: bool = True,
                      filename: Optional[str] = None) -> pd.DataFrame:
        """
        Compare evaluation results from multiple models.
        
        Args:
            evaluation_results: Dict mapping model names to their evaluation results
            save_comparison: Save comparison table
            filename: Custom filename for comparison
            
        Returns:
            pd.DataFrame: Comparison table
        """
        if self.verbose:
            logger.info("Comparing model evaluation results...")
        
        # Extract metrics for comparison
        comparison_data = []
        
        for model_name, results in evaluation_results.items():
            if 'metrics' in results:
                row = {'model': model_name}
                row.update(results['metrics'])
                comparison_data.append(row)
        
        if not comparison_data:
            raise ValueError("No valid evaluation results provided for comparison")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('model')
        
        # Sort by F1 score (descending)
        if 'f1_score' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        # Save comparison table
        if save_comparison:
            if filename is None:
                timestamp = utils.get_timestamp()
                filename = f"model_comparison_{timestamp}.csv"
            
            filepath = self.reports_dir / filename
            comparison_df.to_csv(filepath)
            
            if self.verbose:
                logger.info(f"Model comparison saved to: {filepath}")
            
            # Update Prometheus counter
            self._increment_prometheus_counter('reports_exported')
        
        return comparison_df
    
    # Convenience methods to match notebook interface
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs):
        """Convenience method matching notebook interface."""
        return self.generate_confusion_matrix(y_true, y_pred, **kwargs)
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, **kwargs):
        """Convenience method matching notebook interface."""
        return self.plot_roc_curves(y_true, y_pred_proba, **kwargs)
    
    def classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs):
        """Convenience method matching notebook interface."""
        return self.generate_classification_report(y_true, y_pred, **kwargs)
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get summary of current evaluation state.
        
        Returns:
            Dict: Evaluation summary
        """
        return {
            'model_name': self.model_name,
            'has_metrics': bool(self.metrics_),
            'has_confusion_matrix': self.confusion_matrix_ is not None,
            'has_classification_report': self.classification_report_ is not None,
            'metrics': self.metrics_,
            'reports_directory': str(self.reports_dir),
            'prometheus_enabled': self.enable_prometheus,
            'class_labels': self.class_labels,
            'pos_label': self.pos_label
        }
