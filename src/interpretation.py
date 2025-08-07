"""
Model Interpretation Module.

This module provides the ModelInterpreter class for:
- SHAP (SHapley Additive exPlanations) analysis
- LIME (Local Interpretable Model-agnostic Explanations)
- Permutation feature importance
- Partial dependence plots
- Visualization saving in artifacts/reports/interpretation
"""

import warnings
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

# Core ML libraries
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.model_selection import train_test_split

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Interpretability libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with 'pip install shap'")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available. Install with 'pip install lime'")

# Import utilities
from . import utils

logger = utils.setup_logging(__name__)


class ModelInterpreter:
    """
    Comprehensive model interpretation class.
    
    Provides multiple interpretability methods including SHAP, LIME, 
    permutation importance, and partial dependence plots with automatic
    visualization saving.
    """
    
    def __init__(self, 
                 model: BaseEstimator,
                 feature_names: Optional[List[str]] = None,
                 class_names: Optional[List[str]] = None,
                 save_dir: Optional[Union[str, Path]] = None,
                 random_state: Optional[int] = None,
                 verbose: bool = True):
        """
        Initialize the ModelInterpreter.
        
        Args:
            model: Trained scikit-learn compatible model
            feature_names: Names of input features
            class_names: Names of target classes for classification
            save_dir: Directory to save interpretation plots
            random_state: Random state for reproducibility
            verbose: Enable verbose logging
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.random_state = random_state or utils.get_setting('environment.random_seed', 42)
        self.verbose = verbose
        
        # Set up save directory
        if save_dir is None:
            self.save_dir = utils.resolve_path("artifacts/reports/interpretation")
        else:
            self.save_dir = utils.resolve_path(save_dir)
        utils.ensure_dir(self.save_dir)
        
        # Initialize explainers as None - will be created when needed
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Store computed explanations
        self.shap_values_ = None
        self.lime_explanations_ = None
        self.permutation_importance_ = None
        self.partial_dependence_ = None
        
        if self.verbose:
            logger.info(f"ModelInterpreter initialized for {type(model).__name__}")
            logger.info(f"Visualizations will be saved to: {self.save_dir}")
    
    def _validate_inputs(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Validate input data."""
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError("X must be numpy array or pandas DataFrame")
        
        if len(X.shape) != 2:
            raise ValueError("X must be 2-dimensional")
        
        if self.feature_names and len(self.feature_names) != X.shape[1]:
            raise ValueError("Number of feature names must match number of features")
        
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have same number of samples")
    
    def _get_feature_names(self, X: np.ndarray) -> List[str]:
        """Get feature names, creating default ones if not provided."""
        if self.feature_names:
            return self.feature_names
        return [f"feature_{i}" for i in range(X.shape[1])]
    
    def _save_figure(self, fig: plt.Figure, filename: str, **kwargs) -> Path:
        """Save figure with standard formatting."""
        filepath = self.save_dir / f"{filename}.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight', **kwargs)
        plt.close(fig)
        
        if self.verbose:
            logger.info(f"Saved plot: {filepath}")
        
        return filepath
    
    def explain_shap(self, 
                    X: np.ndarray, 
                    sample_size: Optional[int] = None,
                    explainer_type: str = 'auto',
                    plot_types: List[str] = ['summary', 'bar', 'waterfall'],
                    max_display: int = 20) -> Dict[str, Any]:
        """
        Generate SHAP explanations and visualizations.
        
        Args:
            X: Input features for explanation
            sample_size: Number of samples to use for explanation (None for all)
            explainer_type: Type of SHAP explainer ('auto', 'tree', 'linear', 'kernel')
            plot_types: Types of plots to generate
            max_display: Maximum number of features to display in plots
            
        Returns:
            Dictionary containing SHAP values and saved plot paths
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not available. Install with 'pip install shap'")
        
        self._validate_inputs(X)
        
        if self.verbose:
            logger.info("Generating SHAP explanations...")
        
        # Convert to numpy if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Sample data if requested
        if sample_size and sample_size < len(X):
            np.random.seed(self.random_state)
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Create SHAP explainer based on model type
        if explainer_type == 'auto':
            # Auto-select explainer based on model type
            model_name = type(self.model).__name__.lower()
            if any(tree_model in model_name for tree_model in ['forest', 'tree', 'xgb', 'lgb', 'catboost']):
                self.shap_explainer = shap.TreeExplainer(self.model)
            elif any(linear_model in model_name for linear_model in ['linear', 'logistic', 'svm']):
                self.shap_explainer = shap.LinearExplainer(self.model, X_sample)
            else:
                # Fallback to kernel explainer (slower but model-agnostic)
                self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, X_sample[:100])
        elif explainer_type == 'tree':
            self.shap_explainer = shap.TreeExplainer(self.model)
        elif explainer_type == 'linear':
            self.shap_explainer = shap.LinearExplainer(self.model, X_sample)
        elif explainer_type == 'kernel':
            self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, X_sample[:100])
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
        
        # Calculate SHAP values
        try:
            self.shap_values_ = self.shap_explainer.shap_values(X_sample)
        except Exception as e:
            logger.warning(f"Failed to compute SHAP values with {explainer_type} explainer: {e}")
            # Fallback to kernel explainer
            self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, X_sample[:50])
            self.shap_values_ = self.shap_explainer.shap_values(X_sample)
        
        # Handle multi-class case
        if isinstance(self.shap_values_, list):
            # For multi-class, use class 1 (positive class) for binary classification
            if len(self.shap_values_) == 2:
                shap_values_plot = self.shap_values_[1]
            else:
                shap_values_plot = self.shap_values_[0]  # Use first class for multi-class
        else:
            shap_values_plot = self.shap_values_
        
        feature_names = self._get_feature_names(X_sample)
        saved_plots = {}
        
        # Generate requested plots
        if 'summary' in plot_types:
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values_plot, X_sample, 
                            feature_names=feature_names, 
                            max_display=max_display, show=False)
            plt.title("SHAP Summary Plot")
            saved_plots['summary'] = self._save_figure(plt.gcf(), "shap_summary")
        
        if 'bar' in plot_types:
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values_plot, X_sample,
                            feature_names=feature_names,
                            plot_type="bar", max_display=max_display, show=False)
            plt.title("SHAP Feature Importance")
            saved_plots['bar'] = self._save_figure(plt.gcf(), "shap_importance")
        
        if 'waterfall' in plot_types and len(X_sample) > 0:
            # Create waterfall plot for first sample
            fig, ax = plt.subplots(figsize=(10, 8))
            if hasattr(shap, 'waterfall_plot'):
                # Use new SHAP API if available
                if hasattr(self.shap_explainer, 'expected_value'):
                    expected_value = self.shap_explainer.expected_value
                    if isinstance(expected_value, (list, np.ndarray)):
                        expected_value = expected_value[0] if len(expected_value) > 1 else expected_value[0]
                else:
                    expected_value = 0
                    
                shap.waterfall_plot(shap.Explanation(
                    values=shap_values_plot[0],
                    base_values=expected_value,
                    data=X_sample[0],
                    feature_names=feature_names
                ), show=False)
            else:
                # Fallback for older SHAP versions
                shap.force_plot(
                    self.shap_explainer.expected_value[0] if isinstance(self.shap_explainer.expected_value, list) 
                    else self.shap_explainer.expected_value,
                    shap_values_plot[0], X_sample[0], feature_names=feature_names,
                    matplotlib=True, show=False
                )
            plt.title("SHAP Waterfall Plot (First Sample)")
            saved_plots['waterfall'] = self._save_figure(plt.gcf(), "shap_waterfall")
        
        result = {
            'shap_values': self.shap_values_,
            'expected_value': getattr(self.shap_explainer, 'expected_value', None),
            'feature_names': feature_names,
            'explainer_type': explainer_type,
            'saved_plots': saved_plots
        }
        
        if self.verbose:
            logger.info(f"SHAP analysis completed. Generated {len(saved_plots)} plots.")
        
        return result
    
    def explain_lime(self, 
                    X_train: np.ndarray,
                    X_explain: np.ndarray,
                    y_train: Optional[np.ndarray] = None,
                    num_samples: int = 5000,
                    num_features: int = 10,
                    sample_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Generate LIME explanations and visualizations.
        
        Args:
            X_train: Training data for LIME explainer
            X_explain: Samples to explain
            y_train: Training labels (optional, used for discretization)
            num_samples: Number of samples for LIME to generate
            num_features: Number of top features to show
            sample_indices: Specific sample indices to explain (None for all)
            
        Returns:
            Dictionary containing LIME explanations and saved plot paths
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is not available. Install with 'pip install lime'")
        
        self._validate_inputs(X_train)
        self._validate_inputs(X_explain)
        
        if self.verbose:
            logger.info("Generating LIME explanations...")
        
        # Convert to numpy if pandas DataFrame
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(X_explain, pd.DataFrame):
            X_explain = X_explain.values
        
        # Create LIME explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self._get_feature_names(X_train),
            class_names=self.class_names or ['Class 0', 'Class 1'],
            mode='classification',
            discretize_continuous=True,
            random_state=self.random_state
        )
        
        # Select samples to explain
        if sample_indices is None:
            sample_indices = list(range(min(5, len(X_explain))))  # Explain first 5 samples
        
        explanations = []
        saved_plots = {}
        
        for i, idx in enumerate(sample_indices):
            if idx >= len(X_explain):
                continue
                
            # Generate explanation
            explanation = self.lime_explainer.explain_instance(
                X_explain[idx], 
                self.model.predict_proba,
                num_features=num_features,
                num_samples=num_samples
            )
            explanations.append(explanation)
            
            # Save plot
            fig = explanation.as_pyplot_figure()
            fig.suptitle(f'LIME Explanation - Sample {idx}', fontsize=16)
            saved_plots[f'sample_{idx}'] = self._save_figure(fig, f"lime_explanation_sample_{idx}")
        
        self.lime_explanations_ = explanations
        
        # Create summary plot of feature importance across samples
        if len(explanations) > 1:
            feature_importance_summary = self._create_lime_summary(explanations, num_features)
            fig, ax = plt.subplots(figsize=(10, 6))
            
            features = list(feature_importance_summary.keys())
            importances = list(feature_importance_summary.values())
            
            ax.barh(features, importances)
            ax.set_xlabel('Average Absolute Importance')
            ax.set_title('LIME Feature Importance Summary')
            plt.tight_layout()
            
            saved_plots['summary'] = self._save_figure(fig, "lime_summary")
        
        result = {
            'explanations': explanations,
            'feature_names': self._get_feature_names(X_train),
            'saved_plots': saved_plots,
            'sample_indices': sample_indices
        }
        
        if self.verbose:
            logger.info(f"LIME analysis completed. Explained {len(explanations)} samples.")
        
        return result
    
    def _create_lime_summary(self, explanations: List, num_features: int) -> Dict[str, float]:
        """Create summary of LIME explanations across multiple samples."""
        feature_importance = {}
        
        for explanation in explanations:
            for feature, importance in explanation.as_list():
                if feature not in feature_importance:
                    feature_importance[feature] = []
                feature_importance[feature].append(abs(importance))
        
        # Calculate average absolute importance
        feature_avg_importance = {
            feature: np.mean(importances) 
            for feature, importances in feature_importance.items()
        }
        
        # Sort by importance and take top features
        sorted_features = sorted(feature_avg_importance.items(), 
                               key=lambda x: x[1], reverse=True)[:num_features]
        
        return dict(sorted_features)
    
    def permutation_importance(self,
                             X: np.ndarray,
                             y: np.ndarray,
                             n_repeats: int = 10,
                             random_state: Optional[int] = None,
                             scoring: str = 'accuracy',
                             n_jobs: int = -1,
                             max_features: int = 20) -> Dict[str, Any]:
        """
        Calculate and visualize permutation feature importance.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_repeats: Number of times to permute each feature
            random_state: Random state for reproducibility
            scoring: Scoring metric to use
            n_jobs: Number of parallel jobs
            max_features: Maximum number of features to display
            
        Returns:
            Dictionary containing importance scores and saved plot path
        """
        self._validate_inputs(X, y)
        
        if self.verbose:
            logger.info("Calculating permutation feature importance...")
        
        # Convert to numpy if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            self.model, X, y,
            n_repeats=n_repeats,
            random_state=random_state or self.random_state,
            scoring=scoring,
            n_jobs=n_jobs
        )
        
        self.permutation_importance_ = perm_importance
        
        # Create DataFrame for easier handling
        feature_names = self._get_feature_names(X)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Select top features
        top_features = importance_df.head(max_features)
        
        # Create horizontal bar plot with error bars
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['importance_mean'], 
                xerr=top_features['importance_std'],
                align='center', alpha=0.8, capsize=3)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()  # Top feature at the top
        ax.set_xlabel(f'Permutation Importance ({scoring})')
        ax.set_title(f'Permutation Feature Importance (±std, n={n_repeats})')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        saved_plot = self._save_figure(fig, "permutation_importance")
        
        result = {
            'importances_mean': perm_importance.importances_mean,
            'importances_std': perm_importance.importances_std,
            'importances': perm_importance.importances,
            'importance_df': importance_df,
            'feature_names': feature_names,
            'saved_plot': saved_plot,
            'scoring': scoring
        }
        
        if self.verbose:
            logger.info(f"Permutation importance calculated using {scoring} scoring.")
            logger.info(f"Top 5 features: {list(importance_df.head()['feature'])}")
        
        return result
    
    def partial_dependence_plot(self,
                               X: np.ndarray,
                               features: Union[List[int], List[str], int, str],
                               grid_resolution: int = 100,
                               percentiles: Tuple[float, float] = (0.05, 0.95),
                               kind: str = 'average',
                               subsample: Optional[int] = 1000,
                               n_jobs: int = -1) -> Dict[str, Any]:
        """
        Generate partial dependence plots.
        
        Args:
            X: Feature matrix
            features: Feature indices or names for PDP (can be single feature or list)
            grid_resolution: Number of points in the grid
            percentiles: Percentiles to use for feature range
            kind: Type of PDP ('average', 'individual', 'both')
            subsample: Number of samples to use (None for all)
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary containing PDP results and saved plot paths
        """
        self._validate_inputs(X)
        
        if self.verbose:
            logger.info("Generating partial dependence plots...")
        
        # Convert to numpy if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Subsample if requested
        if subsample and subsample < len(X):
            np.random.seed(self.random_state)
            indices = np.random.choice(len(X), subsample, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Handle feature specification
        if isinstance(features, (int, str)):
            features = [features]
        
        # Convert feature names to indices if needed
        feature_indices = []
        feature_names = self._get_feature_names(X)
        
        for feature in features:
            if isinstance(feature, str):
                if feature in feature_names:
                    feature_indices.append(feature_names.index(feature))
                else:
                    raise ValueError(f"Feature '{feature}' not found in feature names")
            else:
                feature_indices.append(feature)
        
        # Calculate partial dependence
        pd_results = partial_dependence(
            self.model, X_sample, 
            features=feature_indices,
            grid_resolution=grid_resolution,
            percentiles=percentiles,
            kind=kind,
            n_jobs=n_jobs
        )
        
        self.partial_dependence_ = pd_results
        
        saved_plots = {}
        
        # Create plots for each feature
        for i, feature_idx in enumerate(feature_indices):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get the results for this feature
            pd_values = pd_results['average'][i] if 'average' in pd_results else pd_results[0][i]
            grid_values = pd_results['grid_values'][i] if 'grid_values' in pd_results else pd_results[1][i]
            
            # Plot
            ax.plot(grid_values, pd_values, linewidth=2, label='Partial Dependence')
            
            # Add individual lines if requested
            if kind in ['individual', 'both'] and 'individual' in pd_results:
                individual_values = pd_results['individual'][i]
                for j in range(min(50, individual_values.shape[0])):  # Show max 50 individual lines
                    ax.plot(grid_values, individual_values[j], 
                           color='gray', alpha=0.1, linewidth=0.5)
            
            ax.set_xlabel(f'{feature_names[feature_idx]}')
            ax.set_ylabel('Partial Dependence')
            ax.set_title(f'Partial Dependence Plot - {feature_names[feature_idx]}')
            ax.grid(True, alpha=0.3)
            
            if kind in ['individual', 'both']:
                ax.legend()
            
            plt.tight_layout()
            saved_plots[feature_names[feature_idx]] = self._save_figure(
                fig, f"pdp_{feature_names[feature_idx].replace(' ', '_').lower()}")
        
        # Create summary plot if multiple features
        if len(feature_indices) > 1:
            n_cols = min(3, len(feature_indices))
            n_rows = (len(feature_indices) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, feature_idx in enumerate(feature_indices):
                ax = axes[i] if len(feature_indices) > 1 else axes
                
                pd_values = pd_results['average'][i] if 'average' in pd_results else pd_results[0][i]
                grid_values = pd_results['grid_values'][i] if 'grid_values' in pd_results else pd_results[1][i]
                
                ax.plot(grid_values, pd_values, linewidth=2)
                ax.set_xlabel(f'{feature_names[feature_idx]}')
                ax.set_ylabel('Partial Dependence')
                ax.set_title(f'PDP - {feature_names[feature_idx]}')
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for j in range(len(feature_indices), len(axes)):
                axes[j].axis('off')
            
            plt.tight_layout()
            saved_plots['summary'] = self._save_figure(fig, "pdp_summary")
        
        result = {
            'partial_dependence': pd_results,
            'feature_names': [feature_names[i] for i in feature_indices],
            'feature_indices': feature_indices,
            'grid_resolution': grid_resolution,
            'kind': kind,
            'saved_plots': saved_plots
        }
        
        if self.verbose:
            logger.info(f"Partial dependence plots generated for {len(feature_indices)} features.")
        
        return result
    
    def generate_comprehensive_report(self,
                                    X_train: np.ndarray,
                                    X_test: np.ndarray,
                                    y_train: np.ndarray,
                                    y_test: np.ndarray,
                                    include_shap: bool = True,
                                    include_lime: bool = True,
                                    include_permutation: bool = True,
                                    include_pdp: bool = True,
                                    top_features_pdp: int = 5) -> Dict[str, Any]:
        """
        Generate comprehensive interpretation report with all methods.
        
        Args:
            X_train: Training features
            X_test: Test features  
            y_train: Training targets
            y_test: Test targets
            include_shap: Whether to include SHAP analysis
            include_lime: Whether to include LIME analysis
            include_permutation: Whether to include permutation importance
            include_pdp: Whether to include partial dependence plots
            top_features_pdp: Number of top features for PDP
            
        Returns:
            Dictionary containing all interpretation results
        """
        if self.verbose:
            logger.info("Generating comprehensive interpretation report...")
        
        report = {
            'model_type': type(self.model).__name__,
            'feature_names': self._get_feature_names(X_train),
            'n_features': X_train.shape[1],
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test)
        }
        
        # SHAP Analysis
        if include_shap and SHAP_AVAILABLE:
            try:
                shap_results = self.explain_shap(X_test, sample_size=min(500, len(X_test)))
                report['shap'] = shap_results
                logger.info("✓ SHAP analysis completed")
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {e}")
                report['shap'] = {'error': str(e)}
        
        # LIME Analysis
        if include_lime and LIME_AVAILABLE:
            try:
                lime_results = self.explain_lime(
                    X_train, X_test[:min(3, len(X_test))], y_train
                )
                report['lime'] = lime_results
                logger.info("✓ LIME analysis completed")
            except Exception as e:
                logger.warning(f"LIME analysis failed: {e}")
                report['lime'] = {'error': str(e)}
        
        # Permutation Importance
        if include_permutation:
            try:
                perm_results = self.permutation_importance(X_test, y_test)
                report['permutation_importance'] = perm_results
                logger.info("✓ Permutation importance completed")
            except Exception as e:
                logger.warning(f"Permutation importance failed: {e}")
                report['permutation_importance'] = {'error': str(e)}
        
        # Partial Dependence Plots
        if include_pdp:
            try:
                # Get top features from permutation importance if available
                if 'permutation_importance' in report and 'importance_df' in report['permutation_importance']:
                    top_features = report['permutation_importance']['importance_df'].head(top_features_pdp)['feature'].tolist()
                else:
                    # Use first N features as fallback
                    top_features = list(range(min(top_features_pdp, X_train.shape[1])))
                
                pdp_results = self.partial_dependence_plot(X_train, top_features)
                report['partial_dependence'] = pdp_results
                logger.info("✓ Partial dependence plots completed")
            except Exception as e:
                logger.warning(f"Partial dependence plots failed: {e}")
                report['partial_dependence'] = {'error': str(e)}
        
        # Create summary report file
        self._create_summary_report(report)
        
        if self.verbose:
            logger.info("Comprehensive interpretation report generated successfully")
            logger.info(f"All visualizations saved to: {self.save_dir}")
        
        return report
    
    def _create_summary_report(self, report: Dict[str, Any]) -> Path:
        """Create a text summary of the interpretation results."""
        summary_path = self.save_dir / "interpretation_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("MODEL INTERPRETATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model Type: {report['model_type']}\n")
            f.write(f"Number of Features: {report['n_features']}\n")
            f.write(f"Training Samples: {report['n_samples_train']}\n")
            f.write(f"Test Samples: {report['n_samples_test']}\n\n")
            
            # SHAP Summary
            if 'shap' in report:
                f.write("SHAP ANALYSIS\n")
                f.write("-" * 20 + "\n")
                if 'error' in report['shap']:
                    f.write(f"Error: {report['shap']['error']}\n")
                else:
                    f.write(f"Explainer Type: {report['shap']['explainer_type']}\n")
                    f.write(f"Plots Generated: {', '.join(report['shap']['saved_plots'].keys())}\n")
                f.write("\n")
            
            # LIME Summary
            if 'lime' in report:
                f.write("LIME ANALYSIS\n")
                f.write("-" * 20 + "\n")
                if 'error' in report['lime']:
                    f.write(f"Error: {report['lime']['error']}\n")
                else:
                    f.write(f"Samples Explained: {len(report['lime']['explanations'])}\n")
                    f.write(f"Plots Generated: {', '.join(report['lime']['saved_plots'].keys())}\n")
                f.write("\n")
            
            # Permutation Importance Summary
            if 'permutation_importance' in report:
                f.write("PERMUTATION IMPORTANCE\n")
                f.write("-" * 25 + "\n")
                if 'error' in report['permutation_importance']:
                    f.write(f"Error: {report['permutation_importance']['error']}\n")
                else:
                    df = report['permutation_importance']['importance_df']
                    f.write("Top 10 Most Important Features:\n")
                    for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
                        f.write(f"{i:2d}. {row['feature']}: {row['importance_mean']:.4f} (±{row['importance_std']:.4f})\n")
                f.write("\n")
            
            # Partial Dependence Summary
            if 'partial_dependence' in report:
                f.write("PARTIAL DEPENDENCE PLOTS\n")
                f.write("-" * 30 + "\n")
                if 'error' in report['partial_dependence']:
                    f.write(f"Error: {report['partial_dependence']['error']}\n")
                else:
                    f.write(f"Features Analyzed: {', '.join(report['partial_dependence']['feature_names'])}\n")
                    f.write(f"Plots Generated: {', '.join(report['partial_dependence']['saved_plots'].keys())}\n")
                f.write("\n")
            
            f.write(f"Report generated at: {utils.get_timestamp()}\n")
            f.write(f"All visualizations saved to: {self.save_dir}\n")
        
        if self.verbose:
            logger.info(f"Summary report saved to: {summary_path}")
        
        return summary_path
    
    def get_interpretation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all computed interpretations.
        
        Returns:
            Dictionary summarizing available interpretations
        """
        summary = {
            'model_type': type(self.model).__name__,
            'available_methods': [],
            'computed_interpretations': {}
        }
        
        # Check what's available
        if SHAP_AVAILABLE:
            summary['available_methods'].append('SHAP')
        if LIME_AVAILABLE:
            summary['available_methods'].append('LIME')
        summary['available_methods'].extend(['Permutation Importance', 'Partial Dependence'])
        
        # Check what's computed
        if self.shap_values_ is not None:
            summary['computed_interpretations']['SHAP'] = {
                'computed': True,
                'type': type(self.shap_explainer).__name__ if self.shap_explainer else None
            }
        
        if self.lime_explanations_ is not None:
            summary['computed_interpretations']['LIME'] = {
                'computed': True,
                'n_explanations': len(self.lime_explanations_)
            }
        
        if self.permutation_importance_ is not None:
            summary['computed_interpretations']['Permutation Importance'] = {
                'computed': True,
                'n_features': len(self.permutation_importance_.importances_mean)
            }
        
        if self.partial_dependence_ is not None:
            summary['computed_interpretations']['Partial Dependence'] = {
                'computed': True,
                'n_features': len(self.partial_dependence_['grid_values']) if 'grid_values' in self.partial_dependence_ else len(self.partial_dependence_[1])
            }
        
        return summary
