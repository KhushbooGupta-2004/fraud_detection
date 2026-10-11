import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import joblib
import logging

logger = logging.getLogger(__name__)


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    sample_size: int = 1000,
    random_state: int = 42
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values for any supported model type.
    
    Args:
        model: Trained model (tree-based or linear)
        X: Feature matrix
        sample_size: Number of samples to use (for speed)
        random_state: Random seed for sampling
        
    Returns:
        Tuple of (shap_values, X_sample)
    """
    # Sample for speed
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=random_state)
    else:
        X_sample = X
    
    # Determine model type and use appropriate explainer
    model_type = type(model).__name__
    
    # Tree-based models: RandomForest, XGBoost, CalibratedClassifierCV (with tree base)
    if any(t in model_type for t in ['RandomForest', 'XGB', 'GradientBoosting', 'DecisionTree', 'CalibratedClassifierCV']):
        # For calibrated models, get the base estimator
        if hasattr(model, 'base_estimator'):
            base_model = model.base_estimator
        elif hasattr(model, 'calibrated_classifiers_') and len(model.calibrated_classifiers_) > 0:
            # CalibratedClassifierCV with CV > 1 stores calibrated classifiers
            base_model = model.calibrated_classifiers_[0].estimator
        elif hasattr(model, 'estimator') and 'CalibratedClassifierCV' in model_type:
            # Only use estimator for CalibratedClassifierCV, not for RandomForest etc.
            base_model = model.estimator
        else:
            base_model = model
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_sample)
    
    # Linear models: LogisticRegression, LinearRegression
    elif 'LogisticRegression' in model_type or 'LinearRegression' in model_type:
        # Use KernelExplainer with small background for speed
        # Sample background data
        background = X_sample.sample(n=min(50, len(X_sample)), random_state=random_state)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample)
        # KernelExplainer for binary classification returns (n_samples, n_features, n_classes)
        # Select class 1 (fraud) - index 1
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        elif isinstance(shap_values, list):
            shap_values = shap_values[1]
    
    # Fallback: PermutationExplainer (model-agnostic, slower)
    else:
        logger.warning(f"Model type {model_type} not directly supported, using PermutationExplainer")
        explainer = shap.PermutationExplainer(model.predict_proba, X_sample)
        shap_values = explainer.shap_values(X_sample)
    
    # For binary classification, shap_values can be:
    # - list of 2 arrays (class 0, class 1) - older SHAP versions
    # - 3D array (n_samples, n_features, n_classes) - newer SHAP versions
    # We want class 1 (fraud) - positive values push toward fraud
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    
    return shap_values, X_sample


def get_feature_importance(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame
) -> pd.DataFrame:
    """Get feature importance ranked by mean absolute SHAP value."""
    importance = pd.DataFrame({
        'feature': X_sample.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    return importance


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    max_display: int = 15,
    save_path: Optional[str] = None
):
    """Plot SHAP summary (beeswarm) plot."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
    plt.title("SHAP Feature Importance - What Drives Fraud Predictions?")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_shap_bar(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    max_display: int = 15,
    save_path: Optional[str] = None
):
    """Plot SHAP bar chart (mean |SHAP| per feature)."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=max_display, show=False)
    plt.title("Mean |SHAP Value| per Feature")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_shap_waterfall(
    model: Any,
    X: pd.DataFrame,
    instance_idx: int = 0,
    save_path: Optional[str] = None
):
    """Plot SHAP waterfall for a single prediction."""
    model_type = type(model).__name__
    
    # Tree-based models
    if any(t in model_type for t in ['RandomForest', 'XGB', 'GradientBoosting', 'DecisionTree', 'CalibratedClassifierCV']):
        if hasattr(model, 'base_estimator'):
            base_model = model.base_estimator
        elif hasattr(model, 'calibrated_classifiers_') and len(model.calibrated_classifiers_) > 0:
            base_model = model.calibrated_classifiers_[0].estimator
        elif hasattr(model, 'estimator') and 'CalibratedClassifierCV' in model_type:
            base_model = model.estimator
        else:
            base_model = model
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X.iloc[[instance_idx]])
        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
    
    # Linear models: LogisticRegression
    elif 'LogisticRegression' in model_type:
        background = X.sample(n=min(50, len(X)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X.iloc[[instance_idx]])
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        elif isinstance(shap_values, list):
            shap_values = shap_values[1]
        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
    
    else:
        logger.warning(f"Model type {model_type} not supported for waterfall plot")
        return
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # KernelExplainer for single instance returns (1, n_features, n_classes)
        shap_values = shap_values[0, :, 1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        # TreeExplainer for single instance returns (n_features, n_classes)
        shap_values = shap_values[:, 1]
    
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap.Explanation(
        values=shap_values,
        base_values=expected_value,
        data=X.iloc[instance_idx].values,
        feature_names=X.columns.tolist()
    ), show=False)
    plt.title(f"SHAP Waterfall - Instance {instance_idx}")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_shap_force(
    model: Any,
    X: pd.DataFrame,
    instance_idx: int = 0,
    save_path: Optional[str] = None
):
    """Plot SHAP force plot for a single prediction."""
    model_type = type(model).__name__
    
    # Tree-based models
    if any(t in model_type for t in ['RandomForest', 'XGB', 'GradientBoosting', 'DecisionTree', 'CalibratedClassifierCV']):
        if hasattr(model, 'base_estimator'):
            base_model = model.base_estimator
        elif hasattr(model, 'calibrated_classifiers_') and len(model.calibrated_classifiers_) > 0:
            base_model = model.calibrated_classifiers_[0].estimator
        elif hasattr(model, 'estimator') and 'CalibratedClassifierCV' in model_type:
            base_model = model.estimator
        else:
            base_model = model
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X.iloc[[instance_idx]])
        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
    
    # Linear models: LogisticRegression
    elif 'LogisticRegression' in model_type:
        background = X.sample(n=min(50, len(X)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X.iloc[[instance_idx]])
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        elif isinstance(shap_values, list):
            shap_values = shap_values[1]
        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
    
    else:
        logger.warning(f"Model type {model_type} not supported for force plot")
        return
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[0, :, 1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        shap_values = shap_values[:, 1]
    
    shap.force_plot(
        expected_value,
        shap_values[0],
        X.iloc[instance_idx],
        feature_names=X.columns.tolist(),
        matplotlib=True,
        show=False
    )
    plt.title(f"SHAP Force Plot - Instance {instance_idx}")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def get_top_fraud_drivers(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    top_k: int = 10
) -> pd.DataFrame:
    """Get top features pushing toward fraud (positive SHAP) and legitimate (negative SHAP)."""
    mean_shap = pd.DataFrame({
        'feature': X_sample.columns,
        'mean_shap': shap_values.mean(axis=0),
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    # Top fraud drivers (positive SHAP)
    fraud_drivers = mean_shap[mean_shap['mean_shap'] > 0].head(top_k)
    # Top legit drivers (negative SHAP)
    legit_drivers = mean_shap[mean_shap['mean_shap'] < 0].head(top_k)
    
    return fraud_drivers, legit_drivers


def explain_model(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    output_dir: str = "reports/figures",
    sample_size: int = 1000
) -> Dict[str, Any]:
    """Full SHAP explainability pipeline for a model."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Computing SHAP values for {model_name}...")
    shap_values, X_sample = compute_shap_values(model, X_test, sample_size)
    
    logger.info(f"Generating feature importance for {model_name}...")
    importance = get_feature_importance(shap_values, X_sample)
    
    logger.info(f"Generating plots for {model_name}...")
    plot_shap_summary(shap_values, X_sample, save_path=f"{output_dir}/shap_summary_{model_name.lower().replace(' ', '_')}.png")
    plot_shap_bar(shap_values, X_sample, save_path=f"{output_dir}/shap_bar_{model_name.lower().replace(' ', '_')}.png")
    
    # Find a fraud case and a legit case for waterfall/force plots
    fraud_idx = y_test[y_test == 1].index[0] if (y_test == 1).any() else 0
    legit_idx = y_test[y_test == 0].index[0] if (y_test == 0).any() else 0
    
    # Get row indices in X_test
    fraud_row = X_test.index.get_loc(fraud_idx)
    legit_row = X_test.index.get_loc(legit_idx)
    
    plot_shap_waterfall(model, X_test, fraud_row, save_path=f"{output_dir}/shap_waterfall_fraud_{model_name.lower().replace(' ', '_')}.png")
    plot_shap_waterfall(model, X_test, legit_row, save_path=f"{output_dir}/shap_waterfall_legit_{model_name.lower().replace(' ', '_')}.png")
    
    # Top fraud drivers
    fraud_drivers, legit_drivers = get_top_fraud_drivers(shap_values, X_sample)
    
    # Save importance
    importance.to_csv(f"{output_dir}/shap_importance_{model_name.lower().replace(' ', '_')}.csv", index=False)
    
    results = {
        "importance": importance,
        "shap_values": shap_values,
        "X_sample": X_sample,
        "fraud_drivers": fraud_drivers,
        "legit_drivers": legit_drivers,
    }
    
    logger.info(f"SHAP analysis complete for {model_name}")
    return results


def save_shap_values(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    output_path: str
):
    """Save SHAP values for Power BI import."""
    df = pd.DataFrame(shap_values, columns=X_sample.columns)
    df.to_parquet(output_path, index=False)
    logger.info(f"SHAP values saved to {output_path}")


if __name__ == "__main__":
    # Demo with synthetic data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    X, y = make_classification(n_samples=5000, n_features=20, weights=[0.99, 0.01], random_state=42)
    feature_names = [f"feature_{i}" for i in range(20)]
    X = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    
    results = explain_model(rf, X_train, X_test, y_test, "Random Forest")
    print("Top 10 features by SHAP importance:")
    print(results["importance"].head(10))