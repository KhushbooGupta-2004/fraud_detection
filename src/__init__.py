"""
Fraud Detection Pipeline - India Edition
Credit card fraud detection with cost-sensitive threshold optimization.
"""

__version__ = "1.0.0"
__author__ = "Khushboo Gupta"

from .data_preprocessing import load_and_preprocess
from .model_training import train_models, save_models, load_models, select_best_model
from .evaluation import (
    evaluate_predictions,
    compare_models,
    threshold_sweep_metrics,
    find_optimal_threshold,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_confusion_matrices,
    print_results,
    get_best_model,
)
from .threshold_optimizer import (
    compute_cost,
    find_cost_optimal_threshold,
    find_f1_optimal_threshold,
    evaluate_strategies,
    optimize_all_models,
    plot_cost_curves,
    calculate_savings,
    save_threshold_analysis,
)
from .explainability import (
    compute_shap_values,
    get_feature_importance,
    plot_shap_summary,
    plot_shap_bar,
    plot_shap_waterfall,
    plot_shap_force,
    get_top_fraud_drivers,
    explain_model,
    save_shap_values,
)

__all__ = [
    # Data preprocessing
    "load_and_preprocess",
    # Model training
    "train_models",
    "save_models",
    "load_models",
    "select_best_model",
    # Evaluation
    "evaluate_predictions",
    "compare_models",
    "threshold_sweep_metrics",
    "find_optimal_threshold",
    "plot_precision_recall_curve",
    "plot_roc_curve",
    "plot_confusion_matrices",
    "print_results",
    "get_best_model",
    # Threshold optimization
    "compute_cost",
    "find_cost_optimal_threshold",
    "find_f1_optimal_threshold",
    "evaluate_strategies",
    "optimize_all_models",
    "plot_cost_curves",
    "calculate_savings",
    "save_threshold_analysis",
    # Explainability
    "compute_shap_values",
    "get_feature_importance",
    "plot_shap_summary",
    "plot_shap_bar",
    "plot_shap_waterfall",
    "plot_shap_force",
    "get_top_fraud_drivers",
    "explain_model",
    "save_shap_values",
]