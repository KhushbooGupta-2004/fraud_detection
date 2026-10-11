import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
)
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


def evaluate_predictions(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Compute all classification metrics."""
    metrics = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_proba)
    return metrics


def compare_models(
    y_true: np.ndarray, 
    results: Dict[str, Dict],
    metric: str = "F1_Score"
) -> pd.DataFrame:
    """Compare all models at default threshold (0.5)."""
    rows = []
    for name, metrics in results.items():
        rows.append({
            "Model": name,
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "F1_Score": metrics.get("f1_score", 0),
            "ROC_AUC": metrics.get("roc_auc"),
            "PR_AUC": metrics.get("pr_auc"),
        })
    df = pd.DataFrame(rows).set_index("Model")
    return df.sort_values(metric, ascending=False)


def threshold_sweep_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Compute metrics across a range of thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)
    
    rows = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        rows.append({
            "Threshold": t,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        })
    
    return pd.DataFrame(rows)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "F1_Score",
    thresholds: Optional[np.ndarray] = None
) -> tuple:
    """Find threshold that optimizes a given metric."""
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)
    
    sweep_df = threshold_sweep_metrics(y_true, y_proba, thresholds)
    best_idx = sweep_df[metric].idxmax()
    best_threshold = sweep_df.loc[best_idx, "Threshold"]
    best_metrics = sweep_df.loc[best_idx].to_dict()
    return best_threshold, best_metrics, sweep_df


def plot_precision_recall_curve(
    y_true: np.ndarray,
    models_proba: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    show: bool = False
):
    """Plot PR curves for all models."""
    plt.figure(figsize=(8, 6))
    
    for name, proba in models_proba.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.3f})")
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    models_proba: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    show: bool = False
):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(8, 6))
    
    for name, proba in models_proba.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (ROC-AUC = {roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrices(
    y_true: np.ndarray,
    models_preds: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    show: bool = False
):
    """Plot confusion matrices side by side."""
    n_models = len(models_preds)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, preds) in zip(axes, models_preds.items()):
        cm = confusion_matrix(y_true, preds)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def print_results(results: Dict[str, Dict]):
    """Pretty print model comparison."""
    for name, metrics in results.items():
        print(f"\n{name}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")


def get_best_model(results: Dict[str, Dict], metric: str = "f1_score") -> str:
    """Return name of best model by specified metric."""
    return max(results, key=lambda k: results[k][metric])


if __name__ == "__main__":
    # Demo with synthetic data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    
    X, y = make_classification(n_samples=10000, n_features=20, weights=[0.99, 0.01], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    lr = LogisticRegression(class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    proba = lr.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    
    results = {"Logistic Regression": {"pred": pred, "proba": proba}}
    print_results(results)
    plot_precision_recall_curve(y_test, {"Logistic Regression": proba})
    plot_roc_curve(y_test, {"Logistic Regression": proba})
    plot_confusion_matrices(y_test, {"Logistic Regression": pred})
