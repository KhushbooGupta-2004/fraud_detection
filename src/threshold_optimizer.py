import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# Default cost constants (can be overridden)
DEFAULT_C_FN = 10000  # Cost for missed fraud
DEFAULT_C_FP = 500    # Cost for false alarm


def compute_cost(
    y_true: np.ndarray, 
    y_proba: np.ndarray, 
    threshold: float, 
    c_fn: float = DEFAULT_C_FN, 
    c_fp: float = DEFAULT_C_FP
) -> Tuple[float, Dict[str, int]]:
    """Compute total cost and confusion matrix at a given threshold."""
    preds = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    cost = c_fn * fn + c_fp * fp
    return cost, {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    c_fn: float = DEFAULT_C_FN,
    c_fp: float = DEFAULT_C_FP,
    thresholds: Optional[np.ndarray] = None
) -> Tuple[float, float, List[float]]:
    """Find threshold that minimizes total cost."""
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)
    
    best_threshold = 0.5
    best_cost = float('inf')
    all_costs = []
    
    for threshold in thresholds:
        cost, _ = compute_cost(y_true, y_proba, threshold, c_fn, c_fp)
        all_costs.append(cost)
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold
    
    return best_threshold, best_cost, all_costs


def find_f1_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """Find threshold that maximizes F1 score."""
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)
    
    f1_scores = [f1_score(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def evaluate_strategies(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    c_fn: float = DEFAULT_C_FN,
    c_fp: float = DEFAULT_C_FP,
    thresholds: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Compare three threshold strategies: default, F1-optimal, cost-optimal."""
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)
    
    # Find optimal thresholds
    cost_threshold, cost_min, all_costs = find_cost_optimal_threshold(
        y_true, y_proba, c_fn, c_fp, thresholds
    )
    f1_threshold, f1_max = find_f1_optimal_threshold(y_true, y_proba, thresholds)
    
    # Evaluate three strategies
    strategies = {
        "Default (0.50)": 0.5,
        f"F1-Optimal ({f1_threshold:.2f})": f1_threshold,
        f"Cost-Optimal ({cost_threshold:.2f})": cost_threshold,
    }
    
    comparison_rows = []
    for name, threshold in strategies.items():
        preds = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        cost = c_fn * fn + c_fp * fp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        comparison_rows.append({
            "Strategy": name,
            "Threshold": round(threshold, 2),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1_Score": round(f1, 4),
            "Total_Cost": cost,
        })
    
    return pd.DataFrame(comparison_rows), all_costs, cost_threshold, f1_threshold


def optimize_all_models(
    y_true: np.ndarray,
    models_proba: Dict[str, np.ndarray],
    c_fn: float = DEFAULT_C_FN,
    c_fp: float = DEFAULT_C_FP,
    thresholds: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, Any]]:
    """Run threshold optimization for each model independently."""
    results = {}
    for name, proba in models_proba.items():
        logger.info(f"Optimizing threshold for {name}...")
        comparison_df, all_costs, cost_thresh, f1_thresh = evaluate_strategies(
            y_true, proba, c_fn, c_fp, thresholds
        )
        results[name] = {
            "comparison": comparison_df,
            "cost_curve": all_costs,
            "cost_optimal_threshold": cost_thresh,
            "f1_optimal_threshold": f1_thresh,
            "thresholds": thresholds if thresholds is not None else np.arange(0.01, 1.00, 0.01),
        }
    return results


def plot_cost_curves(
    optimization_results: Dict[str, Dict[str, Any]],
    c_fn: float = DEFAULT_C_FN,
    c_fp: float = DEFAULT_C_FP,
    save_path: Optional[str] = None,
    show: bool = False
):
    """Plot cost vs threshold for all models."""
    plt.figure(figsize=(10, 6))
    
    for name, res in optimization_results.items():
        thresholds = res["thresholds"]
        costs = res["cost_curve"]
        best_thresh = res["cost_optimal_threshold"]
        
        plt.plot(thresholds, costs, label=name, linewidth=2)
        plt.axvline(x=best_thresh, linestyle='--', alpha=0.7)
    
    plt.axvline(x=0.5, color='red', linestyle='--', label='Default (0.5)', alpha=0.5)
    plt.title(f'Total Cost vs Decision Threshold (C_FN={c_fn}, C_FP={c_fp})')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Total Cost (Rs.)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def calculate_savings(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    c_fn: float = DEFAULT_C_FN,
    c_fp: float = DEFAULT_C_FP,
    annual_transactions: int = 60_000_000
) -> Dict[str, float]:
    """Calculate savings from cost-optimized threshold vs default."""
    # Default threshold
    preds_default = (y_proba >= 0.5).astype(int)
    tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_true, preds_default).ravel()
    cost_default = c_fn * fn_d + c_fp * fp_d
    
    # Cost-optimal threshold
    best_thresh, _, _ = find_cost_optimal_threshold(y_true, y_proba, c_fn, c_fp)
    preds_optimal = (y_proba >= best_thresh).astype(int)
    tn_o, fp_o, fn_o, tp_o = confusion_matrix(y_true, preds_optimal).ravel()
    cost_optimal = c_fn * fn_o + c_fp * fp_o
    
    savings = cost_default - cost_optimal
    savings_per_10k = savings / len(y_true) * 10000
    annual_savings = savings_per_10k * (annual_transactions / 10000)
    
    return {
        "cost_default": cost_default,
        "cost_optimal": cost_optimal,
        "savings_test": savings,
        "savings_per_10k": savings_per_10k,
        "annual_savings": annual_savings,
        "annual_savings_crore": annual_savings / 1e7,
        "best_threshold": best_thresh,
    }


def save_threshold_analysis(
    optimization_results: Dict[str, Dict[str, Any]],
    output_dir: str = "reports/figures"
):
    """Save threshold analysis data for Power BI."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for name, res in optimization_results.items():
        df = pd.DataFrame({
            "threshold": res["thresholds"],
            "total_cost": res["cost_curve"],
        })
        df["model"] = name
        df.to_csv(Path(output_dir) / f"threshold_analysis_{name.lower().replace(' ', '_')}.csv", index=False)
    
    # Save comparison
    comparison_rows = []
    for name, res in optimization_results.items():
        comp = res["comparison"]
        for _, row in comp.iterrows():
            row_dict = row.to_dict()
            row_dict["model"] = name
            comparison_rows.append(row_dict)
    
    pd.DataFrame(comparison_rows).to_csv(
        Path(output_dir) / "threshold_strategy_comparison.csv", index=False
    )
    
    logger.info(f"Threshold analysis saved to {output_dir}/")


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
    
    # Run optimization
    comparison_df, costs, cost_thresh, f1_thresh = evaluate_strategies(y_test.values, proba)
    print("Strategy Comparison:")
    print(comparison_df.to_string(index=False))
    
    savings = calculate_savings(y_test.values, proba)
    print(f"\nTest set savings: ₹{savings['savings_test']:,.0f}")
    print(f"Savings per 10K: ₹{savings['savings_per_10k']:,.0f}")
    print(f"Annual projection: ₹{savings['annual_savings_crore']:.2f} crore")
    print(f"Best threshold: {savings['best_threshold']:.2f}")
    
    plot_cost_curves({"Logistic Regression": {
        "thresholds": np.arange(0.01, 1.00, 0.01),
        "cost_curve": costs,
        "cost_optimal_threshold": cost_thresh,
    }})


