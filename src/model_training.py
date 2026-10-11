import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
from pathlib import Path

from src.data_preprocessing import load_and_preprocess


def train_models(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series,
    X_train_scaled: pd.DataFrame, X_test_scaled: pd.DataFrame,
    calibrate_xgb: bool = True,
    cv_folds: int = 5
) -> dict:
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    models = {}
    results = {}

    # --- Logistic Regression (uses scaled data) ---
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=1)
    lr.fit(X_train_scaled, y_train)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
    lr_pred = (lr_proba >= 0.5).astype(int)
    models["Logistic Regression"] = lr

    # --- Random Forest (uses raw data - trees don't need scaling) ---
    rf = RandomForestClassifier(
        class_weight='balanced', n_estimators=200, random_state=1, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_pred = (rf_proba >= 0.5).astype(int)
    models["Random Forest"] = rf

    # --- XGBoost (uses raw data, with optional calibration) ---
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=1,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb.fit(X_train, y_train, verbose=False)
    
    if calibrate_xgb:
        # Calibrate on a validation split to avoid overfitting
        xgb_cal = CalibratedClassifierCV(xgb, method='isotonic', cv=3)
        xgb_cal.fit(X_train, y_train)
        xgb_proba = xgb_cal.predict_proba(X_test)[:, 1]
        models["XGBoost"] = xgb_cal
    else:
        xgb_proba = xgb.predict_proba(X_test)[:, 1]
        models["XGBoost"] = xgb
    
    xgb_pred = (xgb_proba >= 0.5).astype(int)

    # --- Evaluate all models at default threshold (0.5) ---
    model_data = {
        "Logistic Regression": (lr, lr_pred, lr_proba, X_test_scaled),
        "Random Forest": (rf, rf_pred, rf_proba, X_test),
        "XGBoost": (models["XGBoost"], xgb_pred, xgb_proba, X_test),
    }

    for name, (model, pred, proba, X_eval) in model_data.items():
        results[name] = {
            "model": model,
            "pred": pred,
            "proba": proba,
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1_score": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),  # FIXED: use proba not pred
            "pr_auc": average_precision_score(y_test, proba),
            "confusion_matrix": confusion_matrix(y_test, pred),
            "classification_report": classification_report(y_test, pred, output_dict=True),
        }

    # Cross-validation for robust comparison
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=1)
    cv_results = {}
    for name, model in models.items():
        if name == "Logistic Regression":
            X_cv, y_cv = X_train_scaled, y_train
        else:
            X_cv, y_cv = X_train, y_train
        
        cv_roc = cross_val_score(model, X_cv, y_cv, cv=cv, scoring='roc_auc', n_jobs=-1)
        cv_pr = cross_val_score(model, X_cv, y_cv, cv=cv, scoring='average_precision', n_jobs=-1)
        cv_results[name] = {
            "roc_auc_mean": cv_roc.mean(),
            "roc_auc_std": cv_roc.std(),
            "pr_auc_mean": cv_pr.mean(),
            "pr_auc_std": cv_pr.std(),
        }
        results[name]["cv"] = cv_results[name]

    return {
        "models": models,
        "results": results,
        "cv_results": cv_results,
    }


def select_best_model(
    results: dict,
    X_test: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
    models: dict,
    selection_criteria: str = "cost_optimal_recall",
    c_fn: float = 10000,
    c_fp: float = 500
) -> dict:
    """Select best model based on specified criteria.
    
        selection_criteria: One of:
            - "cost_optimal_recall": Highest recall at cost-optimal threshold
            - "cost_optimal_f1": Highest F1 at cost-optimal threshold
            - "min_total_cost": Lowest total cost at cost-optimal threshold
            - "recall_at_05": Highest recall at default 0.5 threshold
            - "f1_at_05": Highest F1 at default 0.5 threshold
        c_fn: Cost of false negative
        c_fp: Cost of false positive
        
    """
    from src.threshold_optimizer import find_cost_optimal_threshold
    
    y_true = y_test.values
    best_model_name = None
    best_score = float('-inf')
    best_threshold = 0.5
    best_metrics = {}
    
    for name, res in results.items():
        if name == "Logistic Regression":
            proba = res["proba"]
        else:
            proba = res["proba"]
        
        if selection_criteria == "cost_optimal_recall":
            # Find cost-optimal threshold, then evaluate recall
            best_thresh, _, _ = find_cost_optimal_threshold(y_true, proba, c_fn, c_fp)
            preds = (proba >= best_thresh).astype(int)
            from sklearn.metrics import recall_score
            score = recall_score(y_true, preds)
            threshold = best_thresh
            
        elif selection_criteria == "cost_optimal_f1":
            best_thresh, _, _ = find_cost_optimal_threshold(y_true, proba, c_fn, c_fp)
            preds = (proba >= best_thresh).astype(int)
            from sklearn.metrics import f1_score
            score = f1_score(y_true, preds)
            threshold = best_thresh
            
        elif selection_criteria == "min_total_cost":
            best_thresh, min_cost, _ = find_cost_optimal_threshold(y_true, proba, c_fn, c_fp)
            score = -min_cost  # Negative because we want to maximize
            threshold = best_thresh
            
        elif selection_criteria == "recall_at_05":
            from sklearn.metrics import recall_score
            preds = (proba >= 0.5).astype(int)
            score = recall_score(y_true, preds)
            threshold = 0.5
            
        elif selection_criteria == "f1_at_05":
            from sklearn.metrics import f1_score
            preds = (proba >= 0.5).astype(int)
            score = f1_score(y_true, preds)
            threshold = 0.5
            
        else:
            raise ValueError(f"Unknown selection_criteria: {selection_criteria}")
        
        if score > best_score:
            best_score = score
            best_model_name = name
            best_threshold = threshold
            best_metrics = {
                "selection_criteria": selection_criteria,
                "score": score,
                "threshold": threshold,
            }
    
    # Compute full metrics at best threshold
    best_proba = results[best_model_name]["proba"]
    best_preds = (best_proba >= best_threshold).astype(int)
    
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, 
        roc_auc_score, average_precision_score, confusion_matrix
    )
    
    best_metrics.update({
        "model_name": best_model_name,
        "threshold": best_threshold,
        "precision": precision_score(y_true, best_preds),
        "recall": recall_score(y_true, best_preds),
        "f1_score": f1_score(y_true, best_preds),
        "roc_auc": roc_auc_score(y_true, best_proba),
        "pr_auc": average_precision_score(y_true, best_proba),
        "confusion_matrix": confusion_matrix(y_true, best_preds),
        "total_cost": c_fn * confusion_matrix(y_true, best_preds)[1, 0] + 
                      c_fp * confusion_matrix(y_true, best_preds)[0, 1],
    })
    
    return best_metrics


def save_models(models: dict, output_dir: str = "models") -> None:
    """Save trained models to disk."""
    Path(output_dir).mkdir(exist_ok=True)
    for name, model in models.items():
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, Path(output_dir) / filename)
    print(f"Models saved to {output_dir}/")


def load_models(model_dir: str = "models") -> dict:
    """Load trained models from disk."""
    models = {}
    name_mapping = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }
    for f in Path(model_dir).glob("*.joblib"):
        key = f.stem
        if key in name_mapping:
            name = name_mapping[key]
            models[name] = joblib.load(f)
    return models


if __name__ == "__main__":
    # Load and preprocess data
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, sc = load_and_preprocess()
    
    # Train models
    output = train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
    results = output["results"]
    models = output["models"]
    
    # Print results
    for name, metrics in results.items():
        print(f"\n{name}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        if 'cv' in metrics:
            print(f"  CV ROC-AUC: {metrics['cv']['roc_auc_mean']:.4f} ± {metrics['cv']['roc_auc_std']:.4f}")
            print(f"  CV PR-AUC:  {metrics['cv']['pr_auc_mean']:.4f} ± {metrics['cv']['pr_auc_std']:.4f}")
        print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    # Best model by F1
    best_model_name = max(results, key=lambda k: results[k]["f1_score"])
    print(f"\nBest Model by F1 Score: {best_model_name}")
    
    # Save models
    save_models(models)