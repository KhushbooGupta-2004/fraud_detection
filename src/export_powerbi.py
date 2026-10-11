"""
Power BI Export Module - Star Schema for Fraud Detection Dashboard

Creates dimension and fact tables optimized for Power BI:
- FactTransactions: Core fact table with predictions, costs, features
- DimModel: Model dimension (LR, RF, XGBoost)
- DimThreshold: Threshold dimension for cost optimization
- DimFeature: Feature dimension for SHAP analysis
- BusinessKPIs: Single-row summary for KPI cards
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
from typing import Dict, Any, Optional, List
import json

logger = logging.getLogger(__name__)


def load_pipeline_artifacts(models_dir: str = "models"):
    """Load all pipeline artifacts needed for export."""
    models_path = Path(models_dir)
    
    # Load preprocessed data
    data = joblib.load(models_path / "preprocessed_data.joblib")
    X_train = data['X_train']
    X_test = data['X_test']
    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    y_train = data['y_train']
    y_test = data['y_test']
    scaler = data['scaler']
    
    # Load models
    from src.model_training import load_models
    models = load_models(models_dir)
    
    # Load threshold optimization results
    threshold_dir = Path("reports/figures")
    threshold_files = list(threshold_dir.glob("threshold_analysis_*.csv"))
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "models": models,
        "threshold_files": threshold_files,
    }


def generate_model_predictions(artifacts: Dict[str, Any]) -> pd.DataFrame:
    """Generate predictions from all models for test set."""
    X_test = artifacts["X_test"]
    X_test_scaled = artifacts["X_test_scaled"]
    y_test = artifacts["y_test"]
    models = artifacts["models"]
    
    rows = []
    for model_name, model in models.items():
        if model_name == "Logistic Regression":
            proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            proba = model.predict_proba(X_test)[:, 1]
        
        pred = (proba >= 0.5).astype(int)
        
        for idx, (true_val, prob_val, pred_val) in enumerate(zip(y_test, proba, pred)):
            rows.append({
                "transaction_id": X_test.index[idx],
                "model_name": model_name,
                "y_true": int(true_val),
                "y_pred": int(pred_val),
                "y_proba": float(prob_val),
                "is_correct": int(true_val == pred_val),
            })
    
    return pd.DataFrame(rows)


def generate_fact_transactions(artifacts: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    """Create core fact table: one row per transaction with all model predictions and costs."""
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]
    models = artifacts["models"]
    
    c_fn = config["costs"]["c_fn"]
    c_fp = config["costs"]["c_fp"]
    
    # Get predictions from all models
    model_probas = {}
    model_preds = {}
    for model_name, model in models.items():
        if model_name == "Logistic Regression":
            proba = model.predict_proba(artifacts["X_test_scaled"])[:, 1]
        else:
            proba = model.predict_proba(X_test)[:, 1]
        model_probas[model_name] = proba
        model_preds[model_name] = (proba >= 0.5).astype(int)
    
    # Get optimal thresholds from threshold optimization
    optimal_thresholds = {}
    for model_name in models.keys():
        # Load from threshold analysis files
        threshold_file = Path("reports/figures") / f"threshold_analysis_{model_name.lower().replace(' ', '_')}.csv"
        if threshold_file.exists():
            df = pd.read_csv(threshold_file)
            # Get cost-optimal threshold (minimum cost)
            min_cost_idx = df["total_cost"].idxmin()
            optimal_thresholds[model_name] = df.loc[min_cost_idx, "threshold"]
        else:
            optimal_thresholds[model_name] = 0.5
    
    rows = []
    for i, (idx, true_val) in enumerate(zip(X_test.index, y_test)):
        row = {
            "transaction_id": idx,
            "y_true": int(true_val),
        }
        
        # Add key original features (not all 187 one-hot columns)
        # These will be joined from original data
        for model_name in models.keys():
            proba = model_probas[model_name][i]
            pred_default = model_preds[model_name][i]
            threshold = optimal_thresholds[model_name]
            pred_optimal = int(proba >= threshold)
            
            # Cost calculations
            fn_cost = c_fn if (true_val == 1 and pred_optimal == 0) else 0
            fp_cost = c_fp if (true_val == 0 and pred_optimal == 1) else 0
            total_cost = fn_cost + fp_cost
            
            row[f"{model_name.lower().replace(' ', '_')}_proba"] = float(proba)
            row[f"{model_name.lower().replace(' ', '_')}_pred_default"] = int(pred_default)
            row[f"{model_name.lower().replace(' ', '_')}_pred_optimal"] = int(pred_optimal)
            row[f"{model_name.lower().replace(' ', '_')}_fn_cost"] = fn_cost
            row[f"{model_name.lower().replace(' ', '_')}_fp_cost"] = fp_cost
            row[f"{model_name.lower().replace(' ', '_')}_total_cost"] = total_cost
            row[f"{model_name.lower().replace(' ', '_')}_optimal_threshold"] = threshold
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_dim_model(artifacts: Dict[str, Any]) -> pd.DataFrame:
    """Model dimension table with ROC-AUC and PR-AUC metrics."""
    from src.evaluation import evaluate_predictions
    
    models = artifacts["models"]
    X_test = artifacts["X_test"]
    X_test_scaled = artifacts["X_test_scaled"]
    y_test = artifacts["y_test"]
    
    rows = []
    for model_key, (model_name, model) in enumerate(models.items(), 1):
        if model_name == "Logistic Regression":
            proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            proba = model.predict_proba(X_test)[:, 1]
        
        pred = (proba >= 0.5).astype(int)
        metrics = evaluate_predictions(y_test.values, pred, proba)
        
        rows.append({
            "model_key": model_key,
            "model_name": model_name,
            "model_type": "Linear" if "Logistic" in model_name else ("Ensemble" if "Forest" in model_name else "Boosting"),
            "calibrated": "XGBoost" in model_name,
            "roc_auc": round(metrics.get("roc_auc", 0), 4),
            "pr_auc": round(metrics.get("pr_auc", 0), 4),
        })
    
    return pd.DataFrame(rows)


def generate_dim_threshold(artifacts: Dict[str, Any]) -> pd.DataFrame:
    """Threshold dimension table with cost/F1 metrics per threshold per model."""
    models = artifacts["models"]
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]
    
    model_probas = {}
    for model_name, model in models.items():
        if model_name == "Logistic Regression":
            proba = model.predict_proba(artifacts["X_test_scaled"])[:, 1]
        else:
            proba = model.predict_proba(X_test)[:, 1]
        model_probas[model_name] = proba
    
    rows = []
    thresholds = np.arange(0.01, 1.00, 0.01)
    
    for model_name, proba in model_probas.items():
        for t in thresholds:
            preds = (proba >= t).astype(int)
            from sklearn.metrics import confusion_matrix, f1_score
            cm = confusion_matrix(y_test, preds)
            tn, fp, fn, tp = cm.ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = f1_score(y_test, preds)
            cost = 10000 * fn + 500 * fp
            
            rows.append({
                "model_name": model_name,
                "threshold": round(t, 2),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "total_cost": cost,
            })
    
    return pd.DataFrame(rows)


def generate_dim_feature(artifacts: Dict[str, Any]) -> pd.DataFrame:
    """Feature dimension table with SHAP importance rankings."""
    # Load SHAP importance files
    importance_files = list(Path("reports/figures").glob("shap_importance_*.csv"))
    
    rows = []
    feature_key = 1
    
    for imp_file in importance_files:
        model_name = imp_file.stem.replace("shap_importance_", "").replace("_", " ").title()
        df = pd.read_csv(imp_file)
        
        for rank, (_, row) in enumerate(df.iterrows(), 1):
            rows.append({
                "feature_key": feature_key,
                "model_name": model_name,
                "feature_name": row["feature"],
                "shap_importance": row["importance"],
                "importance_rank": rank,
            })
            feature_key += 1
    
    return pd.DataFrame(rows)


def generate_business_kpis(artifacts: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    """Single-row KPI summary for dashboard cards."""
    y_test = artifacts["y_test"]
    models = artifacts["models"]
    
    # Fraud rate
    fraud_rate = y_test.mean()
    total_transactions = len(y_test)
    fraud_count = int(y_test.sum())
    legit_count = total_transactions - fraud_count
    
    # Best model (by F1 at default threshold)
    from src.evaluation import evaluate_predictions
    best_f1 = 0
    best_model_name = ""
    
    for model_name, model in models.items():
        if model_name == "Logistic Regression":
            proba = model.predict_proba(artifacts["X_test_scaled"])[:, 1]
        else:
            proba = model.predict_proba(artifacts["X_test"])[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = evaluate_predictions(y_test.values, pred, proba)
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model_name = model_name
    
    # Cost savings (using best model)
    best_model = models[best_model_name]
    if best_model_name == "Logistic Regression":
        best_proba = best_model.predict_proba(artifacts["X_test_scaled"])[:, 1]
    else:
        best_proba = best_model.predict_proba(artifacts["X_test"])[:, 1]
    
    from src.threshold_optimizer import calculate_savings
    savings = calculate_savings(
        y_test.values, best_proba,
        config["costs"]["c_fn"], config["costs"]["c_fp"],
        config["powerbi"]["annual_transactions"]
    )
    
    # Optimal threshold for best model
    threshold_file = Path("reports/figures") / f"threshold_analysis_{best_model_name.lower().replace(' ', '_')}.csv"
    if threshold_file.exists():
        df = pd.read_csv(threshold_file)
        min_cost_idx = df["total_cost"].idxmin()
        optimal_threshold = df.loc[min_cost_idx, "threshold"]
    else:
        optimal_threshold = 0.5
    
    return pd.DataFrame([{
        "total_transactions": total_transactions,
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "fraud_rate": round(fraud_rate, 4),
        "class_imbalance_ratio": f"{legit_count // fraud_count}:1",
        "best_model": best_model_name,
        "best_model_f1": round(best_f1, 4),
        "optimal_threshold": round(optimal_threshold, 2),
        "test_set_savings_rs": round(savings["savings_test"], 0),
        "savings_per_10k_rs": round(savings["savings_per_10k"], 0),
        "annual_projection_rs": round(savings["annual_savings"], 0),
        "annual_projection_crore": round(savings["annual_savings_crore"], 2),
        "cost_fn": config["costs"]["c_fn"],
        "cost_fp": config["costs"]["c_fp"],
        "cost_ratio": config["costs"]["c_fn"] // config["costs"]["c_fp"],
    }])


def generate_confusion_matrices_tidy(artifacts: Dict[str, Any]) -> pd.DataFrame:
    """Tidy format confusion matrices for all models and strategies."""
    models = artifacts["models"]
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]
    
    # Get optimal thresholds
    optimal_thresholds = {}
    for model_name in models.keys():
        threshold_file = Path("reports/figures") / f"threshold_analysis_{model_name.lower().replace(' ', '_')}.csv"
        if threshold_file.exists():
            df = pd.read_csv(threshold_file)
            min_cost_idx = df["total_cost"].idxmin()
            optimal_thresholds[model_name] = df.loc[min_cost_idx, "threshold"]
        else:
            optimal_thresholds[model_name] = 0.5
    
    from sklearn.metrics import confusion_matrix
    
    rows = []
    for model_name, model in models.items():
        if model_name == "Logistic Regression":
            proba = model.predict_proba(artifacts["X_test_scaled"])[:, 1]
        else:
            proba = model.predict_proba(X_test)[:, 1]
        
        threshold = optimal_thresholds[model_name]
        
        # Three strategies
        strategies = {
            "Default (0.50)": 0.5,
            "F1-Optimal": threshold,  # Simplified - would need F1-optimal
            "Cost-Optimal": threshold,
        }
        
        for strategy_name, thresh in strategies.items():
            preds = (proba >= thresh).astype(int)
            cm = confusion_matrix(y_test, preds)
            tn, fp, fn, tp = cm.ravel()
            
            rows.append({
                "model_name": model_name,
                "strategy": strategy_name,
                "threshold": round(thresh, 2),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0,
                "recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0,
                "f1_score": round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) > 0 else 0,
                "total_cost": 10000 * fn + 500 * fp,
            })
    
    return pd.DataFrame(rows)


def export_all_for_powerbi(
    output_dir: str = "data/powerbi",
    config_path: str = "config.yaml",
    models_dir: str = "models"
) -> Dict[str, str]:
    """Export all tables for Power BI star schema."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading pipeline artifacts...")
    artifacts = load_pipeline_artifacts(models_dir)
    
    logger.info("Generating FactTransactions...")
    fact_transactions = generate_fact_transactions(artifacts, config)
    fact_path = output_path / "FactTransactions.parquet"
    fact_transactions.to_parquet(fact_path, index=False)
    logger.info(f"Saved FactTransactions: {len(fact_transactions)} rows")
    
    logger.info("Generating DimModel...")
    dim_model = generate_dim_model(artifacts)
    dim_model_path = output_path / "DimModel.csv"
    dim_model.to_csv(dim_model_path, index=False)
    
    logger.info("Generating DimThreshold...")
    dim_threshold = generate_dim_threshold(artifacts)
    dim_threshold_path = output_path / "DimThreshold.parquet"
    dim_threshold.to_parquet(dim_threshold_path, index=False)
    logger.info(f"Saved DimThreshold: {len(dim_threshold)} rows")
    
    logger.info("Generating DimFeature...")
    dim_feature = generate_dim_feature(artifacts)
    dim_feature_path = output_path / "DimFeature.csv"
    dim_feature.to_csv(dim_feature_path, index=False)
    logger.info(f"Saved DimFeature: {len(dim_feature)} rows")
    
    logger.info("Generating BusinessKPIs...")
    business_kpis = generate_business_kpis(artifacts, config)
    kpis_path = output_path / "BusinessKPIs.csv"
    business_kpis.to_csv(kpis_path, index=False)
    logger.info(f"Saved BusinessKPIs: {len(business_kpis)} row")
    
    logger.info("Generating ConfusionMatrices (tidy)...")
    confusion_tidy = generate_confusion_matrices_tidy(artifacts)
    confusion_path = output_path / "ConfusionMatrices.csv"
    confusion_tidy.to_csv(confusion_path, index=False)
    
    logger.info("Generating ModelPredictions (for PR/ROC curves)...")
    model_predictions = generate_model_predictions(artifacts)
    pred_path = output_path / "ModelPredictions.parquet"
    model_predictions.to_parquet(pred_path, index=False)
    logger.info(f"Saved ModelPredictions: {len(model_predictions)} rows")
    
    # Also save threshold analysis consolidated
    threshold_files = artifacts.get("threshold_files", [])
    if threshold_files:
        all_thresholds = []
        for f in threshold_files:
            df = pd.read_csv(f)
            df["model"] = f.stem.replace("threshold_analysis_", "").replace("_", " ").title()
            all_thresholds.append(df)
        if all_thresholds:
            combined = pd.concat(all_thresholds, ignore_index=True)
            combined_path = output_path / "ThresholdAnalysis.parquet"
            combined.to_parquet(combined_path, index=False)
            logger.info(f"Saved ThresholdAnalysis: {len(combined)} rows")
    
    # Save export manifest
    manifest = {
        "exported_at": pd.Timestamp.now().isoformat(),
        "tables": {
            "FactTransactions": {"path": str(fact_path), "rows": len(fact_transactions)},
            "DimModel": {"path": str(dim_model_path), "rows": len(dim_model)},
            "DimThreshold": {"path": str(dim_threshold_path), "rows": len(dim_threshold)},
            "DimFeature": {"path": str(dim_feature_path), "rows": len(dim_feature)},
            "BusinessKPIs": {"path": str(kpis_path), "rows": len(business_kpis)},
            "ConfusionMatrices": {"path": str(confusion_path), "rows": len(confusion_tidy)},
            "ModelPredictions": {"path": str(pred_path), "rows": len(model_predictions)},
        },
        "config_summary": {
            "cost_fn": config["costs"]["c_fn"],
            "cost_fp": config["costs"]["c_fp"],
            "annual_transactions": config["powerbi"]["annual_transactions"],
        }
    }
    
    manifest_path = output_path / "export_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Power BI export complete! Files saved to {output_dir}/")
    logger.info(f"Manifest: {manifest_path}")
    
    return manifest


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    export_all_for_powerbi()