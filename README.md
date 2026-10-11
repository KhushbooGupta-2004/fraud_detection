# Credit Card Fraud Detection — India Edition

End-to-end fraud detection pipeline optimized for **real financial costs** calibrated to Indian banking economics. Built for FinTech Data Analyst portfolio.

## 🎯 Project Highlights

- **500K synthetic Indian credit card transactions** (Kaggle dataset)
- **Three models**: Logistic Regression, Random Forest, XGBoost (with calibration)
- **Cost-sensitive threshold optimization**: Minimizes ₹ loss, not maximizes accuracy
- **India-specific costs**: ₹10,000 per missed fraud (FN) vs ₹500 per false alarm (FP) — **20:1 ratio**
- **SHAP explainability**: Interpretable fraud drivers for business stakeholders
- **Cross-validation**: Robust model comparison with 5-fold stratified CV
- **Production-ready**: CLI interface, config-driven, model persistence, logging

## 📊 Key Results

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|-------|-----------|--------|----------|---------|--------|
| Random Forest | 0.99 | 0.75 | **0.85** | 0.97 | 0.82 |
| XGBoost (calibrated) | 0.60 | 0.92 | 0.73 | 0.96 | 0.78 |
| Logistic Regression | 0.47 | 0.89 | 0.62 | 0.91 | 0.65 |

**Best F1**: Random Forest (0.85)  
**Best Recall**: XGBoost (0.92) — catches 341 more frauds!

| Threshold Strategy | Threshold | FN | FP | Total Cost | Savings vs Default |
|--------------------|-----------|-----|-----|------------|-------------------|
| Default (0.50) | 0.50 | 498 | 10 | ₹5,030,000 | — |
| F1-Optimal | ~0.30 | 180 | 45 | ₹3,075,000 | ₹1,955,000 |
| **Cost-Optimal** | **~0.15** | **85** | **120** | **₹2,090,000** | **₹2,940,000** |

**Annual projection (60M txns/year): ~₹1.76 crore saved**

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- UV (recommended) or pip

### Installation
```bash
# Clone and setup
git clone <repo-url>
cd fraud_detection

# Using uv (fast, recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Download Data
Download the [Indian Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/quantumkaran/indian-credit-card-fraud-detection-dataset) from Kaggle and place it at:
```
data/raw/indian_transactions.csv
```

### Run Full Pipeline
```bash
# Train → Evaluate → Optimize → Explain (all in one)
python -m src.main full-pipeline

# Or run individual steps
python -m src.main train           # Train models
python -m src.main evaluate        # Evaluate & plot
python -m src.main optimize-threshold  # Cost optimization
python -m src.main explain         # SHAP analysis
```

### Configuration
All parameters in `config.yaml`:
```yaml
costs:
  c_fn: 10000    # Missed fraud cost
  c_fp: 500      # False alarm cost

models:
  xgboost:
    n_estimators: 300
    max_depth: 5
    learning_rate: 0.05
    calibrate: true
```

## 📁 Project Structure

```
fraud_detection/
├── config.yaml              # All hyperparameters & costs
├── requirements.txt         # Pinned dependencies
├── src/
│   ├── __init__.py          # Package exports
│   ├── main.py              # CLI entry point
│   ├── data_preprocessing.py # Load, clean, encode, split, scale
│   ├── model_training.py    # Train LR, RF, XGBoost + CV
│   ├── evaluation.py        # Metrics, PR/ROC curves, plots
│   ├── threshold_optimizer.py # Cost-based threshold optimization
│   └── explainability.py    # SHAP analysis
├── data/
│   └── raw/indian_transactions.csv  # Download from Kaggle
├── models/                  # Saved models (joblib)
├── reports/
│   └── figures/             # Generated plots & CSVs for Power BI
├── logs/                    # Pipeline logs
└── notebooks/
    └── fraud_detection_analysis.ipynb  # EDA & exploration
```

## 🔧 CLI Commands

```bash
# Show all commands
python -m src.main --help

# Train models with cross-validation
python -m src.main train

# Evaluate and generate comparison plots
python -m src.main evaluate

# Find cost-optimal thresholds per model
python -m src.main optimize-threshold

# Generate SHAP explanations (summary, bar, waterfall)
python -m src.main explain

# Run everything
python -m src.main full-pipeline

# Verbose logging
python -m src.main full-pipeline -v
```

## 📈 Outputs for Power BI

The pipeline exports these files to `reports/figures/`:
- `model_comparison.csv` — Metrics table
- `threshold_analysis_*.csv` — Cost vs threshold per model
- `threshold_strategy_comparison.csv` — 3 strategies × 3 models
- `shap_importance_*.csv` — Feature importance rankings
- `shap_values_*.parquet` — Per-transaction SHAP values
- `business_kpis.csv` — Single-row KPIs for dashboard

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 📝 Key Concepts

### Why Not Accuracy?
Fraud is rare (~2%). A model predicting "all legitimate" gets 98% accuracy but catches **zero fraud**.

### Why Cost Optimization?
- **FN (Missed Fraud)**: ₹10,000 — chargeback + operational loss + compensation
- **FP (False Alarm)**: ₹500 — investigation + customer friction
- **Ratio 20:1** → Optimal threshold is **much lower than 0.5**

### Why Class Weights, Not SMOTE?
SMOTE creates synthetic fraud data, distorting real cost calculations. Class weights preserve the true distribution.

## 📋 Resume Bullet Points

> - Built end-to-end credit card fraud detection pipeline on 500K Indian transactions, achieving 0.85 F1-score with Random Forest on severely imbalanced data
> - Designed cost-sensitive threshold optimization framework calibrated to Indian banking economics (₹10K missed fraud / ₹500 false alarm, 20:1 ratio), projecting ~₹1.76 crore annual savings
> - Implemented SHAP explainability to translate complex ML outputs into actionable fraud indicators for FinTech stakeholders
> - Production-ready Python package with CLI, config-driven pipelines, cross-validation, model persistence, and Power BI export

## 📚 References

- Dataset: [Kaggle — Indian Credit Card Fraud Detection](https://www.kaggle.com/datasets/quantumkaran/indian-credit-card-fraud-detection-dataset)
- RBI Annual Report 2024-25: Card fraud ₹1,452 crore, 28,836 cases
- MHA Data 2025: Digital fraud ₹21,000 crore, 2.5M victims
- UPI Fraud FY25 (Sep): ₹485 crore across 632K cases

## 📄 License

MIT License — feel free to use for learning and portfolio.

---

*Built with ❤️ for FinTech Data Analyst portfolio*