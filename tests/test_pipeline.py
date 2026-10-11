"""Tests for fraud detection pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Import from src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_preprocessing import load_and_preprocess
from model_training import train_models, save_models, load_models
from evaluation import (
    evaluate_predictions,
    compare_models,
    threshold_sweep_metrics,
    find_optimal_threshold,
    get_best_model,
)
from threshold_optimizer import (
    compute_cost,
    find_cost_optimal_threshold,
    find_f1_optimal_threshold,
    evaluate_strategies,
    optimize_all_models,
    calculate_savings,
)


@pytest.fixture
def sample_data():
    """Generate synthetic imbalanced data for testing."""
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        n_redundant=2,
        weights=[0.98, 0.02],  # 2% fraud
        random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(15)]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name="is_fraud")
    return X, y


@pytest.fixture
def trained_models(sample_data):
    """Train models on sample data."""
    X, y = sample_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Simple scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    output = train_models(
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled,
        calibrate_xgb=False,  # Skip for speed
        cv_folds=3
    )
    
    return output['models'], output['results'], X_test, X_test_scaled, y_test


class TestDataPreprocessing:
    """Tests for data preprocessing."""
    
    def test_load_and_preprocess_returns_correct_types(self):
        """Test that load_and_preprocess returns expected types with proper data."""
        # Create synthetic data with the expected categorical columns
        n_samples = 1000
        np.random.seed(42)
        
        df = pd.DataFrame({
            'trans_datetime': pd.date_range('2023-01-01', periods=n_samples, freq='h'),
            'trans_date': pd.date_range('2023-01-01', periods=n_samples, freq='h').date,
            'trans_time': pd.date_range('2023-01-01', periods=n_samples, freq='h').time,
            'cc_num': range(n_samples),
            'cardholder_name': [f'User_{i}' for i in range(n_samples)],
            'gender': np.random.choice(['M', 'F'], n_samples),
            'dob': pd.date_range('1950-01-01', periods=n_samples, freq='365D'),
            'age': np.random.randint(18, 80, n_samples),
            'job': np.random.choice(['Engineer', 'Doctor', 'Teacher', 'Manager'], n_samples),
            'state': np.random.choice(['Maharashtra', 'Karnataka', 'Delhi', 'Gujarat'], n_samples),
            'merchant': np.random.choice(['Amazon', 'Flipkart', 'Reliance', 'BigBasket'], n_samples),
            'category': np.random.choice(['shopping', 'grocery', 'travel', 'food'], n_samples),
            'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'UPI', 'Net Banking'], n_samples),
            'upi_id': [f'user{i}@bank' if i % 2 == 0 else np.nan for i in range(n_samples)],
            'amt_inr': np.random.lognormal(6, 1, n_samples),
            'trans_num': [f'TXN{i:08d}' for i in range(n_samples)],
            'is_fraud': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
            'fraud_reason': np.nan,
        })
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            result = load_and_preprocess(temp_path)
            assert len(result) == 7  # X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler
            X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = result
            
            assert isinstance(X_train, pd.DataFrame)
            assert isinstance(X_test, pd.DataFrame)
            assert isinstance(y_train, pd.Series)
            assert isinstance(y_test, pd.Series)
            assert X_train.shape[0] + X_test.shape[0] == n_samples
            assert list(X_train.columns) == list(X_test.columns)
            # Should have one-hot encoded columns
            assert any('gender_' in c for c in X_train.columns)
            assert any('job_' in c for c in X_train.columns)
        finally:
            import os
            os.unlink(temp_path)


class TestModelTraining:
    """Tests for model training."""
    
    def test_train_models_returns_all_three(self, trained_models):
        """Test that all three models are trained."""
        models, results, _, _, _ = trained_models
        assert "Logistic Regression" in models
        assert "Random Forest" in models
        assert "XGBoost" in models
        assert len(models) == 3
    
    def test_results_have_expected_keys(self, trained_models):
        """Test that results contain expected metrics."""
        _, results, _, _, _ = trained_models
        for name, metrics in results.items():
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1_score" in metrics
            assert "roc_auc" in metrics
            assert "pr_auc" in metrics
            assert "confusion_matrix" in metrics
            assert "model" in metrics
    
    def test_cv_results_present(self, trained_models):
        """Test that cross-validation results are included."""
        _, results, _, _, _ = trained_models
        for name, metrics in results.items():
            assert "cv" in metrics
            assert "roc_auc_mean" in metrics["cv"]
            assert "pr_auc_mean" in metrics["cv"]
    
    def test_save_and_load_models(self, trained_models, tmp_path):
        """Test model persistence."""
        models, _, _, _, _ = trained_models
        save_models(models, str(tmp_path))
        
        loaded = load_models(str(tmp_path))
        assert len(loaded) == 3
        assert "Logistic Regression" in loaded
        assert "Random Forest" in loaded
        assert "Xgboost" in loaded or "XGBoost" in loaded


class TestEvaluation:
    """Tests for evaluation metrics."""
    
    def test_evaluate_predictions(self, sample_data):
        """Test evaluate_predictions function."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        model = LogisticRegression(class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        
        metrics = evaluate_predictions(y_test.values, pred, proba)
        
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert "confusion_matrix" in metrics
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
    
    def test_compare_models(self, trained_models):
        """Test compare_models function."""
        _, results, _, _, y_test = trained_models
        y_true = y_test.values
        
        df = compare_models(y_true, results, metric="F1_Score")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "Precision" in df.columns
        assert "Recall" in df.columns
        assert "F1_Score" in df.columns
    
    def test_threshold_sweep(self, sample_data):
        """Test threshold sweep metrics."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        model = LogisticRegression(class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        
        sweep_df = threshold_sweep_metrics(y_test.values, proba)
        assert isinstance(sweep_df, pd.DataFrame)
        assert len(sweep_df) == 99  # 0.01 to 0.99 step 0.01
        assert "Threshold" in sweep_df.columns
        assert "F1_Score" in sweep_df.columns
    
    def test_find_optimal_threshold(self, sample_data):
        """Test finding optimal threshold."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        model = LogisticRegression(class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        
        best_thresh, best_metrics, sweep_df = find_optimal_threshold(y_test.values, proba, metric="F1_Score")
        assert 0 < best_thresh < 1
        assert "F1_Score" in best_metrics
        assert isinstance(sweep_df, pd.DataFrame)
    
    def test_get_best_model(self, trained_models):
        """Test get_best_model function."""
        _, results, _, _, _ = trained_models
        best = get_best_model(results, "f1_score")
        assert best in results


class TestThresholdOptimization:
    """Tests for threshold optimization."""
    
    def test_compute_cost(self, sample_data):
        """Test cost computation."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        model = LogisticRegression(class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        
        cost, cm = compute_cost(y_test.values, proba, 0.5, c_fn=10000, c_fp=500)
        assert cost >= 0
        assert all(k in cm for k in ["tn", "fp", "fn", "tp"])
    
    def test_find_cost_optimal_threshold(self, sample_data):
        """Test cost-optimal threshold finding."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        model = LogisticRegression(class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        
        best_thresh, best_cost, costs = find_cost_optimal_threshold(
            y_test.values, proba, c_fn=10000, c_fp=500
        )
        assert 0 < best_thresh < 1
        assert best_cost >= 0
        assert len(costs) == 99
    
    def test_find_f1_optimal_threshold(self, sample_data):
        """Test F1-optimal threshold finding."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        model = LogisticRegression(class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        
        best_thresh, best_f1 = find_f1_optimal_threshold(y_test.values, proba)
        assert 0 < best_thresh < 1
        assert 0 <= best_f1 <= 1
    
    def test_evaluate_strategies(self, sample_data):
        """Test three-strategy comparison."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        model = LogisticRegression(class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        
        df, costs, cost_thresh, f1_thresh = evaluate_strategies(
            y_test.values, proba, c_fn=10000, c_fp=500
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "Default (0.50)" in df["Strategy"].values
        assert any("F1-Optimal" in s for s in df["Strategy"])
        assert any("Cost-Optimal" in s for s in df["Strategy"])
    
    def test_optimize_all_models(self, trained_models):
        """Test per-model threshold optimization."""
        models, results, _, _, y_test = trained_models
        y_true = y_test.values
        
        models_proba = {}
        for name, res in results.items():
            models_proba[name] = res['proba']
        
        opt_results = optimize_all_models(y_true, models_proba, c_fn=10000, c_fp=500)
        assert len(opt_results) == 3
        for name, res in opt_results.items():
            assert "comparison" in res
            assert "cost_curve" in res
            assert "cost_optimal_threshold" in res
    
    def test_calculate_savings(self, sample_data):
        """Test savings calculation."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        model = LogisticRegression(class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        
        savings = calculate_savings(y_test.values, proba, c_fn=10000, c_fp=500)
        assert "cost_default" in savings
        assert "cost_optimal" in savings
        assert "savings_test" in savings
        assert "savings_per_10k" in savings
        assert "annual_savings_crore" in savings
        assert savings["savings_test"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])