import argparse
import sys
import logging
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from src.data_preprocessing import load_and_preprocess
from src.model_training import train_models, save_models, load_models
from src.evaluation import (
    compare_models, plot_precision_recall_curve, plot_roc_curve,
    plot_confusion_matrices, print_results, find_optimal_threshold,
    evaluate_predictions
)
from src.threshold_optimizer import (
    optimize_all_models, plot_cost_curves, calculate_savings,
    save_threshold_analysis, evaluate_strategies
)
from src.explainability import explain_model, save_shap_values
from src.export_powerbi import export_all_for_powerbi
from src.model_training import select_best_model


def setup_logging(config: dict):
    """Configure logging from config."""
    log_config = config.get('logging', {})
    log_file = log_config.get('file', 'logs/pipeline.log')
    log_level = log_config.get('level', 'INFO')
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def cmd_train(args, config):
    """Train models command."""
    logging.info("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = load_and_preprocess()
    
    logging.info("Training models...")
    output = train_models(
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled,
        calibrate_xgb=config['models']['xgboost']['calibrate'],
        cv_folds=config['cv']['n_splits']
    )
    
    models = output['models']
    results = output['results']
    
    # Print results
    print_results(results)
    
    # Save models
    save_models(models, config['output']['models_dir'])
    
    # Save preprocessed data for later use
    import joblib
    joblib.dump({
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled,
        'scaler': scaler
    }, Path(config['output']['models_dir']) / 'preprocessed_data.joblib')
    
    logging.info("Training complete!")


def cmd_evaluate(args, config):
    """Evaluate models command."""
    import joblib
    from src.evaluation import evaluate_predictions
    
    # Load preprocessed data and models
    data = joblib.load(Path(config['output']['models_dir']) / 'preprocessed_data.joblib')
    X_test = data['X_test']
    X_test_scaled = data['X_test_scaled']
    y_test = data['y_test']
    y_true = y_test.values
    
    models = load_models(config['output']['models_dir'])
    
    # Generate predictions and compute metrics
    results = {}
    models_proba = {}
    models_preds = {}
    for name, model in models.items():
        if name == "Logistic Regression":
            proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = evaluate_predictions(y_true, pred, proba)
        results[name] = metrics
        models_proba[name] = proba
        models_preds[name] = pred
    
    # Compare models
    print_results(results)
    
    figures_dir = Path(config['output']['figures_dir'])
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    plot_precision_recall_curve(y_true, models_proba, save_path=figures_dir / 'pr_curve.png')
    plot_roc_curve(y_true, models_proba, save_path=figures_dir / 'roc_curve.png')
    plot_confusion_matrices(y_true, models_preds, save_path=figures_dir / 'confusion_matrices.png')
    
    # Comparison table
    comparison_df = compare_models(y_true, results)
    print("\nModel Comparison (sorted by F1):")
    print(comparison_df.to_string())
    comparison_df.to_csv(figures_dir / 'model_comparison.csv')
    
    logging.info("Evaluation complete!")


def cmd_optimize(args, config):
    """Optimize thresholds command."""
    import joblib
    
    # Load preprocessed data and models
    data = joblib.load(Path(config['output']['models_dir']) / 'preprocessed_data.joblib')
    X_test = data['X_test']
    X_test_scaled = data['X_test_scaled']
    y_test = data['y_test']
    y_true = y_test.values
    
    models = load_models(config['output']['models_dir'])
    
    # Generate probabilities for each model
    models_proba = {}
    for name, model in models.items():
        if name == "Logistic Regression":
            models_proba[name] = model.predict_proba(X_test_scaled)[:, 1]
        else:
            models_proba[name] = model.predict_proba(X_test)[:, 1]
    
    # Run per-model threshold optimization
    c_fn = config['costs']['c_fn']
    c_fp = config['costs']['c_fp']
    thresholds = np.arange(
        config['threshold']['min_threshold'],
        config['threshold']['max_threshold'],
        config['threshold']['step']
    )
    
    optimization_results = optimize_all_models(
        y_true, models_proba, c_fn, c_fp, thresholds
    )
    
    # Print comparison for each model
    for name, res in optimization_results.items():
        print(f"\n{'='*60}")
        print(f"Threshold Optimization: {name}")
        print(f"{'='*60}")
        print(res['comparison'].to_string(index=False))
        
        savings = calculate_savings(y_true, models_proba[name], c_fn, c_fp)
        print(f"\nSavings: Rs.{savings['savings_test']:,.0f} on test set")
        print(f"Per 10K transactions: Rs.{savings['savings_per_10k']:,.0f}")
        print(f"Annual projection: Rs.{savings['annual_savings_crore']:.2f} crore")
        print(f"Optimal threshold: {savings['best_threshold']:.2f}")
    
    # Plot cost curves
    figures_dir = Path(config['output']['figures_dir'])
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_cost_curves(optimization_results, c_fn, c_fp, save_path=figures_dir / 'cost_curves.png', show=False)
    
    # Save for Power BI
    save_threshold_analysis(optimization_results, config['output']['figures_dir'])
    
    logging.info("Threshold optimization complete!")


def cmd_explain(args, config):
    """Generate SHAP explanations command."""
    import joblib
    
    if not config['shap']['enabled']:
        logging.warning("SHAP is disabled in config. Enable it to run explanations.")
        return
    
    try:
        import shap
    except ImportError:
        logging.error("SHAP not installed. Run: pip install shap")
        return
    
    # Load preprocessed data and models
    data = joblib.load(Path(config['output']['models_dir']) / 'preprocessed_data.joblib')
    X_train = data['X_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    models = load_models(config['output']['models_dir'])
    
    figures_dir = Path(config['output']['figures_dir'])
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    sample_size = config['shap']['sample_size']
    
    for name, model in models.items():
        if name == "Logistic Regression":
            # Use scaled data for LR
            X_test_for_model = data['X_test_scaled']
        else:
            X_test_for_model = X_test
        
        explain_model(
            model, X_train, X_test_for_model, y_test, name,
            output_dir=str(figures_dir),
            sample_size=sample_size
        )
    
    logging.info("SHAP explanations complete!")


def cmd_export(args, config):
    """Export all data for Power BI dashboard."""
    output_dir = config['output'].get('powerbi_dir', 'data/powerbi')
    config_path = args.config if hasattr(args, 'config') else 'config.yaml'
    models_dir = config['output']['models_dir']
    
    manifest = export_all_for_powerbi(
        output_dir=output_dir,
        config_path=config_path,
        models_dir=models_dir
    )
    
    print("\nPower BI Export Complete!")
    print(f"Output directory: {output_dir}")
    print("\nGenerated tables:")
    for table_name, info in manifest["tables"].items():
        print(f"  {table_name}: {info['rows']} rows -> {info['path']}")
    print(f"\nManifest: {output_dir}/export_manifest.json")
    
    logging.info("Power BI export complete!")


def cmd_select(args, config):
    """Select best model based on criteria."""
    import joblib
    
    # Load preprocessed data and models
    data = joblib.load(Path(config['output']['models_dir']) / 'preprocessed_data.joblib')
    X_test = data['X_test']
    X_test_scaled = data['X_test_scaled']
    y_test = data['y_test']
    
    from src.model_training import load_models
    models = load_models(config['output']['models_dir'])
    
    # Get results by running evaluation
    from src.evaluation import evaluate_predictions
    results = {}
    for name, model in models.items():
        if name == "Logistic Regression":
            proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = evaluate_predictions(y_test.values, pred, proba)
        results[name] = metrics
        results[name]["proba"] = proba
        results[name]["pred"] = pred
    
    # Selection criteria
    criteria = getattr(args, 'criteria', 'cost_optimal_recall')
    c_fn = config['costs']['c_fn']
    c_fp = config['costs']['c_fp']
    
    best = select_best_model(
        results, X_test, X_test_scaled, y_test, models,
        selection_criteria=criteria,
        c_fn=c_fn, c_fp=c_fp
    )
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL SELECTION")
    print(f"{'='*60}")
    print(f"Criteria: {criteria}")
    print(f"Selected Model: {best['model_name']}")
    print(f"Optimal Threshold: {best['threshold']:.4f}")
    print(f"Score ({criteria}): {best['score']:.4f}")
    print(f"\nMetrics at Optimal Threshold:")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Recall:    {best['recall']:.4f}")
    print(f"  F1-Score:  {best['f1_score']:.4f}")
    print(f"  ROC-AUC:   {best['roc_auc']:.4f}")
    print(f"  PR-AUC:    {best['pr_auc']:.4f}")
    print(f"  Total Cost: Rs.{best['total_cost']:,.0f}")
    print(f"\nConfusion Matrix:")
    print(best['confusion_matrix'])
    
    # Save selection
    import json
    output_dir = Path(config['output'].get('figures_dir', 'reports/figures'))
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'best_model_selection.json', 'w') as f:
        json.dump(best, f, indent=2, default=str)
    
    logging.info(f"Best model selected: {best['model_name']} at threshold {best['threshold']:.4f}")


def cmd_full_pipeline(args, config):
    """Run full pipeline: train -> evaluate -> optimize -> explain -> export."""
    logging.info("="*60)
    logging.info("STARTING FULL PIPELINE")
    logging.info("="*60)
    
    cmd_train(args, config)
    cmd_evaluate(args, config)
    cmd_optimize(args, config)
    cmd_explain(args, config)
    cmd_export(args, config)
    
    logging.info("="*60)
    logging.info("FULL PIPELINE COMPLETE")
    logging.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main train                    # Train all models
  python -m src.main evaluate                 # Evaluate trained models
  python -m src.main optimize-threshold       # Find optimal thresholds
  python -m src.main explain                  # Generate SHAP explanations
  python -m src.main export-powerbi           # Export for Power BI dashboard
  python -m src.main select-model             # Select best model (recall + cost)
  python -m src.main select-model --criteria min_total_cost  # Minimize cost
  python -m src.main full-pipeline            # Run everything
        """
    )
    
    parser.add_argument(
        'command',
        choices=['train', 'evaluate', 'optimize-threshold', 'explain', 'export-powerbi', 'select-model', 'full-pipeline'],
        help='Command to run'
    )
    
    parser.add_argument(
        '--criteria',
        choices=['cost_optimal_recall', 'cost_optimal_f1', 'min_total_cost', 'recall_at_05', 'f1_at_05'],
        default='cost_optimal_recall',
        help='Model selection criteria (default: cost_optimal_recall)'
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    if args.verbose:
        config['logging']['level'] = 'DEBUG'
    setup_logging(config)
    
    # Route to command
    commands = {
        'train': cmd_train,
        'evaluate': cmd_evaluate,
        'optimize-threshold': cmd_optimize,
        'explain': cmd_explain,
        'export-powerbi': cmd_export,
        'select-model': cmd_select,
        'full-pipeline': cmd_full_pipeline,
    }
    
    cmd_func = commands[args.command]
    cmd_func(args, config)


if __name__ == "__main__":
    import numpy as np  # needed for optimize command
    main()