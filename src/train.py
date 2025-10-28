"""Model training utilities for Wine Quality prediction.

This module handles model training with hyperparameter tuning using GridSearchCV.
Follows PEP8 conventions with type hints.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import config

# Configure logging
logging.basicConfig(format=config.LOG_FORMAT, level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


def train_model_with_grid_search(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, Any] = None,
    cv: int = config.CV_FOLDS,
    n_jobs: int = config.CV_N_JOBS,
    verbose: int = config.CV_VERBOSE,
) -> GridSearchCV:
    """Train model with hyperparameter tuning using GridSearchCV.

    Args:
        pipeline: Scikit-learn pipeline with preprocessing and model.
        X_train: Training features.
        y_train: Training target.
        param_grid: Hyperparameter grid for GridSearchCV.
            If None, uses default from config.
        cv: Number of cross-validation folds.
        n_jobs: Number of parallel jobs (-1 for all cores).
        verbose: Verbosity level for GridSearchCV.

    Returns:
        Fitted GridSearchCV object with best model.

    Raises:
        ValueError: If training data is empty or invalid.
    """
    if X_train.empty or y_train.empty:
        raise ValueError("Training data cannot be empty")

    if param_grid is None:
        param_grid = config.HYPERPARAM_GRID

    logger.info("=" * 70)
    logger.info("Starting model training with GridSearchCV")
    logger.info("=" * 70)

    logger.info(f"Training samples: {X_train.shape[0]}")
    logger.info(f"Features: {X_train.shape[1]}")
    logger.info(f"Cross-validation folds: {cv}")
    logger.info(f"Hyperparameter grid: {param_grid}")

    total_combinations = 1
    for param_values in param_grid.values():
        total_combinations *= len(param_values)

    logger.info(f"Total hyperparameter combinations: {total_combinations}")
    logger.info(f"Total model fits: {total_combinations * cv}")

    # Create GridSearchCV
    clf = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
        refit=True,  # Automatically refit best model on full training set
    )

    # Fit the model
    logger.info("Starting hyperparameter search...")
    clf.fit(X_train, y_train)

    logger.info("GridSearchCV completed!")
    logger.info("=" * 70)
    logger.info("Best hyperparameters found:")
    for param_name, param_value in clf.best_params_.items():
        logger.info(f"  {param_name}: {param_value}")

    logger.info(f"\nBest cross-validation score: {clf.best_score_:.4f}")
    logger.info(f"Model automatically refit on full training set: {clf.refit}")

    return clf


def save_model(model: GridSearchCV, filepath: Path = config.MODEL_PATH) -> None:
    """Save trained model to disk.

    Args:
        model: Trained GridSearchCV model to save.
        filepath: Path where model will be saved.

    Raises:
        IOError: If model cannot be saved.
    """
    logger.info("=" * 70)
    logger.info(f"Saving trained model to: {filepath}")

    try:
        joblib.dump(model, filepath)
        file_size_kb = filepath.stat().st_size / 1024
        logger.info("Model saved successfully!")
        logger.info(f"File size: {file_size_kb:.2f} KB")
        logger.info("=" * 70)
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise IOError(f"Could not save model to {filepath}") from e


def load_model(filepath: Path = config.MODEL_PATH) -> GridSearchCV:
    """Load trained model from disk.

    Args:
        filepath: Path to saved model file.

    Returns:
        Loaded GridSearchCV model.

    Raises:
        FileNotFoundError: If model file doesn't exist.
        IOError: If model cannot be loaded.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    logger.info(f"Loading model from: {filepath}")

    try:
        model = joblib.load(filepath)
        logger.info("Model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise IOError(f"Could not load model from {filepath}") from e


def get_best_params(model: GridSearchCV) -> Dict[str, Any]:
    """Extract best hyperparameters from trained GridSearchCV model.

    Args:
        model: Trained GridSearchCV model.

    Returns:
        Dictionary of best hyperparameters.
    """
    return model.best_params_


def get_cv_results(model: GridSearchCV) -> pd.DataFrame:
    """Extract cross-validation results from GridSearchCV.

    Args:
        model: Trained GridSearchCV model.

    Returns:
        DataFrame with CV results for all hyperparameter combinations.
    """
    cv_results = pd.DataFrame(model.cv_results_)
    return cv_results[
        [
            "params",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
            "mean_fit_time",
        ]
    ].sort_values("rank_test_score")


def compare_pipelines(
    pipeline_rf: Pipeline,
    pipeline_gb: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param_grid_rf: Dict[str, Any] = None,
    param_grid_gb: Dict[str, Any] = None,
) -> Dict[str, GridSearchCV]:
    """Compare RandomForest and GradientBoosting pipelines.

    Each pipeline is logged in a separate MLflow run.

    Args:
        pipeline_rf: RandomForest pipeline with preprocessing.
        pipeline_gb: GradientBoosting pipeline with preprocessing.
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        param_grid_rf: Hyperparameter grid for RandomForest.
        param_grid_gb: Hyperparameter grid for GradientBoosting.

    Returns:
        Dictionary with trained models: {"rf": GridSearchCV, "gb": GridSearchCV}.

    Example:
        >>> results = compare_pipelines(
        ...     pipe_rf, pipe_gb, X_train, y_train, X_test, y_test
        ... )
        >>> best = (
        ...     results["rf"]
        ...     if results["rf"].best_score_ > results["gb"].best_score_
        ...     else results["gb"]
        ... )
    """
    from src.evaluate import evaluate_model

    if param_grid_rf is None:
        param_grid_rf = config.HYPERPARAM_GRID
    if param_grid_gb is None:
        param_grid_gb = config.GB_HYPERPARAM_GRID

    results = {}

    logger.info("=" * 70)
    logger.info("COMPARING PIPELINES: RandomForest vs GradientBoosting")
    logger.info("=" * 70)

    # Train and log RandomForest pipeline
    logger.info("\n### PIPELINE 1: RandomForest ###")
    with mlflow.start_run(run_name="RandomForest_Pipeline"):
        logger.info("Training RandomForest pipeline...")
        trained_rf = train_model_with_grid_search(
            pipeline_rf, X_train, y_train, param_grid_rf
        )
        logger.info("\nEvaluating RandomForest on test set...")
        metrics_rf = evaluate_model(trained_rf, X_test, y_test)
        logger.info("\nLogging RandomForest to MLflow...")
        log_model_to_mlflow(trained_rf, metrics_rf)
        results["rf"] = trained_rf
        logger.info("RandomForest pipeline completed!")

    # Train and log GradientBoosting pipeline
    logger.info("\n### PIPELINE 2: GradientBoosting ###")
    with mlflow.start_run(run_name="GradientBoosting_Pipeline"):
        logger.info("Training GradientBoosting pipeline...")
        trained_gb = train_model_with_grid_search(
            pipeline_gb, X_train, y_train, param_grid_gb
        )
        logger.info("\nEvaluating GradientBoosting on test set...")
        metrics_gb = evaluate_model(trained_gb, X_test, y_test)
        logger.info("\nLogging GradientBoosting to MLflow...")
        log_model_to_mlflow(trained_gb, metrics_gb)
        results["gb"] = trained_gb
        logger.info("GradientBoosting pipeline completed!")

    # Compare results
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 70)
    logger.info("\nRandomForest Results:")
    logger.info(f"  Best CV Score: {trained_rf.best_score_:.4f}")
    logger.info(f"  Test R² Score: {metrics_rf['r2_score']:.4f}")
    logger.info(f"  Test RMSE:     {metrics_rf['rmse']:.4f}")

    logger.info("\nGradientBoosting Results:")
    logger.info(f"  Best CV Score: {trained_gb.best_score_:.4f}")
    logger.info(f"  Test R² Score: {metrics_gb['r2_score']:.4f}")
    logger.info(f"  Test RMSE:     {metrics_gb['rmse']:.4f}")

    # Determine winner
    if metrics_rf["r2_score"] > metrics_gb["r2_score"]:
        logger.info("\n*** WINNER: RandomForest ***")
    elif metrics_gb["r2_score"] > metrics_rf["r2_score"]:
        logger.info("\n*** WINNER: GradientBoosting ***")
    else:
        logger.info("\n*** TIE: Both models have equal performance ***")

    logger.info("=" * 70)

    return results


def log_model_to_mlflow(
    model: GridSearchCV,
    metrics: Dict[str, float],
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """Log trained model and metrics to MLflow.

    Args:
        model: Trained GridSearchCV model to log.
        metrics: Dictionary of evaluation metrics.
        params: Additional parameters to log. If None, logs best params from model.

    Example:
        >>> metrics = {"r2_score": 0.47, "mse": 0.34}
        >>> log_model_to_mlflow(trained_model, metrics)
    """
    logger.info("=" * 70)
    logger.info("Logging to MLflow")
    logger.info("=" * 70)

    # Log best hyperparameters
    if params is None:
        if hasattr(model, 'best_params_'):
            params = model.best_params_
        else:
            params = {}

    logger.info("Logging hyperparameters...")
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)
        logger.info(f"  {param_name}: {param_value}")

    # Log cross-validation score
    mlflow.log_param("cv_folds", config.CV_FOLDS)
    if hasattr(model, 'best_score_'):
        mlflow.log_metric("cv_score", model.best_score_)

    # Log evaluation metrics
    logger.info("\nLogging metrics...")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            mlflow.log_metric(metric_name, metric_value)
            logger.info(f"  {metric_name}: {metric_value:.4f}")

    # Log model artifact and register to Models Registry
    logger.info("\nLogging model artifact and registering to Models Registry...")
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="wine_quality_rf_model",
    )
    logger.info("  ✓ Model logged to run artifacts")
    logger.info("  ✓ Model registered to Models Registry")

    logger.info("=" * 70)
