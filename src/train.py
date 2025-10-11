"""Model training utilities for Wine Quality prediction.

This module handles model training with hyperparameter tuning using GridSearchCV.
Follows PEP8 conventions with type hints.
"""

import logging
from pathlib import Path
from typing import Dict, Any

import joblib
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
