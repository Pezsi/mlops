"""Model evaluation utilities for Wine Quality prediction.

This module handles model evaluation with various regression metrics.
Follows PEP8 conventions with type hints.
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

import config

# Configure logging
logging.basicConfig(format=config.LOG_FORMAT, level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


def evaluate_model(
    model: GridSearchCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    log_results: bool = True,
) -> Dict[str, float]:
    """Evaluate trained model on test set.

    Calculates multiple regression metrics:
    - R² Score (coefficient of determination)
    - MSE (Mean Squared Error)
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)

    Args:
        model: Trained GridSearchCV model.
        X_test: Test features.
        y_test: Test target values.
        log_results: Whether to log evaluation results.

    Returns:
        Dictionary containing evaluation metrics.

    Raises:
        ValueError: If test data is empty.
    """
    if X_test.empty or y_test.empty:
        raise ValueError("Test data cannot be empty")

    if log_results:
        logger.info("=" * 70)
        logger.info("Evaluating model on test set")
        logger.info("=" * 70)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    metrics = {
        "r2_score": r2,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "test_samples": len(y_test),
    }

    if log_results:
        logger.info(f"Test set size: {len(y_test)} samples")
        logger.info(f"\nTest Set Performance:")
        logger.info(f"  R² Score:  {r2:.4f}")
        logger.info(f"  MSE:       {mse:.4f}")
        logger.info(f"  RMSE:      {rmse:.4f}")
        logger.info(f"  MAE:       {mae:.4f}")
        logger.info("=" * 70)

    return metrics


def calculate_residuals(y_true: pd.Series, y_pred: np.ndarray) -> np.ndarray:
    """Calculate prediction residuals.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        Array of residuals (y_true - y_pred).
    """
    return np.array(y_true) - y_pred


def get_prediction_errors(
    model: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """Get detailed prediction errors for test set.

    Args:
        model: Trained GridSearchCV model.
        X_test: Test features.
        y_test: Test target values.

    Returns:
        DataFrame with actual values, predictions, and errors.
    """
    y_pred = model.predict(X_test)
    residuals = calculate_residuals(y_test, y_pred)

    error_df = pd.DataFrame(
        {
            "actual": y_test.values,
            "predicted": y_pred,
            "error": residuals,
            "absolute_error": np.abs(residuals),
            "squared_error": residuals**2,
        }
    )

    return error_df


def print_prediction_summary(
    model: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series, n_samples: int = 5
) -> None:
    """Print sample predictions with actual values.

    Args:
        model: Trained GridSearchCV model.
        X_test: Test features.
        y_test: Test target values.
        n_samples: Number of sample predictions to print.
    """
    logger.info("=" * 70)
    logger.info(f"Sample predictions (first {n_samples} test samples):")
    logger.info("=" * 70)

    y_pred = model.predict(X_test[:n_samples])

    for i in range(min(n_samples, len(y_test))):
        actual = y_test.iloc[i]
        predicted = y_pred[i]
        error = abs(actual - predicted)

        logger.info(f"\nSample {i + 1}:")
        logger.info(f"  Actual quality:    {actual}")
        logger.info(f"  Predicted quality: {predicted:.2f}")
        logger.info(f"  Absolute error:    {error:.2f}")

    logger.info("=" * 70)


def compare_with_baseline(metrics: Dict[str, float]) -> None:
    """Compare model performance with baseline metrics.

    Args:
        metrics: Dictionary containing evaluation metrics.
    """
    # Simple baseline: always predict mean
    baseline_r2 = 0.0  # Always predicting mean gives R²=0
    baseline_description = "always predicting mean"

    logger.info("=" * 70)
    logger.info("Comparison with baseline:")
    logger.info(f"  Baseline ({baseline_description}): R² = {baseline_r2:.4f}")
    logger.info(f"  Our model: R² = {metrics['r2_score']:.4f}")

    if metrics["r2_score"] > baseline_r2:
        improvement = metrics["r2_score"] - baseline_r2
        logger.info(f"  ✓ Improvement: +{improvement:.4f} R²")
    else:
        logger.warning(f"  ✗ Model performs worse than baseline!")

    logger.info("=" * 70)
