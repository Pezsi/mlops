"""Unit tests for evaluation module.

Tests for src/evaluate.py functions.
Follows PEP8 conventions.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    evaluate_model,
    calculate_residuals,
    get_prediction_errors,
    print_prediction_summary,
    compare_with_baseline,
)


class TestEvaluateModel:
    """Test suite for evaluate_model function."""

    def test_returns_dict(self, trained_simple_model, sample_train_test_split):
        """Test that function returns a dictionary."""
        _, X_test, _, y_test = sample_train_test_split

        metrics = evaluate_model(trained_simple_model, X_test, y_test, log_results=False)

        assert isinstance(metrics, dict)

    def test_contains_expected_metrics(self, trained_simple_model, sample_train_test_split):
        """Test that returned dict contains expected metrics."""
        _, X_test, _, y_test = sample_train_test_split

        metrics = evaluate_model(trained_simple_model, X_test, y_test, log_results=False)

        expected_metrics = ["r2_score", "mse", "rmse", "mae", "test_samples"]
        for metric in expected_metrics:
            assert metric in metrics

    def test_r2_score_in_valid_range(self, trained_simple_model, sample_train_test_split):
        """Test that R² score is in valid range."""
        _, X_test, _, y_test = sample_train_test_split

        metrics = evaluate_model(trained_simple_model, X_test, y_test, log_results=False)

        # R² can be negative for very bad models, but should be reasonable
        assert metrics["r2_score"] >= -1.0
        assert metrics["r2_score"] <= 1.0

    def test_mse_is_positive(self, trained_simple_model, sample_train_test_split):
        """Test that MSE is positive."""
        _, X_test, _, y_test = sample_train_test_split

        metrics = evaluate_model(trained_simple_model, X_test, y_test, log_results=False)

        assert metrics["mse"] >= 0

    def test_rmse_is_positive(self, trained_simple_model, sample_train_test_split):
        """Test that RMSE is positive."""
        _, X_test, _, y_test = sample_train_test_split

        metrics = evaluate_model(trained_simple_model, X_test, y_test, log_results=False)

        assert metrics["rmse"] >= 0

    def test_rmse_equals_sqrt_mse(self, trained_simple_model, sample_train_test_split):
        """Test that RMSE equals square root of MSE."""
        _, X_test, _, y_test = sample_train_test_split

        metrics = evaluate_model(trained_simple_model, X_test, y_test, log_results=False)

        expected_rmse = np.sqrt(metrics["mse"])
        assert abs(metrics["rmse"] - expected_rmse) < 1e-10

    def test_mae_is_positive(self, trained_simple_model, sample_train_test_split):
        """Test that MAE is positive."""
        _, X_test, _, y_test = sample_train_test_split

        metrics = evaluate_model(trained_simple_model, X_test, y_test, log_results=False)

        assert metrics["mae"] >= 0

    def test_test_samples_count_correct(self, trained_simple_model, sample_train_test_split):
        """Test that test_samples count is correct."""
        _, X_test, _, y_test = sample_train_test_split

        metrics = evaluate_model(trained_simple_model, X_test, y_test, log_results=False)

        assert metrics["test_samples"] == len(y_test)

    def test_empty_test_data_raises_error(self, trained_simple_model):
        """Test that empty test data raises ValueError."""
        X_test_empty = pd.DataFrame()
        y_test_empty = pd.Series(dtype=float)

        with pytest.raises(ValueError, match="Test data cannot be empty"):
            evaluate_model(trained_simple_model, X_test_empty, y_test_empty)


class TestCalculateResiduals:
    """Test suite for calculate_residuals function."""

    def test_returns_numpy_array(self):
        """Test that function returns numpy array."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.0])

        residuals = calculate_residuals(y_true, y_pred)

        assert isinstance(residuals, np.ndarray)

    def test_residuals_correct_length(self):
        """Test that residuals have correct length."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.0])

        residuals = calculate_residuals(y_true, y_pred)

        assert len(residuals) == len(y_true)

    def test_residuals_correct_calculation(self):
        """Test that residuals are calculated correctly."""
        y_true = pd.Series([5, 10, 15])
        y_pred = np.array([4, 11, 14])

        residuals = calculate_residuals(y_true, y_pred)

        expected = np.array([1, -1, 1])
        np.testing.assert_array_almost_equal(residuals, expected)

    def test_perfect_predictions_zero_residuals(self):
        """Test that perfect predictions give zero residuals."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        residuals = calculate_residuals(y_true, y_pred)

        np.testing.assert_array_almost_equal(residuals, np.zeros(5))


class TestGetPredictionErrors:
    """Test suite for get_prediction_errors function."""

    def test_returns_dataframe(self, trained_simple_model, sample_train_test_split):
        """Test that function returns a DataFrame."""
        _, X_test, _, y_test = sample_train_test_split

        error_df = get_prediction_errors(trained_simple_model, X_test, y_test)

        assert isinstance(error_df, pd.DataFrame)

    def test_contains_expected_columns(self, trained_simple_model, sample_train_test_split):
        """Test that DataFrame contains expected columns."""
        _, X_test, _, y_test = sample_train_test_split

        error_df = get_prediction_errors(trained_simple_model, X_test, y_test)

        expected_columns = [
            "actual",
            "predicted",
            "error",
            "absolute_error",
            "squared_error",
        ]
        for col in expected_columns:
            assert col in error_df.columns

    def test_correct_number_of_rows(self, trained_simple_model, sample_train_test_split):
        """Test that DataFrame has correct number of rows."""
        _, X_test, _, y_test = sample_train_test_split

        error_df = get_prediction_errors(trained_simple_model, X_test, y_test)

        assert len(error_df) == len(y_test)

    def test_absolute_error_is_positive(self, trained_simple_model, sample_train_test_split):
        """Test that absolute errors are positive."""
        _, X_test, _, y_test = sample_train_test_split

        error_df = get_prediction_errors(trained_simple_model, X_test, y_test)

        assert (error_df["absolute_error"] >= 0).all()

    def test_squared_error_is_positive(self, trained_simple_model, sample_train_test_split):
        """Test that squared errors are positive."""
        _, X_test, _, y_test = sample_train_test_split

        error_df = get_prediction_errors(trained_simple_model, X_test, y_test)

        assert (error_df["squared_error"] >= 0).all()

    def test_absolute_error_calculation(self, trained_simple_model, sample_train_test_split):
        """Test that absolute error is calculated correctly."""
        _, X_test, _, y_test = sample_train_test_split

        error_df = get_prediction_errors(trained_simple_model, X_test, y_test)

        # Absolute error should equal |error|
        expected_abs_error = np.abs(error_df["error"])
        np.testing.assert_array_almost_equal(
            error_df["absolute_error"], expected_abs_error
        )


class TestPrintPredictionSummary:
    """Test suite for print_prediction_summary function."""

    def test_runs_without_error(self, trained_simple_model, sample_train_test_split):
        """Test that function runs without error."""
        _, X_test, _, y_test = sample_train_test_split

        # Should not raise any exception
        print_prediction_summary(trained_simple_model, X_test, y_test, n_samples=3)

    def test_handles_fewer_samples_than_requested(
        self, trained_simple_model, sample_train_test_split
    ):
        """Test that function handles when n_samples > test set size."""
        _, X_test, _, y_test = sample_train_test_split

        # Request more samples than available
        print_prediction_summary(trained_simple_model, X_test, y_test, n_samples=1000)


class TestCompareWithBaseline:
    """Test suite for compare_with_baseline function."""

    def test_runs_without_error(self):
        """Test that function runs without error."""
        metrics = {"r2_score": 0.5, "mse": 0.3, "rmse": 0.548, "mae": 0.4}

        # Should not raise any exception
        compare_with_baseline(metrics)

    def test_handles_negative_r2(self):
        """Test that function handles negative R² score."""
        metrics = {"r2_score": -0.5, "mse": 1.5, "rmse": 1.225, "mae": 1.0}

        # Should not raise any exception
        compare_with_baseline(metrics)

    def test_handles_perfect_score(self):
        """Test that function handles perfect R² score."""
        metrics = {"r2_score": 1.0, "mse": 0.0, "rmse": 0.0, "mae": 0.0}

        # Should not raise any exception
        compare_with_baseline(metrics)

    def test_handles_zero_score(self):
        """Test that function handles R² score of 0."""
        metrics = {"r2_score": 0.0, "mse": 0.5, "rmse": 0.707, "mae": 0.6}

        # Should not raise any exception
        compare_with_baseline(metrics)
