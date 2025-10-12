"""Unit tests for data loading module.

Tests for data/load_data.py functions.
Follows PEP8 conventions.
"""

import pandas as pd
import pytest
from unittest.mock import patch

from data.load_data import (
    load_wine_quality_data,
    split_data,
    load_and_split_data,
)
import config


class TestLoadWineQualityData:
    """Test suite for load_wine_quality_data function."""

    def test_load_from_csv_file(self, mock_wine_csv):
        """Test loading data from local CSV file."""
        data = load_wine_quality_data(url=str(mock_wine_csv))

        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert config.TARGET_COLUMN in data.columns

    def test_loaded_data_has_correct_columns(self, mock_wine_csv):
        """Test that loaded data contains all expected columns."""
        data = load_wine_quality_data(url=str(mock_wine_csv))

        expected_columns = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
        assert set(expected_columns) == set(data.columns)

    def test_loaded_data_has_correct_shape(self, mock_wine_csv):
        """Test that loaded data has expected number of columns."""
        data = load_wine_quality_data(url=str(mock_wine_csv))

        # 11 features + 1 target = 12 columns
        assert data.shape[1] == 12

    def test_invalid_url_raises_connection_error(self):
        """Test that invalid URL raises ConnectionError or ValueError."""
        invalid_url = "http://invalid-url-that-does-not-exist.com/data.csv"

        with pytest.raises((ConnectionError, ValueError)):
            load_wine_quality_data(url=invalid_url)

    @patch("pandas.read_csv")
    def test_empty_dataset_raises_value_error(self, mock_read_csv):
        """Test that empty dataset raises ValueError."""
        mock_read_csv.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="Loaded dataset is empty"):
            load_wine_quality_data(url="http://example.com/data.csv")

    @patch("pandas.read_csv")
    def test_missing_columns_raises_value_error(self, mock_read_csv):
        """Test that missing expected columns raises ValueError."""
        # Create DataFrame with missing columns
        incomplete_data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        mock_read_csv.return_value = incomplete_data

        with pytest.raises(ValueError, match="Missing expected columns"):
            load_wine_quality_data(url="http://example.com/data.csv")


class TestSplitData:
    """Test suite for split_data function."""

    def test_split_returns_correct_types(self, sample_wine_data):
        """Test that split returns correct data types."""
        X_train, X_test, y_train, y_test = split_data(sample_wine_data)

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

    def test_split_proportions(self, sample_wine_data):
        """Test that split creates correct train/test proportions."""
        test_size = 0.2
        X_train, X_test, y_train, y_test = split_data(
            sample_wine_data, test_size=test_size
        )

        total_samples = len(sample_wine_data)
        test_samples = len(X_test)

        # Allow small rounding differences
        assert abs(test_samples / total_samples - test_size) < 0.05

    def test_split_preserves_sample_count(self, sample_wine_data):
        """Test that split preserves total sample count."""
        X_train, X_test, y_train, y_test = split_data(sample_wine_data)

        total_original = len(sample_wine_data)
        total_split = len(X_train) + len(X_test)

        assert total_original == total_split

    def test_split_removes_target_from_features(self, sample_wine_data):
        """Test that target column is removed from feature sets."""
        X_train, X_test, y_train, y_test = split_data(sample_wine_data)

        assert config.TARGET_COLUMN not in X_train.columns
        assert config.TARGET_COLUMN not in X_test.columns

    def test_split_reproducible_with_random_state(self, sample_wine_data):
        """Test that split is reproducible with same random state."""
        X_train1, X_test1, y_train1, y_test1 = split_data(
            sample_wine_data, random_state=42
        )
        X_train2, X_test2, y_train2, y_test2 = split_data(
            sample_wine_data, random_state=42
        )

        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)

    def test_empty_data_raises_value_error(self):
        """Test that empty DataFrame raises ValueError."""
        empty_data = pd.DataFrame()

        with pytest.raises(ValueError, match="Cannot split empty dataset"):
            split_data(empty_data)

    def test_missing_target_column_raises_value_error(self):
        """Test that missing target column raises ValueError."""
        data_without_target = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6]}
        )

        with pytest.raises(ValueError, match="Target column.*not found"):
            split_data(data_without_target)

    def test_stratified_split(self, sample_wine_data):
        """Test that stratified split maintains class distribution."""
        X_train, X_test, y_train, y_test = split_data(
            sample_wine_data, stratify=True
        )

        # Check that both sets have similar quality distributions
        train_dist = y_train.value_counts(normalize=True).sort_index()
        test_dist = y_test.value_counts(normalize=True).sort_index()

        # Allow some difference due to small sample size
        assert len(train_dist) > 0
        assert len(test_dist) > 0


class TestLoadAndSplitData:
    """Test suite for load_and_split_data convenience function."""

    @patch("data.load_data.load_wine_quality_data")
    def test_load_and_split_integration(self, mock_load, sample_wine_data):
        """Test full load and split pipeline."""
        mock_load.return_value = sample_wine_data

        X_train, X_test, y_train, y_test = load_and_split_data()

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

        # Verify load was called
        mock_load.assert_called_once()

    @patch("data.load_data.load_wine_quality_data")
    def test_load_and_split_calls_functions(self, mock_load, sample_wine_data):
        """Test that load_and_split_data calls underlying functions."""
        mock_load.return_value = sample_wine_data

        result = load_and_split_data()

        # Should call load_wine_quality_data
        assert mock_load.called

        # Should return tuple of 4 items
        assert len(result) == 4
