"""Pytest fixtures and shared test utilities.

This module provides common fixtures used across all test modules.
Follows PEP8 conventions.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

import config


@pytest.fixture
def sample_wine_data():
    """Create sample wine quality dataset for testing.

    Returns:
        DataFrame with sample wine features and quality scores.
    """
    np.random.seed(42)
    n_samples = 100

    data = {
        "fixed acidity": np.random.uniform(4, 16, n_samples),
        "volatile acidity": np.random.uniform(0.1, 1.6, n_samples),
        "citric acid": np.random.uniform(0, 1, n_samples),
        "residual sugar": np.random.uniform(0.9, 15, n_samples),
        "chlorides": np.random.uniform(0.01, 0.6, n_samples),
        "free sulfur dioxide": np.random.uniform(1, 72, n_samples),
        "total sulfur dioxide": np.random.uniform(6, 289, n_samples),
        "density": np.random.uniform(0.99, 1.01, n_samples),
        "pH": np.random.uniform(2.7, 4.0, n_samples),
        "sulphates": np.random.uniform(0.3, 2.0, n_samples),
        "alcohol": np.random.uniform(8, 15, n_samples),
        "quality": np.random.randint(3, 9, n_samples),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_train_test_split(sample_wine_data):
    """Create sample train/test split from wine data.

    Args:
        sample_wine_data: Sample wine quality dataset.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    from sklearn.model_selection import train_test_split

    X = sample_wine_data.drop(config.TARGET_COLUMN, axis=1)
    y = sample_wine_data[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture
def simple_pipeline():
    """Create a simple sklearn pipeline for testing.

    Returns:
        Pipeline with StandardScaler and RandomForestRegressor.
    """
    from sklearn.pipeline import make_pipeline

    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(n_estimators=10, random_state=42),
    )

    return pipeline


@pytest.fixture
def trained_simple_model(simple_pipeline, sample_train_test_split):
    """Create a trained model for testing.

    Args:
        simple_pipeline: Simple sklearn pipeline.
        sample_train_test_split: Sample train/test split.

    Returns:
        Trained pipeline model.
    """
    X_train, X_test, y_train, y_test = sample_train_test_split
    simple_pipeline.fit(X_train, y_train)
    return simple_pipeline


@pytest.fixture
def temp_model_path(tmp_path):
    """Create temporary path for model saving.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path object for temporary model file.
    """
    return tmp_path / "test_model.pkl"


@pytest.fixture
def mock_wine_csv(tmp_path, sample_wine_data):
    """Create temporary CSV file with wine data.

    Args:
        tmp_path: Pytest temporary directory fixture.
        sample_wine_data: Sample wine quality dataset.

    Returns:
        Path to temporary CSV file.
    """
    csv_path = tmp_path / "wine_data.csv"
    sample_wine_data.to_csv(csv_path, sep=";", index=False)
    return csv_path
