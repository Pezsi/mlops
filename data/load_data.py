"""Data loading and splitting utilities for Wine Quality dataset.

This module handles downloading the Wine Quality dataset and splitting it into
training and test sets. Follows PEP8 conventions with type hints.
"""

import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

import config

# Configure logging
logging.basicConfig(format=config.LOG_FORMAT, level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


def load_wine_quality_data(url: str = config.DATASET_URL) -> pd.DataFrame:
    """Load Wine Quality dataset from URL.

    Args:
        url: URL to the Wine Quality CSV file.
            Default: UCI ML Repository Wine Quality dataset.

    Returns:
        DataFrame containing wine quality data with features and target.

    Raises:
        ValueError: If dataset cannot be loaded or is empty.
        ConnectionError: If URL cannot be reached.
    """
    logger.info(f"Loading Wine Quality dataset from {url}")

    try:
        data = pd.read_csv(url, sep=";")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise ConnectionError(f"Could not download dataset from {url}") from e

    if data.empty:
        raise ValueError("Loaded dataset is empty")

    logger.info(
        f"Dataset loaded successfully: {data.shape[0]} samples, "
        f"{data.shape[1]} features"
    )
    logger.info(f"Features: {list(data.columns)}")

    # Validate expected columns
    expected_columns = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    missing_columns = set(expected_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    return data


def split_data(
    data: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    stratify: bool = config.STRATIFY_SPLIT,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and test sets.

    Args:
        data: DataFrame containing features and target.
        test_size: Proportion of dataset to include in test split (0.0-1.0).
        random_state: Random seed for reproducibility.
        stratify: Whether to stratify split based on target variable.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: If target column is missing or data is empty.
    """
    if data.empty:
        raise ValueError("Cannot split empty dataset")

    if config.TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{config.TARGET_COLUMN}' not found in data")

    logger.info("Splitting data into train/test sets...")

    # Separate features and target
    X = data.drop(config.TARGET_COLUMN, axis=1)
    y = data[config.TARGET_COLUMN]

    # Perform train/test split
    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    logger.info(f"Test set proportion: {test_size:.1%}")

    return X_train, X_test, y_train, y_test


def load_and_split_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load Wine Quality dataset and split into train/test sets.

    Convenience function that combines loading and splitting.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: If data loading or splitting fails.
        ConnectionError: If dataset cannot be downloaded.
    """
    logger.info("=" * 70)
    logger.info("Starting data loading and splitting pipeline")
    logger.info("=" * 70)

    # Load dataset
    data = load_wine_quality_data()

    # Split into train/test
    X_train, X_test, y_train, y_test = split_data(data)

    logger.info("Data loading and splitting completed successfully")
    logger.info("=" * 70)

    return X_train, X_test, y_train, y_test
