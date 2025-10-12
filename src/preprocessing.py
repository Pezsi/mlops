"""Preprocessing utilities for Wine Quality dataset.

This module handles feature scaling and pipeline creation for the ML model.
Follows PEP8 conventions with type hints.
"""

import logging
from typing import Any

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline, make_pipeline

import config

# Configure logging
logging.basicConfig(format=config.LOG_FORMAT, level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


def create_preprocessing_pipeline(
    model_params: dict[str, Any] = None,
) -> Pipeline:
    """Create preprocessing and model pipeline.

    Creates a scikit-learn pipeline that includes:
    1. StandardScaler for feature normalization
    2. RandomForestRegressor model

    Args:
        model_params: Dictionary of model hyperparameters.
            If None, uses default parameters from config.

    Returns:
        sklearn Pipeline with StandardScaler and RandomForestRegressor.

    Example:
        >>> pipeline = create_preprocessing_pipeline()
        >>> pipeline.fit(X_train, y_train)
        >>> predictions = pipeline.predict(X_test)
    """
    if model_params is None:
        model_params = config.MODEL_PARAMS

    logger.info("Creating preprocessing + model pipeline")
    logger.info(f"Model parameters: {model_params}")

    # Create pipeline with StandardScaler and RandomForest
    pipeline = make_pipeline(
        preprocessing.StandardScaler(), RandomForestRegressor(**model_params)
    )

    logger.info(f"Pipeline created: {pipeline}")
    logger.info("Pipeline steps:")
    for i, (name, transformer) in enumerate(pipeline.steps, 1):
        logger.info(f"  {i}. {name}: {transformer.__class__.__name__}")

    return pipeline


def create_alternative_pipeline(
    model_params: dict[str, Any] = None,
    pca_components: int = 8,
) -> Pipeline:
    """Create alternative preprocessing and model pipeline.

    Uses PCA and GradientBoosting.

    Creates a scikit-learn pipeline that includes:
    1. StandardScaler for feature normalization
    2. PCA for dimensionality reduction
    3. GradientBoostingRegressor model

    Args:
        model_params: Dictionary of model hyperparameters.
            If None, uses default parameters from config.
        pca_components: Number of PCA components to keep (default: 8).

    Returns:
        sklearn Pipeline with StandardScaler, PCA, and GradientBoostingRegressor.

    Example:
        >>> pipeline = create_alternative_pipeline()
        >>> pipeline.fit(X_train, y_train)
        >>> predictions = pipeline.predict(X_test)
    """
    if model_params is None:
        model_params = config.GB_MODEL_PARAMS

    logger.info(
        "Creating alternative preprocessing + model pipeline (GradientBoosting)"
    )
    logger.info(f"Model parameters: {model_params}")
    logger.info(f"PCA components: {pca_components}")

    # Create pipeline with StandardScaler, PCA, and GradientBoosting
    pipeline = make_pipeline(
        preprocessing.StandardScaler(),
        PCA(n_components=pca_components),
        GradientBoostingRegressor(**model_params),
    )

    logger.info(f"Pipeline created: {pipeline}")
    logger.info("Pipeline steps:")
    for i, (name, transformer) in enumerate(pipeline.steps, 1):
        logger.info(f"  {i}. {name}: {transformer.__class__.__name__}")

    return pipeline


def get_feature_names() -> list[str]:
    """Get list of feature column names.

    Returns:
        List of feature column names from config.
    """
    return config.FEATURE_COLUMNS


def validate_feature_columns(feature_names: list[str]) -> bool:
    """Validate that provided feature names match expected features.

    Args:
        feature_names: List of feature column names to validate.

    Returns:
        True if features match, False otherwise.
    """
    expected_features = set(config.FEATURE_COLUMNS)
    provided_features = set(feature_names)

    if expected_features != provided_features:
        missing = expected_features - provided_features
        extra = provided_features - expected_features

        if missing:
            logger.warning(f"Missing expected features: {missing}")
        if extra:
            logger.warning(f"Unexpected extra features: {extra}")

        return False

    logger.info("Feature validation passed")
    return True
