"""Configuration file for Wine Quality ML pipeline.

This module contains all constants, hyperparameters, and paths used throughout
the project. Following PEP8 conventions with type hints.
"""

from pathlib import Path
from typing import Dict, List, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Data source
DATASET_URL: str = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/winequality-red.csv"
)

# Data splitting parameters
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 123
STRATIFY_SPLIT: bool = True

# Model hyperparameters - RandomForest
MODEL_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "random_state": RANDOM_STATE,
}

# GridSearchCV hyperparameter grid - RandomForest
HYPERPARAM_GRID: Dict[str, List[Any]] = {
    "randomforestregressor__max_features": ["sqrt", "log2"],
    "randomforestregressor__max_depth": [None, 5, 3, 1],
}

# Model hyperparameters - GradientBoosting
GB_MODEL_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "random_state": RANDOM_STATE,
}

# GridSearchCV hyperparameter grid - GradientBoosting
GB_HYPERPARAM_GRID: Dict[str, List[Any]] = {
    "gradientboostingregressor__n_estimators": [50, 100, 150],
    "gradientboostingregressor__learning_rate": [0.05, 0.1, 0.15],
    "gradientboostingregressor__max_depth": [3, 5, 7],
    "pca__n_components": [5, 8, 10],
}

# Cross-validation settings
CV_FOLDS: int = 10
CV_N_JOBS: int = -1  # Use all available cores
CV_VERBOSE: int = 1

# Model saving - RandomForest
MODEL_FILENAME: str = "rf_regressor.pkl"
MODEL_PATH: Path = MODELS_DIR / MODEL_FILENAME

# Model saving - GradientBoosting
GB_MODEL_FILENAME: str = "gb_regressor.pkl"
GB_MODEL_PATH: Path = MODELS_DIR / GB_MODEL_FILENAME

# Feature columns (11 wine quality features)
FEATURE_COLUMNS: List[str] = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

# Target column
TARGET_COLUMN: str = "quality"

# Logging configuration
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL: str = "INFO"
LOG_FILE: Path = LOGS_DIR / "training.log"

# MLflow settings (for future use)
MLFLOW_TRACKING_URI: str = "mlruns"
MLFLOW_EXPERIMENT_NAME: str = "wine_quality_experiment"

# API settings (for future use)
API_HOST: str = "0.0.0.0"
API_PORT: int = 8000
API_RELOAD: bool = True
