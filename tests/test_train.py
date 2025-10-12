"""Unit tests for training module.

Tests for src/train.py functions.
Follows PEP8 conventions.
"""

import joblib
import pytest
from pathlib import Path
from sklearn.model_selection import GridSearchCV

from src.train import (
    train_model_with_grid_search,
    save_model,
    load_model,
    get_best_params,
    get_cv_results,
)


class TestTrainModelWithGridSearch:
    """Test suite for train_model_with_grid_search function."""

    def test_returns_gridsearchcv_object(self, simple_pipeline, sample_train_test_split):
        """Test that function returns GridSearchCV object."""
        X_train, _, y_train, _ = sample_train_test_split

        param_grid = {"randomforestregressor__max_depth": [3, 5]}

        model = train_model_with_grid_search(
            simple_pipeline, X_train, y_train, param_grid, cv=2
        )

        assert isinstance(model, GridSearchCV)

    def test_model_is_fitted(self, simple_pipeline, sample_train_test_split):
        """Test that returned model is fitted."""
        X_train, _, y_train, _ = sample_train_test_split

        param_grid = {"randomforestregressor__max_depth": [3, 5]}

        model = train_model_with_grid_search(
            simple_pipeline, X_train, y_train, param_grid, cv=2
        )

        # Fitted GridSearchCV has best_params_ attribute
        assert hasattr(model, "best_params_")
        assert hasattr(model, "best_score_")

    def test_model_can_predict(self, simple_pipeline, sample_train_test_split):
        """Test that trained model can make predictions."""
        X_train, X_test, y_train, _ = sample_train_test_split

        param_grid = {"randomforestregressor__max_depth": [3, 5]}

        model = train_model_with_grid_search(
            simple_pipeline, X_train, y_train, param_grid, cv=2
        )

        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_uses_correct_cv_folds(self, simple_pipeline, sample_train_test_split):
        """Test that GridSearchCV uses correct number of CV folds."""
        X_train, _, y_train, _ = sample_train_test_split

        param_grid = {"randomforestregressor__max_depth": [3, 5]}

        model = train_model_with_grid_search(
            simple_pipeline, X_train, y_train, param_grid, cv=3
        )

        assert model.cv == 3

    def test_empty_training_data_raises_error(self, simple_pipeline):
        """Test that empty training data raises ValueError."""
        import pandas as pd

        X_train_empty = pd.DataFrame()
        y_train_empty = pd.Series(dtype=float)

        param_grid = {"randomforestregressor__max_depth": [3, 5]}

        with pytest.raises(ValueError, match="Training data cannot be empty"):
            train_model_with_grid_search(
                simple_pipeline, X_train_empty, y_train_empty, param_grid
            )

    def test_best_params_stored(self, simple_pipeline, sample_train_test_split):
        """Test that best parameters are stored correctly."""
        X_train, _, y_train, _ = sample_train_test_split

        param_grid = {"randomforestregressor__max_depth": [3, 5, 7]}

        model = train_model_with_grid_search(
            simple_pipeline, X_train, y_train, param_grid, cv=2
        )

        assert "randomforestregressor__max_depth" in model.best_params_
        assert model.best_params_["randomforestregressor__max_depth"] in [3, 5, 7]


class TestSaveModel:
    """Test suite for save_model function."""

    def test_saves_model_to_file(self, trained_simple_model, temp_model_path):
        """Test that model is saved to specified file."""
        save_model(trained_simple_model, temp_model_path)

        assert temp_model_path.exists()
        assert temp_model_path.stat().st_size > 0

    def test_saved_model_can_be_loaded(self, trained_simple_model, temp_model_path):
        """Test that saved model can be loaded back."""
        save_model(trained_simple_model, temp_model_path)

        loaded_model = joblib.load(temp_model_path)
        assert loaded_model is not None

    def test_saved_model_maintains_functionality(
        self, trained_simple_model, temp_model_path, sample_train_test_split
    ):
        """Test that saved model maintains prediction functionality."""
        X_train, X_test, y_train, y_test = sample_train_test_split

        save_model(trained_simple_model, temp_model_path)
        loaded_model = joblib.load(temp_model_path)

        predictions = loaded_model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_save_to_nonexistent_directory_raises_error(self, trained_simple_model):
        """Test that saving to nonexistent directory raises error."""
        invalid_path = Path("/nonexistent/directory/model.pkl")

        with pytest.raises(IOError):
            save_model(trained_simple_model, invalid_path)


class TestLoadModel:
    """Test suite for load_model function."""

    def test_loads_model_from_file(self, trained_simple_model, temp_model_path):
        """Test that model is loaded from file."""
        joblib.dump(trained_simple_model, temp_model_path)

        loaded_model = load_model(temp_model_path)
        assert loaded_model is not None

    def test_loaded_model_can_predict(
        self, trained_simple_model, temp_model_path, sample_train_test_split
    ):
        """Test that loaded model can make predictions."""
        X_train, X_test, y_train, y_test = sample_train_test_split

        joblib.dump(trained_simple_model, temp_model_path)
        loaded_model = load_model(temp_model_path)

        predictions = loaded_model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        nonexistent_path = Path("/tmp/nonexistent_model.pkl")

        with pytest.raises(FileNotFoundError):
            load_model(nonexistent_path)


class TestGetBestParams:
    """Test suite for get_best_params function."""

    def test_returns_dict(self, simple_pipeline, sample_train_test_split):
        """Test that function returns a dictionary."""
        X_train, _, y_train, _ = sample_train_test_split

        param_grid = {"randomforestregressor__max_depth": [3, 5]}

        model = train_model_with_grid_search(
            simple_pipeline, X_train, y_train, param_grid, cv=2
        )

        best_params = get_best_params(model)
        assert isinstance(best_params, dict)

    def test_contains_best_params(self, simple_pipeline, sample_train_test_split):
        """Test that returned dict contains best parameters."""
        X_train, _, y_train, _ = sample_train_test_split

        param_grid = {
            "randomforestregressor__max_depth": [3, 5],
            "randomforestregressor__n_estimators": [10, 20],
        }

        model = train_model_with_grid_search(
            simple_pipeline, X_train, y_train, param_grid, cv=2
        )

        best_params = get_best_params(model)

        assert "randomforestregressor__max_depth" in best_params
        assert "randomforestregressor__n_estimators" in best_params


class TestGetCVResults:
    """Test suite for get_cv_results function."""

    def test_returns_dataframe(self, simple_pipeline, sample_train_test_split):
        """Test that function returns a DataFrame."""
        X_train, _, y_train, _ = sample_train_test_split

        param_grid = {"randomforestregressor__max_depth": [3, 5]}

        model = train_model_with_grid_search(
            simple_pipeline, X_train, y_train, param_grid, cv=2
        )

        cv_results = get_cv_results(model)

        import pandas as pd

        assert isinstance(cv_results, pd.DataFrame)

    def test_contains_expected_columns(self, simple_pipeline, sample_train_test_split):
        """Test that DataFrame contains expected columns."""
        X_train, _, y_train, _ = sample_train_test_split

        param_grid = {"randomforestregressor__max_depth": [3, 5]}

        model = train_model_with_grid_search(
            simple_pipeline, X_train, y_train, param_grid, cv=2
        )

        cv_results = get_cv_results(model)

        expected_columns = [
            "params",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
            "mean_fit_time",
        ]

        for col in expected_columns:
            assert col in cv_results.columns

    def test_results_sorted_by_rank(self, simple_pipeline, sample_train_test_split):
        """Test that results are sorted by rank."""
        X_train, _, y_train, _ = sample_train_test_split

        param_grid = {"randomforestregressor__max_depth": [3, 5, 7]}

        model = train_model_with_grid_search(
            simple_pipeline, X_train, y_train, param_grid, cv=2
        )

        cv_results = get_cv_results(model)

        # First row should have rank 1
        assert cv_results.iloc[0]["rank_test_score"] == 1
