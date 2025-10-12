"""Unit tests for preprocessing module.

Tests for src/preprocessing.py functions.
Follows PEP8 conventions.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.preprocessing import (
    create_preprocessing_pipeline,
    create_alternative_pipeline,
    get_feature_names,
    validate_feature_columns,
)
import config


class TestCreatePreprocessingPipeline:
    """Test suite for create_preprocessing_pipeline function."""

    def test_returns_pipeline_object(self):
        """Test that function returns a Pipeline object."""
        pipeline = create_preprocessing_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_correct_steps(self):
        """Test that pipeline contains expected steps."""
        pipeline = create_preprocessing_pipeline()

        # Should have 2 steps: scaler and model
        assert len(pipeline.steps) == 2

        step_names = [name for name, _ in pipeline.steps]
        assert "standardscaler" in step_names
        assert "randomforestregressor" in step_names

    def test_pipeline_uses_standard_scaler(self):
        """Test that pipeline uses StandardScaler."""
        pipeline = create_preprocessing_pipeline()

        scaler = pipeline.steps[0][1]
        assert isinstance(scaler, StandardScaler)

    def test_pipeline_uses_random_forest(self):
        """Test that pipeline uses RandomForestRegressor."""
        pipeline = create_preprocessing_pipeline()

        model = pipeline.steps[-1][1]
        assert isinstance(model, RandomForestRegressor)

    def test_custom_model_params(self):
        """Test pipeline creation with custom model parameters."""
        custom_params = {"n_estimators": 50, "max_depth": 5, "random_state": 42}
        pipeline = create_preprocessing_pipeline(model_params=custom_params)

        model = pipeline.steps[-1][1]
        assert model.n_estimators == 50
        assert model.max_depth == 5
        assert model.random_state == 42

    def test_pipeline_can_fit_and_predict(self, sample_train_test_split):
        """Test that pipeline can fit and predict."""
        pipeline = create_preprocessing_pipeline()
        X_train, X_test, y_train, y_test = sample_train_test_split

        # Should be able to fit
        pipeline.fit(X_train, y_train)

        # Should be able to predict
        predictions = pipeline.predict(X_test)
        assert len(predictions) == len(X_test)


class TestCreateAlternativePipeline:
    """Test suite for create_alternative_pipeline function."""

    def test_returns_pipeline_object(self):
        """Test that function returns a Pipeline object."""
        pipeline = create_alternative_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_correct_steps(self):
        """Test that pipeline contains expected steps."""
        pipeline = create_alternative_pipeline()

        # Should have 3 steps: scaler, PCA, and model
        assert len(pipeline.steps) == 3

        step_names = [name for name, _ in pipeline.steps]
        assert "standardscaler" in step_names
        assert "pca" in step_names
        assert "gradientboostingregressor" in step_names

    def test_pipeline_uses_standard_scaler(self):
        """Test that pipeline uses StandardScaler."""
        pipeline = create_alternative_pipeline()

        scaler = pipeline.steps[0][1]
        assert isinstance(scaler, StandardScaler)

    def test_pipeline_uses_pca(self):
        """Test that pipeline uses PCA."""
        pipeline = create_alternative_pipeline()

        pca = pipeline.steps[1][1]
        assert isinstance(pca, PCA)

    def test_pipeline_uses_gradient_boosting(self):
        """Test that pipeline uses GradientBoostingRegressor."""
        pipeline = create_alternative_pipeline()

        model = pipeline.steps[-1][1]
        assert isinstance(model, GradientBoostingRegressor)

    def test_custom_pca_components(self):
        """Test pipeline creation with custom PCA components."""
        pipeline = create_alternative_pipeline(pca_components=5)

        pca = pipeline.steps[1][1]
        assert pca.n_components == 5

    def test_custom_model_params(self):
        """Test pipeline creation with custom model parameters."""
        custom_params = {
            "n_estimators": 50,
            "learning_rate": 0.05,
            "max_depth": 5,
            "random_state": 42,
        }
        pipeline = create_alternative_pipeline(model_params=custom_params)

        model = pipeline.steps[-1][1]
        assert model.n_estimators == 50
        assert model.learning_rate == 0.05
        assert model.max_depth == 5
        assert model.random_state == 42

    def test_pipeline_can_fit_and_predict(self, sample_train_test_split):
        """Test that pipeline can fit and predict."""
        pipeline = create_alternative_pipeline()
        X_train, X_test, y_train, y_test = sample_train_test_split

        # Should be able to fit
        pipeline.fit(X_train, y_train)

        # Should be able to predict
        predictions = pipeline.predict(X_test)
        assert len(predictions) == len(X_test)


class TestGetFeatureNames:
    """Test suite for get_feature_names function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        features = get_feature_names()
        assert isinstance(features, list)

    def test_returns_expected_features(self):
        """Test that function returns expected feature names."""
        features = get_feature_names()

        expected_features = config.FEATURE_COLUMNS
        assert features == expected_features

    def test_returns_11_features(self):
        """Test that function returns 11 wine features."""
        features = get_feature_names()
        assert len(features) == 11

    def test_does_not_include_target(self):
        """Test that feature list does not include target column."""
        features = get_feature_names()
        assert config.TARGET_COLUMN not in features


class TestValidateFeatureColumns:
    """Test suite for validate_feature_columns function."""

    def test_valid_features_return_true(self):
        """Test that valid features return True."""
        valid_features = config.FEATURE_COLUMNS.copy()
        result = validate_feature_columns(valid_features)
        assert result is True

    def test_missing_features_return_false(self):
        """Test that missing features return False."""
        incomplete_features = config.FEATURE_COLUMNS[:5]  # Only first 5
        result = validate_feature_columns(incomplete_features)
        assert result is False

    def test_extra_features_return_false(self):
        """Test that extra features return False."""
        features_with_extra = config.FEATURE_COLUMNS + ["extra_feature"]
        result = validate_feature_columns(features_with_extra)
        assert result is False

    def test_wrong_features_return_false(self):
        """Test that wrong features return False."""
        wrong_features = ["wrong1", "wrong2", "wrong3"]
        result = validate_feature_columns(wrong_features)
        assert result is False

    def test_empty_list_return_false(self):
        """Test that empty feature list returns False."""
        result = validate_feature_columns([])
        assert result is False

    def test_reordered_features_return_true(self):
        """Test that reordered features still return True."""
        reordered_features = config.FEATURE_COLUMNS[::-1]  # Reversed order
        result = validate_feature_columns(reordered_features)
        assert result is True
