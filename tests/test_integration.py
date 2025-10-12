"""Integration tests for end-to-end ML pipeline.

Tests complete workflows combining multiple modules.
Follows PEP8 conventions.
"""

from unittest.mock import patch

from data.load_data import load_and_split_data
from src.preprocessing import (
    create_preprocessing_pipeline,
    create_alternative_pipeline,
)
from src.train import train_model_with_grid_search, save_model, load_model
from src.evaluate import evaluate_model


class TestEndToEndPipeline:
    """Test suite for complete end-to-end ML pipeline."""

    @patch("data.load_data.load_wine_quality_data")
    def test_complete_pipeline_rf(self, mock_load, sample_wine_data, temp_model_path):
        """Test complete RandomForest pipeline from data loading to evaluation."""
        # Mock data loading
        mock_load.return_value = sample_wine_data

        # Step 1: Load and split data
        X_train, X_test, y_train, y_test = load_and_split_data()

        assert len(X_train) > 0
        assert len(X_test) > 0

        # Step 2: Create preprocessing pipeline
        pipeline = create_preprocessing_pipeline()

        assert pipeline is not None

        # Step 3: Train model with grid search
        param_grid = {"randomforestregressor__max_depth": [3, 5]}

        trained_model = train_model_with_grid_search(
            pipeline, X_train, y_train, param_grid, cv=2
        )

        assert trained_model is not None
        assert hasattr(trained_model, "best_params_")

        # Step 4: Evaluate model
        metrics = evaluate_model(trained_model, X_test, y_test, log_results=False)

        assert "r2_score" in metrics
        assert "rmse" in metrics
        assert metrics["test_samples"] == len(y_test)

        # Step 5: Save model
        save_model(trained_model, temp_model_path)

        assert temp_model_path.exists()

        # Step 6: Load model
        loaded_model = load_model(temp_model_path)

        assert loaded_model is not None

        # Step 7: Evaluate loaded model
        predictions = loaded_model.predict(X_test)

        assert len(predictions) == len(y_test)

    @patch("data.load_data.load_wine_quality_data")
    def test_complete_pipeline_gb(self, mock_load, sample_wine_data, temp_model_path):
        """Test complete GradientBoosting pipeline from data loading to evaluation."""
        # Mock data loading
        mock_load.return_value = sample_wine_data

        # Step 1: Load and split data
        X_train, X_test, y_train, y_test = load_and_split_data()

        # Step 2: Create alternative (GB) preprocessing pipeline
        pipeline = create_alternative_pipeline(pca_components=5)

        assert pipeline is not None

        # Step 3: Train model with grid search
        param_grid = {
            "gradientboostingregressor__n_estimators": [50, 100],
            "gradientboostingregressor__learning_rate": [0.1],
            "pca__n_components": [5],
        }

        trained_model = train_model_with_grid_search(
            pipeline, X_train, y_train, param_grid, cv=2
        )

        assert trained_model is not None

        # Step 4: Evaluate model
        metrics = evaluate_model(trained_model, X_test, y_test, log_results=False)

        assert metrics["r2_score"] is not None

        # Step 5: Save and load model
        save_model(trained_model, temp_model_path)
        loaded_model = load_model(temp_model_path)

        # Verify loaded model works
        predictions = loaded_model.predict(X_test)
        assert len(predictions) == len(y_test)


class TestPipelineComponents:
    """Test suite for pipeline component integration."""

    def test_preprocessing_to_training(self, sample_train_test_split):
        """Test integration between preprocessing and training."""
        X_train, X_test, y_train, y_test = sample_train_test_split

        # Create pipeline
        pipeline = create_preprocessing_pipeline()

        # Train with minimal grid search
        param_grid = {"randomforestregressor__max_depth": [3]}

        trained_model = train_model_with_grid_search(
            pipeline, X_train, y_train, param_grid, cv=2
        )

        # Verify training worked
        assert trained_model is not None
        predictions = trained_model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_training_to_evaluation(self, sample_train_test_split):
        """Test integration between training and evaluation."""
        X_train, X_test, y_train, y_test = sample_train_test_split

        # Create and train pipeline
        pipeline = create_preprocessing_pipeline()
        param_grid = {"randomforestregressor__max_depth": [3]}

        trained_model = train_model_with_grid_search(
            pipeline, X_train, y_train, param_grid, cv=2
        )

        # Evaluate model
        metrics = evaluate_model(trained_model, X_test, y_test, log_results=False)

        # Verify evaluation worked
        assert metrics["r2_score"] is not None
        assert metrics["rmse"] > 0

    def test_model_save_load_cycle(self, trained_simple_model, temp_model_path):
        """Test complete save/load cycle maintains model functionality."""
        # Save model
        save_model(trained_simple_model, temp_model_path)

        # Load model
        loaded_model = load_model(temp_model_path)

        # Verify both models make same predictions
        import numpy as np

        test_data = np.random.randn(5, 11)

        orig_preds = trained_simple_model.predict(test_data)
        loaded_preds = loaded_model.predict(test_data)

        np.testing.assert_array_almost_equal(orig_preds, loaded_preds)


class TestPipelineEdgeCases:
    """Test suite for pipeline edge cases and error handling."""

    def test_pipeline_with_minimal_data(self):
        """Test pipeline works with minimal amount of data."""
        import pandas as pd
        import numpy as np

        # Create minimal dataset (30 samples)
        np.random.seed(42)
        n_samples = 30

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

        df = pd.DataFrame(data)

        X = df.drop("quality", axis=1)
        y = df["quality"]

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create and train pipeline
        pipeline = create_preprocessing_pipeline()

        # Use minimal grid search with 2 folds
        param_grid = {"randomforestregressor__max_depth": [3]}

        trained_model = train_model_with_grid_search(
            pipeline, X_train, y_train, param_grid, cv=2
        )

        # Verify it works
        predictions = trained_model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_pipeline_preserves_data_types(self, sample_train_test_split):
        """Test that pipeline preserves correct data types throughout."""
        X_train, X_test, y_train, y_test = sample_train_test_split

        import pandas as pd

        # Verify input types
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)

        # Create and train pipeline
        pipeline = create_preprocessing_pipeline()
        pipeline.fit(X_train, y_train)

        # Verify predictions are numpy arrays
        predictions = pipeline.predict(X_test)

        import numpy as np

        assert isinstance(predictions, np.ndarray)

    def test_both_pipelines_produce_valid_results(self, sample_train_test_split):
        """Test that both RF and GB pipelines produce valid results."""
        X_train, X_test, y_train, y_test = sample_train_test_split

        # Test RandomForest pipeline
        pipeline_rf = create_preprocessing_pipeline()
        param_grid_rf = {"randomforestregressor__max_depth": [3]}

        trained_rf = train_model_with_grid_search(
            pipeline_rf, X_train, y_train, param_grid_rf, cv=2
        )

        metrics_rf = evaluate_model(trained_rf, X_test, y_test, log_results=False)

        # Test GradientBoosting pipeline
        pipeline_gb = create_alternative_pipeline(pca_components=5)
        param_grid_gb = {
            "gradientboostingregressor__n_estimators": [50],
            "pca__n_components": [5],
        }

        trained_gb = train_model_with_grid_search(
            pipeline_gb, X_train, y_train, param_grid_gb, cv=2
        )

        metrics_gb = evaluate_model(trained_gb, X_test, y_test, log_results=False)

        # Both should produce valid metrics
        assert metrics_rf["r2_score"] is not None
        assert metrics_gb["r2_score"] is not None
        assert metrics_rf["rmse"] > 0
        assert metrics_gb["rmse"] > 0
