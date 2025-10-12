"""Unit tests for FastAPI application.

Tests for fastapi_app.py endpoints.
Follows PEP8 conventions.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from fastapi_app import app, models_cache


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_wine_features():
    """Sample wine features for testing."""
    return {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }


@pytest.fixture
def mock_trained_model(trained_simple_model):
    """Mock trained model in cache."""
    models_cache["rf"] = trained_simple_model
    models_cache["gb"] = trained_simple_model
    yield
    models_cache.clear()


class TestGeneralEndpoints:
    """Test suite for general endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns welcome message."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Wine Quality" in data["message"]

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data

    def test_get_features(self, client):
        """Test get features endpoint."""
        response = client.get("/features")
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert len(data["features"]) == 11


class TestModelsEndpoints:
    """Test suite for models endpoints."""

    def test_list_models(self, client):
        """Test list models endpoint."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2  # RF and GB models

    def test_get_model_metrics_invalid_name(self, client):
        """Test get metrics with invalid model name."""
        response = client.get("/metrics/invalid")
        assert response.status_code == 422  # Validation error


class TestPredictionEndpoints:
    """Test suite for prediction endpoints."""

    def test_predict_without_model(self, client, sample_wine_features):
        """Test prediction when model doesn't exist."""
        models_cache.clear()

        request_data = {
            "features": sample_wine_features,
            "model_name": "rf",
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 404

    def test_predict_with_model(
        self, client, sample_wine_features, mock_trained_model
    ):
        """Test successful prediction."""
        request_data = {
            "features": sample_wine_features,
            "model_name": "rf",
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "quality_prediction" in data
        assert "model_used" in data
        assert data["model_used"] == "rf"

    def test_predict_invalid_features(self, client, mock_trained_model):
        """Test prediction with invalid features."""
        invalid_request = {
            "features": {
                "fixed_acidity": -1,  # Invalid: negative value
                "volatile_acidity": 0.7,
            },
            "model_name": "rf",
        }

        response = client.post("/predict", json=invalid_request)
        assert response.status_code == 422  # Validation error

    def test_predict_missing_features(self, client, mock_trained_model):
        """Test prediction with missing features."""
        incomplete_request = {
            "features": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.7,
            },
            "model_name": "rf",
        }

        response = client.post("/predict", json=incomplete_request)
        assert response.status_code == 422  # Validation error

    def test_predict_invalid_model_name(self, client, sample_wine_features):
        """Test prediction with invalid model name."""
        request_data = {
            "features": sample_wine_features,
            "model_name": "invalid",
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_batch_predict(
        self, client, sample_wine_features, mock_trained_model
    ):
        """Test batch prediction."""
        request_data = {
            "features_list": [sample_wine_features, sample_wine_features],
            "model_name": "rf",
        }

        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        assert "model_used" in data

    def test_model_predict_alias(
        self, client, sample_wine_features, mock_trained_model
    ):
        """Test /model/predict alias endpoint."""
        request_data = {
            "features": sample_wine_features,
            "model_name": "rf",
        }

        response = client.post("/model/predict", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "quality_prediction" in data


class TestTrainingEndpoints:
    """Test suite for training endpoints."""

    @patch("fastapi_app.load_wine_quality_data")
    @patch("fastapi_app.train_model_with_grid_search")
    @patch("fastapi_app.evaluate_model")
    @patch("fastapi_app.save_model")
    def test_train_single_model(
        self,
        mock_save,
        mock_evaluate,
        mock_train,
        mock_load_data,
        client,
        sample_wine_data,
        trained_simple_model,
    ):
        """Test training single model endpoint."""
        # Setup mocks
        mock_load_data.return_value = sample_wine_data
        mock_train.return_value = trained_simple_model
        mock_evaluate.return_value = {
            "r2_score": 0.5,
            "mse": 0.4,
            "rmse": 0.632,
            "mae": 0.5,
        }

        request_data = {"pipeline_type": "rf"}

        response = client.post("/model/train", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "model" in data

    @patch("fastapi_app.load_wine_quality_data")
    @patch("fastapi_app.compare_pipelines")
    @patch("fastapi_app.evaluate_model")
    @patch("fastapi_app.save_model")
    def test_train_compare_models(
        self,
        mock_save,
        mock_evaluate,
        mock_compare,
        mock_load_data,
        client,
        sample_wine_data,
        trained_simple_model,
    ):
        """Test training with compare pipeline type."""
        # Setup mocks
        mock_load_data.return_value = sample_wine_data
        mock_compare.return_value = {
            "rf": trained_simple_model,
            "gb": trained_simple_model,
        }
        mock_evaluate.return_value = {
            "r2_score": 0.5,
            "mse": 0.4,
            "rmse": 0.632,
            "mae": 0.5,
        }

        request_data = {"pipeline_type": "compare"}

        response = client.post("/model/train", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "models" in data
        assert "rf" in data["models"]
        assert "gb" in data["models"]


class TestInputValidation:
    """Test suite for input validation."""

    def test_wine_features_validation_ranges(self, client, mock_trained_model):
        """Test that feature ranges are validated."""
        # Test out of range values
        out_of_range_features = {
            "fixed_acidity": 100,  # Too high
            "volatile_acidity": 0.7,
            "citric_acid": 0.0,
            "residual_sugar": 1.9,
            "chlorides": 0.076,
            "free_sulfur_dioxide": 11.0,
            "total_sulfur_dioxide": 34.0,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4,
        }

        request_data = {
            "features": out_of_range_features,
            "model_name": "rf",
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 422

    def test_batch_prediction_empty_list(self, client, mock_trained_model):
        """Test batch prediction with empty list."""
        request_data = {"features_list": [], "model_name": "rf"}

        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 422
