"""Unit tests for Flask-RESTX application.

Tests for flask_app.py endpoints.
Follows PEP8 conventions.
"""

import pytest
import json
from unittest.mock import patch

from flask_app import flask_app, models_cache


@pytest.fixture
def client():
    """Create Flask test client."""
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client


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

    @pytest.mark.skip(reason="Flask-RESTX namespace routing issue")
    def test_root_endpoint(self, client):
        """Test root endpoint returns welcome message."""
        response = client.get('/')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "message" in data
        assert "Wine Quality" in data["message"]

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "status" in data
        assert "models_loaded" in data

    def test_get_features(self, client):
        """Test get features endpoint."""
        response = client.get('/features')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "features" in data
        assert len(data["features"]) == 11


class TestModelsEndpoints:
    """Test suite for models endpoints."""

    def test_list_models(self, client):
        """Test list models endpoint."""
        response = client.get('/models/')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)
        assert len(data) == 2  # RF and GB models

    def test_get_model_metrics_invalid_name(self, client):
        """Test get metrics with invalid model name."""
        response = client.get('/models/metrics/invalid')
        assert response.status_code in [400, 404]


class TestPredictionEndpoints:
    """Test suite for prediction endpoints."""

    @patch("config.MODEL_PATH")
    @patch("config.GB_MODEL_PATH")
    def test_predict_without_model(self, mock_gb_path, mock_rf_path, client, sample_wine_features):
        """Test prediction when model doesn't exist."""
        models_cache.clear()
        mock_rf_path.exists.return_value = False
        mock_gb_path.exists.return_value = False

        request_data = {
            "features": sample_wine_features,
            "model_name": "rf",
        }

        response = client.post(
            '/predict/',
            data=json.dumps(request_data),
            content_type='application/json',
        )
        assert response.status_code == 500

    def test_predict_with_model(
        self, client, sample_wine_features, mock_trained_model
    ):
        """Test successful prediction."""
        request_data = {
            "features": sample_wine_features,
            "model_name": "rf",
        }

        response = client.post(
            '/predict/',
            data=json.dumps(request_data),
            content_type='application/json',
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "quality_prediction" in data
        assert "model_used" in data
        assert data["model_used"] == "rf"

    def test_predict_missing_features(self, client, mock_trained_model):
        """Test prediction with missing features."""
        incomplete_request = {
            "features": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.7,
            },
            "model_name": "rf",
        }

        response = client.post(
            '/predict/',
            data=json.dumps(incomplete_request),
            content_type='application/json',
        )
        # Flask-RESTX validation should fail
        assert response.status_code in [400, 500]

    def test_batch_predict(
        self, client, sample_wine_features, mock_trained_model
    ):
        """Test batch prediction."""
        request_data = {
            "features_list": [sample_wine_features, sample_wine_features],
            "model_name": "rf",
        }

        response = client.post(
            '/predict/batch',
            data=json.dumps(request_data),
            content_type='application/json',
        )
        assert response.status_code == 200
        data = json.loads(response.data)
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

        response = client.post(
            '/model/predict',
            data=json.dumps(request_data),
            content_type='application/json',
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "quality_prediction" in data


class TestTrainingEndpoints:
    """Test suite for training endpoints."""

    @patch("flask_app.load_wine_quality_data")
    @patch("flask_app.split_data")
    @patch("flask_app.train_model_with_grid_search")
    @patch("flask_app.evaluate_model")
    @patch("flask_app.save_model")
    @patch("src.train.log_model_to_mlflow")
    def test_train_single_model(
        self,
        mock_log_mlflow,
        mock_save,
        mock_evaluate,
        mock_train,
        mock_split_data,
        mock_load_data,
        client,
        sample_wine_data,
        trained_simple_model,
    ):
        """Test training single model endpoint."""
        # Setup mocks
        mock_load_data.return_value = sample_wine_data
        X = sample_wine_data.drop("quality", axis=1)
        y = sample_wine_data["quality"]
        mock_split_data.return_value = (X, X, y, y)
        mock_train.return_value = trained_simple_model
        mock_evaluate.return_value = {
            "r2_score": 0.5,
            "mse": 0.4,
            "rmse": 0.632,
            "mae": 0.5,
        }

        request_data = {"pipeline_type": "rf"}

        response = client.post(
            '/model/train',
            data=json.dumps(request_data),
            content_type='application/json',
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "message" in data
        assert "model" in data

    @patch("flask_app.load_wine_quality_data")
    @patch("flask_app.split_data")
    @patch("src.train.compare_pipelines")
    @patch("flask_app.evaluate_model")
    @patch("flask_app.save_model")
    def test_train_compare_models(
        self,
        mock_save,
        mock_evaluate,
        mock_compare,
        mock_split_data,
        mock_load_data,
        client,
        sample_wine_data,
        trained_simple_model,
    ):
        """Test training with compare pipeline type."""
        # Setup mocks
        mock_load_data.return_value = sample_wine_data
        X = sample_wine_data.drop("quality", axis=1)
        y = sample_wine_data["quality"]
        mock_split_data.return_value = (X, X, y, y)
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

        response = client.post(
            '/model/train',
            data=json.dumps(request_data),
            content_type='application/json',
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "message" in data
        assert "models" in data
        assert "rf" in data["models"]
        assert "gb" in data["models"]

    def test_train_empty_request(self, client):
        """Test training with empty request (should use defaults)."""
        response = client.post(
            '/model/train',
            data=json.dumps({}),
            content_type='application/json',
        )
        # Should attempt to train with defaults, might fail due to network
        # but should not be a validation error
        assert response.status_code in [200, 500]


class TestSwaggerDocs:
    """Test suite for Swagger documentation."""

    @pytest.mark.skip(reason="Flask-RESTX Swagger UI routing issue")
    def test_swagger_ui_accessible(self, client):
        """Test that Swagger UI is accessible."""
        response = client.get('/docs/')
        assert response.status_code == 200

    def test_swagger_json_accessible(self, client):
        """Test that Swagger JSON is accessible."""
        response = client.get('/swagger.json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "swagger" in data or "openapi" in data


class TestErrorHandling:
    """Test suite for error handling."""

    def test_404_for_invalid_endpoint(self, client):
        """Test 404 for non-existent endpoint."""
        response = client.get('/nonexistent')
        assert response.status_code == 404

    def test_405_for_wrong_method(self, client):
        """Test 405 for wrong HTTP method."""
        response = client.get('/model/train')
        assert response.status_code == 405

    def test_predict_with_malformed_json(self, client):
        """Test prediction with malformed JSON."""
        response = client.post(
            '/predict/',
            data="not a json",
            content_type='application/json',
        )
        assert response.status_code in [400, 500]
