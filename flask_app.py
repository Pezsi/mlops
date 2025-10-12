"""Flask-RESTX application for Wine Quality prediction.

This module provides REST API endpoints for:
- Model training with MLflow tracking
- Wine quality prediction
- Model management and metrics
- Health checks
- Swagger UI documentation

Follows PEP8 conventions with type hints.
"""

import logging
import traceback
from typing import Dict, Any

import mlflow
import pandas as pd
from flask import Flask, request
from flask_restx import Api, Resource, fields, Namespace

import config
from data.load_data import load_wine_quality_data, split_data
from src.preprocessing import (
    create_preprocessing_pipeline,
    create_alternative_pipeline,
)
from src.train import (
    train_model_with_grid_search,
    save_model,
    load_model,
)
from src.evaluate import evaluate_model

# Configure logging
logging.basicConfig(
    format=config.LOG_FORMAT,
    level=config.LOG_LEVEL,
)
logger = logging.getLogger(__name__)

# Initialize Flask app
flask_app = Flask(__name__)
flask_app.config['RESTX_MASK_SWAGGER'] = False

# Initialize Flask-RESTX API
api = Api(
    flask_app,
    version='1.0.0',
    title='Wine Quality Prediction API',
    description='MLOps API for wine quality prediction with MLflow integration',
    doc='/docs',
)

# Create namespaces
ns_general = Namespace('general', description='General operations')
ns_models = Namespace('models', description='Model management')
ns_predict = Namespace('predict', description='Prediction operations')
ns_train = Namespace('train', description='Model training')

api.add_namespace(ns_general, path='/')
api.add_namespace(ns_models, path='/models')
api.add_namespace(ns_predict, path='/predict')
api.add_namespace(ns_train, path='/model')

# Global model cache
models_cache: Dict[str, Any] = {}


# Flask-RESTX models for request/response validation
wine_features_model = api.model('WineFeatures', {
    'fixed_acidity': fields.Float(
        required=True,
        description='Fixed acidity (g/dm³)',
        min=0,
        max=20,
        example=7.4
    ),
    'volatile_acidity': fields.Float(
        required=True,
        description='Volatile acidity (g/dm³)',
        min=0,
        max=2,
        example=0.7
    ),
    'citric_acid': fields.Float(
        required=True,
        description='Citric acid (g/dm³)',
        min=0,
        max=2,
        example=0.0
    ),
    'residual_sugar': fields.Float(
        required=True,
        description='Residual sugar (g/dm³)',
        min=0,
        max=20,
        example=1.9
    ),
    'chlorides': fields.Float(
        required=True,
        description='Chlorides (g/dm³)',
        min=0,
        max=1,
        example=0.076
    ),
    'free_sulfur_dioxide': fields.Float(
        required=True,
        description='Free sulfur dioxide (mg/dm³)',
        min=0,
        max=100,
        example=11.0
    ),
    'total_sulfur_dioxide': fields.Float(
        required=True,
        description='Total sulfur dioxide (mg/dm³)',
        min=0,
        max=400,
        example=34.0
    ),
    'density': fields.Float(
        required=True,
        description='Density (g/cm³)',
        min=0.98,
        max=1.01,
        example=0.9978
    ),
    'pH': fields.Float(
        required=True,
        description='pH value',
        min=2.5,
        max=4.5,
        example=3.51
    ),
    'sulphates': fields.Float(
        required=True,
        description='Sulphates (g/dm³)',
        min=0,
        max=2.5,
        example=0.56
    ),
    'alcohol': fields.Float(
        required=True,
        description='Alcohol (% vol.)',
        min=8,
        max=15,
        example=9.4
    ),
})

prediction_request_model = api.model('PredictionRequest', {
    'features': fields.Nested(wine_features_model, required=True),
    'model_name': fields.String(
        required=False,
        default='rf',
        description="Model to use: 'rf' or 'gb'",
        enum=['rf', 'gb']
    ),
})

batch_prediction_request_model = api.model('BatchPredictionRequest', {
    'features_list': fields.List(
        fields.Nested(wine_features_model),
        required=True,
        description='List of wine features'
    ),
    'model_name': fields.String(
        required=False,
        default='rf',
        description="Model to use: 'rf' or 'gb'",
        enum=['rf', 'gb']
    ),
})

prediction_response_model = api.model('PredictionResponse', {
    'quality_prediction': fields.Float(description='Predicted wine quality'),
    'model_used': fields.String(description='Model used for prediction'),
})

batch_prediction_response_model = api.model('BatchPredictionResponse', {
    'predictions': fields.List(fields.Float),
    'model_used': fields.String(description='Model used for prediction'),
})

training_request_model = api.model('TrainingRequest', {
    'pipeline_type': fields.String(
        required=False,
        default='rf',
        description="Pipeline type: 'rf', 'gb', or 'compare'",
        enum=['rf', 'gb', 'compare']
    ),
    'data_url': fields.String(
        required=False,
        description='URL to dataset (default: UCI dataset)'
    ),
})

health_response_model = api.model('HealthResponse', {
    'status': fields.String(description='Health status'),
    'models_loaded': fields.Raw(description='Dictionary of loaded models'),
})

model_info_model = api.model('ModelInfo', {
    'name': fields.String(description='Model name'),
    'path': fields.String(description='Model file path'),
    'exists': fields.Boolean(description='Whether model file exists'),
    'metrics': fields.Raw(description='Model metrics'),
})


# Helper functions
def load_models_on_startup():
    """Load models into cache on startup."""
    logger.info("Loading models on startup...")

    try:
        if config.MODEL_PATH.exists():
            models_cache["rf"] = load_model(config.MODEL_PATH)
            logger.info("✓ RandomForest model loaded")
        else:
            logger.warning(f"RandomForest model not found at {config.MODEL_PATH}")

        if config.GB_MODEL_PATH.exists():
            models_cache["gb"] = load_model(config.GB_MODEL_PATH)
            logger.info("✓ GradientBoosting model loaded")
        else:
            logger.warning(
                f"GradientBoosting model not found at {config.GB_MODEL_PATH}"
            )

    except Exception as e:
        logger.error(f"Error loading models: {e}")


def get_model(model_name: str):
    """Get model from cache or load it."""
    if model_name not in models_cache:
        model_path = (
            config.MODEL_PATH if model_name == "rf" else config.GB_MODEL_PATH
        )

        if not model_path.exists():
            api.abort(
                404,
                f"Model '{model_name}' not found. Train the model first."
            )

        try:
            models_cache[model_name] = load_model(model_path)
            logger.info(f"Model '{model_name}' loaded from {model_path}")
        except Exception as e:
            api.abort(500, f"Error loading model: {str(e)}")

    return models_cache[model_name]


def wine_features_dict_to_dataframe(features: dict) -> pd.DataFrame:
    """Convert wine features dictionary to pandas DataFrame."""
    data = {
        "fixed acidity": [features['fixed_acidity']],
        "volatile acidity": [features['volatile_acidity']],
        "citric acid": [features['citric_acid']],
        "residual sugar": [features['residual_sugar']],
        "chlorides": [features['chlorides']],
        "free sulfur dioxide": [features['free_sulfur_dioxide']],
        "total sulfur dioxide": [features['total_sulfur_dioxide']],
        "density": [features['density']],
        "pH": [features['pH']],
        "sulphates": [features['sulphates']],
        "alcohol": [features['alcohol']],
    }
    return pd.DataFrame(data)


# Initialize on startup
with flask_app.app_context():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    load_models_on_startup()
    logger.info("Flask-RESTX API initialized!")


# API Endpoints - General namespace
@ns_general.route('/')
class RootResource(Resource):
    """Root endpoint."""

    def get(self):
        """Welcome message."""
        return {
            'message': 'Wine Quality Prediction API',
            'version': '1.0.0',
            'docs': '/docs',
            'health': '/health',
        }


@ns_general.route('/health')
class HealthResource(Resource):
    """Health check endpoint."""

    @ns_general.marshal_with(health_response_model)
    def get(self):
        """Health check."""
        return {
            'status': 'healthy',
            'models_loaded': {
                'rf': 'rf' in models_cache,
                'gb': 'gb' in models_cache,
            },
        }


@ns_general.route('/features')
class FeaturesResource(Resource):
    """Features information endpoint."""

    def get(self):
        """Get list of required input features."""
        return {
            'features': config.FEATURE_COLUMNS,
            'count': len(config.FEATURE_COLUMNS),
            'description': '11 wine quality features',
        }


# API Endpoints - Models namespace
@ns_models.route('/')
class ModelsListResource(Resource):
    """List models endpoint."""

    @ns_models.marshal_list_with(model_info_model)
    def get(self):
        """List available models with their status."""
        models = []

        # RandomForest model
        rf_exists = config.MODEL_PATH.exists()
        models.append({
            'name': 'RandomForest',
            'path': str(config.MODEL_PATH),
            'exists': rf_exists,
            'metrics': None,
        })

        # GradientBoosting model
        gb_exists = config.GB_MODEL_PATH.exists()
        models.append({
            'name': 'GradientBoosting',
            'path': str(config.GB_MODEL_PATH),
            'exists': gb_exists,
            'metrics': None,
        })

        return models


@ns_models.route('/metrics/<string:model_name>')
class ModelMetricsResource(Resource):
    """Model metrics endpoint."""

    @ns_models.doc(params={'model_name': 'Model name (rf or gb)'})
    def get(self, model_name):
        """Get metrics for a specific model from MLflow."""
        if model_name not in ['rf', 'gb']:
            api.abort(400, "model_name must be 'rf' or 'gb'")

        try:
            # Get latest run for the model
            experiment = mlflow.get_experiment_by_name(
                config.MLFLOW_EXPERIMENT_NAME
            )
            if experiment is None:
                api.abort(404, 'MLflow experiment not found')

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.mlflow.runName LIKE '%{model_name.upper()}%'",
                order_by=["start_time DESC"],
                max_results=1,
            )

            if runs.empty:
                api.abort(
                    404,
                    f"No MLflow runs found for model '{model_name}'"
                )

            run = runs.iloc[0]
            metrics = {
                'r2_score': run.get('metrics.r2_score'),
                'mse': run.get('metrics.mse'),
                'rmse': run.get('metrics.rmse'),
                'mae': run.get('metrics.mae'),
                'run_id': run.get('run_id'),
            }

            return {
                'model_name': model_name,
                'metrics': metrics,
                'timestamp': str(run.get('start_time')),
            }

        except Exception as e:
            api.abort(500, f"Error retrieving metrics: {str(e)}")


# API Endpoints - Prediction namespace
@ns_predict.route('/')
class PredictResource(Resource):
    """Single prediction endpoint."""

    @ns_predict.expect(prediction_request_model, validate=True)
    @ns_predict.marshal_with(prediction_response_model)
    def post(self):
        """Predict wine quality for a single sample."""
        try:
            data = request.json
            model_name = data.get('model_name', 'rf')
            features = data['features']

            # Get model
            model = get_model(model_name)

            # Convert features to DataFrame
            X = wine_features_dict_to_dataframe(features)

            # Make prediction
            prediction = model.predict(X)[0]

            return {
                'quality_prediction': float(prediction),
                'model_used': model_name,
            }

        except Exception as e:
            logger.error(f"Prediction error: {traceback.format_exc()}")
            api.abort(500, f"Prediction failed: {str(e)}")


@ns_predict.route('/batch')
class BatchPredictResource(Resource):
    """Batch prediction endpoint."""

    @ns_predict.expect(batch_prediction_request_model, validate=True)
    @ns_predict.marshal_with(batch_prediction_response_model)
    def post(self):
        """Predict wine quality for multiple samples."""
        try:
            data = request.json
            model_name = data.get('model_name', 'rf')
            features_list = data['features_list']

            # Get model
            model = get_model(model_name)

            # Convert all features to DataFrame
            data_list = []
            for features in features_list:
                df = wine_features_dict_to_dataframe(features)
                data_list.append(df)

            X = pd.concat(data_list, ignore_index=True)

            # Make predictions
            predictions = model.predict(X)

            return {
                'predictions': [float(p) for p in predictions],
                'model_used': model_name,
            }

        except Exception as e:
            logger.error(f"Batch prediction error: {traceback.format_exc()}")
            api.abort(500, f"Batch prediction failed: {str(e)}")


# API Endpoints - Training namespace
@ns_train.route('/train')
class TrainModelResource(Resource):
    """Model training endpoint."""

    @ns_train.expect(training_request_model, validate=False)
    def post(self):
        """Train a new model with the specified configuration."""
        try:
            data = request.json or {}
            pipeline_type = data.get('pipeline_type', 'rf')
            data_url = data.get('data_url', config.DATASET_URL)

            logger.info(f"Training request received: {pipeline_type}")

            # Load data
            wine_data = load_wine_quality_data(url=data_url)
            X_train, X_test, y_train, y_test = split_data(wine_data)

            if pipeline_type == 'compare':
                # Train both models
                from src.train import compare_pipelines

                pipeline_rf = create_preprocessing_pipeline()
                pipeline_gb = create_alternative_pipeline()

                results = compare_pipelines(
                    pipeline_rf,
                    pipeline_gb,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                )

                # Save models
                save_model(results['rf'], config.MODEL_PATH)
                save_model(results['gb'], config.GB_MODEL_PATH)

                # Update cache
                models_cache['rf'] = results['rf']
                models_cache['gb'] = results['gb']

                # Get metrics
                rf_metrics = evaluate_model(
                    results['rf'], X_test, y_test, log_results=False
                )
                gb_metrics = evaluate_model(
                    results['gb'], X_test, y_test, log_results=False
                )

                return {
                    'message': 'Both models trained successfully',
                    'models': {
                        'rf': {
                            'path': str(config.MODEL_PATH),
                            'metrics': rf_metrics,
                        },
                        'gb': {
                            'path': str(config.GB_MODEL_PATH),
                            'metrics': gb_metrics,
                        },
                    },
                }

            else:
                # Train single model
                model_type = (
                    'RandomForest' if pipeline_type == 'rf'
                    else 'GradientBoosting'
                )
                model_path = (
                    config.MODEL_PATH if pipeline_type == 'rf'
                    else config.GB_MODEL_PATH
                )
                param_grid = (
                    config.HYPERPARAM_GRID if pipeline_type == 'rf'
                    else config.GB_HYPERPARAM_GRID
                )

                # Start MLflow run
                with mlflow.start_run(run_name=f"{model_type}_API_Training"):
                    # Create pipeline
                    if pipeline_type == 'rf':
                        pipeline = create_preprocessing_pipeline()
                    else:
                        pipeline = create_alternative_pipeline()

                    # Train model
                    trained_model = train_model_with_grid_search(
                        pipeline, X_train, y_train, param_grid, cv=5
                    )

                    # Evaluate
                    metrics = evaluate_model(
                        trained_model, X_test, y_test, log_results=False
                    )

                    # Log to MLflow
                    from src.train import log_model_to_mlflow
                    log_model_to_mlflow(trained_model, metrics)

                    # Save model
                    save_model(trained_model, model_path)

                    # Update cache
                    models_cache[pipeline_type] = trained_model

                    return {
                        'message': f'{model_type} model trained successfully',
                        'model': {
                            'name': model_type,
                            'path': str(model_path),
                            'metrics': metrics,
                        },
                    }

        except Exception as e:
            logger.error(f"Training error: {traceback.format_exc()}")
            api.abort(500, f"Training failed: {str(e)}")


@ns_train.route('/predict')
class ModelPredictResource(Resource):
    """Alias for prediction endpoint."""

    @ns_train.expect(prediction_request_model, validate=True)
    @ns_train.marshal_with(prediction_response_model)
    def post(self):
        """Predict wine quality (alias endpoint)."""
        try:
            data = request.json
            model_name = data.get('model_name', 'rf')
            features = data['features']

            # Get model
            model = get_model(model_name)

            # Convert features to DataFrame
            X = wine_features_dict_to_dataframe(features)

            # Make prediction
            prediction = model.predict(X)[0]

            return {
                'quality_prediction': float(prediction),
                'model_used': model_name,
            }

        except Exception as e:
            logger.error(f"Prediction error: {traceback.format_exc()}")
            api.abort(500, f"Prediction failed: {str(e)}")


if __name__ == '__main__':
    flask_app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.API_RELOAD,
    )
