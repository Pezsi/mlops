"""FastAPI application for Wine Quality prediction.

This module provides REST API endpoints for:
- Model training with MLflow tracking
- Wine quality prediction
- Model management and metrics
- Health checks

Follows PEP8 conventions with type hints.
"""

import logging
from typing import Dict, List, Optional, Any
import traceback

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

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

# Initialize FastAPI app
app = FastAPI(
    title="Wine Quality Prediction API",
    description="MLOps API for wine quality prediction with MLflow integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global model cache
models_cache: Dict[str, Any] = {}


# Pydantic models for request/response validation
class WineFeatures(BaseModel):
    """Wine features for prediction."""

    fixed_acidity: float = Field(
        ..., ge=0, le=20, description="Fixed acidity (g/dm³)"
    )
    volatile_acidity: float = Field(
        ..., ge=0, le=2, description="Volatile acidity (g/dm³)"
    )
    citric_acid: float = Field(
        ..., ge=0, le=2, description="Citric acid (g/dm³)"
    )
    residual_sugar: float = Field(
        ..., ge=0, le=20, description="Residual sugar (g/dm³)"
    )
    chlorides: float = Field(
        ..., ge=0, le=1, description="Chlorides (g/dm³)"
    )
    free_sulfur_dioxide: float = Field(
        ..., ge=0, le=100, description="Free sulfur dioxide (mg/dm³)"
    )
    total_sulfur_dioxide: float = Field(
        ..., ge=0, le=400, description="Total sulfur dioxide (mg/dm³)"
    )
    density: float = Field(
        ..., ge=0.98, le=1.01, description="Density (g/cm³)"
    )
    pH: float = Field(
        ..., ge=2.5, le=4.5, description="pH value"
    )
    sulphates: float = Field(
        ..., ge=0, le=2.5, description="Sulphates (g/dm³)"
    )
    alcohol: float = Field(
        ..., ge=8, le=15, description="Alcohol (% vol.)"
    )

    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionRequest(BaseModel):
    """Prediction request with wine features."""

    features: WineFeatures
    model_name: Optional[str] = Field(
        "rf", description="Model to use: 'rf' or 'gb'"
    )

    @validator("model_name")
    def validate_model_name(cls, v):
        """Validate model name."""
        if v not in ["rf", "gb"]:
            raise ValueError("model_name must be 'rf' or 'gb'")
        return v


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    features_list: List[WineFeatures]
    model_name: Optional[str] = Field(
        "rf", description="Model to use: 'rf' or 'gb'"
    )


class PredictionResponse(BaseModel):
    """Prediction response."""

    quality_prediction: float = Field(..., description="Predicted wine quality")
    model_used: str = Field(..., description="Model used for prediction")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: List[float]
    model_used: str


class TrainingRequest(BaseModel):
    """Model training request."""

    pipeline_type: str = Field(
        "rf", description="Pipeline type: 'rf', 'gb', or 'compare'"
    )
    data_url: Optional[str] = Field(
        None, description="URL to dataset (default: UCI dataset)"
    )

    @validator("pipeline_type")
    def validate_pipeline_type(cls, v):
        """Validate pipeline type."""
        if v not in ["rf", "gb", "compare"]:
            raise ValueError("pipeline_type must be 'rf', 'gb', or 'compare'")
        return v


class ModelInfo(BaseModel):
    """Model information."""

    name: str
    path: str
    exists: bool
    metrics: Optional[Dict[str, float]] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    models_loaded: Dict[str, bool]


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
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Train the model first.",
            )

        try:
            models_cache[model_name] = load_model(model_path)
            logger.info(f"Model '{model_name}' loaded from {model_path}")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error loading model: {str(e)}"
            )

    return models_cache[model_name]


def wine_features_to_dataframe(features: WineFeatures) -> pd.DataFrame:
    """Convert WineFeatures to pandas DataFrame."""
    data = {
        "fixed acidity": [features.fixed_acidity],
        "volatile acidity": [features.volatile_acidity],
        "citric acid": [features.citric_acid],
        "residual sugar": [features.residual_sugar],
        "chlorides": [features.chlorides],
        "free sulfur dioxide": [features.free_sulfur_dioxide],
        "total sulfur dioxide": [features.total_sulfur_dioxide],
        "density": [features.density],
        "pH": [features.pH],
        "sulphates": [features.sulphates],
        "alcohol": [features.alcohol],
    }
    return pd.DataFrame(data)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("Starting Wine Quality Prediction API...")
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    load_models_on_startup()
    logger.info("API startup completed!")


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint."""
    return {
        "message": "Wine Quality Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "rf": "rf" in models_cache,
            "gb": "gb" in models_cache,
        },
    )


@app.get("/features", tags=["General"])
async def get_features():
    """Get list of required input features."""
    return {
        "features": config.FEATURE_COLUMNS,
        "count": len(config.FEATURE_COLUMNS),
        "description": "11 wine quality features",
    }


@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models():
    """List available models with their status."""
    models = []

    # RandomForest model
    rf_exists = config.MODEL_PATH.exists()
    rf_info = ModelInfo(
        name="RandomForest",
        path=str(config.MODEL_PATH),
        exists=rf_exists,
        metrics=None,
    )
    models.append(rf_info)

    # GradientBoosting model
    gb_exists = config.GB_MODEL_PATH.exists()
    gb_info = ModelInfo(
        name="GradientBoosting",
        path=str(config.GB_MODEL_PATH),
        exists=gb_exists,
        metrics=None,
    )
    models.append(gb_info)

    return models


@app.get("/metrics/{model_name}", tags=["Models"])
async def get_model_metrics(model_name: str):
    """Get metrics for a specific model from MLflow."""
    if model_name not in ["rf", "gb"]:
        raise HTTPException(
            status_code=400,
            detail="model_name must be 'rf' or 'gb'"
        )
    try:
        # Get latest run for the model
        experiment = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            raise HTTPException(
                status_code=404, detail="MLflow experiment not found"
            )

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName LIKE '%{model_name.upper()}%'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if runs.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No MLflow runs found for model '{model_name}'",
            )

        run = runs.iloc[0]
        metrics = {
            "r2_score": run.get("metrics.r2_score"),
            "mse": run.get("metrics.mse"),
            "rmse": run.get("metrics.rmse"),
            "mae": run.get("metrics.mae"),
            "run_id": run.get("run_id"),
        }

        return {
            "model_name": model_name,
            "metrics": metrics,
            "timestamp": run.get("start_time"),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving metrics: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """Predict wine quality for a single sample."""
    try:
        # Get model
        model = get_model(request.model_name)

        # Convert features to DataFrame
        X = wine_features_to_dataframe(request.features)

        # Make prediction
        prediction = model.predict(X)[0]

        return PredictionResponse(
            quality_prediction=float(prediction), model_used=request.model_name
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
)
async def predict_batch(request: BatchPredictionRequest):
    """Predict wine quality for multiple samples."""
    try:
        # Get model
        model = get_model(request.model_name)

        # Validate non-empty list
        if not request.features_list:
            raise HTTPException(status_code=422, detail="features_list cannot be empty")

        # Convert all features to DataFrame
        data_list = []
        for features in request.features_list:
            df = wine_features_to_dataframe(features)
            data_list.append(df)

        X = pd.concat(data_list, ignore_index=True)

        # Make predictions
        predictions = model.predict(X)

        return BatchPredictionResponse(
            predictions=[float(p) for p in predictions],
            model_used=request.model_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/model/train", tags=["Training"])
async def train_model_endpoint(request: TrainingRequest):
    """Train a new model with the specified configuration."""
    try:
        logger.info(f"Training request received: {request.pipeline_type}")

        # Load data
        data_url = request.data_url or config.DATASET_URL
        data = load_wine_quality_data(url=data_url)
        X_train, X_test, y_train, y_test = split_data(data)

        if request.pipeline_type == "compare":
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
            save_model(results["rf"], config.MODEL_PATH)
            save_model(results["gb"], config.GB_MODEL_PATH)

            # Update cache
            models_cache["rf"] = results["rf"]
            models_cache["gb"] = results["gb"]

            # Get metrics
            rf_metrics = evaluate_model(results["rf"], X_test, y_test, log_results=False)
            gb_metrics = evaluate_model(results["gb"], X_test, y_test, log_results=False)

            return {
                "message": "Both models trained successfully",
                "models": {
                    "rf": {
                        "path": str(config.MODEL_PATH),
                        "metrics": rf_metrics,
                    },
                    "gb": {
                        "path": str(config.GB_MODEL_PATH),
                        "metrics": gb_metrics,
                    },
                },
            }

        else:
            # Train single model
            model_type = "RandomForest" if request.pipeline_type == "rf" else "GradientBoosting"
            model_path = config.MODEL_PATH if request.pipeline_type == "rf" else config.GB_MODEL_PATH
            param_grid = config.HYPERPARAM_GRID if request.pipeline_type == "rf" else config.GB_HYPERPARAM_GRID

            # Start MLflow run
            with mlflow.start_run(run_name=f"{model_type}_API_Training"):
                # Create pipeline
                if request.pipeline_type == "rf":
                    pipeline = create_preprocessing_pipeline()
                else:
                    pipeline = create_alternative_pipeline()

                # Train model
                trained_model = train_model_with_grid_search(
                    pipeline, X_train, y_train, param_grid, cv=5
                )

                # Evaluate
                metrics = evaluate_model(trained_model, X_test, y_test, log_results=False)

                # Log to MLflow
                from src.train import log_model_to_mlflow
                log_model_to_mlflow(trained_model, metrics)

                # Save model
                save_model(trained_model, model_path)

                # Update cache
                models_cache[request.pipeline_type] = trained_model

                return {
                    "message": f"{model_type} model trained successfully",
                    "model": {
                        "name": model_type,
                        "path": str(model_path),
                        "metrics": metrics,
                    },
                }

    except Exception as e:
        logger.error(f"Training error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Training failed: {str(e)}"
        )


# Alias endpoint for compatibility
@app.post("/model/predict", response_model=PredictionResponse, tags=["Prediction"])
async def model_predict(request: PredictionRequest):
    """Alias for /predict endpoint."""
    return await predict(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "fastapi_app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_RELOAD,
    )
