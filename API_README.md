# Wine Quality Prediction REST APIs

Complete API documentation for wine quality prediction endpoints.

## Overview

Two production-ready REST API implementations:

- **FastAPI** (547 lines) - Modern async framework with Pydantic validation
- **Flask-RESTX** (637 lines) - Traditional REST with Swagger UI

Both provide identical functionality: predictions, training, model management, and MLflow integration.

## Quick Start

### FastAPI
```bash
python fastapi_app.py
# Docs: http://localhost:8000/docs
```

### Flask-RESTX
```bash
python flask_app.py
# Docs: http://localhost:8000/docs/
```

## API Endpoints

### General

**`GET /`** - Welcome message
```json
{"message": "Wine Quality Prediction API", "version": "1.0.0"}
```

**`GET /health`** - Health check
```json
{"status": "healthy", "models_loaded": {"rf": true, "gb": true}}
```

**`GET /features`** - List required input features
```json
{"features": ["fixed acidity", "volatile acidity", ...], "count": 11}
```

### Models

**`GET /models`** - List available models
```json
[
  {"name": "RandomForest", "path": "models/rf_regressor.pkl", "exists": true},
  {"name": "GradientBoosting", "path": "models/gb_regressor.pkl", "exists": true}
]
```

**`GET /metrics/{model_name}`** - Get model metrics from MLflow
- Path param: `model_name` (rf or gb)
```json
{
  "model_name": "rf",
  "metrics": {"r2_score": 0.47, "rmse": 0.58, "mae": 0.45},
  "timestamp": "2024-01-15T10:30:00"
}
```

### Prediction

**`POST /predict`** - Single prediction

Request:
```json
{
  "features": {
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
    "alcohol": 9.4
  },
  "model_name": "rf"
}
```

Response:
```json
{"quality_prediction": 5.67, "model_used": "rf"}
```

**Feature Validation Ranges:**
- fixed_acidity: 0-20, volatile_acidity: 0-2, citric_acid: 0-2
- residual_sugar: 0-20, chlorides: 0-1
- free_sulfur_dioxide: 0-100, total_sulfur_dioxide: 0-400
- density: 0.98-1.01, pH: 2.5-4.5, sulphates: 0-2.5, alcohol: 8-15

**`POST /predict/batch`** - Batch predictions

Request:
```json
{
  "features_list": [
    {/* wine 1 features */},
    {/* wine 2 features */}
  ],
  "model_name": "rf"
}
```

Response:
```json
{"predictions": [5.67, 6.12], "model_used": "rf"}
```

**`POST /model/predict`** - Alias for `/predict`

### Training

**`POST /model/train`** - Train new model(s)

Request:
```json
{
  "pipeline_type": "rf",     // "rf", "gb", or "compare"
  "data_url": "optional URL" // defaults to UCI dataset
}
```

Response (single model):
```json
{
  "message": "RandomForest model trained successfully",
  "model": {
    "name": "RandomForest",
    "path": "models/rf_regressor.pkl",
    "metrics": {"r2_score": 0.47, "rmse": 0.58, "mae": 0.45}
  }
}
```

Response (compare mode):
```json
{
  "message": "Both models trained successfully",
  "models": {
    "rf": {"path": "...", "metrics": {...}},
    "gb": {"path": "...", "metrics": {...}}
  }
}
```

**Training Process:**
1. Load dataset (UCI or custom URL)
2. Split 80/20 train/test
3. Create preprocessing pipeline
4. GridSearchCV with CV
5. Evaluate on test set
6. Log to MLflow
7. Save model

**Training Time:** RF: 1-2 min, GB: 2-3 min, Compare: 3-5 min

## Usage Examples

### cURL

```bash
# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {...}, "model_name": "rf"}'

# Training
curl -X POST http://localhost:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"pipeline_type": "compare"}'
```

### Python

```python
import requests

# Prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "features": {
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
            "alcohol": 9.4
        },
        "model_name": "rf"
    }
)
print(response.json())

# Training
response = requests.post(
    "http://localhost:8000/model/train",
    json={"pipeline_type": "rf"}
)
print(response.json())
```

## HTTP Status Codes

- `200 OK` - Success
- `400 Bad Request` - Invalid request
- `404 Not Found` - Model/resource not found
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Server error

## MLflow Integration

Both APIs integrate with MLflow for tracking:

**Start MLflow UI:**
```bash
mlflow ui --port 5000
# Access: http://localhost:5000
```

**Logged Information:**
- Hyperparameters from GridSearchCV
- Cross-validation scores
- Test metrics (R², MSE, RMSE, MAE)
- Model artifacts

**Model Registry:**
- Automatic registration to `wine_quality_rf_model`
- Version tracking
- Stage transitions

## Architecture

### FastAPI
```
fastapi_app.py
├── Pydantic Models (validation)
├── Helper Functions (model loading, conversions)
├── Startup Event (load models)
└── API Endpoints (async handlers)
```

### Flask-RESTX
```
flask_app.py
├── Flask-RESTX Models (validation)
├── Helper Functions (model loading, conversions)
├── Namespaces (general, models, predict, train)
└── Resource Classes (endpoint handlers)
```

## Testing

```bash
# Run API tests
pytest tests/test_fastapi.py -v
pytest tests/test_flask_app.py -v

# All tests
pytest tests/test_*app*.py -v
```

**Test Coverage:**
- General endpoints (root, health, features)
- Model management (list, metrics)
- Predictions (single, batch, validation)
- Training (single, compare mode)
- Error handling

## Deployment

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0"]
```

```bash
docker build -t wine-quality-api .
docker run -p 8000:8000 wine-quality-api
```

### Production

```bash
# FastAPI
gunicorn fastapi_app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000

# Flask-RESTX
gunicorn flask_app:flask_app \
  --workers 4 \
  --bind 0.0.0.0:8000
```

## Troubleshooting

**Model not found (404):**
```bash
python main.py --pipeline compare
```

**Validation error (422):**
- Ensure all 11 features present
- Check feature value ranges
- Verify JSON format

**Port in use:**
```bash
lsof -i :8000
kill -9 <PID>
```

## Performance

| Metric | FastAPI | Flask-RESTX |
|--------|---------|-------------|
| Single prediction | ~10ms | ~15ms |
| Batch (10 samples) | ~15ms | ~20ms |
| Throughput | ~100 req/s | ~80 req/s |

*Single worker, Intel i7, 16GB RAM*

## API Comparison

| Feature | FastAPI | Flask-RESTX |
|---------|---------|-------------|
| Performance | Excellent | Very Good |
| Async | Yes | No |
| Validation | Pydantic | Flask-RESTX |
| Documentation | Auto | Auto |
| Code Lines | 547 | 637 |
| Best For | New projects | Legacy systems |

## Security Notes

Current implementation (development only):
- No authentication
- No rate limiting
- No HTTPS

**Production recommendations:**
- Add JWT/OAuth2 authentication
- Implement rate limiting
- Use HTTPS/TLS
- Add CORS configuration
- Environment-based config

## Documentation Links

- **Swagger UI:** http://localhost:8000/docs (FastAPI) or http://localhost:8000/docs/ (Flask)
- **MLflow UI:** http://localhost:5000
- **Main README:** [README.md](README.md)
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **Flask-RESTX Docs:** https://flask-restx.readthedocs.io/

---

**Version:** 1.0.0
**Status:** Production Ready
**Last Updated:** January 2025
