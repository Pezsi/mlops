# Wine Quality MLOps - Teljes API Dokumentáció

Komplett API dokumentáció a Wine Quality MLOps projekt összes REST API szolgáltatásához.

## Áttekintés

A projekt **3 különböző REST API szolgáltatást** tartalmaz, amelyek együtt alkotják a teljes MLOps ökoszisztémát:

### 1. Prediction APIs (Model Serving)
- **FastAPI** (547 lines) - Modern async framework, Pydantic validáció, high performance
- **Flask-RESTX** (637 lines) - Traditional REST, Swagger UI, mature ecosystem

**Funkciók:**
- Wine quality predictions (single & batch)
- Model training trigger
- Model management és metrics
- Health checks és monitoring
- MLflow integration

**Portok:** 8000 (FastAPI), 8000 (Flask - egyikét választva)

### 2. Webhook Trigger API (Airflow Orchestration)
- **Flask-based REST API** - Airflow DAG triggering
- **Port:** 8080

**Funkciók:**
- DAG trigger external rendszerekből (CI/CD, webhooks, stb.)
- Training trigger custom paraméterekkel
- Evaluation trigger
- DAG status check
- API key authentication

### 3. MLflow Tracking Server API
- **MLflow beépített REST API**
- **Port:** 5000

**Funkciók:**
- Experiment tracking
- Model registry operations
- Metrics és artifacts logging
- Model versioning

## API Összehasonlítás

| Feature | FastAPI | Flask-RESTX | Webhook API | MLflow API |
|---------|---------|-------------|-------------|------------|
| **Port** | 8000 | 8000 | 8080 | 5000 |
| **Purpose** | Predictions | Predictions | DAG Trigger | Tracking |
| **Performance** | Excellent | Very Good | Good | Good |
| **Async** | Yes | No | No | No |
| **Auth** | None (dev) | None (dev) | API Key | None (dev) |
| **Swagger UI** | /docs | /docs/ | /docs | Built-in |
| **Best For** | New projects | Legacy | Orchestration | ML Tracking |

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

---

## 2. Webhook Trigger API Részletes Dokumentáció

### Áttekintés

A Webhook Trigger API egy Flask-based REST service, amely lehetővé teszi Airflow DAG-ok külső rendszerekből történő triggerelését.

**Base URL:** `http://localhost:8080`

**Authentication:** API Key (Header: `X-API-Key`)

**Default API Key:** `mlops-secret-key-2025` (változtatható .env fájlban)

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00Z",
  "airflow_connection": "ok"
}
```

#### 2. Trigger Training DAG
```http
POST /trigger/training
Content-Type: application/json
X-API-Key: mlops-secret-key-2025
```

**Request Body:**
```json
{
  "model_name": "wine_quality_rf_model",
  "trigger_source": "ci_cd_pipeline",
  "dataset_version": "v2.1.0",
  "hyperparameters": {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 2
  },
  "force_training": true,
  "deployment_target": "staging",
  "callback_url": "https://your-system.com/callback"
}
```

**Request Parameters:**
- `model_name` (string, required) - Model neve
- `trigger_source` (string, required) - Trigger forrás (ci_cd, manual, scheduled, stb.)
- `dataset_version` (string, optional) - Dataset verzió
- `hyperparameters` (object, optional) - Custom hyperparaméterek
- `force_training` (boolean, optional) - Training kényszerítése (default: false)
- `deployment_target` (string, optional) - Target environment (staging/production)
- `callback_url` (string, optional) - Callback URL eredményekhez

**Response:**
```json
{
  "status": "triggered",
  "dag_id": "webhook_triggered_training",
  "run_id": "manual__2025-01-15T10:30:00+00:00",
  "execution_date": "2025-01-15T10:30:00+00:00",
  "message": "DAG triggered successfully",
  "config": {
    "model_name": "wine_quality_rf_model",
    "trigger_source": "ci_cd_pipeline"
  }
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8080/trigger/training \
  -H "Content-Type: application/json" \
  -H "X-API-Key: mlops-secret-key-2025" \
  -d '{
    "model_name": "wine_quality_rf_model",
    "trigger_source": "ci_cd",
    "force_training": true,
    "hyperparameters": {
      "n_estimators": 200,
      "max_depth": 10
    }
  }'
```

#### 3. Trigger Evaluation DAG
```http
POST /trigger/evaluation
Content-Type: application/json
X-API-Key: mlops-secret-key-2025
```

**Request Body:**
```json
{
  "model_run_id": "run_20250115_103000",
  "evaluation_dataset": "holdout_v1",
  "comparison_baseline": "production_model_v1"
}
```

**Response:**
```json
{
  "status": "triggered",
  "dag_id": "model_evaluation_pipeline",
  "run_id": "manual__2025-01-15T10:31:00+00:00"
}
```

#### 4. Trigger Dataset Check
```http
POST /trigger/dataset-check
Content-Type: application/json
X-API-Key: mlops-secret-key-2025
```

**Request Body:**
```json
{
  "dataset_path": "/opt/airflow/data/incoming/new_data.csv",
  "reference_dataset": "production_v1",
  "drift_threshold": 0.2
}
```

#### 5. Check DAG Status
```http
GET /status/<dag_id>/<run_id>
X-API-Key: mlops-secret-key-2025
```

**Example:**
```bash
curl http://localhost:8080/status/webhook_triggered_training/manual__2025-01-15T10:30:00+00:00 \
  -H "X-API-Key: mlops-secret-key-2025"
```

**Response:**
```json
{
  "dag_id": "webhook_triggered_training",
  "run_id": "manual__2025-01-15T10:30:00+00:00",
  "state": "success",
  "start_date": "2025-01-15T10:30:00+00:00",
  "end_date": "2025-01-15T10:45:00+00:00",
  "execution_date": "2025-01-15T10:30:00+00:00"
}
```

**States:**
- `queued` - Várakozik végrehajtásra
- `running` - Futás alatt
- `success` - Sikeresen befejeződött
- `failed` - Hiba történt
- `skipped` - Kihagyva

#### 6. List Available DAGs
```http
GET /dags
X-API-Key: mlops-secret-key-2025
```

**Response:**
```json
{
  "dags": [
    {
      "dag_id": "daily_model_training_with_notification",
      "is_paused": false,
      "description": "Daily scheduled training with notifications"
    },
    {
      "dag_id": "webhook_triggered_training",
      "is_paused": false,
      "description": "Webhook-triggered training with custom config"
    }
  ],
  "count": 5
}
```

#### 7. API Documentation
```http
GET /docs
```

Swagger UI interaktív API dokumentáció.

### Error Responses

#### 401 Unauthorized
```json
{
  "error": "Unauthorized",
  "message": "Invalid or missing API key"
}
```

#### 400 Bad Request
```json
{
  "error": "Bad Request",
  "message": "Missing required field: model_name"
}
```

#### 500 Internal Server Error
```json
{
  "error": "Internal Server Error",
  "message": "Failed to trigger DAG: connection error"
}
```

### Authentication

**API Key Setup:**

1. Állítsd be az API kulcsot a `.env` fájlban:
```bash
WEBHOOK_API_KEY=your-secret-key-here
```

2. Használd a kulcsot minden kérésben:
```bash
curl -H "X-API-Key: your-secret-key-here" ...
```

### Use Cases

#### 1. CI/CD Integration
```yaml
# GitHub Actions example
- name: Trigger model training
  run: |
    curl -X POST http://mlops-server:8080/trigger/training \
      -H "Content-Type: application/json" \
      -H "X-API-Key: ${{ secrets.MLOPS_API_KEY }}" \
      -d '{"model_name": "wine_quality_rf_model", "trigger_source": "github_actions"}'
```

#### 2. Scheduled External Trigger
```python
import requests
import schedule

def trigger_training():
    response = requests.post(
        "http://localhost:8080/trigger/training",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": "mlops-secret-key-2025"
        },
        json={
            "model_name": "wine_quality_rf_model",
            "trigger_source": "scheduled_script"
        }
    )
    print(f"Status: {response.status_code}, Response: {response.json()}")

schedule.every().day.at("02:00").do(trigger_training)
```

#### 3. Event-Driven Training
```python
# Webhook endpoint a saját alkalmazásodban
@app.route('/new-data-webhook', methods=['POST'])
def handle_new_data():
    data = request.json

    # Trigger training amikor új adat érkezik
    response = requests.post(
        "http://mlops-server:8080/trigger/training",
        headers={"X-API-Key": "mlops-secret-key-2025"},
        json={
            "model_name": "wine_quality_rf_model",
            "trigger_source": "new_data_event",
            "dataset_version": data.get("version"),
            "force_training": True
        }
    )

    return jsonify({"training_triggered": response.json()})
```

---

## 3. MLflow API Referencia

### Base URL
```
http://localhost:5000
```

### Gyakori Endpoints

#### Create Experiment
```http
POST /api/2.0/mlflow/experiments/create
Content-Type: application/json
```

**Body:**
```json
{
  "name": "wine_quality_experiment",
  "artifact_location": "mlruns"
}
```

#### Log Metrics
```http
POST /api/2.0/mlflow/runs/log-metric
Content-Type: application/json
```

**Body:**
```json
{
  "run_id": "abc123",
  "key": "r2_score",
  "value": 0.87,
  "timestamp": 1642252800000
}
```

#### Search Runs
```http
POST /api/2.0/mlflow/runs/search
Content-Type: application/json
```

**Body:**
```json
{
  "experiment_ids": ["1"],
  "filter": "metrics.r2_score > 0.8",
  "order_by": ["metrics.r2_score DESC"],
  "max_results": 10
}
```

### Python SDK
```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Start run
with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("r2_score", 0.87)
    mlflow.sklearn.log_model(model, "model")
```

---

## API Integration Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    External Systems                          │
│  (CI/CD, Monitoring, Data Pipelines, User Apps)            │
└────────────┬──────────────┬──────────────┬─────────────────┘
             │              │              │
             ▼              ▼              ▼
     ┌───────────┐  ┌──────────┐  ┌──────────────┐
     │ Webhook   │  │ FastAPI/ │  │   MLflow     │
     │    API    │  │  Flask   │  │     API      │
     │  (8080)   │  │  (8000)  │  │   (5000)     │
     └─────┬─────┘  └─────┬────┘  └──────┬───────┘
           │              │               │
           ▼              │               ▼
     ┌─────────┐          │         ┌──────────┐
     │ Airflow │          │         │  MLflow  │
     │   DAGs  │          │         │ Tracking │
     └─────┬───┘          │         └─────┬────┘
           │              │               │
           ▼              ▼               ▼
     ┌─────────────────────────────────────┐
     │        Models + Data + Metrics       │
     │     (Shared Volumes & Databases)     │
     └─────────────────────────────────────┘
```

## Documentation Links

- **FastAPI Swagger:** http://localhost:8000/docs
- **Flask-RESTX Swagger:** http://localhost:8000/docs/
- **Webhook API Swagger:** http://localhost:8080/docs
- **MLflow UI:** http://localhost:5000
- **Airflow UI:** http://localhost:8081
- **Main README:** [README.md](README.md)
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **Flask-RESTX Docs:** https://flask-restx.readthedocs.io/
- **MLflow Docs:** https://mlflow.org/docs/latest/rest-api.html
- **Airflow API Docs:** https://airflow.apache.org/docs/apache-airflow/stable/stable-rest-api-ref.html

---

**Version:** 2.0.0
**Status:** Production Ready
**Last Updated:** November 2025
**Services:** 3 REST APIs (Prediction, Webhook, MLflow)
