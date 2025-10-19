# Wine Quality MLOps Project

Production-ready machine learning application for wine quality prediction with complete MLOps infrastructure.

## Overview

Full ML pipeline for predicting wine quality using Random Forest and Gradient Boosting models, with MLflow tracking, REST APIs, and comprehensive testing.

### Key Features

- **Two ML Models** - Random Forest & Gradient Boosting
- **MLflow Integration** - Experiment tracking, model registry
- **Two REST APIs** - FastAPI (async) & Flask-RESTX
- **Docker Containerization** - Multi-container orchestration with Docker Compose
- **Apache Airflow** - Automated training pipelines and model deployment
- **Streamlit Dashboard** - Real-time monitoring and visualization
- **Evidently Integration** - Data drift and model drift detection
- **91+ Tests** - Unit, integration, and API tests
- **PEP8 Compliant** - 0 linting errors
- **Production Ready** - Full MLSecOps infrastructure

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python main.py --pipeline compare

# Start API
python fastapi_app.py        # or python flask_app.py

# View experiments
mlflow ui --port 5000

# Run tests
pytest tests/ -v
```

### Docker Deployment (Recommended)

```bash
# Start complete MLOps platform
docker-compose up -d

# Access services:
# - Airflow UI: http://localhost:8081 (admin/admin)
# - MLflow UI: http://localhost:5000
# - FastAPI: http://localhost:8000/docs
# - Monitoring: http://localhost:8501

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

For detailed deployment instructions, see [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)

## Project Structure

```
wine_quality_mlops/
├── config.py                  # Configuration & hyperparameters
├── main.py                    # Pipeline orchestrator
├── fastapi_app.py             # FastAPI REST API (547 lines)
├── flask_app.py               # Flask-RESTX REST API (637 lines)
│
├── data/
│   └── load_data.py           # Data loading & splitting
│
├── src/
│   ├── preprocessing.py       # Feature preprocessing
│   ├── train.py               # Model training
│   └── evaluate.py            # Model evaluation
│
├── tests/                     # 91+ tests
│   ├── test_load_data.py      # 17 tests
│   ├── test_preprocessing.py  # 26 tests
│   ├── test_train.py          # 23 tests
│   ├── test_evaluate.py       # 25 tests
│   ├── test_integration.py    # 8 tests
│   ├── test_fastapi.py        # 15+ tests
│   └── test_flask_app.py      # 15+ tests
│
├── airflow/                   # Airflow orchestration
│   ├── dags/                  # DAG definitions
│   │   └── train_wine_quality_dag.py  # Automated training pipeline
│   ├── logs/                  # Airflow logs
│   └── Dockerfile.airflow     # Airflow container
│
├── monitoring/                # Monitoring dashboard
│   └── app.py                 # Streamlit monitoring app
│
├── models/                    # Trained models
├── mlruns/                    # MLflow experiments
├── logs/                      # Training logs
│
├── Dockerfile                 # Main application container
├── Dockerfile.streamlit       # Monitoring container
├── docker-compose.yml         # Multi-service orchestration
├── environment.yml            # Conda environment
└── DOCKER_DEPLOYMENT.md       # Deployment guide
```

## ML Pipeline

### Training

```bash
# Compare both models
python main.py --pipeline compare

# Train specific model
python main.py --pipeline rf    # Random Forest
python main.py --pipeline gb    # Gradient Boosting
```

### Pipeline Steps

1. **Data Loading** - UCI Wine Quality dataset
2. **Splitting** - 80/20 train/test (stratified)
3. **Preprocessing** - StandardScaler, optional PCA
4. **Training** - GridSearchCV with 10-fold CV
5. **Evaluation** - R², MSE, RMSE, MAE
6. **Logging** - MLflow tracking
7. **Persistence** - Model saving

### Model Performance

| Model | R² Score | RMSE | Training Time |
|-------|----------|------|---------------|
| Random Forest | ~0.47 | ~0.58 | 1-2 min |
| Gradient Boosting | ~0.49 | ~0.57 | 2-3 min |

## MLflow

### Start MLflow UI

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### Tracked Information

- Hyperparameters (GridSearchCV)
- Cross-validation scores
- Test metrics (R², MSE, RMSE, MAE)
- Model artifacts
- Dataset info

## REST APIs

### FastAPI (Modern, Async)

```bash
python fastapi_app.py
# Docs: http://localhost:8000/docs
```

**Features:** Async support, Pydantic validation, high performance

### Flask-RESTX (Traditional)

```bash
python flask_app.py
# Docs: http://localhost:8000/docs/
```

**Features:** Namespace organization, mature ecosystem

### API Endpoints

```
GET  /                     # Welcome
GET  /health               # Health check
GET  /features             # Feature list
GET  /models               # Available models
GET  /metrics/{model}      # Model metrics

POST /predict              # Single prediction
POST /predict/batch        # Batch predictions
POST /model/train          # Train model
POST /model/predict        # Alias endpoint
```

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Response:**
```json
{
  "quality_prediction": 5.67,
  "model_used": "rf"
}
```

## Testing

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific modules
pytest tests/test_load_data.py -v
pytest tests/test_integration.py -v

# API tests
pytest tests/test_fastapi.py -v
pytest tests/test_flask_app.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Test Coverage

- **91+ tests total**
- Unit tests for all modules
- Integration tests for pipeline
- API endpoint tests
- 100% PEP8 compliant

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# Clone repository
cd wine_quality_mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train models
python main.py --pipeline compare

# Verify
pytest tests/ -v
```

## Configuration

Edit `config.py` to customize:

```python
# Data
TEST_SIZE = 0.2
RANDOM_STATE = 123

# Models
MODEL_PATH = Path("models/rf_regressor.pkl")
GB_MODEL_PATH = Path("models/gb_regressor.pkl")

# Hyperparameters
HYPERPARAM_GRID = {
    "randomforestregressor__max_features": ["sqrt", "log2"],
    "randomforestregressor__max_depth": [None, 5, 3, 1],
}

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
```

## Usage Examples

### Python API

```python
from data.load_data import load_and_split_data
from src.preprocessing import create_preprocessing_pipeline
from src.train import train_model_with_grid_search
from src.evaluate import evaluate_model

# Load data
X_train, X_test, y_train, y_test = load_and_split_data()

# Train
pipeline = create_preprocessing_pipeline()
model = train_model_with_grid_search(pipeline, X_train, y_train)

# Evaluate
metrics = evaluate_model(model, X_test, y_test)
print(f"R²: {metrics['r2_score']:.4f}")
```

### REST API

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "features": {...},
        "model_name": "rf"
    }
)
print(response.json())
```

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
# FastAPI with Gunicorn
gunicorn fastapi_app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000

# Flask with Gunicorn
gunicorn flask_app:flask_app \
  --workers 4 \
  --bind 0.0.0.0:8000
```

## Code Quality

```bash
# PEP8 check
flake8 . --exclude=venv,mlruns
# Result: 0 errors ✅

# Format code
black . --exclude=venv

# Type checking
mypy . --exclude=venv
```

## Project Metrics

- **Total Code:** ~3,500 lines
- **Test Code:** ~1,300 lines
- **API Code:** ~1,800 lines
- **Tests:** 91+
- **PEP8 Errors:** 0

## Technology Stack

**ML/Data:**
- scikit-learn, pandas, numpy

**MLOps:**
- MLflow - Experiment tracking & model registry
- Apache Airflow - Workflow orchestration
- Evidently - Data & model drift detection

**APIs:**
- FastAPI - Async REST API
- Flask-RESTX - Traditional REST API

**Containerization:**
- Docker - Application containerization
- Docker Compose - Multi-service orchestration

**Monitoring:**
- Streamlit - Interactive dashboards
- Plotly - Data visualization

**Testing:**
- pytest, httpx

**Quality:**
- black, flake8

**Databases:**
- PostgreSQL - Airflow metadata
- Redis - Airflow message broker

## Airflow Automation

### Automated Training Pipeline

The `train_wine_quality_model` DAG automates:

1. **Daily Training** - Runs at 2 AM daily
2. **Model Comparison** - Compares new model with production
3. **Auto Deployment** - Promotes better models to production
4. **Notifications** - Sends email if model doesn't improve

**Enable in Airflow:**
```bash
# Access Airflow UI
http://localhost:8081

# Login: admin / admin
# Toggle DAG switch to ON
```

### DAG Configuration

```python
# Schedule: Daily at 2 AM
schedule_interval='0 2 * * *'

# Automatic model versioning
# Automatic staging/production promotion
# Email notifications on failures
```

## Monitoring Dashboard

### Real-time Monitoring with Streamlit

Access at http://localhost:8501

**Pages:**

1. **Overview**
   - System metrics
   - Recent training runs
   - Performance trends

2. **Model Performance**
   - R² Score trends
   - RMSE analysis
   - Metrics comparison

3. **Data Drift**
   - Evidently drift detection
   - Feature distribution analysis
   - HTML drift reports

4. **Model Registry**
   - Production models
   - Model versions
   - Performance metrics

## Documentation

- **Main README:** This file
- **Docker Guide:** [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)
- **API Docs:** `API_README.md` (detailed API reference)
- **Swagger UI:** http://localhost:8000/docs
- **MLflow UI:** http://localhost:5000
- **Airflow UI:** http://localhost:8081
- **Monitoring:** http://localhost:8501

## Troubleshooting

**Model not found:**
```bash
python main.py --pipeline compare
```

**Port in use:**
```bash
lsof -i :8000
kill -9 <PID>
```

**Import errors:**
```bash
cd wine_quality_mlops
source venv/bin/activate
pip install -r requirements.txt
```

## Contributing

1. Follow PEP8
2. Add tests for new features
3. Update documentation
4. Run `pytest tests/ -v` before commit
5. Check with `flake8 .`

## License

MIT License

## Status

**Production Ready - Full MLSecOps Platform**
- Version: 2.0.0
- Last Updated: January 2025
- Status: Stable

### Recent Updates (v2.0.0)

- Docker multi-container orchestration
- Apache Airflow automated training pipelines
- Streamlit monitoring dashboard
- Evidently data/model drift detection
- Complete MLSecOps infrastructure
- Production-ready deployment

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   Wine Quality MLOps Platform                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Airflow    │  │  MLflow UI   │  │  Streamlit   │      │
│  │   (8081)     │  │   (5000)     │  │   (8501)     │      │
│  │              │  │              │  │              │      │
│  │ - Scheduling │  │ - Tracking   │  │ - Dashboard  │      │
│  │ - Pipelines  │  │ - Registry   │  │ - Monitoring │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  FastAPI     │  │  PostgreSQL  │  │    Redis     │      │
│  │   (8000)     │  │              │  │              │      │
│  │              │  │              │  │              │      │
│  │ - Predictions│  │ - Metadata   │  │ - Queue      │      │
│  │ - REST API   │  │              │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

For detailed API documentation, see [API_README.md](API_README.md)
