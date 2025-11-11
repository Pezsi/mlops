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
- **GCP Cloud Run** - Serverless deployment with CI/CD (GitHub Actions)
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

### Airflow MLOps Platform

A teljes projekt egy production-ready MLOps rendszer Apache Airflow orchestrációval. Az alábbi szolgáltatásokat tartalmazza:

**Szolgáltatások:**
- **Airflow Webserver/Scheduler** (8081) - DAG kezelés és ütemezés
- **MLOps Metadata DB** (PostgreSQL:5433) - Dedikált metadata tracking
- **MLflow** (5000) - Model tracking és registry
- **Webhook API** (8080) - External triggers REST API-val
- **Streamlit Dashboard** (8501) - Real-time monitoring
- **Redis** - Airflow message broker
- **PostgreSQL** - Airflow metadata

### DAG-ok Áttekintése

A projekt **5 különböző DAG**-ot tartalmaz különböző trigger típusokkal:

#### 1. `daily_model_training_with_notification`
**Trigger:** Schedule-based (napi 2:00 AM)

**Funkciók:**
- Automatikus model training MLflow logging-gal
- Metadata tracking PostgreSQL adatbázisba
- Intelligent model comparison (R² és RMSE alapján)
- Multi-stage deployment (Staging → Production)
- Multi-channel notifications (Email, Slack, Database)
- Comprehensive error handling és retry logic

**Workflow:**
```
Train Model → Compare with Production → Better?
   ↓ Yes                                    ↓ No
Deploy to Staging                  Send Warning Notification
   ↓
Deploy to Production
   ↓
Send Success Notification
```

**Callbacks:**
- `on_success_callback` - Task sikeres befejezésekor
- `on_failure_callback` - Task hiba esetén (Email + Slack + DB)
- `on_retry_callback` - Retry kísérletkor (Slack + DB)

#### 2. `dataset_sensor_event_trigger`
**Trigger:** Event-based (FileSensor)

**Funkciók:**
- Automatikusan detektálja új dataset fájlokat (`/opt/airflow/data/incoming`)
- Dataset validáció (schema, quality checks)
- **Data drift detection:**
  - Kolmogorov-Smirnov test numerikus feature-ökre
  - Population Stability Index (PSI) számítás
  - Threshold: PSI > 0.2 → drift detected
- Conditional model retraining trigger drift esetén
- Processed fájl mozgatás

#### 3. `model_deployment_pipeline`
**Trigger:** Schedule + ExternalTaskSensor (napi 4:00 AM)

**Funkciók:**
- Vár a training DAG befejezésére (`ExternalTaskSensor`)
- Comprehensive model evaluation
- A/B test setup
- Multi-stage deployment
- Deployment report generation
- Stakeholder notifications

#### 4. `data_pipeline_orchestrator`
**Trigger:** Schedule-based (napi 1:00 AM)

**Funkciók:**
- Több data pipeline orchestrálása parallel
- Complex dependencies kezelése
- Aggregált adatfeldolgozás
- Conditional training trigger

#### 5. `webhook_triggered_training`
**Trigger:** API/Webhook (on-demand)

**Funkciók:**
- REST API triggered training custom paraméterekkel
- Configuration parsing és validation
- Training szükségesség ellenőrzés
- Deployment target environment-be
- Callback URL support

**API használat:**
```bash
curl -X POST http://localhost:8080/trigger/training \
  -H "Content-Type: application/json" \
  -H "X-API-Key: mlops-secret-key-2025" \
  -d '{
    "model_name": "wine_quality_rf_model",
    "trigger_source": "ci_cd",
    "force_training": true,
    "hyperparameters": {"n_estimators": 200}
  }'
```

### Metadata Tracking System

**PostgreSQL adatbázis 14 táblával:**

**Fő táblák:**
- `model_runs` - Training futások tracking
- `model_metrics` - Metrikák (R², RMSE, MAE, stb.)
- `model_parameters` - Hyperparaméterek
- `dataset_versions` - Dataset verziókezelés
- `data_lineage` - Data-model kapcsolatok
- `model_comparisons` - Model összehasonlítások
- `pipeline_events` - Pipeline események (info, warning, error)
- `notification_history` - Értesítési előzmények
- `data_drift_events` - Drift detection eredmények
- `model_monitoring` - Production monitoring
- `feature_statistics` - Feature store
- `ab_experiments` - A/B tesztek tracking

**View-k:**
- `latest_production_models` - Legfrissebb production modellek
- `recent_pipeline_events` - Friss pipeline események
- `model_performance_comparison` - Performance összehasonlítás

**Python API:**
```python
from airflow.utils.metadata_tracker import MetadataTracker

with MetadataTracker(METADATA_DB_CONN) as tracker:
    run_id = tracker.create_model_run(...)
    tracker.log_metrics(run_id, {'r2_score': 0.87})
    tracker.log_parameters(run_id, {'n_estimators': 100})
```

### Multi-Channel Notification System

**3 csatorna:**
- **Email (SMTP/Gmail)** - HTML formázott emailek színkóddal
- **Slack (Webhooks)** - Rich formatting, emoji indicators
- **Database** - Audit trail és history tracking

**Notification típusok:**
- Training start/success/failure
- Model deployment
- Model comparison results
- Data drift alerts
- Pipeline health issues

**Konfigurálás:**
```bash
# .env fájlban
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
TO_EMAILS=admin@example.com,team@example.com
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

### Webhook Trigger API

**REST API service a DAG-ok triggerelésére:**

**Endpoints:**
- `GET /health` - Health check
- `POST /trigger/training` - Training trigger
- `POST /trigger/evaluation` - Evaluation trigger
- `GET /status/<dag_id>/<run_id>` - Status check
- `GET /dags` - Available DAGs

**Features:**
- API key authentication
- Request validation
- Callback URL support
- Airflow API wrapper

### Airflow UI Elérés

```bash
# Access Airflow UI
http://localhost:8081

# Login: admin / admin
# Toggle DAG switch to ON/OFF
```

## Monitoring Dashboard

### Real-time Monitoring with Streamlit

Access at http://localhost:8501

**Streamlit Dashboard komponensek:**

1. **Overview Page**
   - System health metrics
   - Recent training runs summary
   - Performance trends visualization
   - Active DAG status
   - Recent pipeline events

2. **Model Performance**
   - R² Score trends over time
   - RMSE/MAE analysis
   - Model comparison charts
   - Best performing models
   - Training duration tracking

3. **Data Drift Detection**
   - **Evidently AI integration**
   - Feature distribution comparison (reference vs current)
   - Statistical drift tests (KS-test, Chi-square)
   - Population Stability Index (PSI)
   - HTML drift reports generation
   - Automatic drift alerts

4. **Model Registry**
   - Production model list
   - Model versions és stage tracking
   - Performance metrics comparison
   - Model deployment history
   - A/B experiment tracking

5. **Pipeline Monitoring**
   - DAG execution history
   - Task success/failure rates
   - Average execution times
   - Recent errors és warnings
   - Notification history

**Features:**
- Real-time data refresh
- Interactive Plotly visualizations
- MLflow integration
- Metadata database queries
- Export reports PDF/HTML formátumban

## Teljes Projekt Áttekintés

### Mi van a projektben?

Ez egy **komplett, production-ready MLOps platform**, amely demonstrálja a modern gépi tanulási rendszerek teljes életciklusát:

#### 🤖 Machine Learning
- **2 regressziós model:** Random Forest és Gradient Boosting
- **Preprocessing pipeline:** StandardScaler, feature engineering
- **Hyperparameter tuning:** GridSearchCV 10-fold CV-vel
- **Performance:** R² ~0.47-0.49, RMSE ~0.57-0.58

#### 🔄 MLOps Automation (Apache Airflow)
- **5 DAG különböző trigger típusokkal:**
  1. Schedule-based: Napi automatikus training (2 AM)
  2. Event-based: FileSensor új adatok detektálására
  3. ExternalTaskSensor: DAG-ok közötti függőségek
  4. Webhook: API-triggered training custom config-gal
  5. Orchestration: Multi-pipeline koordináció
- **Metadata tracking:** PostgreSQL 14 táblával
- **Multi-channel notifications:** Email (SMTP/Gmail), Slack, Database
- **Error handling:** Success/failure/retry callbacks

#### 📊 Model Tracking & Registry (MLflow)
- **Experiment tracking:** Metrics, parameters, artifacts
- **Model registry:** Verziókezelés, stage management
- **Model comparison:** Automated best model selection
- **Artifact storage:** Model persistence és versioning

#### 🌐 REST APIs (3 API)
1. **FastAPI** (8000) - Async predictions, Pydantic validation
2. **Flask-RESTX** (8000) - Traditional REST, mature ecosystem
3. **Webhook API** (8080) - DAG triggering, API key auth

**API Features:**
- Single & batch predictions
- Model training trigger
- Model metrics és management
- Health checks
- Swagger UI documentation

#### 📈 Monitoring & Visualization
- **Streamlit Dashboard** (8501):
  - Overview: System metrics, recent runs
  - Model Performance: R² trends, RMSE analysis
  - Data Drift: Evidently integration, distribution comparison
  - Model Registry: Production models, versions
  - Pipeline Monitoring: DAG execution history, error tracking

#### 🔍 Data Quality & Drift Detection
- **Automatic validation:** Schema, quality checks
- **Statistical tests:**
  - Kolmogorov-Smirnov test
  - Population Stability Index (PSI)
  - Chi-square test
- **Drift threshold:** PSI > 0.2
- **Automated retraining:** Conditional trigger drift esetén

#### 🐳 Docker Infrastructure
- **9 containerized services:**
  - Airflow (webserver + scheduler + init)
  - MLflow API
  - Webhook API
  - Streamlit monitoring
  - PostgreSQL (2 instance: Airflow + MLOps)
  - Redis (message broker)
- **Multi-service orchestration:** docker-compose
- **Shared volumes:** Models, data, logs, mlruns
- **Bridge network:** Service-to-service communication

#### 🧪 Testing & Quality
- **91+ tests:** Unit, integration, API tests
- **Test coverage:** All modules tested
- **PEP8 compliant:** 0 linting errors
- **Code formatting:** Black, flake8
- **Type checking:** mypy support

#### 📦 Data Pipeline
- **Dataset:** UCI Wine Quality (red wine)
- **Features:** 11 physicochemical properties
- **Target:** Quality score (0-10)
- **Split:** 80/20 train/test (stratified)
- **Preprocessing:** Standardization, optional PCA

#### 🔐 Security & Configuration
- **Environment variables:** .env file
- **API authentication:** API key-based (Webhook API)
- **Database credentials:** Configurable
- **Secret management:** Not hardcoded
- **Production hardening:** TODO (HTTPS, JWT, rate limiting)

### Workflow Összefoglalás

```
┌─────────────────────────────────────────────────────────────┐
│                    Complete MLOps Workflow                   │
└─────────────────────────────────────────────────────────────┘

1. Data Ingestion
   └─→ FileSensor detektálja új adatot
       └─→ Data validation + drift detection

2. Model Training (3 trigger mód)
   ├─→ Schedule: Napi 2 AM automatikus
   ├─→ Event: Drift detekció esetén
   └─→ API: Webhook trigger external system-ből

3. Tracking & Logging
   ├─→ MLflow: Experiments, runs, metrics, artifacts
   └─→ PostgreSQL: Metadata, lineage, events, notifications

4. Model Comparison & Deployment
   └─→ Compare new model vs production
       ├─→ Better? → Deploy Staging → Production
       └─→ Not better? → Send notification (no deploy)

5. Monitoring & Alerting
   ├─→ Streamlit: Real-time dashboard
   ├─→ Evidently: Drift detection
   └─→ Notifications: Email + Slack + DB

6. Model Serving
   └─→ FastAPI/Flask: REST API predictions
       ├─→ Single prediction
       ├─→ Batch predictions
       └─→ Model management

7. External Integration
   └─→ Webhook API: CI/CD, monitoring systems
       └─→ Trigger training, evaluation, deployment
```

### Technológiai Stack

**ML & Data Science:**
- scikit-learn, pandas, numpy, scipy
- Feature engineering, hyperparameter tuning

**MLOps & Orchestration:**
- Apache Airflow - Workflow automation
- MLflow - Experiment tracking & model registry
- Evidently AI - Data & model drift detection

**APIs & Web:**
- FastAPI - Modern async REST
- Flask-RESTX - Traditional REST
- Streamlit - Interactive dashboards

**Databases:**
- PostgreSQL - Airflow & MLOps metadata
- Redis - Message broker & caching

**Containerization:**
- Docker - Application containers
- Docker Compose - Multi-service orchestration

**Monitoring & Visualization:**
- Plotly - Interactive charts
- Evidently - Drift reports
- Custom dashboards

**Testing & Quality:**
- pytest - Testing framework
- black - Code formatting
- flake8 - Linting
- mypy - Type checking

**Notifications:**
- SMTP/Gmail - Email alerts
- Slack Webhooks - Chat notifications
- Database logging - Audit trail

### Projekt Méretei

- **Total Code:** ~3,500+ lines Python
- **Test Code:** ~1,300 lines
- **API Code:** ~1,800 lines (2 prediction + 1 webhook API)
- **DAG Code:** ~1,500 lines (5 DAG-ok)
- **Tests:** 91+ test cases
- **Docker Services:** 9 containers
- **Database Tables:** 14 metadata tables
- **API Endpoints:** 30+ endpoints összesen

### Használati Példák

**1. Training trigger scheduled:**
```bash
# Airflow automatikusan futtatja naponta 2 AM-kor
# Vagy manuálisan:
docker-compose exec airflow-scheduler airflow dags trigger daily_model_training_with_notification
```

**2. Training trigger webhook-kel:**
```bash
curl -X POST http://localhost:8080/trigger/training \
  -H "X-API-Key: mlops-secret-key-2025" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "wine_quality_rf_model", "trigger_source": "manual"}'
```

**3. Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {...}, "model_name": "rf"}'
```

**4. Monitoring:**
```bash
# Streamlit Dashboard
open http://localhost:8501

# MLflow UI
open http://localhost:5000

# Airflow UI
open http://localhost:8081
```

**5. Metadata query:**
```sql
-- Latest production models
SELECT * FROM latest_production_models;

-- Recent drift events
SELECT * FROM data_drift_events WHERE drift_detected = true;

-- Model performance comparison
SELECT * FROM model_performance_comparison ORDER BY performance_rank;
```

## Documentation

- **Main README:** This file - Teljes projekt áttekintés
- **API Documentation:** [API_README.md](API_README.md) - 3 API részletes dokumentációja
- **Docker Guide:** [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) - Docker deployment
- **FastAPI Swagger:** http://localhost:8000/docs - Interactive API docs
- **Flask-RESTX Swagger:** http://localhost:8000/docs/ - Interactive API docs
- **Webhook API Swagger:** http://localhost:8080/docs - Webhook API docs
- **MLflow UI:** http://localhost:5000 - Experiment tracking
- **Airflow UI:** http://localhost:8081 - DAG monitoring
- **Streamlit Dashboard:** http://localhost:8501 - Real-time monitoring

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
┌──────────────────────────────────────────────────────────────────┐
│                   Wine Quality MLOps Platform                     │
│                     (Docker Compose Multi-Service)                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────┐ │
│  │  Airflow    │  │  MLflow UI  │  │ Streamlit   │  │Webhook │ │
│  │  Webserver  │  │  + FastAPI  │  │  Dashboard  │  │  API   │ │
│  │   (8081)    │  │   (5000)    │  │   (8501)    │  │ (8080) │ │
│  │             │  │             │  │             │  │        │ │
│  │• DAG UI     │  │• Tracking   │  │• Monitoring │  │• Trigger│ │
│  │• Scheduling │  │• Registry   │  │• Drift Det. │  │• REST  │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └───┬────┘ │
│         │                │                │              │      │
│  ┌──────┴────────────────┴────────────────┴──────────────┴───┐ │
│  │                    Shared Volumes                           │ │
│  │  /models  /mlruns  /data  /logs  /airflow                  │ │
│  └──────┬────────────────┬────────────────┬──────────────┬───┘ │
│         │                │                │              │      │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐  ┌──┴────┐ │
│  │  Airflow    │  │ PostgreSQL  │  │ PostgreSQL  │  │ Redis │ │
│  │  Scheduler  │  │  (Airflow)  │  │  (MLOps)    │  │       │ │
│  │             │  │   (5432)    │  │   (5433)    │  │(6379) │ │
│  │             │  │             │  │             │  │       │ │
│  │• Dag Runs   │  │• Airflow    │  │• Metadata   │  │• Queue│ │
│  │• Tasks      │  │  Metadata   │  │• Tracking   │  │• Cache│ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────┘ │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘

Data Flow:
1. Airflow Scheduler → DAG Tasks végrehajtás
2. Tasks → MLflow tracking (experiments, runs, metrics)
3. Tasks → MLOps Metadata DB (pipeline events, model runs, metrics)
4. Webhook API → Airflow API → DAG trigger
5. Streamlit → MLflow + MLOps DB → Real-time visualization
6. FastAPI/Flask → Models → Predictions
```

### Docker Services Összefoglalás

**9 Docker Service:**

1. **postgres** - Airflow metadata database
2. **mlops-metadata-db** - MLOps tracking database (**Dedikált MLOps DB**)
3. **redis** - Airflow Celery backend
4. **airflow-webserver** - Airflow UI (8081)
5. **airflow-scheduler** - DAG scheduling engine
6. **airflow-init** - Database initialization
7. **mlflow-api** - MLflow tracking + FastAPI/Flask (5000, 8000)
8. **streamlit-monitoring** - Monitoring dashboard (8501)
9. **webhook-api** - Webhook trigger service (8080)

**Volumes:**
- `./airflow/dags` → `/opt/airflow/dags`
- `./models` → `/app/models`
- `./mlruns` → `/app/mlruns`
- `./data` → `/app/data` és `/opt/airflow/data`
- `./logs` → `/app/logs`

**Networks:**
- `mlops-network` - Bridge network összes service számára

---

## Cloud Deployment - GCP Cloud Run

A projekt támogatja **Google Cloud Platform Cloud Run** deployment-et automatikus CI/CD pipeline-nal.

### Jellemzők

- Serverless autoscaling (0-10 instances)
- Automatikus CI/CD GitHub Actions-szel
- Production-ready konfiguráció
- Költséghatékony (pay-per-use, free tier elérhető)
- HTTPS endpoint automatikus SSL-lel

### Gyors Setup

```bash
# 1. GCP környezet beállítása
export GCP_PROJECT_ID="your-project-id"
./gcp/setup-gcp.sh

# 2. GitHub Secrets beállítása
# - GCP_PROJECT_ID: your-project-id
# - GCP_SA_KEY: (gcp-sa-key.json tartalma)

# 3. Deploy
git push origin main  # Automatikus deployment
# vagy
./gcp/deploy.sh      # Manuális deployment
```

### Monitoring

```bash
# Interaktív monitoring tool
./gcp/monitor.sh

# Real-time logs
gcloud logging tail "resource.type=cloud_run_revision"
```

### Dokumentáció

- **Teljes útmutató**: [GCP_DEPLOYMENT_GUIDE.md](GCP_DEPLOYMENT_GUIDE.md) - Részletes deployment guide
- **Gyors útmutató**: [gcp/QUICK_START.md](gcp/QUICK_START.md) - 5 perces setup
- **Implementáció**: [GCP_IMPLEMENTATION_SUMMARY.md](GCP_IMPLEMENTATION_SUMMARY.md) - Architektúra és összefoglaló

---

For detailed API documentation, see [API_README.md](API_README.md)
