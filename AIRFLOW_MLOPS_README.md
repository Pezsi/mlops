# Wine Quality MLOps - Comprehensive Airflow Pipeline

## Áttekintés

Ez egy production-ready MLOps rendszer, amely bemutatja a modern gépi tanulási pipeline-ok legfontosabb funkcióit:

### Főbb Jellemzők

🔄 **Event-based és Schedule-based Automatizáció**
- Schedule-based: Napi automatikus model training (cron: `0 2 * * *`)
- Event-based: Dataset file sensor - új adat érkezésekor automatikus indítás
- External task sensor: DAG-ok közötti függőségkezelés
- API/Webhook trigger: Külső rendszerekből történő triggerelés

📊 **Comprehensive Metadata Tracking**
- Dedicated PostgreSQL adatbázis MLOps metadatákhoz
- Model runs, metrics, parameters tracking
- Data lineage követés
- Feature statistics és verziókezelés
- Model comparison history
- AB test tracking

🔔 **Multi-channel Notification System**
- Email értesítések (SMTP)
- Slack integration (webhooks)
- Database logging
- Success, failure, és retry callbacks
- Structured event logging

🎯 **Complex Branching és Deployment**
- Intelligent model comparison
- Multi-stage deployment (Staging → Production)
- Automated rollback on failure
- A/B testing support
- Conditional workflows

🔍 **Data Quality és Drift Detection**
- Automatic data validation
- Statistical drift detection (KS-test, PSI)
- Feature distribution monitoring
- Schema validation

## Architektúra

```
┌─────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │  Schedule    │     │ File Sensor  │     │   Webhook    │    │
│  │  Trigger     │────▶│   Trigger    │────▶│   Trigger    │    │
│  │ (Daily 2AM)  │     │ (New Data)   │     │  (API Call)  │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│         │                     │                     │            │
│         └─────────────────────┼─────────────────────┘            │
│                               ▼                                  │
│                    ┌──────────────────┐                          │
│                    │  Train Model     │                          │
│                    │  + Metadata Log  │                          │
│                    └──────────────────┘                          │
│                               │                                  │
│                               ▼                                  │
│                    ┌──────────────────┐                          │
│                    │ Model Comparison │                          │
│                    │   (Branching)    │                          │
│                    └──────────────────┘                          │
│                         │         │                              │
│                    Better    Not Better                          │
│                         │         │                              │
│                         ▼         ▼                              │
│                  ┌─────────┐  ┌────────────┐                    │
│                  │ Deploy  │  │  Notify    │                    │
│                  │ Staging │  │ (No Deploy)│                    │
│                  └─────────┘  └────────────┘                    │
│                         │                                        │
│                         ▼                                        │
│                  ┌──────────────┐                                │
│                  │   Deploy     │                                │
│                  │  Production  │                                │
│                  └──────────────┘                                │
│                         │                                        │
│                         ▼                                        │
│                  ┌──────────────┐                                │
│                  │ Notification │                                │
│                  │ (Multi-chan) │                                │
│                  └──────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

## Szolgáltatások

### 1. Airflow (Port: 8081)
- **Webserver**: DAG kezelés, monitoring
- **Scheduler**: Automatikus DAG ütemezés
- **Metadata DB**: Airflow state tracking

### 2. MLOps Metadata Database (Port: 5433)
- **PostgreSQL**: Dedikált adatbázis MLOps tracking-hez
- **Schema**: 14 tábla teljes MLOps lifecycle-hoz

### 3. MLflow (Port: 5000)
- **Tracking**: Model metrikák és artifacts
- **Registry**: Model verziókezelés

### 4. Webhook API (Port: 8080)
- **External triggers**: REST API DAG indításhoz
- **Authentication**: API key based

### 5. Streamlit Dashboard (Port: 8501)
- **Monitoring**: Real-time metrics
- **Visualization**: Model performance

## Telepítés és Futtatás

### 1. Előfeltételek

```bash
# Docker és Docker Compose telepítése
docker --version
docker-compose --version
```

### 2. Konfigurálás

```bash
# 1. Környezeti változók beállítása
cp .env.example .env

# 2. Szerkeszd a .env fájlt saját értékeiddel
# - Email beállítások (Gmail App Password)
# - Slack webhook URL
# - API kulcsok

vim .env
```

### 3. Build és Indítás

```bash
# Build all services
docker-compose build

# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 4. Ellenőrzés

```bash
# Airflow webserver
open http://localhost:8081
# Username: admin
# Password: admin

# MLflow UI
open http://localhost:5000

# Webhook API docs
open http://localhost:8080/docs

# Streamlit dashboard
open http://localhost:8501

# Metadata DB (psql)
psql -h localhost -p 5433 -U mlops_user -d mlops_metadata
# Password: mlops_password
```

## DAG-ok Részletes Leírása

### 1. `daily_model_training_with_notification`

**Cél**: Napi automatikus model training teljes életciklussal

**Trigger**: Schedule - Minden nap 2:00 AM

**Tasks**:
1. `initialize_metadata` - Metadata tracking inicializálása
2. `train_model` - Model training MLflow logging-gal
3. `branch_decision` - Model comparison és branching
4. `deploy_to_staging` - Staging környezetbe deploy
5. `deploy_to_production` - Production deploy
6. `send_notification` - Multi-channel értesítés

**Callbacks**:
- `on_success_callback` - Task sikeres befejezésekor
- `on_failure_callback` - Task hiba esetén (email + Slack)
- `on_retry_callback` - Retry kísérletkor

**Példa használat**:
```bash
# Manual trigger via Airflow UI or CLI
airflow dags trigger daily_model_training_with_notification
```

### 2. `dataset_sensor_event_trigger`

**Cél**: Új dataset fájlok automatikus detektálása és feldolgozása

**Trigger**: Event-based - FileSensor (`/opt/airflow/data/incoming`)

**Tasks**:
1. `wait_for_new_dataset` - FileSensor - új fájl várása
2. `validate_dataset` - Schema és quality validáció
3. `detect_data_drift` - Statistical drift detection
4. `log_drift_to_metadata` - Drift eredmények logging
5. `trigger_retraining` - Training DAG indítása drift esetén
6. `move_to_processed` - Fájl mozgatása processed mappába

**Data Drift Detection**:
- **Kolmogorov-Smirnov test**: Numerikus feature-ök eloszlása
- **Population Stability Index (PSI)**: Globális drift score
- **Threshold**: PSI > 0.2 → drift detected

**Példa használat**:
```bash
# Drop a new dataset file
cp new_dataset.csv /path/to/data/incoming/

# FileSensor automatically detects and processes
```

### 3. `model_deployment_pipeline`

**Cél**: External task sensor - training után deployment orchestration

**Trigger**: Schedule - 4:00 AM (training után 2 órával)

**External Dependencies**:
- Vár a `train_wine_quality_model` DAG befejezésére
- `execution_delta=timedelta(hours=2)` - 2 órával korábbi run

**Tasks**:
1. `wait_for_training_completion` - ExternalTaskSensor
2. `evaluate_model_performance` - Comprehensive evaluation
3. `setup_ab_test` - A/B test konfiguráció
4. `deploy_to_staging` - Staging deploy
5. `generate_deployment_report` - Deployment report
6. `notify_stakeholders` - Értesítések

### 4. `data_pipeline_orchestrator`

**Cél**: Több data pipeline orchestrálása complex dependencies-szel

**Trigger**: Schedule - 1:00 AM

**Orchestration**:
- Több external task sensor parallel
- Aggregált adatfeldolgozás
- Conditional training trigger

### 5. `webhook_triggered_training`

**Cél**: API/webhook triggered training custom paraméterekkel

**Trigger**: API call only (no schedule)

**API Endpoint**:
```bash
POST /trigger/training
Headers:
  Content-Type: application/json
  X-API-Key: mlops-secret-key-2025
Body:
{
  "model_name": "wine_quality_rf_model",
  "trigger_source": "ci_cd_pipeline",
  "dataset_version": "v2.1.0",
  "hyperparameters": {
    "n_estimators": 200,
    "max_depth": 10
  },
  "force_training": true,
  "deployment_target": "staging",
  "callback_url": "https://your-system.com/callback"
}
```

**Tasks**:
1. `parse_trigger_config` - Configuration parsing
2. `validate_trigger_request` - Input validation
3. `check_training_needed` - Training szükségesség ellenőrzés
4. `prepare_training_environment` - Environment setup
5. `execute_model_training` - Training végrehajtás
6. `deploy_model` - Deploy target environment-be
7. `send_webhook_response` - Response callback URL-re

## Webhook Trigger API

### Elérhető Endpoint-ok

#### 1. Health Check
```bash
curl http://localhost:8080/health
```

#### 2. Trigger Training
```bash
curl -X POST http://localhost:8080/trigger/training \
  -H "Content-Type: application/json" \
  -H "X-API-Key: mlops-secret-key-2025" \
  -d '{
    "model_name": "wine_quality_rf_model",
    "trigger_source": "manual",
    "force_training": true
  }'
```

#### 3. Trigger Evaluation
```bash
curl -X POST http://localhost:8080/trigger/evaluation \
  -H "Content-Type: application/json" \
  -H "X-API-Key: mlops-secret-key-2025" \
  -d '{
    "model_run_id": "run_20250102_120000",
    "evaluation_dataset": "holdout_v1"
  }'
```

#### 4. Check Status
```bash
curl http://localhost:8080/status/webhook_triggered_training/manual_2025-01-02T10:00:00+00:00 \
  -H "X-API-Key: mlops-secret-key-2025"
```

#### 5. List DAGs
```bash
curl http://localhost:8080/dags \
  -H "X-API-Key: mlops-secret-key-2025"
```

## Metadata Database Schema

### Főbb Táblák

1. **model_runs**: Model training futások
2. **model_metrics**: Metrikák (R², RMSE, MAE, stb.)
3. **model_parameters**: Hyperparaméterek
4. **dataset_versions**: Dataset verziók
5. **data_lineage**: Data-model kapcsolatok
6. **model_comparisons**: Model összehasonlítások
7. **pipeline_events**: Pipeline események (info, warning, error)
8. **notification_history**: Értesítési előzmények
9. **data_drift_events**: Drift detekciós események
10. **model_monitoring**: Production monitoring

### Példa Lekérdezések

```sql
-- Latest production models
SELECT * FROM latest_production_models;

-- Recent pipeline events
SELECT * FROM recent_pipeline_events WHERE event_type = 'error';

-- Model performance comparison
SELECT * FROM model_performance_comparison
WHERE model_name = 'wine_quality_rf_model'
ORDER BY performance_rank;

-- Data drift events
SELECT * FROM data_drift_events
WHERE drift_detected = true
ORDER BY detection_timestamp DESC
LIMIT 10;
```

## Notification System

### Email Konfiguráció

```bash
# Gmail App Password generálás:
# 1. Google Account → Security
# 2. 2-Step Verification bekapcsolás
# 3. App Passwords → Generate
# 4. Copy 16-character password to .env

SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=abcd efgh ijkl mnop
TO_EMAILS=admin@example.com,team@example.com
```

### Slack Konfiguráció

```bash
# Slack Incoming Webhook:
# 1. https://api.slack.com/messaging/webhooks
# 2. Create New App
# 3. Activate Incoming Webhooks
# 4. Add New Webhook to Workspace
# 5. Copy Webhook URL

SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX
```

### Notification Szintek

- **INFO**: Sikeres task befejezések, deployments
- **WARNING**: Model nem javult, retry attempts
- **ERROR**: Task failures, deployment failures
- **CRITICAL**: Systemic failures, data corruption

## Troubleshooting

### 1. DAG nem jelenik meg az Airflow UI-ban

```bash
# Check DAG errors
docker-compose exec airflow-scheduler airflow dags list

# Check specific DAG
docker-compose exec airflow-scheduler python -c "
from airflow.models import DagBag
dagbag = DagBag()
print(dagbag.import_errors)
"
```

### 2. Metadata DB connection hiba

```bash
# Check database connection
docker-compose exec mlops-metadata-db psql -U mlops_user -d mlops_metadata -c "SELECT 1;"

# Reinitialize schema
docker-compose exec mlops-metadata-db psql -U mlops_user -d mlops_metadata -f /docker-entrypoint-initdb.d/01-schema.sql
```

### 3. Notification nem működik

```bash
# Test email
docker-compose exec airflow-scheduler python -c "
from airflow.utils.notification_system import create_notification_system
notifier = create_notification_system(
    smtp_user='your-email@gmail.com',
    smtp_password='your-app-password',
    to_emails=['admin@example.com']
)
notifier.send_notification('Test', 'Test message', level='info')
"
```

### 4. Webhook API nem elérhető

```bash
# Check webhook API logs
docker-compose logs webhook-api

# Test health endpoint
curl http://localhost:8080/health

# Check Airflow API connectivity
curl -u admin:admin http://localhost:8081/api/v1/dags
```

## Best Practices

### 1. Metadata Tracking

```python
# Always use MetadataTracker in context manager
with MetadataTracker(METADATA_DB_CONN) as tracker:
    tracker.create_model_run(...)
    tracker.log_metrics(...)
```

### 2. Error Handling

```python
# Implement proper error handling with metadata logging
try:
    # Your task logic
    pass
except Exception as e:
    with MetadataTracker(METADATA_DB_CONN) as tracker:
        tracker.log_pipeline_event(
            dag_id=dag_id,
            event_type='error',
            event_message=str(e)
        )
    raise
```

### 3. Notification Best Practices

- Use **Slack** for high-frequency, low-priority events
- Use **Email** for important events (failures, deployments)
- Always log to **Database** for audit trail

### 4. Model Versioning

- Use semantic versioning: `v1.0.0`, `v1.1.0`, etc.
- Always link models to datasets (data lineage)
- Track all hyperparameters

## Monitoring és Maintenance

### Napi Feladatok

```bash
# Check pipeline health
docker-compose ps

# Review recent errors
docker-compose logs --tail=100 airflow-scheduler | grep ERROR
```

### Heti Feladatok

```bash
# Database cleanup (old logs)
docker-compose exec mlops-metadata-db psql -U mlops_user -d mlops_metadata -c "
DELETE FROM pipeline_events WHERE created_at < NOW() - INTERVAL '30 days';
"

# Review model performance trends
# Query model_performance_comparison view
```

### Havi Feladatok

```bash
# Backup metadata database
docker-compose exec mlops-metadata-db pg_dump -U mlops_user mlops_metadata > backup_$(date +%Y%m%d).sql

# Review and archive old models
# Archive models older than 6 months
```

## Bővítési Lehetőségek

### 1. Additional Data Sources
- S3 sensor integration
- Database sensors (new records)
- Kafka consumer triggers

### 2. Advanced Monitoring
- Prometheus metrics export
- Grafana dashboards
- Custom alerting rules

### 3. Model Serving
- Deploy to Kubernetes
- Serverless endpoints (AWS Lambda)
- Real-time inference API

### 4. CI/CD Integration
- GitHub Actions workflow
- Automated testing
- Deployment pipelines

## Kapcsolódó Dokumentáció

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Flask Documentation](https://flask.palletsprojects.com/)

## Licenc

MIT License

## Support

Ha kérdésed van, hozz létre egy issue-t a GitHub repository-ban.

---

**Készült 2025-ben tanulási célokra. Production használathoz további security hardening szükséges.**
