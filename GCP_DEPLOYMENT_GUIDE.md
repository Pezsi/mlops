# Wine Quality MLOps - GCP Cloud Run Deployment Guide

## Áttekintés

Ez az útmutató bemutatja, hogyan deploy-olhatod a Wine Quality MLOps projektet Google Cloud Platform Cloud Run szolgáltatásra, CI/CD pipeline-nal GitHub Actions használatával.

## Architektúra

```
┌─────────────────────────────────────────────────────────────────┐
│                    GCP MLOps Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  GitHub Repository                                                │
│         │                                                         │
│         │ (push to main/master)                                   │
│         ▼                                                         │
│  ┌──────────────────┐                                            │
│  │ GitHub Actions   │                                            │
│  │   Workflow       │                                            │
│  └──────────────────┘                                            │
│         │                                                         │
│         │ Build & Test                                            │
│         ▼                                                         │
│  ┌──────────────────┐                                            │
│  │ Docker Build     │                                            │
│  │ (Dockerfile.     │                                            │
│  │  cloudrun)       │                                            │
│  └──────────────────┘                                            │
│         │                                                         │
│         │ Push Image                                              │
│         ▼                                                         │
│  ┌──────────────────┐                                            │
│  │ GCP Artifact     │                                            │
│  │   Registry       │                                            │
│  └──────────────────┘                                            │
│         │                                                         │
│         │ Deploy                                                  │
│         ▼                                                         │
│  ┌──────────────────┐                                            │
│  │  Cloud Run       │◄─────── Autoscaling (0-10 instances)      │
│  │   Service        │                                            │
│  │                  │                                            │
│  │ - FastAPI        │                                            │
│  │ - MLflow         │                                            │
│  │ - ML Model       │                                            │
│  └──────────────────┘                                            │
│         │                                                         │
│         │ HTTPS Endpoint                                          │
│         ▼                                                         │
│  ┌──────────────────┐                                            │
│  │    Users /       │                                            │
│  │  Applications    │                                            │
│  └──────────────────┘                                            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Előfeltételek

### 1. Google Cloud Platform Account

- GCP account létrehozása: https://cloud.google.com
- Billing engedélyezése a projekten
- GCP Project létrehozása

### 2. Helyi Eszközök

```bash
# Google Cloud SDK telepítése
# macOS
brew install google-cloud-sdk

# Linux
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Windows
# Töltsd le a telepítőt: https://cloud.google.com/sdk/docs/install

# Ellenőrzés
gcloud --version
```

### 3. Docker Telepítése

```bash
# Docker telepítése
# https://docs.docker.com/get-docker/

# Ellenőrzés
docker --version
```

## Telepítés - Automatikus (Ajánlott)

### 1. GCP Erőforrások Beállítása

```bash
# Navigálj a projekt könyvtárába
cd /path/to/wine_quality_mlops

# Futtasd a setup szkriptet
./gcp/setup-gcp.sh

# Vagy manuálisan add meg a Project ID-t
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="europe-west1"  # Opcionális, alapértelmezett: europe-west1
./gcp/setup-gcp.sh
```

A szkript elvégzi:
- ✓ GCP API-k engedélyezése (Cloud Run, Artifact Registry, Cloud Build)
- ✓ Artifact Registry repository létrehozása
- ✓ Service Account létrehozása
- ✓ IAM jogosultságok beállítása
- ✓ Service Account key generálása (GitHub Actions-höz)

### 2. GitHub Secrets Beállítása

A `setup-gcp.sh` szkript generál egy `gcp-sa-key.json` fájlt. Ezt kell hozzáadni a GitHub Secrets-hez.

**GitHub Repository → Settings → Secrets and variables → Actions → New repository secret:**

1. **GCP_PROJECT_ID**
   - Name: `GCP_PROJECT_ID`
   - Value: `your-project-id`

2. **GCP_SA_KEY**
   - Name: `GCP_SA_KEY`
   - Value: `gcp-sa-key.json` fájl teljes tartalma

```bash
# gcp-sa-key.json tartalmának másolása
cat gcp-sa-key.json
```

### 3. Deployment GitHub Actions-szel

A GitHub Actions workflow automatikusan elindul minden push esetén a `main` vagy `master` branchre.

```bash
# Commit és push
git add .
git commit -m "Add GCP Cloud Run deployment"
git push origin main
```

**Manuális trigger** (opcionális):
- GitHub → Actions → "Deploy to GCP Cloud Run" → Run workflow

## Telepítés - Manuális

### 1. GCP Projekt Beállítása (Manuális)

```bash
# Projekt ID beállítása
export PROJECT_ID="your-project-id"
export REGION="europe-west1"

# GCP projekt kiválasztása
gcloud config set project $PROJECT_ID

# API-k engedélyezése
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com

# Artifact Registry repository létrehozása
gcloud artifacts repositories create wine-quality-mlops \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker repository for Wine Quality MLOps"

# Service Account létrehozása
gcloud iam service-accounts create mlops-cloud-run-sa \
    --display-name="MLOps Cloud Run Service Account"

# Jogosultságok megadása
SA_EMAIL="mlops-cloud-run-sa@$PROJECT_ID.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/artifactregistry.reader"
```

### 2. Docker Image Build és Push

```bash
# Autentikáció Docker-hez
gcloud auth configure-docker $REGION-docker.pkg.dev

# Image build
docker build \
    -f Dockerfile.cloudrun \
    -t $REGION-docker.pkg.dev/$PROJECT_ID/wine-quality-mlops/wine-quality-mlops:latest \
    .

# Push Artifact Registry-be
docker push $REGION-docker.pkg.dev/$PROJECT_ID/wine-quality-mlops/wine-quality-mlops:latest
```

### 3. Cloud Run Service Deploy

```bash
# Deploy Cloud Run-ra
gcloud run deploy wine-quality-mlops \
    --image $REGION-docker.pkg.dev/$PROJECT_ID/wine-quality-mlops/wine-quality-mlops:latest \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --min-instances 0 \
    --set-env-vars "MLFLOW_TRACKING_URI=file:/app/mlruns,ENVIRONMENT=production"

# Service URL lekérése
SERVICE_URL=$(gcloud run services describe wine-quality-mlops \
    --region $REGION \
    --format 'value(status.url)')

echo "Service deployed at: $SERVICE_URL"
```

## Deployment Szkriptek

### 1. Gyors Deployment

```bash
# Egyszerű deployment egyetlen paranccsal
./gcp/deploy.sh
```

### 2. Cloud Build használata

```bash
# Cloud Build submission
gcloud builds submit --config cloudbuild.yaml
```

### 3. Monitoring

```bash
# Interaktív monitoring tool
./gcp/monitor.sh
```

Opciók:
1. Service details megjelenítése
2. Real-time logs
3. Metrikák
4. Health check
5. Traffic split
6. Revíziók listázása

## Tesztelés

### 1. Health Check

```bash
# Service URL lekérése
SERVICE_URL=$(gcloud run services describe wine-quality-mlops \
    --region europe-west1 \
    --format 'value(status.url)')

# Health check
curl $SERVICE_URL/health
```

### 2. API Dokumentáció

```bash
# FastAPI Swagger UI
open $SERVICE_URL/docs

# Vagy böngészőben
echo "API Documentation: $SERVICE_URL/docs"
```

### 3. Prediction Test

```bash
# POST request prediction endpoint-ra
curl -X POST "$SERVICE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11,
    "total_sulfur_dioxide": 34,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }'
```

Várt válasz:
```json
{
  "prediction": 5.2,
  "model_version": "1.0.0",
  "prediction_time": "2025-11-11T12:00:00Z"
}
```

## Monitoring és Logging

### 1. Cloud Run Logs

```bash
# Real-time logs
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=wine-quality-mlops" \
    --project=$PROJECT_ID

# Logs az elmúlt 1 órából
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=wine-quality-mlops" \
    --limit 50 \
    --format json
```

### 2. Cloud Monitoring Dashboard

```bash
# Nyisd meg a Cloud Console Monitoring oldalát
open "https://console.cloud.google.com/run/detail/$REGION/wine-quality-mlops/metrics?project=$PROJECT_ID"
```

Metrikák:
- Request count
- Request latency
- Container CPU utilization
- Container memory utilization
- Container instance count

### 3. Custom Monitoring Script

```bash
# Interaktív monitoring
./gcp/monitor.sh
```

## CI/CD Workflow Részletek

### GitHub Actions Workflow

Fájl: `.github/workflows/gcp-cloud-run-deploy.yml`

**Trigger események:**
- Push to `main` vagy `master` branch
- Manual workflow dispatch

**Jobs:**

1. **build-and-deploy**
   - Checkout kód
   - GCP autentikáció
   - Docker build
   - Push Artifact Registry-be
   - Deploy Cloud Run-ra
   - Health check

2. **integration-tests**
   - Service URL lekérése
   - API endpoint tesztek
   - Prediction teszt

### Cloud Build (Alternatíva)

Fájl: `cloudbuild.yaml`

```bash
# Cloud Build trigger GitHub-ról
gcloud builds triggers create github \
    --repo-name=wine_quality_mlops \
    --repo-owner=your-github-username \
    --branch-pattern="^main$" \
    --build-config=cloudbuild.yaml

# Manuális build
gcloud builds submit --config cloudbuild.yaml
```

## Konfiguráció és Optimalizáció

### 1. Environment Variables

Cloud Run service környezeti változók:

```bash
gcloud run services update wine-quality-mlops \
    --region $REGION \
    --set-env-vars "MLFLOW_TRACKING_URI=file:/app/mlruns,ENVIRONMENT=production,LOG_LEVEL=INFO"
```

### 2. Resource Limits

```bash
# CPU és memória beállítása
gcloud run services update wine-quality-mlops \
    --region $REGION \
    --memory 4Gi \
    --cpu 4

# Timeout növelése
gcloud run services update wine-quality-mlops \
    --region $REGION \
    --timeout 600
```

### 3. Autoscaling

```bash
# Min és max instances
gcloud run services update wine-quality-mlops \
    --region $REGION \
    --min-instances 1 \
    --max-instances 20
```

### 4. Traffic Splitting (A/B Testing)

```bash
# Új revision deploy 20% traffic-kel
gcloud run deploy wine-quality-mlops \
    --image $REGION-docker.pkg.dev/$PROJECT_ID/wine-quality-mlops/wine-quality-mlops:v2 \
    --region $REGION \
    --no-traffic

# Traffic split beállítása
gcloud run services update-traffic wine-quality-mlops \
    --region $REGION \
    --to-revisions LATEST=20,wine-quality-mlops-00001-abc=80
```

## Költségek

### Cloud Run Díjszabás

**Free tier (havi):**
- 2 millió request
- 360,000 GB-seconds memory
- 180,000 vCPU-seconds

**Becsült költség (kis forgalom):**
- 1000 request/nap, 100ms válaszidő, 2GB RAM
- ~$5-10 USD/hónap

**Tippek költségcsökkentésre:**
- `min-instances: 0` (cold start)
- Request timeout optimalizálása
- Csak szükséges amikor használod
- Monitoring a váratlan költségekhez

## Troubleshooting

### 1. Deployment Failures

**Hiba: "Permission denied"**
```bash
# Service Account jogosultságok ellenőrzése
gcloud projects get-iam-policy $PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:serviceAccount:mlops-cloud-run-sa@$PROJECT_ID.iam.gserviceaccount.com"
```

**Hiba: "Image not found"**
```bash
# Artifact Registry image-ek listázása
gcloud artifacts docker images list $REGION-docker.pkg.dev/$PROJECT_ID/wine-quality-mlops

# Docker authentication újra
gcloud auth configure-docker $REGION-docker.pkg.dev
```

### 2. Runtime Errors

**Container startup timeout**
```bash
# Logs ellenőrzése
gcloud logging read "resource.type=cloud_run_revision" \
    --limit 20 \
    --format json

# Timeout növelése
gcloud run services update wine-quality-mlops \
    --region $REGION \
    --timeout 300
```

**Out of memory**
```bash
# Memória növelése
gcloud run services update wine-quality-mlops \
    --region $REGION \
    --memory 4Gi
```

### 3. Performance Issues

```bash
# Cold start optimalizálás
gcloud run services update wine-quality-mlops \
    --region $REGION \
    --min-instances 1  # Warm instance

# CPU növelése
gcloud run services update wine-quality-mlops \
    --region $REGION \
    --cpu 4
```

## Cleanup (Erőforrások Törlése)

### Automatikus Cleanup

```bash
# Minden GCP erőforrás törlése
./gcp/cleanup.sh
```

### Manuális Cleanup

```bash
# Cloud Run service törlése
gcloud run services delete wine-quality-mlops --region $REGION

# Artifact Registry repository törlése
gcloud artifacts repositories delete wine-quality-mlops --location=$REGION

# Service Account törlése
gcloud iam service-accounts delete mlops-cloud-run-sa@$PROJECT_ID.iam.gserviceaccount.com
```

## Best Practices

### 1. Security

- ✓ Ne commitáld a `gcp-sa-key.json` fájlt
- ✓ Használj GitHub Secrets-et
- ✓ Service Account jogosultságok minimalizálása (principle of least privilege)
- ✓ Artifact Registry private repository

### 2. Reliability

- ✓ Health check endpoint implementálása
- ✓ Graceful shutdown
- ✓ Error handling és logging
- ✓ Retry mechanizmusok

### 3. Performance

- ✓ Docker image optimalizálás (multi-stage build)
- ✓ Cold start csökkentés (min-instances beállítás)
- ✓ Model caching
- ✓ Async endpoints nagy számítási igényhez

### 4. Monitoring

- ✓ Cloud Monitoring alerts beállítása
- ✓ Error rate tracking
- ✓ Latency monitoring
- ✓ Cost monitoring

## Következő Lépések

### Production-ready Fejlesztések

1. **Secrets Management**
   - Google Secret Manager integrálása
   - API kulcsok biztonságos tárolása

2. **Database Integration**
   - Cloud SQL PostgreSQL
   - MLflow tracking backend
   - Model registry

3. **Advanced Monitoring**
   - Prometheus metrics export
   - Grafana dashboards
   - Alerting (PagerDuty, Slack)

4. **Load Balancing**
   - Cloud Load Balancer
   - CDN (Cloud CDN)
   - Global deployment

5. **ML-Specific Features**
   - Model versioning strategy
   - A/B testing framework
   - Feature store integration
   - Drift monitoring

## Referenciák

- [GCP Cloud Run Documentation](https://cloud.google.com/run/docs)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [MLOps Best Practices](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

## Support

Ha kérdésed van:
1. Ellenőrizd a [Troubleshooting](#troubleshooting) szekciót
2. Nézd meg a GCP Cloud Run dokumentációt
3. Nyiss issue-t a GitHub repository-ban

---

**Készítve**: 2025. november
**Verzió**: 1.0
**Technológiák**: GCP Cloud Run, GitHub Actions, Docker, FastAPI, MLflow
