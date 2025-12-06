# GCP Cloud Run - Quick Start Guide

## Gyors Telepítés (5 perc)

### 1. Előfeltételek

```bash
# Google Cloud SDK telepítése (ha még nincs)
# macOS
brew install google-cloud-sdk

# Linux
curl https://sdk.cloud.google.com | bash

# gcloud init
gcloud init
```

### 2. GCP Setup - Automatikus

```bash
# Állítsd be a környezeti változókat
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="europe-west1"

# Futtasd a setup szkriptet
./gcp/setup-gcp.sh
```

### 3. GitHub Secrets Beállítása

A szkript generál egy `gcp-sa-key.json` fájlt.

**GitHub Repository → Settings → Secrets → New secret:**

1. `GCP_PROJECT_ID` → your-project-id
2. `GCP_SA_KEY` → `cat gcp-sa-key.json` teljes kimenete

### 4. Deploy

**Opció A: GitHub Actions (Automatikus)**
```bash
git add .
git commit -m "Add GCP deployment"
git push origin main
```

**Opció B: Manuális Deployment**
```bash
./gcp/deploy.sh
```

**Opció C: Cloud Build**
```bash
gcloud builds submit --config cloudbuild.yaml
```

### 5. Tesztelés

```bash
# Service URL lekérése
SERVICE_URL=$(gcloud run services describe wine-quality-mlops \
    --region europe-west1 \
    --format 'value(status.url)')

# Health check
curl $SERVICE_URL/health

# API docs
open $SERVICE_URL/docs

# Prediction test
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

## Hasznos Parancsok

### Monitoring

```bash
# Real-time logs
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=wine-quality-mlops"

# Service details
gcloud run services describe wine-quality-mlops --region europe-west1

# Interaktív monitoring
./gcp/monitor.sh
```

### Update

```bash
# Újra-deployment (új kód után)
./gcp/deploy.sh

# Environment variable update
gcloud run services update wine-quality-mlops \
    --region europe-west1 \
    --set-env-vars "NEW_VAR=value"
```

### Cleanup

```bash
# Minden erőforrás törlése
./gcp/cleanup.sh
```

## Terminál Parancsok Összefoglalója

### Setup (egyszeri)

```bash
# 1. GCP projekt beállítása
export GCP_PROJECT_ID="your-project-id"
gcloud config set project $GCP_PROJECT_ID

# 2. API-k engedélyezése
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# 3. Artifact Registry létrehozása
gcloud artifacts repositories create wine-quality-mlops \
    --repository-format=docker \
    --location=europe-west1

# 4. Docker auth
gcloud auth configure-docker europe-west1-docker.pkg.dev
```

### Build & Deploy (minden deploy alkalmával)

```bash
# 1. Build
docker build -f Dockerfile.cloudrun \
    -t europe-west1-docker.pkg.dev/$GCP_PROJECT_ID/wine-quality-mlops/wine-quality-mlops:latest .

# 2. Push
docker push europe-west1-docker.pkg.dev/$GCP_PROJECT_ID/wine-quality-mlops/wine-quality-mlops:latest

# 3. Deploy
gcloud run deploy wine-quality-mlops \
    --image europe-west1-docker.pkg.dev/$GCP_PROJECT_ID/wine-quality-mlops/wine-quality-mlops:latest \
    --region europe-west1 \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2
```

### Monitoring

```bash
# Logs
gcloud logging tail "resource.type=cloud_run_revision"

# Service info
gcloud run services list
gcloud run services describe wine-quality-mlops --region europe-west1

# Metrics
gcloud monitoring time-series list --filter="resource.type=cloud_run_revision"
```

### Troubleshooting

```bash
# Revisions listázása
gcloud run revisions list --service wine-quality-mlops --region europe-west1

# Traffic routing
gcloud run services update-traffic wine-quality-mlops --region europe-west1 --to-latest

# Service delete és újra-deploy
gcloud run services delete wine-quality-mlops --region europe-west1
./gcp/deploy.sh
```

## Gyakori Hibák

### "Permission denied"
```bash
# Service account jogok ellenőrzése
gcloud projects get-iam-policy $GCP_PROJECT_ID
```

### "Image not found"
```bash
# Artifact Registry images
gcloud artifacts docker images list europe-west1-docker.pkg.dev/$GCP_PROJECT_ID/wine-quality-mlops
```

### "Container startup timeout"
```bash
# Timeout növelése
gcloud run services update wine-quality-mlops --region europe-west1 --timeout 300
```

### "Out of memory"
```bash
# Memória növelése
gcloud run services update wine-quality-mlops --region europe-west1 --memory 4Gi
```

## Költségek

**Free tier:**
- 2M request/hó
- 360,000 GB-seconds memory/hó

**Becsült költség:**
- Kis forgalom (~1000 request/nap): $5-10/hó
- Közepes forgalom (~10,000 request/nap): $20-50/hó

**Költségcsökkentés:**
```bash
# Min instances = 0 (cold start, de ingyenes idle-ben)
gcloud run services update wine-quality-mlops --region europe-west1 --min-instances 0

# Service leállítása (ha nem használod)
gcloud run services delete wine-quality-mlops --region europe-west1
```

---

**Részletes dokumentáció**: Lásd [GCP_DEPLOYMENT_GUIDE.md](../GCP_DEPLOYMENT_GUIDE.md)
