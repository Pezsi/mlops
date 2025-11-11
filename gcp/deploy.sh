#!/bin/bash

# Manual deployment script for GCP Cloud Run
# This script builds and deploys the Wine Quality MLOps service to Cloud Run

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-europe-west1}"
SERVICE_NAME="wine-quality-mlops"
IMAGE_NAME="wine-quality-mlops"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Wine Quality MLOps - Deployment     ${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Check if PROJECT_ID is set
if [ -z "$PROJECT_ID" ]; then
    echo -e "${YELLOW}Enter your GCP Project ID:${NC}"
    read -r PROJECT_ID
fi

echo -e "${YELLOW}Project ID: $PROJECT_ID${NC}"
echo -e "${YELLOW}Region: $REGION${NC}"
echo -e "${YELLOW}Service Name: $SERVICE_NAME${NC}"
echo ""

# Set project
echo -e "${YELLOW}[1/5] Setting GCP project...${NC}"
gcloud config set project "$PROJECT_ID"

# Build Docker image
echo -e "${YELLOW}[2/5] Building Docker image...${NC}"
IMAGE_TAG="$REGION-docker.pkg.dev/$PROJECT_ID/$SERVICE_NAME/$IMAGE_NAME:latest"

docker build -f Dockerfile.cloudrun -t "$IMAGE_TAG" .

echo -e "${GREEN}✓ Docker image built successfully${NC}"

# Push to Artifact Registry
echo -e "${YELLOW}[3/5] Pushing image to Artifact Registry...${NC}"
docker push "$IMAGE_TAG"
echo -e "${GREEN}✓ Image pushed successfully${NC}"

# Deploy to Cloud Run
echo -e "${YELLOW}[4/5] Deploying to Cloud Run...${NC}"
gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_TAG" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --min-instances 0 \
    --set-env-vars "MLFLOW_TRACKING_URI=file:/app/mlruns,ENVIRONMENT=production" \
    --labels "app=wine-quality-mlops,managed-by=manual-deploy"

echo -e "${GREEN}✓ Deployed successfully${NC}"

# Get service URL
echo -e "${YELLOW}[5/5] Getting service URL...${NC}"
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --region "$REGION" \
    --format 'value(status.url)')

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Deployment Successful! 🚀          ${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "${YELLOW}Service URL:${NC} $SERVICE_URL"
echo ""
echo -e "${YELLOW}Quick Links:${NC}"
echo "  API Documentation: $SERVICE_URL/docs"
echo "  Health Check:      $SERVICE_URL/health"
echo ""
echo -e "${YELLOW}Test the API:${NC}"
echo "  curl $SERVICE_URL/health"
echo ""
echo "  curl -X POST \"$SERVICE_URL/predict\" \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{"
echo "      \"fixed_acidity\": 7.4,"
echo "      \"volatile_acidity\": 0.7,"
echo "      \"citric_acid\": 0.0,"
echo "      \"residual_sugar\": 1.9,"
echo "      \"chlorides\": 0.076,"
echo "      \"free_sulfur_dioxide\": 11,"
echo "      \"total_sulfur_dioxide\": 34,"
echo "      \"density\": 0.9978,"
echo "      \"pH\": 3.51,"
echo "      \"sulphates\": 0.56,"
echo "      \"alcohol\": 9.4"
echo "    }'"
echo ""
