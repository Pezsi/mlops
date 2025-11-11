#!/bin/bash

# Cleanup script for GCP resources
# This script removes all GCP resources created for the Wine Quality MLOps project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-europe-west1}"
SERVICE_NAME="wine-quality-mlops"
SERVICE_ACCOUNT_NAME="mlops-cloud-run-sa"
ARTIFACT_REPO_NAME="wine-quality-mlops"

echo -e "${RED}======================================${NC}"
echo -e "${RED}  WARNING: Resource Cleanup           ${NC}"
echo -e "${RED}======================================${NC}"
echo ""
echo -e "${YELLOW}This will DELETE the following resources:${NC}"
echo "  - Cloud Run service: $SERVICE_NAME"
echo "  - Artifact Registry repository: $ARTIFACT_REPO_NAME"
echo "  - Service account: $SERVICE_ACCOUNT_NAME"
echo ""
echo -e "${RED}This action CANNOT be undone!${NC}"
echo ""

read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo -e "${GREEN}Cleanup cancelled.${NC}"
    exit 0
fi

if [ -z "$PROJECT_ID" ]; then
    echo -e "${YELLOW}Enter your GCP Project ID:${NC}"
    read -r PROJECT_ID
fi

gcloud config set project "$PROJECT_ID"

echo ""
echo -e "${YELLOW}[1/3] Deleting Cloud Run service...${NC}"

if gcloud run services describe "$SERVICE_NAME" --region "$REGION" &> /dev/null; then
    gcloud run services delete "$SERVICE_NAME" \
        --region "$REGION" \
        --quiet
    echo -e "${GREEN}✓ Cloud Run service deleted${NC}"
else
    echo -e "${YELLOW}Cloud Run service not found (already deleted?)${NC}"
fi

echo ""
echo -e "${YELLOW}[2/3] Deleting Artifact Registry repository...${NC}"

if gcloud artifacts repositories describe "$ARTIFACT_REPO_NAME" \
    --location "$REGION" &> /dev/null; then

    # List images first
    echo -e "${YELLOW}Images in repository:${NC}"
    gcloud artifacts docker images list \
        "$REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO_NAME" \
        2>/dev/null || true

    gcloud artifacts repositories delete "$ARTIFACT_REPO_NAME" \
        --location "$REGION" \
        --quiet
    echo -e "${GREEN}✓ Artifact Registry repository deleted${NC}"
else
    echo -e "${YELLOW}Artifact Registry repository not found (already deleted?)${NC}"
fi

echo ""
echo -e "${YELLOW}[3/3] Deleting service account...${NC}"

SERVICE_ACCOUNT_EMAIL="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"

if gcloud iam service-accounts describe "$SERVICE_ACCOUNT_EMAIL" &> /dev/null; then
    # Remove IAM policy bindings first
    echo -e "${YELLOW}Removing IAM policy bindings...${NC}"

    gcloud projects remove-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
        --role="roles/run.admin" \
        --quiet 2>/dev/null || true

    gcloud projects remove-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
        --role="roles/storage.admin" \
        --quiet 2>/dev/null || true

    gcloud projects remove-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
        --role="roles/artifactregistry.reader" \
        --quiet 2>/dev/null || true

    # Delete service account
    gcloud iam service-accounts delete "$SERVICE_ACCOUNT_EMAIL" \
        --quiet
    echo -e "${GREEN}✓ Service account deleted${NC}"
else
    echo -e "${YELLOW}Service account not found (already deleted?)${NC}"
fi

# Cleanup local files
echo ""
echo -e "${YELLOW}Cleaning up local files...${NC}"
if [ -f "gcp-sa-key.json" ]; then
    read -p "Delete local service account key (gcp-sa-key.json)? (yes/no): " delete_key
    if [ "$delete_key" = "yes" ]; then
        rm gcp-sa-key.json
        echo -e "${GREEN}✓ Local key file deleted${NC}"
    fi
fi

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Cleanup Complete!                  ${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "${YELLOW}Summary:${NC}"
echo "  All GCP resources for Wine Quality MLOps have been removed."
echo ""
echo -e "${YELLOW}Note:${NC}"
echo "  The following APIs remain enabled (you may want to disable them):"
echo "    - Cloud Run API"
echo "    - Cloud Build API"
echo "    - Artifact Registry API"
echo ""
echo "  To disable APIs:"
echo "    gcloud services disable run.googleapis.com"
echo "    gcloud services disable cloudbuild.googleapis.com"
echo "    gcloud services disable artifactregistry.googleapis.com"
echo ""
