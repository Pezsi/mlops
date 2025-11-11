#!/bin/bash

# GCP MLOps Setup Script
# This script sets up the necessary GCP resources for deploying the Wine Quality MLOps pipeline

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration variables
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-europe-west1}"
SERVICE_NAME="wine-quality-mlops"
SERVICE_ACCOUNT_NAME="mlops-cloud-run-sa"
ARTIFACT_REPO_NAME="wine-quality-mlops"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Wine Quality MLOps - GCP Setup      ${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Step 1: Check prerequisites
echo -e "${YELLOW}[1/8] Checking prerequisites...${NC}"

if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI is not installed.${NC}"
    echo "Install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo -e "${GREEN}✓ gcloud CLI is installed${NC}"

# Step 2: Set project ID
if [ -z "$PROJECT_ID" ]; then
    echo -e "${YELLOW}Enter your GCP Project ID:${NC}"
    read -r PROJECT_ID
fi

echo -e "${YELLOW}[2/8] Setting GCP project to: $PROJECT_ID${NC}"
gcloud config set project "$PROJECT_ID"

# Step 3: Enable required APIs
echo -e "${YELLOW}[3/8] Enabling required GCP APIs...${NC}"
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    containerregistry.googleapis.com \
    cloudresourcemanager.googleapis.com \
    iam.googleapis.com

echo -e "${GREEN}✓ APIs enabled${NC}"

# Step 4: Create Artifact Registry repository
echo -e "${YELLOW}[4/8] Creating Artifact Registry repository...${NC}"

# Check if repository already exists
if gcloud artifacts repositories describe "$ARTIFACT_REPO_NAME" \
    --location="$REGION" &> /dev/null; then
    echo -e "${GREEN}✓ Artifact Registry repository already exists${NC}"
else
    gcloud artifacts repositories create "$ARTIFACT_REPO_NAME" \
        --repository-format=docker \
        --location="$REGION" \
        --description="Docker repository for Wine Quality MLOps"
    echo -e "${GREEN}✓ Artifact Registry repository created${NC}"
fi

# Step 5: Create service account for Cloud Run
echo -e "${YELLOW}[5/8] Creating service account for Cloud Run...${NC}"

SERVICE_ACCOUNT_EMAIL="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"

# Check if service account already exists
if gcloud iam service-accounts describe "$SERVICE_ACCOUNT_EMAIL" &> /dev/null; then
    echo -e "${GREEN}✓ Service account already exists${NC}"
else
    gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
        --display-name="MLOps Cloud Run Service Account" \
        --description="Service account for Wine Quality MLOps Cloud Run service"
    echo -e "${GREEN}✓ Service account created${NC}"
fi

# Step 6: Grant necessary permissions to service account
echo -e "${YELLOW}[6/8] Granting permissions to service account...${NC}"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/run.admin" \
    --condition=None

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/storage.admin" \
    --condition=None

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/artifactregistry.reader" \
    --condition=None

echo -e "${GREEN}✓ Permissions granted${NC}"

# Step 7: Create service account key for GitHub Actions
echo -e "${YELLOW}[7/8] Creating service account key for GitHub Actions...${NC}"

KEY_FILE="gcp-sa-key.json"

if [ -f "$KEY_FILE" ]; then
    echo -e "${YELLOW}Warning: $KEY_FILE already exists. Skipping key creation.${NC}"
else
    gcloud iam service-accounts keys create "$KEY_FILE" \
        --iam-account="$SERVICE_ACCOUNT_EMAIL"
    echo -e "${GREEN}✓ Service account key created: $KEY_FILE${NC}"
    echo -e "${YELLOW}IMPORTANT: Add this key to GitHub Secrets as 'GCP_SA_KEY'${NC}"
fi

# Step 8: Configure Docker authentication
echo -e "${YELLOW}[8/8] Configuring Docker authentication for Artifact Registry...${NC}"
gcloud auth configure-docker "$REGION-docker.pkg.dev"
echo -e "${GREEN}✓ Docker authentication configured${NC}"

# Summary
echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Setup Complete!                    ${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "${YELLOW}Configuration Summary:${NC}"
echo "  Project ID:        $PROJECT_ID"
echo "  Region:            $REGION"
echo "  Service Name:      $SERVICE_NAME"
echo "  Artifact Repo:     $ARTIFACT_REPO_NAME"
echo "  Service Account:   $SERVICE_ACCOUNT_EMAIL"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Add the following secrets to your GitHub repository:"
echo "     - GCP_PROJECT_ID: $PROJECT_ID"
echo "     - GCP_SA_KEY: (contents of $KEY_FILE)"
echo ""
echo "  2. Review and update the region in .github/workflows/gcp-cloud-run-deploy.yml"
echo ""
echo "  3. Deploy manually using:"
echo "     ./gcp/deploy.sh"
echo ""
echo "  4. Or trigger GitHub Actions by pushing to main/master branch"
echo ""
echo -e "${GREEN}Happy deploying! 🚀${NC}"
