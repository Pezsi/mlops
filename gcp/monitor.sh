#!/bin/bash

# Monitoring script for GCP Cloud Run service
# This script provides real-time monitoring and logging for the deployed service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-europe-west1}"
SERVICE_NAME="wine-quality-mlops"

if [ -z "$PROJECT_ID" ]; then
    echo -e "${YELLOW}Enter your GCP Project ID:${NC}"
    read -r PROJECT_ID
fi

gcloud config set project "$PROJECT_ID"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Cloud Run Service Monitoring        ${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Function to show menu
show_menu() {
    echo -e "${YELLOW}Select monitoring option:${NC}"
    echo "  1) Show service details"
    echo "  2) View real-time logs"
    echo "  3) Show recent metrics"
    echo "  4) Test service health"
    echo "  5) Show traffic split"
    echo "  6) List revisions"
    echo "  7) Exit"
    echo ""
}

# Function to show service details
show_service_details() {
    echo -e "${BLUE}[Service Details]${NC}"
    gcloud run services describe "$SERVICE_NAME" --region "$REGION"
    echo ""
}

# Function to view logs
view_logs() {
    echo -e "${BLUE}[Real-time Logs - Press Ctrl+C to stop]${NC}"
    gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME" \
        --project="$PROJECT_ID"
}

# Function to show metrics
show_metrics() {
    echo -e "${BLUE}[Recent Metrics]${NC}"

    SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
        --region "$REGION" \
        --format 'value(status.url)')

    echo -e "${GREEN}Service URL:${NC} $SERVICE_URL"

    # Get revision info
    LATEST_REVISION=$(gcloud run services describe "$SERVICE_NAME" \
        --region "$REGION" \
        --format 'value(status.latestReadyRevisionName)')

    echo -e "${GREEN}Latest Revision:${NC} $LATEST_REVISION"

    # Request count (last hour)
    echo -e "\n${YELLOW}Request count (last hour):${NC}"
    gcloud monitoring time-series list \
        --filter="resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME" \
        --format="table(metric.type)" \
        --project="$PROJECT_ID" 2>/dev/null || echo "Metrics not available yet"

    echo ""
}

# Function to test service health
test_health() {
    echo -e "${BLUE}[Health Check]${NC}"

    SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
        --region "$REGION" \
        --format 'value(status.url)')

    echo -e "${YELLOW}Testing: $SERVICE_URL/health${NC}"

    if curl -f -s "$SERVICE_URL/health" > /dev/null; then
        echo -e "${GREEN}✓ Service is healthy${NC}"

        # Test docs endpoint
        echo -e "\n${YELLOW}Testing: $SERVICE_URL/docs${NC}"
        if curl -f -s "$SERVICE_URL/docs" > /dev/null; then
            echo -e "${GREEN}✓ API documentation is accessible${NC}"
        else
            echo -e "${RED}✗ API documentation is not accessible${NC}"
        fi
    else
        echo -e "${RED}✗ Service health check failed${NC}"
    fi

    echo ""
}

# Function to show traffic split
show_traffic() {
    echo -e "${BLUE}[Traffic Split]${NC}"
    gcloud run services describe "$SERVICE_NAME" \
        --region "$REGION" \
        --format="table(status.traffic.revisionName,status.traffic.percent)"
    echo ""
}

# Function to list revisions
list_revisions() {
    echo -e "${BLUE}[Service Revisions]${NC}"
    gcloud run revisions list \
        --service "$SERVICE_NAME" \
        --region "$REGION" \
        --format="table(metadata.name,status.conditions[0].status,metadata.creationTimestamp)"
    echo ""
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice [1-7]: " choice
    echo ""

    case $choice in
        1)
            show_service_details
            ;;
        2)
            view_logs
            ;;
        3)
            show_metrics
            ;;
        4)
            test_health
            ;;
        5)
            show_traffic
            ;;
        6)
            list_revisions
            ;;
        7)
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option. Please try again.${NC}"
            echo ""
            ;;
    esac

    read -p "Press Enter to continue..."
    echo ""
done
