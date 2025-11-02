#!/bin/bash

# MLOps Pipeline Setup Script
# This script sets up the complete MLOps environment

set -e

echo "============================================"
echo "  Wine Quality MLOps Pipeline Setup"
echo "============================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker found${NC}"
echo -e "${GREEN}✓ Docker Compose found${NC}"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p airflow/dags
mkdir -p airflow/logs
mkdir -p airflow/plugins
mkdir -p airflow/utils
mkdir -p airflow/api
mkdir -p airflow/db
mkdir -p data/incoming
mkdir -p data/processed
mkdir -p models
mkdir -p mlruns

echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo -e "${YELLOW}Please edit .env file with your actual credentials before starting services${NC}"
    echo ""
fi

# Build Docker images
echo "Building Docker images..."
echo "This may take several minutes on first run..."
docker-compose build

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Docker images built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build Docker images${NC}"
    exit 1
fi
echo ""

# Start services
echo "Starting services..."
docker-compose up -d

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Services started successfully${NC}"
else
    echo -e "${RED}✗ Failed to start services${NC}"
    exit 1
fi
echo ""

# Wait for services to be healthy
echo "Waiting for services to be healthy..."
sleep 10

# Check service health
echo ""
echo "Service Status:"
echo "───────────────────────────────────────────"
docker-compose ps
echo ""

# Show access information
echo "============================================"
echo "  MLOps Pipeline is Ready!"
echo "============================================"
echo ""
echo "Access the services at:"
echo ""
echo "  🌐 Airflow UI:        http://localhost:8081"
echo "     Username: admin"
echo "     Password: admin"
echo ""
echo "  🔬 MLflow UI:         http://localhost:5000"
echo ""
echo "  📊 Streamlit:         http://localhost:8501"
echo ""
echo "  🔗 Webhook API:       http://localhost:8080"
echo "     API Docs:          http://localhost:8080/docs"
echo "     API Key: (check .env)"
echo ""
echo "  🗄️  Metadata DB:      localhost:5433"
echo "     Database: mlops_metadata"
echo "     Username: mlops_user"
echo "     Password: mlops_password"
echo ""
echo "============================================"
echo ""
echo "Next Steps:"
echo "1. Configure .env file with your credentials"
echo "2. Restart services: docker-compose restart"
echo "3. Open Airflow UI and unpause DAGs"
echo "4. Check AIRFLOW_MLOPS_README.md for detailed documentation"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop:      docker-compose down"
echo ""
echo -e "${GREEN}Setup completed successfully!${NC}"
