# Wine Quality MLOps Project

Production-ready machine learning pipeline for Wine Quality prediction.

## Project Structure

```
wine_quality_mlops/
├── data/               # Data loading and splitting
├── src/                # Source code (preprocessing, training, evaluation)
├── tests/              # Unit and integration tests
├── models/             # Saved models
├── logs/               # Application logs
├── notebooks/          # Jupyter notebooks and prototypes
├── config.py           # Configuration and hyperparameters
├── main.py             # Main entry point
├── app.py              # REST API (FastAPI)
├── requirements.txt    # Python dependencies
└── pyproject.toml      # Black formatter configuration
```

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Train model
python main.py

# Run tests
pytest tests/

# Format code
black .

# Check code style
flake8 .

# Start API server
uvicorn app:app --reload
```

## MLOps Features

- ✅ Production-ready code structure
- ✅ PEP8 compliant (Black + Flake8)
- ✅ Type hints
- 🔄 MLflow experiment tracking (in progress)
- 🔄 Automated tests (in progress)
- 🔄 REST API (in progress)
- 🔄 Docker support (planned)

## Model Performance

- **Algorithm**: Random Forest Regressor
- **R² Score**: ~0.47
- **MSE**: ~0.34
- **Features**: 11 wine quality features
- **Target**: Wine quality (0-10 scale)
