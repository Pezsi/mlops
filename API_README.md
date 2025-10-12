# Wine Quality Prediction REST APIs

Ez a dokumentum két teljes körű REST API implementációt tartalmaz a Wine Quality predikció számára.

## Létrehozott API-k

### 1. FastAPI alkalmazás (`fastapi_app.py`)
- **547 sor kód**
- Modern, async framework
- Automatikus OpenAPI (Swagger) dokumentáció
- Pydantic validáció
- Type hints minden végponton

### 2. Flask-RESTX alkalmazás (`flask_app.py`)
- **637 sor kód**
- Flask-alapú REST API
- Swagger UI integrációval
- Namespace-alapú szervezés
- Teljes API dokumentáció

## API Végpontok (mindkét API-ban)

### General Endpoints
- `GET /` - Üdvözlő üzenet
- `GET /health` - Health check
- `GET /features` - Feature lista

### Model Management
- `GET /models` - Elérhető modellek listája
- `GET /metrics/{model_name}` - Model metrikák (MLflow-ból)

### Prediction
- `POST /predict` - Egyedi wine quality predikció
- `POST /predict/batch` - Batch predikció
- `POST /model/predict` - Alias endpoint

### Training
- `POST /model/train` - Modell tanítás
  - Támogatott pipeline_type: `rf`, `gb`, `compare`
  - MLflow tracking integráció

## Példa használat

### 1. FastAPI indítása

```bash
# Telepítés
pip install -r requirements.txt

# Indítás
python fastapi_app.py
```

Elérhető: http://localhost:8000
Docs: http://localhost:8000/docs

### 2. Flask-RESTX indítása

```bash
# Indítás
python flask_app.py
```

Elérhető: http://localhost:8000
Docs: http://localhost:8000/docs/

## Példa kérések

### Predikció

```bash
curl -X POST "http://localhost:8000/predict" \
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

### Modell tanítás

```bash
curl -X POST "http://localhost:8000/model/train" \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline_type": "rf"
  }'
```

### Batch predikció

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "features_list": [
      {
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
      {
        "fixed_acidity": 8.1,
        "volatile_acidity": 0.8,
        "citric_acid": 0.1,
        "residual_sugar": 2.2,
        "chlorides": 0.08,
        "free_sulfur_dioxide": 15.0,
        "total_sulfur_dioxide": 40.0,
        "density": 0.998,
        "pH": 3.45,
        "sulphates": 0.6,
        "alcohol": 10.0
      }
    ],
    "model_name": "rf"
  }'
```

## Tesztek

Mindkét API-hoz teljes teszt lefedettség:

- **test_fastapi.py** - 298 sor, 15+ teszt
- **test_flask_app.py** - 320 sor, 15+ teszt

### Tesztek futtatása

```bash
# FastAPI tesztek
pytest tests/test_fastapi.py -v

# Flask-RESTX tesztek
pytest tests/test_flask_app.py -v

# Minden API teszt
pytest tests/test_*app*.py -v
```

## Funkciók

### ✓ MLflow Integráció
- Automatikus experiment tracking
- Model registry támogatás
- Metrics logging
- Model versioning

### ✓ Model Management
- Model cache startupkor
- Lazy loading
- Multiple model support (RF, GB)
- Model state management

### ✓ Validáció
- Input range validáció (Pydantic / Flask-RESTX)
- Type checking
- Error handling
- HTTP status kódok

### ✓ Dokumentáció
- Automatikus Swagger UI
- OpenAPI specifikáció
- Request/Response példák
- Endpoint leírások

## PEP8 Megfelelés

Minden fájl teljes PEP8 compliance-el:

```bash
flake8 fastapi_app.py flask_app.py tests/test_*app*.py --count
# 0 hibák
```

## Architektúra

```
fastapi_app.py (547 sor)
├── Pydantic models (Request/Response)
├── Helper functions (model loading, conversion)
├── API endpoints (async)
└── MLflow integration

flask_app.py (637 sor)
├── Flask-RESTX namespaces
│   ├── general (/, /health, /features)
│   ├── models (/models, /metrics)
│   ├── predict (/predict, /batch)
│   └── train (/model/train, /model/predict)
├── Flask-RESTX models (validation)
├── Helper functions
└── MLflow integration
```

## Összehasonlítás

| Feature | FastAPI | Flask-RESTX |
|---------|---------|-------------|
| Async támogatás | ✓ | ✗ |
| Type hints | Natív | Decorator-ok |
| Validáció | Pydantic | Flask-RESTX models |
| Docs | Automatikus | Automatikus |
| Teljesítmény | Gyorsabb | Jó |
| Namespace support | Tags | Namespaces |
| Kód méret | 547 sor | 637 sor |

## További fejlesztési lehetőségek

- [ ] Docker konténerizáció
- [ ] API authentication (JWT)
- [ ] Rate limiting
- [ ] CORS konfiguráció
- [ ] Database integráció
- [ ] Async MLflow logging
- [ ] Model A/B testing
- [ ] Prometheus metrics
- [ ] Logging aggregáció
- [ ] CI/CD pipeline

## License

MIT License
