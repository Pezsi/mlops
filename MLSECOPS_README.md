# MLSecOps Security Module

Ez a modul átfogó ML biztonsági funkciókat biztosít a Wine Quality MLOps projekthez.

## Tartalomjegyzék

1. [Áttekintés](#áttekintés)
2. [Telepítés](#telepítés)
3. [Modulok](#modulok)
   - [Adat- és Modellmérgezés Detektálás (Lecke 113)](#adat--és-modellmérgezés-detektálás)
   - [Vertex AI Explainable AI & Monitoring (Lecke 114)](#vertex-ai-explainable-ai--monitoring)
   - [Modellrobosztusság Tesztelés (Lecke 115-116)](#modellrobosztusság-tesztelés)
   - [Függőség Biztonsági Ellenőrzés (Lecke 117-120)](#függőség-biztonsági-ellenőrzés)
   - [Külső Modell Auditálás (Lecke 121-122)](#külső-modell-auditálás)
4. [Használati Példák](#használati-példák)
5. [CI/CD Integráció](#cicd-integráció)
6. [Best Practices](#best-practices)

---

## Áttekintés

A MLSecOps modul az OWASP ML Security Top 10 alapján készült, és a következő biztonsági területeket fedi le:

| Lecke | Téma | Modul |
|-------|------|-------|
| 113 | Adat- és modellmérgezés vizsgálata | `security/poisoning/` |
| 114 | Vertex AI Explainable AI és Model Monitoring | `security/vertex_ai/` |
| 115-116 | Modellrobosztusság tesztelése (CleverHans, ART) | `security/robustness/` |
| 117-118 | Cloud Build + Container Analysis | `security/dependency_audit/` |
| 119-120 | Dependabot és Snyk | `.github/dependabot.yml` |
| 121-122 | Külső modellek auditálása (Hugging Face) | `security/model_audit/` |

---

## Telepítés

### Alap telepítés
```bash
pip install -r requirements.txt
```

### Teljes biztonsági toolkit
```bash
pip install -r requirements-security.txt
```

### Opcionális függőségek
```bash
# SHAP és LIME az explainability-hez
pip install shap lime

# Adversarial Robustness Toolbox
pip install adversarial-robustness-toolbox

# Hugging Face integráció
pip install huggingface_hub

# Biztonsági scannerek
pip install safety pip-audit bandit
```

---

## Modulok

### Adat- és Modellmérgezés Detektálás

**Fájlok:**
- `security/poisoning/data_poisoning_detector.py`
- `security/poisoning/model_poisoning_detector.py`

**Funkciók:**

#### DataPoisoningDetector
```python
from security.poisoning.data_poisoning_detector import DataPoisoningDetector

# Inicializálás
detector = DataPoisoningDetector(
    contamination=0.1,      # Várt outlier arány
    z_score_threshold=3.0,  # Z-score küszöb
    iqr_multiplier=1.5      # IQR szorzó
)

# Baseline statisztikák
baseline = detector.compute_baseline_statistics(X_train, y_train)

# Outlier detektálás
zscore_mask, _ = detector.detect_outliers_zscore(X)
iqr_mask, _ = detector.detect_outliers_iqr(X)
iso_mask, scores = detector.detect_outliers_isolation_forest(X)
lof_mask, scores = detector.detect_outliers_lof(X)

# Label flipping detektálás
suspicious_mask, report = detector.detect_label_flipping(X, y)

# Backdoor pattern detektálás
backdoor_report = detector.detect_backdoor_patterns(X, y)

# Teljes elemzés
full_report = detector.run_full_analysis(X, y)
print(f"Risk Level: {full_report['overall_risk_assessment']['risk_level']}")
```

#### ModelPoisoningDetector
```python
from security.poisoning.model_poisoning_detector import ModelPoisoningDetector

# Inicializálás
detector = ModelPoisoningDetector(
    reference_model=clean_model,  # Opcionális referencia modell
    performance_threshold=0.1
)

# Modell fingerprint
fingerprint = detector.compute_model_fingerprint(model)

# Integritás ellenőrzés
is_valid = detector.verify_model_integrity(model, expected_fingerprint)

# Súly elemzés
weight_report = detector.analyze_weight_distribution(model)

# Backdoor trigger detektálás
backdoor_report = detector.detect_backdoor_triggers(model, X, y)

# Teljes elemzés
report = detector.run_full_analysis(model, X, y, is_classifier=True)
```

---

### Vertex AI Explainable AI & Monitoring

**Fájlok:**
- `security/vertex_ai/explainable_ai.py`
- `security/vertex_ai/model_monitoring.py`

#### ExplainableAI
```python
from security.vertex_ai.explainable_ai import ExplainableAI

# Inicializálás
explainer = ExplainableAI(
    model=trained_model,
    feature_names=feature_names,
    task_type="classification"  # vagy "regression"
)

# SHAP inicializálás és magyarázat
explainer.initialize_shap(X_train)
shap_report = explainer.explain_shap(X_test)
print("Top features:", shap_report["top_features"])

# LIME magyarázat egyetlen mintára
explainer.initialize_lime(X_train)
lime_report = explainer.explain_lime(X_test.iloc[0])

# Permutation importance
perm_report = explainer.compute_permutation_importance(X_test, y_test)

# Vertex AI konfiguráció generálás
vertex_config = explainer.generate_vertex_ai_config(X_train)
```

#### ModelMonitor
```python
from security.vertex_ai.model_monitoring import ModelMonitor

# Inicializálás
monitor = ModelMonitor(
    model=trained_model,
    feature_names=feature_names,
    task_type="classification",
    window_size=1000,
    drift_threshold=0.1
)

# Baseline beállítása
baseline = monitor.set_baseline(X_train, y_train)

# Predikciók logolása (éles környezetben)
for batch in production_data:
    predictions = model.predict(batch)
    status = monitor.log_prediction(batch, predictions)

    if status["drift_detected"]:
        print("ALERT: Drift detected!")
        for alert in status["alerts"]:
            print(f"  - {alert['type']}: {alert['severity']}")

# Drift ellenőrzés
drift_report = monitor.check_drift()

# Teljesítmény értékelés (ha van ground truth)
perf_report = monitor.evaluate_performance()

# Összefoglaló
summary = monitor.get_monitoring_summary()
```

---

### Modellrobosztusság Tesztelés

**Fájl:** `security/robustness/adversarial_tester.py`

```python
from security.robustness.adversarial_tester import AdversarialTester

# Inicializálás
tester = AdversarialTester(
    model=trained_model,
    feature_names=feature_names,
    task_type="classification"
)

# ART inicializálás (ha elérhető)
tester.initialize_art(X_train.values)

# Perturbációs támadás
perturb_report = tester.perturbation_attack(
    X_test, y_test,
    epsilon_values=[0.01, 0.05, 0.1, 0.2]
)

# FGSM támadás
fgsm_report = tester.fgsm_attack(X_test, y_test, epsilon=0.1)

# PGD támadás
pgd_report = tester.pgd_attack(
    X_test, y_test,
    epsilon=0.1,
    max_iter=40
)

# Feature importance támadás
fi_report = tester.feature_importance_attack(
    X_test, y_test,
    top_k_features=3
)

# Boundary támadás (csak classification)
boundary_report = tester.boundary_attack(X_test, y_test)

# Teljes robusztusság teszt
full_report = tester.run_full_robustness_test(X_test, y_test)
print(f"Robustness Grade: {full_report['overall_robustness']['grade']}")
```

---

### Függőség Biztonsági Ellenőrzés

**Fájl:** `security/dependency_audit/dependency_scanner.py`

```python
from security.dependency_audit.dependency_scanner import DependencyScanner, ContainerScanner

# Dependency scanning
scanner = DependencyScanner(
    project_path=".",
    requirements_file="requirements.txt"
)

# Safety scan
safety_report = scanner.scan_with_safety()

# pip-audit scan
audit_report = scanner.scan_with_pip_audit()

# Typosquatting ellenőrzés
typo_report = scanner.check_for_typosquatting()

# Licenc compliance
license_report = scanner.check_license_compliance(
    allowed_licenses=["MIT", "Apache-2.0", "BSD-3-Clause"]
)

# Elavult csomagok
outdated_report = scanner.check_outdated_packages()

# Teljes scan
full_report = scanner.run_full_scan()
print(f"Overall Risk: {full_report['summary']['overall_risk']}")

# Container scanning
container_scanner = ContainerScanner("wine-quality-mlops:latest")
trivy_report = container_scanner.scan_with_trivy()
```

---

### Külső Modell Auditálás

**Fájl:** `security/model_audit/external_model_validator.py`

```python
from security.model_audit.external_model_validator import ExternalModelValidator

# Inicializálás (opcionális HF token)
validator = ExternalModelValidator(
    hf_token="hf_xxx"  # vagy HF_TOKEN env var
)

# Hugging Face modell validálás
report = validator.validate_huggingface_model(
    model_id="bert-base-uncased",
    check_security=True,
    check_license=True,
    check_card=True
)

print(f"Status: {report['overall_status']}")
print(f"Trusted: {report['validation_results']['provenance']['is_trusted']}")
print(f"License OK: {report['validation_results']['license']['is_compliant']}")

# Pickle fájl biztonsági scan
pickle_report = validator.scan_pickle_file("model.pkl")

# Modell signature ellenőrzés
verify_report = validator.verify_model_signature(
    "model.pkl",
    expected_hash="abc123..."
)

# Helyi modell auditálás
audit_report = validator.audit_local_model("./models/")

# Attestation dokumentum generálás
attestation = validator.generate_model_attestation(
    model_path="./models/model.pkl",
    model_id="wine-quality-rf",
    validation_report=report
)
```

---

## Használati Példák

### Teljes Biztonsági Pipeline

```python
"""
Komplett MLSecOps pipeline a Wine Quality modellhez.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from security.poisoning.data_poisoning_detector import DataPoisoningDetector
from security.poisoning.model_poisoning_detector import ModelPoisoningDetector
from security.robustness.adversarial_tester import AdversarialTester
from security.vertex_ai.model_monitoring import ModelMonitor

# 1. Adat betöltés
df = pd.read_csv("data/wine_quality.csv")
X = df.drop("quality", axis=1)
y = df["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Adat mérgezés ellenőrzés
print("=" * 50)
print("1. Data Poisoning Detection")
print("=" * 50)

data_detector = DataPoisoningDetector()
data_report = data_detector.run_full_analysis(X_train, y_train)

print(f"Risk Level: {data_report['overall_risk_assessment']['risk_level']}")
print(f"Outliers found: {data_report['outlier_detection']['consensus']['n_outliers']}")

if data_report['overall_risk_assessment']['risk_level'] == "HIGH":
    print("⚠️  WARNING: High risk of data poisoning!")
    # Implement data cleaning

# 3. Modell tanítás
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Modell mérgezés ellenőrzés
print("\n" + "=" * 50)
print("2. Model Poisoning Detection")
print("=" * 50)

model_detector = ModelPoisoningDetector()
model_report = model_detector.run_full_analysis(model, X_test, y_test)

print(f"Risk Level: {model_report['overall_risk_assessment']['risk_level']}")
print(f"Model Fingerprint: {model_report['model_info']['fingerprint'][:16]}...")

# 5. Robusztusság teszt
print("\n" + "=" * 50)
print("3. Robustness Testing")
print("=" * 50)

tester = AdversarialTester(
    model=model,
    feature_names=X.columns.tolist(),
    task_type="classification"
)
robustness_report = tester.run_full_robustness_test(X_test, y_test, n_samples=100)

print(f"Robustness Grade: {robustness_report['overall_robustness']['grade']}")
print(f"Average Score: {robustness_report['overall_robustness']['average_score']:.3f}")

# 6. Monitoring beállítás
print("\n" + "=" * 50)
print("4. Setting up Production Monitoring")
print("=" * 50)

monitor = ModelMonitor(
    model=model,
    feature_names=X.columns.tolist(),
    task_type="classification"
)
baseline = monitor.set_baseline(X_train, y_train)

print(f"Baseline Accuracy: {baseline['prediction_stats']['accuracy']:.3f}")
print("Monitoring configured and ready!")

# 7. Összefoglaló
print("\n" + "=" * 50)
print("SECURITY ASSESSMENT SUMMARY")
print("=" * 50)

assessment = {
    "data_poisoning": data_report['overall_risk_assessment']['risk_level'],
    "model_poisoning": model_report['overall_risk_assessment']['risk_level'],
    "robustness": robustness_report['overall_robustness']['grade']
}

for check, result in assessment.items():
    status = "✅" if "LOW" in result or "A" in result or "B" in result else "⚠️"
    print(f"  {status} {check}: {result}")
```

---

## CI/CD Integráció

### GitHub Actions Workflow

A projekt tartalmaz egy teljes biztonsági scan workflow-t: `.github/workflows/security-scan.yml`

**Futtatott ellenőrzések:**
- 🔍 Dependency vulnerability scanning (Safety, pip-audit)
- 🐳 Container security scanning (Trivy)
- 🔐 Secret detection (Gitleaks, TruffleHog)
- 📝 Static code analysis (Bandit, CodeQL, Semgrep)
- 🤖 ML-specific security checks (pickle scanning)

### Dependabot konfiguráció

A `.github/dependabot.yml` automatikusan:
- Heti rendszerességgel ellenőrzi a függőségeket
- Pull request-eket nyit a frissítésekhez
- Csoportosítja a hasonló frissítéseket

### Snyk integráció

A `.snyk` fájl konfigurálja a Snyk-ot a projekt számára.

---

## Best Practices

### 1. Adat Biztonság
```python
# Mindig ellenőrizd az adatokat betöltés után
detector = DataPoisoningDetector()
report = detector.run_full_analysis(X, y)

if report['overall_risk_assessment']['risk_level'] == "HIGH":
    # Ne használd az adatot tisztítás nélkül!
    raise SecurityException("Data poisoning detected")
```

### 2. Modell Integritás
```python
# Mentsd el a modell fingerprint-jét
fingerprint = model_detector.compute_model_fingerprint(model)
save_fingerprint_to_registry(model_id, fingerprint)

# Ellenőrizd betöltéskor
loaded_model = load_model("model.pkl")
if not model_detector.verify_model_integrity(loaded_model, expected_fingerprint):
    raise SecurityException("Model integrity check failed!")
```

### 3. Folyamatos Monitoring
```python
# Állíts be alerteket drift esetén
if monitor.check_drift()["drift_detected"]:
    send_alert("Model drift detected - retraining may be needed")
```

### 4. Külső Modellek
```python
# Mindig validáld a külső modelleket használat előtt
validator = ExternalModelValidator()
report = validator.validate_huggingface_model(model_id)

if report["overall_status"] != "PASS":
    logging.warning(f"Model validation issues: {report['risks']}")
```

---

## Tesztelés

```bash
# Biztonsági tesztek futtatása
pytest tests/test_security.py -v

# Teljes teszt suite
pytest tests/ -v

# Coverage report
pytest tests/test_security.py --cov=security --cov-report=html
```

---

## Referenciák

- [OWASP ML Security Top 10](https://owasp.org/www-project-machine-learning-security-top-10/)
- [Vertex AI Explainable AI](https://cloud.google.com/vertex-ai/docs/explainable-ai/overview)
- [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Hugging Face Security](https://huggingface.co/docs/hub/security)

---

## Támogatás

Kérdések vagy problémák esetén nyiss egy GitHub issue-t a projektben.
