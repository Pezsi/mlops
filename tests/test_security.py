"""
Comprehensive Test Suite for MLSecOps Security Modules

Tests cover:
- Lecke 113: Data and Model Poisoning Detection
- Lecke 114: Explainable AI and Model Monitoring
- Lecke 115-116: Adversarial Robustness Testing
- Lecke 117-120: Dependency Security
- Lecke 121-122: External Model Auditing
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import tempfile
import os
from pathlib import Path

# Import security modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from security.poisoning.data_poisoning_detector import DataPoisoningDetector
from security.poisoning.model_poisoning_detector import ModelPoisoningDetector
from security.robustness.adversarial_tester import AdversarialTester
from security.vertex_ai.explainable_ai import ExplainableAI
from security.vertex_ai.model_monitoring import ModelMonitor
from security.dependency_audit.dependency_scanner import DependencyScanner
from security.model_audit.external_model_validator import ExternalModelValidator


# Fixtures
@pytest.fixture
def wine_data():
    """Load wine dataset for testing."""
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target)
    return X, y


@pytest.fixture
def wine_train_test_split(wine_data):
    """Split wine data into train and test sets."""
    X, y = wine_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def trained_classifier(wine_train_test_split):
    """Train a RandomForest classifier."""
    X_train, X_test, y_train, y_test = wine_train_test_split
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def trained_regressor(wine_train_test_split):
    """Train a RandomForest regressor."""
    X_train, X_test, y_train, y_test = wine_train_test_split
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model


# =============================================================================
# Lecke 113: Data Poisoning Detection Tests
# =============================================================================
class TestDataPoisoningDetector:
    """Tests for data poisoning detection module."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = DataPoisoningDetector()
        assert detector.contamination == 0.1
        assert detector.z_score_threshold == 3.0

    def test_compute_baseline_statistics(self, wine_data):
        """Test baseline statistics computation."""
        X, y = wine_data
        detector = DataPoisoningDetector()
        stats = detector.compute_baseline_statistics(X, y)

        assert "n_samples" in stats
        assert stats["n_samples"] == len(X)
        assert "feature_stats" in stats
        assert len(stats["feature_stats"]) == len(X.columns)

    def test_detect_outliers_zscore(self, wine_data):
        """Test Z-score outlier detection."""
        X, y = wine_data
        detector = DataPoisoningDetector(z_score_threshold=2.0)
        outlier_mask, feature_outliers = detector.detect_outliers_zscore(X)

        assert len(outlier_mask) == len(X)
        assert isinstance(feature_outliers, dict)

    def test_detect_outliers_iqr(self, wine_data):
        """Test IQR outlier detection."""
        X, y = wine_data
        detector = DataPoisoningDetector()
        outlier_mask, feature_outliers = detector.detect_outliers_iqr(X)

        assert len(outlier_mask) == len(X)
        assert outlier_mask.dtype == bool

    def test_detect_outliers_isolation_forest(self, wine_data):
        """Test Isolation Forest outlier detection."""
        X, y = wine_data
        detector = DataPoisoningDetector(contamination=0.1)
        outlier_mask, anomaly_scores = detector.detect_outliers_isolation_forest(X)

        assert len(outlier_mask) == len(X)
        assert len(anomaly_scores) == len(X)
        assert outlier_mask.sum() > 0  # Should detect some outliers

    def test_detect_outliers_lof(self, wine_data):
        """Test Local Outlier Factor detection."""
        X, y = wine_data
        detector = DataPoisoningDetector()
        outlier_mask, lof_scores = detector.detect_outliers_lof(X)

        assert len(outlier_mask) == len(X)
        assert len(lof_scores) == len(X)

    def test_detect_label_flipping(self, wine_data):
        """Test label flipping detection."""
        X, y = wine_data
        detector = DataPoisoningDetector()
        suspicious_mask, report = detector.detect_label_flipping(X, y)

        assert len(suspicious_mask) == len(X)
        assert "total_suspicious" in report
        assert "class_analysis" in report

    def test_detect_backdoor_patterns(self, wine_data):
        """Test backdoor pattern detection."""
        X, y = wine_data
        detector = DataPoisoningDetector()
        report = detector.detect_backdoor_patterns(X, y)

        assert "suspicious_patterns" in report
        assert "risk_score" in report
        assert 0 <= report["risk_score"] <= 1

    def test_compute_data_fingerprint(self, wine_data):
        """Test data fingerprint computation."""
        X, y = wine_data
        detector = DataPoisoningDetector()
        fingerprint = detector.compute_data_fingerprint(X, y)

        assert len(fingerprint) == 64  # SHA256 hex
        # Same data should give same fingerprint
        fingerprint2 = detector.compute_data_fingerprint(X, y)
        assert fingerprint == fingerprint2

    def test_run_full_analysis(self, wine_data):
        """Test full data poisoning analysis."""
        X, y = wine_data
        detector = DataPoisoningDetector()
        report = detector.run_full_analysis(X, y)

        assert "analysis_timestamp" in report
        assert "outlier_detection" in report
        assert "label_flipping" in report
        assert "overall_risk_assessment" in report
        assert report["overall_risk_assessment"]["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

    def test_poisoned_data_detection(self, wine_data):
        """Test detection of artificially poisoned data."""
        X, y = wine_data
        X_poisoned = X.copy()

        # Inject poison: add extreme outliers
        n_poison = 10
        for col in X_poisoned.columns[:3]:
            X_poisoned.loc[:n_poison-1, col] = X_poisoned[col].max() * 10

        detector = DataPoisoningDetector()
        report = detector.run_full_analysis(X_poisoned, y)

        # Should detect the poisoned samples
        assert report["outlier_detection"]["consensus"]["n_outliers"] > 0


# =============================================================================
# Lecke 113: Model Poisoning Detection Tests
# =============================================================================
class TestModelPoisoningDetector:
    """Tests for model poisoning detection module."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = ModelPoisoningDetector()
        assert detector.performance_threshold == 0.1

    def test_compute_model_fingerprint(self, trained_classifier):
        """Test model fingerprint computation."""
        detector = ModelPoisoningDetector()
        fingerprint = detector.compute_model_fingerprint(trained_classifier)

        assert len(fingerprint) == 64

    def test_verify_model_integrity(self, trained_classifier):
        """Test model integrity verification."""
        detector = ModelPoisoningDetector()
        fingerprint = detector.compute_model_fingerprint(trained_classifier)

        # Same model should verify
        assert detector.verify_model_integrity(trained_classifier, fingerprint)

        # Different fingerprint should fail
        assert not detector.verify_model_integrity(trained_classifier, "wrong_hash")

    def test_extract_model_weights(self, trained_classifier):
        """Test weight extraction."""
        detector = ModelPoisoningDetector()
        weights = detector.extract_model_weights(trained_classifier)

        assert "feature_importances" in weights

    def test_analyze_weight_distribution(self, trained_classifier):
        """Test weight distribution analysis."""
        detector = ModelPoisoningDetector()
        report = detector.analyze_weight_distribution(trained_classifier)

        assert "weight_statistics" in report
        assert "risk_score" in report

    def test_detect_prediction_anomalies(self, trained_classifier, wine_train_test_split):
        """Test prediction anomaly detection."""
        X_train, X_test, y_train, y_test = wine_train_test_split
        detector = ModelPoisoningDetector()
        report = detector.detect_prediction_anomalies(
            trained_classifier, X_test, y_test, is_classifier=True
        )

        assert "prediction_analysis" in report
        assert "accuracy" in report["prediction_analysis"]

    def test_detect_backdoor_triggers(self, trained_classifier, wine_train_test_split):
        """Test backdoor trigger detection."""
        X_train, X_test, y_train, y_test = wine_train_test_split
        detector = ModelPoisoningDetector()
        report = detector.detect_backdoor_triggers(
            trained_classifier, X_test, y_test
        )

        assert "trigger_analysis" in report
        assert "suspicious_patterns" in report

    def test_run_full_analysis(self, trained_classifier, wine_train_test_split):
        """Test full model poisoning analysis."""
        X_train, X_test, y_train, y_test = wine_train_test_split
        detector = ModelPoisoningDetector()
        report = detector.run_full_analysis(
            trained_classifier, X_test, y_test, is_classifier=True
        )

        assert "model_info" in report
        assert "overall_risk_assessment" in report
        assert report["overall_risk_assessment"]["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


# =============================================================================
# Lecke 114: Explainable AI Tests
# =============================================================================
class TestExplainableAI:
    """Tests for Explainable AI module."""

    def test_initialization(self, trained_classifier):
        """Test explainer initialization."""
        wine = load_wine()
        explainer = ExplainableAI(
            model=trained_classifier,
            feature_names=wine.feature_names,
            task_type="classification"
        )
        assert explainer.model is not None
        assert len(explainer.feature_names) == 13

    def test_permutation_importance(self, trained_classifier, wine_train_test_split):
        """Test permutation importance computation."""
        X_train, X_test, y_train, y_test = wine_train_test_split
        wine = load_wine()

        explainer = ExplainableAI(
            model=trained_classifier,
            feature_names=wine.feature_names,
            task_type="classification"
        )

        report = explainer.compute_permutation_importance(X_test, y_test)

        assert "feature_importance" in report
        assert "top_features" in report
        assert len(report["top_features"]) == 5

    def test_generate_vertex_ai_config(self, trained_classifier, wine_train_test_split):
        """Test Vertex AI config generation."""
        X_train, X_test, y_train, y_test = wine_train_test_split
        wine = load_wine()

        explainer = ExplainableAI(
            model=trained_classifier,
            feature_names=wine.feature_names,
            task_type="classification"
        )

        config = explainer.generate_vertex_ai_config(X_train)

        assert "explanation_metadata" in config
        assert "explanation_parameters" in config
        assert len(config["explanation_metadata"]["inputs"]) == 13


# =============================================================================
# Lecke 114: Model Monitoring Tests
# =============================================================================
class TestModelMonitor:
    """Tests for model monitoring module."""

    def test_initialization(self, trained_classifier):
        """Test monitor initialization."""
        wine = load_wine()
        monitor = ModelMonitor(
            model=trained_classifier,
            feature_names=wine.feature_names,
            task_type="classification"
        )
        assert monitor.window_size == 1000

    def test_set_baseline(self, trained_classifier, wine_train_test_split):
        """Test baseline setting."""
        X_train, X_test, y_train, y_test = wine_train_test_split
        wine = load_wine()

        monitor = ModelMonitor(
            model=trained_classifier,
            feature_names=wine.feature_names,
            task_type="classification"
        )

        baseline = monitor.set_baseline(X_train, y_train)

        assert "feature_stats" in baseline
        assert "prediction_stats" in baseline
        assert "accuracy" in baseline["prediction_stats"]

    def test_log_prediction(self, trained_classifier, wine_train_test_split):
        """Test prediction logging."""
        X_train, X_test, y_train, y_test = wine_train_test_split
        wine = load_wine()

        monitor = ModelMonitor(
            model=trained_classifier,
            feature_names=wine.feature_names,
            task_type="classification",
            window_size=20
        )

        monitor.set_baseline(X_train, y_train)

        predictions = trained_classifier.predict(X_test[:10])
        status = monitor.log_prediction(X_test[:10], predictions)

        assert "logged" in status
        assert status["logged"] == 10

    def test_check_drift(self, trained_classifier, wine_train_test_split):
        """Test drift detection."""
        X_train, X_test, y_train, y_test = wine_train_test_split
        wine = load_wine()

        monitor = ModelMonitor(
            model=trained_classifier,
            feature_names=wine.feature_names,
            task_type="classification",
            window_size=20
        )

        monitor.set_baseline(X_train, y_train)

        # Log predictions
        for i in range(0, len(X_test), 5):
            batch = X_test.iloc[i:i+5]
            preds = trained_classifier.predict(batch)
            monitor.log_prediction(batch, preds)

        drift_report = monitor.check_drift()

        assert "feature_drift" in drift_report
        assert "prediction_drift" in drift_report

    def test_detect_drift_with_shifted_data(self, trained_classifier, wine_train_test_split):
        """Test drift detection with artificially shifted data."""
        X_train, X_test, y_train, y_test = wine_train_test_split
        wine = load_wine()

        monitor = ModelMonitor(
            model=trained_classifier,
            feature_names=wine.feature_names,
            task_type="classification",
            window_size=20,
            drift_threshold=0.1
        )

        monitor.set_baseline(X_train, y_train)

        # Create shifted data
        X_shifted = X_test.copy()
        X_shifted.iloc[:, 0] = X_shifted.iloc[:, 0] * 2

        # Log shifted predictions
        for i in range(0, 30, 5):
            if i < len(X_shifted):
                batch = X_shifted.iloc[i:i+5]
                preds = trained_classifier.predict(batch)
                monitor.log_prediction(batch, preds)

        drift_report = monitor.check_drift()

        # Should detect drift
        assert drift_report["drift_detected"] or len(drift_report["feature_drift"]) > 0

    def test_get_monitoring_summary(self, trained_classifier, wine_train_test_split):
        """Test monitoring summary generation."""
        X_train, X_test, y_train, y_test = wine_train_test_split
        wine = load_wine()

        monitor = ModelMonitor(
            model=trained_classifier,
            feature_names=wine.feature_names,
            task_type="classification"
        )

        monitor.set_baseline(X_train, y_train)
        summary = monitor.get_monitoring_summary()

        assert "buffer_status" in summary
        assert "alert_summary" in summary


# =============================================================================
# Lecke 115-116: Adversarial Robustness Tests
# =============================================================================
class TestAdversarialTester:
    """Tests for adversarial robustness testing module."""

    def test_initialization(self, trained_classifier):
        """Test tester initialization."""
        wine = load_wine()
        tester = AdversarialTester(
            model=trained_classifier,
            feature_names=wine.feature_names,
            task_type="classification"
        )
        assert tester.task_type == "classification"

    def test_perturbation_attack(self, trained_classifier, wine_train_test_split):
        """Test perturbation attack."""
        X_train, X_test, y_train, y_test = wine_train_test_split
        wine = load_wine()

        tester = AdversarialTester(
            model=trained_classifier,
            feature_names=wine.feature_names,
            task_type="classification"
        )

        report = tester.perturbation_attack(
            X_test, y_test,
            epsilon_values=[0.1, 0.2],
            n_samples=30
        )

        assert "perturbation_results" in report
        assert "overall_robustness_score" in report
        assert 0 <= report["overall_robustness_score"] <= 1

    def test_fgsm_attack(self, trained_classifier, wine_train_test_split):
        """Test FGSM attack."""
        X_train, X_test, y_train, y_test = wine_train_test_split
        wine = load_wine()

        tester = AdversarialTester(
            model=trained_classifier,
            feature_names=wine.feature_names,
            task_type="classification"
        )

        report = tester.fgsm_attack(X_test[:30], y_test[:30], epsilon=0.1)

        assert "attack_type" in report
        assert "FGSM" in report["attack_type"]
        assert "robustness_ratio" in report

    def test_feature_importance_attack(self, trained_classifier, wine_train_test_split):
        """Test feature importance attack."""
        X_train, X_test, y_train, y_test = wine_train_test_split
        wine = load_wine()

        tester = AdversarialTester(
            model=trained_classifier,
            feature_names=wine.feature_names,
            task_type="classification"
        )

        report = tester.feature_importance_attack(
            X_test, y_test,
            top_k_features=3
        )

        assert "attacked_features" in report
        assert len(report["attacked_features"]) == 3
        assert "robustness_ratio" in report

    def test_boundary_attack(self, trained_classifier, wine_train_test_split):
        """Test boundary attack."""
        X_train, X_test, y_train, y_test = wine_train_test_split
        wine = load_wine()

        tester = AdversarialTester(
            model=trained_classifier,
            feature_names=wine.feature_names,
            task_type="classification"
        )

        report = tester.boundary_attack(X_test, y_test, n_boundary_samples=20)

        assert "boundary_vulnerability_ratio" in report
        assert "robustness_assessment" in report

    def test_run_full_robustness_test(self, trained_classifier, wine_train_test_split):
        """Test full robustness testing."""
        X_train, X_test, y_train, y_test = wine_train_test_split
        wine = load_wine()

        tester = AdversarialTester(
            model=trained_classifier,
            feature_names=wine.feature_names,
            task_type="classification"
        )

        report = tester.run_full_robustness_test(X_test, y_test, n_samples=30)

        assert "attacks" in report
        assert "overall_robustness" in report
        assert "grade" in report["overall_robustness"]


# =============================================================================
# Lecke 117-120: Dependency Scanning Tests
# =============================================================================
class TestDependencyScanner:
    """Tests for dependency scanning module."""

    def test_initialization(self):
        """Test scanner initialization."""
        scanner = DependencyScanner(project_path=".")
        assert scanner.requirements_file == "requirements.txt"

    def test_check_for_typosquatting(self, tmp_path):
        """Test typosquatting detection."""
        # Create a requirements file with a typosquatting attempt
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("numpi>=1.0\nrequets>=2.0\n")

        scanner = DependencyScanner(
            project_path=str(tmp_path),
            requirements_file="requirements.txt"
        )
        report = scanner.check_for_typosquatting()

        assert report["status"] == "suspicious_found"
        assert len(report["suspicious_packages"]) == 2

    def test_clean_requirements(self, tmp_path):
        """Test with clean requirements."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("numpy>=1.0\nrequests>=2.0\n")

        scanner = DependencyScanner(
            project_path=str(tmp_path),
            requirements_file="requirements.txt"
        )
        report = scanner.check_for_typosquatting()

        assert report["status"] == "clean"


# =============================================================================
# Lecke 121-122: External Model Validation Tests
# =============================================================================
class TestExternalModelValidator:
    """Tests for external model validation module."""

    def test_initialization(self):
        """Test validator initialization."""
        validator = ExternalModelValidator()
        assert len(validator.TRUSTED_ORGS) > 0

    def test_scan_pickle_file(self, tmp_path):
        """Test pickle file scanning."""
        import pickle

        # Create a safe pickle file
        test_data = {"key": "value", "numbers": [1, 2, 3]}
        pickle_file = tmp_path / "test.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(test_data, f)

        validator = ExternalModelValidator()
        report = validator.scan_pickle_file(str(pickle_file))

        assert "file_hash" in report
        assert report["file_hash"] is not None

    def test_verify_model_signature(self, tmp_path):
        """Test model signature verification."""
        import hashlib

        # Create a test file
        test_file = tmp_path / "model.bin"
        test_content = b"test model content"
        test_file.write_bytes(test_content)

        expected_hash = hashlib.sha256(test_content).hexdigest()

        validator = ExternalModelValidator()
        report = validator.verify_model_signature(
            str(test_file),
            expected_hash=expected_hash
        )

        assert report["verified"] == True
        assert report["computed_hash"] == expected_hash

    def test_verify_model_signature_mismatch(self, tmp_path):
        """Test model signature verification with mismatch."""
        test_file = tmp_path / "model.bin"
        test_file.write_bytes(b"test content")

        validator = ExternalModelValidator()
        report = validator.verify_model_signature(
            str(test_file),
            expected_hash="wrong_hash"
        )

        assert report["verified"] == False
        assert "error" in report

    def test_audit_local_model(self, trained_classifier, tmp_path):
        """Test local model auditing."""
        import joblib

        # Save model
        model_path = tmp_path / "model.joblib"
        joblib.dump(trained_classifier, model_path)

        validator = ExternalModelValidator()
        report = validator.audit_local_model(str(model_path))

        assert "audit_results" in report
        assert len(report["audit_results"]) > 0

    def test_generate_model_attestation(self, trained_classifier, tmp_path):
        """Test attestation generation."""
        import joblib

        model_path = tmp_path / "model.joblib"
        joblib.dump(trained_classifier, model_path)

        validator = ExternalModelValidator()

        validation_report = {
            "overall_status": "PASS",
            "risks": [],
            "warnings": [],
            "validation_results": {
                "license": {"is_compliant": True},
                "provenance": {"is_trusted": True},
                "security": {"has_risks": False}
            }
        }

        attestation = validator.generate_model_attestation(
            str(model_path),
            "test-model",
            validation_report
        )

        assert "attestation_hash" in attestation
        assert attestation["validation_summary"]["overall_status"] == "PASS"


# =============================================================================
# Integration Tests
# =============================================================================
class TestSecurityIntegration:
    """Integration tests for security modules."""

    def test_full_security_pipeline(self, trained_classifier, wine_train_test_split):
        """Test running all security checks in sequence."""
        X_train, X_test, y_train, y_test = wine_train_test_split
        wine = load_wine()

        # 1. Data Poisoning Detection
        data_detector = DataPoisoningDetector()
        data_report = data_detector.run_full_analysis(X_train, y_train)
        assert data_report["overall_risk_assessment"]["risk_level"] is not None

        # 2. Model Poisoning Detection
        model_detector = ModelPoisoningDetector()
        model_report = model_detector.run_full_analysis(
            trained_classifier, X_test, y_test, is_classifier=True
        )
        assert model_report["overall_risk_assessment"]["risk_level"] is not None

        # 3. Adversarial Robustness
        tester = AdversarialTester(
            model=trained_classifier,
            feature_names=wine.feature_names,
            task_type="classification"
        )
        robustness_report = tester.run_full_robustness_test(
            X_test, y_test, n_samples=30
        )
        assert robustness_report["overall_robustness"]["grade"] is not None

        # 4. Model Monitoring Setup
        monitor = ModelMonitor(
            model=trained_classifier,
            feature_names=wine.feature_names,
            task_type="classification"
        )
        baseline = monitor.set_baseline(X_train, y_train)
        assert "accuracy" in baseline["prediction_stats"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
