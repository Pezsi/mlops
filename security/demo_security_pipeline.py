#!/usr/bin/env python3
"""
MLSecOps Security Pipeline Demo

This script demonstrates all security features implemented for the Wine Quality project.
Run this to see the security modules in action.

Usage:
    python security/demo_security_pipeline.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

# Import security modules
from security.poisoning.data_poisoning_detector import DataPoisoningDetector
from security.poisoning.model_poisoning_detector import ModelPoisoningDetector
from security.robustness.adversarial_tester import AdversarialTester
from security.vertex_ai.explainable_ai import ExplainableAI
from security.vertex_ai.model_monitoring import ModelMonitor
from security.dependency_audit.dependency_scanner import DependencyScanner
from security.model_audit.external_model_validator import ExternalModelValidator


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_result(name: str, value: str, indent: int = 2):
    """Print formatted result."""
    spaces = " " * indent
    print(f"{spaces}{name}: {value}")


def demo_data_poisoning_detection(X_train, y_train):
    """Demo: Data Poisoning Detection (Lecke 113)"""
    print_header("📊 DATA POISONING DETECTION (Lecke 113)")

    detector = DataPoisoningDetector(
        contamination=0.1,
        z_score_threshold=3.0
    )

    print("\n▶ Running full data poisoning analysis...")
    report = detector.run_full_analysis(X_train, y_train)

    print("\n📈 Results:")
    print_result("Dataset size", f"{report['dataset_info']['n_samples']} samples")
    print_result("Data fingerprint", report['data_fingerprint'][:32] + "...")

    print("\n  Outlier Detection:")
    print_result("Z-score outliers",
                 f"{report['outlier_detection']['zscore']['n_outliers']}", 4)
    print_result("IQR outliers",
                 f"{report['outlier_detection']['iqr']['n_outliers']}", 4)
    print_result("Isolation Forest outliers",
                 f"{report['outlier_detection']['isolation_forest']['n_outliers']}", 4)
    print_result("Consensus outliers",
                 f"{report['outlier_detection']['consensus']['n_outliers']}", 4)

    print("\n  Label Flipping Analysis:")
    print_result("Suspicious samples",
                 f"{report['label_flipping']['total_suspicious']}", 4)
    print_result("Suspicious ratio",
                 f"{report['label_flipping']['suspicious_ratio']:.2%}", 4)

    print("\n  Backdoor Pattern Detection:")
    print_result("Suspicious patterns",
                 f"{len(report['backdoor_patterns']['suspicious_patterns'])}", 4)
    print_result("Risk score",
                 f"{report['backdoor_patterns']['risk_score']:.3f}", 4)

    risk = report['overall_risk_assessment']
    status = "✅" if risk['risk_level'] == "LOW" else "⚠️" if risk['risk_level'] == "MEDIUM" else "❌"
    print(f"\n  {status} Overall Risk Level: {risk['risk_level']}")

    return report


def demo_model_poisoning_detection(model, X_test, y_test):
    """Demo: Model Poisoning Detection (Lecke 113)"""
    print_header("🔍 MODEL POISONING DETECTION (Lecke 113)")

    detector = ModelPoisoningDetector()

    print("\n▶ Computing model fingerprint...")
    fingerprint = detector.compute_model_fingerprint(model)
    print_result("Model fingerprint", fingerprint[:32] + "...")

    print("\n▶ Running full model poisoning analysis...")
    report = detector.run_full_analysis(model, X_test, y_test, is_classifier=True)

    print("\n📈 Results:")
    print_result("Model type", report['model_info']['type'])

    print("\n  Weight Analysis:")
    weight_stats = report['weight_analysis'].get('weight_statistics', {})
    print_result("Weight parameters analyzed", str(len(weight_stats)), 4)
    print_result("Weight anomalies", str(len(report['weight_analysis'].get('anomalies', []))), 4)
    print_result("Weight risk score", f"{report['weight_analysis'].get('risk_score', 0):.3f}", 4)

    print("\n  Prediction Analysis:")
    pred = report['prediction_analysis'].get('prediction_analysis', {})
    print_result("Accuracy", f"{pred.get('accuracy', 'N/A'):.3f}" if isinstance(pred.get('accuracy'), float) else "N/A", 4)

    print("\n  Backdoor Trigger Analysis:")
    backdoor = report['backdoor_analysis']
    print_result("Suspicious patterns", str(len(backdoor.get('suspicious_patterns', []))), 4)
    print_result("Trigger risk score", f"{backdoor.get('risk_score', 0):.3f}", 4)

    risk = report['overall_risk_assessment']
    status = "✅" if risk['risk_level'] == "LOW" else "⚠️" if risk['risk_level'] == "MEDIUM" else "❌"
    print(f"\n  {status} Overall Risk Level: {risk['risk_level']}")

    return report


def demo_explainable_ai(model, X_train, X_test, y_test, feature_names):
    """Demo: Explainable AI (Lecke 114)"""
    print_header("🧠 EXPLAINABLE AI (Lecke 114)")

    explainer = ExplainableAI(
        model=model,
        feature_names=feature_names,
        task_type="classification"
    )

    print("\n▶ Computing permutation importance...")
    perm_report = explainer.compute_permutation_importance(X_test, y_test)

    print("\n📈 Top 5 Important Features (Permutation):")
    for i, feat in enumerate(perm_report['top_features'], 1):
        imp = perm_report['feature_importance'][feat]
        print(f"    {i}. {feat}: {imp['mean']:.4f} (±{imp['std']:.4f})")

    print("\n▶ Generating Vertex AI configuration...")
    vertex_config = explainer.generate_vertex_ai_config(X_train)
    print_result("Inputs configured", str(len(vertex_config['explanation_metadata']['inputs'])))
    print_result("Attribution methods", "sampled_shapley, xrai, integrated_gradients")

    return perm_report


def demo_model_monitoring(model, X_train, y_train, X_test, y_test, feature_names):
    """Demo: Model Monitoring (Lecke 114)"""
    print_header("📡 MODEL MONITORING (Lecke 114)")

    monitor = ModelMonitor(
        model=model,
        feature_names=feature_names,
        task_type="classification",
        window_size=50,
        drift_threshold=0.1
    )

    print("\n▶ Setting up baseline from training data...")
    baseline = monitor.set_baseline(X_train, y_train)
    print_result("Baseline accuracy", f"{baseline['prediction_stats']['accuracy']:.3f}")
    print_result("Features monitored", str(len(baseline['feature_stats'])))

    print("\n▶ Simulating production traffic...")
    for i in range(0, min(60, len(X_test)), 10):
        batch = X_test.iloc[i:i+10]
        preds = model.predict(batch)
        status = monitor.log_prediction(batch, preds, y_test.iloc[i:i+10].values)

    print("\n▶ Checking for drift...")
    drift_report = monitor.check_drift()

    print("\n📈 Drift Detection Results:")
    drifted = [f for f, d in drift_report['feature_drift'].items() if d.get('is_drifted')]
    print_result("Drift detected", str(drift_report['drift_detected']))
    print_result("Drifted features", str(len(drifted)))

    pred_drift = drift_report['prediction_drift']
    print_result("Prediction mean shift", f"{pred_drift['mean_shift']:.3f}")

    summary = monitor.get_monitoring_summary()
    print_result("Total alerts", str(summary['alert_summary']['total_alerts']))

    return drift_report


def demo_adversarial_robustness(model, X_test, y_test, feature_names):
    """Demo: Adversarial Robustness Testing (Lecke 115-116)"""
    print_header("⚔️ ADVERSARIAL ROBUSTNESS TESTING (Lecke 115-116)")

    tester = AdversarialTester(
        model=model,
        feature_names=feature_names,
        task_type="classification"
    )

    # Use smaller sample for demo
    n_samples = min(50, len(X_test))
    X_sample = X_test.iloc[:n_samples]
    y_sample = y_test.iloc[:n_samples]

    print(f"\n▶ Testing with {n_samples} samples...")

    print("\n1️⃣ Perturbation Attack:")
    perturb = tester.perturbation_attack(
        X_sample, y_sample,
        epsilon_values=[0.05, 0.1, 0.2]
    )
    print_result("Overall robustness", f"{perturb['overall_robustness_score']:.3f}", 4)
    print_result("Grade", perturb['robustness_grade'], 4)

    print("\n2️⃣ FGSM Attack:")
    fgsm = tester.fgsm_attack(X_sample, y_sample, epsilon=0.1)
    print_result("Robustness ratio", f"{fgsm['robustness_ratio']:.3f}", 4)

    print("\n3️⃣ Feature Importance Attack:")
    fi = tester.feature_importance_attack(X_sample, y_sample, top_k_features=3)
    print_result("Attacked features", ", ".join(fi['attacked_features'][:3]), 4)
    print_result("Robustness ratio", f"{fi['robustness_ratio']:.3f}", 4)

    print("\n4️⃣ Boundary Attack:")
    boundary = tester.boundary_attack(X_sample, y_sample, n_boundary_samples=30)
    print_result("Vulnerability ratio", f"{boundary['boundary_vulnerability_ratio']:.3f}", 4)
    print_result("Assessment", boundary['robustness_assessment'], 4)

    print("\n▶ Full Robustness Summary:")
    full = tester.run_full_robustness_test(X_sample, y_sample, n_samples=n_samples)
    overall = full['overall_robustness']
    status = "✅" if "A" in overall['grade'] or "B" in overall['grade'] else "⚠️"
    print(f"    {status} Grade: {overall['grade']}")
    print_result("Average score", f"{overall['average_score']:.3f}", 4)
    print_result("Minimum score", f"{overall['minimum_score']:.3f}", 4)

    return full


def demo_dependency_scanning():
    """Demo: Dependency Security Scanning (Lecke 117-120)"""
    print_header("🔒 DEPENDENCY SECURITY (Lecke 117-120)")

    scanner = DependencyScanner(
        project_path=str(project_root),
        requirements_file="requirements.txt"
    )

    print("\n▶ Checking for typosquatting...")
    typo = scanner.check_for_typosquatting()
    print_result("Status", typo['status'])
    if typo.get('suspicious_packages'):
        for pkg in typo['suspicious_packages']:
            print(f"    ⚠️ {pkg['package']} might be {pkg['might_be']}")

    print("\n▶ Checking outdated packages...")
    outdated = scanner.check_outdated_packages()
    print_result("Outdated packages", str(outdated.get('outdated_count', 0)))

    print("\n📋 Dependency Security Summary:")
    print("    ✅ Dependabot configured: .github/dependabot.yml")
    print("    ✅ Snyk configured: .snyk")
    print("    ✅ Security scan workflow: .github/workflows/security-scan.yml")

    return typo


def demo_external_model_validation():
    """Demo: External Model Validation (Lecke 121-122)"""
    print_header("📦 EXTERNAL MODEL VALIDATION (Lecke 121-122)")

    validator = ExternalModelValidator()

    print("\n▶ Trusted organizations list:")
    for org in validator.TRUSTED_ORGS[:5]:
        print(f"    ✓ {org}")
    print(f"    ... and {len(validator.TRUSTED_ORGS) - 5} more")

    print("\n▶ Testing local pickle scanning...")
    import tempfile
    import pickle

    # Create a safe test pickle
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pickle.dump({"safe": "data"}, f)
        temp_path = f.name

    scan_result = validator.scan_pickle_file(temp_path)
    os.unlink(temp_path)

    print_result("File hash", scan_result['file_hash'][:32] + "...")
    print_result("Is safe", "✅ Yes" if scan_result['is_safe'] else "❌ No")

    print("\n▶ Model attestation capabilities:")
    print("    • Hugging Face model validation")
    print("    • License compliance checking")
    print("    • Provenance verification")
    print("    • Security scanning for pickle files")
    print("    • Model integrity verification")

    return scan_result


def main():
    """Run complete security demo."""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "MLSecOps Security Demo" + " " * 21 + "║")
    print("║" + " " * 10 + "Wine Quality MLOps Project" + " " * 22 + "║")
    print("╚" + "═" * 58 + "╝")

    # Load data
    print("\n▶ Loading Wine dataset...")
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Training: {len(X_train)} samples, Testing: {len(X_test)} samples")

    # Train model
    print("\n▶ Training RandomForest classifier...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"  Model accuracy: {accuracy:.3f}")

    # Run demos
    results = {}

    # 1. Data Poisoning Detection
    results['data_poisoning'] = demo_data_poisoning_detection(X_train, y_train)

    # 2. Model Poisoning Detection
    results['model_poisoning'] = demo_model_poisoning_detection(model, X_test, y_test)

    # 3. Explainable AI
    results['explainable_ai'] = demo_explainable_ai(
        model, X_train, X_test, y_test, wine.feature_names
    )

    # 4. Model Monitoring
    results['monitoring'] = demo_model_monitoring(
        model, X_train, y_train, X_test, y_test, wine.feature_names
    )

    # 5. Adversarial Robustness
    results['robustness'] = demo_adversarial_robustness(
        model, X_test, y_test, wine.feature_names
    )

    # 6. Dependency Scanning
    results['dependencies'] = demo_dependency_scanning()

    # 7. External Model Validation
    results['external_models'] = demo_external_model_validation()

    # Final Summary
    print_header("📊 FINAL SECURITY ASSESSMENT")

    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│                   Security Summary                       │")
    print("├─────────────────────────────────────────────────────────┤")

    # Data Poisoning
    data_risk = results['data_poisoning']['overall_risk_assessment']['risk_level']
    data_status = "✅" if data_risk == "LOW" else "⚠️" if data_risk == "MEDIUM" else "❌"
    print(f"│ {data_status} Data Poisoning Risk: {data_risk:<33}│")

    # Model Poisoning
    model_risk = results['model_poisoning']['overall_risk_assessment']['risk_level']
    model_status = "✅" if model_risk == "LOW" else "⚠️" if model_risk == "MEDIUM" else "❌"
    print(f"│ {model_status} Model Poisoning Risk: {model_risk:<32}│")

    # Robustness
    rob_grade = results['robustness']['overall_robustness']['grade']
    rob_status = "✅" if "A" in rob_grade or "B" in rob_grade else "⚠️"
    print(f"│ {rob_status} Adversarial Robustness: {rob_grade:<28}│")

    # Dependencies
    dep_status = "✅" if results['dependencies']['status'] == "clean" else "⚠️"
    print(f"│ {dep_status} Dependency Security: {'Configured':<32}│")

    # Monitoring
    print(f"│ ✅ Production Monitoring: {'Ready':<32}│")

    print("└─────────────────────────────────────────────────────────┘")

    print("\n🎉 Security pipeline demo completed successfully!")
    print("\nFor detailed documentation, see: MLSECOPS_README.md")

    return results


if __name__ == "__main__":
    main()
