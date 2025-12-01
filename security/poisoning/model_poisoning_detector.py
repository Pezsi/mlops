"""
Model Poisoning Detection Module (Lecke 113)

This module provides model poisoning detection capabilities:
- Model behavior analysis
- Weight distribution anomaly detection
- Prediction consistency checks
- Trigger pattern detection
- Model integrity verification

Reference: OWASP ML Security Top 10 - ML04:2023 Model Poisoning
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import hashlib
import pickle
import json
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPoisoningDetector:
    """
    Model poisoning detection for ML models.

    Detects various types of model poisoning:
    1. Weight manipulation - abnormal weight distributions
    2. Backdoor triggers - specific inputs causing wrong predictions
    3. Degradation attacks - subtle performance degradation
    4. Trojan models - hidden malicious behaviors
    """

    def __init__(
        self,
        reference_model: Optional[BaseEstimator] = None,
        performance_threshold: float = 0.1,
        weight_anomaly_threshold: float = 3.0,
        random_state: int = 42
    ):
        """
        Initialize the model poisoning detector.

        Args:
            reference_model: Clean reference model for comparison
            performance_threshold: Maximum allowed performance deviation
            weight_anomaly_threshold: Z-score threshold for weight anomalies
            random_state: Random seed for reproducibility
        """
        self.reference_model = reference_model
        self.performance_threshold = performance_threshold
        self.weight_anomaly_threshold = weight_anomaly_threshold
        self.random_state = random_state
        self.model_fingerprint: Optional[str] = None
        self.baseline_predictions: Optional[np.ndarray] = None

    def compute_model_fingerprint(
        self,
        model: BaseEstimator
    ) -> str:
        """
        Compute cryptographic fingerprint of model for integrity verification.

        Args:
            model: Trained model

        Returns:
            SHA256 hash of model bytes
        """
        model_bytes = pickle.dumps(model)
        fingerprint = hashlib.sha256(model_bytes).hexdigest()
        self.model_fingerprint = fingerprint
        logger.info(f"Model fingerprint: {fingerprint[:16]}...")
        return fingerprint

    def verify_model_integrity(
        self,
        model: BaseEstimator,
        expected_fingerprint: str
    ) -> bool:
        """
        Verify model integrity against expected fingerprint.

        Args:
            model: Model to verify
            expected_fingerprint: Expected SHA256 hash

        Returns:
            True if fingerprints match
        """
        current_fingerprint = self.compute_model_fingerprint(model)
        is_valid = current_fingerprint == expected_fingerprint

        if not is_valid:
            logger.warning("Model integrity verification FAILED!")
        else:
            logger.info("Model integrity verification passed")

        return is_valid

    def extract_model_weights(
        self,
        model: BaseEstimator
    ) -> Dict[str, np.ndarray]:
        """
        Extract weights from various model types.

        Args:
            model: Trained model

        Returns:
            Dictionary of weight arrays
        """
        weights = {}

        # sklearn models
        if hasattr(model, 'coef_'):
            weights['coefficients'] = np.array(model.coef_).flatten()

        if hasattr(model, 'intercept_'):
            intercept = model.intercept_
            if isinstance(intercept, np.ndarray):
                weights['intercept'] = intercept.flatten()
            else:
                weights['intercept'] = np.array([intercept])

        if hasattr(model, 'feature_importances_'):
            weights['feature_importances'] = model.feature_importances_

        # Tree-based models
        if hasattr(model, 'estimators_'):
            # Random Forest, Gradient Boosting
            tree_depths = []
            n_leaves = []
            for estimator in model.estimators_[:min(100, len(model.estimators_))]:
                if hasattr(estimator, 'tree_'):
                    tree = estimator.tree_
                elif hasattr(estimator, 'tree_'):
                    tree = estimator[0].tree_ if hasattr(estimator, '__getitem__') else None
                else:
                    tree = None

                if tree is not None:
                    tree_depths.append(tree.max_depth)
                    n_leaves.append(tree.n_leaves)

            if tree_depths:
                weights['tree_depths'] = np.array(tree_depths)
                weights['n_leaves'] = np.array(n_leaves)

        return weights

    def analyze_weight_distribution(
        self,
        model: BaseEstimator
    ) -> Dict[str, Any]:
        """
        Analyze weight distribution for anomalies.

        Args:
            model: Trained model

        Returns:
            Weight analysis report
        """
        weights = self.extract_model_weights(model)

        if not weights:
            logger.warning("No weights extracted from model")
            return {"error": "Could not extract weights from model"}

        report = {
            "timestamp": datetime.now().isoformat(),
            "weight_statistics": {},
            "anomalies": [],
            "risk_score": 0.0
        }

        risk_factors = []

        for name, w in weights.items():
            if len(w) == 0:
                continue

            w_stats = {
                "name": name,
                "shape": list(w.shape),
                "mean": float(np.mean(w)),
                "std": float(np.std(w)),
                "min": float(np.min(w)),
                "max": float(np.max(w)),
                "median": float(np.median(w)),
                "skewness": float(stats.skew(w.flatten())),
                "kurtosis": float(stats.kurtosis(w.flatten())),
                "sparsity": float(np.mean(np.abs(w) < 1e-6)),
                "inf_count": int(np.sum(np.isinf(w))),
                "nan_count": int(np.sum(np.isnan(w)))
            }

            report["weight_statistics"][name] = w_stats

            # Check for anomalies
            anomalies = []

            # Extreme values
            if w_stats["max"] > 1000 or w_stats["min"] < -1000:
                anomalies.append("extreme_values")
                risk_factors.append(f"{name}_extreme_values")

            # High kurtosis (heavy tails)
            if abs(w_stats["kurtosis"]) > 10:
                anomalies.append("abnormal_distribution")
                risk_factors.append(f"{name}_abnormal_distribution")

            # NaN or Inf
            if w_stats["nan_count"] > 0 or w_stats["inf_count"] > 0:
                anomalies.append("invalid_values")
                risk_factors.append(f"{name}_invalid_values")

            # High sparsity (could indicate pruning attack)
            if w_stats["sparsity"] > 0.9:
                anomalies.append("high_sparsity")

            if anomalies:
                report["anomalies"].append({
                    "weight_name": name,
                    "issues": anomalies
                })

        # Calculate risk score
        report["risk_score"] = min(1.0, len(risk_factors) * 0.2)
        report["risk_factors"] = risk_factors

        logger.info(f"Weight analysis complete. Risk score: {report['risk_score']:.2f}")
        return report

    def detect_prediction_anomalies(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        is_classifier: bool = True
    ) -> Dict[str, Any]:
        """
        Detect anomalous prediction patterns that might indicate poisoning.

        Args:
            model: Model to analyze
            X: Features
            y: True labels
            is_classifier: Whether model is a classifier

        Returns:
            Prediction anomaly report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "prediction_analysis": {},
            "anomalies": [],
            "risk_score": 0.0
        }

        try:
            predictions = model.predict(X)

            if is_classifier:
                # Classification metrics
                accuracy = accuracy_score(y, predictions)
                report["prediction_analysis"]["accuracy"] = float(accuracy)

                # Check for class imbalance in predictions
                unique, counts = np.unique(predictions, return_counts=True)
                pred_distribution = {str(int(u)): int(c) for u, c in zip(unique, counts)}
                report["prediction_analysis"]["prediction_distribution"] = pred_distribution

                # Check if model always predicts one class
                if len(unique) == 1:
                    report["anomalies"].append({
                        "type": "single_class_prediction",
                        "description": f"Model only predicts class {unique[0]}"
                    })

                # Check for suspiciously perfect accuracy on subsets
                for class_val in y.unique():
                    class_mask = y == class_val
                    class_acc = accuracy_score(y[class_mask], predictions[class_mask])
                    if class_acc == 1.0 and class_mask.sum() > 10:
                        report["anomalies"].append({
                            "type": "perfect_class_accuracy",
                            "description": f"100% accuracy on class {class_val}",
                            "samples": int(class_mask.sum())
                        })

            else:
                # Regression metrics
                mse = mean_squared_error(y, predictions)
                r2 = r2_score(y, predictions)
                report["prediction_analysis"]["mse"] = float(mse)
                report["prediction_analysis"]["r2"] = float(r2)

                # Check for prediction distribution anomalies
                residuals = y - predictions
                report["prediction_analysis"]["residual_stats"] = {
                    "mean": float(residuals.mean()),
                    "std": float(residuals.std()),
                    "skewness": float(stats.skew(residuals)),
                    "kurtosis": float(stats.kurtosis(residuals))
                }

                # Check for biased predictions
                if abs(residuals.mean()) > 0.5:
                    report["anomalies"].append({
                        "type": "biased_predictions",
                        "description": f"Systematic bias of {residuals.mean():.3f}"
                    })

            # Cross-validation consistency
            cv_predictions = cross_val_predict(model, X, y, cv=5)
            cv_consistency = np.mean(predictions == cv_predictions) if is_classifier else \
                           1 - mean_squared_error(predictions, cv_predictions)
            report["prediction_analysis"]["cv_consistency"] = float(cv_consistency)

            if cv_consistency < 0.8:
                report["anomalies"].append({
                    "type": "inconsistent_predictions",
                    "description": f"Low cross-validation consistency: {cv_consistency:.3f}"
                })

        except Exception as e:
            logger.error(f"Prediction analysis error: {e}")
            report["error"] = str(e)

        # Calculate risk score
        report["risk_score"] = min(1.0, len(report["anomalies"]) * 0.25)

        logger.info(f"Prediction analysis complete. "
                   f"{len(report['anomalies'])} anomalies found")
        return report

    def detect_backdoor_triggers(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        n_perturbations: int = 100,
        perturbation_scale: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect potential backdoor triggers by analyzing model sensitivity
        to specific input patterns.

        Args:
            model: Model to analyze
            X: Features
            y: True labels
            n_perturbations: Number of perturbation tests
            perturbation_scale: Scale of perturbations

        Returns:
            Backdoor detection report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "trigger_analysis": {},
            "suspicious_patterns": [],
            "risk_score": 0.0
        }

        np.random.seed(self.random_state)
        original_predictions = model.predict(X)

        # Test each feature for trigger behavior
        feature_sensitivities = {}

        for col_idx, col in enumerate(X.columns):
            prediction_changes = []

            for _ in range(n_perturbations):
                X_perturbed = X.copy()
                # Apply specific perturbation pattern
                perturbation = np.random.uniform(
                    -perturbation_scale,
                    perturbation_scale,
                    size=len(X)
                ) * X[col].std()
                X_perturbed[col] = X[col] + perturbation

                new_predictions = model.predict(X_perturbed)

                if hasattr(model, 'predict_proba'):
                    # For classifiers, measure probability change
                    orig_proba = model.predict_proba(X)
                    new_proba = model.predict_proba(X_perturbed)
                    change = np.mean(np.abs(orig_proba - new_proba))
                else:
                    # For regressors, measure prediction change
                    change = np.mean(np.abs(original_predictions - new_predictions))

                prediction_changes.append(change)

            avg_sensitivity = np.mean(prediction_changes)
            std_sensitivity = np.std(prediction_changes)

            feature_sensitivities[col] = {
                "avg_sensitivity": float(avg_sensitivity),
                "std_sensitivity": float(std_sensitivity),
                "max_sensitivity": float(max(prediction_changes))
            }

        report["trigger_analysis"]["feature_sensitivities"] = feature_sensitivities

        # Identify suspiciously sensitive features
        sensitivities = [f["avg_sensitivity"] for f in feature_sensitivities.values()]
        mean_sens = np.mean(sensitivities)
        std_sens = np.std(sensitivities)

        for col, sens in feature_sensitivities.items():
            z_score = (sens["avg_sensitivity"] - mean_sens) / std_sens if std_sens > 0 else 0
            if z_score > 2.5:  # Unusually sensitive
                report["suspicious_patterns"].append({
                    "feature": col,
                    "sensitivity": sens["avg_sensitivity"],
                    "z_score": float(z_score),
                    "reason": "abnormally_high_sensitivity"
                })

        # Test for specific trigger patterns
        # Test extreme value triggers
        for col in X.columns:
            X_triggered = X.copy()
            X_triggered[col] = X[col].max() * 2  # Extreme value

            triggered_predictions = model.predict(X_triggered)
            trigger_effect = np.mean(original_predictions != triggered_predictions) \
                if hasattr(model, 'classes_') else \
                np.mean(np.abs(original_predictions - triggered_predictions))

            if trigger_effect > 0.5:
                report["suspicious_patterns"].append({
                    "feature": col,
                    "trigger_type": "extreme_value",
                    "trigger_effect": float(trigger_effect),
                    "reason": "high_sensitivity_to_extreme_values"
                })

        # Calculate risk score
        n_suspicious = len(report["suspicious_patterns"])
        report["risk_score"] = min(1.0, n_suspicious * 0.2)

        logger.info(f"Backdoor analysis complete. "
                   f"{n_suspicious} suspicious patterns found")
        return report

    def compare_with_reference(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """
        Compare model behavior with reference (clean) model.

        Args:
            model: Model to test
            X: Features
            y: True labels

        Returns:
            Comparison report
        """
        if self.reference_model is None:
            return {"error": "No reference model set"}

        report = {
            "timestamp": datetime.now().isoformat(),
            "comparison_results": {},
            "discrepancies": [],
            "risk_score": 0.0
        }

        try:
            ref_predictions = self.reference_model.predict(X)
            test_predictions = model.predict(X)

            # Prediction agreement
            if hasattr(model, 'classes_'):
                agreement = np.mean(ref_predictions == test_predictions)
            else:
                agreement = 1 - mean_squared_error(
                    ref_predictions, test_predictions
                ) / np.var(y)

            report["comparison_results"]["prediction_agreement"] = float(agreement)

            # Performance comparison
            if hasattr(model, 'classes_'):
                ref_acc = accuracy_score(y, ref_predictions)
                test_acc = accuracy_score(y, test_predictions)
                perf_diff = test_acc - ref_acc
            else:
                ref_r2 = r2_score(y, ref_predictions)
                test_r2 = r2_score(y, test_predictions)
                perf_diff = test_r2 - ref_r2

            report["comparison_results"]["performance_difference"] = float(perf_diff)

            # Check for significant discrepancies
            if agreement < 0.9:
                report["discrepancies"].append({
                    "type": "low_prediction_agreement",
                    "value": float(agreement),
                    "threshold": 0.9
                })

            if abs(perf_diff) > self.performance_threshold:
                report["discrepancies"].append({
                    "type": "performance_deviation",
                    "value": float(perf_diff),
                    "threshold": self.performance_threshold
                })

            # Analyze disagreement cases
            disagreement_mask = ref_predictions != test_predictions
            if disagreement_mask.sum() > 0:
                disagreement_samples = X[disagreement_mask]
                report["comparison_results"]["disagreement_count"] = int(
                    disagreement_mask.sum()
                )

                # Check if disagreements are clustered
                if len(disagreement_samples) > 10:
                    from sklearn.cluster import DBSCAN
                    from sklearn.preprocessing import StandardScaler

                    scaler = StandardScaler()
                    scaled = scaler.fit_transform(disagreement_samples)
                    clusters = DBSCAN(eps=0.5, min_samples=3).fit_predict(scaled)

                    n_clusters = len(set(clusters) - {-1})
                    if n_clusters > 0:
                        report["discrepancies"].append({
                            "type": "clustered_disagreements",
                            "n_clusters": n_clusters,
                            "reason": "disagreements_form_clusters"
                        })

        except Exception as e:
            logger.error(f"Comparison error: {e}")
            report["error"] = str(e)

        # Calculate risk score
        report["risk_score"] = min(1.0, len(report["discrepancies"]) * 0.3)

        logger.info(f"Reference comparison complete. "
                   f"{len(report['discrepancies'])} discrepancies found")
        return report

    def run_full_analysis(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        is_classifier: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete model poisoning analysis.

        Args:
            model: Model to analyze
            X: Features
            y: True labels
            is_classifier: Whether model is a classifier

        Returns:
            Comprehensive analysis report
        """
        logger.info("Starting full model poisoning analysis...")

        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "model_info": {
                "type": type(model).__name__,
                "fingerprint": self.compute_model_fingerprint(model)
            },
            "weight_analysis": {},
            "prediction_analysis": {},
            "backdoor_analysis": {},
            "reference_comparison": {},
            "overall_risk_assessment": {}
        }

        # Weight analysis
        report["weight_analysis"] = self.analyze_weight_distribution(model)

        # Prediction analysis
        report["prediction_analysis"] = self.detect_prediction_anomalies(
            model, X, y, is_classifier
        )

        # Backdoor detection
        report["backdoor_analysis"] = self.detect_backdoor_triggers(
            model, X, y
        )

        # Reference comparison (if available)
        if self.reference_model is not None:
            report["reference_comparison"] = self.compare_with_reference(
                model, X, y
            )

        # Overall risk assessment
        risk_scores = [
            report["weight_analysis"].get("risk_score", 0),
            report["prediction_analysis"].get("risk_score", 0),
            report["backdoor_analysis"].get("risk_score", 0),
            report["reference_comparison"].get("risk_score", 0)
        ]

        overall_risk = np.mean([r for r in risk_scores if r > 0])

        risk_factors = []
        if report["weight_analysis"].get("risk_score", 0) > 0.3:
            risk_factors.append("weight_anomalies")
        if report["prediction_analysis"].get("risk_score", 0) > 0.3:
            risk_factors.append("prediction_anomalies")
        if report["backdoor_analysis"].get("risk_score", 0) > 0.3:
            risk_factors.append("backdoor_patterns")
        if report["reference_comparison"].get("risk_score", 0) > 0.3:
            risk_factors.append("reference_deviation")

        report["overall_risk_assessment"] = {
            "risk_score": float(overall_risk),
            "risk_level": "HIGH" if overall_risk > 0.6 else
                         "MEDIUM" if overall_risk > 0.3 else "LOW",
            "risk_factors": risk_factors,
            "recommendations": self._generate_recommendations(risk_factors)
        }

        logger.info(f"Model analysis complete. Risk level: "
                   f"{report['overall_risk_assessment']['risk_level']}")
        return report

    def _generate_recommendations(self, risk_factors: List[str]) -> List[str]:
        """Generate security recommendations based on detected risks."""
        recommendations = []

        if not risk_factors:
            recommendations.append(
                "No significant risks detected. Continue regular monitoring."
            )
            return recommendations

        if "weight_anomalies" in risk_factors:
            recommendations.extend([
                "Inspect model weights for abnormal values",
                "Verify training data integrity",
                "Consider retraining with validated data"
            ])

        if "prediction_anomalies" in risk_factors:
            recommendations.extend([
                "Analyze prediction distribution across classes",
                "Test model on holdout dataset",
                "Check for data leakage in training"
            ])

        if "backdoor_patterns" in risk_factors:
            recommendations.extend([
                "Investigate sensitive features",
                "Test with adversarial inputs",
                "Consider fine-tuning or Neural Cleanse",
                "Implement input sanitization"
            ])

        if "reference_deviation" in risk_factors:
            recommendations.extend([
                "Compare training procedures",
                "Verify data pipeline integrity",
                "Check for unauthorized model modifications"
            ])

        return recommendations


if __name__ == "__main__":
    # Demo with wine quality data
    from sklearn.datasets import load_wine
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    print("=== Model Poisoning Detection Demo ===\n")

    # Load sample data
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Initialize detector
    detector = ModelPoisoningDetector()

    # Run full analysis
    report = detector.run_full_analysis(model, X_test, y_test, is_classifier=True)

    print("\n=== Analysis Results ===")
    print(f"Model Type: {report['model_info']['type']}")
    print(f"Model Fingerprint: {report['model_info']['fingerprint'][:32]}...")
    print(f"\nOverall Risk Level: {report['overall_risk_assessment']['risk_level']}")
    print(f"Risk Score: {report['overall_risk_assessment']['risk_score']:.2f}")

    if report['overall_risk_assessment']['risk_factors']:
        print(f"Risk Factors: {', '.join(report['overall_risk_assessment']['risk_factors'])}")

    print("\n=== Recommendations ===")
    for rec in report['overall_risk_assessment']['recommendations']:
        print(f"  - {rec}")
