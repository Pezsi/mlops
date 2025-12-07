"""
Model Robustness Testing Module (Lecke 115-116)

This module provides adversarial robustness testing using:
- CleverHans (when available)
- Adversarial Robustness Toolbox (ART) (when available)
- Custom perturbation testing

Features:
- FGSM (Fast Gradient Sign Method) attacks
- PGD (Projected Gradient Descent) attacks
- Feature perturbation testing
- Robustness metrics calculation

Reference: OWASP ML Security Top 10 - ML01:2023 Input Manipulation
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports for adversarial libraries
try:
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
    from art.estimators.classification import SklearnClassifier
    from art.estimators.regression import ScikitlearnRegressor
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    logger.warning("ART not available. Install with: pip install adversarial-robustness-toolbox")

try:
    import cleverhans  # noqa: F401
    CLEVERHANS_AVAILABLE = True
except ImportError:
    CLEVERHANS_AVAILABLE = False
    logger.warning("CleverHans not available. Install with: pip install cleverhans")


class AdversarialTester:
    """
    Adversarial robustness testing for ML models.

    Tests model resilience against:
    1. Perturbation attacks (noise injection)
    2. Gradient-based attacks (FGSM, PGD)
    3. Feature manipulation
    4. Boundary attacks
    """

    def __init__(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        task_type: str = "regression",
        random_state: int = 42
    ):
        """
        Initialize Adversarial Tester.

        Args:
            model: Trained model to test
            feature_names: List of feature names
            task_type: Either "regression" or "classification"
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.feature_names = feature_names
        self.task_type = task_type
        self.random_state = random_state
        np.random.seed(random_state)

        self.art_classifier = None
        self.scaler = None

    def initialize_art(
        self,
        X_train: np.ndarray,
        clip_values: Optional[Tuple[float, float]] = None
    ) -> bool:
        """
        Initialize ART wrapper for the model.

        Args:
            X_train: Training data for normalization
            clip_values: Min/max values for inputs

        Returns:
            True if initialization successful
        """
        if not ART_AVAILABLE:
            logger.error("ART not available")
            return False

        try:
            if clip_values is None:
                clip_values = (float(X_train.min()), float(X_train.max()))

            if self.task_type == "classification":
                self.art_classifier = SklearnClassifier(
                    model=self.model,
                    clip_values=clip_values
                )
            else:
                self.art_classifier = ScikitlearnRegressor(
                    model=self.model,
                    clip_values=clip_values
                )

            logger.info("ART wrapper initialized")
            return True

        except Exception as e:
            logger.error(f"ART initialization error: {e}")
            return False

    def perturbation_attack(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        epsilon_values: List[float] = [0.01, 0.05, 0.1, 0.2, 0.5],
        n_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Test model robustness with random noise perturbations.

        Args:
            X: Features
            y: True labels
            epsilon_values: List of perturbation magnitudes
            n_samples: Number of samples to test

        Returns:
            Perturbation attack report
        """
        if n_samples:
            indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
            X_test = X.iloc[indices]
            y_test = y.iloc[indices]
        else:
            X_test = X
            y_test = y

        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_test)

        # Original predictions
        original_preds = self.model.predict(X_test)

        if self.task_type == "classification":
            original_metric = accuracy_score(y_test, original_preds)
            metric_name = "accuracy"
        else:
            original_metric = r2_score(y_test, original_preds)
            metric_name = "r2_score"

        report = {
            "timestamp": datetime.now().isoformat(),
            "attack_type": "random_perturbation",
            "n_samples": len(X_test),
            "original_metric": float(original_metric),
            "metric_name": metric_name,
            "perturbation_results": []
        }

        for epsilon in epsilon_values:
            # Add random noise
            noise = np.random.normal(0, epsilon, X_scaled.shape)
            X_perturbed = scaler.inverse_transform(X_scaled + noise)
            X_perturbed_df = pd.DataFrame(X_perturbed, columns=X_test.columns)

            # Predictions on perturbed data
            perturbed_preds = self.model.predict(X_perturbed_df)

            if self.task_type == "classification":
                perturbed_metric = accuracy_score(y_test, perturbed_preds)
                attack_success = np.mean(original_preds != perturbed_preds)
            else:
                perturbed_metric = r2_score(y_test, perturbed_preds)
                # Attack success for regression: predictions changed significantly
                pred_change = np.abs(original_preds - perturbed_preds)
                attack_success = np.mean(pred_change > np.std(original_preds))

            robustness = perturbed_metric / original_metric if original_metric > 0 else 0

            result = {
                "epsilon": epsilon,
                "perturbed_metric": float(perturbed_metric),
                "metric_degradation": float(original_metric - perturbed_metric),
                "attack_success_rate": float(attack_success),
                "robustness_ratio": float(robustness)
            }

            report["perturbation_results"].append(result)

        # Calculate overall robustness score
        avg_robustness = np.mean([
            r["robustness_ratio"] for r in report["perturbation_results"]
        ])
        report["overall_robustness_score"] = float(avg_robustness)
        report["robustness_grade"] = self._grade_robustness(avg_robustness)

        logger.info(f"Perturbation attack complete. "
                   f"Robustness: {avg_robustness:.3f} ({report['robustness_grade']})")
        return report

    def fgsm_attack(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        epsilon: float = 0.1
    ) -> Dict[str, Any]:
        """
        Fast Gradient Sign Method (FGSM) attack using ART.

        Args:
            X: Features
            y: True labels
            epsilon: Perturbation magnitude

        Returns:
            FGSM attack report
        """
        if not ART_AVAILABLE or self.art_classifier is None:
            # Fallback to custom implementation
            return self._custom_fgsm(X, y, epsilon)

        try:
            X_array = X.values.astype(np.float32)
            y_array = y.values

            # Create FGSM attack
            fgsm = FastGradientMethod(
                estimator=self.art_classifier,
                eps=epsilon,
                targeted=False
            )

            # Generate adversarial examples
            X_adv = fgsm.generate(x=X_array)
            X_adv_df = pd.DataFrame(X_adv, columns=X.columns)

            # Evaluate
            original_preds = self.model.predict(X)
            adversarial_preds = self.model.predict(X_adv_df)

            if self.task_type == "classification":
                original_acc = accuracy_score(y, original_preds)
                adversarial_acc = accuracy_score(y, adversarial_preds)
                attack_success = np.mean(original_preds != adversarial_preds)

                report = {
                    "timestamp": datetime.now().isoformat(),
                    "attack_type": "FGSM",
                    "epsilon": epsilon,
                    "n_samples": len(X),
                    "original_accuracy": float(original_acc),
                    "adversarial_accuracy": float(adversarial_acc),
                    "attack_success_rate": float(attack_success),
                    "robustness_ratio": float(adversarial_acc / original_acc)
                    if original_acc > 0 else 0
                }
            else:
                original_r2 = r2_score(y, original_preds)
                adversarial_r2 = r2_score(y, adversarial_preds)
                pred_change = np.mean(np.abs(original_preds - adversarial_preds))

                report = {
                    "timestamp": datetime.now().isoformat(),
                    "attack_type": "FGSM",
                    "epsilon": epsilon,
                    "n_samples": len(X),
                    "original_r2": float(original_r2),
                    "adversarial_r2": float(adversarial_r2),
                    "avg_prediction_change": float(pred_change),
                    "robustness_ratio": float(adversarial_r2 / original_r2)
                    if original_r2 > 0 else 0
                }

            logger.info(f"FGSM attack complete. "
                       f"Robustness ratio: {report['robustness_ratio']:.3f}")
            return report

        except Exception as e:
            logger.error(f"FGSM attack error: {e}")
            return {"error": str(e)}

    def _custom_fgsm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        epsilon: float
    ) -> Dict[str, Any]:
        """
        Custom FGSM implementation using numerical gradients.

        For sklearn models, we approximate gradients using finite differences.
        """
        logger.info("Using custom FGSM implementation (numerical gradients)")

        X_array = X.values.astype(np.float64)
        original_preds = self.model.predict(X)

        # Compute numerical gradients
        gradients = np.zeros_like(X_array)
        delta = 1e-5

        for i in range(X_array.shape[1]):
            X_plus = X_array.copy()
            X_minus = X_array.copy()
            X_plus[:, i] += delta
            X_minus[:, i] -= delta

            pred_plus = self.model.predict(pd.DataFrame(X_plus, columns=X.columns))
            pred_minus = self.model.predict(pd.DataFrame(X_minus, columns=X.columns))

            if self.task_type == "classification":
                # For classification, use loss approximation
                gradients[:, i] = (pred_plus != y.values).astype(float) - \
                                 (pred_minus != y.values).astype(float)
            else:
                # For regression, use prediction difference
                gradients[:, i] = (pred_plus - pred_minus) / (2 * delta)

        # FGSM: perturb in direction of gradient sign
        X_adv = X_array + epsilon * np.sign(gradients)
        X_adv_df = pd.DataFrame(X_adv, columns=X.columns)

        adversarial_preds = self.model.predict(X_adv_df)

        if self.task_type == "classification":
            original_acc = accuracy_score(y, original_preds)
            adversarial_acc = accuracy_score(y, adversarial_preds)
            attack_success = np.mean(original_preds != adversarial_preds)

            return {
                "timestamp": datetime.now().isoformat(),
                "attack_type": "FGSM_custom",
                "epsilon": epsilon,
                "n_samples": len(X),
                "original_accuracy": float(original_acc),
                "adversarial_accuracy": float(adversarial_acc),
                "attack_success_rate": float(attack_success),
                "robustness_ratio": float(adversarial_acc / original_acc)
                if original_acc > 0 else 0
            }
        else:
            original_r2 = r2_score(y, original_preds)
            adversarial_r2 = r2_score(y, adversarial_preds)
            pred_change = np.mean(np.abs(original_preds - adversarial_preds))

            return {
                "timestamp": datetime.now().isoformat(),
                "attack_type": "FGSM_custom",
                "epsilon": epsilon,
                "n_samples": len(X),
                "original_r2": float(original_r2),
                "adversarial_r2": float(adversarial_r2),
                "avg_prediction_change": float(pred_change),
                "robustness_ratio": float(adversarial_r2 / original_r2)
                if original_r2 > 0 else 0
            }

    def pgd_attack(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        epsilon: float = 0.1,
        epsilon_step: float = 0.01,
        max_iter: int = 40
    ) -> Dict[str, Any]:
        """
        Projected Gradient Descent (PGD) attack.

        Args:
            X: Features
            y: True labels
            epsilon: Maximum perturbation
            epsilon_step: Step size per iteration
            max_iter: Maximum iterations

        Returns:
            PGD attack report
        """
        if not ART_AVAILABLE or self.art_classifier is None:
            # Fallback to iterative perturbation
            return self._custom_pgd(X, y, epsilon, epsilon_step, max_iter)

        try:
            X_array = X.values.astype(np.float32)

            # Create PGD attack
            pgd = ProjectedGradientDescent(
                estimator=self.art_classifier,
                eps=epsilon,
                eps_step=epsilon_step,
                max_iter=max_iter,
                targeted=False
            )

            # Generate adversarial examples
            X_adv = pgd.generate(x=X_array)
            X_adv_df = pd.DataFrame(X_adv, columns=X.columns)

            # Evaluate
            original_preds = self.model.predict(X)
            adversarial_preds = self.model.predict(X_adv_df)

            if self.task_type == "classification":
                original_acc = accuracy_score(y, original_preds)
                adversarial_acc = accuracy_score(y, adversarial_preds)

                report = {
                    "timestamp": datetime.now().isoformat(),
                    "attack_type": "PGD",
                    "epsilon": epsilon,
                    "epsilon_step": epsilon_step,
                    "max_iter": max_iter,
                    "n_samples": len(X),
                    "original_accuracy": float(original_acc),
                    "adversarial_accuracy": float(adversarial_acc),
                    "attack_success_rate": float(np.mean(original_preds != adversarial_preds)),
                    "robustness_ratio": float(adversarial_acc / original_acc)
                    if original_acc > 0 else 0
                }
            else:
                original_r2 = r2_score(y, original_preds)
                adversarial_r2 = r2_score(y, adversarial_preds)

                report = {
                    "timestamp": datetime.now().isoformat(),
                    "attack_type": "PGD",
                    "epsilon": epsilon,
                    "epsilon_step": epsilon_step,
                    "max_iter": max_iter,
                    "n_samples": len(X),
                    "original_r2": float(original_r2),
                    "adversarial_r2": float(adversarial_r2),
                    "robustness_ratio": float(adversarial_r2 / original_r2)
                    if original_r2 > 0 else 0
                }

            logger.info(f"PGD attack complete. "
                       f"Robustness ratio: {report['robustness_ratio']:.3f}")
            return report

        except Exception as e:
            logger.error(f"PGD attack error: {e}")
            return {"error": str(e)}

    def _custom_pgd(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        epsilon: float,
        epsilon_step: float,
        max_iter: int
    ) -> Dict[str, Any]:
        """Custom PGD implementation using iterative perturbation."""
        logger.info("Using custom PGD implementation")

        X_array = X.values.astype(np.float64)
        original_preds = self.model.predict(X)
        X_adv = X_array.copy()

        for _ in range(max_iter):
            # Compute gradients (similar to FGSM)
            gradients = np.zeros_like(X_adv)
            delta = 1e-5

            for i in range(X_adv.shape[1]):
                X_plus = X_adv.copy()
                X_minus = X_adv.copy()
                X_plus[:, i] += delta
                X_minus[:, i] -= delta

                pred_plus = self.model.predict(pd.DataFrame(X_plus, columns=X.columns))
                pred_minus = self.model.predict(pd.DataFrame(X_minus, columns=X.columns))

                if self.task_type == "classification":
                    gradients[:, i] = (pred_plus != y.values).astype(float) - \
                                     (pred_minus != y.values).astype(float)
                else:
                    gradients[:, i] = (pred_plus - pred_minus) / (2 * delta)

            # Update with step
            X_adv = X_adv + epsilon_step * np.sign(gradients)

            # Project back to epsilon ball
            perturbation = X_adv - X_array
            perturbation = np.clip(perturbation, -epsilon, epsilon)
            X_adv = X_array + perturbation

        X_adv_df = pd.DataFrame(X_adv, columns=X.columns)
        adversarial_preds = self.model.predict(X_adv_df)

        if self.task_type == "classification":
            original_acc = accuracy_score(y, original_preds)
            adversarial_acc = accuracy_score(y, adversarial_preds)

            return {
                "timestamp": datetime.now().isoformat(),
                "attack_type": "PGD_custom",
                "epsilon": epsilon,
                "max_iter": max_iter,
                "n_samples": len(X),
                "original_accuracy": float(original_acc),
                "adversarial_accuracy": float(adversarial_acc),
                "attack_success_rate": float(np.mean(original_preds != adversarial_preds)),
                "robustness_ratio": float(adversarial_acc / original_acc)
                if original_acc > 0 else 0
            }
        else:
            original_r2 = r2_score(y, original_preds)
            adversarial_r2 = r2_score(y, adversarial_preds)

            return {
                "timestamp": datetime.now().isoformat(),
                "attack_type": "PGD_custom",
                "epsilon": epsilon,
                "max_iter": max_iter,
                "n_samples": len(X),
                "original_r2": float(original_r2),
                "adversarial_r2": float(adversarial_r2),
                "robustness_ratio": float(adversarial_r2 / original_r2)
                if original_r2 > 0 else 0
            }

    def feature_importance_attack(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_k_features: int = 3,
        perturbation_magnitude: float = 0.5
    ) -> Dict[str, Any]:
        """
        Attack by perturbing most important features.

        Args:
            X: Features
            y: True labels
            top_k_features: Number of top features to perturb
            perturbation_magnitude: Perturbation scale (relative to std)

        Returns:
            Feature importance attack report
        """
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            # Use permutation importance fallback
            from sklearn.inspection import permutation_importance
            perm_importance = permutation_importance(
                self.model, X, y, n_repeats=10, random_state=self.random_state
            )
            importance = perm_importance.importances_mean

        # Find top-k important features
        top_indices = np.argsort(importance)[-top_k_features:]
        top_features = [self.feature_names[i] for i in top_indices]

        original_preds = self.model.predict(X)

        # Perturb top features
        X_perturbed = X.copy()
        for col in top_features:
            std = X[col].std()
            X_perturbed[col] = X[col] + perturbation_magnitude * std * \
                np.random.choice([-1, 1], size=len(X))

        perturbed_preds = self.model.predict(X_perturbed)

        if self.task_type == "classification":
            original_acc = accuracy_score(y, original_preds)
            perturbed_acc = accuracy_score(y, perturbed_preds)
            attack_success = np.mean(original_preds != perturbed_preds)

            report = {
                "timestamp": datetime.now().isoformat(),
                "attack_type": "feature_importance_attack",
                "top_k_features": top_k_features,
                "attacked_features": top_features,
                "perturbation_magnitude": perturbation_magnitude,
                "n_samples": len(X),
                "original_accuracy": float(original_acc),
                "perturbed_accuracy": float(perturbed_acc),
                "attack_success_rate": float(attack_success),
                "robustness_ratio": float(perturbed_acc / original_acc)
                if original_acc > 0 else 0
            }
        else:
            original_r2 = r2_score(y, original_preds)
            perturbed_r2 = r2_score(y, perturbed_preds)

            report = {
                "timestamp": datetime.now().isoformat(),
                "attack_type": "feature_importance_attack",
                "top_k_features": top_k_features,
                "attacked_features": top_features,
                "perturbation_magnitude": perturbation_magnitude,
                "n_samples": len(X),
                "original_r2": float(original_r2),
                "perturbed_r2": float(perturbed_r2),
                "robustness_ratio": float(perturbed_r2 / original_r2)
                if original_r2 > 0 else 0
            }

        logger.info(f"Feature importance attack on {top_features}. "
                   f"Robustness: {report['robustness_ratio']:.3f}")
        return report

    def boundary_attack(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_boundary_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Test robustness near decision boundaries.

        Args:
            X: Features
            y: True labels
            n_boundary_samples: Number of samples to generate

        Returns:
            Boundary attack report
        """
        if self.task_type != "classification":
            return {"error": "Boundary attack only for classification"}

        # Find samples that are misclassified with small perturbations
        original_preds = self.model.predict(X)

        boundary_samples = []
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for i in range(len(X)):
            if len(boundary_samples) >= n_boundary_samples:
                break

            # Try small perturbations
            for eps in [0.01, 0.05, 0.1]:
                noise = np.random.normal(0, eps, X_scaled.shape[1])
                x_perturbed = X_scaled[i] + noise
                x_perturbed_original = scaler.inverse_transform([x_perturbed])
                x_perturbed_df = pd.DataFrame(x_perturbed_original, columns=X.columns)

                new_pred = self.model.predict(x_perturbed_df)[0]

                if new_pred != original_preds[i]:
                    boundary_samples.append({
                        "original_index": i,
                        "epsilon": eps,
                        "original_label": int(original_preds[i]),
                        "flipped_label": int(new_pred)
                    })
                    break

        # Analyze boundary samples
        report = {
            "timestamp": datetime.now().isoformat(),
            "attack_type": "boundary_attack",
            "n_samples_tested": len(X),
            "n_boundary_samples_found": len(boundary_samples),
            "boundary_vulnerability_ratio": float(len(boundary_samples) / len(X)),
            "avg_epsilon_to_flip": float(np.mean([s["epsilon"] for s in boundary_samples]))
            if boundary_samples else None,
            "sample_analysis": boundary_samples[:20],
            "class_vulnerability": {}
        }

        # Analyze by class
        for sample in boundary_samples:
            label = str(sample["original_label"])
            report["class_vulnerability"][label] = \
                report["class_vulnerability"].get(label, 0) + 1

        # Robustness interpretation
        vuln_ratio = report["boundary_vulnerability_ratio"]
        if vuln_ratio < 0.1:
            report["robustness_assessment"] = "HIGH - Few samples near decision boundary"
        elif vuln_ratio < 0.3:
            report["robustness_assessment"] = "MEDIUM - Some boundary sensitivity"
        else:
            report["robustness_assessment"] = "LOW - Many samples near decision boundary"

        logger.info(f"Boundary attack complete. "
                   f"Vulnerability ratio: {vuln_ratio:.3f}")
        return report

    def run_full_robustness_test(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_samples: int = 200
    ) -> Dict[str, Any]:
        """
        Run comprehensive robustness testing.

        Args:
            X: Features
            y: True labels
            n_samples: Number of samples to use

        Returns:
            Comprehensive robustness report
        """
        logger.info("Starting comprehensive robustness testing...")

        # Sample data if needed
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_test = X.iloc[indices]
            y_test = y.iloc[indices]
        else:
            X_test = X
            y_test = y

        report = {
            "timestamp": datetime.now().isoformat(),
            "model_type": type(self.model).__name__,
            "task_type": self.task_type,
            "n_samples_tested": len(X_test),
            "attacks": {}
        }

        # Initialize ART if available
        if ART_AVAILABLE:
            self.initialize_art(X_test.values)

        # Perturbation attack
        report["attacks"]["perturbation"] = self.perturbation_attack(
            X_test, y_test,
            epsilon_values=[0.01, 0.05, 0.1, 0.2]
        )

        # FGSM attack
        report["attacks"]["fgsm"] = self.fgsm_attack(X_test, y_test, epsilon=0.1)

        # PGD attack
        report["attacks"]["pgd"] = self.pgd_attack(
            X_test, y_test, epsilon=0.1, max_iter=20
        )

        # Feature importance attack
        report["attacks"]["feature_importance"] = self.feature_importance_attack(
            X_test, y_test, top_k_features=3
        )

        # Boundary attack (classification only)
        if self.task_type == "classification":
            report["attacks"]["boundary"] = self.boundary_attack(X_test, y_test)

        # Overall robustness score
        robustness_scores = []
        for attack_name, attack_result in report["attacks"].items():
            if "robustness_ratio" in attack_result:
                robustness_scores.append(attack_result["robustness_ratio"])

        if robustness_scores:
            avg_robustness = np.mean(robustness_scores)
            min_robustness = np.min(robustness_scores)

            report["overall_robustness"] = {
                "average_score": float(avg_robustness),
                "minimum_score": float(min_robustness),
                "grade": self._grade_robustness(avg_robustness),
                "recommendations": self._generate_recommendations(robustness_scores)
            }

        logger.info(f"Robustness testing complete. "
                   f"Overall grade: {report['overall_robustness']['grade']}")
        return report

    def _grade_robustness(self, score: float) -> str:
        """Grade robustness score."""
        if score >= 0.9:
            return "A - Excellent"
        elif score >= 0.8:
            return "B - Good"
        elif score >= 0.7:
            return "C - Fair"
        elif score >= 0.6:
            return "D - Poor"
        else:
            return "F - Critical"

    def _generate_recommendations(self, scores: List[float]) -> List[str]:
        """Generate robustness recommendations."""
        recommendations = []

        min_score = min(scores) if scores else 1.0

        if min_score < 0.6:
            recommendations.extend([
                "Consider adversarial training to improve robustness",
                "Implement input validation and sanitization",
                "Use ensemble models for better stability"
            ])
        elif min_score < 0.8:
            recommendations.extend([
                "Add input noise during training for regularization",
                "Monitor prediction confidence in production"
            ])
        else:
            recommendations.append(
                "Model shows good robustness. Continue monitoring."
            )

        return recommendations


if __name__ == "__main__":
    # Demo with wine quality data
    from sklearn.datasets import load_wine
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    print("=== Adversarial Robustness Testing Demo ===\n")
    print(f"ART Available: {ART_AVAILABLE}")
    print(f"CleverHans Available: {CLEVERHANS_AVAILABLE}\n")

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

    # Initialize tester
    tester = AdversarialTester(
        model=model,
        feature_names=wine.feature_names,
        task_type="classification"
    )

    # Run comprehensive test
    report = tester.run_full_robustness_test(X_test, y_test, n_samples=50)

    print("\n=== Robustness Results ===")
    print(f"Model: {report['model_type']}")
    print(f"Samples tested: {report['n_samples_tested']}")

    print("\n=== Attack Results ===")
    for attack_name, attack_result in report["attacks"].items():
        if "robustness_ratio" in attack_result:
            print(f"  {attack_name}: {attack_result['robustness_ratio']:.3f}")

    print(f"\n=== Overall Robustness ===")
    overall = report["overall_robustness"]
    print(f"Average Score: {overall['average_score']:.3f}")
    print(f"Minimum Score: {overall['minimum_score']:.3f}")
    print(f"Grade: {overall['grade']}")

    print("\n=== Recommendations ===")
    for rec in overall["recommendations"]:
        print(f"  - {rec}")
