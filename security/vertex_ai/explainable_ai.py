"""
Vertex AI Explainable AI Integration (Lecke 114)

This module provides local explainability features that can integrate with
Vertex AI Explainable AI when deployed to GCP, plus local alternatives.

Features:
- SHAP (SHapley Additive exPlanations) integration
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature attribution analysis
- Vertex AI Explainable AI configuration

Reference: https://cloud.google.com/vertex-ai/docs/explainable-ai/overview
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from typing import Dict, List, Optional, Any, Union
import logging
import json
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports for explainability libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not available. Install with: pip install lime")


class ExplainableAI:
    """
    Explainable AI for ML models with Vertex AI integration support.

    Provides multiple explanation methods:
    1. SHAP values for global and local explanations
    2. LIME for local, human-readable explanations
    3. Permutation importance
    4. Feature attribution analysis
    """

    def __init__(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        task_type: str = "regression",
        random_state: int = 42
    ):
        """
        Initialize Explainable AI.

        Args:
            model: Trained model to explain
            feature_names: List of feature names
            task_type: Either "regression" or "classification"
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.feature_names = feature_names
        self.task_type = task_type
        self.random_state = random_state

        self.shap_explainer = None
        self.lime_explainer = None
        self.baseline_shap_values = None

    def initialize_shap(
        self,
        X_background: pd.DataFrame,
        explainer_type: str = "auto"
    ) -> bool:
        """
        Initialize SHAP explainer.

        Args:
            X_background: Background data for SHAP
            explainer_type: Type of explainer (auto, tree, kernel, linear)

        Returns:
            True if initialization successful
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP not available")
            return False

        try:
            if explainer_type == "auto":
                # Automatically select best explainer
                model_type = type(self.model).__name__.lower()
                if any(t in model_type for t in ['forest', 'tree', 'boost', 'xgb', 'lgbm']):
                    explainer_type = "tree"
                elif 'linear' in model_type or 'logistic' in model_type:
                    explainer_type = "linear"
                else:
                    explainer_type = "kernel"

            if explainer_type == "tree":
                self.shap_explainer = shap.TreeExplainer(self.model)
            elif explainer_type == "linear":
                self.shap_explainer = shap.LinearExplainer(
                    self.model,
                    X_background
                )
            else:  # kernel
                # Use smaller sample for kernel explainer (computationally expensive)
                background = shap.sample(X_background, min(100, len(X_background)))
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict,
                    background
                )

            logger.info(f"SHAP {explainer_type} explainer initialized")
            return True

        except Exception as e:
            logger.error(f"SHAP initialization error: {e}")
            return False

    def initialize_lime(
        self,
        X_train: pd.DataFrame,
        categorical_features: Optional[List[int]] = None
    ) -> bool:
        """
        Initialize LIME explainer.

        Args:
            X_train: Training data for LIME
            categorical_features: Indices of categorical features

        Returns:
            True if initialization successful
        """
        if not LIME_AVAILABLE:
            logger.error("LIME not available")
            return False

        try:
            mode = "classification" if self.task_type == "classification" else "regression"

            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=self.feature_names,
                class_names=None,  # Will be set during prediction
                mode=mode,
                categorical_features=categorical_features,
                random_state=self.random_state
            )

            logger.info("LIME explainer initialized")
            return True

        except Exception as e:
            logger.error(f"LIME initialization error: {e}")
            return False

    def explain_shap(
        self,
        X: pd.DataFrame,
        check_additivity: bool = False
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations.

        Args:
            X: Data to explain
            check_additivity: Whether to check SHAP additivity

        Returns:
            SHAP explanation report
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return {"error": "SHAP not initialized"}

        try:
            shap_values = self.shap_explainer.shap_values(X, check_additivity=check_additivity)

            # Handle different output formats
            if isinstance(shap_values, list):
                # Multi-class classification
                shap_values = np.array(shap_values)
                # Use mean absolute for feature importance
                mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 1))
            else:
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

            # Feature importance ranking
            importance_ranking = sorted(
                zip(self.feature_names, mean_abs_shap),
                key=lambda x: x[1],
                reverse=True
            )

            report = {
                "timestamp": datetime.now().isoformat(),
                "method": "SHAP",
                "n_samples": len(X),
                "feature_importance": {
                    name: float(imp) for name, imp in importance_ranking
                },
                "top_features": [name for name, _ in importance_ranking[:5]],
                "shap_values_shape": list(np.array(shap_values).shape),
                "expected_value": float(self.shap_explainer.expected_value)
                if not isinstance(self.shap_explainer.expected_value, (list, np.ndarray))
                else [float(v) for v in self.shap_explainer.expected_value]
            }

            # Store for comparison
            self.baseline_shap_values = shap_values

            logger.info(f"SHAP explanations generated for {len(X)} samples")
            return report

        except Exception as e:
            logger.error(f"SHAP explanation error: {e}")
            return {"error": str(e)}

    def explain_lime(
        self,
        sample: pd.Series,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single sample.

        Args:
            sample: Single sample to explain
            num_features: Number of features to include

        Returns:
            LIME explanation report
        """
        if not LIME_AVAILABLE or self.lime_explainer is None:
            return {"error": "LIME not initialized"}

        try:
            if self.task_type == "classification":
                predict_fn = self.model.predict_proba
            else:
                predict_fn = self.model.predict

            explanation = self.lime_explainer.explain_instance(
                sample.values,
                predict_fn,
                num_features=num_features
            )

            # Extract feature contributions
            feature_weights = explanation.as_list()

            report = {
                "timestamp": datetime.now().isoformat(),
                "method": "LIME",
                "prediction": float(self.model.predict([sample.values])[0]),
                "feature_contributions": [
                    {"feature": feat, "weight": float(weight)}
                    for feat, weight in feature_weights
                ],
                "intercept": float(explanation.intercept[0])
                if hasattr(explanation, 'intercept') else None,
                "local_prediction": float(explanation.local_pred[0])
                if hasattr(explanation, 'local_pred') else None,
                "score": float(explanation.score)
                if hasattr(explanation, 'score') else None
            }

            logger.info("LIME explanation generated")
            return report

        except Exception as e:
            logger.error(f"LIME explanation error: {e}")
            return {"error": str(e)}

    def compute_permutation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10,
        scoring: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute permutation feature importance.

        Args:
            X: Features
            y: Target
            n_repeats: Number of permutation repeats
            scoring: Scoring metric

        Returns:
            Permutation importance report
        """
        try:
            if scoring is None:
                scoring = "accuracy" if self.task_type == "classification" else "r2"

            result = permutation_importance(
                self.model, X, y,
                n_repeats=n_repeats,
                random_state=self.random_state,
                scoring=scoring,
                n_jobs=-1
            )

            importance_ranking = sorted(
                zip(self.feature_names, result.importances_mean, result.importances_std),
                key=lambda x: x[1],
                reverse=True
            )

            report = {
                "timestamp": datetime.now().isoformat(),
                "method": "permutation_importance",
                "scoring": scoring,
                "n_repeats": n_repeats,
                "feature_importance": {
                    name: {"mean": float(mean), "std": float(std)}
                    for name, mean, std in importance_ranking
                },
                "top_features": [name for name, _, _ in importance_ranking[:5]]
            }

            logger.info("Permutation importance computed")
            return report

        except Exception as e:
            logger.error(f"Permutation importance error: {e}")
            return {"error": str(e)}

    def detect_explanation_drift(
        self,
        X_new: pd.DataFrame,
        threshold: float = 0.2
    ) -> Dict[str, Any]:
        """
        Detect drift in model explanations compared to baseline.

        Args:
            X_new: New data to compare
            threshold: Drift detection threshold

        Returns:
            Explanation drift report
        """
        if self.baseline_shap_values is None:
            return {"error": "No baseline SHAP values. Run explain_shap first."}

        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return {"error": "SHAP not available"}

        try:
            new_shap_values = self.shap_explainer.shap_values(X_new)

            # Calculate baseline feature importance
            if isinstance(self.baseline_shap_values, list):
                baseline_importance = np.mean(
                    np.abs(np.array(self.baseline_shap_values)), axis=(0, 1)
                )
            else:
                baseline_importance = np.mean(np.abs(self.baseline_shap_values), axis=0)

            # Calculate new feature importance
            if isinstance(new_shap_values, list):
                new_importance = np.mean(np.abs(np.array(new_shap_values)), axis=(0, 1))
            else:
                new_importance = np.mean(np.abs(new_shap_values), axis=0)

            # Normalize
            baseline_norm = baseline_importance / np.sum(baseline_importance)
            new_norm = new_importance / np.sum(new_importance)

            # Calculate drift per feature
            feature_drift = {}
            drifted_features = []

            for i, name in enumerate(self.feature_names):
                drift = abs(new_norm[i] - baseline_norm[i])
                feature_drift[name] = {
                    "baseline_importance": float(baseline_norm[i]),
                    "new_importance": float(new_norm[i]),
                    "drift": float(drift),
                    "is_drifted": drift > threshold
                }
                if drift > threshold:
                    drifted_features.append(name)

            # Overall drift score
            overall_drift = np.mean(np.abs(new_norm - baseline_norm))

            report = {
                "timestamp": datetime.now().isoformat(),
                "n_samples_baseline": len(self.baseline_shap_values),
                "n_samples_new": len(X_new),
                "feature_drift": feature_drift,
                "drifted_features": drifted_features,
                "overall_drift_score": float(overall_drift),
                "threshold": threshold,
                "drift_detected": overall_drift > threshold
            }

            logger.info(f"Explanation drift: {len(drifted_features)} features drifted, "
                       f"overall score: {overall_drift:.3f}")
            return report

        except Exception as e:
            logger.error(f"Explanation drift detection error: {e}")
            return {"error": str(e)}

    def generate_vertex_ai_config(
        self,
        X_sample: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate Vertex AI Explainable AI configuration.

        Args:
            X_sample: Sample data for baseline
            output_path: Optional path to save configuration

        Returns:
            Vertex AI explanation configuration
        """
        # Calculate feature baselines
        feature_baselines = X_sample.mean().to_dict()

        config = {
            "explanation_metadata": {
                "inputs": {
                    name: {
                        "input_baselines": [float(feature_baselines[name])],
                        "encoding": "bag_of_features",
                        "modality": "numeric"
                    }
                    for name in self.feature_names
                },
                "outputs": {
                    "quality": {
                        "output_tensor_name": "prediction"
                    }
                }
            },
            "explanation_parameters": {
                "sampled_shapley_attribution": {
                    "path_count": 10
                },
                "xrai_attribution": {
                    "step_count": 50
                },
                "integrated_gradients_attribution": {
                    "step_count": 50,
                    "smooth_grad_config": {
                        "noise_sigma": 0.1,
                        "noisy_sample_count": 3
                    }
                }
            }
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Vertex AI config saved to {output_path}")

        return config

    def get_comprehensive_report(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explainability report.

        Args:
            X: Features
            y: Target
            sample_idx: Sample index for local explanation

        Returns:
            Comprehensive explainability report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "model_type": type(self.model).__name__,
            "task_type": self.task_type,
            "n_features": len(self.feature_names),
            "n_samples": len(X),
            "explanations": {}
        }

        # SHAP global explanation
        if SHAP_AVAILABLE and self.shap_explainer is not None:
            report["explanations"]["shap"] = self.explain_shap(X)

        # LIME local explanation
        if LIME_AVAILABLE and self.lime_explainer is not None:
            report["explanations"]["lime"] = self.explain_lime(X.iloc[sample_idx])

        # Permutation importance
        report["explanations"]["permutation"] = self.compute_permutation_importance(X, y)

        # Built-in feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
            report["explanations"]["builtin"] = {
                "method": "tree_feature_importance",
                "feature_importance": {k: float(v) for k, v in importance.items()}
            }

        return report


if __name__ == "__main__":
    # Demo with wine quality data
    from sklearn.datasets import load_wine
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    print("=== Explainable AI Demo ===\n")

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

    # Initialize Explainable AI
    explainer = ExplainableAI(
        model=model,
        feature_names=wine.feature_names,
        task_type="classification"
    )

    # Initialize SHAP
    if SHAP_AVAILABLE:
        explainer.initialize_shap(X_train)

    # Initialize LIME
    if LIME_AVAILABLE:
        explainer.initialize_lime(X_train)

    # Get comprehensive report
    report = explainer.get_comprehensive_report(X_test, y_test)

    print(f"Model: {report['model_type']}")
    print(f"Task: {report['task_type']}")
    print(f"\nExplanation methods available: {list(report['explanations'].keys())}")

    if "shap" in report["explanations"]:
        print("\n=== SHAP Top Features ===")
        shap_report = report["explanations"]["shap"]
        if "top_features" in shap_report:
            for i, feat in enumerate(shap_report["top_features"], 1):
                importance = shap_report["feature_importance"][feat]
                print(f"  {i}. {feat}: {importance:.4f}")

    if "permutation" in report["explanations"]:
        print("\n=== Permutation Importance Top Features ===")
        perm_report = report["explanations"]["permutation"]
        if "top_features" in perm_report:
            for i, feat in enumerate(perm_report["top_features"], 1):
                imp = perm_report["feature_importance"][feat]
                print(f"  {i}. {feat}: {imp['mean']:.4f} (+/- {imp['std']:.4f})")

    # Generate Vertex AI config
    print("\n=== Vertex AI Configuration ===")
    vertex_config = explainer.generate_vertex_ai_config(X_train)
    print(f"Inputs configured: {len(vertex_config['explanation_metadata']['inputs'])}")
    print(f"Attribution methods: sampled_shapley, xrai, integrated_gradients")
