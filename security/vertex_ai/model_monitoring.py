"""
Vertex AI Model Monitoring Integration (Lecke 114)

This module provides model monitoring capabilities that can integrate with
Vertex AI Model Monitoring when deployed to GCP.

Features:
- Prediction drift detection
- Feature skew detection
- Training-serving skew detection
- Alerting configuration
- Local monitoring simulation

Reference: https://cloud.google.com/vertex-ai/docs/model-monitoring/overview
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from typing import Dict, List, Optional, Any, Callable
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Model monitoring for production ML systems with Vertex AI integration.

    Monitors:
    1. Prediction drift - changes in prediction distribution
    2. Feature drift - changes in input feature distribution
    3. Performance degradation - model accuracy/error changes
    4. Anomalous predictions - outlier detection in outputs
    """

    def __init__(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        task_type: str = "regression",
        window_size: int = 1000,
        drift_threshold: float = 0.1
    ):
        """
        Initialize Model Monitor.

        Args:
            model: Trained model to monitor
            feature_names: List of feature names
            task_type: Either "regression" or "classification"
            window_size: Size of sliding window for monitoring
            drift_threshold: Threshold for drift detection
        """
        self.model = model
        self.feature_names = feature_names
        self.task_type = task_type
        self.window_size = window_size
        self.drift_threshold = drift_threshold

        # Baseline statistics
        self.baseline_feature_stats: Dict[str, Dict] = {}
        self.baseline_prediction_stats: Dict[str, float] = {}

        # Sliding window buffers
        self.feature_buffer = deque(maxlen=window_size)
        self.prediction_buffer = deque(maxlen=window_size)
        self.ground_truth_buffer = deque(maxlen=window_size)

        # Alert history
        self.alerts: List[Dict] = []

        # Performance tracking
        self.performance_history: List[Dict] = []

    def set_baseline(
        self,
        X_baseline: pd.DataFrame,
        y_baseline: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Set baseline statistics from training data.

        Args:
            X_baseline: Training features
            y_baseline: Training labels (optional)

        Returns:
            Baseline statistics
        """
        # Feature statistics
        for col in self.feature_names:
            if col in X_baseline.columns:
                data = X_baseline[col]
                self.baseline_feature_stats[col] = {
                    "mean": float(data.mean()),
                    "std": float(data.std()),
                    "min": float(data.min()),
                    "max": float(data.max()),
                    "median": float(data.median()),
                    "q1": float(data.quantile(0.25)),
                    "q3": float(data.quantile(0.75)),
                    "histogram": np.histogram(
                        data, bins=20, density=True
                    )[0].tolist()
                }

        # Prediction statistics
        predictions = self.model.predict(X_baseline)
        self.baseline_prediction_stats = {
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
            "median": float(np.median(predictions))
        }

        if self.task_type == "classification":
            unique, counts = np.unique(predictions, return_counts=True)
            self.baseline_prediction_stats["class_distribution"] = {
                str(int(u)): int(c) for u, c in zip(unique, counts)
            }

        # Performance baseline
        if y_baseline is not None:
            if self.task_type == "classification":
                self.baseline_prediction_stats["accuracy"] = float(
                    accuracy_score(y_baseline, predictions)
                )
            else:
                self.baseline_prediction_stats["r2"] = float(
                    r2_score(y_baseline, predictions)
                )
                self.baseline_prediction_stats["mse"] = float(
                    mean_squared_error(y_baseline, predictions)
                )

        logger.info(f"Baseline set with {len(X_baseline)} samples")

        return {
            "feature_stats": self.baseline_feature_stats,
            "prediction_stats": self.baseline_prediction_stats,
            "n_samples": len(X_baseline),
            "timestamp": datetime.now().isoformat()
        }

    def log_prediction(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
        ground_truth: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Log predictions for monitoring.

        Args:
            features: Input features
            predictions: Model predictions
            ground_truth: Actual values (optional)

        Returns:
            Monitoring status
        """
        # Add to buffers
        for _, row in features.iterrows():
            self.feature_buffer.append(row.to_dict())

        for pred in predictions:
            self.prediction_buffer.append(pred)

        if ground_truth is not None:
            for gt in ground_truth:
                self.ground_truth_buffer.append(gt)

        # Check for drift if buffer is full
        status = {
            "logged": len(predictions),
            "buffer_size": len(self.prediction_buffer),
            "drift_detected": False,
            "alerts": []
        }

        if len(self.prediction_buffer) >= self.window_size:
            drift_report = self.check_drift()
            status["drift_detected"] = drift_report.get("drift_detected", False)
            status["alerts"] = drift_report.get("alerts", [])

        return status

    def check_drift(self) -> Dict[str, Any]:
        """
        Check for various types of drift.

        Returns:
            Drift detection report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "drift_detected": False,
            "feature_drift": {},
            "prediction_drift": {},
            "alerts": []
        }

        # Feature drift detection
        current_features = pd.DataFrame(list(self.feature_buffer))

        for col in self.feature_names:
            if col not in current_features.columns:
                continue
            if col not in self.baseline_feature_stats:
                continue

            baseline = self.baseline_feature_stats[col]
            current = current_features[col]

            # Calculate PSI (Population Stability Index)
            psi = self._calculate_psi(
                np.random.normal(baseline["mean"], baseline["std"], 1000),
                current.values
            )

            # KS test
            ks_stat, ks_pvalue = stats.ks_2samp(
                np.random.normal(baseline["mean"], baseline["std"], 1000),
                current.values
            )

            # Mean shift
            mean_shift = abs(current.mean() - baseline["mean"]) / baseline["std"] \
                if baseline["std"] > 0 else 0

            feature_drift = {
                "psi": float(psi),
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "mean_shift": float(mean_shift),
                "is_drifted": psi > self.drift_threshold or mean_shift > 2
            }

            report["feature_drift"][col] = feature_drift

            if feature_drift["is_drifted"]:
                alert = {
                    "type": "feature_drift",
                    "feature": col,
                    "severity": "HIGH" if psi > 0.25 else "MEDIUM",
                    "psi": float(psi),
                    "timestamp": datetime.now().isoformat()
                }
                report["alerts"].append(alert)
                self.alerts.append(alert)

        # Prediction drift detection
        current_predictions = np.array(list(self.prediction_buffer))

        pred_mean_shift = abs(
            np.mean(current_predictions) - self.baseline_prediction_stats["mean"]
        ) / self.baseline_prediction_stats["std"] \
            if self.baseline_prediction_stats["std"] > 0 else 0

        pred_std_ratio = np.std(current_predictions) / self.baseline_prediction_stats["std"] \
            if self.baseline_prediction_stats["std"] > 0 else 1

        report["prediction_drift"] = {
            "mean_shift": float(pred_mean_shift),
            "std_ratio": float(pred_std_ratio),
            "current_mean": float(np.mean(current_predictions)),
            "baseline_mean": float(self.baseline_prediction_stats["mean"]),
            "is_drifted": pred_mean_shift > 2 or abs(pred_std_ratio - 1) > 0.5
        }

        if report["prediction_drift"]["is_drifted"]:
            alert = {
                "type": "prediction_drift",
                "severity": "HIGH" if pred_mean_shift > 3 else "MEDIUM",
                "mean_shift": float(pred_mean_shift),
                "timestamp": datetime.now().isoformat()
            }
            report["alerts"].append(alert)
            self.alerts.append(alert)

        # Set overall drift flag
        report["drift_detected"] = (
            any(f["is_drifted"] for f in report["feature_drift"].values()) or
            report["prediction_drift"]["is_drifted"]
        )

        if report["drift_detected"]:
            logger.warning(f"Drift detected! {len(report['alerts'])} alerts generated")

        return report

    def _calculate_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Calculate Population Stability Index."""
        bins = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        baseline_counts = np.histogram(baseline, bins=bins)[0]
        current_counts = np.histogram(current, bins=bins)[0]

        baseline_props = np.clip(baseline_counts / len(baseline), 0.001, None)
        current_props = np.clip(current_counts / len(current), 0.001, None)

        psi = np.sum(
            (current_props - baseline_props) *
            np.log(current_props / baseline_props)
        )

        return psi

    def evaluate_performance(
        self,
        force: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate model performance on buffered data.

        Args:
            force: Force evaluation even with incomplete buffer

        Returns:
            Performance report or None
        """
        if len(self.ground_truth_buffer) < self.window_size and not force:
            return None

        if len(self.ground_truth_buffer) == 0:
            return {"error": "No ground truth available"}

        predictions = np.array(list(self.prediction_buffer))[:len(self.ground_truth_buffer)]
        ground_truth = np.array(list(self.ground_truth_buffer))

        report = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(ground_truth),
            "metrics": {}
        }

        if self.task_type == "classification":
            accuracy = accuracy_score(ground_truth, predictions)
            report["metrics"]["accuracy"] = float(accuracy)

            # Compare to baseline
            if "accuracy" in self.baseline_prediction_stats:
                baseline_acc = self.baseline_prediction_stats["accuracy"]
                degradation = baseline_acc - accuracy
                report["degradation"] = float(degradation)
                report["degradation_pct"] = float(degradation / baseline_acc * 100)

                if degradation > 0.1:  # 10% degradation
                    alert = {
                        "type": "performance_degradation",
                        "severity": "HIGH" if degradation > 0.2 else "MEDIUM",
                        "metric": "accuracy",
                        "current": float(accuracy),
                        "baseline": float(baseline_acc),
                        "degradation": float(degradation),
                        "timestamp": datetime.now().isoformat()
                    }
                    report["alert"] = alert
                    self.alerts.append(alert)

        else:
            r2 = r2_score(ground_truth, predictions)
            mse = mean_squared_error(ground_truth, predictions)
            report["metrics"]["r2"] = float(r2)
            report["metrics"]["mse"] = float(mse)

            # Compare to baseline
            if "r2" in self.baseline_prediction_stats:
                baseline_r2 = self.baseline_prediction_stats["r2"]
                degradation = baseline_r2 - r2
                report["degradation"] = float(degradation)

                if degradation > 0.1:
                    alert = {
                        "type": "performance_degradation",
                        "severity": "HIGH" if degradation > 0.2 else "MEDIUM",
                        "metric": "r2",
                        "current": float(r2),
                        "baseline": float(baseline_r2),
                        "degradation": float(degradation),
                        "timestamp": datetime.now().isoformat()
                    }
                    report["alert"] = alert
                    self.alerts.append(alert)

        self.performance_history.append(report)
        return report

    def detect_anomalous_predictions(
        self,
        predictions: np.ndarray,
        threshold_std: float = 3.0
    ) -> Dict[str, Any]:
        """
        Detect anomalous predictions.

        Args:
            predictions: Predictions to check
            threshold_std: Standard deviation threshold for anomalies

        Returns:
            Anomaly detection report
        """
        mean = self.baseline_prediction_stats["mean"]
        std = self.baseline_prediction_stats["std"]

        if std == 0:
            return {"error": "Zero standard deviation in baseline"}

        z_scores = np.abs((predictions - mean) / std)
        anomaly_mask = z_scores > threshold_std

        report = {
            "timestamp": datetime.now().isoformat(),
            "n_predictions": len(predictions),
            "n_anomalies": int(anomaly_mask.sum()),
            "anomaly_ratio": float(anomaly_mask.mean()),
            "anomaly_indices": np.where(anomaly_mask)[0].tolist()[:100],
            "max_z_score": float(z_scores.max()),
            "threshold": threshold_std
        }

        if report["anomaly_ratio"] > 0.1:
            alert = {
                "type": "anomalous_predictions",
                "severity": "HIGH" if report["anomaly_ratio"] > 0.2 else "MEDIUM",
                "anomaly_ratio": float(report["anomaly_ratio"]),
                "timestamp": datetime.now().isoformat()
            }
            report["alert"] = alert
            self.alerts.append(alert)

        return report

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get overall monitoring summary.

        Returns:
            Monitoring summary report
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "buffer_status": {
                "feature_buffer_size": len(self.feature_buffer),
                "prediction_buffer_size": len(self.prediction_buffer),
                "ground_truth_buffer_size": len(self.ground_truth_buffer),
                "window_size": self.window_size,
                "buffer_full": len(self.prediction_buffer) >= self.window_size
            },
            "alert_summary": {
                "total_alerts": len(self.alerts),
                "alerts_by_type": {},
                "alerts_by_severity": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
                "recent_alerts": self.alerts[-10:] if self.alerts else []
            },
            "performance_trend": {
                "n_evaluations": len(self.performance_history),
                "recent_metrics": self.performance_history[-5:]
                if self.performance_history else []
            }
        }

        # Count alerts by type
        for alert in self.alerts:
            alert_type = alert.get("type", "unknown")
            severity = alert.get("severity", "MEDIUM")

            summary["alert_summary"]["alerts_by_type"][alert_type] = \
                summary["alert_summary"]["alerts_by_type"].get(alert_type, 0) + 1
            summary["alert_summary"]["alerts_by_severity"][severity] += 1

        return summary

    def generate_vertex_ai_monitoring_config(
        self,
        project_id: str,
        endpoint_name: str,
        notification_channels: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate Vertex AI Model Monitoring configuration.

        Args:
            project_id: GCP project ID
            endpoint_name: Vertex AI endpoint name
            notification_channels: List of notification channel IDs
            output_path: Optional path to save configuration

        Returns:
            Vertex AI monitoring configuration
        """
        config = {
            "display_name": f"{endpoint_name}_monitoring",
            "model_monitoring_objective_configs": [
                {
                    "objective_config": {
                        "prediction_drift_detection_config": {
                            "drift_thresholds": {
                                feat: {"value": self.drift_threshold}
                                for feat in self.feature_names
                            },
                            "default_drift_threshold": {
                                "value": self.drift_threshold
                            }
                        },
                        "feature_attribution_drift_detection_config": {
                            "default_drift_threshold": {
                                "value": self.drift_threshold
                            }
                        }
                    },
                    "training_dataset": {
                        "gcs_source": {
                            "gcs_uri": [
                                f"gs://{project_id}-mlops/training_data/"
                            ]
                        },
                        "data_format": "csv",
                        "target_field": "quality"
                    }
                }
            ],
            "model_monitoring_alert_config": {
                "email_alert_config": {
                    "user_emails": []
                },
                "enable_logging": True,
                "notification_channels": notification_channels or []
            },
            "model_monitoring_job_configs": {
                "sample_rate": 1.0,
                "log_sampling_strategy": {
                    "random_sample_config": {
                        "sample_rate": 0.8
                    }
                }
            },
            "analysis_instance_schema_uri": "",
            "stats_anomalies_base_directory": {
                "output_uri_prefix": f"gs://{project_id}-mlops/monitoring/"
            }
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Vertex AI monitoring config saved to {output_path}")

        return config

    def clear_buffers(self):
        """Clear all monitoring buffers."""
        self.feature_buffer.clear()
        self.prediction_buffer.clear()
        self.ground_truth_buffer.clear()
        logger.info("Monitoring buffers cleared")

    def clear_alerts(self):
        """Clear alert history."""
        self.alerts.clear()
        logger.info("Alert history cleared")


if __name__ == "__main__":
    # Demo with wine quality data
    from sklearn.datasets import load_wine
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    print("=== Model Monitoring Demo ===\n")

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

    # Initialize monitor
    monitor = ModelMonitor(
        model=model,
        feature_names=wine.feature_names,
        task_type="classification",
        window_size=50,
        drift_threshold=0.1
    )

    # Set baseline
    baseline = monitor.set_baseline(X_train, y_train)
    print(f"Baseline set with {baseline['n_samples']} samples")
    print(f"Baseline accuracy: {baseline['prediction_stats'].get('accuracy', 'N/A'):.3f}")

    # Simulate production predictions
    print("\n=== Simulating Production Traffic ===")

    # Normal traffic
    print("\n1. Normal traffic (no drift):")
    for i in range(0, len(X_test), 10):
        batch = X_test.iloc[i:i+10]
        preds = model.predict(batch)
        status = monitor.log_prediction(batch, preds, y_test.iloc[i:i+10].values)

    drift_report = monitor.check_drift()
    print(f"   Drift detected: {drift_report['drift_detected']}")
    print(f"   Alerts: {len(drift_report['alerts'])}")

    # Simulate drifted data
    print("\n2. Drifted traffic (feature shift):")
    X_drifted = X_test.copy()
    X_drifted[wine.feature_names[0]] = X_drifted[wine.feature_names[0]] * 2  # Double first feature

    monitor.clear_buffers()
    for i in range(0, len(X_drifted), 10):
        batch = X_drifted.iloc[i:i+10]
        preds = model.predict(batch)
        monitor.log_prediction(batch, preds)

    drift_report = monitor.check_drift()
    print(f"   Drift detected: {drift_report['drift_detected']}")
    print(f"   Drifted features: {[f for f, d in drift_report['feature_drift'].items() if d['is_drifted']]}")

    # Get summary
    print("\n=== Monitoring Summary ===")
    summary = monitor.get_monitoring_summary()
    print(f"Total alerts: {summary['alert_summary']['total_alerts']}")
    print(f"Alerts by type: {summary['alert_summary']['alerts_by_type']}")

    # Generate Vertex AI config
    print("\n=== Vertex AI Configuration ===")
    vertex_config = monitor.generate_vertex_ai_monitoring_config(
        project_id="my-project",
        endpoint_name="wine-quality-endpoint"
    )
    print(f"Monitoring objectives configured: {len(vertex_config['model_monitoring_objective_configs'])}")
