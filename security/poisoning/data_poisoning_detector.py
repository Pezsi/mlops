"""
Data Poisoning Detection Module (Lecke 113)

This module provides comprehensive data poisoning detection capabilities:
- Statistical outlier detection (IQR, Z-score, Isolation Forest)
- Label flipping detection
- Data distribution anomaly detection
- Backdoor pattern detection in features

Reference: OWASP ML Security Top 10 - ML04:2023 Model Poisoning
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import logging
import hashlib
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPoisoningDetector:
    """
    Comprehensive data poisoning detection for ML datasets.

    Detects various types of data poisoning attacks:
    1. Outlier injection - anomalous data points
    2. Label flipping - incorrect labels for legitimate data
    3. Backdoor attacks - specific patterns that trigger misclassification
    4. Distribution shifts - subtle changes to data distribution
    """

    def __init__(
        self,
        contamination: float = 0.1,
        z_score_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        random_state: int = 42
    ):
        """
        Initialize the data poisoning detector.

        Args:
            contamination: Expected proportion of outliers in the data
            z_score_threshold: Threshold for Z-score based outlier detection
            iqr_multiplier: Multiplier for IQR-based outlier detection
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.z_score_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier
        self.random_state = random_state
        self.baseline_stats: Optional[Dict] = None
        self.feature_fingerprints: Dict[str, str] = {}

    def compute_baseline_statistics(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Compute baseline statistics for the clean dataset.

        Args:
            X: Feature DataFrame
            y: Target series (optional)

        Returns:
            Dictionary containing baseline statistics
        """
        stats_dict = {
            "computed_at": datetime.now().isoformat(),
            "n_samples": len(X),
            "n_features": len(X.columns),
            "feature_stats": {},
            "correlation_matrix": X.corr().to_dict(),
        }

        for col in X.columns:
            col_data = X[col]
            q1, q3 = col_data.quantile([0.25, 0.75])
            stats_dict["feature_stats"][col] = {
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "median": float(col_data.median()),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "skewness": float(col_data.skew()),
                "kurtosis": float(col_data.kurtosis()),
            }

        if y is not None:
            stats_dict["label_distribution"] = y.value_counts().to_dict()
            stats_dict["label_entropy"] = float(
                stats.entropy(y.value_counts(normalize=True))
            )

        self.baseline_stats = stats_dict
        logger.info(f"Baseline statistics computed for {len(X)} samples")
        return stats_dict

    def detect_outliers_zscore(
        self,
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, List[int]]]:
        """
        Detect outliers using Z-score method.

        Args:
            X: Feature DataFrame

        Returns:
            Tuple of (outlier mask, per-feature outlier indices)
        """
        z_scores = np.abs(stats.zscore(X, nan_policy='omit'))
        outlier_mask = (z_scores > self.z_score_threshold).any(axis=1)

        feature_outliers = {}
        for i, col in enumerate(X.columns):
            outlier_indices = np.where(z_scores[:, i] > self.z_score_threshold)[0]
            if len(outlier_indices) > 0:
                feature_outliers[col] = outlier_indices.tolist()

        n_outliers = outlier_mask.sum()
        logger.info(f"Z-score detection: {n_outliers} outliers found "
                   f"({100*n_outliers/len(X):.2f}%)")
        return outlier_mask, feature_outliers

    def detect_outliers_iqr(
        self,
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, List[int]]]:
        """
        Detect outliers using IQR (Interquartile Range) method.

        Args:
            X: Feature DataFrame

        Returns:
            Tuple of (outlier mask, per-feature outlier indices)
        """
        outlier_mask = np.zeros(len(X), dtype=bool)
        feature_outliers = {}

        for col in X.columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr

            col_outliers = (X[col] < lower_bound) | (X[col] > upper_bound)
            outlier_indices = np.where(col_outliers)[0]

            if len(outlier_indices) > 0:
                feature_outliers[col] = outlier_indices.tolist()
                outlier_mask |= col_outliers.values

        n_outliers = outlier_mask.sum()
        logger.info(f"IQR detection: {n_outliers} outliers found "
                   f"({100*n_outliers/len(X):.2f}%)")
        return outlier_mask, feature_outliers

    def detect_outliers_isolation_forest(
        self,
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outliers using Isolation Forest algorithm.

        Args:
            X: Feature DataFrame

        Returns:
            Tuple of (outlier mask, anomaly scores)
        """
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            n_jobs=-1
        )

        predictions = iso_forest.fit_predict(X)
        anomaly_scores = iso_forest.decision_function(X)
        outlier_mask = predictions == -1

        n_outliers = outlier_mask.sum()
        logger.info(f"Isolation Forest: {n_outliers} outliers found "
                   f"({100*n_outliers/len(X):.2f}%)")
        return outlier_mask, anomaly_scores

    def detect_outliers_lof(
        self,
        X: pd.DataFrame,
        n_neighbors: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outliers using Local Outlier Factor (LOF).

        Args:
            X: Feature DataFrame
            n_neighbors: Number of neighbors for LOF

        Returns:
            Tuple of (outlier mask, LOF scores)
        """
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
            n_jobs=-1
        )

        predictions = lof.fit_predict(X)
        lof_scores = lof.negative_outlier_factor_
        outlier_mask = predictions == -1

        n_outliers = outlier_mask.sum()
        logger.info(f"LOF detection: {n_outliers} outliers found "
                   f"({100*n_outliers/len(X):.2f}%)")
        return outlier_mask, lof_scores

    def detect_label_flipping(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_neighbors: int = 5
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect potential label flipping attacks by identifying samples
        whose labels differ from their nearest neighbors.

        Args:
            X: Feature DataFrame
            y: Target series
            n_neighbors: Number of neighbors to consider

        Returns:
            Tuple of (suspicious sample mask, detection report)
        """
        from sklearn.neighbors import NearestNeighbors

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=-1)
        nn.fit(X_scaled)
        distances, indices = nn.kneighbors(X_scaled)

        suspicious_mask = np.zeros(len(X), dtype=bool)
        mismatch_scores = np.zeros(len(X))

        for i in range(len(X)):
            neighbor_indices = indices[i, 1:]  # Exclude self
            neighbor_labels = y.iloc[neighbor_indices].values
            sample_label = y.iloc[i]

            # Calculate label agreement ratio
            agreement_ratio = np.mean(neighbor_labels == sample_label)
            mismatch_scores[i] = 1 - agreement_ratio

            # Flag as suspicious if majority of neighbors have different labels
            if agreement_ratio < 0.5:
                suspicious_mask[i] = True

        # Analyze by class
        class_analysis = {}
        for label in y.unique():
            label_mask = y == label
            class_analysis[str(label)] = {
                "total_samples": int(label_mask.sum()),
                "suspicious_samples": int((suspicious_mask & label_mask).sum()),
                "avg_mismatch_score": float(mismatch_scores[label_mask].mean())
            }

        report = {
            "total_suspicious": int(suspicious_mask.sum()),
            "suspicious_ratio": float(suspicious_mask.mean()),
            "class_analysis": class_analysis,
            "high_risk_indices": np.where(mismatch_scores > 0.8)[0].tolist()
        }

        logger.info(f"Label flipping detection: {report['total_suspicious']} "
                   f"suspicious samples found")
        return suspicious_mask, report

    def detect_backdoor_patterns(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        min_samples: int = 5
    ) -> Dict[str, Any]:
        """
        Detect potential backdoor patterns in the data.
        Backdoor attacks often involve specific feature value combinations
        that consistently lead to a particular classification.

        Args:
            X: Feature DataFrame
            y: Target series
            min_samples: Minimum samples to consider a pattern suspicious

        Returns:
            Dictionary containing backdoor pattern analysis
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "suspicious_patterns": [],
            "feature_value_clusters": {},
            "risk_score": 0.0
        }

        # Use DBSCAN to find unusual clusters
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        dbscan = DBSCAN(eps=0.5, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)

        # Analyze each cluster for backdoor characteristics
        unique_clusters = set(clusters) - {-1}  # Exclude noise

        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_size = cluster_mask.sum()

            if cluster_size < min_samples:
                continue

            cluster_labels = y[cluster_mask]
            label_purity = cluster_labels.value_counts().max() / len(cluster_labels)

            # High purity small clusters might indicate backdoor
            if label_purity > 0.95 and cluster_size < len(X) * 0.05:
                cluster_features = X[cluster_mask].mean()
                global_features = X.mean()

                # Find features that deviate significantly in this cluster
                feature_deviations = {}
                for col in X.columns:
                    deviation = abs(cluster_features[col] - global_features[col])
                    std = X[col].std()
                    if std > 0 and deviation / std > 2:
                        feature_deviations[col] = {
                            "cluster_mean": float(cluster_features[col]),
                            "global_mean": float(global_features[col]),
                            "std_deviation": float(deviation / std)
                        }

                if feature_deviations:
                    pattern = {
                        "cluster_id": int(cluster_id),
                        "cluster_size": int(cluster_size),
                        "dominant_label": int(cluster_labels.mode().iloc[0]),
                        "label_purity": float(label_purity),
                        "deviating_features": feature_deviations,
                        "sample_indices": np.where(cluster_mask)[0].tolist()[:10]
                    }
                    report["suspicious_patterns"].append(pattern)

        # Calculate overall risk score
        if report["suspicious_patterns"]:
            total_suspicious = sum(
                p["cluster_size"] for p in report["suspicious_patterns"]
            )
            report["risk_score"] = min(1.0, total_suspicious / (len(X) * 0.1))

        logger.info(f"Backdoor detection: {len(report['suspicious_patterns'])} "
                   f"suspicious patterns found, risk score: {report['risk_score']:.3f}")
        return report

    def detect_distribution_shift(
        self,
        X_new: pd.DataFrame,
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detect distribution shifts between baseline and new data.

        Args:
            X_new: New feature DataFrame to compare against baseline
            significance_level: Statistical significance level for tests

        Returns:
            Dictionary containing distribution shift analysis
        """
        if self.baseline_stats is None:
            raise ValueError("Baseline statistics not computed. "
                           "Call compute_baseline_statistics first.")

        report = {
            "timestamp": datetime.now().isoformat(),
            "feature_drift": {},
            "overall_drift_score": 0.0,
            "drifted_features": []
        }

        drift_scores = []

        for col in X_new.columns:
            if col not in self.baseline_stats["feature_stats"]:
                continue

            baseline = self.baseline_stats["feature_stats"][col]
            new_data = X_new[col]

            # Kolmogorov-Smirnov test
            # Generate baseline samples from statistics
            baseline_samples = np.random.normal(
                baseline["mean"],
                baseline["std"],
                size=1000
            )
            ks_stat, ks_pvalue = stats.ks_2samp(baseline_samples, new_data)

            # Population Stability Index (PSI)
            psi = self._calculate_psi(baseline_samples, new_data.values)

            # Mean shift detection
            mean_shift = abs(new_data.mean() - baseline["mean"]) / baseline["std"]

            # Variance change
            var_ratio = new_data.var() / (baseline["std"] ** 2)

            feature_drift = {
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "psi": float(psi),
                "mean_shift_std": float(mean_shift),
                "variance_ratio": float(var_ratio),
                "is_drifted": ks_pvalue < significance_level or psi > 0.2
            }

            report["feature_drift"][col] = feature_drift

            if feature_drift["is_drifted"]:
                report["drifted_features"].append(col)
                drift_scores.append(psi)

        if drift_scores:
            report["overall_drift_score"] = float(np.mean(drift_scores))

        logger.info(f"Distribution shift: {len(report['drifted_features'])} "
                   f"features drifted, overall score: {report['overall_drift_score']:.3f}")
        return report

    def _calculate_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI interpretation:
        - PSI < 0.1: No significant shift
        - 0.1 <= PSI < 0.2: Moderate shift
        - PSI >= 0.2: Significant shift
        """
        # Create bins from baseline data
        bins = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        # Calculate proportions
        baseline_counts = np.histogram(baseline, bins=bins)[0]
        current_counts = np.histogram(current, bins=bins)[0]

        baseline_props = baseline_counts / len(baseline)
        current_props = current_counts / len(current)

        # Avoid division by zero
        baseline_props = np.clip(baseline_props, 0.001, None)
        current_props = np.clip(current_props, 0.001, None)

        # Calculate PSI
        psi = np.sum(
            (current_props - baseline_props) *
            np.log(current_props / baseline_props)
        )

        return psi

    def compute_data_fingerprint(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> str:
        """
        Compute a cryptographic fingerprint of the dataset for integrity verification.

        Args:
            X: Feature DataFrame
            y: Target series (optional)

        Returns:
            SHA256 hash of the dataset
        """
        data_bytes = X.to_numpy().tobytes()
        if y is not None:
            data_bytes += y.to_numpy().tobytes()

        fingerprint = hashlib.sha256(data_bytes).hexdigest()
        logger.info(f"Data fingerprint computed: {fingerprint[:16]}...")
        return fingerprint

    def run_full_analysis(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_baseline: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete data poisoning analysis.

        Args:
            X: Feature DataFrame
            y: Target series
            save_baseline: Whether to save baseline statistics

        Returns:
            Comprehensive analysis report
        """
        logger.info("Starting full data poisoning analysis...")

        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "dataset_info": {
                "n_samples": len(X),
                "n_features": len(X.columns),
                "n_classes": len(y.unique())
            },
            "data_fingerprint": self.compute_data_fingerprint(X, y),
            "outlier_detection": {},
            "label_flipping": {},
            "backdoor_patterns": {},
            "overall_risk_assessment": {}
        }

        # Compute baseline if requested
        if save_baseline:
            self.compute_baseline_statistics(X, y)

        # Z-score outliers
        zscore_mask, zscore_features = self.detect_outliers_zscore(X)
        report["outlier_detection"]["zscore"] = {
            "n_outliers": int(zscore_mask.sum()),
            "outlier_ratio": float(zscore_mask.mean()),
            "feature_outliers": zscore_features
        }

        # IQR outliers
        iqr_mask, iqr_features = self.detect_outliers_iqr(X)
        report["outlier_detection"]["iqr"] = {
            "n_outliers": int(iqr_mask.sum()),
            "outlier_ratio": float(iqr_mask.mean()),
            "feature_outliers": iqr_features
        }

        # Isolation Forest
        iso_mask, iso_scores = self.detect_outliers_isolation_forest(X)
        report["outlier_detection"]["isolation_forest"] = {
            "n_outliers": int(iso_mask.sum()),
            "outlier_ratio": float(iso_mask.mean()),
            "avg_anomaly_score": float(iso_scores.mean())
        }

        # Local Outlier Factor
        lof_mask, lof_scores = self.detect_outliers_lof(X)
        report["outlier_detection"]["lof"] = {
            "n_outliers": int(lof_mask.sum()),
            "outlier_ratio": float(lof_mask.mean()),
            "avg_lof_score": float(lof_scores.mean())
        }

        # Consensus outliers (detected by multiple methods)
        consensus_mask = (
            zscore_mask.astype(int) +
            iqr_mask.astype(int) +
            iso_mask.astype(int) +
            lof_mask.astype(int)
        ) >= 2
        report["outlier_detection"]["consensus"] = {
            "n_outliers": int(consensus_mask.sum()),
            "outlier_ratio": float(consensus_mask.mean()),
            "outlier_indices": np.where(consensus_mask)[0].tolist()[:100]
        }

        # Label flipping detection
        flip_mask, flip_report = self.detect_label_flipping(X, y)
        report["label_flipping"] = flip_report

        # Backdoor pattern detection
        report["backdoor_patterns"] = self.detect_backdoor_patterns(X, y)

        # Overall risk assessment
        risk_factors = []

        # High outlier ratio
        if report["outlier_detection"]["consensus"]["outlier_ratio"] > 0.05:
            risk_factors.append("high_outlier_ratio")

        # Label flipping concerns
        if report["label_flipping"]["suspicious_ratio"] > 0.1:
            risk_factors.append("label_flipping_suspected")

        # Backdoor patterns
        if report["backdoor_patterns"]["risk_score"] > 0.3:
            risk_factors.append("backdoor_pattern_detected")

        overall_risk = len(risk_factors) / 3.0
        report["overall_risk_assessment"] = {
            "risk_score": overall_risk,
            "risk_level": "HIGH" if overall_risk > 0.6 else
                         "MEDIUM" if overall_risk > 0.3 else "LOW",
            "risk_factors": risk_factors,
            "recommendations": self._generate_recommendations(risk_factors)
        }

        logger.info(f"Analysis complete. Risk level: "
                   f"{report['overall_risk_assessment']['risk_level']}")
        return report

    def _generate_recommendations(self, risk_factors: List[str]) -> List[str]:
        """Generate security recommendations based on detected risks."""
        recommendations = []

        if not risk_factors:
            recommendations.append("No significant risks detected. Continue monitoring.")
            return recommendations

        if "high_outlier_ratio" in risk_factors:
            recommendations.extend([
                "Review and validate data collection pipeline",
                "Investigate sources of anomalous data points",
                "Consider removing or correcting outliers before training"
            ])

        if "label_flipping_suspected" in risk_factors:
            recommendations.extend([
                "Audit labeling process and annotators",
                "Implement multi-annotator consensus for labels",
                "Use confident learning to detect label errors"
            ])

        if "backdoor_pattern_detected" in risk_factors:
            recommendations.extend([
                "Investigate small high-purity clusters",
                "Check for external data source contamination",
                "Implement trigger detection mechanisms",
                "Consider adversarial training"
            ])

        return recommendations


if __name__ == "__main__":
    # Demo with wine quality data
    from sklearn.datasets import load_wine

    print("=== Data Poisoning Detection Demo ===\n")

    # Load sample data
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target)

    # Initialize detector
    detector = DataPoisoningDetector(contamination=0.1)

    # Run full analysis
    report = detector.run_full_analysis(X, y)

    print("\n=== Analysis Results ===")
    print(f"Dataset: {report['dataset_info']['n_samples']} samples, "
          f"{report['dataset_info']['n_features']} features")
    print(f"Data Fingerprint: {report['data_fingerprint'][:32]}...")
    print(f"\nOverall Risk Level: {report['overall_risk_assessment']['risk_level']}")
    print(f"Risk Score: {report['overall_risk_assessment']['risk_score']:.2f}")

    if report['overall_risk_assessment']['risk_factors']:
        print(f"Risk Factors: {', '.join(report['overall_risk_assessment']['risk_factors'])}")

    print("\n=== Recommendations ===")
    for rec in report['overall_risk_assessment']['recommendations']:
        print(f"  - {rec}")
