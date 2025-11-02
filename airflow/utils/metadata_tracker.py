"""
MLOps Metadata Tracking Module

This module provides a clean interface for tracking ML pipeline metadata
including model runs, metrics, parameters, data lineage, and events.
"""

import psycopg2
from psycopg2.extras import RealDictCursor, Json
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class MetadataTracker:
    """Tracks metadata for ML pipeline operations."""

    def __init__(self, db_connection_string: str):
        """
        Initialize metadata tracker.

        Args:
            db_connection_string: PostgreSQL connection string
        """
        self.conn_string = db_connection_string
        self.conn = None

    def connect(self):
        """Establish database connection."""
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(self.conn_string)
        return self.conn

    def close(self):
        """Close database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    # ============ Model Run Tracking ============

    def create_model_run(
        self,
        run_id: str,
        dag_id: str,
        task_id: str,
        execution_date: datetime,
        model_name: str,
        model_type: str = "RandomForest",
        framework: str = "sklearn",
        model_version: Optional[str] = None
    ) -> str:
        """
        Create a new model training run entry.

        Returns:
            run_id
        """
        conn = self.connect()
        cursor = conn.cursor()

        query = """
        INSERT INTO model_runs (
            run_id, dag_id, task_id, execution_date, model_name,
            model_version, model_type, framework, training_status,
            training_start_time
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (run_id) DO UPDATE SET
            training_status = EXCLUDED.training_status,
            updated_at = CURRENT_TIMESTAMP
        RETURNING run_id;
        """

        cursor.execute(query, (
            run_id, dag_id, task_id, execution_date, model_name,
            model_version, model_type, framework, 'started',
            datetime.now()
        ))

        conn.commit()
        result = cursor.fetchone()[0]
        cursor.close()

        logger.info(f"Created model run: {run_id}")
        return result

    def update_model_run_status(
        self,
        run_id: str,
        status: str,
        training_duration: Optional[float] = None,
        deployment_stage: Optional[str] = None,
        deployed: bool = False
    ):
        """Update model run status and metadata."""
        conn = self.connect()
        cursor = conn.cursor()

        query = """
        UPDATE model_runs SET
            training_status = %s,
            training_end_time = %s,
            training_duration_seconds = COALESCE(%s, training_duration_seconds),
            deployment_stage = COALESCE(%s, deployment_stage),
            deployed = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE run_id = %s;
        """

        cursor.execute(query, (
            status,
            datetime.now() if status == 'completed' else None,
            training_duration,
            deployment_stage,
            deployed,
            run_id
        ))

        conn.commit()
        cursor.close()
        logger.info(f"Updated model run {run_id}: status={status}")

    # ============ Metrics Tracking ============

    def log_metrics(
        self,
        run_id: str,
        metrics: Dict[str, float],
        metric_type: str = "training"
    ):
        """
        Log multiple metrics for a model run.

        Args:
            run_id: Model run identifier
            metrics: Dictionary of metric_name: metric_value
            metric_type: Type of metrics (training, validation, test)
        """
        conn = self.connect()
        cursor = conn.cursor()

        query = """
        INSERT INTO model_metrics (run_id, metric_name, metric_value, metric_type)
        VALUES (%s, %s, %s, %s);
        """

        data = [(run_id, name, value, metric_type) for name, value in metrics.items()]
        cursor.executemany(query, data)

        conn.commit()
        cursor.close()
        logger.info(f"Logged {len(metrics)} metrics for run {run_id}")

    def get_run_metrics(self, run_id: str) -> Dict[str, float]:
        """Get all metrics for a specific run."""
        conn = self.connect()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        query = """
        SELECT metric_name, metric_value, metric_type
        FROM model_metrics
        WHERE run_id = %s;
        """

        cursor.execute(query, (run_id,))
        results = cursor.fetchall()
        cursor.close()

        return {row['metric_name']: row['metric_value'] for row in results}

    # ============ Parameters Tracking ============

    def log_parameters(
        self,
        run_id: str,
        parameters: Dict[str, Any],
        param_type: str = "hyperparameter"
    ):
        """Log model parameters/hyperparameters."""
        conn = self.connect()
        cursor = conn.cursor()

        query = """
        INSERT INTO model_parameters (run_id, param_name, param_value, param_type)
        VALUES (%s, %s, %s, %s);
        """

        data = [
            (run_id, name, json.dumps(value) if not isinstance(value, str) else value, param_type)
            for name, value in parameters.items()
        ]
        cursor.executemany(query, data)

        conn.commit()
        cursor.close()
        logger.info(f"Logged {len(parameters)} parameters for run {run_id}")

    # ============ Dataset Tracking ============

    def register_dataset(
        self,
        dataset_name: str,
        dataset_version: str,
        dataset_path: str,
        dataset_size_bytes: int,
        row_count: int,
        column_count: int,
        dataset_hash: Optional[str] = None
    ) -> int:
        """Register a new dataset version."""
        conn = self.connect()
        cursor = conn.cursor()

        query = """
        INSERT INTO dataset_versions (
            dataset_name, dataset_version, dataset_path, dataset_size_bytes,
            row_count, column_count, dataset_hash
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (dataset_name, dataset_version) DO UPDATE SET
            dataset_path = EXCLUDED.dataset_path,
            dataset_size_bytes = EXCLUDED.dataset_size_bytes
        RETURNING dataset_id;
        """

        cursor.execute(query, (
            dataset_name, dataset_version, dataset_path, dataset_size_bytes,
            row_count, column_count, dataset_hash
        ))

        dataset_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()

        logger.info(f"Registered dataset: {dataset_name} v{dataset_version}")
        return dataset_id

    def link_dataset_to_run(
        self,
        run_id: str,
        dataset_id: int,
        usage_type: str = "train"
    ):
        """Link a dataset to a model run (data lineage)."""
        conn = self.connect()
        cursor = conn.cursor()

        query = """
        INSERT INTO data_lineage (run_id, dataset_id, usage_type)
        VALUES (%s, %s, %s);
        """

        cursor.execute(query, (run_id, dataset_id, usage_type))
        conn.commit()
        cursor.close()

        logger.info(f"Linked dataset {dataset_id} to run {run_id}")

    # ============ Model Comparison ============

    def log_model_comparison(
        self,
        dag_run_id: str,
        new_run_id: str,
        baseline_run_id: str,
        comparison_result: str,
        metric_compared: str,
        new_metric_value: float,
        baseline_metric_value: float,
        deployed: bool = False
    ) -> int:
        """Log model comparison results."""
        conn = self.connect()
        cursor = conn.cursor()

        improvement = ((new_metric_value - baseline_metric_value) / baseline_metric_value) * 100

        query = """
        INSERT INTO model_comparisons (
            dag_run_id, new_run_id, baseline_run_id, comparison_result,
            metric_compared, new_metric_value, baseline_metric_value,
            improvement_percentage, deployed
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING comparison_id;
        """

        cursor.execute(query, (
            dag_run_id, new_run_id, baseline_run_id, comparison_result,
            metric_compared, new_metric_value, baseline_metric_value,
            improvement, deployed
        ))

        comparison_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()

        logger.info(f"Logged model comparison: {comparison_id}")
        return comparison_id

    # ============ Event Logging ============

    def log_pipeline_event(
        self,
        dag_id: str,
        event_type: str,
        event_message: str,
        dag_run_id: Optional[str] = None,
        task_id: Optional[str] = None,
        event_data: Optional[Dict] = None
    ) -> int:
        """Log a pipeline event."""
        conn = self.connect()
        cursor = conn.cursor()

        query = """
        INSERT INTO pipeline_events (
            dag_id, dag_run_id, task_id, event_type, event_message, event_data
        ) VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING event_id;
        """

        cursor.execute(query, (
            dag_id, dag_run_id, task_id, event_type, event_message,
            Json(event_data) if event_data else None
        ))

        event_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()

        logger.info(f"Logged pipeline event: {event_id} - {event_type}")
        return event_id

    def log_notification(
        self,
        event_id: int,
        channel: str,
        recipient: str,
        subject: str,
        message: str,
        status: str = "pending"
    ) -> int:
        """Log a notification attempt."""
        conn = self.connect()
        cursor = conn.cursor()

        query = """
        INSERT INTO notification_history (
            event_id, channel, recipient, subject, message, status,
            sent_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING notification_id;
        """

        sent_at = datetime.now() if status == "sent" else None

        cursor.execute(query, (
            event_id, channel, recipient, subject, message, status, sent_at
        ))

        notification_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()

        logger.info(f"Logged notification: {notification_id}")
        return notification_id

    # ============ Data Drift Detection ============

    def log_data_drift(
        self,
        dataset_id: int,
        feature_name: str,
        drift_metric: str,
        drift_score: float,
        drift_threshold: float,
        drift_detected: bool
    ) -> int:
        """Log data drift detection results."""
        conn = self.connect()
        cursor = conn.cursor()

        query = """
        INSERT INTO data_drift_events (
            dataset_id, feature_name, drift_metric, drift_score,
            drift_threshold, drift_detected
        ) VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING drift_id;
        """

        cursor.execute(query, (
            dataset_id, feature_name, drift_metric, drift_score,
            drift_threshold, drift_detected
        ))

        drift_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()

        if drift_detected:
            logger.warning(f"Data drift detected: {feature_name} - {drift_score:.4f}")

        return drift_id

    # ============ Queries ============

    def get_latest_production_model(self, model_name: str) -> Optional[Dict]:
        """Get the latest production model information."""
        conn = self.connect()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        query = """
        SELECT * FROM latest_production_models
        WHERE model_name = %s
        LIMIT 1;
        """

        cursor.execute(query, (model_name,))
        result = cursor.fetchone()
        cursor.close()

        return dict(result) if result else None

    def get_recent_events(self, days: int = 7, event_type: Optional[str] = None) -> List[Dict]:
        """Get recent pipeline events."""
        conn = self.connect()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        if event_type:
            query = """
            SELECT * FROM recent_pipeline_events
            WHERE event_type = %s
            ORDER BY created_at DESC;
            """
            cursor.execute(query, (event_type,))
        else:
            query = """
            SELECT * FROM recent_pipeline_events
            ORDER BY created_at DESC;
            """
            cursor.execute(query)

        results = cursor.fetchall()
        cursor.close()

        return [dict(row) for row in results]


# Utility functions

def compute_dataset_hash(file_path: str) -> str:
    """Compute MD5 hash of a dataset file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
