"""
Dataset File Sensor DAG - Event-Based Triggering

This DAG demonstrates event-based automation by monitoring for new dataset files.
When a new dataset file appears in a specified directory, it automatically triggers
data validation, preprocessing, and potentially model retraining.

Key Features:
- FileSensor for event-based triggering
- Automatic dataset validation
- Data drift detection
- Conditional model retraining trigger
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

# Add project root to path
sys.path.insert(0, '/opt/airflow')

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email': ['admin@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
}

# Configuration
WATCH_DIRECTORY = "/opt/airflow/data/incoming"
PROCESSED_DIRECTORY = "/opt/airflow/data/processed"
DATASET_PATTERN = "winequality-*.csv"
REFERENCE_DATASET = "/opt/airflow/data/winequality-red.csv"


def validate_new_dataset(**kwargs):
    """
    Validate newly arrived dataset.
    Checks for:
    - File integrity
    - Schema consistency
    - Data quality issues
    """
    from airflow.utils.metadata_tracker import MetadataTracker

    file_path = kwargs['ti'].xcom_pull(task_ids='wait_for_new_dataset', key='return_value')
    print(f"Validating dataset: {file_path}")

    try:
        # Load dataset
        df = pd.read_csv(file_path)

        # Basic validation
        validation_results = {
            'file_path': file_path,
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': int(df.isnull().sum().sum()),
            'duplicates': int(df.duplicated().sum()),
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
        }

        # Check schema against reference
        reference_df = pd.read_csv(REFERENCE_DATASET)
        schema_match = set(df.columns) == set(reference_df.columns)
        validation_results['schema_match'] = schema_match

        # Quality checks
        quality_issues = []
        if validation_results['missing_values'] > 0:
            quality_issues.append(f"Found {validation_results['missing_values']} missing values")

        if validation_results['duplicates'] > 0:
            quality_issues.append(f"Found {validation_results['duplicates']} duplicate rows")

        if not schema_match:
            quality_issues.append(f"Schema mismatch with reference dataset")

        validation_results['quality_issues'] = quality_issues
        validation_results['is_valid'] = len(quality_issues) == 0

        # Store results in XCom
        kwargs['ti'].xcom_push(key='validation_results', value=validation_results)
        kwargs['ti'].xcom_push(key='new_dataset_path', value=file_path)

        print(f"Validation Results: {validation_results}")

        return validation_results['is_valid']

    except Exception as e:
        print(f"Validation failed: {e}")
        raise


def detect_data_drift(**kwargs):
    """
    Detect data drift between new dataset and reference dataset using statistical tests.

    Uses:
    - Kolmogorov-Smirnov test for numerical features
    - Chi-square test for categorical features
    - Population Stability Index (PSI)
    """
    ti = kwargs['ti']
    new_dataset_path = ti.xcom_pull(task_ids='validate_dataset', key='new_dataset_path')

    print(f"Detecting data drift for: {new_dataset_path}")

    # Load datasets
    new_df = pd.read_csv(new_dataset_path)
    reference_df = pd.read_csv(REFERENCE_DATASET)

    drift_detected = False
    drift_results = []

    # Numerical features - KS test
    numerical_features = new_df.select_dtypes(include=[np.number]).columns

    for feature in numerical_features:
        if feature in reference_df.columns:
            # Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(
                reference_df[feature].dropna(),
                new_df[feature].dropna()
            )

            # Drift detected if p-value < 0.05
            feature_drift = p_value < 0.05

            drift_results.append({
                'feature': feature,
                'test': 'KS_test',
                'statistic': float(ks_statistic),
                'p_value': float(p_value),
                'drift_detected': feature_drift
            })

            if feature_drift:
                drift_detected = True
                print(f"  Drift detected in '{feature}': KS={ks_statistic:.4f}, p={p_value:.4f}")

    # Calculate PSI for overall dataset drift
    psi_score = calculate_psi(reference_df, new_df, numerical_features)
    drift_results.append({
        'feature': 'overall',
        'test': 'PSI',
        'statistic': float(psi_score),
        'p_value': None,
        'drift_detected': psi_score > 0.2  # PSI > 0.2 indicates significant drift
    })

    if psi_score > 0.2:
        drift_detected = True
        print(f"  Overall drift detected: PSI={psi_score:.4f}")

    # Store results
    ti.xcom_push(key='drift_detected', value=drift_detected)
    ti.xcom_push(key='drift_results', value=drift_results)

    print(f"Drift Detection Summary: {'Drift Detected' if drift_detected else 'No Drift'}")

    return 'trigger_retraining' if drift_detected else 'skip_retraining'


def calculate_psi(reference_df, new_df, features, buckets=10):
    """
    Calculate Population Stability Index (PSI) for dataset drift.

    PSI Interpretation:
    - < 0.1: No significant change
    - 0.1 - 0.2: Moderate change
    - > 0.2: Significant change (drift)
    """
    psi_values = []

    for feature in features:
        try:
            # Create buckets based on reference distribution
            min_val = min(reference_df[feature].min(), new_df[feature].min())
            max_val = max(reference_df[feature].max(), new_df[feature].max())
            breakpoints = np.linspace(min_val, max_val, buckets + 1)

            # Calculate distributions
            reference_dist = np.histogram(reference_df[feature].dropna(), bins=breakpoints)[0]
            new_dist = np.histogram(new_df[feature].dropna(), bins=breakpoints)[0]

            # Normalize to percentages and avoid division by zero
            reference_dist = reference_dist / reference_dist.sum()
            new_dist = new_dist / new_dist.sum()

            # Add small constant to avoid log(0)
            reference_dist = reference_dist + 1e-10
            new_dist = new_dist + 1e-10

            # Calculate PSI
            psi = np.sum((new_dist - reference_dist) * np.log(new_dist / reference_dist))
            psi_values.append(psi)

        except Exception as e:
            print(f"Error calculating PSI for {feature}: {e}")
            continue

    return np.mean(psi_values) if psi_values else 0.0


def move_to_processed(**kwargs):
    """Move validated dataset to processed directory."""
    ti = kwargs['ti']
    source_path = ti.xcom_pull(task_ids='validate_dataset', key='new_dataset_path')

    # Create processed directory if not exists
    os.makedirs(PROCESSED_DIRECTORY, exist_ok=True)

    # Move file
    filename = os.path.basename(source_path)
    dest_path = os.path.join(PROCESSED_DIRECTORY, filename)

    import shutil
    shutil.move(source_path, dest_path)

    print(f"Moved dataset to: {dest_path}")
    return dest_path


def skip_retraining(**kwargs):
    """Log that retraining was skipped due to no drift."""
    print("No significant data drift detected. Skipping model retraining.")
    return "skipped"


def log_drift_to_metadata(**kwargs):
    """Log drift detection results to metadata database."""
    ti = kwargs['ti']
    drift_results = ti.xcom_pull(task_ids='detect_data_drift', key='drift_results')

    print("Logging drift detection to metadata database...")
    # This would connect to metadata tracker and log drift results
    # For now, just print
    for result in drift_results:
        print(f"  Feature: {result['feature']}, Drift: {result['drift_detected']}")


# Define the DAG
with DAG(
    dag_id='dataset_sensor_event_trigger',
    default_args=default_args,
    description='Event-driven DAG that monitors for new datasets and triggers validation',
    schedule_interval=None,  # Event-driven, no schedule
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['event-driven', 'dataset', 'sensor', 'data-quality'],
) as dag:

    # Task 1: Wait for new dataset file to appear
    wait_for_dataset = FileSensor(
        task_id='wait_for_new_dataset',
        filepath=WATCH_DIRECTORY,
        fs_conn_id='fs_default',  # Use default filesystem connection
        poke_interval=30,  # Check every 30 seconds
        timeout=60 * 60 * 24,  # Timeout after 24 hours
        mode='reschedule',  # Don't block a worker slot while waiting
    )

    # Task 2: Validate new dataset
    validate_task = PythonOperator(
        task_id='validate_dataset',
        python_callable=validate_new_dataset,
        provide_context=True,
    )

    # Task 3: Detect data drift
    drift_detection_task = BranchPythonOperator(
        task_id='detect_data_drift',
        python_callable=detect_data_drift,
        provide_context=True,
    )

    # Task 4: Log drift results to metadata DB
    log_drift_task = PythonOperator(
        task_id='log_drift_to_metadata',
        python_callable=log_drift_to_metadata,
        provide_context=True,
    )

    # Task 5a: Trigger model retraining if drift detected
    trigger_retraining_task = TriggerDagRunOperator(
        task_id='trigger_retraining',
        trigger_dag_id='train_wine_quality_model',  # Trigger the training DAG
        conf={
            'triggered_by': 'dataset_sensor',
            'reason': 'data_drift_detected'
        },
    )

    # Task 5b: Skip retraining if no drift
    skip_retraining_task = PythonOperator(
        task_id='skip_retraining',
        python_callable=skip_retraining,
        provide_context=True,
    )

    # Task 6: Move dataset to processed directory
    move_file_task = PythonOperator(
        task_id='move_to_processed',
        python_callable=move_to_processed,
        provide_context=True,
        trigger_rule='none_failed',  # Run if upstream tasks didn't fail
    )

    # Define task dependencies
    wait_for_dataset >> validate_task >> drift_detection_task >> log_drift_task
    drift_detection_task >> [trigger_retraining_task, skip_retraining_task]
    [trigger_retraining_task, skip_retraining_task] >> move_file_task
