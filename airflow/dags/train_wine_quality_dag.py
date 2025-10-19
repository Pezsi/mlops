"""
Airflow DAG for automated Wine Quality model training and deployment.

This DAG:
1. Trains a new Random Forest model
2. Compares performance with current production model
3. Updates model registry if new model performs better
4. Sends email notification if model doesn't improve
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
import sys
import os
import mlflow
from mlflow import MlflowClient
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Add project root to path
sys.path.insert(0, '/opt/airflow')

# MLflow configuration
MLFLOW_TRACKING_URI = "file:/opt/airflow/mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Model configuration
MODEL_NAME = "wine_quality_rf_model"
DATA_PATH = "/opt/airflow/data/winequality-red.csv"
MODEL_PATH = "/opt/airflow/models/rf_regressor.pkl"

# Default arguments for the DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email': ['admin@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}


def train_new_model(**kwargs):
    """Train a new wine quality model and log to MLflow."""
    from data.load_data import load_and_split_data
    from src.preprocessing import create_preprocessing_pipeline
    from src.train import train_model_with_grid_search
    from src.evaluate import evaluate_model

    print("Starting model training...")

    # Load and split data
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Create preprocessing pipeline
    preprocessing_pipeline = create_preprocessing_pipeline(use_pca=False)

    # Start MLflow run
    with mlflow.start_run(run_name=f"airflow_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Train model
        print("Training model with GridSearchCV...")
        model = train_model_with_grid_search(
            preprocessing_pipeline,
            X_train,
            y_train,
            cv_folds=5
        )

        # Evaluate model
        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=MODEL_NAME
        )

        # Save model locally
        joblib.dump(model, MODEL_PATH)

        # Store metrics in XCom for downstream tasks
        kwargs['ti'].xcom_push(key='new_r2_score', value=metrics['r2_score'])
        kwargs['ti'].xcom_push(key='new_rmse', value=metrics['rmse'])
        kwargs['ti'].xcom_push(key='run_id', value=mlflow.active_run().info.run_id)

        print(f"Model trained successfully. R²: {metrics['r2_score']:.4f}, RMSE: {metrics['rmse']:.4f}")

    return metrics['r2_score']


def compare_models(**kwargs):
    """
    Compare new model with production model.
    Returns task ID for branching decision.
    """
    ti = kwargs['ti']
    new_r2_score = ti.xcom_pull(task_ids='train_model', key='new_r2_score')
    new_rmse = ti.xcom_pull(task_ids='train_model', key='new_rmse')

    print(f"New model - R²: {new_r2_score:.4f}, RMSE: {new_rmse:.4f}")

    # Try to get latest production model metrics
    try:
        latest_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])

        if latest_versions:
            latest_version = latest_versions[0]
            run = client.get_run(latest_version.run_id)

            prod_r2_score = run.data.metrics.get('r2_score', 0)
            prod_rmse = run.data.metrics.get('rmse', float('inf'))

            print(f"Production model - R²: {prod_r2_score:.4f}, RMSE: {prod_rmse:.4f}")

            # Compare models (new model should have higher R² and lower RMSE)
            if new_r2_score >= prod_r2_score and new_rmse <= prod_rmse:
                print("New model is better or equal. Proceeding with deployment.")

                # Store comparison results
                kwargs['ti'].xcom_push(key='model_improved', value=True)
                kwargs['ti'].xcom_push(key='prod_r2_score', value=prod_r2_score)
                kwargs['ti'].xcom_push(key='prod_rmse', value=prod_rmse)

                return 'update_model_registry'
            else:
                print("New model is not better. Skipping deployment.")
                kwargs['ti'].xcom_push(key='model_improved', value=False)
                kwargs['ti'].xcom_push(key='prod_r2_score', value=prod_r2_score)
                kwargs['ti'].xcom_push(key='prod_rmse', value=prod_rmse)

                return 'send_notification'
        else:
            # No production model exists, deploy new model
            print("No production model found. Deploying new model.")
            kwargs['ti'].xcom_push(key='model_improved', value=True)
            return 'update_model_registry'

    except Exception as e:
        print(f"Error comparing models: {e}")
        print("Proceeding with deployment due to error.")
        kwargs['ti'].xcom_push(key='model_improved', value=True)
        return 'update_model_registry'


def update_model_registry(**kwargs):
    """Update MLflow model registry with new production model."""
    ti = kwargs['ti']
    run_id = ti.xcom_pull(task_ids='train_model', key='run_id')

    print(f"Updating model registry with run_id: {run_id}")

    try:
        # Get all versions of the model
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")

        # Archive old production models
        for version in versions:
            if version.current_stage == "Production":
                print(f"Archiving old production model version {version.version}")
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=version.version,
                    stage="Archived"
                )

        # Find the latest version (just created from this run)
        latest_version = max([v.version for v in versions])

        # Promote new model to production
        print(f"Promoting model version {latest_version} to Production")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest_version,
            stage="Production"
        )

        print(f"Model version {latest_version} is now in Production stage")

    except Exception as e:
        print(f"Error updating model registry: {e}")
        raise


def skip_deployment(**kwargs):
    """Dummy task when model deployment is skipped."""
    print("Model deployment skipped - new model did not improve performance.")
    return "skipped"


# Define the DAG
with DAG(
    dag_id='train_wine_quality_model',
    default_args=default_args,
    description='Daily training and evaluation of Wine Quality model',
    schedule_interval='0 2 * * *',  # Run daily at 2 AM
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['ml', 'wine-quality', 'training'],
) as dag:

    # Task 1: Train new model
    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_new_model,
        provide_context=True,
    )

    # Task 2: Compare models and decide next step
    compare_task = BranchPythonOperator(
        task_id='compare_models',
        python_callable=compare_models,
        provide_context=True,
    )

    # Task 3a: Update model registry (if model improved)
    update_registry_task = PythonOperator(
        task_id='update_model_registry',
        python_callable=update_model_registry,
        provide_context=True,
    )

    # Task 3b: Send email notification (if model didn't improve)
    notification_task = EmailOperator(
        task_id='send_notification',
        to='admin@example.com',
        subject='Wine Quality Model Training - No Improvement',
        html_content="""
        <h3>Wine Quality Model Training Notification</h3>
        <p>The new model's performance did not improve compared to the current production model.</p>
        <p><strong>Action:</strong> No deployment was performed.</p>
        <p>Please review the MLflow UI for detailed metrics.</p>
        """,
    )

    # Task 4: Skip deployment task
    skip_task = PythonOperator(
        task_id='skip_deployment',
        python_callable=skip_deployment,
        provide_context=True,
    )

    # Define task dependencies
    train_task >> compare_task
    compare_task >> [update_registry_task, notification_task]
