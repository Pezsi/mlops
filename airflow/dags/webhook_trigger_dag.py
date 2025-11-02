"""
Webhook/API Trigger DAG - External API Integration

This DAG is designed to be triggered via external API calls (webhooks).
It demonstrates event-based triggering from external systems like:
- CI/CD pipelines
- External monitoring systems
- Manual triggers from web interfaces
- Integration with other microservices

Trigger via API:
POST /api/v1/dags/webhook_triggered_training/dagRuns
Authorization: Basic <base64_encoded_credentials>
Content-Type: application/json

{
    "conf": {
        "model_name": "wine_quality_rf_model",
        "trigger_source": "ci_cd_pipeline",
        "dataset_version": "v2.1.0",
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 10
        }
    }
}
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import sys

sys.path.insert(0, '/opt/airflow')

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email': ['admin@example.com'],
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=3),
}


def parse_trigger_config(**kwargs):
    """
    Parse configuration from API trigger request.
    The 'conf' parameter contains custom configuration passed via API.
    """
    dag_run = kwargs['dag_run']
    conf = dag_run.conf or {}

    print("=" * 60)
    print("WEBHOOK TRIGGER RECEIVED")
    print("=" * 60)

    trigger_info = {
        'trigger_source': conf.get('trigger_source', 'unknown'),
        'model_name': conf.get('model_name', 'default_model'),
        'dataset_version': conf.get('dataset_version', 'latest'),
        'hyperparameters': conf.get('hyperparameters', {}),
        'force_training': conf.get('force_training', False),
        'deployment_target': conf.get('deployment_target', 'staging'),
        'triggered_at': datetime.now().isoformat(),
        'triggered_by_user': conf.get('user', 'api'),
    }

    print(f"Trigger Source: {trigger_info['trigger_source']}")
    print(f"Model Name: {trigger_info['model_name']}")
    print(f"Dataset Version: {trigger_info['dataset_version']}")
    print(f"Hyperparameters: {trigger_info['hyperparameters']}")
    print(f"Force Training: {trigger_info['force_training']}")
    print(f"Deployment Target: {trigger_info['deployment_target']}")
    print("=" * 60)

    # Store in XCom for downstream tasks
    kwargs['ti'].xcom_push(key='trigger_config', value=trigger_info)

    return trigger_info


def validate_trigger_request(**kwargs):
    """
    Validate the trigger request to ensure all required parameters are present.
    """
    ti = kwargs['ti']
    trigger_config = ti.xcom_pull(task_ids='parse_trigger_config', key='trigger_config')

    print("Validating trigger request...")

    validation_errors = []

    # Validate model name
    valid_models = ['wine_quality_rf_model', 'wine_quality_xgb_model', 'wine_quality_nn_model']
    if trigger_config['model_name'] not in valid_models:
        validation_errors.append(f"Invalid model name: {trigger_config['model_name']}")

    # Validate deployment target
    valid_targets = ['staging', 'production', 'development']
    if trigger_config['deployment_target'] not in valid_targets:
        validation_errors.append(f"Invalid deployment target: {trigger_config['deployment_target']}")

    # Validate hyperparameters (if provided)
    if trigger_config['hyperparameters']:
        required_params = ['n_estimators', 'max_depth']
        for param in required_params:
            if param not in trigger_config['hyperparameters']:
                print(f"Warning: Missing hyperparameter '{param}', will use default")

    if validation_errors:
        error_msg = "Validation failed:\n" + "\n".join(validation_errors)
        print(error_msg)
        raise ValueError(error_msg)

    print("Validation passed!")
    return True


def check_training_needed(**kwargs):
    """
    Determine if training is actually needed based on trigger configuration.
    Can skip training if model is recent and no force flag is set.
    """
    ti = kwargs['ti']
    trigger_config = ti.xcom_pull(task_ids='parse_trigger_config', key='trigger_config')

    force_training = trigger_config['force_training']

    if force_training:
        print("Force training flag is set. Proceeding with training.")
        return 'prepare_training_environment'

    # Check if recent training exists
    print("Checking for recent training runs...")

    # In real scenario, query MLflow or metadata DB
    # For now, simulate check
    hours_since_last_training = 5  # Simulated

    if hours_since_last_training < 6:
        print(f"Recent training exists ({hours_since_last_training}h ago). Skipping training.")
        return 'skip_training'
    else:
        print(f"Last training was {hours_since_last_training}h ago. Proceeding with training.")
        return 'prepare_training_environment'


def prepare_training_environment(**kwargs):
    """
    Prepare training environment based on trigger configuration.
    Set up hyperparameters, load specific dataset version, etc.
    """
    ti = kwargs['ti']
    trigger_config = ti.xcom_pull(task_ids='parse_trigger_config', key='trigger_config')

    print("Preparing training environment...")
    print(f"  - Loading dataset version: {trigger_config['dataset_version']}")
    print(f"  - Configuring hyperparameters: {trigger_config['hyperparameters']}")
    print(f"  - Target deployment: {trigger_config['deployment_target']}")

    training_config = {
        'dataset_path': f"/opt/airflow/data/datasets/{trigger_config['dataset_version']}/winequality-red.csv",
        'model_output_path': f"/opt/airflow/models/{trigger_config['model_name']}_webhook.pkl",
        'hyperparameters': trigger_config['hyperparameters'],
        'mlflow_experiment': f"webhook_trigger_{trigger_config['trigger_source']}"
    }

    ti.xcom_push(key='training_config', value=training_config)

    print("Training environment prepared!")
    return training_config


def execute_training(**kwargs):
    """
    Execute model training with the configured parameters.
    This is a simplified version; in real scenario, it would call training pipeline.
    """
    ti = kwargs['ti']
    trigger_config = ti.xcom_pull(task_ids='parse_trigger_config', key='trigger_config')
    training_config = ti.xcom_pull(task_ids='prepare_training_environment', key='training_config')

    print("=" * 60)
    print("EXECUTING TRAINING")
    print("=" * 60)
    print(f"Model: {trigger_config['model_name']}")
    print(f"Dataset: {training_config['dataset_path']}")
    print(f"Hyperparameters: {training_config['hyperparameters']}")
    print("=" * 60)

    # In real scenario, this would trigger the actual training DAG
    # For demonstration, we'll simulate training

    import time
    print("Training model...")
    time.sleep(2)  # Simulate training time

    # Simulated results
    training_results = {
        'status': 'completed',
        'run_id': f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'metrics': {
            'r2_score': 0.87,
            'rmse': 0.62,
            'mae': 0.48,
            'training_time_seconds': 45
        }
    }

    print(f"Training completed: {training_results}")

    ti.xcom_push(key='training_results', value=training_results)

    return training_results


def skip_training_task(**kwargs):
    """Task executed when training is skipped."""
    print("Training skipped based on trigger configuration.")
    return "skipped"


def deploy_model(**kwargs):
    """
    Deploy trained model to the target environment specified in trigger.
    """
    ti = kwargs['ti']
    trigger_config = ti.xcom_pull(task_ids='parse_trigger_config', key='trigger_config')
    training_results = ti.xcom_pull(task_ids='execute_model_training', key='training_results')

    deployment_target = trigger_config['deployment_target']

    print("=" * 60)
    print(f"DEPLOYING TO {deployment_target.upper()}")
    print("=" * 60)

    if training_results:
        print(f"Deploying model run: {training_results['run_id']}")
        print(f"Model metrics: {training_results['metrics']}")
    else:
        print("No new training results. Deploying latest model.")

    print(f"Deployment to {deployment_target} completed successfully!")

    return {'deployment_target': deployment_target, 'status': 'deployed'}


def send_webhook_response(**kwargs):
    """
    Send response back to triggering system via webhook.
    This would POST results to a callback URL if provided in trigger config.
    """
    ti = kwargs['ti']
    trigger_config = ti.xcom_pull(task_ids='parse_trigger_config', key='trigger_config')
    training_results = ti.xcom_pull(task_ids='execute_model_training', key='training_results')

    callback_url = trigger_config.get('callback_url')

    response_payload = {
        'dag_id': 'webhook_triggered_training',
        'execution_date': kwargs['execution_date'].isoformat(),
        'status': 'success',
        'trigger_source': trigger_config['trigger_source'],
        'training_results': training_results,
        'message': 'Model training and deployment completed successfully'
    }

    print("=" * 60)
    print("WEBHOOK RESPONSE")
    print("=" * 60)

    if callback_url:
        print(f"Sending response to callback URL: {callback_url}")
        # In real scenario, would POST to callback_url
        # requests.post(callback_url, json=response_payload)
    else:
        print("No callback URL provided. Response:")
        print(response_payload)

    print("=" * 60)

    return response_payload


# Define the DAG
with DAG(
    dag_id='webhook_triggered_training',
    default_args=default_args,
    description='API/Webhook triggered training pipeline with custom configuration',
    schedule_interval=None,  # Only triggered via API, no schedule
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['webhook', 'api-trigger', 'event-driven', 'external'],
    # Enable DAG to be triggered via API
    is_paused_upon_creation=False,
) as dag:

    # Add DAG documentation
    dag.doc_md = __doc__

    # Task 1: Parse trigger configuration from API request
    parse_config_task = PythonOperator(
        task_id='parse_trigger_config',
        python_callable=parse_trigger_config,
        provide_context=True,
        doc_md="""
        Parse and extract configuration from API trigger request.
        Validates that all required parameters are present.
        """
    )

    # Task 2: Validate trigger request
    validate_task = PythonOperator(
        task_id='validate_trigger_request',
        python_callable=validate_trigger_request,
        provide_context=True,
    )

    # Task 3: Check if training is needed
    check_training_task = BranchPythonOperator(
        task_id='check_training_needed',
        python_callable=check_training_needed,
        provide_context=True,
    )

    # Task 4a: Prepare training environment
    prepare_task = PythonOperator(
        task_id='prepare_training_environment',
        python_callable=prepare_training_environment,
        provide_context=True,
    )

    # Task 4b: Skip training
    skip_task = PythonOperator(
        task_id='skip_training',
        python_callable=skip_training_task,
        provide_context=True,
    )

    # Task 5: Execute training
    training_task = PythonOperator(
        task_id='execute_model_training',
        python_callable=execute_training,
        provide_context=True,
        trigger_rule='none_failed',  # Run if upstream didn't fail
    )

    # Task 6: Deploy model
    deploy_task = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model,
        provide_context=True,
        trigger_rule='none_failed',
    )

    # Task 7: Send webhook response
    webhook_response_task = PythonOperator(
        task_id='send_webhook_response',
        python_callable=send_webhook_response,
        provide_context=True,
        trigger_rule='all_done',  # Always run to send response
    )

    # Define task dependencies
    parse_config_task >> validate_task >> check_training_task
    check_training_task >> [prepare_task, skip_task]
    prepare_task >> training_task
    [training_task, skip_task] >> deploy_task >> webhook_response_task
