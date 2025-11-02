"""
External Task Sensor DAG - DAG Dependencies

This DAG demonstrates event-based triggering based on other DAG completions.
Uses ExternalTaskSensor to wait for upstream DAGs and create complex workflows.

Use Case:
- Wait for data ingestion DAG to complete
- Wait for model training DAG to complete
- Trigger downstream tasks like model evaluation, deployment, or reporting

This creates a dependency graph across multiple DAGs for complex orchestration.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor, ExternalTaskMarker
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import sys

sys.path.insert(0, '/opt/airflow')

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email': ['admin@example.com'],
    'email_on_failure': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def run_model_evaluation(**kwargs):
    """
    Run comprehensive model evaluation after training completes.
    This task waits for the training DAG to finish.
    """
    print("Running comprehensive model evaluation...")

    # Pull information from upstream DAG via XCom
    ti = kwargs['ti']

    # In a real scenario, we'd fetch XCom data from the external DAG
    print("Evaluating model performance on holdout dataset...")
    print("Generating evaluation reports...")
    print("Computing business metrics...")

    evaluation_results = {
        'holdout_accuracy': 0.85,
        'business_metric_improvement': 0.12,
        'inference_time_ms': 5.2,
        'evaluation_status': 'passed'
    }

    ti.xcom_push(key='evaluation_results', value=evaluation_results)

    print(f"Evaluation Results: {evaluation_results}")
    return evaluation_results


def run_ab_test_setup(**kwargs):
    """
    Set up A/B test configuration for the newly trained model.
    Waits for both training and evaluation to complete.
    """
    print("Setting up A/B test for new model...")

    ab_test_config = {
        'control_model': 'production_v1.2',
        'treatment_model': 'candidate_v1.3',
        'traffic_split': 0.1,  # 10% traffic to new model
        'duration_days': 7,
        'success_metrics': ['accuracy', 'inference_time', 'user_satisfaction']
    }

    print(f"A/B Test Configuration: {ab_test_config}")

    return ab_test_config


def deploy_to_staging(**kwargs):
    """
    Deploy model to staging environment after evaluation passes.
    """
    ti = kwargs['ti']
    evaluation_results = ti.xcom_pull(
        task_ids='evaluate_model_performance',
        key='evaluation_results'
    )

    if evaluation_results['evaluation_status'] == 'passed':
        print("Deploying model to staging environment...")
        print("  - Creating staging endpoint")
        print("  - Loading model artifacts")
        print("  - Running smoke tests")
        print("Model successfully deployed to staging!")
        return "deployed"
    else:
        print("Evaluation did not pass. Skipping deployment.")
        return "skipped"


def generate_deployment_report(**kwargs):
    """
    Generate comprehensive deployment report.
    Includes metrics, test results, and deployment status.
    """
    print("Generating deployment report...")

    report = {
        'deployment_date': datetime.now().isoformat(),
        'environment': 'staging',
        'status': 'success',
        'tests_passed': 45,
        'tests_failed': 0,
        'deployment_duration_minutes': 8
    }

    print(f"Deployment Report: {report}")

    # In real scenario, this would be saved to S3/database/sent via email
    return report


def notify_stakeholders(**kwargs):
    """
    Send notification to stakeholders about model deployment.
    """
    print("Notifying stakeholders...")
    print("  - Sending email to ML team")
    print("  - Posting to Slack #ml-deployments channel")
    print("  - Updating JIRA ticket status")
    print("Notifications sent successfully!")


# ============ Main DAG: Model Deployment Pipeline ============

with DAG(
    dag_id='model_deployment_pipeline',
    default_args=default_args,
    description='Waits for training DAG completion and handles deployment',
    schedule_interval='0 4 * * *',  # Run daily at 4 AM (after training at 2 AM)
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['deployment', 'external-sensor', 'production'],
) as deployment_dag:

    # Sensor 1: Wait for training DAG to complete
    wait_for_training = ExternalTaskSensor(
        task_id='wait_for_training_completion',
        external_dag_id='train_wine_quality_model',
        external_task_id='train_model',  # Wait specifically for training task
        allowed_states=['success'],
        failed_states=['failed', 'skipped'],
        mode='reschedule',
        poke_interval=60,  # Check every 60 seconds
        timeout=60 * 60 * 2,  # Timeout after 2 hours
        execution_delta=timedelta(hours=2),  # Look for run 2 hours earlier
    )

    # Task: Evaluate model performance
    evaluate_task = PythonOperator(
        task_id='evaluate_model_performance',
        python_callable=run_model_evaluation,
        provide_context=True,
    )

    # Task: Set up A/B test
    ab_test_setup_task = PythonOperator(
        task_id='setup_ab_test',
        python_callable=run_ab_test_setup,
        provide_context=True,
    )

    # Task: Deploy to staging
    deploy_staging_task = PythonOperator(
        task_id='deploy_to_staging',
        python_callable=deploy_to_staging,
        provide_context=True,
    )

    # Task: Generate report
    report_task = PythonOperator(
        task_id='generate_deployment_report',
        python_callable=generate_deployment_report,
        provide_context=True,
    )

    # Task: Notify stakeholders
    notify_task = PythonOperator(
        task_id='notify_stakeholders',
        python_callable=notify_stakeholders,
        provide_context=True,
    )

    # Dependencies
    wait_for_training >> evaluate_task
    evaluate_task >> [ab_test_setup_task, deploy_staging_task]
    [ab_test_setup_task, deploy_staging_task] >> report_task >> notify_task


# ============ Companion DAG: Multiple Dependencies Example ============

with DAG(
    dag_id='data_pipeline_orchestrator',
    default_args=default_args,
    description='Orchestrates multiple data pipelines with complex dependencies',
    schedule_interval='0 1 * * *',  # Run daily at 1 AM
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['orchestration', 'external-sensor', 'data-pipeline'],
) as orchestrator_dag:

    def process_aggregated_data(**kwargs):
        """Process data after all ingestion pipelines complete."""
        print("Processing aggregated data from all sources...")
        print("  - Merging datasets")
        print("  - Applying transformations")
        print("  - Running quality checks")
        return "completed"

    def trigger_training_if_ready(**kwargs):
        """Trigger training DAG if data processing is successful."""
        print("All data pipelines completed successfully.")
        print("Triggering model training...")
        return {'status': 'ready_for_training'}

    # Wait for dataset sensor DAG (if it ran)
    wait_for_dataset_processing = ExternalTaskSensor(
        task_id='wait_for_dataset_sensor',
        external_dag_id='dataset_sensor_event_trigger',
        external_task_id='move_to_processed',
        allowed_states=['success', 'skipped'],  # OK if it didn't run
        failed_states=['failed'],
        mode='reschedule',
        poke_interval=60,
        timeout=60 * 30,  # 30 minutes timeout
        execution_delta=timedelta(hours=0),
    )

    # Process aggregated data
    process_data_task = PythonOperator(
        task_id='process_aggregated_data',
        python_callable=process_aggregated_data,
        provide_context=True,
    )

    # Conditionally trigger training
    check_and_trigger_task = PythonOperator(
        task_id='check_and_trigger_training',
        python_callable=trigger_training_if_ready,
        provide_context=True,
    )

    # Trigger training DAG
    trigger_training = TriggerDagRunOperator(
        task_id='trigger_model_training',
        trigger_dag_id='train_wine_quality_model',
        conf={'triggered_by': 'orchestrator', 'data_ready': True},
    )

    # Mark task completion for downstream sensors
    mark_completion = ExternalTaskMarker(
        task_id='mark_orchestrator_complete',
        external_dag_id='model_deployment_pipeline',
        external_task_id='wait_for_training_completion',
    )

    # Dependencies
    wait_for_dataset_processing >> process_data_task >> check_and_trigger_task
    check_and_trigger_task >> trigger_training >> mark_completion


# ============ Companion DAG: Monitoring and Alerts ============

with DAG(
    dag_id='monitoring_and_alerting',
    default_args=default_args,
    description='Monitors all DAGs and sends alerts on failures',
    schedule_interval='*/15 * * * *',  # Run every 15 minutes
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['monitoring', 'alerts', 'external-sensor'],
) as monitoring_dag:

    def check_pipeline_health(**kwargs):
        """
        Check health of all pipelines and send alerts if needed.
        """
        print("Checking pipeline health...")

        health_status = {
            'training_dag': 'healthy',
            'deployment_dag': 'healthy',
            'sensor_dag': 'healthy',
            'orchestrator_dag': 'healthy',
            'overall_status': 'all_systems_operational'
        }

        print(f"Health Status: {health_status}")
        return health_status

    def collect_pipeline_metrics(**kwargs):
        """
        Collect metrics from all DAGs for monitoring dashboard.
        """
        print("Collecting pipeline metrics...")

        metrics = {
            'dag_runs_last_hour': 12,
            'success_rate': 0.95,
            'average_duration_minutes': 25,
            'failed_tasks': 1,
            'pending_tasks': 3
        }

        print(f"Pipeline Metrics: {metrics}")
        return metrics

    def send_health_report(**kwargs):
        """Send health report to monitoring system."""
        print("Sending health report to monitoring dashboard...")
        print("  - Updating Grafana dashboard")
        print("  - Logging to monitoring database")
        print("Health report sent successfully!")

    # Tasks
    health_check_task = PythonOperator(
        task_id='check_pipeline_health',
        python_callable=check_pipeline_health,
        provide_context=True,
    )

    metrics_task = PythonOperator(
        task_id='collect_pipeline_metrics',
        python_callable=collect_pipeline_metrics,
        provide_context=True,
    )

    report_task = PythonOperator(
        task_id='send_health_report',
        python_callable=send_health_report,
        provide_context=True,
    )

    # Dependencies
    [health_check_task, metrics_task] >> report_task
