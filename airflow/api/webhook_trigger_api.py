"""
Webhook Trigger API - External DAG Triggering Service

This Flask API provides a simple interface for external systems to trigger
Airflow DAGs via webhooks. It wraps the Airflow REST API and provides
additional features like authentication, validation, and logging.

Endpoints:
- POST /trigger/training - Trigger model training
- POST /trigger/evaluation - Trigger model evaluation
- POST /trigger/deployment - Trigger model deployment
- GET /status/<dag_id>/<run_id> - Check DAG run status
- GET /health - Health check

Usage:
    python webhook_trigger_api.py

Example Request:
    curl -X POST http://localhost:8080/trigger/training \
      -H "Content-Type: application/json" \
      -H "X-API-Key: your-api-key" \
      -d '{
        "model_name": "wine_quality_rf_model",
        "trigger_source": "ci_cd",
        "force_training": true
      }'
"""

from flask import Flask, request, jsonify
from functools import wraps
import requests
import logging
import os
from datetime import datetime
import base64
import sys

sys.path.insert(0, '/opt/airflow')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
AIRFLOW_API_URL = os.getenv('AIRFLOW_API_URL', 'http://localhost:8081/api/v1')
AIRFLOW_USERNAME = os.getenv('AIRFLOW_USERNAME', 'admin')
AIRFLOW_PASSWORD = os.getenv('AIRFLOW_PASSWORD', 'admin')
API_KEY = os.getenv('WEBHOOK_API_KEY', 'mlops-secret-key-2025')

# Create basic auth header for Airflow API
auth_string = f"{AIRFLOW_USERNAME}:{AIRFLOW_PASSWORD}"
auth_bytes = auth_string.encode('utf-8')
auth_base64 = base64.b64encode(auth_bytes).decode('utf-8')
AIRFLOW_AUTH_HEADER = f"Basic {auth_base64}"


# ============ Authentication Decorator ============

def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')

        if not api_key:
            logger.warning("Request without API key")
            return jsonify({'error': 'API key required'}), 401

        if api_key != API_KEY:
            logger.warning(f"Invalid API key: {api_key}")
            return jsonify({'error': 'Invalid API key'}), 403

        return f(*args, **kwargs)

    return decorated_function


# ============ Helper Functions ============

def trigger_airflow_dag(dag_id, conf=None):
    """
    Trigger an Airflow DAG via REST API.

    Args:
        dag_id: ID of the DAG to trigger
        conf: Configuration dictionary to pass to DAG

    Returns:
        Response from Airflow API
    """
    url = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns"

    headers = {
        'Authorization': AIRFLOW_AUTH_HEADER,
        'Content-Type': 'application/json'
    }

    payload = {
        'conf': conf or {},
        'execution_date': datetime.utcnow().isoformat() + 'Z'
    }

    try:
        logger.info(f"Triggering DAG: {dag_id} with config: {conf}")
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()

        result = response.json()
        logger.info(f"DAG triggered successfully: {result.get('dag_run_id')}")

        return result

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to trigger DAG: {e}")
        raise


def get_dag_run_status(dag_id, dag_run_id):
    """
    Get status of a DAG run.

    Args:
        dag_id: ID of the DAG
        dag_run_id: ID of the DAG run

    Returns:
        DAG run status information
    """
    url = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns/{dag_run_id}"

    headers = {
        'Authorization': AIRFLOW_AUTH_HEADER,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get DAG run status: {e}")
        raise


# ============ API Endpoints ============

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'webhook-trigger-api',
        'timestamp': datetime.utcnow().isoformat()
    }), 200


@app.route('/trigger/training', methods=['POST'])
@require_api_key
def trigger_training():
    """
    Trigger model training DAG.

    Request Body:
    {
        "model_name": "wine_quality_rf_model",
        "trigger_source": "ci_cd_pipeline",
        "dataset_version": "v2.1.0",
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 10
        },
        "force_training": true,
        "callback_url": "https://your-system.com/callback"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Request body required'}), 400

        # Validate required fields
        model_name = data.get('model_name', 'wine_quality_rf_model')
        trigger_source = data.get('trigger_source', 'api')

        # Build configuration for DAG
        dag_conf = {
            'model_name': model_name,
            'trigger_source': trigger_source,
            'dataset_version': data.get('dataset_version', 'latest'),
            'hyperparameters': data.get('hyperparameters', {}),
            'force_training': data.get('force_training', False),
            'deployment_target': data.get('deployment_target', 'staging'),
            'callback_url': data.get('callback_url'),
            'user': data.get('user', 'api_user'),
            'triggered_at': datetime.utcnow().isoformat()
        }

        # Trigger DAG
        result = trigger_airflow_dag('webhook_triggered_training', conf=dag_conf)

        return jsonify({
            'status': 'triggered',
            'dag_id': 'webhook_triggered_training',
            'dag_run_id': result.get('dag_run_id'),
            'execution_date': result.get('execution_date'),
            'conf': dag_conf,
            'message': 'Training pipeline triggered successfully'
        }), 202

    except Exception as e:
        logger.error(f"Error triggering training: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/trigger/evaluation', methods=['POST'])
@require_api_key
def trigger_evaluation():
    """
    Trigger model evaluation DAG.

    Request Body:
    {
        "model_run_id": "run_20250102_120000",
        "evaluation_dataset": "holdout_v1",
        "metrics": ["accuracy", "precision", "recall", "f1"]
    }
    """
    try:
        data = request.get_json()

        dag_conf = {
            'model_run_id': data.get('model_run_id'),
            'evaluation_dataset': data.get('evaluation_dataset', 'default'),
            'metrics': data.get('metrics', ['r2_score', 'rmse', 'mae']),
            'trigger_source': data.get('trigger_source', 'api'),
            'triggered_at': datetime.utcnow().isoformat()
        }

        result = trigger_airflow_dag('model_deployment_pipeline', conf=dag_conf)

        return jsonify({
            'status': 'triggered',
            'dag_id': 'model_deployment_pipeline',
            'dag_run_id': result.get('dag_run_id'),
            'message': 'Evaluation pipeline triggered successfully'
        }), 202

    except Exception as e:
        logger.error(f"Error triggering evaluation: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/trigger/dataset-check', methods=['POST'])
@require_api_key
def trigger_dataset_check():
    """
    Trigger dataset validation and drift detection.

    Request Body:
    {
        "dataset_path": "/path/to/new/dataset.csv",
        "reference_dataset": "/path/to/reference.csv",
        "drift_threshold": 0.2
    }
    """
    try:
        data = request.get_json()

        dag_conf = {
            'dataset_path': data.get('dataset_path'),
            'reference_dataset': data.get('reference_dataset', REFERENCE_DATASET),
            'drift_threshold': data.get('drift_threshold', 0.2),
            'trigger_source': data.get('trigger_source', 'api'),
            'triggered_at': datetime.utcnow().isoformat()
        }

        result = trigger_airflow_dag('dataset_sensor_event_trigger', conf=dag_conf)

        return jsonify({
            'status': 'triggered',
            'dag_id': 'dataset_sensor_event_trigger',
            'dag_run_id': result.get('dag_run_id'),
            'message': 'Dataset validation triggered successfully'
        }), 202

    except Exception as e:
        logger.error(f"Error triggering dataset check: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/status/<dag_id>/<dag_run_id>', methods=['GET'])
@require_api_key
def get_status(dag_id, dag_run_id):
    """
    Get the status of a DAG run.

    Path Parameters:
        dag_id: ID of the DAG
        dag_run_id: ID of the DAG run
    """
    try:
        status = get_dag_run_status(dag_id, dag_run_id)

        return jsonify({
            'dag_id': dag_id,
            'dag_run_id': dag_run_id,
            'state': status.get('state'),
            'start_date': status.get('start_date'),
            'end_date': status.get('end_date'),
            'execution_date': status.get('execution_date'),
            'conf': status.get('conf'),
        }), 200

    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/dags', methods=['GET'])
@require_api_key
def list_dags():
    """
    List all available DAGs that can be triggered.
    """
    available_dags = [
        {
            'dag_id': 'webhook_triggered_training',
            'description': 'Webhook-triggered model training pipeline',
            'trigger_endpoint': '/trigger/training'
        },
        {
            'dag_id': 'model_deployment_pipeline',
            'description': 'Model evaluation and deployment pipeline',
            'trigger_endpoint': '/trigger/evaluation'
        },
        {
            'dag_id': 'dataset_sensor_event_trigger',
            'description': 'Dataset validation and drift detection',
            'trigger_endpoint': '/trigger/dataset-check'
        }
    ]

    return jsonify({
        'dags': available_dags,
        'count': len(available_dags)
    }), 200


@app.route('/docs', methods=['GET'])
def api_docs():
    """API documentation."""
    docs = {
        'service': 'MLOps Webhook Trigger API',
        'version': '1.0.0',
        'description': 'API for triggering Airflow DAGs via webhooks',
        'authentication': 'X-API-Key header required',
        'endpoints': [
            {
                'path': '/health',
                'method': 'GET',
                'description': 'Health check endpoint',
                'auth_required': False
            },
            {
                'path': '/trigger/training',
                'method': 'POST',
                'description': 'Trigger model training pipeline',
                'auth_required': True
            },
            {
                'path': '/trigger/evaluation',
                'method': 'POST',
                'description': 'Trigger model evaluation pipeline',
                'auth_required': True
            },
            {
                'path': '/trigger/dataset-check',
                'method': 'POST',
                'description': 'Trigger dataset validation',
                'auth_required': True
            },
            {
                'path': '/status/<dag_id>/<dag_run_id>',
                'method': 'GET',
                'description': 'Get DAG run status',
                'auth_required': True
            },
            {
                'path': '/dags',
                'method': 'GET',
                'description': 'List available DAGs',
                'auth_required': True
            }
        ],
        'example_request': {
            'url': 'POST /trigger/training',
            'headers': {
                'Content-Type': 'application/json',
                'X-API-Key': 'your-api-key'
            },
            'body': {
                'model_name': 'wine_quality_rf_model',
                'trigger_source': 'ci_cd',
                'force_training': True
            }
        }
    }

    return jsonify(docs), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


# ============ Main ============

if __name__ == '__main__':
    logger.info("Starting Webhook Trigger API...")
    logger.info(f"Airflow API URL: {AIRFLOW_API_URL}")

    app.run(
        host='0.0.0.0',
        port=int(os.getenv('WEBHOOK_API_PORT', 8080)),
        debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    )
