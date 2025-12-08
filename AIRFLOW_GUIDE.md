# Apache Airflow Orchestration Guide

Technical documentation for the Apache Airflow workflows integrated into the Wine Quality MLOps platform. This guide explains the DAG architecture, trigger mechanisms, task dependencies, and operational procedures.

---

## Table of Contents

1. [Introduction to Airflow in MLOps](#introduction-to-airflow-in-mlops)
2. [DAG Architecture](#dag-architecture)
3. [Available Workflows](#available-workflows)
4. [Trigger Mechanisms](#trigger-mechanisms)
5. [Task Design Patterns](#task-design-patterns)
6. [Operational Procedures](#operational-procedures)
7. [Monitoring and Debugging](#monitoring-and-debugging)
8. [Configuration](#configuration)

---

## Introduction to Airflow in MLOps

### Why Workflow Orchestration Matters

Machine learning operations involve complex sequences of dependent tasks. Data must load before preprocessing. Preprocessing must complete before training. Training must finish before evaluation. Evaluation must pass before deployment. Managing these dependencies manually introduces errors and delays.

Apache Airflow provides a framework for defining, scheduling, and monitoring these task sequences. Rather than running scripts manually and hoping they execute in the right order, Airflow ensures tasks run exactly when their dependencies complete. If a task fails, Airflow can retry it, alert operators, or skip dependent tasks as configured.

### Airflow Concepts

Airflow organizes work around several core concepts that shape how pipelines are designed and operated.

**DAGs (Directed Acyclic Graphs)**: A DAG represents a complete workflow as a collection of tasks with dependencies. The "directed" aspect means dependencies flow in one direction. The "acyclic" constraint prevents circular dependencies that would create infinite loops. Each DAG has a unique identifier, schedule, and configuration.

**Tasks**: Individual units of work within a DAG. A task might load data, train a model, send a notification, or any other discrete operation. Tasks define what happens but not when. Airflow determines execution timing based on dependencies and scheduling.

**Operators**: Templates for creating tasks. The PythonOperator runs Python functions. The BashOperator executes shell commands. The BranchPythonOperator enables conditional workflows. Custom operators encapsulate reusable patterns.

**Task Instances**: A specific execution of a task for a particular DAG run. The same task definition creates many task instances across different runs. Task instances track state: queued, running, success, failed, or skipped.

**DAG Runs**: A specific execution of a DAG for a particular logical date. Each DAG run contains task instances for all tasks in the DAG. Multiple DAG runs can execute simultaneously for different logical dates.

### Airflow in This Platform

This platform uses Airflow to automate the complete ML lifecycle. Training pipelines run on schedule or in response to events. Deployment pipelines promote validated models. Monitoring pipelines check production health. The combination ensures continuous operation without manual intervention.

The Airflow components run in Docker containers alongside other platform services. The webserver provides the user interface at port 8081. The scheduler monitors for scheduled and triggered work. Worker processes execute the actual tasks.

---

## DAG Architecture

### Design Principles

The DAGs in this platform follow several design principles that improve reliability and maintainability.

**Single Responsibility**: Each DAG handles one logical workflow. The training DAG focuses on model training. The deployment DAG handles promotion. The monitoring DAG checks health. This separation makes each DAG easier to understand, test, and modify.

**Idempotency**: Tasks produce the same result regardless of how many times they run. If a task fails partway through and retries, the retry starts fresh rather than building on partial state. This property enables safe retries without side effects.

**Graceful Failure Handling**: Tasks anticipate failures and handle them appropriately. Missing data causes a clean error rather than a cryptic crash. Failed validations skip deployment rather than promoting broken models. Clear error messages enable quick diagnosis.

**Observability**: Tasks log their actions comprehensively. Operators can understand what happened by reading logs. Metrics capture key values for monitoring. Alerts notify teams of issues requiring attention.

### DAG File Organization

DAG definitions live in the `airflow/dags/` directory. Airflow's scheduler scans this directory periodically, loading any Python files that define DAG objects. Each file typically defines one DAG, though complex workflows might split across files with shared utilities.

The utility functions that DAGs call live in `airflow/utils/`. This separation keeps DAG files focused on workflow definition while reusable logic stays in shared modules.

Supporting infrastructure includes custom operators, hooks for external systems, and configuration files. These components extend Airflow's capabilities for ML-specific needs.

### Task Dependencies

Dependencies between tasks create the workflow structure. When Task B depends on Task A, Airflow ensures A completes successfully before B starts. Multiple independent tasks can run in parallel when dependencies allow.

Simple linear dependencies chain tasks sequentially. Training depends on preprocessing, which depends on data loading. This pattern suits workflows where each step needs the previous step's output.

Fan-out patterns run multiple independent tasks after a common predecessor. After data loads, parallel tasks might train different model types simultaneously. This pattern improves throughput by utilizing available resources.

Fan-in patterns consolidate multiple predecessors into a single successor. After multiple models train, a comparison task evaluates them all. This pattern enables aggregation and selection.

Branching patterns choose between alternative paths based on runtime conditions. If model performance exceeds a threshold, deploy it. Otherwise, skip deployment. This pattern enables conditional logic within workflows.

---

## Available Workflows

### Daily Model Training DAG

The daily training DAG implements scheduled retraining to incorporate new data and maintain model freshness. Running overnight ensures updated models are ready for the next business day.

**Purpose**: Automate routine model updates without manual intervention. As new wine quality data accumulates, models trained on larger datasets may improve. Regular retraining captures these improvements automatically.

**Schedule**: Daily at 2:00 AM local time. This timing avoids peak usage hours while completing before business hours begin. The schedule is configurable through Airflow variables.

**Workflow Structure**:

The DAG begins with metadata initialization, recording the run's start time and configuration. This metadata supports tracking and debugging.

The training task executes the model training pipeline. It loads current data, preprocesses features, trains the model with cross-validation, and logs results to MLflow. The task captures all parameters and metrics for reproducibility.

The branch decision evaluates whether the new model improves upon the current production model. The task queries MLflow for the production model's metrics and compares against the new model. Improvement above a configured threshold proceeds to deployment.

If the model improves, the staging deployment task registers the new model version and deploys it to a staging environment. Staging deployment enables testing before production exposure.

Production deployment promotes the staged model to production after staging validation passes. This staged approach catches problems before they affect users.

The notification task sends alerts about the training outcome. Successful deployments notify relevant teams. Failures escalate to on-call personnel. Skipped deployments log for review without alerts.

If the model does not improve, the skip notification path logs the outcome without deployment. This prevents deploying models that would degrade quality.

**Configuration**: Key parameters include performance thresholds for deployment decisions, notification recipients, and MLflow experiment names. These configure through Airflow variables for flexibility without code changes.

The implementation resides in `airflow/dags/daily_model_training_with_notification.py`.

### Dataset Sensor DAG

The dataset sensor DAG responds to new data files rather than running on schedule. When external systems deposit new datasets, this DAG automatically triggers processing.

**Purpose**: Enable event-driven processing when data arrives unpredictably. Some data sources update irregularly based on external factors. Polling on schedule might miss data or waste resources checking when no data exists.

**Trigger Mechanism**: A FileSensor monitors a configured directory for new files matching specified patterns. When matching files appear, the sensor triggers the downstream workflow. Configurable poke intervals control how frequently the sensor checks.

**Workflow Structure**:

The sensor task continuously monitors for new files. When files appear, it records the file paths and triggers downstream tasks. The sensor handles multiple files arriving simultaneously.

Data validation checks the new files for expected format and content. Schema validation ensures required columns exist with appropriate types. Statistical validation checks for anomalous distributions. Invalid files trigger alerts rather than processing.

The training trigger initiates model training using the validated data. Rather than duplicating training logic, this task triggers the training DAG with appropriate parameters. This delegation avoids code duplication.

Cleanup tasks archive or remove processed files to prevent reprocessing. The cleanup strategy depends on data retention requirements and storage constraints.

**Use Cases**: This pattern suits scenarios with irregular data delivery. External partners might provide data batches at unpredictable times. Manual data collection processes might complete at varying intervals. The sensor approach handles these patterns gracefully.

The implementation resides in `airflow/dags/dataset_sensor_dag.py`.

### External Task Sensor DAG

The external task sensor DAG coordinates multi-DAG workflows by waiting for tasks in other DAGs to complete. This enables complex orchestration across independent workflows.

**Purpose**: Coordinate dependent workflows that span multiple DAGs. The model deployment might need to wait for training completion. Monitoring might need deployed models before meaningful checks. External task sensors enable these cross-DAG dependencies.

**Trigger Mechanism**: ExternalTaskSensor waits for specified tasks in other DAGs to reach success state. The sensor can wait for specific execution dates or the most recent run. Timeout configuration prevents infinite waits.

**Workflow Structure**:

The sensor task monitors for the target task completion. Configuration specifies the external DAG ID, task ID, and execution date logic. When the external task succeeds, downstream tasks proceed.

Post-dependency tasks execute work that requires the external task's output. For deployment coordination, this might involve moving models between environments. For monitoring coordination, this might involve updating dashboards.

**Coordination Patterns**:

Sequential DAG execution ensures DAGs run in order even when scheduled independently. DAG B's sensor waits for DAG A, guaranteeing A completes first regardless of scheduling vagaries.

Resource coordination prevents multiple resource-intensive DAGs from running simultaneously. Each DAG's sensor waits for others to finish, ensuring only one heavy workload runs at a time.

Data handoff coordination ensures downstream DAGs see committed outputs from upstream DAGs. The sensor waits not just for task completion but for data availability.

The implementation resides in `airflow/dags/external_task_sensor_dag.py`.

### Webhook Triggered Training DAG

The webhook DAG responds to HTTP requests, enabling external systems to trigger workflows. CI/CD pipelines, data platforms, and operational tools integrate through this API.

**Purpose**: Enable programmatic workflow triggering from external systems. When code changes deploy, CI/CD can trigger model validation. When data platforms complete ETL, they can trigger training. This integration creates cohesive automation across tools.

**Trigger Mechanism**: A REST API endpoint accepts HTTP POST requests with configuration parameters. The webhook API validates requests, authenticates callers, and triggers the appropriate DAG runs. API key authentication prevents unauthorized triggering.

**Workflow Structure**:

The webhook API receives trigger requests at a dedicated endpoint. Request validation ensures required parameters exist with valid values. Authentication verifies the API key matches configured secrets.

Parameter extraction pulls configuration from the request body. Callers can specify model names, hyperparameter overrides, force-training flags, and other options. Default values apply when parameters are omitted.

DAG triggering invokes Airflow's trigger mechanism with extracted parameters. The triggered DAG run receives parameters through Airflow's conf mechanism, making them available to all tasks.

Status tracking provides feedback about the triggered run. The API returns the DAG run ID for tracking. Callers can poll for completion or receive webhook callbacks.

**Integration Examples**:

CI/CD integration triggers model validation after code deployments. When training code changes, the pipeline automatically validates that models still train correctly and meet performance requirements.

Data platform integration triggers training after ETL completion. When new data loads into the data warehouse, the platform notifies the webhook, initiating model updates with fresh data.

Operational integration enables manual triggering through tooling. Operations dashboards might include buttons that trigger retraining or deployment through the webhook API.

The webhook API implementation resides in `airflow/api/webhook_trigger_api.py`. The DAG definition resides in `airflow/dags/webhook_trigger_dag.py`.

### Primary Training DAG

The primary training DAG provides the core model training workflow used by other DAGs and manual execution. It encapsulates the complete training process in a reusable form.

**Purpose**: Provide a comprehensive training workflow that other DAGs can trigger or that operators can run manually. Centralizing training logic in one DAG ensures consistency across different triggering mechanisms.

**Workflow Structure**:

Data loading retrieves the training dataset from configured sources. The task handles data versioning, ensuring reproducibility by recording exactly which data version was used.

Preprocessing transforms raw data into model-ready features. Standard scaling normalizes feature distributions. Feature selection identifies relevant inputs. The preprocessing configuration logs to MLflow.

Model training fits the configured model architecture on preprocessed data. Cross-validation provides robust performance estimates. Hyperparameter optimization explores the search space. All parameters and intermediate results log to MLflow.

Model evaluation calculates comprehensive metrics on held-out test data. Beyond simple accuracy, evaluation captures multiple metrics that reveal different performance aspects. Evaluation results determine deployment eligibility.

Model comparison compares the new model against existing models. If a production model exists, comparison determines whether the new model improves sufficiently to warrant deployment. Without existing models, comparison against baselines occurs.

Registry update logs successful models to MLflow's model registry. Versioning tracks model lineage. Metadata captures all information needed to reproduce the model.

Deployment decisions determine whether to proceed with production deployment. Configurable thresholds define minimum improvement requirements. Conservative thresholds prevent unnecessary churn. Aggressive thresholds enable rapid iteration.

The implementation resides in `airflow/dags/train_wine_quality_dag.py`.

---

## Trigger Mechanisms

### Schedule-Based Triggering

Most DAGs run on defined schedules using cron expressions. Airflow's scheduler monitors these schedules and creates DAG runs at appropriate times.

**Cron Expressions**: Standard cron syntax defines schedules. Five fields specify minute, hour, day of month, month, and day of week. The expression "0 2 * * *" means 2:00 AM daily. The expression "0 0 * * 0" means midnight every Sunday.

**Catchup Behavior**: When DAGs enable catchup, Airflow creates runs for all missed schedules since the DAG's start date. This ensures no scheduled runs are skipped even if Airflow was down. Production DAGs typically disable catchup to avoid overwhelming backlog processing.

**Schedule Intervals**: The schedule_interval parameter accepts cron expressions, timedelta objects, or preset strings like "@daily" or "@hourly". Choose intervals that match business requirements and data freshness needs.

### Event-Based Triggering

Sensors enable event-driven triggering by monitoring for specific conditions. When conditions are met, sensors trigger downstream tasks.

**File Sensors**: Monitor directories for new files matching patterns. Useful for processing data drops from external systems. Configuration includes the file path pattern and poke interval.

**External Task Sensors**: Monitor for task completion in other DAGs. Enable coordination across workflows. Configuration includes the external DAG ID, task ID, and execution date logic.

**Custom Sensors**: Extend Airflow's sensor base class for custom conditions. The platform could implement sensors for MLflow model registration events, monitoring alert triggers, or external API availability.

### API-Based Triggering

The webhook API enables programmatic triggering from external systems. HTTP requests initiate DAG runs with custom parameters.

**Authentication**: API keys protect against unauthorized triggering. Keys configure through environment variables and validate on each request. Key rotation procedures update keys without service interruption.

**Parameter Passing**: Request bodies contain JSON parameters passed to DAG runs. Tasks access these parameters through Airflow's templating system. This enables dynamic workflow behavior based on trigger context.

**Response Handling**: The API returns immediately with the DAG run ID. Callers can poll the Airflow API for run status or implement callback webhooks for completion notification.

### Manual Triggering

Operators can trigger DAG runs manually through the Airflow UI or CLI. Manual triggering supports testing, recovery, and ad-hoc operations.

**UI Triggering**: The Airflow web interface provides trigger buttons for each DAG. Operators can optionally provide configuration parameters. Manual runs appear in the same history as scheduled runs.

**CLI Triggering**: The airflow command-line tool supports DAG triggering. This enables scripting and automation outside the UI. The CLI also supports clearing failed tasks and backfilling historical runs.

---

## Task Design Patterns

### Idempotent Task Design

Tasks should produce identical results regardless of how many times they execute. This property enables safe retries without accumulating side effects.

**Database Operations**: Use upsert patterns that insert or update rather than blind inserts. Check for existing records before creating new ones. Handle duplicate key errors gracefully.

**File Operations**: Write to temporary locations first, then move atomically to final locations. Check for existing outputs before processing. Support overwriting when appropriate.

**External API Calls**: Design for retry safety. POST operations that create resources should check for existing resources first. Use idempotency keys when APIs support them.

### Error Handling Patterns

Tasks should handle errors gracefully, providing clear information for debugging while maintaining workflow integrity.

**Expected Errors**: Anticipate common failure modes and handle them specifically. Missing data should produce a clear message about what's missing. Invalid inputs should explain the validation failure.

**Unexpected Errors**: Catch broad exceptions to prevent cryptic failures. Log full stack traces for debugging. Re-raise after logging to let Airflow handle retry logic.

**Retry Configuration**: Configure retries for transient failures. Set appropriate retry delays that give external systems time to recover. Limit retry counts to prevent infinite retry loops.

### Task Communication Patterns

Tasks often need to pass information to downstream tasks. Airflow provides several mechanisms for this communication.

**XCom**: Airflow's cross-communication system stores small values in the database. Upstream tasks push values. Downstream tasks pull them. Suitable for metadata, status flags, and small results. Not suitable for large data.

**External Storage**: For large data, tasks write to shared storage (databases, object stores, file systems). Downstream tasks read from the same locations. XCom might pass the storage location rather than the data itself.

**Task Context**: Airflow provides execution context including logical date, DAG run configuration, and task instance information. Tasks use this context to coordinate behavior and locate relevant data.

### Branching Patterns

BranchPythonOperator enables conditional workflow paths based on runtime decisions.

**Binary Branching**: Choose between two paths based on a condition. If model improves, deploy. Otherwise, skip. The branch function returns the task ID of the chosen path.

**Multi-Way Branching**: Choose among multiple paths. Different model types might trigger different deployment procedures. The branch function returns whichever task ID matches the condition.

**Skip Handling**: Tasks on non-chosen branches receive "skipped" state. Downstream tasks that depend on skipped tasks also skip. Join tasks can use trigger rules to run regardless of upstream skips.

---

## Operational Procedures

### Monitoring DAG Health

Regular monitoring ensures workflows execute successfully and issues receive prompt attention.

**Dashboard Review**: The Airflow UI provides at-a-glance status of all DAGs. Green indicates recent success. Red indicates recent failure. The grid view shows historical patterns revealing intermittent problems.

**Log Review**: Task logs contain detailed execution information. Successful tasks log their actions. Failed tasks log error messages and stack traces. Regular log review catches issues that don't trigger alerts.

**Metric Monitoring**: Airflow exports metrics about scheduler performance, task duration, and queue depth. Monitoring these metrics reveals systemic issues like growing backlogs or slowing tasks.

### Handling Failures

When tasks fail, operators must diagnose and resolve the issue while minimizing impact on downstream processes.

**Diagnosis**: Start with the failed task's logs. Error messages usually indicate the problem. Check recent changes that might have introduced the failure. Verify external dependencies are available.

**Resolution**: Fix the underlying problem before retrying. Retrying without fixing causes repeated failures. Document the root cause and resolution for future reference.

**Retry Strategies**: Clear the failed task to retry it. Clearing also clears downstream tasks that were skipped due to the failure. For persistent failures, consider running the workflow manually with debugging enabled.

### Backfill Operations

Backfilling runs historical DAG executions for dates that were missed or need reprocessing.

**When to Backfill**: Backfill when scheduled runs were missed due to outages. Backfill when logic changes require reprocessing historical dates. Backfill when data corrections require regenerating outputs.

**Backfill Execution**: The Airflow CLI provides backfill commands specifying date ranges. Backfill runs execute sequentially by default to avoid overwhelming resources. Parallel backfill is possible but requires careful resource management.

**Considerations**: Backfill creates many DAG runs that may take significant time to process. Monitor resource usage during backfill. Consider processing recent dates first when recency matters.

### DAG Updates and Deployments

Updating DAG definitions requires care to avoid disrupting running workflows.

**Code Deployment**: New DAG code deploys through normal deployment processes. Airflow's scheduler picks up changes within minutes. No restart required for DAG definition changes.

**Breaking Changes**: Changes that modify task IDs break historical continuity. Renamed tasks appear as new tasks with no history. Consider keeping task IDs stable when possible.

**Testing Changes**: Test DAG changes in development environments before production deployment. Verify task execution order, dependency handling, and error behavior.

---

## Monitoring and Debugging

### Airflow UI Navigation

The Airflow web interface provides comprehensive workflow visibility.

**DAGs View**: Lists all DAGs with status indicators. Quick access to trigger, pause, or view each DAG. Tags help organize DAGs by purpose.

**Grid View**: Shows task status across historical runs in a matrix format. Columns represent runs. Rows represent tasks. Colors indicate status. This view reveals patterns in success and failure.

**Graph View**: Displays task dependencies as a visual graph. Useful for understanding workflow structure and identifying bottlenecks.

**Gantt View**: Shows task duration over time. Reveals which tasks take longest and how parallelism affects total runtime.

**Task Instance Details**: Clicking a task instance shows its logs, rendered templates, XCom values, and other details. Essential for debugging failures.

### Log Analysis

Task logs contain detailed information about execution.

**Standard Output**: Print statements and logging calls appear in task logs. Use logging liberally to document what tasks do. Include relevant context like data counts and parameter values.

**Error Traces**: Exceptions include full stack traces in logs. These traces pinpoint exactly where failures occur. Read traces carefully to understand failure causes.

**Log Aggregation**: For production deployments, consider aggregating logs to centralized systems. This enables searching across all tasks and correlating with other system logs.

### Common Issues

Certain issues appear frequently and have established solutions.

**Import Errors**: DAG files with import errors don't load. Check the scheduler logs for import error messages. Fix the code and redeploy.

**Task Timeout**: Tasks exceeding execution timeout are killed. Increase timeout for legitimate long-running tasks. Investigate unexpected slowdowns.

**Resource Exhaustion**: Workers running out of memory or CPU cause failures. Increase worker resources or reduce task concurrency. Optimize tasks that consume excessive resources.

**Dependency Issues**: Incorrectly specified dependencies cause tasks to run out of order or deadlock. Review dependency definitions carefully. Test in development environments.

---

## Configuration

### Airflow Variables

Airflow Variables store configuration values accessible to all DAGs. Variables suit values that might change without code deployments.

**Usage**: Set variables through the UI or CLI. Access in DAGs using Variable.get(). Provide default values for resilience.

**Examples**: Performance thresholds for deployment decisions. Notification email addresses. External system URLs. Feature flags for conditional behavior.

### Airflow Connections

Connections store credentials for external systems. Connections separate sensitive credentials from DAG code.

**Configuration**: Set connections through the UI, CLI, or environment variables. Connections include host, port, login, password, and extra configuration.

**Usage**: Hooks use connections to authenticate with external systems. The MLflow hook uses connections to reach the tracking server. Database hooks use connections for database access.

### Environment Variables

Environment variables configure Airflow itself and provide secrets to tasks.

**Airflow Configuration**: Environment variables starting with AIRFLOW__ override airflow.cfg settings. This enables configuration without modifying files.

**Task Secrets**: Sensitive values like API keys can pass through environment variables. Tasks read these variables rather than hardcoding secrets.

### Docker Compose Configuration

The docker-compose.yml file defines the Airflow deployment for local development.

**Service Definitions**: Separate services for webserver, scheduler, and workers enable independent scaling. Shared volumes provide common access to DAG files and logs.

**Resource Limits**: Configure memory and CPU limits per service. Prevent individual components from consuming all available resources.

**Network Configuration**: Services communicate over a shared Docker network. Port mappings expose the webserver UI. Internal services communicate without external exposure.

---

## Version Information

- Version: 1.0.0
- Last Updated: December 2025
- Airflow Version: 2.7.3
- Compatible With: Docker Compose deployment, Kubernetes deployment
