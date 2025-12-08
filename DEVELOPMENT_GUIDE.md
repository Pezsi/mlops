# Development and Deployment Guide

Technical documentation for setting up development environments, running the platform locally, and deploying with Docker. This guide covers everything needed to work with the Wine Quality MLOps platform.

---

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Running Without Docker](#running-without-docker)
3. [Docker Deployment](#docker-deployment)
4. [Docker Compose Services](#docker-compose-services)
5. [Local Development Workflow](#local-development-workflow)
6. [Testing](#testing)
7. [Code Quality](#code-quality)
8. [Troubleshooting](#troubleshooting)

---

## Development Environment Setup

### System Requirements

The platform runs on Linux, macOS, and Windows with WSL2. Development requires Python 3.8 or higher, with Python 3.10+ recommended for best compatibility. Docker deployment requires Docker Engine 20.10+ and Docker Compose 2.0+.

Memory requirements depend on usage patterns. Training models requires approximately 2GB of available RAM. Running the full Docker stack requires approximately 4GB. Development workstations with 8GB or more RAM provide comfortable headroom.

Disk space requirements include the codebase (approximately 50MB), Python virtual environment (approximately 500MB), Docker images (approximately 2GB), and MLflow artifacts (grows with experiment count). Plan for at least 5GB of available disk space.

### Python Environment Setup

Virtual environments isolate project dependencies from system Python packages. This isolation prevents conflicts between projects and ensures reproducible environments.

**Creating the Environment**

Navigate to the project directory. Create a virtual environment in a subdirectory, conventionally named venv. This directory stores the isolated Python installation and all installed packages.

Activation modifies the shell's PATH to use the virtual environment's Python instead of the system Python. The command differs between operating systems and shells. On Linux and macOS with bash or zsh, use the source command with the activate script in the bin directory. On Windows, use the activate script in the Scripts directory.

After activation, the shell prompt typically shows the environment name, confirming activation succeeded. The which python command (or where python on Windows) should show the virtual environment's Python.

**Installing Dependencies**

The requirements.txt file lists all Python packages needed to run the platform. Installing these packages downloads them from PyPI and installs into the virtual environment. The process might take several minutes depending on internet speed.

Some packages require compilation. On Linux, this might require installing development headers (python3-dev on Debian/Ubuntu, python3-devel on Fedora/RHEL). On macOS, Xcode command line tools provide necessary compilers.

After installation, verify key packages are available by importing them in Python. Testing imports for sklearn, mlflow, fastapi, and airflow confirms successful installation.

**Optional Dependencies**

Additional requirements files provide dependencies for specific purposes. The requirements-security.txt file adds security scanning tools. The requirements-airflow.txt file adds Airflow-specific packages. Install these when needed for specific development tasks.

### IDE Configuration

Most Python IDEs work well with this project. VS Code and PyCharm are popular choices with good Python support.

**Interpreter Configuration**

Configure the IDE to use the virtual environment's Python interpreter. This ensures the IDE finds installed packages for code completion and linting. Point to the python executable in the venv/bin directory (or venv/Scripts on Windows).

**Linting and Formatting**

The project uses flake8 for linting and black for formatting. Configure the IDE to run these tools automatically. The .flake8 file contains project-specific linting configuration.

Black enforces consistent formatting automatically. Many IDEs can run black on save, keeping code formatted without manual effort. The pyproject.toml file contains black configuration.

---

## Running Without Docker

### Training Models

Model training runs directly with Python, without requiring Docker. This approach suits development and testing where quick iteration matters more than environment isolation.

**Basic Training**

The main.py script provides the command-line interface for training. Running with the compare flag trains both Random Forest and Gradient Boosting models, comparing their performance. This full comparison takes several minutes.

Training with specific pipeline flags trains individual models. The rf flag trains only Random Forest. The gb flag trains only Gradient Boosting. Single-model training completes faster.

**Training Output**

Training produces console output showing progress and results. Dataset loading confirms successful data retrieval. Training steps show cross-validation progress. Final metrics summarize model performance.

Trained models save to the models directory. MLflow logs experiments to the mlruns directory. Both directories should exist before training; create them if missing.

**MLflow Tracking**

MLflow tracks experiments automatically during training. View results by starting the MLflow UI server on port 5000. Open the URL in a browser to explore experiments, compare runs, and view metrics.

The MLflow UI shows all training runs with their parameters and metrics. Selecting runs enables side-by-side comparison. Artifacts include saved models and visualizations.

### Running the Prediction API

The FastAPI prediction service runs independently of Docker. This enables rapid iteration during API development.

**Prerequisites**

Trained models must exist before starting the API. Run training at least once to produce model files. The API loads models at startup and fails if none exist.

**Starting the Service**

Uvicorn serves FastAPI applications. Start the service specifying the application module and desired port. The host setting 0.0.0.0 enables access from other machines on the network.

The service logs startup information including the listening address. Access the interactive documentation at the /docs path. Test endpoints directly through the documentation interface.

**Development Mode**

Uvicorn's reload flag enables automatic restart when code changes. This speeds development by eliminating manual restart after each change. The server watches Python files and reloads when they change.

Development mode has performance overhead and shouldn't be used in production. The reload mechanism occasionally misses changes; manual restart resolves this.

### Running the Streamlit Dashboard

The monitoring dashboard runs with the Streamlit command. Specify the application file and optional port configuration.

**Starting the Dashboard**

Streamlit starts a local web server and opens a browser automatically. If the browser doesn't open, access the URL shown in console output. The default port is 8501.

The dashboard queries MLflow for experiment data. Ensure the MLflow tracking directory contains experiments; otherwise, the dashboard shows empty views.

**Development Features**

Streamlit provides hot reloading during development. Code changes trigger automatic page refresh. This enables rapid iteration on visualizations and layouts.

The Streamlit menu provides additional options. The settings menu configures themes and developer options. The rerun option manually refreshes the page.

---

## Docker Deployment

### Docker Architecture

Docker containers package applications with their dependencies, ensuring consistent behavior across environments. The platform uses multiple containers orchestrated by Docker Compose.

**Container Benefits**

Isolation prevents conflicts between service dependencies. The Airflow container's Python environment is completely separate from the prediction API's environment. Updates to one don't affect others.

Reproducibility ensures identical behavior across machines. The same Docker images run identically on development laptops and production servers. No more "works on my machine" problems.

Simplified deployment packages everything needed. Recipients don't need to install Python, configure environments, or manage dependencies. Running docker-compose up starts everything.

**Image Organization**

The project includes several Dockerfiles for different purposes. The main Dockerfile builds the prediction service image. Dockerfile.airflow builds the Airflow components. Dockerfile.cloudrun optimizes for Cloud Run deployment. Dockerfile.streamlit builds the monitoring dashboard.

Separate Dockerfiles enable targeted optimization. The Airflow image includes workflow-specific dependencies. The prediction image minimizes size for fast deployment. Each image contains exactly what its service needs.

### Docker Compose Overview

Docker Compose defines multi-container applications. The docker-compose.yml file specifies all services, their configurations, and relationships.

**Service Definition**

Each service defines a container configuration. The build context specifies which Dockerfile to use. Environment variables configure runtime behavior. Port mappings expose services for access.

**Dependencies**

Services can depend on other services. Database services start before services that need them. Health checks ensure dependencies are ready before dependent services start.

**Volumes**

Volumes persist data beyond container lifecycle. Database containers use volumes to preserve data across restarts. Shared volumes enable services to access common files.

**Networks**

Docker Compose creates a network connecting all services. Services communicate using service names as hostnames. External access requires explicit port mappings.

### Starting the Platform

**Initial Startup**

First startup builds images and creates containers. This process downloads base images, installs dependencies, and configures services. Initial startup takes several minutes.

The detached flag runs containers in the background. Without this flag, the terminal shows combined logs from all containers. Background execution frees the terminal for other work.

**Verifying Services**

After startup, verify services are running. The docker-compose ps command shows container status. All services should show as running.

Test service accessibility by accessing their endpoints. The Airflow UI should be accessible on port 8081. MLflow should respond on port 5000. The prediction API should answer on port 8000.

**Viewing Logs**

Combined logs show output from all containers. Individual service logs filter to specific containers. The follow flag streams logs continuously.

Log output helps diagnose startup problems. Error messages indicate configuration issues. Stack traces pinpoint code problems.

### Stopping and Cleanup

**Stopping Services**

The down command stops and removes containers. Networks created by compose are also removed. Volumes persist by default for data preservation.

**Removing Volumes**

Adding the volumes flag removes named volumes. This deletes persistent data including database contents. Use carefully as data loss is permanent.

**Rebuilding Images**

Code changes require rebuilding images. The build command reconstructs images from Dockerfiles. The no-cache flag forces fresh builds, useful when cached layers might be stale.

---

## Docker Compose Services

### Airflow Services

**Webserver**

The Airflow webserver provides the user interface for workflow management. Access it on port 8081 with default credentials admin/admin. The interface shows DAG status, enables manual triggering, and provides access to logs.

Configuration passes through environment variables. The executor setting determines how tasks run. The database connection specifies metadata storage. Additional settings configure authentication and features.

**Scheduler**

The scheduler monitors for work and dispatches tasks to workers. It runs continuously, checking for scheduled DAGs and triggered runs. Without the scheduler, no DAGs execute.

The scheduler shares configuration with the webserver. Both connect to the same metadata database. They must agree on DAG definitions through shared volume mounts.

**Database**

PostgreSQL stores Airflow metadata including DAG runs, task states, and configuration. The database persists through container restarts via volume mounts.

Initialization creates necessary tables on first startup. The airflow db init command runs automatically. Subsequent startups use existing database state.

### MLflow Service

MLflow tracking server stores experiment data and serves the UI. The backend store configures where metadata persists. The artifact store specifies where model files save.

File-based storage keeps everything local. This suits development and small deployments. Production deployments might use cloud storage for artifacts and databases for metadata.

The UI port exposes the web interface. Access it to view experiments, compare runs, and manage models. The API endpoints enable programmatic access for training scripts.

### Prediction Services

**FastAPI Service**

The FastAPI container runs the prediction API. It loads models from shared volumes and serves inference requests. Port 8000 exposes the API for external access.

Environment variables configure model paths and MLflow connectivity. The service queries MLflow for production model information. Model files load from shared volume mounts.

**Webhook API**

The webhook service handles external triggers. It runs on port 8080, accepting HTTP requests and initiating DAG runs. API key authentication protects against unauthorized access.

Configuration includes the API key secret and Airflow connection details. The service communicates with Airflow through its API to trigger DAGs.

### Streamlit Service

The monitoring dashboard runs in its own container. It connects to MLflow to query experiment data. Port 8501 exposes the web interface.

The container mounts the monitoring code directory. Changes to dashboard code reflect after container restart. Development mode isn't available in containerized deployment.

### Database Services

**Airflow PostgreSQL**

Dedicated PostgreSQL instance for Airflow metadata. Runs on port 5432 (the standard PostgreSQL port). Volume mount ensures data persistence.

Credentials configure through environment variables. The database name, user, and password must match Airflow configuration. Initialization happens automatically on first startup.

**MLOps PostgreSQL**

Second PostgreSQL instance for platform metadata. Runs on port 5433 to avoid conflict with Airflow's database. Separate instances provide isolation and independent management.

This database stores additional metadata beyond MLflow's tracking. Monitoring data, audit logs, and configuration might persist here.

### Redis Service

Redis provides the message broker for Airflow's Celery executor. Tasks queue in Redis for worker pickup. The lightweight service requires minimal resources.

Default configuration works for development. Production deployments might tune memory limits and persistence settings.

---

## Local Development Workflow

### Code Organization

Understanding the codebase structure accelerates development. Files organize by function into logical directories.

**Root Directory**

Configuration files live at the root level. Requirements files list dependencies. Docker files define container builds. The main.py script provides the CLI.

Core ML code also lives at the root. Training logic in train.py, evaluation in evaluate.py, and configuration in config.py. This flat organization keeps essential code easily accessible.

**Airflow Directory**

DAG definitions live in airflow/dags. API implementations live in airflow/api. Shared utilities live in airflow/utils. Docker files specific to Airflow live here too.

**Monitoring Directory**

The Streamlit dashboard code lives in monitoring. The main application file and supporting modules reside here.

**Security Directory**

Security module implementations organize into subdirectories. Poisoning detection, robustness testing, and dependency scanning each have dedicated directories.

**Tests Directory**

Test files mirror the source structure. Test modules correspond to source modules they test. Fixtures and utilities support test implementation.

### Development Iteration

**Without Docker**

For rapid iteration on specific components, run them directly without Docker. Train models with Python directly. Start the API with Uvicorn. Run the dashboard with Streamlit.

This approach provides fastest feedback. Code changes take effect immediately with hot reload. Debugging works naturally with IDE integration.

The tradeoff is inconsistency with production. Dependencies might differ. Service interactions require manual coordination. Use Docker periodically to verify integrated behavior.

**With Docker**

For testing integrated behavior, use Docker Compose. Start the full stack and test end-to-end workflows. Verify services communicate correctly.

Code changes require rebuilding images. This adds overhead compared to direct execution. Consider rebuilding only changed services to minimize iteration time.

Hybrid approaches work well. Run services under active development directly. Run stable dependencies in Docker. This balances iteration speed with integration testing.

### Making Changes

**Source Changes**

Modify source files with your editor. For non-Docker development, changes take effect immediately with hot reload or on restart.

For Docker development, rebuild the affected image. The build process copies current source into the image. Restart the container to use the new image.

**Configuration Changes**

Environment variables configure runtime behavior. For non-Docker development, export variables in the shell before starting services.

For Docker development, modify docker-compose.yml environment sections. Restart containers to apply changes. Compose down/up applies all changes cleanly.

**Dependency Changes**

Adding or updating dependencies requires updating requirements files. For non-Docker development, reinstall with pip.

For Docker development, rebuild images to incorporate new dependencies. The build process installs requirements from the updated file.

---

## Testing

### Test Organization

Tests live in the tests directory, organized by component. Test files follow the naming convention test_*.py for pytest discovery.

**Unit Tests**

Unit tests verify individual functions and classes in isolation. They don't require external services or databases. Fast execution enables running frequently during development.

The test modules correspond to source modules. test_load_data.py tests data loading. test_preprocessing.py tests feature transformation. test_train.py tests model training.

**Integration Tests**

Integration tests verify component interactions. They might require databases or external services. Longer execution suits less frequent runs.

The test_integration.py module tests end-to-end workflows. Training followed by evaluation followed by prediction exercises the complete pipeline.

**API Tests**

API tests verify endpoint behavior. They start the API service and make HTTP requests. Response validation ensures correct behavior.

test_fastapi.py covers FastAPI endpoints. test_flask_app.py covers Flask endpoints. Both verify the same functionality through different implementations.

### Running Tests

**Full Suite**

Run the complete test suite with pytest. All tests in the tests directory execute. Output shows pass/fail status for each test.

The verbose flag provides detailed output. Test names and status display individually. Failure details include assertion information.

**Selective Execution**

Run specific test files by passing the path. Run specific test functions by appending ::test_name. This speeds iteration when focusing on particular functionality.

Markers enable running test categories. Tests might be marked as slow, integration, or api. Running with marker filters executes only matching tests.

**Coverage Reporting**

Coverage analysis shows which code tests exercise. The coverage report shows percentage covered by file. HTML reports provide detailed line-by-line coverage visualization.

High coverage doesn't guarantee quality but identifies untested code. Focus coverage efforts on critical paths and complex logic.

### Writing Tests

**Test Structure**

Tests follow the arrange-act-assert pattern. Arrange sets up test conditions. Act executes the code under test. Assert verifies expected outcomes.

Fixtures provide reusable setup. pytest fixtures define common test data and configurations. Fixture scope controls how often setup runs.

**Mocking**

External dependencies require mocking for unit tests. Mock objects replace real dependencies with controllable substitutes. This isolates the code under test from external factors.

The unittest.mock module provides mocking tools. Patch decorators replace objects temporarily. Mock objects track calls and return configured values.

**Test Data**

Tests need input data. Small datasets suffice for unit tests. Fixtures can generate synthetic data or load samples.

Avoid depending on external data sources in tests. Tests should run without network access. Include necessary data in the repository or generate programmatically.

---

## Code Quality

### Formatting with Black

Black enforces consistent code formatting. It makes style decisions automatically, eliminating debates. Running black reformats code to match its style.

**Running Black**

Execute black on the project directory. Files reformat in place. The check flag verifies formatting without changing files, useful in CI.

Exclude virtual environment and generated directories. The --exclude flag accepts patterns. Common exclusions include venv, mlruns, and __pycache__.

**Integration**

Configure editors to run black on save. This maintains formatting without manual intervention. Black's deterministic output means everyone's code looks identical.

Pre-commit hooks can enforce formatting. Commits with improperly formatted code reject automatically. This prevents formatting violations from entering the repository.

### Linting with Flake8

Flake8 identifies potential issues through static analysis. It catches syntax errors, undefined variables, unused imports, and style violations.

**Running Flake8**

Execute flake8 on the project directory. Issues display with file, line, and description. Exit code indicates whether issues were found.

The .flake8 configuration file customizes behavior. Line length settings, ignored rules, and excluded directories configure here. Project conventions might differ from defaults.

**Common Issues**

Unused imports indicate cleanup opportunities. Remove imports that aren't used. Alternatively, mark intentional unused imports with noqa comments.

Line length violations require reformatting. Black usually handles this automatically. Complex expressions might need manual restructuring.

Undefined variables indicate bugs. These should always be fixed. Missing imports or typos in variable names are common causes.

### Type Checking

Type hints document expected types and enable static analysis. While not enforced at runtime, they catch errors before execution.

**Adding Type Hints**

Function signatures should include type hints. Parameter types document expected inputs. Return types document outputs. This serves as executable documentation.

Complex types use typing module constructs. List, Dict, Optional, and Union express compound types. Type aliases simplify repeated complex types.

**Running mypy**

mypy performs static type checking. It verifies that code respects declared types. Type errors indicate potential bugs or incorrect hints.

Strict mode enables comprehensive checking. Some legacy code might not pass strict checks initially. Incremental adoption adds hints over time.

---

## Troubleshooting

### Common Development Issues

**Import Errors**

Import errors indicate missing packages or incorrect paths. Verify the virtual environment is activated. Check that dependencies are installed.

Circular imports cause confusing errors. Reorganize code to break circular dependencies. Import within functions if necessary to defer resolution.

**Model Not Found**

Model loading errors occur when model files are missing. Run training to produce model files. Verify model paths in configuration match actual locations.

For Docker, verify volume mounts expose model directories. Container paths must match what the application expects.

**Port Conflicts**

Port already in use errors indicate another process uses the desired port. Find the conflicting process with lsof or netstat. Stop the conflicting process or use a different port.

Multiple Docker Compose runs might conflict. Ensure previous deployments are stopped before starting new ones.

### Docker Issues

**Build Failures**

Build failures indicate Dockerfile or dependency problems. Read error messages carefully. Package installation failures might indicate network issues or incompatible packages.

Cache invalidation can help. The --no-cache flag forces fresh builds. Stale caches sometimes cause confusing failures.

**Container Startup Failures**

Containers that exit immediately indicate startup problems. Check logs for error messages. Common causes include missing configuration or dependency failures.

Health check failures prevent dependent services from starting. Verify database connections and required services are available.

**Volume Permission Issues**

Permission denied errors often involve Docker volumes. On Linux, UID/GID mismatches cause issues. Run containers as the appropriate user or adjust volume permissions.

**Networking Issues**

Services unable to connect indicate networking problems. Verify services are on the same Docker network. Use service names, not localhost, for inter-container communication.

### Performance Issues

**Slow Training**

Training taking longer than expected might indicate resource constraints. Monitor CPU and memory usage. Consider reducing data size for development.

**Slow API Responses**

High latency suggests model loading problems or resource constraints. Verify models load at startup, not per-request. Monitor memory usage.

**Docker Resource Limits**

Default Docker resource limits might be too restrictive. Configure memory and CPU limits in docker-compose.yml. Increase Docker's overall resource allocation in Docker settings.

---

## Version Information

- Version: 1.0.0
- Last Updated: December 2025
- Python Version: 3.8+
- Docker Version: 20.10+
- Docker Compose Version: 2.0+
