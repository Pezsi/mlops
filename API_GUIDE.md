# REST API Services Guide

Technical documentation for the REST API services in the Wine Quality MLOps platform. This guide covers the FastAPI prediction service, Flask alternative implementation, and webhook trigger API.

---

## Table of Contents

1. [API Architecture Overview](#api-architecture-overview)
2. [FastAPI Prediction Service](#fastapi-prediction-service)
3. [Flask Alternative Implementation](#flask-alternative-implementation)
4. [Webhook Trigger API](#webhook-trigger-api)
5. [Authentication and Security](#authentication-and-security)
6. [Error Handling](#error-handling)
7. [Performance Considerations](#performance-considerations)
8. [Integration Patterns](#integration-patterns)

---

## API Architecture Overview

### Role of APIs in the Platform

APIs serve as the interface between the machine learning models and external consumers. Without APIs, models would only be accessible through direct Python code. APIs enable web applications, mobile apps, IoT devices, and other systems to obtain predictions without understanding the underlying ML implementation.

The platform provides multiple API implementations serving different purposes. The primary prediction API handles model inference requests. The webhook API enables external systems to trigger workflows. Alternative implementations demonstrate different framework approaches.

### Design Philosophy

The APIs follow REST conventions, providing predictable endpoint structures and standard HTTP semantics. Resources have consistent URLs. Operations use appropriate HTTP methods (GET for retrieval, POST for creation). Status codes indicate success or failure categories.

JSON serves as the primary data format. Request bodies contain JSON payloads. Responses return JSON structures. This format works well across programming languages and platforms.

OpenAPI documentation generates automatically from code annotations. Developers can explore available endpoints, understand request formats, and test functionality through interactive documentation. This self-documenting approach reduces integration friction.

### Service Topology

The prediction API runs as a standalone service, typically on port 8000. It loads models from MLflow or local storage and serves predictions independently of other platform components. This isolation enables scaling the prediction service separately from training and orchestration components.

The webhook API runs on port 8080, handling external triggers for Airflow workflows. It authenticates requests, validates parameters, and communicates with Airflow to initiate DAG runs. Separation from the prediction API reflects its different purpose and security requirements.

---

## FastAPI Prediction Service

### Why FastAPI

FastAPI provides several advantages for ML serving workloads. Native async support enables handling concurrent requests efficiently. Automatic validation catches malformed requests before they reach application code. Built-in OpenAPI documentation eliminates separate documentation maintenance.

Type hints drive both validation and documentation. Defining a request model with typed fields automatically validates incoming data and generates documentation showing expected formats. This single-source-of-truth approach prevents documentation drift.

Performance matches or exceeds alternatives. FastAPI builds on Starlette for networking and Pydantic for validation, both highly optimized libraries. The async architecture handles many concurrent requests with minimal overhead.

### Service Architecture

The FastAPI application organizes endpoints into logical groups. General endpoints provide health checks and metadata. Model endpoints expose model information. Prediction endpoints handle inference requests. Training endpoints enable model updates.

Model loading occurs at startup. The service retrieves the current production model from MLflow or loads from local storage. Keeping the model in memory enables fast inference without per-request loading overhead.

Request processing validates inputs, transforms features, generates predictions, and formats responses. Each step is isolated for clarity and testability. Errors at any step return appropriate error responses rather than crashing.

### Endpoint Categories

**General Endpoints**

The root endpoint returns a welcome message confirming the API is operational. While simple, this endpoint verifies basic connectivity and routing.

The health endpoint provides detailed service status. It checks model availability, dependency connectivity, and resource status. Load balancers and orchestrators use this endpoint to determine service health.

The features endpoint returns the list of expected input features. Client applications can query this endpoint to build input forms dynamically, ensuring they collect all required values.

**Model Endpoints**

The models list endpoint returns available model names and versions. When multiple models exist, clients can discover options rather than hardcoding names.

The metrics endpoint returns performance information for specified models. Applications might display model quality alongside predictions, helping users understand prediction reliability.

**Prediction Endpoints**

The primary predict endpoint accepts feature values and returns quality predictions. The request body contains feature names and values. The response contains the predicted quality score and metadata.

The batch predict endpoint handles multiple samples in a single request. Batch processing amortizes per-request overhead across many predictions, improving throughput for bulk operations.

The model-specific predict endpoint allows targeting specific model versions. By default, predictions use the production model. Explicitly specifying a model enables A/B testing, gradual rollouts, or accessing experimental models.

**Training Endpoints**

The train endpoint triggers model training. While most training runs through Airflow, the API endpoint enables direct triggering for testing or emergency updates. Training requests can specify hyperparameters and other configuration.

### Request and Response Formats

**Prediction Request**

Prediction requests contain feature values for the wine sample. All eleven physicochemical properties are required: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol content.

Optional parameters include model name for targeting specific models. When omitted, the service uses the current production model.

**Prediction Response**

Responses contain the predicted quality score as a floating-point number. The prediction typically falls between 3 and 8, matching the range observed in training data. Extreme predictions outside this range might indicate unusual inputs.

Metadata accompanies the prediction: the model name and version used, prediction timestamp, and processing duration. This information supports debugging and audit requirements.

**Batch Request**

Batch requests contain an array of feature sets. Each element follows the same format as single predictions. The array can contain any number of samples, though very large batches might timeout.

**Batch Response**

Batch responses contain an array of predictions in the same order as inputs. Each prediction includes the quality score and metadata. Failed predictions within a batch return error information rather than failing the entire batch.

### Interactive Documentation

Accessing the /docs endpoint opens Swagger UI, an interactive documentation interface. The interface lists all endpoints with descriptions, shows expected request formats, and enables testing directly in the browser.

The /redoc endpoint provides an alternative documentation style. ReDoc presents similar information with a different visual design that some developers prefer.

Both documentation interfaces generate automatically from code. Updating endpoint definitions automatically updates documentation. This eliminates documentation maintenance burden and prevents documentation from becoming stale.

The implementation resides in `airflow/api/fastapi_app.py` and the root-level `fastapi_app.py`.

---

## Flask Alternative Implementation

### Implementation Purpose

The Flask implementation provides an alternative to FastAPI, demonstrating that the same functionality can be achieved with different frameworks. Organizations with existing Flask expertise might prefer this implementation. The comparison also illustrates different architectural approaches.

Flask follows a more traditional synchronous model compared to FastAPI's async-first approach. For CPU-bound ML inference, the difference matters less than for I/O-bound workloads. Both implementations achieve similar performance for prediction requests.

### Flask-RESTX Integration

Flask-RESTX extends Flask with API-focused features. Namespaces organize related endpoints. Models define request and response schemas. Swagger documentation generates automatically. These features bring Flask closer to FastAPI's developer experience.

Namespaces group endpoints by function. The prediction namespace contains inference endpoints. The model namespace contains model management endpoints. This organization mirrors the FastAPI implementation's router structure.

Request parsing with Flask-RESTX validates incoming data against defined expectations. Required fields, type constraints, and value ranges all validate automatically. Invalid requests return informative error messages.

### Endpoint Comparison

Both implementations provide equivalent endpoints with identical functionality. The root endpoint, health check, feature list, model list, and prediction endpoints all exist in both versions. Request and response formats match exactly.

The implementation approaches differ. FastAPI uses Pydantic models for validation. Flask-RESTX uses request parsers and API models. The underlying validation logic is similar, but the syntactic expression differs.

Documentation in Flask-RESTX requires more explicit configuration than FastAPI's automatic inference from type hints. Both produce comprehensive Swagger documentation, but FastAPI requires less boilerplate.

### When to Choose Each

Choose FastAPI for new projects, async requirements, or teams comfortable with modern Python features. FastAPI's type-driven approach reduces boilerplate and catches errors earlier.

Choose Flask for teams with Flask experience, projects integrating with existing Flask applications, or environments where async complexity isn't justified. Flask's maturity provides extensive ecosystem support.

Both frameworks produce equivalent results for ML serving. The choice often comes down to team preference and organizational standards rather than technical requirements.

The implementation resides in `flask_app.py`.

---

## Webhook Trigger API

### Purpose and Architecture

The webhook API enables external systems to trigger platform workflows. Rather than requiring direct Airflow access, external systems call HTTP endpoints to initiate DAG runs. This decoupling simplifies integration and improves security.

The API authenticates requests using API keys. Valid keys must accompany every request. This prevents unauthorized triggering while remaining simple to implement across diverse client systems.

Built on Flask, the webhook API keeps dependencies minimal. The service only needs to validate requests and communicate with Airflow, not perform ML inference. This simplicity improves reliability and reduces resource requirements.

### Trigger Endpoint

The primary endpoint accepts training trigger requests. The request body specifies which DAG to trigger and provides configuration parameters. The API validates the request, checks authentication, and initiates the DAG run.

**Request Contents**

Model name identifies which model configuration to train. The platform might support multiple model types with different architectures or hyperparameters.

Trigger source documents what system initiated the request. This metadata helps with debugging and audit trails. Common sources include CI/CD pipelines, data platforms, and manual operations.

Force training overrides normal checks that might skip training. When data hasn't changed significantly, training might otherwise be skipped. Force training ensures a new model is produced regardless.

Hyperparameters optionally override default training configuration. Callers can specify learning rate, tree depth, or other parameters. This enables experimentation without modifying training code.

**Response Contents**

Successful triggers return the Airflow DAG run ID. Callers use this ID to track execution progress. The response confirms which DAG was triggered and what parameters were applied.

Failed triggers return error information. Authentication failures indicate invalid API keys. Validation failures detail which parameters were problematic. Airflow communication failures indicate infrastructure issues.

### Integration Patterns

**CI/CD Integration**

Continuous integration pipelines trigger validation after code changes. When training code updates, the pipeline calls the webhook to verify models still train correctly. Failed training blocks deployment of broken code.

The CI/CD system stores the API key as a secure variable. Pipeline scripts construct requests with appropriate parameters. Pipeline stages wait for training completion before proceeding.

**Data Platform Integration**

Data pipelines trigger training after ETL completion. When fresh data loads into the data warehouse, the platform notifies the webhook. Training incorporates the new data automatically.

Scheduling coordination ensures training doesn't start before data is ready. The data platform only calls the webhook after confirming data availability. Retry logic handles transient failures.

**Operational Integration**

Operations dashboards provide manual trigger capabilities. Operators can initiate training without accessing Airflow directly. The dashboard constructs appropriate requests based on operator selections.

Incident response procedures might include triggering retraining. If model degradation is detected, operators can quickly initiate training to potentially resolve the issue.

The implementation resides in `airflow/api/webhook_trigger_api.py`.

---

## Authentication and Security

### API Key Authentication

The webhook API requires API key authentication on all requests. Keys pass through HTTP headers rather than URL parameters, avoiding exposure in logs and browser history.

Key validation compares submitted keys against configured valid keys. The comparison uses constant-time algorithms to prevent timing attacks. Invalid keys return generic error messages without revealing whether the key format was correct.

Key management includes rotation procedures. Periodic rotation limits exposure from compromised keys. During rotation, both old and new keys work temporarily, enabling gradual client migration.

### Prediction API Security

The prediction API in this implementation allows unauthenticated access. This suits internal deployments where network security provides the primary protection. Production deployments might add authentication layers.

Authentication options include API keys similar to the webhook API, OAuth2 for user-specific access, or mutual TLS for service-to-service communication. The choice depends on deployment context and security requirements.

Rate limiting prevents abuse even without authentication. Limits on requests per IP address or time window prevent denial-of-service attacks and ensure fair resource allocation.

### Input Validation

All APIs validate inputs before processing. Type validation ensures values match expected types. Range validation catches obviously incorrect values. Schema validation confirms all required fields are present.

Validation failures return informative error messages. Messages identify which field failed and why. This information helps clients fix requests without guessing.

Validation prevents injection attacks. Sanitizing inputs before use in queries or commands blocks common attack vectors. The APIs don't construct queries from user input, but validation provides defense in depth.

### Secure Configuration

Sensitive configuration uses environment variables rather than code or configuration files. API keys, database credentials, and other secrets never appear in source control.

Production deployments integrate with secret management systems. Google Secret Manager, HashiCorp Vault, or similar systems provide secure storage with audit logging and access controls.

HTTPS encrypts communication in production. Self-signed certificates might suffice for internal deployments. Public deployments should use certificates from recognized authorities.

---

## Error Handling

### Error Response Format

All APIs return errors in consistent JSON format. The structure includes an error code for programmatic handling, a human-readable message, and additional details when relevant.

HTTP status codes indicate error categories. 400-series codes indicate client errors (bad request, unauthorized, not found). 500-series codes indicate server errors. Clients can handle different categories appropriately.

### Client Errors

**400 Bad Request**: The request format is invalid. Missing required fields, incorrect types, or malformed JSON trigger this response. The message details what's wrong.

**401 Unauthorized**: Authentication failed. For the webhook API, this means the API key was missing or invalid. Clients should verify their credentials.

**404 Not Found**: The requested resource doesn't exist. Requesting predictions for a non-existent model returns this error. Clients should verify resource identifiers.

**422 Unprocessable Entity**: The request format is valid but content is problematic. Feature values outside acceptable ranges might trigger this. The message indicates which values were problematic.

### Server Errors

**500 Internal Server Error**: Something unexpected went wrong. These errors indicate bugs or infrastructure issues. Server logs contain details for debugging.

**503 Service Unavailable**: The service is temporarily unable to handle requests. Model loading failures or dependency unavailability might cause this. Clients should retry after a delay.

### Error Handling Best Practices

Clients should handle errors gracefully. Displaying raw error messages to end users is poor experience. Translate technical errors into user-friendly messages while logging details for debugging.

Retry logic handles transient failures. Server errors and some client errors (like rate limiting) might succeed on retry. Implement exponential backoff to avoid overwhelming struggling services.

Circuit breakers prevent cascade failures. If an API consistently fails, clients should temporarily stop calling it rather than adding load. After a cooling period, cautiously resume requests.

---

## Performance Considerations

### Model Loading Strategy

Models load at service startup rather than per-request. Loading involves deserializing model parameters, which takes seconds. Keeping the model in memory after loading enables millisecond-level inference.

Model reloading handles updates. When a new model deploys, the service must load it without dropping requests. Graceful reloading loads the new model alongside the old one, then switches traffic.

Memory management matters for large models. The wine quality model is small, but larger models might consume gigabytes. Monitor memory usage and allocate appropriately.

### Request Processing Optimization

Feature validation happens early. Catching invalid requests before expensive processing saves resources. Pydantic validation in FastAPI is highly optimized.

Prediction computation is CPU-bound. Scikit-learn models use NumPy for efficient numerical computation. The wine quality model predicts in milliseconds.

Response serialization adds minimal overhead. JSON encoding is fast for small response bodies. Larger responses might benefit from compression.

### Concurrency Handling

FastAPI handles concurrent requests through async processing. The event loop manages multiple requests simultaneously, switching between them during I/O waits. CPU-bound prediction blocks the event loop briefly but not long enough to matter.

Flask uses a synchronous model where each request occupies a worker thread. Thread pools enable concurrent handling up to the pool size. For CPU-bound ML inference, this model works well.

Scaling horizontally handles load beyond single-service capacity. Load balancers distribute requests across multiple service instances. Each instance maintains its own model copy.

### Batch Processing Benefits

Batch endpoints process multiple predictions efficiently. Setup costs (request parsing, model preparation) amortize across all predictions. NumPy operations vectorize well across batch dimensions.

Batch size tradeoffs exist. Very small batches don't benefit much from batching. Very large batches increase latency for all predictions. Medium batches balance throughput and latency.

Client applications can batch naturally. If an application needs multiple predictions simultaneously, batching them reduces total latency compared to sequential requests.

---

## Integration Patterns

### Synchronous Integration

Simple integrations call the API and wait for responses. The caller blocks until the prediction returns. This pattern suits interactive applications where users wait for results.

Timeout handling prevents indefinite waits. Set reasonable timeouts based on expected response times. Handle timeout errors gracefully, perhaps offering retry options.

Error handling displays appropriate messages. Technical errors shouldn't leak to users. Parse error responses and display user-friendly equivalents.

### Asynchronous Integration

Background processing enables non-blocking integration. The caller submits requests and continues other work. Responses arrive later through callbacks, polling, or message queues.

Job queues manage prediction workloads. Submit prediction requests to a queue. Worker processes call the API and store results. Callers retrieve results when needed.

Event-driven architectures integrate through messaging. Prediction requests publish to a topic. The prediction service subscribes, processes, and publishes results. This decoupling improves resilience.

### Batch Integration

High-volume scenarios process predictions in batches. Collect multiple samples, submit as a batch request, and distribute results. This maximizes throughput when latency per individual prediction isn't critical.

Scheduled batch processing handles periodic workloads. Nightly jobs might predict quality for all new wines received that day. Batch endpoints handle these volumes efficiently.

Streaming integration processes continuous data flows. As samples arrive, buffer them into batches. Submit batches at regular intervals or when buffers reach threshold sizes.

### Monitoring Integration

Logging predictions supports debugging and analysis. Log input features, predictions, and request metadata. Structured logging enables querying and aggregation.

Metrics track prediction service health. Monitor request rates, latency distributions, and error rates. Alerts trigger when metrics exceed thresholds.

Prediction tracking enables model monitoring. Store predictions with timestamps and eventually actual outcomes. Compare predictions against outcomes to detect model degradation.

---

## Version Information

- Version: 1.0.0
- Last Updated: December 2025
- FastAPI Version: 0.100+
- Flask Version: 2.0+
