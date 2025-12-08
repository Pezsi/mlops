# GCP Cloud Run Deployment Guide

Technical documentation for deploying the Wine Quality MLOps platform to Google Cloud Platform using Cloud Run serverless infrastructure with automated CI/CD through GitHub Actions.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Why Cloud Run](#why-cloud-run)
3. [Deployment Process](#deployment-process)
4. [Configuration and Scaling](#configuration-and-scaling)
5. [Monitoring and Operations](#monitoring-and-operations)
6. [Cost Management](#cost-management)
7. [Security Considerations](#security-considerations)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Topics](#advanced-topics)

---

## Architecture Overview

The deployment architecture positions GCP Cloud Run as the production serving layer while GitHub Actions handles continuous integration and deployment. This combination provides automated deployments triggered by code changes, with Cloud Run managing the operational complexity of running containerized workloads.

### The Deployment Pipeline

Code changes flow through a structured pipeline from development to production. When developers push changes to the main branch, GitHub Actions automatically triggers. The workflow builds a Docker image containing the application code and dependencies, runs tests to verify functionality, and pushes the validated image to Google Artifact Registry. Finally, the workflow deploys the new image to Cloud Run, which handles the traffic transition from old to new versions.

This automated pipeline eliminates manual deployment steps that introduce human error. Every deployment follows the same process, ensuring consistency between environments. Failed deployments automatically roll back, preventing broken code from reaching users.

### Component Relationships

**GitHub Repository**: The source of truth for application code. Changes here trigger the deployment pipeline. The repository contains application source code, Dockerfile definitions, GitHub Actions workflow configurations, and deployment scripts.

**GitHub Actions**: The automation engine coordinating the deployment process. Workflows define the sequence of build, test, and deploy steps. Actions authenticate with GCP using service account credentials stored as repository secrets.

**Artifact Registry**: Google's managed container registry storing Docker images. Each successful build pushes a new image tagged with the git commit hash. The registry maintains image history, enabling rollback to any previous version if problems emerge after deployment.

**Cloud Run**: The execution environment for production workloads. Cloud Run pulls images from Artifact Registry and runs them in managed containers. The platform handles load balancing, SSL termination, automatic scaling, and health monitoring without requiring infrastructure management.

### Traffic Flow

User requests arrive at a Cloud Run endpoint URL, which provides HTTPS with automatic certificate management. Cloud Run routes requests to healthy container instances, starting new instances if needed to handle load. Containers process requests using the FastAPI application, which loads the ML model and returns predictions. Responses flow back through Cloud Run's load balancer to users.

The endpoint URL remains stable across deployments. Cloud Run manages traffic transitions between old and new container versions, ensuring zero downtime during updates.

---

## Why Cloud Run

### Serverless Benefits

Cloud Run abstracts away infrastructure management. Traditional deployments require provisioning servers, configuring networking, managing operating system updates, and scaling capacity. Cloud Run handles all of this automatically, letting teams focus on application development rather than operations.

The serverless model charges only for actual resource consumption. Containers that receive no traffic cost nothing. This differs fundamentally from traditional hosting where servers cost money whether they handle requests or sit idle. For ML inference workloads with variable traffic patterns, this model can significantly reduce costs.

### Scaling Characteristics

Cloud Run scales automatically based on incoming request volume. When traffic increases, the platform starts additional container instances to handle load. When traffic decreases, instances terminate to reduce costs. This elasticity matches capacity to demand without manual intervention.

The scale-to-zero capability particularly benefits development and staging environments. These environments often sit idle most of the time, yet traditional hosting charges continuously. Cloud Run's ability to scale to zero means idle environments cost nothing, enabling teams to maintain multiple environments without budget concerns.

### Container-Based Deployment

Cloud Run runs standard Docker containers, providing deployment flexibility. The same container image that runs locally during development runs identically in production. This consistency eliminates the "works on my machine" problems that plague traditional deployments.

Container images capture the complete application environment: code, dependencies, configuration, and runtime. Teams can reproduce any deployment by running the corresponding image. This reproducibility simplifies debugging production issues and enables confident rollbacks.

### ML Inference Suitability

ML inference workloads match Cloud Run's strengths well. Prediction requests arrive unpredictably, sometimes in bursts. Models load slowly but serve predictions quickly once loaded. Traffic often varies significantly by time of day.

Cloud Run handles these patterns efficiently. Auto-scaling manages traffic bursts without pre-provisioned capacity. Warm instances keep models loaded for fast responses. Scale-to-zero eliminates costs during quiet periods while allowing quick scale-up when traffic returns.

---

## Deployment Process

### Prerequisites

Successful deployment requires several prerequisites. A Google Cloud Platform account with billing enabled provides access to GCP services. A GCP project serves as the organizational container for resources. The Google Cloud SDK installed locally enables command-line interaction with GCP services.

Service account credentials authorize GitHub Actions to deploy resources. The service account needs specific IAM roles: Cloud Run Admin for deploying services, Artifact Registry Writer for pushing images, and Service Account User for acting as the runtime service account.

### Initial Setup

First-time deployment requires one-time setup steps. These steps create the necessary GCP resources and configure GitHub to authenticate with GCP.

Artifact Registry repository creation establishes storage for Docker images. The repository lives in a specific GCP region, chosen to minimize latency for the target user base. The Europe West 1 region suits European users while US Central suits North American deployments.

Service account creation establishes the identity used for deployments. The service account receives only the permissions needed for deployment, following the principle of least privilege. A JSON key file generated for this account enables authentication from GitHub Actions.

GitHub Secrets configuration stores the GCP credentials securely. The project ID and service account key become repository secrets, accessible to workflows but hidden from repository viewers. This separation keeps credentials out of source code where they might accidentally leak.

The setup script at `gcp/setup-gcp.sh` automates these steps. Running the script with appropriate environment variables creates all necessary resources and outputs instructions for configuring GitHub secrets.

### Automated Deployment Flow

After initial setup, deployments proceed automatically. Pushing code to the main branch triggers the GitHub Actions workflow. The workflow authenticates with GCP using the stored service account credentials.

The build stage constructs a Docker image from the Dockerfile.cloudrun definition. This specialized Dockerfile optimizes for Cloud Run's environment, including appropriate base images and runtime configurations. The build incorporates the current code and dependencies into a self-contained image.

Testing stages verify the image works correctly before deployment. Unit tests validate individual components. Integration tests confirm services interact properly. Security scans check for vulnerabilities in dependencies.

After tests pass, the workflow pushes the image to Artifact Registry with a unique tag derived from the git commit hash. This tagging scheme enables tracing any deployed image back to its source code.

The deployment stage updates the Cloud Run service to use the new image. Cloud Run handles the traffic transition, gradually shifting requests from old to new instances. If the new version fails health checks, Cloud Run automatically rolls back to the previous working version.

### Manual Deployment Option

While automated deployment handles most cases, manual deployment provides flexibility for special situations. Debugging deployment issues, testing configuration changes, or deploying from branches other than main might require manual steps.

Manual deployment follows the same logical steps as automated deployment, executed through gcloud CLI commands rather than GitHub Actions. The deploy script at `gcp/deploy.sh` provides a convenient wrapper around these commands.

The manual process builds the image locally, pushes to Artifact Registry, and deploys to Cloud Run. Local builds enable quick iteration when debugging Dockerfile issues. Direct CLI access provides visibility into each step that automated workflows abstract away.

---

## Configuration and Scaling

### Resource Allocation

Cloud Run allows configuring CPU and memory allocation per container instance. Appropriate resource allocation balances performance against cost. Under-allocation causes slow responses or out-of-memory errors. Over-allocation wastes money on unused capacity.

ML inference workloads typically need more memory than CPU. Models load entirely into memory, consuming space proportional to model complexity. Inference computations use CPU briefly for each prediction. The wine quality model requires approximately 1-2 GB of memory for comfortable operation with room for request handling overhead.

CPU allocation affects prediction latency and throughput. More CPU cores enable faster individual predictions and better handling of concurrent requests. However, scikit-learn models used in this project don't parallelize prediction across cores, so additional CPU primarily helps with concurrent request handling rather than individual prediction speed.

Recommended starting configuration allocates 2 GB of memory and 2 CPU cores. This provides adequate resources for the model while leaving headroom for traffic spikes. Teams should monitor actual usage and adjust based on observed patterns.

### Scaling Configuration

Scaling configuration controls how Cloud Run responds to traffic changes. The minimum instance count determines how many instances run even without traffic. The maximum instance count caps scaling to control costs during traffic spikes.

Setting minimum instances to zero enables scale-to-zero, eliminating costs during idle periods. However, scaling from zero introduces cold start latency as Cloud Run starts a new instance and the container initializes. For the wine quality model, cold starts take approximately 10-15 seconds as the container starts and the model loads.

Setting minimum instances to one or more eliminates cold start latency by keeping instances warm. Warm instances respond immediately since the model is already loaded. This configuration suits production environments where response latency matters, accepting the cost of running instances continuously.

Maximum instances control cost exposure during traffic spikes. Without a maximum, traffic spikes could start many instances, potentially generating unexpected bills. Setting an appropriate maximum balances handling legitimate traffic against protecting against runaway costs or denial-of-service attacks.

### Environment Configuration

Environment variables configure application behavior without code changes. Cloud Run injects configured variables into containers at startup. This separation enables the same container image to run differently across environments.

The MLflow tracking URI variable specifies where experiment tracking data persists. In Cloud Run's ephemeral environment, this typically points to a file path within the container since external tracking servers require additional setup.

Log level configuration controls output verbosity. Development environments might use DEBUG for detailed troubleshooting. Production environments typically use INFO or WARNING to reduce log volume while capturing important events.

Custom application settings follow the same pattern. Database connection strings, API keys for external services, and feature flags all configure through environment variables. This approach keeps sensitive configuration out of source code and enables environment-specific behavior.

### Request Handling

Timeout configuration determines how long Cloud Run waits for requests to complete. Prediction requests typically complete quickly, but training triggers or batch predictions might take longer. The timeout should accommodate the longest expected request while preventing stuck requests from consuming resources indefinitely.

Concurrency settings control how many simultaneous requests each instance handles. Higher concurrency improves efficiency by handling multiple requests per instance. However, ML inference consumes significant memory per request, so excessive concurrency can cause memory exhaustion.

The recommended configuration starts with concurrency of 80, which Cloud Run's default. Monitoring should track memory usage under load, reducing concurrency if memory pressure causes problems.

---

## Monitoring and Operations

### Understanding Cloud Run Metrics

Cloud Run automatically collects metrics about service operation. These metrics appear in Google Cloud Console and can trigger alerts when values exceed thresholds.

Request count tracks total requests over time. This metric reveals traffic patterns: daily cycles, weekly trends, and growth over time. Sudden changes might indicate problems or unexpected usage.

Request latency measures response time across percentiles. The p50 latency shows typical user experience. The p95 and p99 latencies reveal tail behavior affecting a minority of requests. ML inference latency should remain relatively stable; increasing latency might indicate model degradation or resource constraints.

Container CPU utilization shows compute resource consumption. Consistently high CPU suggests more resources might improve performance. Consistently low CPU indicates over-provisioning that wastes money.

Container memory utilization tracks memory consumption. Memory should remain comfortably below allocation. Approaching the limit risks out-of-memory crashes. The wine quality model's memory usage should remain stable since the model size doesn't change.

Instance count reveals scaling behavior. Frequent scaling up and down might indicate traffic instability. Staying at maximum instances suggests the limit is too low for actual demand.

### Log Analysis

Cloud Logging aggregates logs from all container instances. Application logs, request logs, and system logs all flow into a searchable, filterable interface.

Request logs capture every incoming request with timing information. These logs enable analyzing traffic patterns, identifying slow requests, and debugging specific user issues.

Application logs contain output from the FastAPI application. Prediction requests log their inputs and outputs for monitoring and debugging. Errors log with stack traces for investigation.

System logs record container lifecycle events: starts, stops, and health check results. These logs help diagnose startup problems and understand scaling behavior.

Log-based metrics transform log patterns into numeric metrics. For example, counting error log entries creates an error rate metric. These derived metrics can trigger alerts when error rates spike.

### Alerting Configuration

Alerts notify teams when metrics indicate problems. Effective alerting balances sensitivity against alert fatigue. Too many alerts train teams to ignore them. Too few alerts miss genuine problems.

Latency alerts trigger when response times exceed acceptable thresholds. A p99 latency threshold of 2 seconds catches severe slowdowns while ignoring occasional outliers.

Error rate alerts trigger when failures exceed normal baselines. A sudden spike in 5xx errors indicates something is wrong and needs investigation.

Instance count alerts trigger when scaling approaches limits. Running at maximum instances for extended periods suggests the limit needs increasing.

Memory utilization alerts trigger when approaching allocation limits. Catching memory pressure before crashes enables proactive remediation.

### Operational Procedures

Regular operational tasks maintain healthy service operation. The monitoring script at `gcp/monitor.sh` provides convenient access to common operations.

Health checks verify the service responds correctly. The /health endpoint returns service status, enabling quick verification that deployment succeeded.

Log review identifies emerging problems before they become critical. Regular review catches unusual patterns that automated alerts might miss.

Revision management tracks deployment history. Cloud Run maintains multiple revisions, enabling quick rollback if a deployment causes problems. Listing revisions shows deployment history with traffic allocation.

Traffic management controls which revision receives requests. During incidents, shifting traffic to a previous revision provides immediate remediation while investigation continues.

---

## Cost Management

### Understanding Cloud Run Pricing

Cloud Run pricing has three components: CPU allocation, memory allocation, and request count. Understanding these components enables cost optimization.

CPU charges accumulate whenever containers run, measured in vCPU-seconds. Running one container with 2 vCPUs for one hour costs the same as running two containers with 1 vCPU for one hour. Charges apply only during request processing unless "always allocated" mode is enabled.

Memory charges follow the same pattern as CPU, measured in GB-seconds. Larger memory allocations cost proportionally more. Memory charges apply whenever containers run, regardless of whether they're processing requests.

Request charges add a small per-request fee. For typical ML inference workloads, request charges represent a small fraction of total cost compared to CPU and memory.

### Free Tier Benefits

Google provides a generous free tier for Cloud Run. Each month, the first 2 million requests, 180,000 vCPU-seconds, and 360,000 GB-seconds are free. This free allocation covers significant usage for development and light production workloads.

For the wine quality service with typical configuration (2 vCPUs, 2 GB memory), the free tier covers approximately 25 hours of continuous operation or many more hours with scale-to-zero during idle periods.

### Cost Optimization Strategies

Scale-to-zero provides the most significant cost savings for variable workloads. Instances that aren't running don't cost anything. Development and staging environments benefit most since they sit idle most of the time.

Right-sizing resources eliminates waste from over-provisioning. Start with conservative allocations and increase only if monitoring reveals constraints. Memory and CPU that aren't used still cost money.

Minimum instances represent a cost versus latency tradeoff. Zero minimum instances eliminate idle costs but introduce cold start latency. Teams should evaluate whether latency requirements justify the cost of warm instances.

Request timeout limits prevent runaway costs from stuck requests. A request that hangs indefinitely consumes resources until timeout. Appropriate timeouts release resources from problematic requests.

Concurrency optimization serves more requests per instance. Higher concurrency means fewer instances needed for the same traffic, reducing costs. However, concurrency is limited by memory and CPU availability.

### Cost Estimation

For planning purposes, estimating costs requires traffic projections and configuration decisions. Consider a production workload receiving 10,000 predictions per day with an average response time of 100ms, running with 2 vCPU and 2 GB memory with minimum 1 instance for low latency.

The warm instance runs continuously: 30 days of 24 hours equals 720 hours. At 2 vCPU, this equals 1,440 vCPU-hours. At approximately $0.00002400 per vCPU-second (after free tier), monthly CPU cost runs approximately $100-120.

Memory costs follow similar calculations, typically resulting in comparable amounts. Request charges for 300,000 monthly requests (10,000 daily times 30 days) add minimal cost.

Total monthly cost for this configuration typically ranges $150-250, varying by region and actual usage patterns. Development environments using scale-to-zero cost dramatically less, often staying within free tier limits.

---

## Security Considerations

### Authentication Approaches

Cloud Run supports multiple authentication approaches for different use cases. Public endpoints allow anyone to access the service. Authenticated endpoints require valid credentials.

Public endpoints suit prediction APIs where authentication happens at an application level. The wine quality API validates API keys in application code rather than at the Cloud Run level. This approach provides flexibility in authentication logic while keeping the infrastructure simple.

IAM authentication restricts access to entities with appropriate GCP IAM permissions. This approach suits internal services where all callers have GCP identities. The service automatically validates tokens and rejects unauthorized requests.

Identity-Aware Proxy provides user-level authentication integrated with Google identities. This approach suits web applications where end users authenticate. IAP handles the authentication flow and injects user identity into requests.

### Secret Management

Secrets like API keys and database passwords should never appear in source code or environment variables visible in configuration. Google Secret Manager provides secure secret storage with controlled access.

Secrets store encrypted in Secret Manager and inject into containers at runtime. Cloud Run retrieves secrets during container startup, making them available as environment variables or mounted files without exposure in configuration.

The service account running containers needs Secret Manager Accessor role to read secrets. This permission scoping ensures only authorized containers can access secrets.

Rotation procedures update secrets without service interruption. Creating a new secret version and redeploying containers picks up the new value. The old version remains available for rollback if needed.

### Network Security

Cloud Run services run on Google's network infrastructure with automatic DDoS protection. Google's global load balancing absorbs attack traffic before it reaches containers.

VPC connectors enable private network access when Cloud Run needs to reach internal resources. A connector attaches Cloud Run to a VPC network, enabling access to Compute Engine instances, Cloud SQL databases, or on-premises resources through VPN.

Ingress settings control where traffic can originate. The default allows traffic from anywhere. Restricting to internal traffic only blocks external access, useful for backend services that should only receive requests from other GCP services.

Egress settings control where containers can send traffic. Routing all egress through VPC enables using Cloud NAT for static IP addresses or accessing private resources.

### Service Account Permissions

The runtime service account identity controls what resources containers can access. Following least privilege principles, this account should have only the permissions actually needed.

For the wine quality service, the runtime account needs minimal permissions. It doesn't access other GCP services beyond Secret Manager (if using secrets). Avoiding unnecessary permissions limits blast radius if the service is compromised.

The deployment service account needs broader permissions for deployment operations. However, this account's credentials store securely in GitHub Secrets and only CI/CD workflows use them.

---

## Troubleshooting

### Deployment Failures

Deployment failures typically stem from a few common causes. Understanding these causes enables quick diagnosis and resolution.

Permission errors indicate the service account lacks required IAM roles. Error messages specify which permission is missing. Adding the appropriate role to the service account resolves the issue.

Image not found errors occur when Cloud Run cannot pull the specified image. This might indicate the image wasn't pushed, the image path is incorrect, or permissions don't allow pulling from the registry.

Resource exhaustion errors occur when requested CPU or memory exceeds platform limits. Cloud Run enforces maximum resource allocations. Reducing requested resources or requesting quota increases resolves this.

### Container Startup Failures

Containers that fail to start prevent the service from handling traffic. Cloud Run marks such deployments as failed and maintains the previous working version.

Crash loops indicate the container starts but immediately exits. Application errors, missing dependencies, or configuration problems cause crashes. Container logs reveal the specific error.

Health check failures indicate the container starts but doesn't respond correctly to health probes. The application might be listening on the wrong port or taking too long to initialize.

Resource constraints might prevent successful startup. Insufficient memory causes out-of-memory kills during initialization. The wine quality model requires loading into memory, consuming significant resources during startup.

### Runtime Issues

Runtime issues affect running containers rather than deployment. These issues might be intermittent or dependent on specific request patterns.

Timeout errors occur when requests exceed the configured timeout. For prediction requests, this might indicate model loading problems or unusual input data. For training triggers, longer timeouts might be necessary.

Memory exhaustion causes containers to crash under load. This might happen with high concurrency or unusually large requests. Reducing concurrency or increasing memory allocation helps.

Cold start latency affects requests that arrive when no instances are running. The container must start and the model must load before serving. Setting minimum instances to one eliminates cold starts for latency-sensitive workloads.

### Investigation Procedures

Effective troubleshooting follows a systematic approach. Start with logs, which capture detailed information about what happened. Filter logs by time range, severity, and service name to find relevant entries.

Check metrics for anomalies around the problem time. Latency spikes, error rate increases, or resource utilization changes provide context about what changed.

Review recent deployments that might have introduced problems. Cloud Run's revision history shows exactly when deployments happened. Rolling back to a previous revision tests whether the deployment caused the issue.

Test in isolation to reproduce problems. Using curl to send specific requests helps determine whether issues are request-specific or systemic.

---

## Advanced Topics

### Traffic Splitting

Traffic splitting sends different percentages of traffic to different revisions. This capability enables several advanced deployment patterns.

Canary deployments send a small percentage of traffic to a new revision while most traffic continues to the previous version. If the new version causes problems, only a small fraction of users are affected. Gradually increasing the percentage completes the rollout.

A/B testing sends traffic to different versions based on percentage allocation. Combined with analytics, this reveals which version performs better for business metrics.

Blue-green deployments maintain two complete environments. Traffic switches entirely between them rather than gradually shifting. This approach provides instant rollback capability.

Traffic splitting configuration specifies which revisions receive what percentage of traffic. All percentages must sum to 100. Cloud Run handles the routing automatically based on configuration.

### Custom Domains

Cloud Run provides automatic domain names, but custom domains improve user experience and branding. Domain mapping connects a custom domain to a Cloud Run service.

Domain verification proves ownership of the domain. This typically involves adding a DNS TXT record with a verification code provided by Google.

DNS configuration points the domain to Cloud Run. An A or AAAA record for apex domains, or a CNAME record for subdomains, directs traffic to Google's servers.

SSL certificates provision automatically through Google-managed certificates. After DNS propagates, Cloud Run obtains and renews certificates without manual intervention.

### VPC Integration

VPC connectors enable Cloud Run to access resources in Virtual Private Cloud networks. This capability unlocks several use cases.

Database access becomes possible for Cloud SQL instances without public IPs. The connector routes traffic through the VPC where the database is accessible.

Internal service communication enables Cloud Run services to call other internal services without exposing them publicly. Service-to-service authentication combines with VPC routing for secure internal communication.

On-premises connectivity through Cloud VPN or Interconnect extends to Cloud Run through VPC connectors. The same VPC routing that reaches on-premises resources becomes available to containers.

### Multi-Region Deployment

For global applications, deploying to multiple regions improves latency and availability. Cloud Run services can deploy independently to different regions.

Global load balancing distributes traffic to the nearest healthy region. Users in Europe reach the European deployment while users in Asia reach the Asian deployment.

Regional failover handles region-wide outages. If one region becomes unavailable, traffic automatically routes to healthy regions.

Data consistency becomes complex with multi-region deployments. If the ML model updates, all regions need the new version. Coordination mechanisms ensure consistency across regions.

---

## References

### GCP Documentation

Google's Cloud Run documentation provides comprehensive reference material covering all platform capabilities. The serverless VPC access documentation explains VPC connector configuration. The Cloud Build documentation covers CI/CD integration options beyond GitHub Actions.

### Related Guides

The main README.md provides overall project context. The MLSECOPS_README.md covers security considerations that apply to cloud deployments. The monitoring/ directory contains Streamlit dashboard code that could deploy separately for monitoring the production service.

### Support Resources

Google Cloud Support provides assistance for GCP-specific issues. GitHub Support helps with Actions workflow problems. Community forums and Stack Overflow answer common questions.

---

## Version Information

- Version: 1.0.0
- Last Updated: December 2025
- Platform: Google Cloud Platform
- Services: Cloud Run, Artifact Registry, Cloud Logging, Cloud Monitoring
