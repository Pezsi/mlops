# GCP Cloud Run - Quick Start Guide

This guide provides a streamlined path to deploying the Wine Quality MLOps platform on Google Cloud Platform. For detailed explanations and advanced configuration, see the comprehensive GCP_DEPLOYMENT_GUIDE.md in the project root.

---

## Overview

Deployment to GCP Cloud Run follows four main phases: preparing the GCP environment, configuring GitHub integration, deploying the application, and verifying operation. The process takes approximately 15-30 minutes for first-time setup.

---

## Prerequisites

### Required Accounts

A Google Cloud Platform account with billing enabled is required. Cloud Run usage within free tier limits incurs no charges. Beyond free tier, expect costs of approximately 5-20 USD monthly for development workloads.

A GitHub account is required if using automated deployment through GitHub Actions. Manual deployment works without GitHub.

### Required Tools

The Google Cloud SDK provides the gcloud command-line tool. Installation differs by operating system. On macOS, Homebrew offers the simplest installation method. On Linux, the official installer script handles installation and PATH configuration. On Windows, download the installer from Google Cloud's documentation.

After installation, verify the SDK works by checking its version. The gcloud init command walks through initial configuration including authentication and default project selection.

Docker is required for building container images. Install Docker Desktop on macOS and Windows. On Linux, install Docker Engine through your distribution's package manager.

---

## Phase 1: GCP Environment Setup

### Understanding the Setup

The setup phase creates GCP resources that persist across deployments. Artifact Registry stores Docker images. A service account provides deployment credentials. IAM permissions authorize necessary operations.

Running the automated setup script handles all these tasks. The script is idempotent, meaning running it multiple times is safe. It creates resources if they don't exist and skips creation if they do.

### Running Setup

Before running setup, set environment variables specifying your GCP project ID and preferred region. The project ID appears in the GCP Console header. The region should be geographically close to your users; europe-west1 suits European users, us-central1 suits North American users.

Execute the setup script from the project's gcp directory. The script outputs progress messages as it creates resources. Upon completion, it displays instructions for the next phase.

The setup creates a service account key file. This file contains credentials for deployment automation. Keep it secure and never commit it to version control. The file is needed only for GitHub Actions configuration.

---

## Phase 2: GitHub Integration

### Why GitHub Actions

GitHub Actions automates the build and deployment process. When you push code changes, GitHub automatically builds a new Docker image, runs tests, and deploys to Cloud Run. This automation eliminates manual deployment steps and ensures consistent deployments.

Manual deployment remains an option for those preferring direct control or not using GitHub.

### Configuring Secrets

GitHub Secrets store sensitive values that workflows can access. Two secrets are required: the GCP project ID and the service account key.

Navigate to your GitHub repository's Settings page. Find the Secrets and Variables section, then the Actions subsection. Create new repository secrets for each required value.

The project ID secret stores your GCP project identifier as plain text. The service account key secret stores the entire contents of the JSON key file generated during setup. Copy the complete file contents, including braces.

### Triggering Deployment

After configuring secrets, deployment triggers automatically on pushes to the main branch. You can also trigger deployment manually through the GitHub Actions interface.

Monitor deployment progress in the Actions tab. The workflow shows each step's status. Green checkmarks indicate success. Red X marks indicate failures requiring investigation.

---

## Phase 3: Manual Deployment Alternative

### When to Use Manual Deployment

Manual deployment suits several scenarios: testing deployment before configuring automation, deploying from branches other than main, debugging deployment issues, or organizational policies prohibiting CI/CD secrets.

### Deployment Steps

Manual deployment follows the same logical steps as automated deployment, executed through direct commands.

First, configure Docker to authenticate with GCP's Artifact Registry. This one-time setup enables pushing images to your registry.

Second, build the Docker image using the Cloud Run-specific Dockerfile. The image tag includes the full Artifact Registry path so Docker knows where to push it.

Third, push the built image to Artifact Registry. This uploads the image to GCP's container storage.

Fourth, deploy to Cloud Run specifying the image location and configuration options. The command returns the service URL upon successful deployment.

The deploy script in the gcp directory automates these steps. Running it executes all necessary commands in sequence.

---

## Phase 4: Verification

### Testing the Deployment

After deployment completes, verify the service operates correctly. The deployment process outputs the service URL. If you missed it, retrieve it using the gcloud command to describe the service.

### Health Check

Access the health endpoint to verify basic operation. The health check confirms the service is running and can handle requests. A successful response indicates the deployment succeeded.

### API Documentation

Access the /docs path to view the interactive API documentation. The Swagger UI displays all available endpoints. You can test predictions directly through this interface without writing code.

### Test Prediction

Submit a test prediction to verify the model loads correctly. The prediction endpoint accepts wine feature values and returns a quality prediction. Any reasonable input values should produce a prediction between 3 and 8.

---

## Ongoing Operations

### Monitoring

GCP Console provides monitoring dashboards for Cloud Run services. View request counts, latency distributions, and error rates. Configure alerts to notify you of issues.

The monitoring script in the gcp directory provides convenient access to common monitoring operations through an interactive menu.

### Updating

Code updates deploy automatically through GitHub Actions when pushed to main. The workflow builds a new image, deploys it, and transitions traffic to the new version.

For manual updates, rebuild the image and redeploy. Cloud Run handles traffic transition automatically.

### Cost Control

Cloud Run charges based on resource consumption. Scale-to-zero eliminates costs when the service receives no traffic. This setting is the default and suits development workloads.

For production workloads requiring low latency, minimum instances keep the service warm. This increases costs but eliminates cold start delays.

Monitor costs in the GCP Console billing section. Set budget alerts to avoid surprises.

### Cleanup

When the deployment is no longer needed, cleanup removes all created resources. This eliminates ongoing costs and cleans up the GCP project.

The cleanup script in the gcp directory automates resource deletion. It removes the Cloud Run service, Artifact Registry repository, and service account.

---

## Troubleshooting Quick Reference

### Permission Denied

Permission errors indicate the service account lacks required roles. Verify IAM permissions in GCP Console. The service account needs Cloud Run Admin, Artifact Registry Writer, and Service Account User roles.

### Image Not Found

Image not found errors indicate deployment references a nonexistent image. Verify the image pushed successfully by listing images in Artifact Registry. Rebuild and push if necessary.

### Container Startup Timeout

Startup timeouts indicate the container takes too long to initialize. The default timeout might be insufficient for loading ML models. Increase the timeout through service update commands.

### Out of Memory

Memory errors indicate the container exceeds its memory allocation. ML models require significant memory. Increase memory allocation through service update commands.

---

## Next Steps

After successful deployment, consider the following enhancements:

Configure a custom domain for a branded URL instead of the auto-generated Cloud Run URL.

Set up Cloud Monitoring alerts for proactive issue notification.

Configure minimum instances for production workloads requiring consistent low latency.

Review the comprehensive GCP_DEPLOYMENT_GUIDE.md for advanced topics including traffic splitting, VPC integration, and multi-region deployment.

---

## Related Documentation

The comprehensive deployment guide at GCP_DEPLOYMENT_GUIDE.md covers all topics in depth.

The main README.md provides project overview and architecture context.

The DEVELOPMENT_GUIDE.md covers local development setup for testing before deployment.
