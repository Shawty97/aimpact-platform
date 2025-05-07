# AImpact Platform Deployment Guide

This guide provides comprehensive instructions for deploying the AImpact platform in both development and production environments. It covers environment setup, deployment options, configuration, security considerations, and monitoring setup.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Deployment Options](#deployment-options)
   - [Local Development](#local-development-with-docker-compose)
   - [Production Deployment](#production-deployment-with-kubernetes)
4. [Configuration Guide](#configuration-guide)
5. [Security Considerations](#security-considerations)
6. [Monitoring Setup](#monitoring-setup)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

Before deploying the AImpact platform, ensure you have the following installed:

- Docker and Docker Compose (for local development)
- Kubernetes command-line tool (kubectl) (for production deployment)
- Git
- A container registry account (for production deployment)

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/aimpact.git
cd aimpact
```

### 2. Set Up Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# Core Configuration
ENVIRONMENT=development  # or 'production' for production deployments
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql://postgres:postgres@db:5432/aimpact
VECTOR_DB_URL=postgresql://postgres:postgres@db:5432/aimpact
VECTOR_DB_TYPE=postgres

# Redis Configuration
REDIS_URL=redis://redis:6379/0

# Authentication
JWT_SECRET=your_secure_jwt_secret_here
JWT_EXPIRE=86400  # 24 hours in seconds

# LLM Configuration
DEFAULT_LLM_PROVIDER=local  # 'openai', 'anthropic', 'local'
OPENAI_API_KEY=your_openai_key_here  # Only needed if using OpenAI
ANTHROPIC_API_KEY=your_anthropic_key_here  # Only needed if using Anthropic
OLLAMA_URL=http://ollama:11434  # Local LLM URL
OLLAMA_MODEL=mistral  # Model to use with Ollama

# Voice Processing
VOICE_ENGINE_TYPE=coqui
SPEECH_RECOGNITION_PROVIDER=vosk
TTS_MODEL_PATH=/app/models/tts/coqui
STT_MODEL_PATH=/app/models/stt/vosk
USE_LOCAL_TTS=true
USE_LOCAL_STT=true

# Service Discovery
WORKFLOW_ENGINE_URL=http://workflow-engine:8050/api
VOICE_SERVICE_URL=http://voice-service:8060/api
LLM_ORCHESTRATOR_URL=http://llm-orchestrator:8070/api
```

> **Note**: For production deployments, use proper secrets management as described in the [Security Considerations](#security-considerations) section.

## Deployment Options

### Local Development with Docker Compose

For local development and testing, we use Docker Compose to set up the entire platform with all its dependencies.

#### 1. Start the Development Environment

```bash
docker-compose up -d
```

This command will:
- Build and start all services defined in the `docker-compose.yml` file
- Set up PostgreSQL with vector extensions
- Configure Redis for caching and message queueing
- Start Prometheus and Grafana for monitoring
- Start the local LLM service (Ollama)
- Launch the backend API and all microservices

#### 2. Access Services

- Backend API: http://localhost:8000
- Dashboard: http://localhost:3000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001 (admin/admin)

#### 3. Development Workflow

During development:

```bash
# View logs of a specific service
docker-compose logs -f backend

# Restart a specific service
docker-compose restart backend

# Stop all services
docker-compose down

# Stop all services and remove volumes
docker-compose down -v
```

### Production Deployment with Kubernetes

For production deployments, we use Kubernetes to ensure scalability, reliability, and maintainability.

#### 1. Create the Kubernetes Namespace

```bash
kubectl create namespace aimpact
```

#### 2. Set Up Environment Secrets

```bash
# Create secrets from .env file
kubectl create secret generic aimpact-secrets \
  --namespace=aimpact \
  --from-literal=database-url="postgresql://username:password@db-host:5432/aimpact" \
  --from-literal=redis-url="redis://redis-host:6379/0" \
  --from-literal=openai-api-key="your_openai_key_here" \
  --from-literal=jwt-secret="your_secure_jwt_secret_here"

# Create Docker registry secret
kubectl create secret docker-registry regcred \
  --namespace=aimpact \
  --docker-server=your-registry-server \
  --docker-username=your-username \
  --docker-password=your-password
```

#### 3. Deploy ConfigMap

```bash
kubectl apply -f k8s/configmap.yaml
```

#### 4. Set Up Infrastructure Components

Deploy PostgreSQL and Redis:

```bash
# For production, use managed services when possible
# Example for PostgreSQL using Helm:
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install postgres bitnami/postgresql \
  --namespace=aimpact \
  --set postgresqlUsername=postgres \
  --set postgresqlPassword=postgres \
  --set postgresqlDatabase=aimpact

# Redis
helm install redis bitnami/redis \
  --namespace=aimpact
```

#### 5. Build and Push Docker Images

```bash
# Set variables
export DOCKER_REGISTRY=your-registry.io
export IMAGE_TAG=latest

# Build and push images
docker build -t ${DOCKER_REGISTRY}/aimpact-backend:${IMAGE_TAG} .
docker push ${DOCKER_REGISTRY}/aimpact-backend:${IMAGE_TAG}

# Build dashboard image
cd dashboard
docker build -t ${DOCKER_REGISTRY}/aimpact-dashboard:${IMAGE_TAG} .
docker push ${DOCKER_REGISTRY}/aimpact-dashboard:${IMAGE_TAG}
cd ..
```

#### 6. Deploy Application Components

```bash
# Deploy backend and dashboard
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Deploy ingress
kubectl apply -f k8s/ingress.yaml

# Verify deployments
kubectl get pods -n aimpact
```

#### 7. Set Up Monitoring

```bash
# Install Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace=aimpact

# Deploy custom Grafana dashboards
kubectl apply -f k8s/monitoring/grafana-dashboards.yaml
```

## Configuration Guide

### API Configuration

The API can be configured using the following environment variables or the ConfigMap in Kubernetes:

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment (development/production) | `development` |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `DATABASE_URL` | PostgreSQL database connection string | - |
| `REDIS_URL` | Redis connection string | - |
| `JWT_SECRET` | Secret for JWT token generation | - |
| `JWT_EXPIRE` | JWT token expiration time in seconds | `86400` |

### Voice Processing Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `VOICE_ENGINE_TYPE` | Voice synthesis engine (coqui, openai) | `coqui` |
| `SPEECH_RECOGNITION_PROVIDER` | Speech recognition provider (vosk, whisper) | `vosk` |
| `TTS_MODEL_PATH` | Path to TTS models | `/app/models/tts/coqui` |
| `STT_MODEL_PATH` | Path to STT models | `/app/models/stt/vosk` |

### LLM Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DEFAULT_LLM_PROVIDER` | Default LLM provider (openai, anthropic, local) | `local` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OLLAMA_URL` | URL for local Ollama service | `http://ollama:11434` |
| `OLLAMA_MODEL` | Model to use with Ollama | `mistral` |

## Security Considerations

### 1. Secrets Management

- In production, use a dedicated secrets management solution such as:
  - Kubernetes Secrets (with encryption at rest)
  - HashiCorp Vault
  - AWS Secrets Manager or Azure Key Vault
- Rotate secrets regularly
- Never commit secrets to version control

### 2. Network Security

- Use TLS for all services (enable SSL in ingress configuration)
- Set up network policies to restrict traffic between services
- Implement proper egress control for outbound connections

### 3. Container Security

- Use non-root users in containers (already configured in Dockerfile)
- Scan container images for vulnerabilities using tools like Trivy or Clair
- Implement pod security policies

### 4. Authentication and Authorization

- Use strong, unique passwords and rotate them regularly
- Implement proper RBAC (Role-Based Access Control) in Kubernetes
- Use the JWT authentication system with short-lived tokens

### 5. Regular Updates

- Keep all dependencies updated
- Apply security patches promptly
- Review security bulletins for all components regularly

## Monitoring Setup

### 1. Prometheus Metrics

The AImpact platform exposes metrics at the `/metrics` endpoint. The following metrics are available:

- HTTP request rates, errors, and latencies
- Voice processing throughput and error rates
- Agent execution metrics
- System resource utilization

### 2. Grafana Dashboards

Pre-configured Grafana dashboards are available for:

- System Metrics: CPU, memory, and network usage
- Voice Processing: Throughput, latency, and error rates
- Agent Performance: Success rates, response times, and utilization
- API Performance: Request rates, errors, and latencies

Access Grafana:
- In development: http://localhost:3001 (admin/admin)
- In production: Follow your ingress setup

### 3. Setting Up Alerts

Configure Prometheus alerts using the provided rules:

```bash
# In development
Edit monitoring/prometheus/rules/alerts.yml

# In production
kubectl apply -f k8s/monitoring/prometheus-alerts.yaml
```

Example alerts include:
- High error rates
- Service downtime
- API latency issues
- Resource constraints

### 4. Logging

Logs from all services are collected and can be aggregated using a solution such as:

- ELK Stack (Elasticsearch, Logstash, Kibana)
- Grafana Loki
- Cloud-native logging (e.g., AWS CloudWatch, Google Cloud Logging)

## Troubleshooting

### Common Issues

#### Services Not Starting

Check logs:
```bash
# Development
docker-compose logs <service>

# Production
kubectl logs -n aimpact <pod-name>
```

#### Database Connection Issues

- Verify the database is running
- Check connection strings
- Ensure database users have the correct permissions
- Verify network connectivity between services

#### Authentication Problems

- Check JWT_SECRET is correctly set
- Verify token expiration settings
- Ensure proper CORS configuration

#### Performance Issues

- Monitor resource usage with Prometheus
- Check for memory leaks
- Optimize database queries
- Consider scaling up resources or horizontal scaling

### Getting Help

If you encounter issues not covered in this guide:

- Check the [GitHub Issues](https://github.com/yourusername/aimpact/issues)
- Join our [Discord community](https://discord.gg/aimpact)
- Contact support at support@aimpact.example.com

