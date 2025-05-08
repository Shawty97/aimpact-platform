# Deployment Guide

This guide covers deploying the AImPact platform in various environments.

## Prerequisites

- Docker and Docker Compose for local/development deployments
- Kubernetes cluster for production deployments
- PostgreSQL database
- Redis instance
- Object storage (S3-compatible)

## Environment Setup

Create an appropriate `.env` file based on the example:

```
# Core Settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info
SECRET_KEY=your-secret-key

# Database
DATABASE_URL=postgresql://user:password@host:port/dbname
REDIS_URL=redis://host:port/0

# Storage
S3_ENDPOINT=https://s3.amazonaws.com
S3_BUCKET=aimpact-storage
S3_ACCESS_KEY=your-access-key
S3_SECRET_KEY=your-secret-key

# Authentication
JWT_SECRET=your-jwt-secret
JWT_ALGORITHM=HS256
JWT_EXPIRATION=86400

# API Configuration
ENABLE_CORS=true
ALLOWED_ORIGINS=https://yourdomain.com
API_RATE_LIMIT=true
```

## Local Deployment

Use Docker Compose for local deployment:

```bash
# Build and start services
docker-compose up -d

# Run database migrations
docker-compose exec backend alembic upgrade head

# Seed initial data (optional)
docker-compose exec backend python scripts/seed_data.py
```

## Kubernetes Deployment

The platform can be deployed on Kubernetes using the provided manifests or Helm charts.

### Using kubectl

```bash
# Apply ConfigMaps and Secrets
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml

# Deploy database (if not using managed service)
kubectl apply -f k8s/database.yaml

# Deploy Redis (if not using managed service)
kubectl apply -f k8s/redis.yaml

