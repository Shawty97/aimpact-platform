# AImpact Authentication Service

A secure, scalable authentication and user management service for the AImpact platform. This service provides JWT-based authentication, role-based access control, and comprehensive user management capabilities.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Security Measures](#security-measures)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Deployment Guide](#deployment-guide)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Features

- **JWT-based Authentication**: Secure token-based authentication with access and refresh tokens
- **User Management**: Complete user lifecycle management (registration, updates, deactivation)
- **Role-Based Access Control**: Fine-grained access control with customizable roles and permissions
- **Token Management**: Token validation, blacklisting, and refresh capabilities
- **Security Features**: Password strength validation, rate limiting, and protection against common attacks
- **Async Database Support**: Asynchronous database operations for high-performance applications
- **Comprehensive Testing**: Test coverage for all critical authentication and user management flows

## Architecture

The authentication service follows a clean architecture pattern with the following components:

- **API Layer**: FastAPI-based endpoints for authentication and user management
- **Service Layer**: Business logic for authentication, user management, and token handling
- **Repository Layer**: Database access for user and token data
- **Models**: Pydantic models for request/response validation and SQLAlchemy models for database interactions
- **Core**: Configuration, dependencies, and security utilities

### Key Components:

- `services/auth_service.py`: Authentication business logic
- `services/user_service.py`: User management business logic
- `services/token_service.py`: Token generation, validation, and blacklisting
- `models/user.py`: User data model
- `models/role.py`: Role and permission models
- `models/token.py`: Token blacklist model
- `routes/auth.py`: Authentication endpoints
- `routes/users.py`: User management endpoints

## Security Measures

The authentication service implements multiple layers of security:

- **Secure Password Storage**: Passwords are hashed using bcrypt with per-user salt
- **JWT with Short Lifetimes**: Access tokens have short lifetimes with refresh token rotation
- **Token Blacklisting**: Revoked tokens are tracked to prevent reuse
- **Rate Limiting**: Protection against brute force attacks
- **Password Strength Validation**: Enforces strong password policies
- **HTTPS Only**: All API endpoints require HTTPS in production
- **CORS Protection**: Configurable CORS settings to prevent cross-site attacks
- **Role-Based Access**: Granular permission control for all operations

## Prerequisites

- Python 3.10+
- PostgreSQL 13+
- Docker (optional, for containerized deployment)
- Redis (optional, for rate limiting and token blacklisting)

## Installation

### Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/aimpact.git
   cd aimpact/services/auth
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables (create a `.env` file in the project root):
   ```env
   # Database
   DB_USER=postgres
   DB_PASSWORD=your_password
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=aimpact_auth
   
   # JWT
   JWT_SECRET_KEY=your_secret_key
   JWT_ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   REFRESH_TOKEN_EXPIRE_MINUTES=10080  # 7 days
   
   # Service
   HOST=0.0.0.0
   PORT=8000
   ENVIRONMENT=development  # development, testing, production
   ```

5. Set up the database:
   ```bash
   # Create database
   createdb aimpact_auth
   
   # Run migrations
   alembic upgrade head
   ```

6. Start the service:
   ```bash
   uvicorn main:app --reload
   ```

### Docker Setup

1. Build the Docker image:
   ```bash
   docker build -t aimpact-auth .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 --env-file .env aimpact-auth
   ```

## Configuration

Configuration is managed through environment variables, with sensible defaults provided. The main configuration options are:

### Database Configuration

- `DB_USER`: Database username (default: postgres)
- `DB_PASSWORD`: Database password
- `DB_HOST`: Database host (default: localhost)
- `DB_PORT`: Database port (default: 5432)
- `DB_NAME`: Database name (default: aimpact_auth)

### JWT Configuration

- `JWT_SECRET_KEY`: Secret key for JWT token signing
- `JWT_ALGORITHM`: Algorithm for JWT token signing (default: HS256)
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Access token lifetime in minutes (default: 30)
- `REFRESH_TOKEN_EXPIRE_MINUTES`: Refresh token lifetime in minutes (default: 10080)

### Service Configuration

- `HOST`: Host to bind the service to (default: 0.0.0.0)
- `PORT`: Port to run the service on (default: 8000)
- `ENVIRONMENT`: Environment (development, testing, production)
- `LOG_LEVEL`: Logging level (default: INFO)

## API Documentation

### Authentication Endpoints

#### POST /auth/login

Authenticates a user and issues access and refresh tokens.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "securePassword123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### POST /auth/refresh

Issues a new access token using a valid refresh token.

**Request:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### POST /auth/logout

Invalidates the current access token.

**Request:**
Headers: `Authorization: Bearer {token}`

**Response:**
```json
{
  "message": "Successfully logged out"
}
```

### User Management Endpoints

#### POST /users/

Creates a new user (admin only).

**Request:**
```json
{
  "email": "newuser@example.com",
  "password": "SecurePass123!",
  "first_name": "John",
  "last_name": "Doe",
  "role_id": 2
}
```

**Response:**
```json
{
  "id": 1,
  "email": "newuser@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "is_active": true,
  "role_id": 2,
  "created_at": "2023-01-01T00:00:00"
}
```

#### GET /users/

List all users (admin only).

**Response:**
```json
[
  {
    "id": 1,
    "email": "user@example.com",
    "first_name": "John",
    "last_name": "Doe",
    "is_active": true,
    "role_id": 2,
    "created_at": "2023-01-01T00:00:00"
  }
]
```

#### GET /users/me

Get the current user's profile.

**Response:**
```json
{
  "id": 1,
  "email": "user@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "is_active": true,
  "role_id": 2,
  "created_at": "2023-01-01T00:00:00"
}
```

#### PATCH /users/{user_id}

Update a user's information.

**Request:**
```json
{
  "first_name": "Updated",
  "last_name": "Name"
}
```

**Response:**
```json
{
  "id": 1,
  "email": "user@example.com",
  "first_name": "Updated",
  "last_name": "Name",
  "is_active": true,
  "role_id": 2,
  "created_at": "2023-01-01T00:00:00"
}
```

#### POST /users/{user_id}/activate

Activate a user account (admin only).

**Response:**
```json
{
  "id": 1,
  "email": "user@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "is_active": true,
  "role_id": 2,
  "created_at": "2023-01-01T00:00:00"
}
```

#### POST /users/{user_id}/deactivate

Deactivate a user account (admin only).

**Response:**
```json
{
  "id": 1,
  "email": "user@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "is_active": false,
  "role_id": 2,
  "created_at": "2023-01-01T00:00:00"
}
```

#### POST /users/me/change-password

Change the current user's password.

**Request:**
```json
{
  "current_password": "OldPassword123",
  "new_password": "NewSecurePassword456!"
}
```

**Response:**
```json
{
  "message": "Password updated successfully"
}
```

## Deployment Guide

### Production Configuration

For a production deployment, make sure to:

1. Set `ENVIRONMENT=production` in your environment variables
2. Use a strong, randomly generated `JWT_SECRET_KEY`
3. Configure SSL/TLS for HTTPS
4. Set up proper monitoring and logging
5. Set up database backups
6. Consider using a managed database service

### Database Setup

1. Create a production database:

   ```bash
   createdb aimpact_auth_prod
   ```

2. Set up the database schema:

   ```bash
   # Run migrations
   alembic upgrade head
   ```

3. Create initial roles and admin user:

   ```bash
   # Run the setup script
   python -m scripts.init_db
   ```

### Kubernetes Deployment

1. Create Kubernetes secrets for sensitive configuration:

   ```bash
   kubectl create secret generic aimpact-auth-secrets \
     --from-literal=JWT_SECRET_KEY=your_secret_key \
     --from-literal=DB_PASSWORD=your_db_password
   ```

2. Apply the Kubernetes manifests:

   ```bash
   kubectl apply -f k8s/
   ```

### Docker Compose Deployment

For a simple multi-container setup, use Docker Compose:

```yaml
# docker-compose.yml
version: '3'

services:
  auth:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env.prod
    depends_on:
      - db
      - redis
    restart: always

  db:
    image: postgres:13
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=aimpact_auth
    restart: always

  redis:
    image: redis:6
    restart: always

volumes:
  postgres_data:
```

### Security Considerations

1. **Secrets Management**: Use a secure vault for managing secrets in production
2. **Network Security**: Implement proper network policies and firewall rules
3. **Regular Updates**: Keep dependencies updated to patch security vulnerabilities
4. **Audit Logging**: Enable comprehensive audit logging for all authentication events
5. **Monitoring**: Set up alerting for suspicious authentication patterns
6. **Backups**: Ensure regular database backups with secure storage

## Testing

The authentication service includes comprehensive test coverage for all components.

### Running Tests

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_auth.py
pytest tests/test_users.py

# Run with coverage report
pytest --cov=src
```

### Test Structure

- `tests/conftest.py`: Test fixtures and configuration
- `tests/test_auth.py`: Authentication flow tests
- `tests/test_users.py`: User management tests
- `tests/test_token.py`: Token management tests

## Troubleshooting

### Common Issues

1. **Database Connection Issues**:
   - Check the database credentials in your `.env` file
   - Ensure the database server is running
   - Verify network connectivity to the database

2. **Token Validation Failures**:
   - Check if the `JWT_SECRET_KEY` is consistent across all instances
   - Ensure the system time is synchronized
   - Verify that tokens haven't expired or been blacklisted

3. **Permission Errors**:
   - Verify that the user has the correct role assigned
   - Check that the role has the necessary permissions

### Support

For additional support, please contact the AImpact team at support@aimpact.ai or open an issue in the project repository.

