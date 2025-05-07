import json
import pytest
import time
from datetime import datetime, timedelta
from unittest import mock

import jwt
from fastapi import status
from fastapi.testclient import TestClient
from pydantic import EmailStr

from core.config import settings
from models.user import User
from services.auth_service import AuthService
from services.token_service import TokenService


@pytest.mark.asyncio
async def test_login_success(client: TestClient, users: list[User]):
    """Test successful login with valid credentials."""
    # Arrange
    login_data = {
        "email": "user@example.com",
        "password": "User@123"
    }
    
    # Act
    response = client.post("/auth/login", json=login_data)
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert "access_token" in response.json()
    assert "refresh_token" in response.json()
    assert "token_type" in response.json()
    assert response.json()["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_invalid_credentials(client: TestClient):
    """Test login with invalid credentials."""
    # Arrange
    login_data = {
        "email": "user@example.com",
        "password": "WrongPassword"
    }
    
    # Act
    response = client.post("/auth/login", json=login_data)
    
    # Assert
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "detail" in response.json()
    assert "Invalid credentials" in response.json()["detail"]


@pytest.mark.asyncio
async def test_login_nonexistent_user(client: TestClient):
    """Test login with non-existent user."""
    # Arrange
    login_data = {
        "email": "nonexistent@example.com",
        "password": "Password123"
    }
    
    # Act
    response = client.post("/auth/login", json=login_data)
    
    # Assert
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_login_inactive_user(client: TestClient, users: list[User]):
    """Test login with inactive user account."""
    # Arrange
    login_data = {
        "email": "inactive@example.com",
        "password": "Inactive@123"
    }
    
    # Act
    response = client.post("/auth/login", json=login_data)
    
    # Assert
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Account is inactive" in response.json()["detail"]


@pytest.mark.asyncio
async def test_login_rate_limiting(client: TestClient, monkeypatch):
    """Test rate limiting on login endpoint."""
    # Mock rate limiter to simulate too many requests
    def mock_is_rate_limited(*args, **kwargs):
        return True
    
    monkeypatch.setattr("services.auth_service.is_rate_limited", mock_is_rate_limited)
    
    # Arrange
    login_data = {
        "email": "user@example.com",
        "password": "User@123"
    }
    
    # Act
    response = client.post("/auth/login", json=login_data)
    
    # Assert
    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert "Too many login attempts" in response.json()["detail"]


@pytest.mark.asyncio
async def test_logout(client: TestClient, tokens: dict):
    """Test successful logout."""
    # Arrange
    token = tokens["user@example.com"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Act
    response = client.post("/auth/logout", headers=headers)
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert "message" in response.json()
    assert "Successfully logged out" in response.json()["message"]
    
    # Attempt to use the token again
    response2 = client.get("/users/me", headers=headers)
    assert response2.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_refresh_token(client: TestClient, tokens: dict):
    """Test refresh token endpoint."""
    # Arrange
    refresh_token = tokens["user@example.com"]["refresh_token"]
    refresh_data = {"refresh_token": refresh_token}
    
    # Act
    response = client.post("/auth/refresh", json=refresh_data)
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert "access_token" in response.json()
    assert "token_type" in response.json()
    assert response.json()["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_refresh_token_blacklisted(client: TestClient, tokens: dict, token_service: TokenService):
    """Test refresh token endpoint with blacklisted token."""
    # Arrange
    refresh_token = tokens["user@example.com"]["refresh_token"]
    # Blacklist the token
    await token_service.blacklist_token(refresh_token)
    
    refresh_data = {"refresh_token": refresh_token}
    
    # Act
    response = client.post("/auth/refresh", json=refresh_data)
    
    # Assert
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Token has been revoked" in response.json()["detail"]


@pytest.mark.asyncio
async def test_refresh_invalid_token(client: TestClient):
    """Test refresh token endpoint with invalid token."""
    # Arrange
    invalid_token = "invalid.token.format"
    refresh_data = {"refresh_token": invalid_token}
    
    # Act
    response = client.post("/auth/refresh", json=refresh_data)
    
    # Assert
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Could not validate credentials" in response.json()["detail"]


@pytest.mark.asyncio
async def test_refresh_expired_token(client: TestClient, expired_token: str):
    """Test refresh token endpoint with expired token."""
    # Arrange
    refresh_data = {"refresh_token": expired_token}
    
    # Act
    response = client.post("/auth/refresh", json=refresh_data)
    
    # Assert
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Token has expired" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_me(client: TestClient, tokens: dict):
    """Test get current user info endpoint."""
    # Arrange
    token = tokens["user@example.com"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Act
    response = client.get("/users/me", headers=headers)
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert "email" in response.json()
    assert response.json()["email"] == "user@example.com"
    assert "id" in response.json()
    assert "is_active" in response.json()
    assert response.json()["is_active"] is True


@pytest.mark.asyncio
async def test_get_me_no_token(client: TestClient):
    """Test get current user info without token."""
    # Act
    response = client.get("/users/me")
    
    # Assert
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Not authenticated" in response.json()["detail"]


@pytest.mark.asyncio
async def test_token_blacklisting(token_service: TokenService, tokens: dict):
    """Test token blacklisting functionality."""
    # Arrange
    token = tokens["user@example.com"]["access_token"]
    
    # Act - Blacklist the token
    await token_service.blacklist_token(token)
    
    # Assert - Check if token is blacklisted
    is_blacklisted = await token_service.is_token_blacklisted(token)
    assert is_blacklisted is True


@pytest.mark.asyncio
async def test_token_validation(token_service: TokenService, tokens: dict):
    """Test token validation."""
    # Arrange
    token = tokens["user@example.com"]["access_token"]
    
    # Act
    payload = await token_service.decode_token(token)
    
    # Assert
    assert "sub" in payload
    assert "email" in payload
    assert payload["email"] == "user@example.com"
    assert "scopes" in payload
    assert "read" in payload["scopes"]


@pytest.mark.asyncio
async def test_token_scopes(token_service: TokenService, tokens: dict):
    """Test token scopes for different user types."""
    # Arrange - Admin token
    admin_token = tokens["admin@example.com"]["access_token"]
    
    # Act
    admin_payload = await token_service.decode_token(admin_token)
    
    # Assert - Admin should have read and write scopes
    assert "scopes" in admin_payload
    assert "read" in admin_payload["scopes"]
    assert "write" in admin_payload["scopes"]
    
    # Arrange - Regular user token
    user_token = tokens["user@example.com"]["access_token"]
    
    # Act
    user_payload = await token_service.decode_token(user_token)
    
    # Assert - Regular user should only have read scope
    assert "scopes" in user_payload
    assert "read" in user_payload["scopes"]
    assert "write" not in user_payload["scopes"]


@pytest.mark.asyncio
async def test_purge_expired_blacklist_tokens(token_service: TokenService, tokens: dict):
    """Test purging expired tokens from blacklist."""
    # Arrange - Blacklist some tokens
    token1 = tokens["user@example.com"]["access_token"]
    token2 = tokens["guest@example.com"]["access_token"]
    
    await token_service.blacklist_token(token1)
    await token_service.blacklist_token(token2)
    
    # Act - Mock current time to be after token expiration
    with mock.patch('services.token_service.datetime') as mock_datetime:
        # Set current time to future
        future_time = datetime.utcnow() + timedelta(hours=24)
        mock_datetime.utcnow.return_value = future_time
        
        # Purge expired tokens
        purged_count = await token_service.purge_expired_blacklist_tokens()
    
    # Assert
    assert purged_count >= 2  # Should have purged at least our 2 tokens


@pytest.mark.asyncio
async def test_security_token_tampering(token_service: TokenService, tokens: dict):
    """Test security against token tampering."""
    # Arrange
    token = tokens["user@example.com"]["access_token"]
    
    # Decode token without verification to tamper with it
    payload = jwt.decode(
        token,
        options={"verify_signature": False}
    )
    
    # Tamper with the payload
    payload["email"] = "admin@example.com"  # Try to escalate privileges
    
    # Re-encode with the tampered payload but without the proper signature
    tampered_token = jwt.encode(
        payload,
        "wrong_secret_key",  # Wrong key
        algorithm=settings.JWT_ALGORITHM
    )
    
    # Act & Assert - Should raise an exception
    with pytest.raises(Exception):
        await token_service.decode_token(tampered_token)


@pytest.mark.asyncio
async def test_protected_endpoint_access_control(client: TestClient, tokens: dict):
    """Test role-based access control for protected endpoints."""
    # Arrange - Admin endpoint
    admin_endpoint = "/admin/users"
    
    # Admin token should work
    admin_token = tokens["admin@example.com"]["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    
    # Regular user token should be denied
    user_token = tokens["user@example.com"]["access_token"]
    user_headers = {"Authorization": f"Bearer {user_token}"}
    
    # Act
    admin_response = client.get(admin_endpoint, headers=admin_headers)
    user_response = client.get(admin_endpoint, headers=user_headers)
    
    # Assert
    assert admin_response.status_code == status.HTTP_200_OK
    assert user_response.status_code == status.HTTP_403_FORBIDDEN
    assert "Not enough permissions" in user_response.json()["detail"]

