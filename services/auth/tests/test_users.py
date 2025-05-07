import pytest
import re
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from fastapi import status
from fastapi.testclient import TestClient
from pydantic import EmailStr

from models.user import User
from models.role import Role
from services.user_service import UserService


@pytest.mark.asyncio
async def test_create_user_success(user_service: UserService, roles: list[Role]):
    """Test successful user creation with valid data."""
    # Arrange
    email = EmailStr("newuser@example.com")
    password = "Password@123"
    first_name = "New"
    last_name = "User"
    
    # Act
    user = await user_service.create_user(
        email=email,
        password=password,
        first_name=first_name,
        last_name=last_name,
        role_id=roles[1].id  # use regular user role
    )
    
    # Assert
    assert user is not None
    assert user.email == email
    assert user.first_name == first_name
    assert user.last_name == last_name
    assert user.is_active is True
    assert user.role_id == roles[1].id
    assert user.hashed_password is not None
    assert user.hashed_password != password  # password should be hashed


@pytest.mark.asyncio
async def test_create_user_api(client: TestClient, tokens: dict):
    """Test user creation through API endpoint."""
    # Arrange
    admin_token = tokens["admin@example.com"]["access_token"]
    headers = {"Authorization": f"Bearer {admin_token}"}
    
    user_data = {
        "email": "apicreated@example.com",
        "password": "ApiUser@123",
        "first_name": "API",
        "last_name": "Created",
        "role_id": 2  # Regular user role
    }
    
    # Act
    response = client.post("/users/", headers=headers, json=user_data)
    
    # Assert
    assert response.status_code == status.HTTP_201_CREATED
    assert "id" in response.json()
    assert response.json()["email"] == "apicreated@example.com"
    assert response.json()["first_name"] == "API"
    assert response.json()["is_active"] is True
    assert "password" not in response.json()  # Password should not be returned


@pytest.mark.asyncio
async def test_create_user_duplicate_email(user_service: UserService, users: list[User], roles: list[Role]):
    """Test user creation with duplicate email."""
    # Arrange - Use an email that already exists
    existing_email = users[0].email
    
    # Act & Assert - Should raise exception
    with pytest.raises(Exception) as exc_info:
        await user_service.create_user(
            email=existing_email,
            password="Password@123",
            first_name="Duplicate",
            last_name="User",
            role_id=roles[1].id
        )
    
    # Check exception details
    assert "already exists" in str(exc_info.value).lower() or "duplicate" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_create_user_weak_password(client: TestClient, tokens: dict):
    """Test user creation with weak password that doesn't meet requirements."""
    # Arrange
    admin_token = tokens["admin@example.com"]["access_token"]
    headers = {"Authorization": f"Bearer {admin_token}"}
    
    # Test cases for weak passwords
    weak_passwords = [
        "short",  # Too short
        "nouppercase123",  # No uppercase
        "NOLOWERCASE123",  # No lowercase
        "NoSpecialChar123",  # No special characters
        "NoNumbers@abc"  # No numbers
    ]
    
    for password in weak_passwords:
        user_data = {
            "email": f"weakpass_{password}@example.com",
            "password": password,
            "first_name": "Weak",
            "last_name": "Password",
            "role_id": 2
        }
        
        # Act
        response = client.post("/users/", headers=headers, json=user_data)
        
        # Assert
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "password" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_create_user_invalid_email(client: TestClient, tokens: dict):
    """Test user creation with invalid email format."""
    # Arrange
    admin_token = tokens["admin@example.com"]["access_token"]
    headers = {"Authorization": f"Bearer {admin_token}"}
    
    invalid_emails = [
        "notanemail",
        "missing@tld",
        "@missinguser.com",
        "spaces in@email.com",
        "multiple@@at.com"
    ]
    
    for email in invalid_emails:
        user_data = {
            "email": email,
            "password": "ValidPass@123",
            "first_name": "Invalid",
            "last_name": "Email",
            "role_id": 2
        }
        
        # Act
        response = client.post("/users/", headers=headers, json=user_data)
        
        # Assert
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "email" in str(response.json()).lower()


@pytest.mark.asyncio
async def test_get_user_by_id(user_service: UserService, users: list[User]):
    """Test retrieving user by ID."""
    # Arrange
    user_id = users[1].id  # Use regular user
    
    # Act
    user = await user_service.get_user_by_id(user_id)
    
    # Assert
    assert user is not None
    assert user.id == user_id
    assert user.email == users[1].email


@pytest.mark.asyncio
async def test_get_user_by_email(user_service: UserService, users: list[User]):
    """Test retrieving user by email."""
    # Arrange
    user_email = users[1].email  # Use regular user email
    
    # Act
    user = await user_service.get_user_by_email(user_email)
    
    # Assert
    assert user is not None
    assert user.email == user_email
    assert user.id == users[1].id


@pytest.mark.asyncio
async def test_get_users(client: TestClient, tokens: dict, users: list[User]):
    """Test retrieving list of users."""
    # Arrange
    admin_token = tokens["admin@example.com"]["access_token"]
    headers = {"Authorization": f"Bearer {admin_token}"}
    
    # Act
    response = client.get("/users/", headers=headers)
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert isinstance(response.json(), list)
    assert len(response.json()) >= len(users)
    
    # Check if our test users are in the results
    user_emails = [user["email"] for user in response.json()]
    assert "user@example.com" in user_emails
    assert "admin@example.com" in user_emails


@pytest.mark.asyncio
async def test_get_users_pagination(client: TestClient, tokens: dict):
    """Test user pagination."""
    # Arrange
    admin_token = tokens["admin@example.com"]["access_token"]
    headers = {"Authorization": f"Bearer {admin_token}"}
    
    # Act - Get first page with limit=2
    response1 = client.get("/users/?skip=0&limit=2", headers=headers)
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert len(response1.json()) <= 2
    
    # Get second page
    response2 = client.get("/users/?skip=2&limit=2", headers=headers)
    
    # Check that pages have different users
    if len(response2.json()) > 0:
        page1_ids = [user["id"] for user in response1.json()]
        page2_ids = [user["id"] for user in response2.json()]
        assert not any(user_id in page1_ids for user_id in page2_ids)


@pytest.mark.asyncio
async def test_update_user(client: TestClient, tokens: dict, users: list[User]):
    """Test updating user details."""
    # Arrange
    admin_token = tokens["admin@example.com"]["access_token"]
    headers = {"Authorization": f"Bearer {admin_token}"}
    
    user_id = users[1].id  # Use regular user
    update_data = {
        "first_name": "Updated",
        "last_name": "UserName"
    }
    
    # Act
    response = client.patch(f"/users/{user_id}", headers=headers, json=update_data)
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["first_name"] == "Updated"
    assert response.json()["last_name"] == "UserName"
    assert response.json()["id"] == user_id
    
    # Verify update was persisted
    get_response = client.get(f"/users/{user_id}", headers=headers)
    assert get_response.json()["first_name"] == "Updated"


@pytest.mark.asyncio
async def test_update_user_email(client: TestClient, tokens: dict, users: list[User]):
    """Test updating user email."""
    # Arrange
    admin_token = tokens["admin@example.com"]["access_token"]
    headers = {"Authorization": f"Bearer {admin_token}"}
    
    user_id = users[2].id  # Use guest user
    update_data = {
        "email": "updated_email@example.com"
    }
    
    # Act
    response = client.patch(f"/users/{user_id}", headers=headers, json=update_data)
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["email"] == "updated_email@example.com"
    
    # Verify login works with new email
    login_data = {
        "email": "updated_email@example.com",
        "password": "Guest@123"  # Original password
    }
    login_response = client.post("/auth/login", json=login_data)
    assert login_response.status_code == status.HTTP_200_OK


@pytest.mark.asyncio
async def test_update_self(client: TestClient, tokens: dict):
    """Test users updating their own profile."""
    # Arrange
    user_token = tokens["user@example.com"]["access_token"]
    headers = {"Authorization": f"Bearer {user_token}"}
    
    update_data = {
        "first_name": "Self",
        "last_name": "Updated"
    }
    
    # Act
    response = client.patch("/users/me", headers=headers, json=update_data)
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["first_name"] == "Self"
    assert response.json()["last_name"] == "Updated"


@pytest.mark.asyncio
async def test_change_password(client: TestClient, tokens: dict):
    """Test password change functionality."""
    # Arrange
    user_token = tokens["user@example.com"]["access_token"]
    headers = {"Authorization": f"Bearer {user_token}"}
    
    password_data = {
        "current_password": "User@123",
        "new_password": "NewUserPass@456"
    }
    
    # Act
    response = client.post("/users/me/change-password", headers=headers, json=password_data)
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert "message" in response.json()
    assert "password updated" in response.json()["message"].lower()
    
    # Verify login works with new password
    login_data = {
        "email": "user@example.com",
        "password": "NewUserPass@456"
    }
    login_response = client.post("/auth/login", json=login_data)
    assert login_response.status_code == status.HTTP_200_OK
    
    # Verify old password no longer works
    old_login_data = {
        "email": "user@example.com",
        "password": "User@123"
    }
    old_login_response = client.post("/auth/login", json=old_login_data)
    assert old_login_response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_change_password_wrong_current(client: TestClient, tokens: dict):
    """Test password change with incorrect current password."""
    # Arrange
    user_token = tokens["user@example.com"]["access_token"]
    headers = {"Authorization": f"Bearer {user_token}"}
    
    password_data = {
        "current_password": "WrongCurrentPass",
        "new_password": "NewUserPass@456"
    }
    
    # Act
    response = client.post("/users/me/change-password", headers=headers, json=password_data)
    
    # Assert
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "current password" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_deactivate_user(client: TestClient, tokens: dict, users: list[User]):
    """Test deactivating a user account."""
    # Arrange
    admin_token = tokens["admin@example.com"]["access_token"]
    headers = {"Authorization": f"Bearer {admin_token}"}
    
    user_id = users[2].id  # Use guest user
    
    # Act
    response = client.post(f"/users/{user_id}/deactivate", headers=headers)
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert "is_active" in response.json()
    assert response.json()["is_active"] is False
    
    # Verify user can't login after deactivation
    login_data = {
        "email": users[2].email,
        "password": "Guest@123"
    }
    login_response = client.post("/auth/login", json=login_data)
    assert login_response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "inactive" in login_response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_activate_user(client: TestClient, tokens: dict, users: list[User]):
    """Test activating a deactivated user account."""
    # Arrange
    admin_token = tokens["admin@example.com"]["access_token"]
    headers = {"Authorization": f"Bearer {admin_token}"}
    
    # Use inactive user
    inactive_user_id = users[3].id
    
    # Act
    response = client.post(f"/users/{inactive_user_id}/activate", headers=headers)
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert "is_active" in response.json()
    assert response.json()["is_active"] is True
    
    # Verify user can login after activation
    login_data = {
        "email": users[3].email,
        "password": "Inactive@123"
    }
    login_response = client.post("/auth/login", json=login_data)
    assert login_response.status_code == status.HTTP_200_OK
    assert "access_token" in login_response.json()


@pytest.mark.asyncio
async def test_change_role(client: TestClient, tokens: dict, users: list[User]):
    """Test changing a user's role."""
    # Arrange
    admin_token = tokens["admin@example.com"]["access_token"]
    headers = {"Authorization": f"Bearer {admin_token}"}
    
    user_id = users[1].id  # Regular user
    update_data = {
        "role_id": 3  # Guest role
    }
    
    # Act
    response = client.patch(f"/users/{user_id}", headers=headers, json=update_data)
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["role_id"] == 3
    
    # Get the user details to confirm
    get_response = client.get(f"/users/{user_id}", headers=headers)
    assert get_response.json()["role_id"] == 3


@pytest.mark.asyncio
async def test_non_admin_cannot_change_roles(client: TestClient, tokens: dict, users: list[User]):
    """Test that non-admin users cannot change roles."""
    # Arrange
    user_token = tokens["user@example.com"]["access_token"]
    headers = {"Authorization": f"Bearer {user_token}"}
    
    # Try to change another user's role
    another_user_id = users[2].id  # Guest user
    update_data = {
        "role_id": 1  # Admin role
    }
    
    # Act
    response = client.patch(f"/users/{another_user_id}", headers=headers, json=update_data)
    
    # Assert
    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert "permission" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_delete_user(client: TestClient, tokens: dict, user_service: UserService):
    """Test deleting a user account."""
    # Arrange - Create a user to delete
    email = EmailStr("delete_me@example.com")
    user_to_delete = await user_service.create_user(
        email=email,
        password="Delete@123",
        first_name="Delete",
        last_name="Me",
        role_id=2  # Regular user role
    )
    
    admin_token = tokens["admin@example.com"]["access_token"]
    headers = {"Authorization": f"Bearer {admin_token}"}
    
    # Act
    response = client.delete(f"/users/{user_to_delete.id}", headers=headers)
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert "success" in response.json()["message"].lower()
    
    # Verify user cannot be found after deletion
    get_response = client.get(f"/users/{user_to_delete.id}", headers=headers)
    assert get_response.status_code == status.HTTP_404_NOT_FOUND
    
    # Verify login no longer works
    login_data = {
        "email": "delete_me@example.com",
        "password": "Delete@123"
    }
    login_response = client.post("/auth/login", json=login_data)
    assert login_response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_bulk_operations(client: TestClient, tokens: dict, users: list[User]):
    """Test bulk operations like deactivating multiple users."""
    # Arrange
    admin_token = tokens["admin@example.com"]["access_token"]
    headers = {"Authorization": f"Bearer {admin_token}"}
    
    # Get regular and guest user IDs
    user_ids = [users[1].id, users[2].id]
    
    # Act - Bulk deactivate
    response = client.post("/users/bulk/deactivate", headers=headers, json={"user_ids": user_ids})
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    assert "deactivated" in response.json()["message"].lower()
    assert len(response.json()["affected_users"]) == 2
    
    # Verify users are deactivated
    for user_id in user_ids:
        get_response = client.get(f"/users/{user_id}", headers=headers)
        assert get_response.json()["is_active"] is False


@pytest.mark.asyncio
async def test_role_based_access(client: TestClient, tokens: dict):
    """Test role-based access to endpoints."""
    # Test different endpoints with different roles
    
    # Endpoints mapping
    endpoints = {
        "admin_only": "/admin/settings",
        "user_can_access": "/users/me",
        "public": "/health"
    }
    
    # Admin should access all endpoints
    admin_token = tokens["admin@example.com"]["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    
    admin_responses = {
        endpoint: client.get(url, headers=admin_headers) 
        for endpoint, url in endpoints.items()
    }
    
    for endpoint, response in admin_responses.items():
        assert response.status_code != status.HTTP_403_FORBIDDEN, f"Admin should access {endpoint}"
    
    # Regular user should not access admin endpoints
    user_token = tokens["user@example.com"]["access_token"]
    user_headers = {"Authorization": f"Bearer {user_token}"}
    
    user_admin_response = client.get(endpoints["admin_only"], headers=user_headers)
    assert user_admin_response.status_code == status.HTTP_403_FORBIDDEN
    
    # Regular user should access user endpoints
    user_self_response = client.get(endpoints["user_can_access"], headers=user_headers)
    assert user_self_response.status_code == status.HTTP_200_OK
    
    # Public endpoint should be accessible without token
    public_response = client.get(endpoints["public"])
    assert public_response.status_code == status.HTTP_200_OK


@pytest.mark.asyncio
async def test_password_validation(user_service: UserService, roles: list[Role]):
    """Test password validation rules."""
    email = EmailStr("passwordtest@example.com")
    role_id = roles[1].id  # Regular user role
    
    # Test different invalid password scenarios
    invalid_passwords = [
        ("short", "too short"),
        ("onlylowercase123$", "no uppercase letter"),
        ("ONLYUPPERCASE123$", "no lowercase letter"),
        ("NoNumbers$", "no numbers"),
        ("NoSpecial123", "no special characters"),
        ("Admin12345", "common pattern"),
        ("pass word", "contains spaces")
    ]
    
    for password, reason in invalid_passwords:
        with pytest.raises(Exception, match=r".*[Pp]assword.*") as exc_info:
            await user_service.create_user(
                email=email,
                password=password,
                first_name="Password",
                last_name="Test",
                role_id=role_id
            )

