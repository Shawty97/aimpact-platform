import asyncio
import os
from datetime import datetime, timedelta
from typing import AsyncGenerator, Dict, Generator, List

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import EmailStr
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from core.config import settings
from db.base import Base
from db.session import get_db
from models.role import Role
from models.user import User
from services.auth_service import AuthService
from services.user_service import UserService
from services.token_service import TokenService

# Use an in-memory SQLite database for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


# Create a new test settings
def get_test_settings():
    test_settings = settings.copy()
    test_settings.SQLALCHEMY_DATABASE_URI = TEST_DATABASE_URL
    test_settings.JWT_SECRET_KEY = "test_secret_key"
    test_settings.JWT_ALGORITHM = "HS256"
    test_settings.ACCESS_TOKEN_EXPIRE_MINUTES = 15
    test_settings.REFRESH_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day
    test_settings.ENVIRONMENT = "test"
    return test_settings


# Override settings for testing
@pytest.fixture(scope="session")
def test_settings():
    return get_test_settings()


# Create test async engine
@pytest.fixture(scope="session")
def engine():
    engine = create_async_engine(
        TEST_DATABASE_URL,
        poolclass=NullPool,
    )
    yield engine
    asyncio.run(engine.dispose())


# Create test session
@pytest_asyncio.fixture(scope="function")
async def db_session(engine) -> AsyncGenerator[AsyncSession, None]:
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create session
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session
        # Clean up after test
        await session.rollback()

    # Drop all tables after test
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# Dependency override for database session
@pytest.fixture(scope="function")
def override_get_db(db_session: AsyncSession):
    async def _override_get_db():
        try:
            yield db_session
        finally:
            pass

    return _override_get_db


# Create FastAPI test application
@pytest.fixture(scope="function")
def app(override_get_db):
    from main import app as fastapi_app

    # Override the dependency
    fastapi_app.dependency_overrides[get_db] = override_get_db
    return fastapi_app


# Create test client
@pytest.fixture(scope="function")
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client


# Create token service
@pytest.fixture(scope="function")
def token_service(db_session: AsyncSession) -> TokenService:
    return TokenService(db=db_session)


# Create user service
@pytest.fixture(scope="function")
def user_service(db_session: AsyncSession) -> UserService:
    return UserService(db=db_session)


# Create auth service
@pytest.fixture(scope="function")
def auth_service(db_session: AsyncSession, token_service: TokenService, user_service: UserService) -> AuthService:
    return AuthService(db=db_session, token_service=token_service, user_service=user_service)


# Create test roles
@pytest_asyncio.fixture(scope="function")
async def roles(db_session: AsyncSession) -> List[Role]:
    # Create basic roles
    roles = [
        Role(name="admin", description="Administrator with full access"),
        Role(name="user", description="Regular user with limited access"),
        Role(name="guest", description="Guest user with minimal access"),
    ]
    
    # Add roles to database
    for role in roles:
        db_session.add(role)
    
    await db_session.commit()
    
    # Refresh roles to get IDs
    for role in roles:
        await db_session.refresh(role)
    
    return roles


# Create test users
@pytest_asyncio.fixture(scope="function")
async def users(db_session: AsyncSession, roles: List[Role], user_service: UserService) -> List[User]:
    # Create test users with different roles
    admin_user = await user_service.create_user(
        email=EmailStr("admin@example.com"),
        password="Admin@123",
        first_name="Admin",
        last_name="User",
        role_id=roles[0].id  # admin role
    )
    
    regular_user = await user_service.create_user(
        email=EmailStr("user@example.com"),
        password="User@123",
        first_name="Regular",
        last_name="User",
        role_id=roles[1].id  # user role
    )
    
    guest_user = await user_service.create_user(
        email=EmailStr("guest@example.com"),
        password="Guest@123",
        first_name="Guest",
        last_name="User",
        role_id=roles[2].id  # guest role
    )
    
    inactive_user = await user_service.create_user(
        email=EmailStr("inactive@example.com"),
        password="Inactive@123",
        first_name="Inactive",
        last_name="User",
        role_id=roles[1].id,  # user role
        is_active=False
    )
    
    return [admin_user, regular_user, guest_user, inactive_user]


# Create test tokens
@pytest_asyncio.fixture(scope="function")
async def tokens(db_session: AsyncSession, users: List[User], token_service: TokenService) -> Dict[str, Dict[str, str]]:
    # Create tokens for each user
    all_tokens = {}
    
    for user in users:
        if user.is_active:  # Only create tokens for active users
            access_token = await token_service.create_access_token(
                user_id=user.id,
                email=user.email,
                scopes=["read", "write"] if user.role.name == "admin" else ["read"]
            )
            
            refresh_token = await token_service.create_refresh_token(
                user_id=user.id,
                email=user.email
            )
            
            all_tokens[user.email.lower()] = {
                "access_token": access_token,
                "refresh_token": refresh_token
            }
    
    return all_tokens


# Fixture for expired tokens
@pytest_asyncio.fixture(scope="function")
async def expired_token(db_session: AsyncSession, users: List[User]) -> str:
    # Create an expired token for testing
    active_user = users[1]  # regular user
    
    # Create payload with past expiration date
    expired_time = datetime.utcnow() - timedelta(minutes=30)
    
    import jwt
    payload = {
        "sub": str(active_user.id),
        "email": str(active_user.email),
        "exp": expired_time.timestamp(),
        "iat": (expired_time - timedelta(minutes=30)).timestamp(),
        "type": "access"
    }
    
    # Create token with test secret key
    expired_token = jwt.encode(
        payload,
        get_test_settings().JWT_SECRET_KEY,
        algorithm=get_test_settings().JWT_ALGORITHM
    )
    
    return expired_token


# Fixture for blacklisted token
@pytest_asyncio.fixture(scope="function")
async def blacklisted_token(db_session: AsyncSession, tokens: Dict, token_service: TokenService) -> str:
    # Get a valid token
    user_email = "user@example.com"
    token = tokens[user_email]["access_token"]
    
    # Blacklist it
    await token_service.blacklist_token(token)
    
    return token

