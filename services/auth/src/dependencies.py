"""
Authentication Service Dependencies

This module provides dependency functions for the FastAPI application.
These are used for dependency injection in route handlers.
"""
from typing import Optional, Union

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import jwt
from jwt.exceptions import PyJWTError

from .config import settings
from .models import TokenData, User, UserRole
from .database import SessionLocal
from .services import UserService

# Create services
user_service = UserService()

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


def get_db():
    """
    Get database session. Dependency for route handlers.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Get current user from token. Dependency for route handlers.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode JWT token
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        
        # Extract user ID from token
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
        # Check if token is blacklisted
        jti = payload.get("jti")
        if jti and await user_service.is_token_blacklisted(jti, db):
            raise credentials_exception
        
        # Create token data
        token_data = TokenData(
            sub=user_id,
            email=payload.get("email"),
            roles=payload.get("roles", [UserRole.USER]),
            exp=payload.get("exp"),
            iat=payload.get("iat"),
            jti=jti
        )
    except PyJWTError:
        raise credentials_exception
    
    # Get user from database
    user = await user_service.get_user_by_id(token_data.sub, db)
    if user is None:
        raise credentials_exception
    
    return user


async def get_current_active_user(
    current_

