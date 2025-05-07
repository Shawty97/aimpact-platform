"""
Authentication Service - Auth Service

This module provides the core authentication functionality,
including user authentication and token generation.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import jwt
from fastapi import Depends
from sqlalchemy.orm import Session

from ..config import settings
from ..models import User, Token, TokenData
from .user_service import UserService
from .token_service import TokenService

logger = logging.getLogger("auth_service.auth")


class AuthService:
    """Service for handling authentication operations."""
    
    def __init__(self, user_service: UserService):
        """Initialize with user service."""
        self.user_service = user_service
        self.token_service = TokenService()
    
    async def authenticate_user(
        self, username_or_email: str, password: str, db: Session
    ) -> Optional[User]:
        """
        Authenticate a user by username/email and password.
        
        Args:
            username_or_email: Username or email to authenticate
            password: Plain text password
            db: Database session
            
        Returns:
            User object if authentication successful, None otherwise
        """
        logger.debug(f"Authenticating user: {username_or_email}")
        
        # Try to find user by email
        user = await self.user_service.get_user_by_email(username_or_email, db)
        
        # If not found, try username
        if not user:
            user = await self.user_service.get_user_by_username(username_or_email, db)
        
        # If still not found or password doesn't match, return None
        if not user:
            logger.warning(f"User not found: {username_or_email}")
            return None
        
        if not self.user_service.verify_password(password, user.hashed_password):
            logger.warning(f"Invalid password for user: {username_or_email}")
            return None
        
        # Convert to User model from UserInDB
        return await self.user_service.get_user_model(user, db)
    
    def create_access_token(self, user: User) -> Token:
        """
        Create a JWT access token for the user.
        
        Args:
            user: User object to create token for
            
        Returns:
            Token object with access token and expiry
        """
        now = datetime.utcnow()
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        # Create JWT payload
        token_data = TokenData(
            sub=user.id,
            email=user.email,
            roles=user.roles,
            exp=int(expire.timestamp()),
            iat=int(now.timestamp()),
            jti=f"{user.id}:{now.timestamp()}"
        )
        
        # Create JWT token
        jwt_token = jwt.encode(
            token_data.dict(),
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )
        
        return Token(
            access_token=jwt_token,
            token_type="bearer",
            expires_at=int(expire.timestamp())
        )
    
    async def blacklist_token(self, token: str, user_id: str, db: Session) -> bool:
        """
        Blacklist a token to prevent further use.
        
        Args:
            token: JWT token to blacklist
            user_id: ID of user who owns the token
            db: Database session
            
        Returns:
            True if token was blacklisted, False otherwise
        """
        try:
            # Decode token to get jti and expiry
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM],
                options={"verify_exp": False}  # Allow expired tokens to be blacklisted
            )
            
            jti = payload.get("jti")
            exp = payload.get("exp")
            
            if not jti or not exp:
                logger.warning("Token missing jti or exp claims")
                return False
            
            # Add token to blacklist
            return await self.token_service.blacklist_token(jti, user_id, exp, db)
            
        except jwt.PyJWTError as e:
            logger.error(f"Error decoding token for blacklisting: {e}")
            return False

