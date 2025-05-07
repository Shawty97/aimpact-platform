import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import jwt
from fastapi import Depends, HTTPException, status
from pydantic import EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from models.token import TokenBlacklist
from models.user import User
from core.config import settings
from db.session import get_db

logger = logging.getLogger(__name__)

class TokenService:
    """
    Service for handling JWT token creation, validation, and blacklisting
    """
    
    def __init__(self, db: AsyncSession = Depends(get_db)):
        self.db = db
        self.algorithm = settings.JWT_ALGORITHM
        self.secret_key = settings.JWT_SECRET_KEY
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_minutes = settings.REFRESH_TOKEN_EXPIRE_MINUTES
        self._blacklisted_tokens: Dict[str, datetime] = {}
        
    async def create_access_token(self, user_id: int, email: EmailStr, scopes: List[str] = None) -> str:
        """
        Create a new JWT access token
        
        Args:
            user_id: The user ID to include in the token
            email: The user email to include in the token
            scopes: Optional list of permission scopes to include in the token
            
        Returns:
            str: The encoded JWT access token
        """
        if scopes is None:
            scopes = []
            
        # Set token expiration time
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        # Create token payload
        to_encode = {
            "sub": str(user_id),
            "email": email,
            "scopes": scopes,
            "exp": expire.timestamp(),
            "iat": datetime.utcnow().timestamp(),
            "type": "access"
        }
        
        # Encode the JWT token
        try:
            encoded_jwt = jwt.encode(
                to_encode, 
                self.secret_key, 
                algorithm=self.algorithm
            )
            logger.debug(f"Created access token for user {user_id}")
            return encoded_jwt
        except Exception as e:
            logger.error(f"Error creating access token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create access token"
            )
    
    async def create_refresh_token(self, user_id: int, email: EmailStr) -> str:
        """
        Create a new JWT refresh token
        
        Args:
            user_id: The user ID to include in the token
            email: The user email to include in the token
            
        Returns:
            str: The encoded JWT refresh token
        """
        # Set token expiration time
        expire = datetime.utcnow() + timedelta(minutes=self.refresh_token_expire_minutes)
        
        # Create token payload
        to_encode = {
            "sub": str(user_id),
            "email": email,
            "exp": expire.timestamp(),
            "iat": datetime.utcnow().timestamp(),
            "type": "refresh"
        }
        
        # Encode the JWT token
        try:
            encoded_jwt = jwt.encode(
                to_encode, 
                self.secret_key, 
                algorithm=self.algorithm
            )
            logger.debug(f"Created refresh token for user {user_id}")
            return encoded_jwt
        except Exception as e:
            logger.error(f"Error creating refresh token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create refresh token"
            )
    
    async def decode_token(self, token: str) -> Dict:
        """
        Decode and validate a JWT token
        
        Args:
            token: The JWT token to decode and validate
            
        Returns:
            Dict: The decoded token payload
            
        Raises:
            HTTPException: If the token is invalid or expired
        """
        try:
            # Check if token is blacklisted
            if await self.is_token_blacklisted(token):
                logger.warning("Attempt to use blacklisted token")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Decode the token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Check token expiration
            if "exp" in payload and payload["exp"] < time.time():
                logger.warning("Attempt to use expired token")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return payload
        
        except jwt.PyJWTError as e:
            logger.warning(f"JWT validation error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    async def blacklist_token(self, token: str) -> None:
        """
        Add a token to the blacklist
        
        Args:
            token: The JWT token to blacklist
            
        Returns:
            None
        """
        try:
            # Decode token without verification to get expiration
            payload = jwt.decode(
                token,
                options={"verify_signature": False}
            )
            
            # Extract expiration time
            exp_timestamp = payload.get("exp", 0)
            expiration = datetime.fromtimestamp(exp_timestamp)
            
            # Get token identifier (hash of the token)
            token_id = hash(token)
            
            # Add to database
            new_blacklist_entry = TokenBlacklist(
                token_id=token_id,
                expires_at=expiration
            )
            
            self.db.add(new_blacklist_entry)
            await self.db.commit()
            
            # Also add to in-memory cache for faster lookups
            self._blacklisted_tokens[token] = expiration
            
            logger.info(f"Token blacklisted, expires at {expiration}")
            
        except Exception as e:
            logger.error(f"Error blacklisting token: {str(e)}")
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error blacklisting token"
            )
    
    async def is_token_blacklisted(self, token: str) -> bool:
        """
        Check if a token is blacklisted
        
        Args:
            token: The JWT token to check
            
        Returns:
            bool: True if token is blacklisted, False otherwise
        """
        # First check in-memory cache for faster response
        if token in self._blacklisted_tokens:
            # Check if token blacklist entry has expired
            if self._blacklisted_tokens[token] < datetime.utcnow():
                # Remove expired token from cache
                del self._blacklisted_tokens[token]
                return False
            return True
        
        # If not in cache, check database
        token_id = hash(token)
        
        # Query database for token
        stmt = select(TokenBlacklist).where(TokenBlacklist.token_id == token_id)
        result = await self.db.execute(stmt)
        blacklisted_token = result.scalar_one_or_none()
        
        if blacklisted_token:
            # Check if token has expired
            if blacklisted_token.expires_at < datetime.utcnow():
                # If expired, remove from blacklist to keep it clean
                await self.db.delete(blacklisted_token)
                await self.db.commit()
                return False
            
            # Add to cache for future lookups
            self._blacklisted_tokens[token] = blacklisted_token.expires_at
            return True
        
        return False
    
    async def purge_expired_blacklist_tokens(self) -> int:
        """
        Remove expired tokens from the blacklist database
        
        Returns:
            int: Number of expired tokens removed
        """
        try:
            # Get current time
            now = datetime.utcnow()
            
            # Find expired tokens
            stmt = select(TokenBlacklist).where(TokenBlacklist.expires_at < now)
            result = await self.db.execute(stmt)
            expired_tokens = result.scalars().all()
            
            # Delete expired tokens
            count = 0
            for token in expired_tokens:
                await self.db.delete(token)
                count += 1
            
            # Commit changes
            if count > 0:
                await self.db.commit()
                logger.info(f"Removed {count} expired tokens from blacklist")
            
            # Also clean up in-memory cache
            current_time = datetime.utcnow()
            expired_cache_tokens = [
                token for token, exp_time in self._blacklisted_tokens.items() 
                if exp_time < current_time
            ]
            
            for token in expired_cache_tokens:
                del self._blacklisted_tokens[token]
            
            return count
        
        except Exception as e:
            logger.error(f"Error purging blacklist: {str(e)}")
            await self.db.rollback()
            return 0

    async def validate_user_from_token(self, token: str) -> Optional[User]:
        """
        Validate token and return the corresponding user
        
        Args:
            token: JWT token to validate
            
        Returns:
            Optional[User]: The user associated with the token if valid, None otherwise
        """
        try:
            # Decode and validate the token
            payload = await self.decode_token(token)
            
            # Extract user ID from token
            user_id = int(payload.get("sub"))
            
            # Query database for user
            stmt = select(User).where(User.id == user_id)
            result = await self.db.execute(stmt)
            user = result.scalar_one_or_none()
            
            if not user:
                logger.warning(f"Token has valid format but user {user_id} not found")
                return None
            
            # Check if user is active
            if not user.is_active:
                logger.warning(f"User {user_id} is deactivated but attempted to use token")
                return None
            
            return user
            
        except HTTPException:
            # Already handled in decode_token
            return None
        except Exception as e:
            logger.error(f"Error validating user from token: {str(e)}")
            return None

