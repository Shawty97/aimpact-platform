"""
Authentication Service - User Service

This module provides user management functionality,
including user creation, retrieval, and updates.
"""
import logging
import uuid
from datetime import datetime
from typing import List, Optional

from passlib.context import CryptContext
from sqlalchemy.orm import Session

from ..models import UserDB, User, UserCreate, UserUpdate, UserRole, UserBase

logger = logging.getLogger("auth_service.user")


class UserService:
    """Service for user management operations."""
    
    # Password context for hashing and verification
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against a hash.
        
        Args:
            plain_password: Plain text password
            hashed_password: Hashed password
            
        Returns:
            True if password matches hash, False otherwise
        """
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """
        Generate a password hash.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        return self.pwd_context.hash(password)
    
    async def create_user(self, user_create: UserCreate, db: Session) -> User:
        """
        Create a new user.
        
        Args:
            user_create: User creation data
            db: Database session
            
        Returns:
            Created user
            
        Raises:
            ValueError: If user with email or username already exists
        """
        # Check if user with email already exists
        existing_user = await self.get_user_by_email(user_create.email, db)
        if existing_user:
            logger.warning(f"User with email already exists: {user_create.email}")
            raise ValueError("User with this email already exists")
        
        # Check if user with username already exists
        existing_user = await self.get_user_by_username(user_create.username, db)
        if existing_user:
            logger.warning(f"User with username already exists: {user_create.username}")
            raise ValueError("User with this username already exists")
        
        # Create user ID
        user_id = str(uuid.uuid4())
        
        # Get current timestamp
        now = datetime.utcnow()
        
        # Create user in database
        db_user = UserDB(
            id=user_id,
            email=user_create.email,
            username=user_create.username,
            hashed_password=self.get_password_hash(user_create.password),
            full_name=user_create.full_name,
            is_active=user_create.is_active,
            created_at=now,
            updated_at=now,
        )
        
        # Add user to database
        db.add(db_user)
        
        # Add roles
        for role in user_create.roles:
            db.execute(
                f"INSERT INTO user_roles (user_id, role) VALUES ('{user_id}', '{role}')"
            )
        
        # Commit changes
        db.commit()
        db.refresh(db_user)
        
        logger.info(f"Created user: {user_id} ({user_create.email})")
        
        # Return user model
        return User(
            id=user_id,
            email=user_create.email,
            username=user_create.username,
            full_name=user_create.full_name,
            is_active=user_create.is_active,
            roles=user_create.roles,
            created_at=now,
            updated_at=now,
        )
    
    async def get_user_by_id(self, user_id: str, db: Session) -> Optional[UserDB]:
        """
        Get a user by ID.
        
        Args:
            user_id: User ID
            db: Database session
            
        Returns:
            User object if found, None otherwise
        """
        return db.query(UserDB).filter(UserDB.id == user_id).first()
    
    async def get_user_by_email(self, email: str, db: Session) -> Optional[UserDB]:
        """
        Get a user by email.
        
        Args:
            email: User email
            db: Database session
            
        Returns:
            User object if found, None otherwise
        """
        return db.query(UserDB).filter(UserDB.email == email).first()
    
    async def get_user_by_username(self, username: str, db: Session) -> Optional[UserDB]:
        """
        Get a user by username.
        
        Args:
            username: Username
            db: Database session
            
        Returns:
            User object if found, None otherwise
        """
        return db.query(UserDB).filter(UserDB.username == username).first()
    
    async def get_user_model(self, db_user: UserDB, db: Session) -> User:
        """
        Convert a UserDB object to a User model.
        
        Args:
            db_user: UserDB object
            db: Database session
            
        Returns:
            User model
        """
        # Get user roles from database
        roles = db.execute(
            f"SELECT role FROM user_roles WHERE user_id = '{db_user.id}'"
        ).fetchall()
        
        # Convert to UserRole enum values
        user_roles = [UserRole(role[0]) for role in roles]
        
        # Create and return User model
        return User(
            id=db_user.id,
            email=db_user.email,
            username=db_user.username,
            full_name=db_user.full_name,
            is_active=db_user.is_active,
            roles=user_roles,
            created_at=db_user.created_at,
            updated_at=db_user.updated_at,
        )
    
    async def update_user(
        self, user_id: str, user_update: UserUpdate, db: Session
    ) -> Optional[User]:
        """
        Update a user.
        
        Args:
            user_id: User ID
            user_update: User update data
            db: Database session
            
        Returns:
            Updated user if successful, None if user not found
            
        Raises:
            ValueError: If email or username already exists for another user
        """
        # Get user from database
        db_user = await self.get_user_by_id(user_id, db)
        if not db_user:
            logger.warning(f"User not found for update: {user_id}")
            return None
        
        # Check if email is being updated and already exists
        if user_update.email and user_update.email != db_user.email:
            existing_user = await self.get_user_by_email(user_update.email, db)
            if existing_user and existing_user.id != user_id:
                logger.warning(f"Email already in use: {user_update.email}")
                raise ValueError("Email already in use")
        
        # Check if username is being updated and already exists
        if user_update.username and user_update.username != db_user.username:
            existing_user = await self.get_user_by_username(user_update.username, db)
            if existing_user and existing_user.id != user_id:
                logger.warning(f"Username already in use: {user_update.username}")
                raise ValueError("Username already in use")
        
        # Update user fields
        if user_update.email:
            db_user.email = user_update.email
        if user_update.username:
            db_user.username = user_update.username
        if user_update.full_name is not None:
            db_user.full_name = user_update.full_name
        if user_update.is_active is not None:
            db_user.is_active = user_update.is_active
        if user_update.password:
            db_user.hashed_password = self.get_password_hash(user_update.password)
        
        # Update timestamp
        db_user.updated_at = datetime.utcnow()
        
        # Update roles if provided
        if user_update.roles is not None:
            # Delete existing roles
            db.execute(f"DELETE FROM user_roles WHERE user_i

