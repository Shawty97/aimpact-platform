"""User service for user management operations."""
from datetime import datetime
from typing import List, Optional
import uuid

from backend.auth.models.user import User, UserCreate, UserInDB, UserRole
from backend.auth.services.password_service import get_password_hash


class UserService:
    """Service for user management operations."""
    
    # This would be replaced with a database connection in production
    _users_db = {}  # Temporary in-memory storage
    
    async def create_user(self, user_create: UserCreate) -> User:
        """Create a new user."""
        # Check if user already exists
        existing_user = await self.get_user_by_email(user_create.email)
        if existing_user:
            raise ValueError("User with this email already exists")
        
        # Create new user
        now = datetime.utcnow()
        user_id = str(uuid.uuid4())
        
        user_in_db = UserInDB(
            _id=user_id,
            hashed_password=get_password_hash(user_create.password),
            created_at=now,
            updated_at=now,
            **user_create.dict(exclude={"password"})
        )
        
        # Store user in database (temporary in-memory storage)
        self._users_db[user_id] = user_in_db
        
        return User(
            id=user_id,
            created_at=now,
            **user_create.dict(exclude={"password"})
        )
    
    async def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        """Get a user by ID."""
        return self._users_db.get(user_id)
    
    async def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        """Get a user by email."""
        for user in self._users_db.values():
            if user.email == email:
                return user
        return None
    
    async def update_user(self, user_id: str, **kwargs) -> Optional[User]:
        """Update a user."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return None
        
        # Update user fields
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        user.updated_at = datetime.utcnow()
        self._users_db[user_id] = user
        
        return User(
            id=user_id,
            created_at=user.created_at,
            **user.dict(exclude={"hashed_password", "_id"})
        )
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        if user_id in self._users_db:
            del self._users_db[user_id]
            return True
        return False
    
    async def add_role(self, user_id: str, role: UserRole) -> Optional[User]:
        """Add a role to a user."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return None
        
        if role not in user.roles:
            user.roles.append(role)
            user.updated_at = datetime.utcnow()
            self._users_db[user_id] = user
        
        return User(
            id=user_id,
            created_at=user.created_at,
            **user.dict(exclude={"hashed_password", "_id"})
        )
    
    async def remove_role(self, user_id: str, role: UserRole) -> Optional[User]:
        """Remove a role from a user."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return None
        
        if role in user.roles:
            user.roles.remove(role)
            user.updated_at = datetime.utcnow()
            self._users_db[user_id] = user
        
        return User(
            id=user_id,
            created_at=user.created_at,
            **user.dict(exclude={"hashed_password", "_id"})
        )

