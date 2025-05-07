"""Dependencies for authentication API."""
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from backend.auth.models.user import User, UserRole
from backend.auth.services.auth_service import AuthService
from backend.auth.services.user_service import UserService


# Create services
user_service = UserService()
auth_service = AuthService(user_service)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/token")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get the current authenticated user."""
    return await auth_service.get_current_user(token)


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get the current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


def check_admin_role(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """Check if the current user has admin role."""
    if UserRole.ADMIN not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    return current_user


def check_role(role: UserRole):
    """Create a dependency to check for a specific role."""
    
    async def check_user_role(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        if role not in current_user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {role} required"
            )
        return current_user
    
    return check_user_role

