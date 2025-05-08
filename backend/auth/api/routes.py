"""API routes for authentication."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from backend.auth.api.dependencies import (
    auth_service,
    get_current_active_user,
    check_admin_role,
    user_service
)
from backend.auth.models.token import Token
from backend.auth.models.user import User, UserCreate, UserRole


router = APIRouter()


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """Login endpoint to get access token."""
    user = await auth_service.authenticate_user(
        form_data.username,  # OAuth2 uses username field for email
        form_data.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account",
        )
        
    return auth_service.create_access_token(user)


@router.post("/register", response_model=User)
async def register_user(user_create: UserCreate):
    """Register a new user."""
    try:
        user = await user_service.create_user(user_create)
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/me", response_model=User)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get the current user information."""
    return current_user


@router.post("/users/{user_id}/roles/{role}", response_model=User)
async def add_user_role(
    user_id: str,
    role: UserRole,
    admin_user: User = Depends(check_admin_role)
):
    """Add a role to a user (admin only)."""
    user = await user_service.add_role(user_id, role)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


@router.delete("/users/{user_id}/roles/{role}", response_model=User)
async def remove_user_role(
    user_id: str,
    role: UserRole,
    admin_user: User = Depends(check_admin_role)
):
    """Remove a role from a user (admin only)."""
    user = await user_service.remove_role(user_id, role)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user
