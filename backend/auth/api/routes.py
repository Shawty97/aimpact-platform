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
        form_data.

