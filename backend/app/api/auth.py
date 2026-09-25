"""
Authentication API Router and Dependencies.
Provides /api/auth/register, /api/auth/login, /api/auth/me,
and reusable FastAPI dependencies get_current_user and get_current_user_id.
"""

import re
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field, field_validator

from app.auth.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
)
from app.db import repository
from app.utils.logger import setup_logger

logger = setup_logger("api.auth")

router = APIRouter(prefix="/api/auth", tags=["Authentication"])
security = HTTPBearer(auto_error=False)


# ==============================================================================
# SCHEMAS
# ==============================================================================

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


class UserRegisterRequest(BaseModel):
    email: str = Field(..., description="Valid user email address")
    password: str = Field(..., min_length=6, description="User password (minimum 6 characters)")

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        trimmed = v.strip().lower()
        if not EMAIL_REGEX.match(trimmed):
            raise ValueError("Invalid email format.")
        return trimmed

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        trimmed = v.strip()
        if len(trimmed) < 6:
            raise ValueError("Password must be at least 6 characters long.")
        return trimmed


class UserLoginRequest(BaseModel):
    email: str = Field(..., description="Registered user email address")
    password: str = Field(..., description="User password")

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        trimmed = v.strip().lower()
        if not EMAIL_REGEX.match(trimmed):
            raise ValueError("Invalid email format.")
        return trimmed

    @field_validator("password")
    @classmethod
    def validate_login_password(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Password cannot be empty.")
        return trimmed



class UserResponse(BaseModel):
    id: str
    email: str
    created_at: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ==============================================================================
# FASTAPI DEPENDENCIES
# ==============================================================================

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    """
    FastAPI dependency that extracts and validates the Bearer JWT token,
    returning the authenticated user record from the SQLite database.
    Raises HTTP 401 if missing, invalid, or expired.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials.strip()
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload["sub"]
    user = repository.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_current_user_id(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> str:
    """
    Convenience dependency returning the string ID of the authenticated user.
    """
    return str(current_user["id"])


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
def register_user(request: UserRegisterRequest) -> UserResponse:
    """
    Create a new user account.
    Enforces unique email check and stores a secure bcrypt password hash.
    """
    clean_email = request.email.strip().lower()

    # Check for existing account
    existing = repository.get_user_by_email(clean_email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email address already exists.",
        )

    try:
        clean_password = request.password.strip()
        pw_hash = hash_password(clean_password)
        user_record = repository.create_user(
            email=clean_email,
            password_hash=pw_hash,
        )
        logger.info(f"Successfully registered user: {clean_email} ({user_record['id']})")
        return UserResponse(
            id=user_record["id"],
            email=user_record["email"],
            created_at=user_record["created_at"],
        )
    except Exception as e:
        logger.error(f"Error registering user '{clean_email}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user account.",
        )


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Authenticate user and obtain JWT access token",
)
def login_user(request: UserLoginRequest) -> TokenResponse:
    """
    Authenticate user credentials and return a signed JWT access token.
    """
    clean_email = request.email.strip().lower()
    clean_password = request.password.strip()
    user_record = repository.get_user_by_email(clean_email)

    # Use constant-time or safe comparison, do not leak user existence
    if not user_record or not verify_password(clean_password, user_record.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user_record["id"], "email": user_record["email"]}
    )

    logger.info(f"User logged in successfully: {clean_email} ({user_record['id']})")
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            id=user_record["id"],
            email=user_record["email"],
            created_at=user_record["created_at"],
        ),
    )


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get current authenticated user profile",
)
def get_current_user_profile(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> UserResponse:
    """
    Return the profile details of the currently authenticated user.
    """
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        created_at=current_user["created_at"],
    )
