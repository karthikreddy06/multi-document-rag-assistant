"""
Authentication and Cryptographic Security Module.
Provides bcrypt password hashing and PyJWT token generation/decoding.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional
import bcrypt
import jwt

from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("auth.security")


def hash_password(password: str) -> str:
    """
    Hash a plaintext password using bcrypt with an automatically generated salt.
    """
    if not password:
        raise ValueError("Password cannot be empty.")
    pwd_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Safely verify a plaintext password against a stored bcrypt hash.
    Returns False on mismatch, invalid format, or empty input.
    """
    if not plain_password or not hashed_password:
        return False
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception as e:
        logger.warning(f"Password verification encountered an error: {e}")
        return False


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Generate a signed JWT access token containing claims and an expiration time.
    """
    if not settings.jwt_secret_key:
        raise RuntimeError("JWT_SECRET_KEY is not configured in the environment.")

    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.jwt_access_token_expire_minutes)
    )
    to_encode.update({"exp": expire, "iat": now})
    token = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    return token


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate a JWT access token.
    Returns the decoded claims dictionary if valid, or None if expired or invalid.
    """
    if not settings.jwt_secret_key:
        raise RuntimeError("JWT_SECRET_KEY is not configured in the environment.")

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.debug("JWT token has expired.")
        return None
    except jwt.InvalidTokenError as e:
        logger.debug(f"Invalid JWT token: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error decoding JWT: {e}")
        return None
