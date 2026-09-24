"""
Tests for Authentication Core, Database Migration, and User Management (Phase 1 & Phase 2).
Validates password hashing, JWT signing/verification, database migration, and auth API endpoints.
"""

from datetime import timedelta
import pytest
from fastapi.testclient import TestClient

from app.api.routes import app
from app.auth.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
)
from app.db import repository, database, init_db


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    """Provide an isolated test database with fresh schema initialization."""
    test_db = tmp_path / "test_auth.db"
    init_db(test_db)

    orig_get_conn = database.get_db_connection

    def mock_get_conn(custom_path=None):
        return orig_get_conn(custom_path or test_db)

    monkeypatch.setattr(database, "get_db_connection", mock_get_conn)
    monkeypatch.setattr(repository, "get_db_connection", mock_get_conn)

    return test_db


@pytest.fixture
def client(auth_db):
    """Provide a TestClient connected to the isolated auth_db."""
    return TestClient(app)


# ==============================================================================
# 1. SECURITY & CRYPTO UNIT TESTS
# ==============================================================================

def test_password_hashing():
    """Verify bcrypt hashes are salted, unique per call, and verify correctly."""
    pwd = "MySecurePassword123!"
    h1 = hash_password(pwd)
    h2 = hash_password(pwd)

    # Different salts produce different hashes
    assert h1 != h2
    assert h1.startswith("$2b$") or h1.startswith("$2a$")

    # Correct password verifies
    assert verify_password(pwd, h1) is True
    assert verify_password(pwd, h2) is True

    # Wrong password fails
    assert verify_password("WrongPassword123!", h1) is False
    assert verify_password("", h1) is False
    assert verify_password(pwd, "") is False


def test_jwt_token_flow():
    """Verify JWT access token creation, claims encoding, and decoding."""
    claims = {"sub": "usr_test123", "email": "test@example.com"}
    token = create_access_token(claims, expires_delta=timedelta(minutes=15))

    decoded = decode_access_token(token)
    assert decoded is not None
    assert decoded["sub"] == "usr_test123"
    assert decoded["email"] == "test@example.com"
    assert "exp" in decoded
    assert "iat" in decoded


def test_jwt_expired_token():
    """Verify expired JWT tokens are rejected cleanly."""
    claims = {"sub": "usr_expired", "email": "expired@example.com"}
    expired_token = create_access_token(claims, expires_delta=timedelta(seconds=-10))

    assert decode_access_token(expired_token) is None


def test_jwt_invalid_token():
    """Verify malformed or forged tokens return None."""
    assert decode_access_token("not-a-valid-jwt-token") is None
    assert decode_access_token("") is None


# ==============================================================================
# 2. DATABASE REPOSITORY & MIGRATION TESTS
# ==============================================================================

def test_database_migration_legacy_user(auth_db):
    """Verify init_db creates users table and sets up legacy_user automatically."""
    legacy = repository.get_user_by_id("legacy_user")
    assert legacy is not None
    assert legacy["id"] == "legacy_user"
    assert legacy["email"] == "legacy@local.dev"


def test_create_and_get_user(auth_db):
    """Verify user creation with email normalization and lookup by email and ID."""
    pw_hash = hash_password("secret123")
    user = repository.create_user(
        email="Alice.Smith@Example.COM",
        password_hash=pw_hash,
    )

    assert user["id"].startswith("usr_")
    # Email should be normalized to lowercase
    assert user["email"] == "alice.smith@example.com"
    assert user["created_at"]

    # Lookup by case-insensitive email
    by_email = repository.get_user_by_email("ALICE.SMITH@example.com")
    assert by_email is not None
    assert by_email["id"] == user["id"]

    # Lookup by ID
    by_id = repository.get_user_by_id(user["id"])
    assert by_id is not None
    assert by_id["email"] == "alice.smith@example.com"


def test_user_email_uniqueness(auth_db):
    """Verify unique constraint prevents duplicate user registration in repository."""
    repository.create_user("bob@example.com", hash_password("pass123"))

    with pytest.raises(Exception):
        repository.create_user("bob@example.com", hash_password("pass456"))


# ==============================================================================
# 3. FASTAPI AUTH ENDPOINTS TESTS
# ==============================================================================

def test_register_endpoint_success(client):
    """Test POST /api/auth/register creates a new user."""
    payload = {
        "email": "Carol@domain.org",
        "password": "StrongPassword2026!",
    }
    res = client.post("/api/auth/register", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["id"].startswith("usr_")
    assert data["email"] == "carol@domain.org"
    assert "password_hash" not in data
    assert "password" not in data
    assert data["created_at"]


def test_register_duplicate_email(client):
    """Test POST /api/auth/register with existing email returns 409 Conflict."""
    payload = {
        "email": "duplicate@test.com",
        "password": "Password123!",
    }
    res1 = client.post("/api/auth/register", json=payload)
    assert res1.status_code == 201

    # Second registration with same email
    res2 = client.post("/api/auth/register", json=payload)
    assert res2.status_code == 409
    assert "already exists" in res2.json()["detail"].lower()


def test_register_validation_errors(client):
    """Test POST /api/auth/register validation: malformed email and short password."""
    # Malformed email
    res1 = client.post("/api/auth/register", json={"email": "not-an-email", "password": "validPassword123"})
    assert res1.status_code == 422

    # Short password (< 6 chars)
    res2 = client.post("/api/auth/register", json={"email": "valid@test.com", "password": "123"})
    assert res2.status_code == 422


def test_login_success(client):
    """Test POST /api/auth/login returns valid JWT access token."""
    # Register user first
    reg_payload = {"email": "david@example.com", "password": "MyPassword123"}
    client.post("/api/auth/register", json=reg_payload)

    # Login
    login_payload = {"email": "david@example.com", "password": "MyPassword123"}
    res = client.post("/api/auth/login", json=login_payload)
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "david@example.com"
    assert data["user"]["id"].startswith("usr_")


def test_login_invalid_password(client):
    """Test POST /api/auth/login with incorrect password returns 401."""
    client.post("/api/auth/register", json={"email": "eve@example.com", "password": "CorrectPassword123"})

    res = client.post("/api/auth/login", json={"email": "eve@example.com", "password": "WrongPassword"})
    assert res.status_code == 401
    assert "incorrect" in res.json()["detail"].lower()


def test_login_nonexistent_user(client):
    """Test POST /api/auth/login with unknown email returns 401."""
    res = client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "AnyPassword123"})
    assert res.status_code == 401
    assert "incorrect" in res.json()["detail"].lower()


def test_get_me_endpoint_authenticated(client):
    """Test GET /api/auth/me with valid Bearer token returns current user profile."""
    # Register & Login
    client.post("/api/auth/register", json={"email": "frank@example.com", "password": "FrankPassword123"})
    login_res = client.post("/api/auth/login", json={"email": "frank@example.com", "password": "FrankPassword123"})
    token = login_res.json()["access_token"]

    # Call /api/auth/me
    headers = {"Authorization": f"Bearer {token}"}
    res = client.get("/api/auth/me", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["email"] == "frank@example.com"
    assert data["id"].startswith("usr_")


def test_get_me_endpoint_unauthenticated(client):
    """Test GET /api/auth/me without token returns 401."""
    res = client.get("/api/auth/me")
    assert res.status_code == 401
    assert "not provided" in res.json()["detail"].lower()


def test_get_me_endpoint_invalid_token(client):
    """Test GET /api/auth/me with bad token returns 401."""
    headers = {"Authorization": "Bearer totally-fake-token-string"}
    res = client.get("/api/auth/me", headers=headers)
    assert res.status_code == 401
    assert "invalid or expired" in res.json()["detail"].lower()
