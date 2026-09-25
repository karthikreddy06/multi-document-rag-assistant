"""
Production Database Persistence and Authentication Consistency Test Suite.
Validates:
1. Register -> Logout -> Login flow
2. Email case normalization
3. Surrounding password whitespace consistency (preserving internal spaces)
4. Duplicate exact email registration rejection (409 Conflict)
5. Database path configuration (file and directory handling, parent creation)
6. Existing users remaining accessible after restart/init_db
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.api.routes import app
from app.auth.security import hash_password, verify_password
from app.db import repository, database, init_db
from app.db.database import get_db_path


@pytest.fixture
def test_db_path(tmp_path, monkeypatch):
    """Provide an isolated, persistent SQLite database path for testing."""
    db_file = tmp_path / "persistent_test.db"
    init_db(db_file)

    orig_get_conn = database.get_db_connection

    def mock_get_conn(custom_path=None):
        return orig_get_conn(custom_path or db_file)

    monkeypatch.setattr(database, "get_db_connection", mock_get_conn)
    monkeypatch.setattr(repository, "get_db_connection", mock_get_conn)

    return db_file


@pytest.fixture
def client(test_db_path):
    """Provide a TestClient connected to the isolated test database."""
    return TestClient(app)


# ==============================================================================
# 1. Register -> Logout -> Login Flow
# ==============================================================================

def test_register_logout_login_flow(client):
    """Test full cycle: register account, log out (clear session), log back in."""
    reg_payload = {
        "email": "persist_user@example.com",
        "password": "SecurePassword2026!",
    }
    # 1. Register
    reg_res = client.post("/api/auth/register", json=reg_payload)
    assert reg_res.status_code == 201
    user_data = reg_res.json()
    user_id = user_data["id"]
    assert user_data["email"] == "persist_user@example.com"

    # 2. Simulate Logout (client discards access token)
    # 3. Login with the same credentials
    login_payload = {
        "email": "persist_user@example.com",
        "password": "SecurePassword2026!",
    }
    login_res = client.post("/api/auth/login", json=login_payload)
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    assert token_data["user"]["id"] == user_id
    assert token_data["user"]["email"] == "persist_user@example.com"

    # 4. Access protected profile using token
    headers = {"Authorization": f"Bearer {token_data['access_token']}"}
    me_res = client.get("/api/auth/me", headers=headers)
    assert me_res.status_code == 200
    me_data = me_res.json()
    assert me_data["id"] == user_id
    assert me_data["email"] == "persist_user@example.com"


# ==============================================================================
# 2. Email Case Normalization
# ==============================================================================

def test_email_case_normalization(client):
    """Verify that email registration and login are case-insensitive and normalized."""
    reg_payload = {
        "email": "MiXeD.Case@Example.COM",
        "password": "ValidPassword123!",
    }
    reg_res = client.post("/api/auth/register", json=reg_payload)
    assert reg_res.status_code == 201
    assert reg_res.json()["email"] == "mixed.case@example.com"

    # Log in with lowercase
    login_res_lower = client.post("/api/auth/login", json={
        "email": "mixed.case@example.com",
        "password": "ValidPassword123!",
    })
    assert login_res_lower.status_code == 200

    # Log in with uppercase
    login_res_upper = client.post("/api/auth/login", json={
        "email": "MIXED.CASE@EXAMPLE.COM",
        "password": "ValidPassword123!",
    })
    assert login_res_upper.status_code == 200

    # Log in with surrounding whitespace
    login_res_ws = client.post("/api/auth/login", json={
        "email": "   mixed.case@example.com   ",
        "password": "ValidPassword123!",
    })
    assert login_res_ws.status_code == 200


# ==============================================================================
# 3. Surrounding Password Whitespace Consistency
# ==============================================================================

def test_surrounding_password_whitespace_consistency(client):
    """
    Verify that surrounding whitespace is stripped consistently between registration
    and login, while internal spaces and characters inside the password are preserved.
    """
    # Register with surrounding whitespace and internal spaces
    reg_payload = {
        "email": "whitespace_test@example.com",
        "password": "   My Secure P@ssword With Spaces   ",
    }
    reg_res = client.post("/api/auth/register", json=reg_payload)
    assert reg_res.status_code == 201

    # Login without surrounding whitespace succeeds
    login_res_clean = client.post("/api/auth/login", json={
        "email": "whitespace_test@example.com",
        "password": "My Secure P@ssword With Spaces",
    })
    assert login_res_clean.status_code == 200

    # Login with extra surrounding whitespace also succeeds
    login_res_padded = client.post("/api/auth/login", json={
        "email": "whitespace_test@example.com",
        "password": "     My Secure P@ssword With Spaces     ",
    })
    assert login_res_padded.status_code == 200

    # Login with internal spaces removed fails (verifying internal spaces are preserved)
    login_res_no_spaces = client.post("/api/auth/login", json={
        "email": "whitespace_test@example.com",
        "password": "MySecureP@sswordWithSpaces",
    })
    assert login_res_no_spaces.status_code == 401

    # Login with incorrect password fails
    login_res_wrong = client.post("/api/auth/login", json={
        "email": "whitespace_test@example.com",
        "password": "CompletelyDifferentPassword",
    })
    assert login_res_wrong.status_code == 401


# ==============================================================================
# 4. Duplicate Exact Email Registration Rejection
# ==============================================================================

def test_duplicate_exact_email_registration(client):
    """Verify that attempting to register an existing email returns 409 Conflict."""
    payload = {
        "email": "first_registrant@example.com",
        "password": "Password12345!",
    }
    res1 = client.post("/api/auth/register", json=payload)
    assert res1.status_code == 201

    # Exact duplicate registration
    res2 = client.post("/api/auth/register", json=payload)
    assert res2.status_code == 409
    assert "already exists" in res2.json()["detail"].lower()

    # Uppercase duplicate registration
    res3 = client.post("/api/auth/register", json={
        "email": "FIRST_REGISTRANT@EXAMPLE.COM",
        "password": "Password12345!",
    })
    assert res3.status_code == 409
    assert "already exists" in res3.json()["detail"].lower()

    # Duplicate with surrounding whitespace
    res4 = client.post("/api/auth/register", json={
        "email": "   first_registrant@example.com   ",
        "password": "Password12345!",
    })
    assert res4.status_code == 409
    assert "already exists" in res4.json()["detail"].lower()


# ==============================================================================
# 5. Database Path Configuration
# ==============================================================================

def test_database_path_configuration(tmp_path):
    """Verify that get_db_path handles explicit files, directory paths, and parent creation."""
    # 1. Explicit file path in a nested non-existent directory
    nested_file = tmp_path / "deep" / "nested" / "app_data.db"
    resolved_file = get_db_path(nested_file)
    assert resolved_file == nested_file.resolve()
    assert resolved_file.parent.exists()

    # 2. Directory path with existing directory (auto-appends rag_app.db)
    target_dir = tmp_path / "persistent_mount"
    target_dir.mkdir(parents=True, exist_ok=True)
    resolved_dir = get_db_path(target_dir)
    assert resolved_dir == (target_dir / "rag_app.db").resolve()

    # 3. Path ending with trailing slash
    trailing_slash_path = str(tmp_path / "mount_point") + "/"
    resolved_slash = get_db_path(trailing_slash_path)
    assert resolved_slash.name == "rag_app.db"
    assert resolved_slash.parent.exists()


# ==============================================================================
# 6. Existing Users Remain Accessible After Restart / init_db
# ==============================================================================

def test_existing_users_remain_accessible_after_restart(tmp_path):
    """
    Verify that calling init_db() on an existing database (simulating container restart)
    does NOT wipe or corrupt existing users, chats, or passwords.
    """
    persistent_db = tmp_path / "restart_simulation.db"

    # Step 1: Initial startup and schema creation
    init_db(persistent_db)

    # Step 2: Register a user and create a chat in the database
    conn1 = database.get_db_connection(persistent_db)
    user = repository.create_user(
        email="durable_user@example.com",
        password_hash=hash_password("DurablePassword123!"),
        conn=conn1,
    )
    user_id = user["id"]
    user_email = user["email"]
    stored_hash = user["password_hash"]

    chat = repository.create_chat(
        title="Durable Chat Session",
        user_id=user_id,
        conn=conn1,
    )
    chat_id = chat["id"]
    conn1.commit()
    conn1.close()

    # Step 3: Simulate server restart / reboot / redeployment by calling init_db again
    init_db(persistent_db)

    # Step 4: Verify the user and chat are fully preserved and intact
    conn2 = database.get_db_connection(persistent_db)
    fetched_user = repository.get_user_by_id(user_id, conn=conn2)
    assert fetched_user is not None
    assert fetched_user["id"] == user_id
    assert fetched_user["email"] == user_email
    assert fetched_user["password_hash"] == stored_hash

    # Verify password verification succeeds with original password
    assert verify_password("DurablePassword123!", fetched_user["password_hash"]) is True

    # Verify chat is preserved
    fetched_chat = repository.get_chat(chat_id, user_id=user_id, conn=conn2)
    assert fetched_chat is not None
    assert fetched_chat["id"] == chat_id
    assert fetched_chat["title"] == "Durable Chat Session"

    # Verify existing user can still be looked up by email
    by_email = repository.get_user_by_email("durable_user@example.com", conn=conn2)
    assert by_email is not None
    assert by_email["id"] == user_id

    # Verify new users can still be added to the persistent database
    new_user = repository.create_user(
        email="second_user@example.com",
        password_hash=hash_password("SecondPassword123!"),
        conn=conn2,
    )
    assert new_user["id"].startswith("usr_")
    conn2.commit()

    all_users = repository.list_users(conn=conn2)
    # legacy_user + durable_user + second_user = 3 users
    assert len(all_users) >= 3
    user_emails = [u["email"] for u in all_users]
    assert "durable_user@example.com" in user_emails
    assert "second_user@example.com" in user_emails

    conn2.close()
