"""
Pytest global configuration and environment initialization.
Ensures local chroma_db at repository root is located during test runs.
Provides automatic authentication for legacy test suites while preserving
full real authentication for test_auth and test_user_isolation.
"""

import os
from pathlib import Path
import pytest

from app.api.routes import app
from app.api.auth import get_current_user, get_current_user_id
from app.config import settings

# If CHROMA_PATH is not explicitly set, default to chroma_db at repository root
if "CHROMA_PATH" not in os.environ:
    repo_root = Path(__file__).resolve().parent.parent.parent
    root_chroma = repo_root / "chroma_db"
    if root_chroma.exists():
        os.environ["CHROMA_PATH"] = str(root_chroma)


@pytest.fixture(autouse=True)
def default_sqlite_for_tests(monkeypatch, request):
    """
    By default in unit test suites, ensure tests run against local SQLite
    unless the test specifically targets PostgreSQL/Supabase/pgvector.
    """
    mod_name = request.module.__name__
    if any(k in mod_name for k in ("postgres", "pgvector", "supabase")):
        yield
        return
    monkeypatch.setattr(settings, "database_url", None)
    yield


@pytest.fixture(autouse=True)
def auto_auth_for_legacy_tests(request):
    """
    Automatically provide legacy_user authentication for existing legacy tests
    that were written before user auth existed, while ensuring test_auth and
    test_user_isolation run against the real JWT authentication layer.
    """
    mod_name = request.module.__name__
    if any(k in mod_name for k in ("test_auth", "test_user_isolation", "frontend_auth", "persistence", "file_library", "conversion", "smoke")):
        yield
        return

    legacy_user = {
        "id": "legacy_user",
        "email": "legacy@local.dev",
        "created_at": "2026-01-01T00:00:00Z"
    }

    app.dependency_overrides[get_current_user] = lambda: legacy_user
    app.dependency_overrides[get_current_user_id] = lambda: "legacy_user"
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_current_user_id, None)
