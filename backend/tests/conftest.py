"""
Pytest global configuration and environment initialization.
Ensures local chroma_db at repository root is located during test runs.
"""

import os
from pathlib import Path

# If CHROMA_PATH is not explicitly set, default to chroma_db at repository root
# (conftest.py is in backend/tests/, so repo root is parent.parent)
if "CHROMA_PATH" not in os.environ:
    repo_root = Path(__file__).resolve().parent.parent.parent
    root_chroma = repo_root / "chroma_db"
    if root_chroma.exists():
        os.environ["CHROMA_PATH"] = str(root_chroma)
