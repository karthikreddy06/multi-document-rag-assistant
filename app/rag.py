"""
RAG Application Entrypoint (Backward-Compatible).
Directs execution to the modular production RAG application.
"""

import sys
from pathlib import Path

# Ensure rag-project root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.main import main

if __name__ == "__main__":
    main()