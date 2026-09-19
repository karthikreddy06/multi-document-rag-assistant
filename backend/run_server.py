"""
Production Server Entrypoint for Multi-Document RAG Assistant.
Binds to 0.0.0.0 and PORT environment variable.
"""

import os
import uvicorn

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.api.routes:app", host=host, port=port, reload=False)
