"""
Centralized Configuration Module for RAG Application.
Loads settings from environment variables or .env file with sensible production defaults.
"""

import os
from pathlib import Path
from typing import Any, List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


# Project root is the parent of the backend directory (where this config.py's grandparent is)
# config.py -> backend/app/ -> backend/ -> project_root/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Provider & Model settings
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    llm_model: str = Field(default="llama3.2:1b", alias="LLM_MODEL")
    llm_api_url: str = Field(default="", alias="LLM_API_URL")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_num_predict: int = Field(default=120, alias="LLM_NUM_PREDICT")
    llm_num_ctx: int = Field(default=1536, alias="LLM_NUM_CTX")

    embedding_provider: str = Field(default="ollama", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="nomic-embed-text", alias="EMBEDDING_MODEL")
    embedding_api_url: str = Field(default="", alias="EMBEDDING_API_URL")
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_dimensions: int = Field(default=768, alias="EMBEDDING_DIMENSIONS")

    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    ollama_timeout: float = Field(default=180.0, alias="OLLAMA_TIMEOUT")

    # Storage settings - defaults relative to project structure
    # chroma_db is at project root, documents is in backend/
    chroma_path: str = Field(default=str(_PROJECT_ROOT / "chroma_db"), alias="CHROMA_PATH")
    collection_name: str = Field(default="documents", alias="COLLECTION_NAME")
    documents_dir: str = Field(default=str(_BACKEND_DIR / "documents"), alias="DOCUMENTS_DIR")

    # Database settings
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")
    database_path: str = Field(default=str(_BACKEND_DIR / "data" / "rag_app.db"), alias="DATABASE_PATH")
    upload_dir: str = Field(default=str(_BACKEND_DIR / "data" / "uploads"), alias="UPLOAD_DIR")

    # Supabase Storage settings
    supabase_url: str = Field(default="https://ldytwnvxskfajjwcxxtb.supabase.co", alias="SUPABASE_URL")
    supabase_service_role_key: Optional[str] = Field(default=None, alias="SUPABASE_SERVICE_ROLE_KEY")
    supabase_storage_bucket: str = Field(default="rag-files", alias="SUPABASE_STORAGE_BUCKET")

    # Vector store provider: "chroma" (default) or "pgvector"
    vector_store_provider: str = Field(default="chroma", alias="VECTOR_STORE_PROVIDER")

    # Chunking settings
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP")

    # Retrieval settings
    top_k: int = Field(default=3, alias="TOP_K")

    # API & CORS settings
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ],
        alias="CORS_ORIGINS"
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_str = v.strip()
            if v_str.startswith("[") and v_str.endswith("]"):
                import json
                try:
                    return json.loads(v_str)
                except Exception:
                    pass
            return [origin.strip() for origin in v_str.split(",") if origin.strip()]
        return v

    # Logging settings
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Authentication & Security settings
    jwt_secret_key: str = Field(
        default="rag_assistant_dev_secret_key_change_in_production_32b+",
        alias="JWT_SECRET_KEY"
    )
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=1440, alias="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")

    model_config = SettingsConfigDict(
        env_file=(".env", str(_BACKEND_DIR / ".env")),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def chroma_abs_path(self) -> Path:
        return Path(self.chroma_path).resolve()

    @property
    def documents_abs_path(self) -> Path:
        return Path(self.documents_dir).resolve()

    @property
    def database_abs_path(self) -> Path:
        return Path(self.database_path).resolve()

    @property
    def is_postgres(self) -> bool:
        return bool(self.database_url and self.database_url.strip())

    @property
    def is_supabase_storage(self) -> bool:
        return bool(
            self.supabase_url
            and self.supabase_service_role_key
            and self.supabase_service_role_key.strip()
        )

    @property
    def is_pgvector(self) -> bool:
        return bool(self.vector_store_provider.lower() == "pgvector" and self.is_postgres)

    @property
    def upload_abs_path(self) -> Path:
        return Path(self.upload_dir).resolve()

    @property
    def effective_llm_api_key(self) -> str:
        p = self.llm_provider.lower()
        if p == "groq":
            return self.groq_api_key or self.llm_api_key
        if p == "cloud":
            return self.gemini_api_key or self.llm_api_key
        return self.llm_api_key or self.groq_api_key or self.gemini_api_key

    @property
    def effective_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.gemini_api_key

    @property
    def effective_llm_api_url(self) -> str:
        if self.llm_api_url:
            return self.llm_api_url
        p = self.llm_provider.lower()
        if p == "groq":
            return "https://api.groq.com/openai/v1"
        if p == "cloud":
            return "https://generativelanguage.googleapis.com/v1beta/openai/"
        return ""

    @property
    def effective_embedding_api_url(self) -> str:
        if self.embedding_api_url:
            return self.embedding_api_url
        if self.embedding_provider.lower() == "cloud":
            return "https://generativelanguage.googleapis.com/v1beta/openai/"
        return ""

    @property
    def effective_llm_model(self) -> str:
        p = self.llm_provider.lower()
        if p == "groq":
            if self.llm_model and self.llm_model != "llama3.2:1b":
                return self.llm_model
            return "openai/gpt-oss-20b"
        if p == "cloud" and self.llm_model == "llama3.2:1b":
            return "gemini-3.6-flash"
        return self.llm_model

    @property
    def effective_embedding_model(self) -> str:
        p = self.embedding_provider.lower()
        if p in ("local", "onnx"):
            if self.embedding_model and self.embedding_model != "nomic-embed-text":
                return self.embedding_model
            return "all-MiniLM-L6-v2"
        if p == "cloud" and self.embedding_model == "nomic-embed-text":
            return "gemini-embedding-2"
        return self.embedding_model

    @property
    def effective_embedding_dimensions(self) -> int:
        p = self.embedding_provider.lower()
        if p in ("local", "onnx"):
            if self.embedding_dimensions != 768:
                return self.embedding_dimensions
            return 384
        return self.embedding_dimensions


settings = Settings()
