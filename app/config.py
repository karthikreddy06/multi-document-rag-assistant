"""
Centralized Configuration Module for RAG Application.
Loads settings from environment variables or .env file with sensible production defaults.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # Ollama settings
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    embedding_model: str = Field(default="nomic-embed-text", alias="EMBEDDING_MODEL")
    llm_model: str = Field(default="llama3.2", alias="LLM_MODEL")
    ollama_timeout: float = Field(default=60.0, alias="OLLAMA_TIMEOUT")

    # Storage settings
    chroma_path: str = Field(default="./chroma_db", alias="CHROMA_PATH")
    collection_name: str = Field(default="documents", alias="COLLECTION_NAME")
    documents_dir: str = Field(default="./documents", alias="DOCUMENTS_DIR")

    # Chunking settings
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP")

    # Retrieval settings
    top_k: int = Field(default=5, alias="TOP_K")

    # Logging settings
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def chroma_abs_path(self) -> Path:
        return Path(self.chroma_path).resolve()

    @property
    def documents_abs_path(self) -> Path:
        return Path(self.documents_dir).resolve()


settings = Settings()
