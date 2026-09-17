"""
Embedding Service Module.
Interfaces with Ollama using the nomic-embed-text model to generate dense vector embeddings.
"""

from typing import List, Optional
import time
import ollama

from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("embeddings.service")


class EmbeddingService:
    """Production wrapper for Ollama embeddings with health checks and error management."""

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.model = model or settings.embedding_model
        self.host = host or settings.ollama_host
        self.timeout = timeout or settings.ollama_timeout
        self.client = ollama.Client(host=self.host, timeout=self.timeout)

    def check_health(self) -> bool:
        """Verify Ollama availability and presence of the embedding model."""
        try:
            available_models = self.client.list()
            model_names = [m.model for m in available_models.models]
            # Match base name or with tag (e.g. nomic-embed-text:latest)
            matched = any(
                m == self.model or m.startswith(f"{self.model}:")
                for m in model_names
            )
            if not matched:
                logger.warning(
                    f"Embedding model '{self.model}' not found in Ollama models: {model_names}. "
                    f"Run 'ollama pull {self.model}' to download it."
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Cannot connect to Ollama at {self.host}: {e}")
            return False

    def embed_text(self, text: str, max_retries: int = 3) -> List[float]:
        """
        Generate embedding vector for a single text string with retry logic on transient errors.
        """
        if not text or not text.strip():
            logger.warning("Attempted to embed empty text string.")
            return []

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.embed(
                    model=self.model,
                    input=text.strip()
                )
                return response["embeddings"][0]
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Embedding attempt {attempt}/{max_retries} failed for model {self.model}: {e}. "
                    f"{'Retrying...' if attempt < max_retries else 'No more retries.'}"
                )
                if attempt < max_retries:
                    time.sleep(0.5 * attempt)

        logger.error(f"Failed to generate embedding after {max_retries} attempts: {last_error}")
        raise RuntimeError(
            f"Ollama embedding failure: {last_error}. Ensure Ollama is running at {self.host} "
            f"and model '{self.model}' is pulled."
        ) from last_error

    def embed_batch(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """
        Generate embeddings for a list of texts in batch with sequential fallback.
        """
        if not texts:
            return []

        logger.info(f"Generating embeddings for {len(texts)} chunks using {self.model}...")

        # Ollama embed endpoint supports list of inputs
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.embed(
                    model=self.model,
                    input=[t.strip() for t in texts]
                )
                return response["embeddings"]
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(0.5 * attempt)

        logger.warning(f"Batch embedding failed ({last_error}), falling back to sequential embedding...")
        results: List[List[float]] = []
        for t in texts:
            results.append(self.embed_text(t))
        return results
