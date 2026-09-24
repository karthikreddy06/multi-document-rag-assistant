"""
Embedding Service Module.
Interfaces with Ollama or Cloud API providers to generate dense vector embeddings.
"""

from typing import List, Optional
import time
import httpx
import ollama

from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("embeddings.service")


class EmbeddingService:
    """Production wrapper for embeddings (Ollama or Cloud API) with health checks and error management."""

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout: Optional[float] = None,
        provider: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        self.provider = (provider or settings.embedding_provider).lower()
        self._local_ef = None

        if self.provider in ("local", "onnx"):
            if model:
                self.model = model
            elif settings.embedding_model != "nomic-embed-text":
                self.model = settings.embedding_model
            else:
                self.model = "all-MiniLM-L6-v2"
            self.api_url = ""
            if dimensions is not None:
                self.dimensions = dimensions
            elif settings.embedding_dimensions != 768:
                self.dimensions = settings.embedding_dimensions
            else:
                self.dimensions = 384
        elif self.provider == "cloud":
            if model:
                self.model = model
            elif settings.embedding_model != "nomic-embed-text":
                self.model = settings.embedding_model
            else:
                self.model = "gemini-embedding-2"
            self.api_url = api_url or settings.embedding_api_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
            self.dimensions = dimensions if dimensions is not None else settings.embedding_dimensions
        else:
            self.model = model or settings.embedding_model
            self.api_url = api_url or settings.embedding_api_url
            self.dimensions = dimensions if dimensions is not None else settings.embedding_dimensions

        self.host = host or settings.ollama_host
        self.timeout = timeout or settings.ollama_timeout
        self.api_key = api_key or settings.effective_embedding_api_key

        if self.provider == "ollama":
            self.client = ollama.Client(host=self.host, timeout=self.timeout)
        else:
            self.client = None

    def _get_local_ef(self):
        """Lazily initialize lightweight local CPU ONNX embedding function."""
        if self._local_ef is None:
            try:
                from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
                self._local_ef = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
            except Exception as e:
                logger.error(f"Failed to initialize local ONNX embedding model: {e}")
                raise RuntimeError(f"Local ONNX embedding initialization error: {e}") from e
        return self._local_ef

    def check_health(self) -> bool:
        """Verify provider availability and embedding model readiness."""
        if self.provider == "ollama":
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
        elif self.provider in ("local", "onnx"):
            try:
                ef = self._get_local_ef()
                return ef is not None
            except Exception as e:
                logger.error(f"Local embedding provider health check failed: {e}")
                return False
        else:
            if not self.api_url:
                logger.error("Cloud embedding provider specified, but EMBEDDING_API_URL is missing.")
                return False
            if not self.api_key:
                logger.warning("Cloud API key (GEMINI_API_KEY or EMBEDDING_API_KEY) is not set. Cloud embedding request may fail if authentication is required.")
            return True

    def _local_embed(self, input_texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local CPU ONNX model with dimension validation."""
        ef = self._get_local_ef()
        raw_embeddings = ef(input_texts)
        embeddings: List[List[float]] = []
        for emb in raw_embeddings:
            if hasattr(emb, "tolist"):
                vec = emb.tolist()
            else:
                vec = [float(x) for x in emb]
            if self.dimensions and len(vec) != self.dimensions:
                raise ValueError(
                    f"Expected embedding dimension {self.dimensions} from {self.model} ({self.provider}), "
                    f"but received vector of dimension {len(vec)}."
                )
            embeddings.append(vec)
        return embeddings

    def _cloud_embed(self, input_texts: List[str]) -> List[List[float]]:
        """Call OpenAI-compatible cloud REST embeddings endpoint using httpx."""
        url = self.api_url.rstrip("/")
        if not url.endswith("/embeddings"):
            url = f"{url}/embeddings"

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["x-goog-api-key"] = self.api_key

        payload: dict = {
            "model": self.model,
            "input": input_texts if len(input_texts) > 1 else input_texts[0],
        }
        if self.dimensions:
            payload["dimensions"] = self.dimensions

        with httpx.Client(timeout=self.timeout) as http_client:
            resp = http_client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        raw_data = data.get("data", [])
        # Sort by index if present, preserving list order if index is omitted
        raw_data_sorted = sorted(enumerate(raw_data), key=lambda x: x[1].get("index", x[0]))
        embeddings = [item["embedding"] for _, item in raw_data_sorted]

        # Strict validation: verify returned vector dimension matches expected dimensions
        if self.dimensions:
            for i, emb in enumerate(embeddings):
                if len(emb) != self.dimensions:
                    raise ValueError(
                        f"Expected embedding dimension {self.dimensions} from {self.model} ({self.provider}), "
                        f"but received vector of dimension {len(emb)}."
                    )

        return embeddings

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
                if self.provider == "ollama":
                    response = self.client.embed(
                        model=self.model,
                        input=text.strip()
                    )
                    return response["embeddings"][0]
                elif self.provider in ("local", "onnx"):
                    embeddings = self._local_embed([text.strip()])
                    return embeddings[0]
                else:
                    embeddings = self._cloud_embed([text.strip()])
                    return embeddings[0]
            except ValueError:
                # Do not retry on dimension validation error
                raise
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
            f"Embedding failure ({self.provider}): {last_error}."
        ) from last_error

    def embed_batch(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """
        Generate embeddings for a list of texts in batch with sequential fallback.
        """
        if not texts:
            return []

        logger.info(f"Generating embeddings for {len(texts)} chunks using {self.model} ({self.provider})...")

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                if self.provider == "ollama":
                    response = self.client.embed(
                        model=self.model,
                        input=[t.strip() for t in texts]
                    )
                    return response["embeddings"]
                elif self.provider in ("local", "onnx"):
                    return self._local_embed([t.strip() for t in texts])
                else:
                    return self._cloud_embed([t.strip() for t in texts])
            except ValueError:
                # Do not retry or fall back on dimension validation error
                raise
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(0.5 * attempt)

        logger.warning(f"Batch embedding failed ({last_error}), falling back to sequential embedding...")
        results: List[List[float]] = []
        for t in texts:
            results.append(self.embed_text(t))
        return results

