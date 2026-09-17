"""
Legacy embedding compatibility module.
Delegates to app.embeddings.service.EmbeddingService.
"""

from app.embeddings.service import EmbeddingService

_service = EmbeddingService()


def create_embedding(text: str):
    """Generate embedding for text using nomic-embed-text."""
    return _service.embed_text(text)