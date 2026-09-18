"""
Image Parser — extracts metadata from image files using Pillow.
Stores a structured descriptive placeholder that enables the RAG system to
acknowledge the image's existence without hallucinating visual content.

A pluggable vision/OCR hook is provided for future extension.
"""

from pathlib import Path
from typing import List, Optional

from app.ingestion.parsers.base import BaseParser
from app.models import Document
from app.utils.logger import setup_logger

logger = setup_logger("ingestion.parsers.image")


class ImageParser(BaseParser):
    """
    Parse image files (.png, .jpg, .jpeg, .webp) into descriptive Document objects.

    Strategy: graceful degradation with a pluggable vision hook.
    - By default, a structured metadata description is indexed so the RAG system
      can acknowledge the image without fabricating visual content.
    - If a vision_hook callable is provided (e.g., a VLM or OCR function), it will
      be called with the file_path and its output will replace the placeholder text.
    """

    SUPPORTED_EXTENSIONS: frozenset = frozenset({".png", ".jpg", ".jpeg", ".webp"})

    def __init__(self, vision_hook=None):
        """
        Args:
            vision_hook: Optional callable(file_path: Path) -> str that returns
                         extracted/described text. If None, a metadata placeholder
                         is used.
        """
        self._vision_hook = vision_hook

    def parse(
        self,
        file_path: Path,
        original_filename: str,
        file_hash: str,
    ) -> List[Document]:
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        logger.info(f"Parsing image: {original_filename}")

        # Attempt to read image metadata with Pillow
        image_info = self._read_image_metadata(file_path)

        # If a vision hook is available, use it for richer text extraction
        if self._vision_hook is not None:
            try:
                extracted_text = self._vision_hook(file_path)
                if extracted_text and extracted_text.strip():
                    content = extracted_text.strip()
                    logger.info(f"Vision hook produced {len(content)} chars for '{original_filename}'")
                else:
                    content = self._build_placeholder(original_filename, image_info)
            except Exception as e:
                logger.warning(f"Vision hook failed for '{original_filename}': {e}. Using placeholder.")
                content = self._build_placeholder(original_filename, image_info)
        else:
            content = self._build_placeholder(original_filename, image_info)

        meta = self._base_metadata(
            file_path=file_path,
            original_filename=original_filename,
            file_hash=file_hash,
            page_number=1,
            total_pages=1,
            char_count=len(content),
            format="image",
            image_format=image_info.get("format", "unknown"),
            image_width=image_info.get("width"),
            image_height=image_info.get("height"),
            image_mode=image_info.get("mode"),
            is_image=True,
            is_placeholder=(self._vision_hook is None),
        )

        logger.info(f"ImageParser: indexed '{original_filename}' as descriptive placeholder.")
        return [Document(page_content=content, metadata=meta)]

    def _read_image_metadata(self, file_path: Path) -> dict:
        """Extract basic image metadata using Pillow; returns empty dict on failure."""
        try:
            from PIL import Image
            with Image.open(str(file_path)) as img:
                return {
                    "format": img.format or file_path.suffix.lstrip(".").upper(),
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                }
        except Exception as e:
            logger.warning(f"Could not read image metadata for '{file_path.name}': {e}")
            return {"format": file_path.suffix.lstrip(".").upper()}

    @staticmethod
    def _build_placeholder(original_filename: str, image_info: dict) -> str:
        """Build a structured descriptive placeholder for the image."""
        parts = [f"[Image File: {original_filename}"]
        fmt = image_info.get("format")
        if fmt:
            parts.append(f"Format: {fmt}")
        w = image_info.get("width")
        h = image_info.get("height")
        if w and h:
            parts.append(f"Dimensions: {w}x{h} pixels")
        mode = image_info.get("mode")
        if mode:
            parts.append(f"Color mode: {mode}")
        parts.append(
            "Note: This is an image file. Text content is not available without OCR or vision processing."
        )
        parts.append("]")
        return "\n".join(parts)
