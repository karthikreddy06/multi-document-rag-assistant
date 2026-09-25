"""
Vision Service Module.
Provides multimodal image understanding and OCR extraction using Groq's vision endpoint.
Extracts structured Markdown (Extracted Text and Visual Description) for RAG indexing.
"""

import base64
import io
from pathlib import Path
from typing import Optional, Tuple, Union
import httpx
from PIL import Image

from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("services.vision")

VISION_PROMPT = (
    "You are an expert vision and document analysis system optimized for Retrieval-Augmented Generation (RAG).\n"
    "Your goal is to extract complete, faithful, and granular information from this image so that any text, "
    "numbers, diagrams, charts, UI components, or visual context can be accurately retrieved and answered.\n\n"
    "Structure your response in clean, valid Markdown using the following exact headings:\n\n"
    "### Extracted Text\n"
    "- Transcribe ALL visible text, headings, paragraphs, labels, numbers, codes, and data values verbatim.\n"
    "- If the image contains tables or tabular data, format them as Markdown tables.\n"
    "- If no text is visible in the image, write: 'No visible text in image.'\n\n"
    "### Visual Description\n"
    "- Document / UI Layout: describe the structure, hierarchy, forms, buttons, cards, navigation, or screenshots.\n"
    "- Charts & Diagrams: explain axes, units, series, trends, data points, flows, connected nodes, and sequences.\n"
    "- Photographic / Scene Content: describe objects, people, scenes, colors, spatial arrangement, and meaningful context.\n"
    "- Key Takeaways: summarize the primary message, purpose, or conclusion conveyed by the image."
)

MIME_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "JPG": "image/jpeg",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}


class VisionService:
    """Production multimodal vision service leveraging Groq's vision API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key if api_key is not None else settings.groq_api_key
        raw_url = api_url if api_url is not None else settings.effective_llm_api_url
        self.api_url = raw_url.rstrip("/") if raw_url else "https://api.groq.com/openai/v1"
        self.model = model or settings.effective_vision_model
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        """Returns True if the required API credentials are present."""
        return bool(self.api_key and self.api_key.strip())

    def _prepare_image(
        self,
        file_path_or_bytes: Union[str, Path, bytes],
    ) -> Tuple[str, str]:
        """
        Reads image data, ensures minimum dimension requirements, and encodes as base64 data URI.

        Returns:
            Tuple of (mime_type, base64_encoded_string)
        """
        if isinstance(file_path_or_bytes, (str, Path)):
            path = Path(file_path_or_bytes)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            raw_bytes = path.read_bytes()
        elif isinstance(file_path_or_bytes, bytes):
            raw_bytes = file_path_or_bytes
        else:
            raise TypeError(f"Unsupported image input type: {type(file_path_or_bytes)}")

        if not raw_bytes:
            raise ValueError("Image data is empty (0 bytes)")

        # Validate with Pillow and ensure minimum dimensions (Groq requires at least 32x32)
        with Image.open(io.BytesIO(raw_bytes)) as img:
            fmt = (img.format or "PNG").upper()
            mime_type = MIME_TYPES.get(fmt, "image/png")
            width, height = img.size

            if width < 32 or height < 32:
                new_w = max(32, width)
                new_h = max(32, height)
                logger.info(f"Upscaling small image ({width}x{height}) to ({new_w}x{new_h}) for API compliance")
                resample = getattr(Image, "Resampling", Image).NEAREST
                resized_img = img.resize((new_w, new_h), resample=resample)
                buffer = io.BytesIO()
                resized_img.save(buffer, format=fmt if fmt in ("PNG", "JPEG", "WEBP") else "PNG")
                raw_bytes = buffer.getvalue()

        b64_str = base64.b64encode(raw_bytes).decode("utf-8")
        return mime_type, b64_str

    def describe_image(
        self,
        file_path_or_bytes: Union[str, Path, bytes],
        original_filename: Optional[str] = None,
    ) -> Optional[str]:
        """
        Sends image to Groq's multimodal endpoint to extract verbatim text (OCR) and rich visual context.

        Args:
            file_path_or_bytes: Path to image file on disk or raw bytes.
            original_filename: Optional name for logging.

        Returns:
            Structured Markdown string or None if processing fails.
        """
        fn = original_filename or (
            Path(file_path_or_bytes).name if isinstance(file_path_or_bytes, (str, Path)) else "image"
        )

        if not self.is_configured:
            logger.warning(
                f"VisionService: GROQ_API_KEY is not configured. Skipping vision extraction for '{fn}'."
            )
            return None

        try:
            mime_type, b64_data = self._prepare_image(file_path_or_bytes)
        except Exception as e:
            logger.warning(f"VisionService: Failed to load/prepare image '{fn}': {e}")
            return None

        endpoint_url = self.api_url
        if not endpoint_url.endswith("/chat/completions") and not endpoint_url.endswith("/completions"):
            endpoint_url = f"{endpoint_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": VISION_PROMPT,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_data}",
                            },
                        },
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
        }

        logger.info(f"VisionService: Sending vision request for '{fn}' to Groq ({self.model})...")
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

            choices = data.get("choices", [])
            if not choices:
                logger.warning(f"VisionService: Groq returned no choices for '{fn}'.")
                return None

            message = choices[0].get("message", {})
            content = message.get("content", "").strip()

            if content:
                logger.info(
                    f"VisionService: Successfully extracted vision description for '{fn}' ({len(content)} chars)."
                )
                return content
            else:
                logger.warning(f"VisionService: Empty content returned from vision model for '{fn}'.")
                return None

        except httpx.HTTPStatusError as e:
            logger.warning(
                f"VisionService: Groq API HTTP error for '{fn}': {e.response.status_code} - {e.response.text}"
            )
            return None
        except Exception as e:
            logger.warning(f"VisionService: Failed during vision generation for '{fn}': {e}")
            return None


# Module-level singleton
_default_vision_service: Optional[VisionService] = None


def get_vision_service() -> VisionService:
    """Return the shared VisionService singleton."""
    global _default_vision_service
    if _default_vision_service is None:
        _default_vision_service = VisionService()
    return _default_vision_service


def reset_vision_service() -> None:
    """Reset the shared VisionService singleton (primarily for testing)."""
    global _default_vision_service
    _default_vision_service = None
