"""
Vision Service Module.
Provides multimodal image understanding and OCR extraction using Google Gemini (with Groq fallback).
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
    "numbers, diagrams, charts, UI components, photographic details, people, clothing, colors, objects, "
    "or visual context can be accurately retrieved and answered.\n\n"
    "Structure your response in clean, valid Markdown using the following exact headings:\n\n"
    "### Extracted Text\n"
    "- Transcribe ALL visible text, headings, paragraphs, labels, numbers, codes, and data values verbatim.\n"
    "- If the image contains tables or tabular data, format them as Markdown tables.\n"
    "- If no text is visible in the image, write: 'No visible text in image.'\n\n"
    "### Visual Description\n"
    "- Photographic / Scene Content: describe in detail the subject, people, facial features, expressions, glasses, clothing/shirt style and color, accessories, background, colors, lighting, and spatial arrangement.\n"
    "- Document / UI Layout: describe the structure, hierarchy, forms, buttons, cards, navigation, or screenshots if applicable.\n"
    "- Charts & Diagrams: explain axes, units, series, trends, data points, flows, connected nodes, and sequences if applicable.\n"
    "- Key Takeaways: summarize the primary message, identity, or purpose conveyed by the image."
)

MIME_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "JPG": "image/jpeg",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}


class VisionService:
    """Multimodal vision service leveraging Google Gemini (default) with Groq fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.provider = (provider or settings.effective_vision_provider).lower()
        self.model = model or settings.effective_vision_model

        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = settings.effective_vision_api_key

        self.api_url = api_url
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
        Reads image data, ensures minimum dimension requirements, and encodes as base64 data.

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

        # Validate with Pillow and ensure minimum dimensions (at least 32x32)
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

    def _call_gemini(self, mime_type: str, b64_data: str, filename: str) -> Optional[str]:
        """Call Google Gemini's multimodal REST API."""
        endpoint_url = self.api_url
        if not endpoint_url:
            endpoint_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        # Check if caller configured an OpenAI-compatible Gemini URL
        if "/openai/" in endpoint_url or endpoint_url.endswith("/chat/completions"):
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VISION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
                            },
                        ],
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 2048,
            }
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            # Native Google Generative Language generateContent API
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": VISION_PROMPT},
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": b64_data,
                                }
                            },
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 2048,
                },
            }

        logger.info(f"VisionService: Sending vision request for '{filename}' to Gemini ({self.model})...")
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

            # 1. Parse native Gemini response structure
            candidates = data.get("candidates", [])
            if candidates:
                candidate = candidates[0]
                parts = candidate.get("content", {}).get("parts", [])
                if parts:
                    content = parts[0].get("text", "").strip()
                    if content:
                        logger.info(
                            f"VisionService: Gemini successfully analyzed '{filename}' ({len(content)} chars)."
                        )
                        return content

            # 2. Parse OpenAI-compatible fallback structure
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "").strip()
                if content:
                    logger.info(
                        f"VisionService: Gemini analyzed '{filename}' via OpenAI endpoint ({len(content)} chars)."
                    )
                    return content

            logger.warning(f"VisionService: Empty content returned from Gemini for '{filename}'.")
            return None

        except httpx.HTTPStatusError as e:
            logger.warning(
                f"VisionService: Gemini API HTTP error for '{filename}': {e.response.status_code} - {e.response.text}"
            )
            return None
        except Exception as e:
            logger.warning(f"VisionService: Gemini API call failed for '{filename}': {e}")
            return None

    def _call_groq(self, mime_type: str, b64_data: str, filename: str) -> Optional[str]:
        """Call Groq's multimodal chat completions endpoint (fallback provider)."""
        endpoint_url = self.api_url or settings.effective_llm_api_url or "https://api.groq.com/openai/v1"
        if not endpoint_url.endswith("/chat/completions") and not endpoint_url.endswith("/completions"):
            endpoint_url = f"{endpoint_url.rstrip('/')}/chat/completions"

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
                        {"type": "text", "text": VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
                        },
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
        }

        logger.info(f"VisionService: Sending vision request for '{filename}' to Groq ({self.model})...")
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

            choices = data.get("choices", [])
            if not choices:
                logger.warning(f"VisionService: Groq returned no choices for '{filename}'.")
                return None

            message = choices[0].get("message", {})
            content = message.get("content", "").strip()
            if content:
                logger.info(
                    f"VisionService: Groq successfully analyzed '{filename}' ({len(content)} chars)."
                )
                return content
            else:
                logger.warning(f"VisionService: Empty content returned from Groq for '{filename}'.")
                return None

        except httpx.HTTPStatusError as e:
            logger.warning(
                f"VisionService: Groq API HTTP error for '{filename}': {e.response.status_code} - {e.response.text}"
            )
            return None
        except Exception as e:
            logger.warning(f"VisionService: Groq vision call failed for '{filename}': {e}")
            return None

    def _call_openrouter(self, mime_type: str, b64_data: str, filename: str) -> Optional[str]:
        """Call OpenRouter chat completions endpoint for multimodal vision understanding."""
        endpoint_url = self.api_url or "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/karthikreddy06/multi-document-rag-assistant",
            "X-Title": "Multi-Document RAG Assistant",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
                        },
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1500,
        }

        # Provide resilient fallback models on OpenRouter (max 3 allowed)
        if "stealth/space-bunny-alpha" in (self.model or ""):
            payload["models"] = [
                "stealth/space-bunny-alpha",
                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
                "google/gemma-4-26b-a4b-it:free",
            ]

        logger.info(f"VisionService: Sending vision request for '{filename}' to OpenRouter ({self.model})...")
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

            choices = data.get("choices", [])
            if not choices:
                logger.warning(f"VisionService: OpenRouter returned no choices for '{filename}'.")
                return None

            message = choices[0].get("message", {})
            content = (message.get("content") or "").strip()

            # Handle reasoning-focused models that may put description in reasoning field
            if not content or len(content) < 10:
                reasoning = (message.get("reasoning") or "").strip()
                if reasoning:
                    content = f"### Visual Description\n{reasoning}"

            if content:
                logger.info(
                    f"VisionService: OpenRouter successfully analyzed '{filename}' ({len(content)} chars)."
                )
                return content
            else:
                logger.warning(f"VisionService: Empty content returned from OpenRouter for '{filename}'.")
                return None

        except httpx.HTTPStatusError as e:
            logger.warning(
                f"VisionService: OpenRouter API HTTP error for '{filename}': {e.response.status_code} - {e.response.text}"
            )
            return None
        except Exception as e:
            logger.warning(f"VisionService: OpenRouter vision call failed for '{filename}': {e}")
            return None

    def describe_image(
        self,
        file_path_or_bytes: Union[str, Path, bytes],
        original_filename: Optional[str] = None,
    ) -> Optional[str]:
        """
        Sends image to vision model (OpenRouter by default) to extract verbatim text and rich visual description.

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
                f"VisionService: Vision API key ({self.provider.upper()}_API_KEY) is not configured. "
                f"Skipping vision extraction for '{fn}'."
            )
            return None

        try:
            mime_type, b64_data = self._prepare_image(file_path_or_bytes)
        except Exception as e:
            logger.warning(f"VisionService: Failed to load/prepare image '{fn}': {e}")
            return None

        if self.provider == "openrouter":
            return self._call_openrouter(mime_type, b64_data, fn)
        elif self.provider == "gemini":
            return self._call_gemini(mime_type, b64_data, fn)
        else:
            return self._call_groq(mime_type, b64_data, fn)


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
