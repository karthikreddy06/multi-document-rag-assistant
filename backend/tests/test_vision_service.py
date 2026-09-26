"""
Unit tests for VisionService (Gemini primary with Groq fallback) and ImageParser integration.
All tests use mocked HTTP/Gemini responses so no real external API calls or exposed keys are needed.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import httpx
from PIL import Image
import pytest

from app.config import Settings
from app.ingestion.parsers.image_parser import ImageParser
from app.ingestion.parsers.registry import ParserRegistry
from app.services.vision import VisionService, get_vision_service, reset_vision_service, VISION_PROMPT


@pytest.fixture
def sample_png_path(tmp_path) -> Path:
    """Create a valid 64x64 PNG image on disk."""
    img_path = tmp_path / "sample_photo.png"
    img = Image.new("RGB", (64, 64), color="white")
    img.save(img_path, format="PNG")
    return img_path


@pytest.fixture
def tiny_png_path(tmp_path) -> Path:
    """Create an image smaller than 32x32 to test auto-resizing for API compliance."""
    img_path = tmp_path / "tiny.png"
    img = Image.new("RGB", (16, 16), color="red")
    img.save(img_path, format="PNG")
    return img_path


MOCK_VISION_MARKDOWN = (
    "### Extracted Text\n"
    "No visible text in image.\n\n"
    "### Visual Description\n"
    "- Photographic / Scene Content: A close-up portrait of a young man with dark hair, a mustache, beard, "
    "and dark rectangular eyeglass frames. He wears a plain white collared button-up shirt against a light background."
)


class TestVisionServiceConfig:
    def test_default_openrouter_provider_and_model(self):
        service = VisionService(api_key="sk-or-test-key", provider="openrouter")
        assert service.provider == "openrouter"
        assert service.model == "stealth/space-bunny-alpha"

    def test_gemini_provider_and_model(self):
        service = VisionService(api_key="ai-gemini-test-key", provider="gemini", model="gemini-1.5-flash")
        assert service.provider == "gemini"
        assert service.model == "gemini-1.5-flash"

    def test_custom_openrouter_model(self):
        service = VisionService(api_key="sk-or-test-key", provider="openrouter", model="qwen/qwen3.8-27b:free")
        assert service.model == "qwen/qwen3.8-27b:free"

    def test_groq_fallback_provider_config(self):
        service = VisionService(api_key="gsk-test-key", provider="groq", model="qwen/qwen3.8-27b")
        assert service.provider == "groq"
        assert service.model == "qwen/qwen3.8-27b"

    def test_is_configured_logic(self):
        s_unconf = VisionService(api_key="")
        assert not s_unconf.is_configured

        s_conf = VisionService(api_key="valid-key")
        assert s_conf.is_configured

    def test_settings_vision_properties(self):
        s = Settings(OPENROUTER_API_KEY="test_or_val", VISION_PROVIDER="openrouter")
        assert s.effective_vision_provider == "openrouter"
        assert s.effective_vision_model == "stealth/space-bunny-alpha"
        assert s.effective_vision_api_key == "test_or_val"


class TestVisionServicePreparation:
    def test_prepare_image_png(self, sample_png_path):
        service = VisionService(api_key="test-key")
        mime_type, b64_str = service._prepare_image(sample_png_path)
        assert mime_type == "image/png"
        assert len(b64_str) > 50

    def test_prepare_image_upscales_tiny_dimensions(self, tiny_png_path):
        service = VisionService(api_key="test-key")
        mime_type, b64_str = service._prepare_image(tiny_png_path)
        assert mime_type == "image/png"
        assert len(b64_str) > 0

    def test_prepare_image_empty_raises(self, tmp_path):
        empty_file = tmp_path / "empty.png"
        empty_file.write_bytes(b"")
        service = VisionService(api_key="test-key")
        with pytest.raises(ValueError, match="empty"):
            service._prepare_image(empty_file)

    def test_prepare_image_missing_file_raises(self, tmp_path):
        missing = tmp_path / "nonexistent.png"
        service = VisionService(api_key="test-key")
        with pytest.raises(FileNotFoundError):
            service._prepare_image(missing)


class TestVisionServiceDescribeImageOpenRouter:
    def test_describe_image_success_openrouter_mocked(self, sample_png_path):
        service = VisionService(api_key="test-or-key", provider="openrouter", model="stealth/space-bunny-alpha")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": MOCK_VISION_MARKDOWN
                    }
                }
            ]
        }

        captured_requests = []

        def mock_post(url, headers, json):
            captured_requests.append({"url": url, "headers": headers, "json": json})
            return mock_response

        with patch("httpx.Client.post", side_effect=mock_post):
            result = service.describe_image(sample_png_path, original_filename="image.jpg")

        assert result is not None
        assert "white collared button-up shirt" in result
        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert "openrouter.ai/api/v1/chat/completions" in req["url"]
        assert req["headers"]["Authorization"] == "Bearer test-or-key"
        assert req["headers"]["HTTP-Referer"] == "https://github.com/karthikreddy06/multi-document-rag-assistant"
        assert req["json"]["model"] == "stealth/space-bunny-alpha"
        assert "models" in req["json"]
        msg = req["json"]["messages"][0]["content"]
        assert any(item["type"] == "text" and item["text"] == VISION_PROMPT for item in msg)
        assert any(item["type"] == "image_url" and item["image_url"]["url"].startswith("data:image/png;base64,") for item in msg)

    def test_describe_image_openrouter_http_error_returns_none(self, sample_png_path):
        service = VisionService(api_key="test-or-key", provider="openrouter")

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limited upstream."
        error = httpx.HTTPStatusError("Too Many Requests", request=MagicMock(), response=mock_response)

        with patch("httpx.Client.post", side_effect=error):
            result = service.describe_image(sample_png_path)

        assert result is None


class TestVisionServiceDescribeImageGemini:
    def test_describe_image_success_gemini_mocked(self, sample_png_path):
        service = VisionService(api_key="test-gemini-key", provider="gemini", model="gemini-1.5-flash")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": MOCK_VISION_MARKDOWN}
                        ]
                    }
                }
            ]
        }

        captured_requests = []

        def mock_post(url, headers, json):
            captured_requests.append({"url": url, "headers": headers, "json": json})
            return mock_response

        with patch("httpx.Client.post", side_effect=mock_post):
            result = service.describe_image(sample_png_path, original_filename="image.jpg")

        assert result is not None
        assert "white collared button-up shirt" in result
        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert "generativelanguage.googleapis.com" in req["url"]
        assert "gemini-1.5-flash:generateContent" in req["url"]
        assert req["headers"]["x-goog-api-key"] == "test-gemini-key"
        parts = req["json"]["contents"][0]["parts"]
        assert parts[0]["text"] == VISION_PROMPT
        assert parts[1]["inline_data"]["mime_type"] == "image/png"
        assert len(parts[1]["inline_data"]["data"]) > 0

    def test_describe_image_unconfigured_returns_none(self, sample_png_path):
        service = VisionService(api_key="")
        result = service.describe_image(sample_png_path)
        assert result is None

    def test_describe_image_gemini_http_error_returns_none(self, sample_png_path):
        service = VisionService(api_key="test-gemini-key", provider="gemini")

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "API key not valid."
        error = httpx.HTTPStatusError("Forbidden", request=MagicMock(), response=mock_response)

        with patch("httpx.Client.post", side_effect=error):
            result = service.describe_image(sample_png_path)

        assert result is None


class TestImageParserVisionHookIntegration:
    def test_successful_vision_sets_placeholder_false(self, sample_png_path):
        mock_hook = MagicMock(return_value=MOCK_VISION_MARKDOWN)
        parser = ImageParser(vision_hook=mock_hook)

        docs = parser.parse(sample_png_path, original_filename="image.jpg", file_hash="hash_123")
        assert len(docs) == 1
        doc = docs[0]
        assert doc.metadata["is_placeholder"] is False
        assert doc.metadata["is_image"] is True
        assert "white collared button-up shirt" in doc.page_content

    def test_failed_vision_hook_sets_placeholder_true(self, sample_png_path):
        mock_hook = MagicMock(side_effect=RuntimeError("Gemini service timeout"))
        parser = ImageParser(vision_hook=mock_hook)

        docs = parser.parse(sample_png_path, original_filename="image.jpg", file_hash="hash_123")
        assert len(docs) == 1
        doc = docs[0]
        assert doc.metadata["is_placeholder"] is True
        assert "[Image File: image.jpg" in doc.page_content


class TestParserRegistryWiring:
    def test_parser_registry_wires_configured_vision_service(self):
        reset_vision_service()
        with patch.object(VisionService, "is_configured", True):
            with patch.object(VisionService, "describe_image", return_value="Mocked Vision Description"):
                registry = ParserRegistry()
                image_parser = registry.get_parser(".jpg")
                assert image_parser._vision_hook is not None

    def test_parser_registry_respects_custom_hook_override(self):
        custom_hook = lambda p: "Custom Vision Hook"
        registry = ParserRegistry(vision_hook=custom_hook)
        image_parser = registry.get_parser(".png")
        assert image_parser._vision_hook == custom_hook
