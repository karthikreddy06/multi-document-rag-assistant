"""
Unit tests for VisionService and ImageParser vision hook integration.
All tests use mocked HTTP/Groq responses so no real external API calls are made.
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
    img_path = tmp_path / "sample_diagram.png"
    img = Image.new("RGB", (64, 64), color="blue")
    img.save(img_path, format="PNG")
    return img_path


@pytest.fixture
def tiny_png_path(tmp_path) -> Path:
    """Create an image smaller than 32x32 to test auto-resizing for API compliance."""
    img_path = tmp_path / "tiny.png"
    img = Image.new("RGB", (10, 10), color="green")
    img.save(img_path, format="PNG")
    return img_path


MOCK_VISION_MARKDOWN = (
    "### Extracted Text\n"
    "Quarterly Revenue: $4.2M\n"
    "Growth Rate: +18%\n\n"
    "### Visual Description\n"
    "A bar chart comparing Q1 to Q4 financial performance with an upward trendline."
)


class TestVisionServiceConfig:
    def test_default_model(self):
        service = VisionService(api_key="test-key", model=None)
        assert service.model == "qwen/qwen3.8-27b"

    def test_custom_vision_model(self):
        service = VisionService(api_key="test-key", model="custom-vision-model")
        assert service.model == "custom-vision-model"

    def test_is_configured_logic(self):
        s_unconf = VisionService(api_key="")
        assert not s_unconf.is_configured

        s_conf = VisionService(api_key="gsk_valid_key")
        assert s_conf.is_configured

    def test_settings_vision_model_field(self):
        s = Settings(VISION_MODEL="qwen/qwen3.8-27b")
        assert s.vision_model == "qwen/qwen3.8-27b"
        assert s.effective_vision_model == "qwen/qwen3.8-27b"


class TestVisionServicePreparation:
    def test_prepare_image_png(self, sample_png_path):
        service = VisionService(api_key="test-key")
        mime_type, b64_str = service._prepare_image(sample_png_path)
        assert mime_type == "image/png"
        assert len(b64_str) > 50

    def test_prepare_image_upscales_tiny_dimensions(self, tiny_png_path):
        """Images under 32x32 must be upscaled so Groq API does not reject them."""
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


class TestVisionServiceDescribeImage:
    def test_describe_image_success_mocked(self, sample_png_path):
        service = VisionService(api_key="test-key", model="qwen/qwen3.8-27b")

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

        with patch("httpx.Client.post", return_value=mock_response) as mock_post:
            result = service.describe_image(sample_png_path)

            assert result == MOCK_VISION_MARKDOWN
            assert "### Extracted Text" in result
            assert "### Visual Description" in result

            # Verify request arguments sent to Groq
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            headers = call_kwargs["headers"]
            payload = call_kwargs["json"]

            assert headers["Authorization"] == "Bearer test-key"
            assert payload["model"] == "qwen/qwen3.8-27b"
            msg = payload["messages"][0]["content"]
            assert any(item["type"] == "text" for item in msg)
            assert any(item["type"] == "image_url" for item in msg)
            img_url = next(item["image_url"]["url"] for item in msg if item["type"] == "image_url")
            assert img_url.startswith("data:image/png;base64,")

    def test_describe_image_unconfigured_returns_none(self, sample_png_path):
        service = VisionService(api_key="")
        result = service.describe_image(sample_png_path)
        assert result is None

    def test_describe_image_api_error_returns_none(self, sample_png_path):
        service = VisionService(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("httpx.Client.post", side_effect=httpx.HTTPStatusError("500 Error", request=MagicMock(), response=mock_resp)):
            result = service.describe_image(sample_png_path)
            assert result is None


class TestImageParserVisionHookIntegration:
    def test_successful_vision_sets_placeholder_false(self, sample_png_path):
        """When vision succeeds, is_placeholder must be False and page_content contains vision text."""
        hook = lambda p: MOCK_VISION_MARKDOWN
        parser = ImageParser(vision_hook=hook)
        docs = parser.parse(sample_png_path, "sample_diagram.png", "hash123")

        assert len(docs) == 1
        doc = docs[0]
        assert doc.page_content == MOCK_VISION_MARKDOWN
        assert doc.metadata["is_placeholder"] is False
        assert doc.metadata["is_image"] is True
        assert doc.metadata["format"] == "image"
        assert doc.metadata["image_width"] == 64
        assert doc.metadata["image_height"] == 64

    def test_failed_vision_hook_sets_placeholder_true(self, sample_png_path):
        """When vision fails, parser falls back to Pillow placeholder with is_placeholder=True."""
        def bad_hook(p):
            raise RuntimeError("Groq rate limit exceeded")

        parser = ImageParser(vision_hook=bad_hook)
        docs = parser.parse(sample_png_path, "sample_diagram.png", "hash123")

        assert len(docs) == 1
        doc = docs[0]
        assert "Format: PNG" in doc.page_content
        assert "Note: This is an image file" in doc.page_content
        assert doc.metadata["is_placeholder"] is True
        assert doc.metadata["is_image"] is True

    def test_empty_vision_result_sets_placeholder_true(self, sample_png_path):
        hook = lambda p: ""
        parser = ImageParser(vision_hook=hook)
        docs = parser.parse(sample_png_path, "sample_diagram.png", "hash123")

        assert docs[0].metadata["is_placeholder"] is True
        assert "Note: This is an image file" in docs[0].page_content


class TestParserRegistryDefaultWiring:
    def test_parser_registry_wires_configured_vision_service(self):
        reset_vision_service()
        with patch("app.services.vision.VisionService.is_configured", True):
            registry = ParserRegistry()
            image_parser = registry.get_parser(".png")
            assert isinstance(image_parser, ImageParser)
            assert image_parser._vision_hook is not None

    def test_parser_registry_respects_custom_hook_override(self):
        custom_hook = lambda p: "Custom output"
        registry = ParserRegistry(vision_hook=custom_hook)
        image_parser = registry.get_parser(".png")
        assert image_parser._vision_hook == custom_hook
