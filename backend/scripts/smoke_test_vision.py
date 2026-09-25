"""
Local smoke-test script for Groq multimodal VisionService (qwen/qwen3.8-27b).
Validates real end-to-end vision extraction against Groq API without exposing secrets.

Usage:
    python backend/scripts/smoke_test_vision.py
"""

import io
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image, ImageDraw

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Load .env
env_file = backend_dir / ".env"
if env_file.exists():
    load_dotenv(env_file)

from app.config import settings
from app.ingestion.parsers.image_parser import ImageParser
from app.services.vision import VisionService, get_vision_service


def create_test_image(output_path: Path) -> Path:
    """Generate a test image with distinct text, shapes, and colors for visual validation."""
    width, height = 320, 200
    img = Image.new("RGB", (width, height), color="#1E293B")  # Slate dark background
    draw = ImageDraw.Draw(img)

    # Draw a colored card / box
    draw.rectangle([20, 20, 300, 90], fill="#0EA5E9", outline="#38BDF8", width=2)
    # Simple lines mimicking text structure
    draw.text((35, 35), "SYSTEM METRIC: 99.98% UPTIME", fill="#FFFFFF")
    draw.text((35, 55), "STATUS: OPERATIONAL - NODE 07", fill="#E0F2FE")

    # Draw a bottom table/bar representation
    draw.rectangle([20, 110, 140, 175], fill="#10B981")  # Green bar
    draw.text((25, 120), "Active: 4,520", fill="#FFFFFF")

    draw.rectangle([160, 110, 300, 175], fill="#F59E0B")  # Amber bar
    draw.text((165, 120), "Queued: 128", fill="#FFFFFF")

    img.save(output_path, format="PNG")
    return output_path


def main():
    print("=" * 70)
    print("      GROQ MULTIMODAL VISION SMOKE TEST (qwen/qwen3.8-27b)")
    print("=" * 70)

    # Verify environment configuration without exposing secret values
    key = settings.groq_api_key
    if not key or not key.strip():
        print("[ERROR] GROQ_API_KEY is not configured in environment or backend/.env.")
        sys.exit(1)

    masked_key = f"...{key[-6:]}" if len(key) > 6 else "***"
    print(f"[*] GROQ_API_KEY detected: True (Key suffix: {masked_key})")
    print(f"[*] Target Model:         {settings.effective_vision_model}")
    print(f"[*] API Base URL:         {settings.effective_llm_api_url}")
    print("-" * 70)

    # Generate temporary test image
    temp_dir = backend_dir / "data" / "temp_test"
    temp_dir.mkdir(parents=True, exist_ok=True)
    test_image_path = temp_dir / "smoke_test_metric_card.png"
    create_test_image(test_image_path)
    print(f"[*] Created synthetic test image: {test_image_path.name} (320x200)")

    try:
        # Step 1: Direct VisionService invocation
        print("\n[Step 1] Calling VisionService.describe_image() via Groq...")
        vision_service = VisionService()
        description = vision_service.describe_image(test_image_path, original_filename="smoke_test_metric_card.png")

        if not description:
            print("[FAIL] VisionService returned None or empty response.")
            sys.exit(1)

        print("[PASS] Successfully received multimodal response from Groq!")
        print(f"       Extracted character length: {len(description)} chars")

        has_ocr = "### Extracted Text" in description or "Extracted Text" in description
        has_vis = "### Visual Description" in description or "Visual Description" in description
        print(f"       Contains 'Extracted Text' section:    {has_ocr}")
        print(f"       Contains 'Visual Description' section: {has_vis}")

        print("\n--- Preview of Generated Description ---")
        preview = description[:450] + ("..." if len(description) > 450 else "")
        print(preview)
        print("----------------------------------------")

        # Step 2: Test ImageParser integration
        print("\n[Step 2] Testing ImageParser with VisionService hook...")
        parser = ImageParser(vision_hook=vision_service.describe_image)
        docs = parser.parse(test_image_path, original_filename="smoke_test_metric_card.png", file_hash="smoke_hash_01")

        assert len(docs) == 1, "Expected exactly 1 Document object"
        doc = docs[0]

        is_placeholder = doc.metadata.get("is_placeholder")
        is_image = doc.metadata.get("is_image")
        width = doc.metadata.get("image_width")
        height = doc.metadata.get("image_height")

        print(f"[*] Document Metadata:")
        print(f"    - is_image:       {is_image} (Expected: True)")
        print(f"    - is_placeholder: {is_placeholder} (Expected: False)")
        print(f"    - dimensions:     {width}x{height} (Expected: 320x200)")
        print(f"    - page_content:   {len(doc.page_content)} characters")

        if is_placeholder is False and is_image is True:
            print("\n[SUCCESS] ImageParser verified: is_placeholder=False with rich vision content!")
        else:
            print(f"\n[FAIL] ImageParser failed: is_placeholder={is_placeholder}")
            sys.exit(1)

    finally:
        # Clean up temporary test file
        if test_image_path.exists():
            test_image_path.unlink()
        try:
            temp_dir.rmdir()
        except OSError:
            pass

    print("\n" + "=" * 70)
    print("  ALL SMOKE TESTS PASSED - GROQ VISION PIPELINE IS OPERATIONAL")
    print("=" * 70)


if __name__ == "__main__":
    main()
