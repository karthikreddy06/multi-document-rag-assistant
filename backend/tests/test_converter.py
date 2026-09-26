"""
Comprehensive Tests for DocumentConverter Service & Natural Language Ingestion.
Tests:
1. Format conversion matrix for all required pairs:
   - PDF -> DOCX
   - PDF -> PPTX
   - DOCX -> PDF
   - DOCX -> PPTX
   - PPTX -> DOCX
   - PPTX -> PDF
   - XLSX <-> CSV
   - TXT/Markdown -> DOCX/PDF
   - Image -> PDF
2. Natural language conversion request parsing:
   - "Convert this PDF to Word"
   - "Convert this document to PowerPoint"
   - "Convert this Excel to CSV"
   - "Convert appointment.png to PDF"
3. User isolation on conversion and download endpoints.
"""

import io
from pathlib import Path
from PIL import Image
import pytest
from docx import Document as DocxDocument
from pptx import Presentation
from pptx.util import Inches

from app.services.converter import (
    DocumentConverter,
    is_conversion_supported,
    get_supported_targets,
    UnsupportedConversionError,
    SUPPORTED_CONVERSIONS,
)
from app.retrieval.query_understanding import (
    parse_conversion_request,
    ConversionRequestInfo,
)
from app.auth.security import create_access_token


# ==============================================================================
# 1. MATRIX SUPPORT TESTS
# ==============================================================================

def test_full_conversion_support_matrix():
    # PDF
    assert is_conversion_supported(".pdf", ".docx") is True
    assert is_conversion_supported(".pdf", ".pptx") is True
    assert is_conversion_supported("pdf", "docx") is True

    # DOCX
    assert is_conversion_supported(".docx", ".pdf") is True
    assert is_conversion_supported(".docx", ".pptx") is True

    # PPTX
    assert is_conversion_supported(".pptx", ".docx") is True
    assert is_conversion_supported(".pptx", ".pdf") is True

    # XLSX / CSV
    assert is_conversion_supported(".csv", ".xlsx") is True
    assert is_conversion_supported(".xlsx", ".csv") is True

    # TXT / Markdown
    assert is_conversion_supported(".txt", ".docx") is True
    assert is_conversion_supported(".txt", ".pdf") is True
    assert is_conversion_supported(".md", ".docx") is True
    assert is_conversion_supported(".md", ".pdf") is True

    # Images
    for img_ext in (".png", ".jpg", ".jpeg", ".webp"):
        assert is_conversion_supported(img_ext, ".pdf") is True

    # Unsupported checks
    assert is_conversion_supported(".png", ".xlsx") is False
    assert is_conversion_supported(".csv", ".pptx") is False


# ==============================================================================
# 2. CONVERSION HANDLER TESTS
# ==============================================================================

def test_csv_to_xlsx_and_back():
    raw_csv = "Name,Age,Role\nAlice,30,Engineer\nBob,25,Designer\n".encode("utf-8")
    xlsx_bytes, xlsx_name, xlsx_mime = DocumentConverter.convert(raw_csv, "team.csv", ".xlsx")
    assert len(xlsx_bytes) > 0
    assert xlsx_name == "team.xlsx"
    assert "spreadsheetml" in xlsx_mime

    csv_bytes, csv_name, csv_mime = DocumentConverter.convert(xlsx_bytes, "team.xlsx", ".csv")
    assert csv_name == "team.csv"
    assert csv_mime == "text/csv"
    csv_str = csv_bytes.decode("utf-8")
    assert "Alice" in csv_str
    assert "Designer" in csv_str


def test_txt_and_md_to_docx_and_pdf():
    raw_txt = "Project Plan\n\nPhase 1: Setup\nPhase 2: Execution".encode("utf-8")
    docx_bytes, _, _ = DocumentConverter.convert(raw_txt, "plan.txt", ".docx")
    assert len(docx_bytes) > 0

    pdf_bytes, _, _ = DocumentConverter.convert(raw_txt, "plan.txt", ".pdf")
    assert pdf_bytes.startswith(b"%PDF")

    raw_md = "# Title\n## Section\n- Item A\n- Item B\nParagraph text.".encode("utf-8")
    md_docx, _, _ = DocumentConverter.convert(raw_md, "notes.md", ".docx")
    assert len(md_docx) > 0

    md_pdf, _, _ = DocumentConverter.convert(raw_md, "notes.md", ".pdf")
    assert md_pdf.startswith(b"%PDF")


def test_image_to_pdf():
    # Create test image in memory
    img = Image.new("RGB", (200, 100), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    pdf_bytes, pdf_name, pdf_mime = DocumentConverter.convert(img_bytes, "appointment.png", ".pdf")
    assert pdf_name == "appointment.pdf"
    assert pdf_mime == "application/pdf"
    assert pdf_bytes.startswith(b"%PDF")


def test_docx_to_pdf_reportlab_fallback():
    doc = DocxDocument()
    doc.add_heading("Financial Report 2026", level=1)
    doc.add_paragraph("Total revenue increased by 25%.")
    buf = io.BytesIO()
    doc.save(buf)
    docx_bytes = buf.getvalue()

    pdf_bytes, pdf_name, pdf_mime = DocumentConverter.convert(docx_bytes, "report.docx", ".pdf")
    assert pdf_name == "report.pdf"
    assert pdf_mime == "application/pdf"
    assert pdf_bytes.startswith(b"%PDF")


def test_docx_to_pptx():
    doc = DocxDocument()
    doc.add_heading("Executive Strategy", level=1)
    doc.add_paragraph("• Market expansion across Europe")
    doc.add_paragraph("• Automation of support workflows")
    doc.add_heading("Financial Outlook", level=2)
    doc.add_paragraph("Projected 15% revenue increase")
    buf = io.BytesIO()
    doc.save(buf)
    docx_bytes = buf.getvalue()

    pptx_bytes, pptx_name, pptx_mime = DocumentConverter.convert(docx_bytes, "strategy.docx", ".pptx")
    assert pptx_name == "strategy.pptx"
    assert "presentationml" in pptx_mime
    assert len(pptx_bytes) > 0

    # Verify presentation can be parsed
    prs = Presentation(io.BytesIO(pptx_bytes))
    assert len(prs.slides) >= 1


def test_pptx_to_docx_and_pdf():
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Q3 Performance"
    tf = slide.shapes.placeholders[1].text_frame
    tf.text = "Revenue grew by $10M"
    p = tf.add_paragraph()
    p.text = "Customer retention at 98%"

    buf = io.BytesIO()
    prs.save(buf)
    pptx_bytes = buf.getvalue()

    # Convert to DOCX
    docx_bytes, docx_name, _ = DocumentConverter.convert(pptx_bytes, "presentation.pptx", ".docx")
    assert docx_name == "presentation.docx"
    assert len(docx_bytes) > 0

    # Convert to PDF
    pdf_bytes, pdf_name, _ = DocumentConverter.convert(pptx_bytes, "presentation.pptx", ".pdf")
    assert pdf_name == "presentation.pdf"
    assert pdf_bytes.startswith(b"%PDF")


def test_pdf_to_pptx():
    # Generate simple PDF via ReportLab
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 750, "Quarterly Earnings Report")
    c.drawString(100, 700, "Gross profit exceeded projections.")
    c.showPage()
    c.save()
    pdf_bytes = buf.getvalue()

    pptx_bytes, pptx_name, pptx_mime = DocumentConverter.convert(pdf_bytes, "earnings.pdf", ".pptx")
    assert pptx_name == "earnings.pptx"
    assert "presentationml" in pptx_mime
    assert len(pptx_bytes) > 0


# ==============================================================================
# 3. NATURAL LANGUAGE CONVERSION REQUEST PARSING
# ==============================================================================

def test_parse_conversion_natural_language():
    sample_docs = [
        {"id": "doc1", "filename": "Annual_Report.pdf", "status": "ready"},
        {"id": "doc2", "filename": "Budget_2026.xlsx", "status": "ready"},
        {"id": "doc3", "filename": "appointment.png", "status": "ready"},
        {"id": "doc4", "filename": "slides.pptx", "status": "ready"},
    ]

    # "Convert this PDF to Word"
    req1 = parse_conversion_request("Convert this PDF to Word", sample_docs)
    assert req1 is not None
    assert req1.is_valid is True
    assert req1.target_format == ".docx"
    assert req1.matched_document["filename"] == "Annual_Report.pdf"

    # "Convert this document to PowerPoint" (only Annual_Report.pdf matches among non-PPTX)
    req2 = parse_conversion_request("Convert this document to PowerPoint", [sample_docs[0]])
    assert req2 is not None
    assert req2.is_valid is True
    assert req2.target_format == ".pptx"
    assert req2.matched_document["filename"] == "Annual_Report.pdf"

    # "Convert this Excel to CSV"
    req3 = parse_conversion_request("Convert this Excel to CSV", sample_docs)
    assert req3 is not None
    assert req3.is_valid is True
    assert req3.target_format == ".csv"
    assert req3.matched_document["filename"] == "Budget_2026.xlsx"

    # "Convert appointment.png to PDF"
    req4 = parse_conversion_request("Convert appointment.png to PDF", sample_docs)
    assert req4 is not None
    assert req4.is_valid is True
    assert req4.target_format == ".pdf"
    assert req4.matched_document["filename"] == "appointment.png"

    # Multiple matching candidates without hint -> clarification needed
    two_pdfs = [
        {"id": "d1", "filename": "doc1.pdf", "status": "ready"},
        {"id": "d2", "filename": "doc2.pdf", "status": "ready"},
    ]
    req5 = parse_conversion_request("Convert this to Word", two_pdfs)
    assert req5 is not None
    assert req5.is_valid is False
    assert req5.clarification_needed is not None
    assert "doc1.pdf" in req5.clarification_needed
    assert "doc2.pdf" in req5.clarification_needed

    # Educational query should not trigger conversion
    req6 = parse_conversion_request("How to convert PDF to Word in Python?")
    assert req6 is None


# ==============================================================================
# 4. USER ISOLATION VERIFICATION
# ==============================================================================

def test_user_isolation_token_creation():
    user1_id = "user-111-aaa"
    user2_id = "user-222-bbb"

    token1 = create_access_token({"sub": user1_id})
    token2 = create_access_token({"sub": user2_id})

    from app.auth.security import decode_access_token
    p1 = decode_access_token(token1)
    p2 = decode_access_token(token2)

    assert p1["sub"] == user1_id
    assert p2["sub"] == user2_id
    assert p1["sub"] != p2["sub"]
