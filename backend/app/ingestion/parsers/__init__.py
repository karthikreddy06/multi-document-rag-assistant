"""
Multi-format document parsers sub-package.

Exports the ParserRegistry (main entry point) and all concrete parsers
for direct import when needed.
"""

from app.ingestion.parsers.base import BaseParser
from app.ingestion.parsers.registry import ParserRegistry, UnsupportedFileTypeError
from app.ingestion.parsers.pdf_parser import PDFParser
from app.ingestion.parsers.docx_parser import DOCXParser
from app.ingestion.parsers.pptx_parser import PPTXParser
from app.ingestion.parsers.xlsx_parser import XLSXParser
from app.ingestion.parsers.csv_parser import CSVParser
from app.ingestion.parsers.text_parser import TextParser
from app.ingestion.parsers.image_parser import ImageParser

__all__ = [
    "BaseParser",
    "ParserRegistry",
    "UnsupportedFileTypeError",
    "PDFParser",
    "DOCXParser",
    "PPTXParser",
    "XLSXParser",
    "CSVParser",
    "TextParser",
    "ImageParser",
]
