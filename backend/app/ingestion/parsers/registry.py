"""
Parser Registry — maps file extensions to concrete parser instances.
Central dispatch point for the multi-format ingestion architecture.
"""

from pathlib import Path
from typing import Dict, List, Optional

from app.ingestion.parsers.base import BaseParser
from app.ingestion.parsers.csv_parser import CSVParser
from app.ingestion.parsers.docx_parser import DOCXParser
from app.ingestion.parsers.image_parser import ImageParser
from app.ingestion.parsers.pdf_parser import PDFParser
from app.ingestion.parsers.pptx_parser import PPTXParser
from app.ingestion.parsers.text_parser import TextParser
from app.ingestion.parsers.xlsx_parser import XLSXParser
from app.models import Document
from app.utils.logger import setup_logger

logger = setup_logger("ingestion.parsers.registry")


class UnsupportedFileTypeError(ValueError):
    """Raised when no parser is registered for a given file extension."""


class ParserRegistry:
    """
    Central registry that maps file extensions to parser instances.

    Usage::

        registry = ParserRegistry()
        docs = registry.parse(file_path, original_filename, file_hash)

    Custom parsers can be registered at runtime::

        registry.register(".myext", MyCustomParser())
    """

    def __init__(self, vision_hook=None):
        """
        Args:
            vision_hook: Optional callable for image vision/OCR passed to ImageParser.
        """
        self._parsers: Dict[str, BaseParser] = {}
        self._register_defaults(vision_hook=vision_hook)

    def _register_defaults(self, vision_hook=None) -> None:
        """Register all built-in parsers for their supported extensions."""
        default_parsers: List[BaseParser] = [
            PDFParser(),
            DOCXParser(),
            PPTXParser(),
            XLSXParser(),
            CSVParser(),
            TextParser(),
            ImageParser(vision_hook=vision_hook),
        ]
        for parser in default_parsers:
            for ext in parser.SUPPORTED_EXTENSIONS:
                self._parsers[ext.lower()] = parser

    def register(self, extension: str, parser: BaseParser) -> None:
        """
        Register a custom parser for a file extension.

        Args:
            extension: File extension including the dot, e.g. ``".myext"``.
            parser: A BaseParser instance.
        """
        self._parsers[extension.lower()] = parser
        logger.info(f"ParserRegistry: registered {type(parser).__name__} for '{extension}'")

    @property
    def supported_extensions(self) -> frozenset:
        """Return a frozenset of all currently supported file extensions."""
        return frozenset(self._parsers.keys())

    def get_parser(self, extension: str) -> BaseParser:
        """
        Return the parser for *extension*.

        Raises:
            UnsupportedFileTypeError: If no parser handles this extension.
        """
        parser = self._parsers.get(extension.lower())
        if parser is None:
            supported = ", ".join(sorted(self.supported_extensions))
            raise UnsupportedFileTypeError(
                f"No parser registered for extension '{extension}'. "
                f"Supported types: {supported}"
            )
        return parser

    def is_supported(self, extension: str) -> bool:
        """Return True if *extension* has a registered parser."""
        return extension.lower() in self._parsers

    def parse(
        self,
        file_path: Path,
        original_filename: str,
        file_hash: str,
    ) -> List[Document]:
        """
        Detect the file type and dispatch to the appropriate parser.

        Args:
            file_path: Absolute path to the file on disk.
            original_filename: The user-facing filename.
            file_hash: Pre-computed SHA-256 hex digest.

        Returns:
            List of Document objects ready for chunking and embedding.

        Raises:
            UnsupportedFileTypeError: If the file extension is not supported.
        """
        ext = Path(original_filename).suffix.lower()
        if not ext:
            ext = Path(file_path).suffix.lower()

        parser = self.get_parser(ext)
        logger.info(
            f"ParserRegistry: dispatching '{original_filename}' "
            f"(ext='{ext}') to {type(parser).__name__}"
        )
        return parser.parse(file_path, original_filename, file_hash)
