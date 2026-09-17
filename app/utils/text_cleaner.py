"""
Text Preprocessing and Cleaning Utilities.
Normalizes extracted text, fixes PDF kerning artifacts, and handles unicode characters safely.
"""

import re


def clean_extracted_text(text: str) -> str:
    """
    Clean and normalize raw text extracted from PDF documents.
    """
    if not text:
        return ""

    # Replace null bytes
    text = text.replace("\x00", "")

    # Normalize unicode spaces
    text = text.replace("\xa0", " ")

    # Normalize unicode dashes and quotes
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # Normalize bullet points and unknown replacement characters
    text = text.replace("\u2022", "• ").replace("\ufffd", "- ")

    # Generic PDF kerning & typography repairs:
    # 1. Degree / abbreviation spacing: 'B. T ech' -> 'B.Tech', 'M. S c' -> 'M.Sc'
    text = re.sub(r"\b([A-Z]\.)\s*([A-Z])\s+([a-z]+)\b", r"\1\2\3", text)
    text = re.sub(r"\b([A-Z]\.)\s*([A-Z]\b)", r"\1\2", text)

    # 2. Drop-cap or split first letter of word / camel-case: 'T ravel' -> 'Travel', 'F ull' -> 'Full'
    # Run twice to handle adjacent repairs like 'T ravelT rack' -> 'TravelTrack'
    dropcap_pattern = r"(?:^|\b|(?<=[a-z]))([A-Z])\s+([a-z]{2,})(?=[A-Z\b\s\.,;:\-\(\)\[\]]|$)"
    text = re.sub(dropcap_pattern, r"\1\2", text)
    text = re.sub(dropcap_pattern, r"\1\2", text)

    # 3. Acronym / uppercase token split: 'CGP A' -> 'CGPA', 'HTM L' -> 'HTML'
    text = re.sub(r"\b([A-Z]{2,})\s+([A-Z])\b", r"\1\2", text)

    # Clean up multiple spaces on a line (excluding newlines)
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize multiple newlines (keep max 2 newlines)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()
