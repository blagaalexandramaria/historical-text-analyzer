"""
Module responsible for text preprocessing in the Historical Text Analyzer.

Includes:
- reading multiple file formats (.txt, .docx, .pdf)
- cleaning and normalizing text
- tokenization
- stop-word removal with fallback support
"""

import json
import os
import re
from typing import Callable

from stop_words import STOP_WORDS


# Get absolute path to current directory (used for loading JSON files reliably)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STOP_WORDS_PATH = os.path.join(ROOT_DIR, "stop_words.json")


def load_stop_words() -> set[str]:
    """
    Loads stop words from JSON file.
    Falls back to hardcoded list if file cannot be read.
    """
    try:
        with open(STOP_WORDS_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
            # Normalize all stop words to lowercase
            return {str(word).lower() for word in data}
    except Exception:
        # Fallback to Python list if JSON is missing/corrupted
        return set(STOP_WORDS)


class MissingDependencyError(RuntimeError):
    """Raised when an optional dependency (docx/pdf) is not installed."""
    pass


def _read_txt(filepath: str) -> str:
    """Reads plain text files."""
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()


def _read_docx(filepath: str) -> str:
    """
    Reads .docx files using python-docx.
    Raises error if dependency is missing.
    """
    try:
        import docx  # type: ignore
    except Exception as exc:
        raise MissingDependencyError(
            "Reading .docx requires 'python-docx'. Install with: pip install python-docx"
        ) from exc

    document = docx.Document(filepath)

    # Extract text from all paragraphs
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def _read_pdf(filepath: str) -> str:
    """
    Reads .pdf files using PyPDF2.
    Extracts text page by page.
    """
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception as exc:
        raise MissingDependencyError(
            "Reading .pdf requires 'PyPDF2'. Install with: pip install PyPDF2"
        ) from exc

    reader = PdfReader(filepath)
    parts: list[str] = []

    # Extract text from each page
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            parts.append(text)

    return "\n".join(parts)


def read_text_file(filepath: str) -> str:
    """
    Detects file type and uses the appropriate reader function.
    Supports: .txt, .docx, .pdf
    """
    ext = os.path.splitext(filepath)[1].lower()

    # Map file extensions to corresponding reader functions
    readers: dict[str, Callable[[str], str]] = {
        ".txt": _read_txt,
        ".docx": _read_docx,
        ".pdf": _read_pdf,
    }

    if ext not in readers:
        raise ValueError(f"Unsupported file type: {ext}")

    return readers[ext](filepath)


def clean_text(text: str) -> str:
    """
    Cleans raw text by:
    - converting to lowercase
    - removing punctuation
    - normalizing whitespace

    Keeps digits (important for historical years like 1918).
    """
    text = text.lower()

    # Remove punctuation but keep letters, digits and spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str) -> list[str]:
    """Splits cleaned text into words."""
    return text.split()


def remove_stop_words(words: list[str]) -> list[str]:
    """
    Filters tokens by:
    - removing stop words
    - removing short/irrelevant words
    - keeping only meaningful historical tokens

    Special rule:
    - keeps 4-digit numbers (likely years: 1914, 1918, etc.)
    """
    stop_words = load_stop_words()
    filtered: list[str] = []

    for word in words:

        # Skip common stop words
        if word in stop_words:
            continue

        # Keep only 4-digit numbers (historical years)
        if word.isdigit():
            if len(word) == 4:
                filtered.append(word)
            continue

        # Remove very short words (usually not meaningful)
        if len(word) <= 2:
            continue

        filtered.append(word)

    return filtered


def process_text(filepath: str) -> list[str]:
    """
    Full preprocessing pipeline:
    file → clean → tokenize → filter

    Returns a list of meaningful words ready for NLP analysis.
    """
    text = read_text_file(filepath)
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    filtered = remove_stop_words(tokens)
    return filtered