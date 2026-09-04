"""
Transliteration module for Kaithi to Devanagari conversion.

This module handles:
- Character-level Kaithi to Devanagari mapping
- Legal terminology glossary
- KenLM-based error correction
- Bhashini NMT integration for modernization
"""

from kaithi_drishyam.transliteration.service import (
    KAITHI_TO_DEVANAGARI,
    LEGAL_GLOSSARY,
    TransliterationResult,
    TransliterationService,
)

__all__ = [
    "KAITHI_TO_DEVANAGARI",
    "LEGAL_GLOSSARY",
    "TransliterationResult",
    "TransliterationService",
]
