"""Unit tests for Kaithi transliteration."""

from __future__ import annotations

from kaithi_drishyam.transliteration import TransliterationService


def test_transliterates_kaithi_characters_to_devanagari() -> None:
    """Known Kaithi characters should map to Devanagari."""
    result = TransliterationService().transliterate("𑂍𑂰𑂧")

    assert result.devanagari_text == "कार"


def test_preserves_unmapped_characters_and_reports_them() -> None:
    """Unknown characters should pass through and be reported."""
    result = TransliterationService().transliterate("𑂍?")

    assert result.devanagari_text == "क?"
    assert result.unmapped_characters == ["?"]


def test_legal_glossary_detects_known_terms() -> None:
    """Known legal terms should be reported with modern meanings."""
    result = TransliterationService().transliterate("मौजा खेसरा")

    assert result.legal_term_mappings["मौजा"] == "राजस्व ग्राम"
    assert result.legal_term_mappings["खेसरा"] == "भूमि खाता/प्लॉट संख्या"
