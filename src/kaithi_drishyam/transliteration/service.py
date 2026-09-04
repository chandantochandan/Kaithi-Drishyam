"""Kaithi to Devanagari transliteration service."""

from __future__ import annotations

from dataclasses import dataclass, field


KAITHI_TO_DEVANAGARI = {
    "𑂃": "अ",
    "𑂄": "आ",
    "𑂅": "इ",
    "𑂆": "ई",
    "𑂇": "उ",
    "𑂈": "ऊ",
    "𑂉": "ए",
    "𑂊": "ऐ",
    "𑂋": "ओ",
    "𑂌": "औ",
    "𑂍": "क",
    "𑂎": "ख",
    "𑂏": "ग",
    "𑂐": "घ",
    "𑂑": "ङ",
    "𑂒": "च",
    "𑂓": "छ",
    "𑂔": "ज",
    "𑂕": "झ",
    "𑂖": "ञ",
    "𑂗": "ट",
    "𑂘": "ठ",
    "𑂙": "ड",
    "𑂚": "ढ",
    "𑂛": "ण",
    "𑂜": "त",
    "𑂝": "थ",
    "𑂞": "द",
    "𑂟": "ध",
    "𑂠": "न",
    "𑂡": "प",
    "𑂢": "फ",
    "𑂣": "ब",
    "𑂤": "भ",
    "𑂥": "म",
    "𑂦": "य",
    "𑂧": "र",
    "𑂨": "ल",
    "𑂩": "व",
    "𑂪": "श",
    "𑂫": "ष",
    "𑂬": "स",
    "𑂭": "ह",
    "𑂮": "़",
    "𑂯": "ऽ",
    "𑂰": "ा",
    "𑂱": "ि",
    "𑂲": "ी",
    "𑂳": "ु",
    "𑂴": "ू",
    "𑂵": "े",
    "𑂶": "ै",
    "𑂷": "ो",
    "𑂸": "ौ",
    "𑂹": "्",
}

LEGAL_GLOSSARY = {
    "मौजा": "राजस्व ग्राम",
    "खेसरा": "भूमि खाता/प्लॉट संख्या",
    "जमीन": "भूमि",
    "रैयत": "कृषक/भूमिधारी",
}


@dataclass
class TransliterationResult:
    """Transliteration and legal-term modernization output."""

    devanagari_text: str
    modern_hindi_text: str
    legal_term_mappings: dict[str, str] = field(default_factory=dict)
    unmapped_characters: list[str] = field(default_factory=list)


class TransliterationService:
    """Convert recognized Kaithi text into Devanagari/modern Hindi."""

    def transliterate(self, kaithi_text: str) -> TransliterationResult:
        """Transliterate Kaithi script characters to Devanagari."""
        output: list[str] = []
        unmapped: list[str] = []

        for character in kaithi_text:
            if character in KAITHI_TO_DEVANAGARI:
                output.append(KAITHI_TO_DEVANAGARI[character])
            else:
                output.append(character)
                if not character.isspace() and character not in {".", ",", ":", ";", "-"}:
                    unmapped.append(character)

        devanagari = "".join(output)
        modern_hindi, mappings = self.apply_legal_glossary(devanagari)
        return TransliterationResult(
            devanagari_text=devanagari,
            modern_hindi_text=modern_hindi,
            legal_term_mappings=mappings,
            unmapped_characters=sorted(set(unmapped)),
        )

    def apply_legal_glossary(self, text: str) -> tuple[str, dict[str, str]]:
        """Append modern legal meanings for known historical terms."""
        mappings: dict[str, str] = {}
        modern = text
        for term, meaning in LEGAL_GLOSSARY.items():
            if term in text:
                mappings[term] = meaning
        return modern, mappings
