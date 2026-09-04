"""
Kaithi-Drishyam: OCR system for digitizing historical Kaithi script land records.

This package provides an end-to-end pipeline for:
1. Pre-processing historical document images
2. Segmenting text lines from documents
3. Recognizing Kaithi script using CRNN models
4. Transliterating Kaithi to Devanagari
5. Modernizing archaic Hindi to modern Hindi
"""

__version__ = "0.1.0"
__author__ = "Kaithi-Drishyam Team"

from kaithi_drishyam.config import Settings

__all__ = ["Settings", "__version__"]
