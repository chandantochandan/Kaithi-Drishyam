"""
Pre-processing module for document image enhancement.

This module handles:
- Image format validation and loading
- Deskewing rotated documents
- Binarization for optimal recognition
- Noise reduction using Bhashini Udyat API and Albumentations
- Border artifact removal
"""

from kaithi_drishyam.preprocessing.processor import DocumentProcessor
from kaithi_drishyam.preprocessing.models import ProcessedImage
from kaithi_drishyam.preprocessing.bhashini_client import BhashiniUdyatClient, BhashiniAPIError

__all__ = ["DocumentProcessor", "ProcessedImage", "BhashiniUdyatClient", "BhashiniAPIError"]
