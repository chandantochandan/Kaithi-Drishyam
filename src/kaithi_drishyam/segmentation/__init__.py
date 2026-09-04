"""
Segmentation module for text line detection.

This module handles:
- Text line identification with bounding boxes
- Reading order maintenance (top-to-bottom)
- Multi-column layout handling
- Non-text element exclusion
"""

from kaithi_drishyam.segmentation.engine import SegmentationEngine
from kaithi_drishyam.segmentation.models import TextLine

__all__ = ["SegmentationEngine", "TextLine"]
