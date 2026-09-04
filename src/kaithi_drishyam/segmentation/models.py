"""Data models for segmentation module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class TextLine:
    """Represents a detected text line in a document.

    Attributes:
        bounding_box: Tuple of (x, y, width, height) for the text line region.
        cropped_image: The extracted text line image as numpy array.
        reading_order: Position in document reading order (0-indexed).
        confidence_score: Detection confidence (0.0 to 1.0).
        column_index: Column index for multi-column layouts (0-indexed).
        line_index_in_column: Line index within the column.
    """

    bounding_box: tuple[int, int, int, int]
    cropped_image: np.ndarray
    reading_order: int
    confidence_score: float = 1.0
    column_index: int = 0
    line_index_in_column: int = 0

    @property
    def x(self) -> int:
        """Get x coordinate of bounding box."""
        return self.bounding_box[0]

    @property
    def y(self) -> int:
        """Get y coordinate of bounding box."""
        return self.bounding_box[1]

    @property
    def width(self) -> int:
        """Get width of bounding box."""
        return self.bounding_box[2]

    @property
    def height(self) -> int:
        """Get height of bounding box."""
        return self.bounding_box[3]

    @property
    def area(self) -> int:
        """Get area of bounding box."""
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        """Get center point of bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)
