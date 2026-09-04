"""Data models for preprocessing module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np


@dataclass
class ProcessedImage:
    """Represents a preprocessed document image.

    Attributes:
        image: The preprocessed image as a numpy array (grayscale or binary).
        original_shape: Original image dimensions before processing.
        deskew_angle: Detected and corrected skew angle in degrees.
        noise_level: Estimated noise level (0.0 to 1.0).
        binarized: Whether the image has been binarized.
        preprocessing_metadata: Additional metadata from preprocessing steps.
    """

    image: np.ndarray
    original_shape: tuple[int, int]
    deskew_angle: float = 0.0
    noise_level: float = 0.0
    binarized: bool = False
    preprocessing_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        """Get image height."""
        return self.image.shape[0]

    @property
    def width(self) -> int:
        """Get image width."""
        return self.image.shape[1]

    @property
    def shape(self) -> tuple[int, ...]:
        """Get current image shape."""
        return self.image.shape
