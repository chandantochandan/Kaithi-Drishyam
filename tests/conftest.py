"""Pytest configuration and shared fixtures for Kaithi-Drishyam tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pytest

from kaithi_drishyam.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Provide test settings."""
    return Settings()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_grayscale_image() -> np.ndarray:
    """Create a sample grayscale test image with text-like features."""
    # Create a white background
    image = np.ones((300, 400), dtype=np.uint8) * 255

    # Add some black "text" lines
    cv2.rectangle(image, (50, 50), (350, 70), 0, -1)
    cv2.rectangle(image, (50, 90), (300, 110), 0, -1)
    cv2.rectangle(image, (50, 130), (320, 150), 0, -1)
    cv2.rectangle(image, (50, 170), (280, 190), 0, -1)

    return image


@pytest.fixture
def sample_color_image() -> np.ndarray:
    """Create a sample color test image."""
    # Create grayscale first
    gray = np.ones((300, 400), dtype=np.uint8) * 255
    cv2.rectangle(gray, (50, 50), (350, 70), 0, -1)
    cv2.rectangle(gray, (50, 90), (300, 110), 0, -1)

    # Convert to BGR
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def sample_noisy_image() -> np.ndarray:
    """Create a sample image with noise."""
    image = np.ones((300, 400), dtype=np.uint8) * 255

    # Add text lines
    cv2.rectangle(image, (50, 50), (350, 70), 0, -1)
    cv2.rectangle(image, (50, 100), (300, 120), 0, -1)

    # Add salt-and-pepper noise
    noise = np.random.random(image.shape)
    image[noise < 0.02] = 0
    image[noise > 0.98] = 255

    return image


@pytest.fixture
def sample_skewed_image() -> np.ndarray:
    """Create a sample skewed test image."""
    # Create base image with text
    image = np.ones((400, 500), dtype=np.uint8) * 255
    cv2.rectangle(image, (50, 100), (450, 130), 0, -1)
    cv2.rectangle(image, (50, 160), (400, 190), 0, -1)
    cv2.rectangle(image, (50, 220), (420, 250), 0, -1)

    # Apply rotation (5 degrees skew)
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 5.0, 1.0)
    skewed = cv2.warpAffine(
        image,
        rotation_matrix,
        (image.shape[1], image.shape[0]),
        borderValue=255,
    )

    return skewed


@pytest.fixture
def sample_multicolumn_image() -> np.ndarray:
    """Create a sample image with multiple text columns."""
    image = np.ones((400, 600), dtype=np.uint8) * 255

    # Left column
    cv2.rectangle(image, (30, 50), (250, 70), 0, -1)
    cv2.rectangle(image, (30, 90), (240, 110), 0, -1)
    cv2.rectangle(image, (30, 130), (260, 150), 0, -1)

    # Right column
    cv2.rectangle(image, (320, 50), (570, 70), 0, -1)
    cv2.rectangle(image, (320, 90), (550, 110), 0, -1)
    cv2.rectangle(image, (320, 130), (560, 150), 0, -1)

    return image


@pytest.fixture
def sample_image_file(temp_dir: Path, sample_grayscale_image: np.ndarray) -> Path:
    """Create a sample image file."""
    image_path = temp_dir / "test_document.png"
    cv2.imwrite(str(image_path), sample_grayscale_image)
    return image_path


@pytest.fixture
def sample_jpeg_file(temp_dir: Path, sample_grayscale_image: np.ndarray) -> Path:
    """Create a sample JPEG image file."""
    image_path = temp_dir / "test_document.jpg"
    cv2.imwrite(str(image_path), sample_grayscale_image)
    return image_path


@pytest.fixture
def sample_tiff_file(temp_dir: Path, sample_grayscale_image: np.ndarray) -> Path:
    """Create a sample TIFF image file."""
    image_path = temp_dir / "test_document.tiff"
    cv2.imwrite(str(image_path), sample_grayscale_image)
    return image_path


# Hypothesis settings for property-based tests
from hypothesis import settings as hypothesis_settings

hypothesis_settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,
)

hypothesis_settings.register_profile(
    "dev",
    max_examples=10,
    deadline=None,
)

hypothesis_settings.register_profile(
    "exhaustive",
    max_examples=500,
    deadline=None,
)


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers and settings."""
    # Register custom markers
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for combined components")
    config.addinivalue_line("markers", "property: Property-based tests using Hypothesis")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")
    config.addinivalue_line("markers", "gpu: Tests that require GPU")
