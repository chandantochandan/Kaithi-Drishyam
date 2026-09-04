"""Unit tests for the preprocessing module."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from kaithi_drishyam.preprocessing import DocumentProcessor, ProcessedImage


class TestDocumentProcessor:
    """Tests for DocumentProcessor class."""

    @pytest.fixture
    def processor(self) -> DocumentProcessor:
        """Create a document processor instance."""
        return DocumentProcessor(use_bhashini_denoiser=False)

    @pytest.mark.unit
    def test_preprocess_grayscale_image(
        self, processor: DocumentProcessor, sample_grayscale_image: np.ndarray
    ) -> None:
        """Test preprocessing a grayscale image."""
        result = processor.preprocess_image(sample_grayscale_image)

        assert isinstance(result, ProcessedImage)
        assert result.image is not None
        assert result.binarized is True
        assert "binarization" in result.preprocessing_metadata["steps_applied"]

    @pytest.mark.unit
    def test_preprocess_color_image(
        self, processor: DocumentProcessor, sample_color_image: np.ndarray
    ) -> None:
        """Test preprocessing a color image converts to grayscale."""
        result = processor.preprocess_image(sample_color_image)

        assert isinstance(result, ProcessedImage)
        assert len(result.image.shape) == 2  # Should be grayscale
        assert "grayscale_conversion" in result.preprocessing_metadata["steps_applied"]

    @pytest.mark.unit
    def test_preprocess_from_file(
        self, processor: DocumentProcessor, sample_image_file: Path
    ) -> None:
        """Test preprocessing from file path."""
        result = processor.preprocess_image(sample_image_file)

        assert isinstance(result, ProcessedImage)
        assert result.image is not None

    @pytest.mark.unit
    def test_supported_formats(
        self,
        processor: DocumentProcessor,
        sample_jpeg_file: Path,
        sample_tiff_file: Path,
    ) -> None:
        """Test that all supported formats can be loaded."""
        # Test JPEG
        result_jpeg = processor.preprocess_image(sample_jpeg_file)
        assert isinstance(result_jpeg, ProcessedImage)

        # Test TIFF
        result_tiff = processor.preprocess_image(sample_tiff_file)
        assert isinstance(result_tiff, ProcessedImage)

    @pytest.mark.unit
    def test_unsupported_format(
        self, processor: DocumentProcessor, temp_dir: Path
    ) -> None:
        """Test that unsupported formats raise ValueError."""
        fake_path = temp_dir / "test.bmp"
        fake_path.touch()

        with pytest.raises(ValueError, match="Unsupported image format"):
            processor.preprocess_image(fake_path)

    @pytest.mark.unit
    def test_nonexistent_file(self, processor: DocumentProcessor) -> None:
        """Test that nonexistent files raise ValueError."""
        with pytest.raises(ValueError, match="Image file not found"):
            processor.preprocess_image(Path("/nonexistent/path.png"))

    @pytest.mark.unit
    def test_empty_image(self, processor: DocumentProcessor) -> None:
        """Test that empty images raise ValueError."""
        empty_image = np.array([], dtype=np.uint8)

        with pytest.raises(ValueError, match="Invalid or empty image"):
            processor.preprocess_image(empty_image)

    @pytest.mark.unit
    def test_deskew_detection(
        self, processor: DocumentProcessor, sample_skewed_image: np.ndarray
    ) -> None:
        """Test that skewed images are detected and corrected."""
        result = processor.preprocess_image(sample_skewed_image, apply_deskew=True)

        assert "deskewing" in result.preprocessing_metadata["steps_applied"]
        # The deskew angle should be close to the original 5 degrees
        # (within tolerance due to detection variance)

    @pytest.mark.unit
    def test_skip_deskew(
        self, processor: DocumentProcessor, sample_grayscale_image: np.ndarray
    ) -> None:
        """Test that deskewing can be skipped."""
        result = processor.preprocess_image(
            sample_grayscale_image, apply_deskew=False
        )

        assert "deskewing" not in result.preprocessing_metadata["steps_applied"]

    @pytest.mark.unit
    def test_skip_binarization(
        self, processor: DocumentProcessor, sample_grayscale_image: np.ndarray
    ) -> None:
        """Test that binarization can be skipped."""
        result = processor.preprocess_image(
            sample_grayscale_image, apply_binarization=False
        )

        assert result.binarized is False
        assert "binarization" not in result.preprocessing_metadata["steps_applied"]

    @pytest.mark.unit
    def test_noise_level_estimation(
        self, processor: DocumentProcessor, sample_noisy_image: np.ndarray
    ) -> None:
        """Test that noise level is estimated."""
        result = processor.preprocess_image(sample_noisy_image)

        assert 0.0 <= result.noise_level <= 1.0

    @pytest.mark.unit
    def test_original_shape_preserved(
        self, processor: DocumentProcessor, sample_grayscale_image: np.ndarray
    ) -> None:
        """Test that original shape is recorded."""
        result = processor.preprocess_image(sample_grayscale_image)

        assert result.original_shape == sample_grayscale_image.shape[:2]


class TestProcessedImage:
    """Tests for ProcessedImage dataclass."""

    @pytest.mark.unit
    def test_properties(self) -> None:
        """Test ProcessedImage properties."""
        image = np.zeros((100, 200), dtype=np.uint8)
        processed = ProcessedImage(
            image=image,
            original_shape=(150, 250),
            deskew_angle=2.5,
            noise_level=0.1,
            binarized=True,
        )

        assert processed.height == 100
        assert processed.width == 200
        assert processed.shape == (100, 200)
        assert processed.deskew_angle == 2.5
        assert processed.noise_level == 0.1
        assert processed.binarized is True
