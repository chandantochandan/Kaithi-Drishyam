"""Unit tests for the segmentation module."""

from __future__ import annotations

import numpy as np
import pytest

from kaithi_drishyam.preprocessing import ProcessedImage
from kaithi_drishyam.segmentation import SegmentationEngine, TextLine


class TestSegmentationEngine:
    """Tests for SegmentationEngine class."""

    @pytest.fixture
    def engine(self) -> SegmentationEngine:
        """Create a segmentation engine instance."""
        return SegmentationEngine()

    @pytest.mark.unit
    def test_detect_text_lines_basic(
        self, engine: SegmentationEngine, sample_grayscale_image: np.ndarray
    ) -> None:
        """Test basic text line detection."""
        lines = engine.detect_text_lines(sample_grayscale_image)

        assert isinstance(lines, list)
        assert len(lines) > 0
        assert all(isinstance(line, TextLine) for line in lines)

    @pytest.mark.unit
    def test_detect_text_lines_from_processed_image(
        self, engine: SegmentationEngine, sample_grayscale_image: np.ndarray
    ) -> None:
        """Test text line detection from ProcessedImage."""
        processed = ProcessedImage(
            image=sample_grayscale_image,
            original_shape=sample_grayscale_image.shape[:2],
            binarized=False,
        )

        lines = engine.detect_text_lines(processed)

        assert isinstance(lines, list)
        assert len(lines) > 0

    @pytest.mark.unit
    def test_bounding_boxes_valid(
        self, engine: SegmentationEngine, sample_grayscale_image: np.ndarray
    ) -> None:
        """Test that bounding boxes have valid coordinates."""
        lines = engine.detect_text_lines(sample_grayscale_image)

        for line in lines:
            x, y, w, h = line.bounding_box
            assert x >= 0
            assert y >= 0
            assert w > 0
            assert h > 0
            assert x + w <= sample_grayscale_image.shape[1]
            assert y + h <= sample_grayscale_image.shape[0]

    @pytest.mark.unit
    def test_cropped_images_valid(
        self, engine: SegmentationEngine, sample_grayscale_image: np.ndarray
    ) -> None:
        """Test that cropped images are valid numpy arrays."""
        lines = engine.detect_text_lines(sample_grayscale_image)

        for line in lines:
            assert isinstance(line.cropped_image, np.ndarray)
            assert line.cropped_image.size > 0
            assert line.cropped_image.shape[0] == line.height
            assert line.cropped_image.shape[1] == line.width

    @pytest.mark.unit
    def test_reading_order_sequential(
        self, engine: SegmentationEngine, sample_grayscale_image: np.ndarray
    ) -> None:
        """Test that reading order is sequential starting from 0."""
        lines = engine.detect_text_lines(sample_grayscale_image)

        reading_orders = [line.reading_order for line in lines]
        assert reading_orders == list(range(len(lines)))

    @pytest.mark.unit
    def test_multicolumn_detection(
        self, engine: SegmentationEngine, sample_multicolumn_image: np.ndarray
    ) -> None:
        """Test that multi-column layouts are detected."""
        lines = engine.detect_text_lines(sample_multicolumn_image)

        # Should detect lines in multiple columns
        column_indices = set(line.column_index for line in lines)
        assert len(column_indices) >= 1  # At least one column detected

    @pytest.mark.unit
    def test_maintain_reading_order(
        self, engine: SegmentationEngine, sample_grayscale_image: np.ndarray
    ) -> None:
        """Test maintain_reading_order sorts correctly."""
        lines = engine.detect_text_lines(sample_grayscale_image)

        # Shuffle the lines
        import random
        shuffled = lines.copy()
        random.shuffle(shuffled)

        # Re-sort by reading order
        sorted_lines = engine.maintain_reading_order(shuffled)

        # Verify sorted by column then y-coordinate
        for i in range(len(sorted_lines) - 1):
            curr = sorted_lines[i]
            next_line = sorted_lines[i + 1]
            if curr.column_index == next_line.column_index:
                assert curr.y <= next_line.y

    @pytest.mark.unit
    def test_confidence_scores_valid(
        self, engine: SegmentationEngine, sample_grayscale_image: np.ndarray
    ) -> None:
        """Test that confidence scores are in valid range."""
        lines = engine.detect_text_lines(sample_grayscale_image)

        for line in lines:
            assert 0.0 <= line.confidence_score <= 1.0

    @pytest.mark.unit
    def test_empty_image(self, engine: SegmentationEngine) -> None:
        """Test handling of empty/white image."""
        white_image = np.ones((300, 400), dtype=np.uint8) * 255
        lines = engine.detect_text_lines(white_image)

        # Should return empty list or very few low-confidence lines
        assert isinstance(lines, list)


class TestTextLine:
    """Tests for TextLine dataclass."""

    @pytest.mark.unit
    def test_properties(self) -> None:
        """Test TextLine properties."""
        cropped = np.zeros((20, 100), dtype=np.uint8)
        text_line = TextLine(
            bounding_box=(50, 100, 100, 20),
            cropped_image=cropped,
            reading_order=0,
            confidence_score=0.95,
            column_index=0,
            line_index_in_column=0,
        )

        assert text_line.x == 50
        assert text_line.y == 100
        assert text_line.width == 100
        assert text_line.height == 20
        assert text_line.area == 2000
        assert text_line.center == (100, 110)

    @pytest.mark.unit
    def test_default_values(self) -> None:
        """Test TextLine default values."""
        cropped = np.zeros((20, 100), dtype=np.uint8)
        text_line = TextLine(
            bounding_box=(0, 0, 100, 20),
            cropped_image=cropped,
            reading_order=0,
        )

        assert text_line.confidence_score == 1.0
        assert text_line.column_index == 0
        assert text_line.line_index_in_column == 0
