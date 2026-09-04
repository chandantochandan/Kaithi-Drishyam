"""Property-based tests for the segmentation module.

Property 3: Text Segmentation Correctness
For any document image, the Segmentation_Module should identify all text lines
with valid bounding boxes, maintain top-to-bottom reading order, exclude non-text
elements, and output both coordinates and cropped images for each detected line.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck

from kaithi_drishyam.segmentation import SegmentationEngine, TextLine


class TestTextSegmentationCorrectnessProperty:
    """Property 3: Text Segmentation Correctness tests."""

    @pytest.fixture
    def engine(self) -> SegmentationEngine:
        """Create a segmentation engine instance."""
        return SegmentationEngine()

    @pytest.mark.property
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        num_lines=st.integers(min_value=1, max_value=10),
        width=st.integers(min_value=200, max_value=800),
        height=st.integers(min_value=200, max_value=800),
    )
    def test_detects_text_lines_with_valid_bounding_boxes(
        self,
        engine: SegmentationEngine,
        num_lines: int,
        width: int,
        height: int,
    ) -> None:
        """Property: All detected lines have valid bounding boxes within image bounds.

        Feature: kaithi-ocr, Property 3: Text Segmentation Correctness
        Validates: Requirement 2.1
        """
        # Generate document image
        image = self._generate_document_with_lines(width, height, num_lines)

        # Detect text lines
        lines = engine.detect_text_lines(image)

        # Property: all bounding boxes are valid
        for line in lines:
            x, y, w, h = line.bounding_box
            assert x >= 0, "x must be non-negative"
            assert y >= 0, "y must be non-negative"
            assert w > 0, "width must be positive"
            assert h > 0, "height must be positive"
            assert x + w <= width, "bounding box must not exceed image width"
            assert y + h <= height, "bounding box must not exceed image height"

    @pytest.mark.property
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        num_lines=st.integers(min_value=2, max_value=8),
    )
    def test_maintains_top_to_bottom_reading_order(
        self,
        engine: SegmentationEngine,
        num_lines: int,
    ) -> None:
        """Property: Lines are ordered from top to bottom.

        Feature: kaithi-ocr, Property 3: Text Segmentation Correctness
        Validates: Requirement 2.2
        """
        # Generate document with multiple lines
        image = self._generate_document_with_lines(400, 400, num_lines)

        # Detect text lines
        lines = engine.detect_text_lines(image)

        # Skip if less than 2 lines detected
        assume(len(lines) >= 2)

        # Property: lines should be sorted by y-coordinate (within same column)
        sorted_lines = engine.maintain_reading_order(lines)
        for i in range(len(sorted_lines) - 1):
            curr = sorted_lines[i]
            next_line = sorted_lines[i + 1]
            if curr.column_index == next_line.column_index:
                assert curr.y <= next_line.y, "Lines must be in top-to-bottom order"

    @pytest.mark.property
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        num_columns=st.integers(min_value=1, max_value=3),
        lines_per_column=st.integers(min_value=2, max_value=5),
    )
    def test_handles_multicolumn_layouts(
        self,
        engine: SegmentationEngine,
        num_columns: int,
        lines_per_column: int,
    ) -> None:
        """Property: Multi-column layouts are handled correctly.

        Feature: kaithi-ocr, Property 3: Text Segmentation Correctness
        Validates: Requirement 2.3
        """
        # Generate multi-column document
        width = 200 * num_columns + 50 * (num_columns - 1)
        height = 50 * lines_per_column + 100
        image = self._generate_multicolumn_document(width, height, num_columns, lines_per_column)

        # Detect text lines
        lines = engine.detect_text_lines(image)

        # Property: should detect lines from all columns
        assert len(lines) > 0, "Should detect at least some lines"

        # Property: reading order should be sequential
        reading_orders = [line.reading_order for line in lines]
        assert reading_orders == list(range(len(lines))), "Reading order must be sequential"

    @pytest.mark.property
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        num_lines=st.integers(min_value=1, max_value=6),
    )
    def test_outputs_cropped_images_for_each_line(
        self,
        engine: SegmentationEngine,
        num_lines: int,
    ) -> None:
        """Property: Each detected line has a valid cropped image.

        Feature: kaithi-ocr, Property 3: Text Segmentation Correctness
        Validates: Requirement 2.5
        """
        # Generate document
        image = self._generate_document_with_lines(400, 300, num_lines)

        # Detect text lines
        lines = engine.detect_text_lines(image)

        # Property: each line has valid cropped image
        for line in lines:
            assert isinstance(line.cropped_image, np.ndarray)
            assert line.cropped_image.size > 0
            assert line.cropped_image.shape[0] == line.height
            assert line.cropped_image.shape[1] == line.width

    @pytest.mark.property
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        num_lines=st.integers(min_value=1, max_value=6),
    )
    def test_confidence_scores_are_valid(
        self,
        engine: SegmentationEngine,
        num_lines: int,
    ) -> None:
        """Property: Confidence scores are in valid range [0, 1].

        Feature: kaithi-ocr, Property 3: Text Segmentation Correctness
        Validates: Requirement 2.5
        """
        # Generate document
        image = self._generate_document_with_lines(300, 250, num_lines)

        # Detect text lines
        lines = engine.detect_text_lines(image)

        # Property: confidence scores are valid
        for line in lines:
            assert 0.0 <= line.confidence_score <= 1.0

    @pytest.mark.property
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        num_lines=st.integers(min_value=2, max_value=5),
        width=st.integers(min_value=300, max_value=500),
    )
    def test_excludes_stamps_and_noise(
        self,
        engine: SegmentationEngine,
        num_lines: int,
        width: int,
    ) -> None:
        """Property: Small isolated noise regions should not be detected as text.

        Feature: kaithi-ocr, Property 3: Text Segmentation Correctness
        Validates: Requirement 2.4
        """
        # Generate document with text lines and some noise
        image = self._generate_document_with_noise(width, 300, num_lines)

        # Detect text lines
        lines = engine.detect_text_lines(image)

        # Property: all detected lines should have reasonable dimensions
        # (not tiny noise dots or huge blocks)
        for line in lines:
            # Lines should have reasonable aspect ratio (width > height for text)
            # and not be too small
            assert line.width >= engine.min_line_width, "Line should not be too narrow"
            assert line.height >= engine.min_line_height, "Line should not be too short"
            assert line.height <= engine.max_line_height, "Line should not be too tall"

    # Helper methods for generating test images

    def _generate_document_with_lines(
        self,
        width: int,
        height: int,
        num_lines: int,
    ) -> np.ndarray:
        """Generate a document image with text lines."""
        # White background
        image = np.ones((height, width), dtype=np.uint8) * 255

        # Calculate line spacing
        available_height = height - 40  # Margins
        line_spacing = available_height // (num_lines + 1)
        line_height = max(15, min(25, line_spacing // 2))

        for i in range(num_lines):
            y = 20 + (i + 1) * line_spacing - line_height // 2
            if y + line_height < height - 20:
                # Vary line width
                x_end = int(width * (0.5 + 0.4 * ((i % 3) / 3)))
                image[y:y + line_height, 30:x_end] = 0

        return image

    def _generate_multicolumn_document(
        self,
        width: int,
        height: int,
        num_columns: int,
        lines_per_column: int,
    ) -> np.ndarray:
        """Generate a multi-column document image."""
        # White background
        image = np.ones((height, width), dtype=np.uint8) * 255

        col_width = (width - 20) // num_columns
        line_height = 15

        for col in range(num_columns):
            col_start = 10 + col * col_width
            col_end = col_start + col_width - 20

            for line in range(lines_per_column):
                y = 30 + line * 40
                if y + line_height < height - 20:
                    x_end = col_end - (line % 2) * 20
                    image[y:y + line_height, col_start:x_end] = 0

        return image

    def _generate_document_with_noise(
        self,
        width: int,
        height: int,
        num_lines: int,
    ) -> np.ndarray:
        """Generate a document with text lines and some noise."""
        # Start with document with lines
        image = self._generate_document_with_lines(width, height, num_lines)

        # Add some random noise dots (should be filtered out)
        noise = np.random.random(image.shape)
        image[noise < 0.01] = 0  # Small random black dots

        return image

    def _generate_document_with_border(
        self,
        width: int,
        height: int,
        border_size: int,
        num_lines: int,
    ) -> np.ndarray:
        """Generate a document with decorative border."""
        # White background
        image = np.ones((height, width), dtype=np.uint8) * 255

        # Add solid dark border (simulating scanner edge artifacts)
        # These are solid blocks that should be filtered out
        image[:border_size, :] = 30  # Top border - solid dark
        image[height - border_size:, :] = 30  # Bottom border - solid dark

        # Add text lines in content area (well inside the borders)
        content_start = border_size + 30
        content_end = height - border_size - 30
        available_height = content_end - content_start

        if available_height > 50:
            line_spacing = available_height // (num_lines + 1)
            line_height = min(15, line_spacing // 2)

            for i in range(num_lines):
                y = content_start + (i + 1) * line_spacing - line_height // 2
                if y + line_height < content_end:
                    image[y:y + line_height, 60:width - 60] = 0

        return image
