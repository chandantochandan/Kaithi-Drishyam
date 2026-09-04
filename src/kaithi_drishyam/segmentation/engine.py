"""Text line segmentation engine."""

from __future__ import annotations

from typing import List, Tuple, Union

import cv2
import numpy as np
from loguru import logger

from kaithi_drishyam.preprocessing.models import ProcessedImage
from kaithi_drishyam.segmentation.models import TextLine


class SegmentationEngine:
    """Handles text line detection and segmentation.

    This engine implements text line segmentation:
    1. Detect text regions using projection profiles
    2. Identify individual text lines with bounding boxes
    3. Handle multi-column layouts
    4. Maintain reading order (top-to-bottom, left-to-right)
    5. Exclude non-text elements (stamps, decorative borders)
    """

    def __init__(
        self,
        min_line_height: int = 10,
        max_line_height: int = 200,
        min_line_width: int = 50,
        column_gap_threshold: float = 0.1,
    ):
        """Initialize the segmentation engine.

        Args:
            min_line_height: Minimum height in pixels for a valid text line.
            max_line_height: Maximum height in pixels for a valid text line.
            min_line_width: Minimum width in pixels for a valid text line.
            column_gap_threshold: Threshold for detecting column gaps (fraction of image width).
        """
        self.min_line_height = min_line_height
        self.max_line_height = max_line_height
        self.min_line_width = min_line_width
        self.column_gap_threshold = column_gap_threshold

    def detect_text_lines(
        self,
        image: Union[ProcessedImage, np.ndarray],
    ) -> list[TextLine]:
        """Detect and extract text lines from a document image.

        Args:
            image: ProcessedImage or numpy array of the document.

        Returns:
            List of TextLine objects sorted by reading order.
        """
        # Extract numpy array from ProcessedImage if needed
        if isinstance(image, ProcessedImage):
            img_array = image.image
        else:
            img_array = image

        # Ensure binary image
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        if img_array.max() > 1:
            _, img_array = cv2.threshold(
                img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        # Detect columns first
        columns = self._detect_columns(img_array)

        # Detect lines within each column
        all_lines: list[TextLine] = []
        reading_order = 0

        for col_idx, (col_start, col_end) in enumerate(columns):
            col_image = img_array[:, col_start:col_end]
            line_regions = self._detect_line_regions(col_image)

            for line_idx, (y_start, y_end) in enumerate(line_regions):
                # Refine horizontal bounds within the line
                line_img = col_image[y_start:y_end, :]
                x_start, x_end = self._refine_horizontal_bounds(line_img)

                if x_end - x_start < self.min_line_width:
                    continue

                # Crop the text line
                cropped = col_image[y_start:y_end, x_start:x_end]

                # Calculate confidence based on text density
                confidence = self._calculate_line_confidence(cropped)

                text_line = TextLine(
                    bounding_box=(col_start + x_start, y_start, x_end - x_start, y_end - y_start),
                    cropped_image=cropped.copy(),
                    reading_order=reading_order,
                    confidence_score=confidence,
                    column_index=col_idx,
                    line_index_in_column=line_idx,
                )
                all_lines.append(text_line)
                reading_order += 1

        logger.info(
            f"Detected {len(all_lines)} text lines across {len(columns)} column(s)"
        )

        return all_lines

    def maintain_reading_order(self, lines: list[TextLine]) -> list[TextLine]:
        """Sort text lines by reading order (top-to-bottom, left-to-right).

        Args:
            lines: List of detected text lines.

        Returns:
            Lines sorted by reading order.
        """
        # Sort by column first, then by y-coordinate within each column
        return sorted(lines, key=lambda l: (l.column_index, l.y))

    def _detect_columns(self, image: np.ndarray) -> list[tuple[int, int]]:
        """Detect column regions in a document.

        Args:
            image: Binary document image.

        Returns:
            List of (start_x, end_x) tuples for each column.
        """
        # Compute vertical projection profile
        vertical_proj = np.sum(image == 0, axis=0)  # Count black pixels per column

        # Smooth the projection
        kernel_size = max(5, image.shape[1] // 50)
        if kernel_size % 2 == 0:
            kernel_size += 1
        vertical_proj_smooth = np.convolve(
            vertical_proj, np.ones(kernel_size) / kernel_size, mode="same"
        )

        # Find gaps (potential column separators)
        threshold = np.max(vertical_proj_smooth) * 0.1
        gap_threshold = int(image.shape[1] * self.column_gap_threshold)

        # Find regions with content
        in_content = vertical_proj_smooth > threshold
        columns: list[tuple[int, int]] = []
        start = None

        for i, has_content in enumerate(in_content):
            if has_content and start is None:
                start = i
            elif not has_content and start is not None:
                if i - start > gap_threshold:
                    columns.append((start, i))
                start = None

        if start is not None:
            columns.append((start, len(in_content)))

        # If no columns detected, treat entire width as one column
        if not columns:
            columns = [(0, image.shape[1])]

        return columns

    def _detect_line_regions(self, image: np.ndarray) -> list[tuple[int, int]]:
        """Detect line regions using horizontal projection profile.

        Args:
            image: Binary column image.

        Returns:
            List of (start_y, end_y) tuples for each line.
        """
        # Compute horizontal projection profile
        horizontal_proj = np.sum(image == 0, axis=1)  # Count black pixels per row

        # Smooth the projection
        kernel_size = max(3, image.shape[0] // 100)
        if kernel_size % 2 == 0:
            kernel_size += 1
        horizontal_proj_smooth = np.convolve(
            horizontal_proj, np.ones(kernel_size) / kernel_size, mode="same"
        )

        # Find line regions
        threshold = np.max(horizontal_proj_smooth) * 0.05
        in_line = horizontal_proj_smooth > threshold
        lines: list[tuple[int, int]] = []
        start = None

        for i, has_content in enumerate(in_line):
            if has_content and start is None:
                start = i
            elif not has_content and start is not None:
                height = i - start
                if self.min_line_height <= height <= self.max_line_height:
                    lines.append((start, i))
                start = None

        if start is not None:
            height = len(in_line) - start
            if self.min_line_height <= height <= self.max_line_height:
                lines.append((start, len(in_line)))

        return lines

    def _refine_horizontal_bounds(self, line_image: np.ndarray) -> tuple[int, int]:
        """Refine horizontal bounds of a text line to trim whitespace.

        Args:
            line_image: Binary line image.

        Returns:
            Tuple of (start_x, end_x) for the refined bounds.
        """
        vertical_proj = np.sum(line_image == 0, axis=0)
        threshold = np.max(vertical_proj) * 0.05 if np.max(vertical_proj) > 0 else 0

        # Find first and last columns with content
        content_cols = np.where(vertical_proj > threshold)[0]
        if len(content_cols) == 0:
            return 0, line_image.shape[1]

        return int(content_cols[0]), int(content_cols[-1] + 1)

    def _calculate_line_confidence(self, line_image: np.ndarray) -> float:
        """Calculate confidence score for a detected text line.

        Based on text density and consistency.

        Args:
            line_image: Binary line image.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if line_image.size == 0:
            return 0.0

        # Calculate text density (ratio of black pixels)
        text_density = np.sum(line_image == 0) / line_image.size

        # Ideal density for handwritten text is between 0.1 and 0.5
        if 0.1 <= text_density <= 0.5:
            density_score = 1.0
        elif text_density < 0.1:
            density_score = text_density / 0.1
        else:
            density_score = max(0.0, 1.0 - (text_density - 0.5) / 0.5)

        return density_score
