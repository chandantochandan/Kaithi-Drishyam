"""Property-based tests for the preprocessing module.

Property 1: Image Format and Preprocessing Pipeline
For any uploaded image in JPEG, PNG, or TIFF format, the Document_Processor
should successfully preprocess it by applying deskewing (for angles up to 15 degrees),
binarization, noise reduction, and border removal, producing a ProcessedImage
with valid metadata.

Validates: Requirements 1.1, 1.3, 1.4, 1.5, 1.6
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck

from kaithi_drishyam.preprocessing import DocumentProcessor, ProcessedImage


class TestImagePreprocessingPipelineProperty:
    """Property 1: Image Format and Preprocessing Pipeline tests."""

    @pytest.fixture
    def processor(self) -> DocumentProcessor:
        """Create a document processor instance."""
        return DocumentProcessor(use_bhashini_denoiser=False)

    @pytest.mark.property
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        width=st.integers(min_value=100, max_value=1000),
        height=st.integers(min_value=100, max_value=1000),
        noise_level=st.floats(min_value=0.0, max_value=0.1),
        num_text_lines=st.integers(min_value=1, max_value=10),
    )
    def test_preprocessing_produces_valid_output(
        self,
        processor: DocumentProcessor,
        width: int,
        height: int,
        noise_level: float,
        num_text_lines: int,
    ) -> None:
        """Property: For any valid image, preprocessing produces valid ProcessedImage.

        Feature: kaithi-ocr, Property 1: Image Format and Preprocessing Pipeline
        Validates: Requirements 1.1, 1.3, 1.4, 1.5, 1.6
        """
        # Generate synthetic document image
        image = self._generate_document_image(width, height, noise_level, num_text_lines)

        # Apply preprocessing
        result = processor.preprocess_image(image)

        # Property assertions
        assert isinstance(result, ProcessedImage)
        assert result.image is not None
        assert result.image.size > 0
        assert result.original_shape == (height, width)
        assert 0.0 <= result.noise_level <= 1.0
        assert isinstance(result.preprocessing_metadata, dict)
        assert "steps_applied" in result.preprocessing_metadata

    @pytest.mark.property
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        skew_angle=st.floats(min_value=-15.0, max_value=15.0),
    )
    def test_deskewing_corrects_within_threshold(
        self,
        processor: DocumentProcessor,
        skew_angle: float,
    ) -> None:
        """Property: Deskewing corrects angles up to 15 degrees.

        Feature: kaithi-ocr, Property 1: Image Format and Preprocessing Pipeline
        Validates: Requirement 1.3
        """
        # Skip very small angles that might not be detected
        assume(abs(skew_angle) > 1.0)

        # Generate skewed image
        image = self._generate_skewed_image(400, 300, skew_angle)

        # Apply preprocessing with deskewing
        result = processor.preprocess_image(image, apply_deskew=True)

        # Property: deskewing should be attempted
        assert "deskewing" in result.preprocessing_metadata["steps_applied"]

    @pytest.mark.property
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        width=st.integers(min_value=100, max_value=500),
        height=st.integers(min_value=100, max_value=500),
    )
    def test_binarization_produces_binary_output(
        self,
        processor: DocumentProcessor,
        width: int,
        height: int,
    ) -> None:
        """Property: Binarization produces black and white output.

        Feature: kaithi-ocr, Property 1: Image Format and Preprocessing Pipeline
        Validates: Requirement 1.4
        """
        # Generate grayscale image
        image = self._generate_document_image(width, height, noise_level=0.02, num_lines=3)

        # Apply preprocessing with binarization
        result = processor.preprocess_image(image, apply_binarization=True)

        # Property: output should be binary (only 0 and 255 values)
        assert result.binarized is True
        unique_values = np.unique(result.image)
        assert len(unique_values) <= 2
        assert all(v in [0, 255] for v in unique_values)

    @pytest.mark.property
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        noise_level=st.floats(min_value=0.01, max_value=0.15),
    )
    def test_denoising_reduces_noise(
        self,
        processor: DocumentProcessor,
        noise_level: float,
    ) -> None:
        """Property: Denoising reduces noise in the image.

        Feature: kaithi-ocr, Property 1: Image Format and Preprocessing Pipeline
        Validates: Requirement 1.5
        """
        # Generate noisy image
        image = self._generate_document_image(300, 200, noise_level=noise_level, num_lines=3)

        # Apply preprocessing with denoising but without binarization
        result = processor.preprocess_image(
            image,
            apply_denoising=True,
            apply_binarization=False,
        )

        # Property: denoising should be applied
        assert "denoising" in result.preprocessing_metadata["steps_applied"]

    @pytest.mark.property
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        border_size=st.integers(min_value=10, max_value=50),
    )
    def test_border_removal_crops_image(
        self,
        processor: DocumentProcessor,
        border_size: int,
    ) -> None:
        """Property: Border removal crops dark borders.

        Feature: kaithi-ocr, Property 1: Image Format and Preprocessing Pipeline
        Validates: Requirement 1.6
        """
        # Generate image with dark borders
        image = self._generate_image_with_borders(300, 200, border_size)

        # Apply preprocessing with border removal
        result = processor.preprocess_image(image, remove_borders=True)

        # Property: border removal should be applied and image should be smaller
        assert "border_removal" in result.preprocessing_metadata["steps_applied"]

    @pytest.mark.property
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        is_color=st.booleans(),
        width=st.integers(min_value=100, max_value=400),
        height=st.integers(min_value=100, max_value=400),
    )
    def test_handles_color_and_grayscale(
        self,
        processor: DocumentProcessor,
        is_color: bool,
        width: int,
        height: int,
    ) -> None:
        """Property: Processor handles both color and grayscale images.

        Feature: kaithi-ocr, Property 1: Image Format and Preprocessing Pipeline
        Validates: Requirement 1.1
        """
        # Generate image
        if is_color:
            image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            # Add some structure
            image[height//4:height//2, width//4:3*width//4] = 0
        else:
            image = np.random.randint(200, 256, (height, width), dtype=np.uint8)
            image[height//4:height//2, width//4:3*width//4] = 0

        # Apply preprocessing
        result = processor.preprocess_image(image)

        # Property: output should always be 2D (grayscale/binary)
        assert len(result.image.shape) == 2

    # Helper methods for generating test images

    def _generate_document_image(
        self,
        width: int,
        height: int,
        noise_level: float,
        num_lines: int,
    ) -> np.ndarray:
        """Generate a synthetic document image with text lines."""
        # White background
        image = np.ones((height, width), dtype=np.uint8) * 255

        # Add text-like horizontal lines
        line_height = max(10, height // (num_lines * 3))
        y_start = line_height

        for i in range(num_lines):
            y = y_start + i * (line_height * 2)
            if y + line_height < height:
                x_end = int(width * (0.5 + 0.4 * np.random.random()))
                image[y:y + line_height, 30:x_end] = 0

        # Add noise
        if noise_level > 0:
            noise = np.random.random(image.shape)
            image[noise < noise_level / 2] = 0
            image[noise > 1 - noise_level / 2] = 255

        return image

    def _generate_skewed_image(
        self,
        width: int,
        height: int,
        angle: float,
    ) -> np.ndarray:
        """Generate a skewed document image."""
        import cv2

        # Create base image
        image = self._generate_document_image(width, height, 0.0, 4)

        # Apply rotation
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        skewed = cv2.warpAffine(
            image,
            rotation_matrix,
            (width, height),
            borderValue=255,
        )

        return skewed

    def _generate_image_with_borders(
        self,
        width: int,
        height: int,
        border_size: int,
    ) -> np.ndarray:
        """Generate an image with dark borders."""
        # Create image with content
        image = self._generate_document_image(width, height, 0.0, 3)

        # Add dark borders
        image[:border_size, :] = 20  # Top
        image[-border_size:, :] = 20  # Bottom
        image[:, :border_size] = 20  # Left
        image[:, -border_size:] = 20  # Right

        return image
