"""Document preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
from loguru import logger

from kaithi_drishyam.config import settings
from kaithi_drishyam.preprocessing.models import ProcessedImage
from kaithi_drishyam.preprocessing.bhashini_client import BhashiniUdyatClient, BhashiniAPIError


class DocumentProcessor:
    """Handles document image preprocessing for OCR.

    This processor implements the Phase 1 preprocessing pipeline:
    1. Image loading and format validation
    2. Deskewing (up to 15 degrees)
    3. Binarization (grayscale to black-and-white)
    4. Noise reduction (Bhashini Udyat API or Albumentations fallback)
    5. Border artifact removal
    """

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
    MAX_SKEW_ANGLE = settings.max_skew_angle

    def __init__(self, use_bhashini_denoiser: bool = True):
        """Initialize the document processor.

        Args:
            use_bhashini_denoiser: Whether to use Bhashini Udyat API for denoising.
        """
        self.use_bhashini_denoiser = use_bhashini_denoiser
        self._bhashini_client = None

    def preprocess_image(
        self,
        image: Union[np.ndarray, str, Path],
        apply_deskew: bool = True,
        apply_binarization: bool = True,
        apply_denoising: bool = True,
        remove_borders: bool = True,
    ) -> ProcessedImage:
        """Apply full preprocessing pipeline to a document image.

        Args:
            image: Input image as numpy array or path to image file.
            apply_deskew: Whether to correct document skew.
            apply_binarization: Whether to binarize the image.
            apply_denoising: Whether to apply noise reduction.
            remove_borders: Whether to remove dark border artifacts.

        Returns:
            ProcessedImage with all preprocessing applied.

        Raises:
            ValueError: If image format is not supported or image is invalid.
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = self._load_image(Path(image))

        if image is None or image.size == 0:
            raise ValueError("Invalid or empty image provided")

        original_shape = image.shape[:2]
        metadata: dict = {"steps_applied": []}

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            metadata["steps_applied"].append("grayscale_conversion")

        # Step 1: Remove borders
        if remove_borders:
            image = self._remove_borders(image)
            metadata["steps_applied"].append("border_removal")

        # Step 2: Deskew
        deskew_angle = 0.0
        if apply_deskew:
            image, deskew_angle = self._deskew(image)
            metadata["steps_applied"].append("deskewing")
            metadata["deskew_angle"] = deskew_angle

        # Step 3: Denoise
        noise_level = self._estimate_noise_level(image)
        if apply_denoising:
            image = self._denoise(image)
            metadata["steps_applied"].append("denoising")

        # Step 4: Binarize
        binarized = False
        if apply_binarization:
            image = self._binarize(image)
            binarized = True
            metadata["steps_applied"].append("binarization")

        logger.info(
            f"Preprocessing complete: {len(metadata['steps_applied'])} steps applied, "
            f"deskew={deskew_angle:.2f}deg, noise_level={noise_level:.3f}"
        )

        return ProcessedImage(
            image=image,
            original_shape=original_shape,
            deskew_angle=deskew_angle,
            noise_level=noise_level,
            binarized=binarized,
            preprocessing_metadata=metadata,
        )

    def _load_image(self, path: Path) -> np.ndarray:
        """Load image from file path.

        Args:
            path: Path to the image file.

        Returns:
            Image as numpy array.

        Raises:
            ValueError: If file format is not supported or file doesn't exist.
        """
        if not path.exists():
            raise ValueError(f"Image file not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format: {suffix}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )

        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > settings.max_image_size_mb:
            raise ValueError(
                f"Image file too large: {file_size_mb:.1f}MB. "
                f"Maximum allowed: {settings.max_image_size_mb}MB"
            )

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        return image

    def _deskew(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Detect and correct document skew.

        Args:
            image: Grayscale input image.

        Returns:
            Tuple of (deskewed image, detected angle in degrees).
        """
        # Use Hough Line Transform to detect skew angle
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
        )

        if lines is None:
            return image, 0.0

        # Calculate angles of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Normalize to [-45, 45] range
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            angles.append(angle)

        if not angles:
            return image, 0.0

        # Use median angle for robustness
        median_angle = np.median(angles)

        # Only correct if within maximum allowed skew
        if abs(median_angle) > self.MAX_SKEW_ANGLE:
            logger.warning(
                f"Detected skew angle {median_angle:.2f}deg exceeds maximum "
                f"{self.MAX_SKEW_ANGLE}deg, skipping correction"
            )
            return image, 0.0

        # Rotate image to correct skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rotated, median_angle

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """Convert grayscale image to binary using adaptive thresholding.

        Args:
            image: Grayscale input image.

        Returns:
            Binary image (black text on white background).
        """
        # Use adaptive Gaussian thresholding for varying illumination
        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )
        return binary

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction to the image.

        Attempts to use Bhashini Udyat API first, falls back to
        local denoising if unavailable.

        Args:
            image: Input image.

        Returns:
            Denoised image.
        """
        if self.use_bhashini_denoiser:
            try:
                return self._denoise_with_bhashini(image)
            except Exception as e:
                logger.warning(f"Bhashini denoiser failed, using fallback: {e}")

        return self._denoise_local(image)

    def _denoise_with_bhashini(self, image: np.ndarray) -> np.ndarray:
        """Denoise using Bhashini Udyat API.

        Args:
            image: Input image.

        Returns:
            Denoised image from Bhashini API.

        Raises:
            BhashiniAPIError: If API call fails.
        """
        if self._bhashini_client is None:
            self._bhashini_client = BhashiniUdyatClient()

        return self._bhashini_client.denoise_image(image)

    def _denoise_local(self, image: np.ndarray) -> np.ndarray:
        """Apply local denoising using OpenCV.

        Uses Non-local Means Denoising for effective noise removal
        while preserving text edges.

        Args:
            image: Grayscale input image.

        Returns:
            Denoised image.
        """
        # Apply Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)

        # Apply morphological operations to remove salt-and-pepper noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)

        return denoised

    def _remove_borders(self, image: np.ndarray) -> np.ndarray:
        """Remove dark border artifacts from scanned documents.

        Args:
            image: Input image.

        Returns:
            Image with borders cropped.
        """
        # Create binary mask
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        # Find the largest contour (document area)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Add small margin
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        return image[y : y + h, x : x + w]

    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level in the image.

        Uses Laplacian variance method for noise estimation.

        Args:
            image: Grayscale input image.

        Returns:
            Estimated noise level (0.0 to 1.0, higher means noisier).
        """
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        # Normalize to 0-1 range (empirically determined thresholds)
        noise_level = min(1.0, laplacian_var / 1000.0)
        return noise_level
