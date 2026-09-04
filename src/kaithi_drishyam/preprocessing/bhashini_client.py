"""Bhashini Udyat API client for image denoising."""

from __future__ import annotations

import base64
from typing import Optional

import cv2
import httpx
import numpy as np
from loguru import logger

from kaithi_drishyam.config import settings


class BhashiniUdyatClient:
    """Client for Bhashini Udyat Denoiser API.

    This client handles:
    - API authentication
    - Image encoding/decoding
    - Request/response handling
    - Error recovery with exponential backoff
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """Initialize the Bhashini Udyat client.

        Args:
            api_key: Bhashini API key. Defaults to settings.bhashini_api_key.
            api_url: Bhashini Udyat API URL. Defaults to settings.bhashini_udyat_url.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.
        """
        self.api_key = api_key or settings.bhashini_api_key
        self.api_url = api_url or settings.bhashini_udyat_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                headers=self._get_headers(),
            )
        return self._client

    def _get_headers(self) -> dict:
        """Get request headers with authentication."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Denoise an image using Bhashini Udyat API.

        Args:
            image: Input image as numpy array (grayscale or BGR).

        Returns:
            Denoised image as numpy array.

        Raises:
            BhashiniAPIError: If API call fails after all retries.
            ValueError: If API key is not configured.
        """
        if not self.api_key:
            raise ValueError(
                "Bhashini API key not configured. "
                "Set KAITHI_BHASHINI_API_KEY environment variable."
            )

        # Encode image to base64
        image_b64 = self._encode_image(image)

        # Prepare request payload
        payload = {
            "image": image_b64,
            "task": "denoise",
            "config": {
                "strength": "medium",
                "preserve_text": True,
            },
        }

        # Make request with retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.post(
                    f"{self.api_url}/enhance",
                    json=payload,
                )
                response.raise_for_status()

                # Decode response
                result = response.json()
                if "image" in result:
                    return self._decode_image(result["image"])
                elif "enhanced_image" in result:
                    return self._decode_image(result["enhanced_image"])
                else:
                    raise BhashiniAPIError("Invalid response format from Bhashini API")

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"Bhashini API timeout (attempt {attempt + 1}/{self.max_retries})")
                self._exponential_backoff(attempt)

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429:  # Rate limited
                    logger.warning(f"Rate limited (attempt {attempt + 1}/{self.max_retries})")
                    self._exponential_backoff(attempt)
                elif e.response.status_code == 401:
                    raise BhashiniAPIError("Authentication failed. Check API key.")
                else:
                    raise BhashiniAPIError(f"HTTP error: {e.response.status_code}")

            except httpx.RequestError as e:
                last_error = e
                logger.warning(f"Request error (attempt {attempt + 1}/{self.max_retries}): {e}")
                self._exponential_backoff(attempt)

        raise BhashiniAPIError(f"API call failed after {self.max_retries} attempts: {last_error}")

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64 string.

        Args:
            image: Image as numpy array.

        Returns:
            Base64 encoded image string.
        """
        # Encode as PNG
        success, buffer = cv2.imencode(".png", image)
        if not success:
            raise ValueError("Failed to encode image")

        return base64.b64encode(buffer).decode("utf-8")

    def _decode_image(self, image_b64: str) -> np.ndarray:
        """Decode base64 string to image.

        Args:
            image_b64: Base64 encoded image string.

        Returns:
            Image as numpy array.
        """
        image_bytes = base64.b64decode(image_b64)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError("Failed to decode image from response")

        return image

    def _exponential_backoff(self, attempt: int) -> None:
        """Sleep with exponential backoff.

        Args:
            attempt: Current attempt number (0-indexed).
        """
        import time

        delay = min(2 ** attempt, 30)  # Max 30 seconds
        time.sleep(delay)

    def check_health(self) -> bool:
        """Check if Bhashini API is available.

        Returns:
            True if API is healthy, False otherwise.
        """
        try:
            response = self.client.get(
                f"{self.api_url}/health",
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "BhashiniUdyatClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


class BhashiniAPIError(Exception):
    """Exception raised for Bhashini API errors."""

    pass
