"""Document loading utilities for images and PDFs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np


@dataclass
class DocumentPage:
    """One rasterized page from an input document."""

    page_number: int
    image: np.ndarray


class DocumentLoader:
    """Load supported image and PDF inputs into page images."""

    IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
    PDF_FORMATS = {".pdf"}
    SUPPORTED_FORMATS = IMAGE_FORMATS | PDF_FORMATS

    def load(self, path: Path, dpi: int = 200) -> list[DocumentPage]:
        """Load an image/PDF as one or more page images."""
        suffix = path.suffix.lower()
        if suffix in self.IMAGE_FORMATS:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Failed to load image: {path}")
            return [DocumentPage(page_number=1, image=image)]

        if suffix in self.PDF_FORMATS:
            return self._load_pdf(path, dpi=dpi)

        raise ValueError(f"Unsupported document format: {suffix}")

    def _load_pdf(self, path: Path, dpi: int) -> list[DocumentPage]:
        if not path.exists():
            raise ValueError(f"Document file not found: {path}")

        pages: list[DocumentPage] = []
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        with fitz.open(path) as document:
            for index, page in enumerate(document):
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                    pixmap.height, pixmap.width, pixmap.n
                )
                if pixmap.n == 3:
                    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                pages.append(DocumentPage(page_number=index + 1, image=array))

        if not pages:
            raise ValueError(f"PDF has no pages: {path}")
        return pages
