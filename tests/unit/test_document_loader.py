"""Unit tests for document loading."""

from __future__ import annotations

from pathlib import Path

import fitz

from kaithi_drishyam.document_loader import DocumentLoader


def _write_sample_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=360, height=180)
    page.insert_text((40, 60), "Kaithi Drishyam sample", fontsize=18)
    page.insert_text((40, 100), "Line two", fontsize=18)
    document.save(path)
    document.close()


def test_load_image_as_single_page(sample_image_file: Path) -> None:
    """Image inputs should load as one page."""
    pages = DocumentLoader().load(sample_image_file)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].image.size > 0


def test_load_pdf_as_pages(temp_dir: Path) -> None:
    """PDF inputs should rasterize into page images."""
    pdf_path = temp_dir / "sample.pdf"
    _write_sample_pdf(pdf_path)

    pages = DocumentLoader().load(pdf_path)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].image.size > 0
