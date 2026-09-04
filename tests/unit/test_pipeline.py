"""Unit tests for the document pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import fitz

from kaithi_drishyam.pipeline import DocumentPipeline
from kaithi_drishyam.preprocessing import DocumentProcessor


def test_pipeline_processes_image_and_writes_outputs(
    sample_image_file: Path,
    temp_dir: Path,
) -> None:
    """Pipeline should write processed image, line crops, and serializable metadata."""
    pipeline = DocumentPipeline(processor=DocumentProcessor(use_bhashini_denoiser=False))

    result = pipeline.process(sample_image_file, output_dir=temp_dir / "out")
    payload = result.to_dict()

    assert payload["source_image"] == str(sample_image_file)
    assert payload["segmentation"]["line_count"] == len(result.text_lines)
    assert payload["recognition"]["status"] == "model_unavailable"
    assert payload["transliteration"]["status"] == "waiting_for_recognition"
    assert result.processed_image_path is not None
    assert cv2.imread(result.processed_image_path, cv2.IMREAD_GRAYSCALE) is not None

    for line in payload["segmentation"]["lines"]:
        assert line["image_path"] is not None
        assert Path(line["image_path"]).exists()


def test_pipeline_json_is_serializable(sample_image_file: Path) -> None:
    """Pipeline JSON should be valid for CLI/API consumers."""
    pipeline = DocumentPipeline(processor=DocumentProcessor(use_bhashini_denoiser=False))

    result = pipeline.process(sample_image_file, write_images=False)
    payload = json.loads(result.to_json())

    assert payload["segmentation"]["line_count"] == len(result.text_lines)
    assert payload["processed_image_path"] is None


def test_pipeline_processes_pdf_pages(temp_dir: Path) -> None:
    """Pipeline should process rasterized PDF pages."""
    pdf_path = temp_dir / "sample.pdf"
    document = fitz.open()
    page = document.new_page(width=360, height=180)
    page.insert_text((40, 60), "Kaithi Drishyam sample", fontsize=18)
    document.save(pdf_path)
    document.close()

    pipeline = DocumentPipeline(processor=DocumentProcessor(use_bhashini_denoiser=False))
    results = pipeline.process_pages(pdf_path, write_images=False)

    assert len(results) == 1
    assert results[0].page_number == 1
