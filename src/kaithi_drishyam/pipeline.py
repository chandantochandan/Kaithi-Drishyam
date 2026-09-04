"""Document processing pipeline orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2

from kaithi_drishyam.document_loader import DocumentLoader
from kaithi_drishyam.preprocessing import DocumentProcessor, ProcessedImage
from kaithi_drishyam.recognition import CRNNRecognizer, RecognitionSummary
from kaithi_drishyam.segmentation import SegmentationEngine, TextLine
from kaithi_drishyam.transliteration import TransliterationResult, TransliterationService


@dataclass
class DocumentPipelineResult:
    """Result produced by preprocessing and text-line segmentation."""

    source_image: str
    processed_image_path: Optional[str]
    line_image_dir: Optional[str]
    processed_image: ProcessedImage
    text_lines: list[TextLine]
    recognition: RecognitionSummary
    transliteration: Optional[TransliterationResult] = None
    page_number: int = 1
    source_language: str = "kaithi"
    target_language: str = "devanagari"

    def to_dict(self) -> dict[str, Any]:
        """Convert the result into a JSON-serializable dictionary."""
        return {
            "source_image": self.source_image,
            "page_number": self.page_number,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "processed_image_path": self.processed_image_path,
            "line_image_dir": self.line_image_dir,
            "preprocessing": {
                "original_shape": list(self.processed_image.original_shape),
                "processed_shape": list(self.processed_image.shape),
                "deskew_angle": self.processed_image.deskew_angle,
                "noise_level": self.processed_image.noise_level,
                "binarized": self.processed_image.binarized,
                "metadata": self.processed_image.preprocessing_metadata,
            },
            "segmentation": {
                "line_count": len(self.text_lines),
                "lines": [
                    {
                        "reading_order": line.reading_order,
                        "bounding_box": {
                            "x": line.x,
                            "y": line.y,
                            "width": line.width,
                            "height": line.height,
                        },
                        "confidence_score": line.confidence_score,
                        "column_index": line.column_index,
                        "line_index_in_column": line.line_index_in_column,
                        "image_path": self._line_image_path(line),
                    }
                    for line in self.text_lines
                ],
            },
            "recognition": self._recognition_dict(),
            "transliteration": self._transliteration_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the result as JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def _line_image_path(self, line: TextLine) -> Optional[str]:
        if self.line_image_dir is None:
            return None
        filename = f"line_{line.reading_order:04d}.png"
        return str(Path(self.line_image_dir) / filename)

    def _recognition_dict(self) -> dict[str, Any]:
        return {
            "status": self.recognition.status,
            "message": self.recognition.message,
            "text": self.recognition.text,
            "average_confidence": self.recognition.average_confidence,
            "lines": [
                {
                    "kaithi_text": line.kaithi_text,
                    "confidence_score": line.confidence_score,
                    "status": line.status,
                    "low_confidence": line.low_confidence,
                    "metadata": line.metadata,
                }
                for line in self.recognition.lines
            ],
        }

    def _transliteration_dict(self) -> dict[str, Any]:
        if self.transliteration is None:
            return {
                "status": "waiting_for_recognition",
                "message": "Transliteration is ready, but no recognized Kaithi text is available.",
            }
        return {
            "status": "transliterated",
            "devanagari_text": self.transliteration.devanagari_text,
            "modern_hindi_text": self.transliteration.modern_hindi_text,
            "legal_term_mappings": self.transliteration.legal_term_mappings,
            "unmapped_characters": self.transliteration.unmapped_characters,
        }


class DocumentPipeline:
    """Runs currently implemented document processing stages."""

    def __init__(
        self,
        processor: Optional[DocumentProcessor] = None,
        segmenter: Optional[SegmentationEngine] = None,
        recognizer: Optional[CRNNRecognizer] = None,
        transliterator: Optional[TransliterationService] = None,
    ) -> None:
        self.processor = processor or DocumentProcessor()
        self.segmenter = segmenter or SegmentationEngine()
        self.recognizer = recognizer or CRNNRecognizer()
        self.transliterator = transliterator or TransliterationService()
        self.loader = DocumentLoader()

    def process(
        self,
        image_path: Path,
        output_dir: Optional[Path] = None,
        write_images: bool = True,
        source_language: str = "kaithi",
        target_language: str = "devanagari",
    ) -> DocumentPipelineResult:
        """Preprocess the first page of a document and segment it into text-line images."""
        return self.process_pages(
            image_path,
            output_dir=output_dir,
            write_images=write_images,
            source_language=source_language,
            target_language=target_language,
        )[0]

    def process_pages(
        self,
        document_path: Path,
        output_dir: Optional[Path] = None,
        write_images: bool = True,
        source_language: str = "kaithi",
        target_language: str = "devanagari",
    ) -> list[DocumentPipelineResult]:
        """Preprocess and segment every page in an image/PDF document."""
        pages = self.loader.load(document_path)
        return [
            self._process_page(
                document_path=document_path,
                page_number=page.page_number,
                image=page.image,
                output_dir=output_dir / f"page_{page.page_number:04d}" if output_dir else None,
                write_images=write_images,
                source_language=source_language,
                target_language=target_language,
            )
            for page in pages
        ]

    def _process_page(
        self,
        document_path: Path,
        page_number: int,
        image,
        output_dir: Optional[Path],
        write_images: bool,
        source_language: str,
        target_language: str,
    ) -> DocumentPipelineResult:
        processed = self.processor.preprocess_image(image)
        lines = self.segmenter.detect_text_lines(processed)
        recognition = self.recognizer.recognize_lines(lines)
        transliteration = None
        if recognition.text:
            transliteration = self.transliterator.transliterate(recognition.text)

        processed_image_path: Optional[str] = None
        line_image_dir: Optional[str] = None

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

            if write_images:
                processed_path = output_dir / "processed.png"
                cv2.imwrite(str(processed_path), processed.image)
                processed_image_path = str(processed_path)

                lines_dir = output_dir / "lines"
                lines_dir.mkdir(parents=True, exist_ok=True)
                line_image_dir = str(lines_dir)

                for line in lines:
                    line_path = lines_dir / f"line_{line.reading_order:04d}.png"
                    cv2.imwrite(str(line_path), line.cropped_image)

        return DocumentPipelineResult(
            source_image=str(document_path),
            processed_image_path=processed_image_path,
            line_image_dir=line_image_dir,
            processed_image=processed,
            text_lines=lines,
            recognition=recognition,
            transliteration=transliteration,
            page_number=page_number,
            source_language=source_language,
            target_language=target_language,
        )
