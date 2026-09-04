"""FastAPI application for Kaithi-Drishyam."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from kaithi_drishyam.config import settings
from kaithi_drishyam.document_loader import DocumentLoader
from kaithi_drishyam.pipeline import DocumentPipeline
from kaithi_drishyam.preprocessing import DocumentProcessor
from kaithi_drishyam.recognition import CRNNRecognizer
from kaithi_drishyam.transliteration import TransliterationService


app = FastAPI(
    title="Kaithi-Drishyam API",
    description="OCR pipeline API for historical Kaithi-script land records.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health."""
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, Any]:
    """Return API entrypoint information."""
    return {
        "service": "kaithi-drishyam",
        "status": "ok",
        "links": {
            "health": "/health",
            "status": "/status",
            "docs": "/docs",
            "process_document": "/api/v1/documents/process",
        },
    }


@app.get("/status")
def status() -> dict[str, Any]:
    """Return implemented and pending capabilities."""
    return {
        "service": "kaithi-drishyam",
        "implemented": [
            "image_preprocessing",
            "text_line_segmentation",
            "crnn_model_architecture",
            "kaithi_to_devanagari_transliteration",
            "synthetic_data_generator",
            "pdf_document_ingestion",
            "multilingual_request_metadata",
            "pipeline_json_metadata",
            "cer_wer_metrics",
        ],
        "pending": [
            "trained_crnn_checkpoint",
            "bhashini_nmt_modernization",
        ],
        "recognition": {
            "architecture": "implemented",
            "checkpoint_path": str(CRNNRecognizer().checkpoint_path),
            "checkpoint_available": CRNNRecognizer().checkpoint_path.exists(),
        },
        "bhashini": {
            "udyat_denoiser_client": "implemented",
            "api_key_configured": bool(settings.bhashini_api_key),
            "live_contract_verified": False,
        },
    }


@app.post("/api/v1/documents/process")
async def process_document(file: UploadFile = File(...)) -> dict[str, Any]:
    """Preprocess and segment an uploaded document image."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in DocumentLoader.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail={
                "message": "Unsupported document format",
                "supported_formats": sorted(DocumentLoader.SUPPORTED_FORMATS),
            },
        )

    max_bytes = settings.max_image_size_mb * 1024 * 1024
    with tempfile.TemporaryDirectory(prefix="kaithi_drishyam_") as tmpdir:
        tmp_path = Path(tmpdir) / f"upload{suffix}"
        total_bytes = 0

        async with aiofiles.open(tmp_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Image file too large. Maximum allowed: {settings.max_image_size_mb}MB",
                    )
                await out_file.write(chunk)

        try:
            pipeline = DocumentPipeline(
                processor=DocumentProcessor(use_bhashini_denoiser=bool(settings.bhashini_api_key))
            )
            results = pipeline.process_pages(tmp_path, write_images=False)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Document processing failed") from exc

    pages = [result.to_dict() for result in results]
    for page in pages:
        page["source_image"] = file.filename
    payload = {
        "source_image": file.filename,
        "file_size_bytes": total_bytes,
        "page_count": len(pages),
        "pages": pages,
    }
    if len(pages) == 1:
        payload.update(pages[0])
        payload["page_count"] = 1
        payload["pages"] = pages
    return payload


@app.post("/api/v1/transliterate")
async def transliterate(payload: dict[str, str]) -> dict[str, Any]:
    """Transliterate typed Kaithi text to Devanagari."""
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' is required")

    result = TransliterationService().transliterate(text)
    return {
        "source_text": text,
        "devanagari_text": result.devanagari_text,
        "modern_hindi_text": result.modern_hindi_text,
        "legal_term_mappings": result.legal_term_mappings,
        "unmapped_characters": result.unmapped_characters,
    }
