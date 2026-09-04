"""FastAPI application for Kaithi-Drishyam."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from kaithi_drishyam.config import settings
from kaithi_drishyam.pipeline import DocumentPipeline
from kaithi_drishyam.preprocessing import DocumentProcessor


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


@app.get("/status")
def status() -> dict[str, Any]:
    """Return implemented and pending capabilities."""
    return {
        "service": "kaithi-drishyam",
        "implemented": [
            "image_preprocessing",
            "text_line_segmentation",
            "pipeline_json_metadata",
            "cer_wer_metrics",
        ],
        "pending": [
            "crnn_recognition",
            "kaithi_to_devanagari_transliteration",
            "bhashini_nmt_modernization",
            "web_backend_integration",
        ],
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
    if suffix not in DocumentProcessor.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail={
                "message": "Unsupported image format",
                "supported_formats": sorted(DocumentProcessor.SUPPORTED_FORMATS),
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
            result = pipeline.process(tmp_path, write_images=False)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Document processing failed") from exc

    payload = result.to_dict()
    payload["source_image"] = file.filename
    payload["file_size_bytes"] = total_bytes
    return payload
