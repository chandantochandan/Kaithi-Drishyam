"""Unit tests for the FastAPI application."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from kaithi_drishyam.api.main import app


def test_health_endpoint() -> None:
    """Health endpoint should report ok."""
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint_lists_api_links() -> None:
    """Root endpoint should make manual browser testing clear."""
    client = TestClient(app)

    response = client.get("/")
    payload = response.json()

    assert response.status_code == 200
    assert payload["service"] == "kaithi-drishyam"
    assert payload["links"]["docs"] == "/docs"


def test_status_endpoint_reports_current_capabilities() -> None:
    """Status endpoint should distinguish implemented and pending capabilities."""
    client = TestClient(app)

    response = client.get("/status")
    payload = response.json()

    assert response.status_code == 200
    assert "image_preprocessing" in payload["implemented"]
    assert "trained_crnn_checkpoint" in payload["pending"]


def test_process_document_endpoint(sample_image_file: Path) -> None:
    """Document endpoint should process an uploaded image and return metadata."""
    client = TestClient(app)

    with sample_image_file.open("rb") as image_file:
        response = client.post(
            "/api/v1/documents/process",
            files={"file": ("sample.png", image_file, "image/png")},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["source_image"] == "sample.png"
    assert "line_count" in payload["segmentation"]
    assert isinstance(payload["segmentation"]["lines"], list)
    assert payload["recognition"]["status"] == "model_unavailable"


def test_process_document_rejects_unsupported_format(temp_dir: Path) -> None:
    """Document endpoint should reject unsupported file extensions."""
    client = TestClient(app)
    upload = temp_dir / "sample.bmp"
    upload.write_bytes(b"not an image")

    with upload.open("rb") as image_file:
        response = client.post(
            "/api/v1/documents/process",
            files={"file": ("sample.bmp", image_file, "image/bmp")},
        )

    assert response.status_code == 415


def test_transliterate_endpoint() -> None:
    """Transliteration endpoint should convert typed Kaithi text."""
    client = TestClient(app)

    response = client.post("/api/v1/transliterate", json={"text": "𑂍𑂰𑂧"})

    assert response.status_code == 200
    assert response.json()["devanagari_text"] == "कार"
