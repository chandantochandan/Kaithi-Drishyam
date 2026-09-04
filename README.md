# Kaithi-Drishyam

Kaithi-Drishyam is an OCR system for digitizing historical Kaithi-script land records and converting scanned document images into modern Hindi text.

## Current Status

This repository currently contains the implemented foundation for:

- document image preprocessing
- Bhashini Udyat denoiser API client with fallback behavior
- text-line segmentation
- OCR accuracy metric utilities
- unit and property tests for preprocessing, segmentation, and metrics

The CRNN recognition engine, transliteration and modernization services, REST API, web interface, and full end-to-end document pipeline are planned but not yet implemented.

## Tests

Run the test suite with:

```bash
pytest -q
```

Latest local verification:

```text
52 passed
```

## Project Layout

```text
src/kaithi_drishyam/     Python package source
tests/                   Unit and property tests
data/                    Dataset placeholders
models/                  Model checkpoint/pretrained placeholders
requirements.md          Product requirements
design.md                System design
tasks.md                 Implementation task plan
```

## Configuration

Copy `.env.example` and provide the required credentials locally. Do not commit real credentials.
