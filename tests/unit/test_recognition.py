"""Unit tests for CRNN recognition components."""

from __future__ import annotations

import torch

from kaithi_drishyam.recognition import CRNNRecognizer, KaithiCRNN


def test_crnn_forward_returns_time_major_logits() -> None:
    """CRNN model should produce CTC-compatible logits."""
    model = KaithiCRNN(num_classes=12, hidden_size=16, num_layers=1)
    images = torch.rand(2, 1, 48, 256)

    logits = model(images)

    assert logits.ndim == 3
    assert logits.shape[1] == 2
    assert logits.shape[2] == 12


def test_recognizer_reports_missing_checkpoint(sample_grayscale_image) -> None:
    """Recognizer should be explicit when no trained model is available."""
    recognizer = CRNNRecognizer()

    summary = recognizer.recognize_lines([])

    assert summary.status == "model_unavailable"
    assert "no trained checkpoint" in summary.message
