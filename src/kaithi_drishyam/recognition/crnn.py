"""CRNN architecture and inference wrapper for Kaithi OCR."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch import nn

from kaithi_drishyam.config import settings
from kaithi_drishyam.recognition.models import RecognitionResult, RecognitionSummary
from kaithi_drishyam.segmentation import TextLine


KAITHI_CHARSET = [
    "𑂃",
    "𑂄",
    "𑂅",
    "𑂆",
    "𑂇",
    "𑂈",
    "𑂉",
    "𑂊",
    "𑂋",
    "𑂌",
    "𑂍",
    "𑂎",
    "𑂏",
    "𑂐",
    "𑂑",
    "𑂒",
    "𑂓",
    "𑂔",
    "𑂕",
    "𑂖",
    "𑂗",
    "𑂘",
    "𑂙",
    "𑂚",
    "𑂛",
    "𑂜",
    "𑂝",
    "𑂞",
    "𑂟",
    "𑂠",
    "𑂡",
    "𑂢",
    "𑂣",
    "𑂤",
    "𑂥",
    "𑂦",
    "𑂧",
    "𑂨",
    "𑂩",
    "𑂪",
    "𑂫",
    "𑂬",
    "𑂭",
    "𑂮",
    "𑂯",
    "𑂰",
    "𑂱",
    "𑂲",
    "𑂳",
    "𑂴",
    "𑂵",
    "𑂶",
    "𑂷",
    "𑂸",
    "𑂹",
]


class KaithiCRNN(nn.Module):
    """Compact CNN + BiLSTM + CTC-ready recognizer."""

    def __init__(self, num_classes: int, hidden_size: int = 256, num_layers: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.sequence = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=settings.crnn_dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return time-major logits for CTC decoding."""
        features = self.features(images).squeeze(2).permute(0, 2, 1)
        sequence, _ = self.sequence(features)
        logits = self.classifier(sequence)
        return logits.permute(1, 0, 2)


class CRNNRecognizer:
    """Inference wrapper around a trained KaithiCRNN checkpoint."""

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        charset: Sequence[str] = KAITHI_CHARSET,
    ) -> None:
        self.charset = list(charset)
        self.blank_index = 0
        self.model = KaithiCRNN(
            num_classes=len(self.charset) + 1,
            hidden_size=settings.crnn_hidden_size,
            num_layers=settings.crnn_num_layers,
        )
        self.checkpoint_path = checkpoint_path or settings.get_model_path("checkpoints/kaithi_crnn.pt")
        self.is_loaded = False

    def load(self) -> bool:
        """Load trained model weights if present."""
        if not self.checkpoint_path.exists():
            return False
        state = torch.load(self.checkpoint_path, map_location=settings.device)
        self.model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        self.model.to(settings.device)
        self.model.eval()
        self.is_loaded = True
        return True

    def recognize_lines(self, lines: list[TextLine]) -> RecognitionSummary:
        """Recognize Kaithi text from segmented line images."""
        if not self.is_loaded and not self.load():
            return RecognitionSummary(
                status="model_unavailable",
                message=(
                    "CRNN architecture is implemented, but no trained checkpoint was found at "
                    f"{self.checkpoint_path}."
                ),
            )

        results = [self._recognize_line(line) for line in lines]
        return RecognitionSummary(status="recognized", message="Recognition completed.", lines=results)

    def _recognize_line(self, line: TextLine) -> RecognitionResult:
        tensor = self._prepare_line_image(line.cropped_image)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=2)
            confidence, predicted = torch.max(probabilities, dim=2)

        text = self._ctc_greedy_decode(predicted[:, 0].cpu().tolist())
        score = float(confidence[:, 0].mean().cpu().item())
        return RecognitionResult(
            kaithi_text=text,
            confidence_score=score,
            low_confidence=score < settings.confidence_threshold,
        )

    def _prepare_line_image(self, image: np.ndarray) -> torch.Tensor:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(image, (256, 48), interpolation=cv2.INTER_AREA)
        normalized = resized.astype("float32") / 255.0
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(settings.device)

    def _ctc_greedy_decode(self, indices: list[int]) -> str:
        output: list[str] = []
        previous = self.blank_index
        for index in indices:
            if index != self.blank_index and index != previous:
                output.append(self.charset[index - 1])
            previous = index
        return "".join(output)
