"""Recognition model data structures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RecognitionResult:
    """OCR output for one segmented line."""

    kaithi_text: str
    confidence_score: float
    status: str = "recognized"
    low_confidence: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class RecognitionSummary:
    """OCR output for a full document."""

    status: str
    message: str
    lines: list[RecognitionResult] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Join line-level recognized text."""
        return "\n".join(line.kaithi_text for line in self.lines)

    @property
    def average_confidence(self) -> float:
        """Return average confidence across recognized lines."""
        if not self.lines:
            return 0.0
        return sum(line.confidence_score for line in self.lines) / len(self.lines)
