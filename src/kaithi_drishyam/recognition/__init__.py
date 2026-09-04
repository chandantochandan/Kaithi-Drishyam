"""
Recognition module for Kaithi script OCR.

This module handles:
- CRNN model architecture (CNN + Bi-LSTM + CTC)
- Synthetic training data generation
- Model training and inference
- Confidence score calculation
"""

from kaithi_drishyam.recognition.crnn import CRNNRecognizer, KaithiCRNN
from kaithi_drishyam.recognition.models import RecognitionResult, RecognitionSummary
from kaithi_drishyam.recognition.synthetic_data import SyntheticDataGenerator, SyntheticSample

__all__ = [
    "CRNNRecognizer",
    "KaithiCRNN",
    "RecognitionResult",
    "RecognitionSummary",
    "SyntheticDataGenerator",
    "SyntheticSample",
]
