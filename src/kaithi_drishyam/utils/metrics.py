"""Metrics calculation utilities for OCR evaluation."""

from __future__ import annotations

from typing import List, Union


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER).

    CER = (S + D + I) / N

    Where:
    - S = number of substitutions
    - D = number of deletions
    - I = number of insertions
    - N = number of characters in reference

    Args:
        reference: Ground truth text.
        hypothesis: Predicted text.

    Returns:
        Character Error Rate (0.0 = perfect, higher = worse).
    """
    if len(reference) == 0:
        return 1.0 if len(hypothesis) > 0 else 0.0

    # Use dynamic programming (Levenshtein distance)
    distance = _levenshtein_distance(reference, hypothesis)
    return distance / len(reference)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER).

    WER = (S + D + I) / N

    Where:
    - S = number of word substitutions
    - D = number of word deletions
    - I = number of word insertions
    - N = number of words in reference

    Args:
        reference: Ground truth text.
        hypothesis: Predicted text.

    Returns:
        Word Error Rate (0.0 = perfect, higher = worse).
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0

    distance = _levenshtein_distance(ref_words, hyp_words)
    return distance / len(ref_words)


def _levenshtein_distance(s1: Union[str, List], s2: Union[str, List]) -> int:
    """Calculate Levenshtein distance between two sequences.

    Args:
        s1: First sequence (string or list of words).
        s2: Second sequence (string or list of words).

    Returns:
        Edit distance between sequences.
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
