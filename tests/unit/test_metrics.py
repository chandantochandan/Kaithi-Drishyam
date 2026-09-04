"""Unit tests for the metrics module."""

from __future__ import annotations

import pytest

from kaithi_drishyam.utils.metrics import calculate_cer, calculate_wer


class TestCER:
    """Tests for Character Error Rate calculation."""

    @pytest.mark.unit
    def test_identical_strings(self) -> None:
        """Test CER is 0 for identical strings."""
        assert calculate_cer("hello", "hello") == 0.0

    @pytest.mark.unit
    def test_completely_different(self) -> None:
        """Test CER for completely different strings."""
        # "abc" -> "xyz" requires 3 substitutions, CER = 3/3 = 1.0
        assert calculate_cer("abc", "xyz") == 1.0

    @pytest.mark.unit
    def test_empty_reference(self) -> None:
        """Test CER with empty reference."""
        assert calculate_cer("", "hello") == 1.0
        assert calculate_cer("", "") == 0.0

    @pytest.mark.unit
    def test_empty_hypothesis(self) -> None:
        """Test CER with empty hypothesis."""
        # All characters deleted, CER = 5/5 = 1.0
        assert calculate_cer("hello", "") == 1.0

    @pytest.mark.unit
    def test_one_substitution(self) -> None:
        """Test CER with one substitution."""
        # "hello" -> "hallo" is 1 substitution, CER = 1/5 = 0.2
        assert calculate_cer("hello", "hallo") == 0.2

    @pytest.mark.unit
    def test_insertion(self) -> None:
        """Test CER with insertion."""
        # "hello" -> "helloo" is 1 insertion, CER = 1/5 = 0.2
        assert calculate_cer("hello", "helloo") == 0.2

    @pytest.mark.unit
    def test_deletion(self) -> None:
        """Test CER with deletion."""
        # "hello" -> "helo" is 1 deletion, CER = 1/5 = 0.2
        assert calculate_cer("hello", "helo") == 0.2

    @pytest.mark.unit
    def test_hindi_text(self) -> None:
        """Test CER with Hindi text."""
        ref = "नमस्ते"
        hyp = "नमस्ते"
        assert calculate_cer(ref, hyp) == 0.0

        # One character different
        hyp2 = "नमस्तो"
        cer = calculate_cer(ref, hyp2)
        assert 0.0 < cer < 1.0


class TestWER:
    """Tests for Word Error Rate calculation."""

    @pytest.mark.unit
    def test_identical_sentences(self) -> None:
        """Test WER is 0 for identical sentences."""
        assert calculate_wer("hello world", "hello world") == 0.0

    @pytest.mark.unit
    def test_completely_different(self) -> None:
        """Test WER for completely different sentences."""
        # 2 substitutions / 2 words = 1.0
        assert calculate_wer("hello world", "foo bar") == 1.0

    @pytest.mark.unit
    def test_empty_reference(self) -> None:
        """Test WER with empty reference."""
        assert calculate_wer("", "hello world") == 1.0
        assert calculate_wer("", "") == 0.0

    @pytest.mark.unit
    def test_empty_hypothesis(self) -> None:
        """Test WER with empty hypothesis."""
        # All words deleted, WER = 2/2 = 1.0
        assert calculate_wer("hello world", "") == 1.0

    @pytest.mark.unit
    def test_one_word_wrong(self) -> None:
        """Test WER with one word wrong."""
        # "hello world" -> "hello earth" is 1 substitution, WER = 1/2 = 0.5
        assert calculate_wer("hello world", "hello earth") == 0.5

    @pytest.mark.unit
    def test_word_insertion(self) -> None:
        """Test WER with word insertion."""
        # "hello world" -> "hello big world" is 1 insertion, WER = 1/2 = 0.5
        assert calculate_wer("hello world", "hello big world") == 0.5

    @pytest.mark.unit
    def test_word_deletion(self) -> None:
        """Test WER with word deletion."""
        # "hello big world" -> "hello world" is 1 deletion, WER = 1/3
        result = calculate_wer("hello big world", "hello world")
        assert abs(result - 1 / 3) < 0.001

    @pytest.mark.unit
    def test_hindi_text(self) -> None:
        """Test WER with Hindi text."""
        ref = "यह एक परीक्षण है"
        hyp = "यह एक परीक्षण है"
        assert calculate_wer(ref, hyp) == 0.0

        # One word different
        hyp2 = "यह एक जांच है"
        wer = calculate_wer(ref, hyp2)
        assert 0.0 < wer < 1.0
