# tests/test_preprocessing/test_cleaner.py
"""
Unit tests for audio preprocessing.
Tests run on synthetic audio — no real EIT recordings required.
"""

import numpy as np
import pytest

from src.preprocessing.cleaner import (
    normalize_loudness,
    remove_filler_words,
    trim_silence,
)
from src.postprocessing.corrector import postprocess


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def silent_audio():
    """1 second of silence at 16kHz."""
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def tone_audio():
    """1 second of a 440Hz sine tone at 16kHz (clearly not silent)."""
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def padded_audio(tone_audio):
    """Tone with 0.5s of silence on each end."""
    silence = np.zeros(8000, dtype=np.float32)
    return np.concatenate([silence, tone_audio, silence])


# ── Normalization tests ────────────────────────────────────────────────────────

def test_normalize_loudness_changes_rms(tone_audio):
    original_rms = np.sqrt(np.mean(tone_audio ** 2))
    normalized = normalize_loudness(tone_audio, target_lufs=-23.0)
    new_rms = np.sqrt(np.mean(normalized ** 2))
    assert abs(new_rms - original_rms) > 1e-4, "RMS should change after normalization"


def test_normalize_loudness_clips_to_range(tone_audio):
    # Amplify first, then normalize — result must stay in [-1, 1]
    loud = tone_audio * 100
    normalized = normalize_loudness(loud, target_lufs=-3.0)
    assert normalized.max() <= 1.0
    assert normalized.min() >= -1.0


def test_normalize_silent_audio_returns_unchanged(silent_audio):
    """Near-silent audio should pass through without error."""
    result = normalize_loudness(silent_audio)
    np.testing.assert_array_equal(result, silent_audio)


# ── Silence trimming tests ─────────────────────────────────────────────────────

def test_trim_silence_removes_padding(padded_audio):
    sr = 16000
    trimmed = trim_silence(padded_audio, sr=sr, top_db=30)
    # Trimmed should be shorter than original (padding removed)
    assert len(trimmed) < len(padded_audio)


def test_trim_silence_preserves_tone_content(padded_audio, tone_audio):
    sr = 16000
    trimmed = trim_silence(padded_audio, sr=sr, top_db=30)
    # Should be close in length to the original tone
    assert abs(len(trimmed) - len(tone_audio)) < 2000  # within 0.125s


# ── Post-processing tests ──────────────────────────────────────────────────────

def test_remove_filler_words_basic():
    text = "um yo uh quiero ir a la tienda"
    result = postprocess(text, lowercase=True, strip_fillers=True, apply_corrections=False)
    assert "um" not in result
    assert "uh" not in result
    assert "quiero" in result


def test_postprocess_lowercases():
    text = "Yo QUIERO Comer"
    result = postprocess(text, lowercase=True, strip_fillers=False, apply_corrections=False)
    assert result == "yo quiero comer"


def test_postprocess_collapses_spaces():
    text = "yo   quiero    comer"
    result = postprocess(text, lowercase=True, strip_fillers=False, apply_corrections=False)
    assert "  " not in result


def test_postprocess_empty_string():
    result = postprocess("", lowercase=True, strip_fillers=True, apply_corrections=False)
    assert result == ""
