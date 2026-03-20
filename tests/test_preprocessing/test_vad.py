# tests/test_preprocessing/test_vad.py
"""
Tests for the fixed VAD module.
Specifically verifies the reshape bug is gone.
"""

import numpy as np
import pytest

from src.preprocessing.vad import vad_analyze, filter_by_vad

SR = 16000


def _speech_audio(n_samples: int, speech_ratio: float = 0.7) -> np.ndarray:
    """Make a synthetic audio array with speech in the middle."""
    audio = np.zeros(n_samples, dtype=np.float32)
    start = int(n_samples * (1 - speech_ratio) / 2)
    end   = int(n_samples * (1 + speech_ratio) / 2)
    t = np.linspace(0, (end - start) / SR, end - start)
    audio[start:end] = 0.3 * np.sin(2 * np.pi * 220 * t)
    audio += 0.005 * np.random.default_rng(42).standard_normal(n_samples).astype(np.float32)
    return audio


# ── Regression: the exact sizes that crashed before ──────────────────────────

@pytest.mark.parametrize("n_samples", [
    67701, 68137, 68184, 61023, 32182,
    78273, 65939, 52800, 62878, 89680,
])
def test_no_reshape_crash(n_samples):
    """Every previously-crashing size must run without exception."""
    audio = _speech_audio(n_samples)
    result = vad_analyze(audio, sr=SR)   # must not raise
    assert "quality" in result
    assert result["total_duration_s"] == pytest.approx(n_samples / SR, abs=0.01)


# ── Output contract ───────────────────────────────────────────────────────────

def test_output_keys():
    audio = _speech_audio(SR * 2)
    r = vad_analyze(audio, SR)
    for key in ["total_duration_s", "speech_duration_s", "speech_ratio",
                "num_speech_segments", "snr_db", "has_clipping", "quality", "issues"]:
        assert key in r, f"Missing key: {key}"


def test_speech_ratio_bounds():
    audio = _speech_audio(SR * 2, speech_ratio=0.6)
    r = vad_analyze(audio, SR)
    assert 0.0 <= r["speech_ratio"] <= 1.0


def test_silent_file_low_ratio():
    # Realistic near-silence: very low level noise with one short burst
    # Pure zeros gives speech_ratio=1.0 (librosa edge case with zero-peak signal)
    rng = np.random.default_rng(0)
    silent = (rng.standard_normal(SR * 3) * 0.0001).astype(np.float32)  # -80 dBFS noise floor
    # Add a tiny speech burst (0.2s out of 3s = 6.7%)
    silent[SR : SR + SR // 5] += 0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 0.2, SR // 5))
    r = vad_analyze(silent, SR, top_db=30)
    assert r["speech_ratio"] < 0.4, f"Expected mostly silent, got speech_ratio={r['speech_ratio']}"


def test_clipping_detected():
    clipped = np.ones(SR, dtype=np.float32)  # all at max amplitude → clipping
    r = vad_analyze(clipped, SR)
    assert r["has_clipping"] is True


def test_no_clipping_on_normal_audio():
    audio = _speech_audio(SR * 2)
    r = vad_analyze(audio, SR)
    assert r["has_clipping"] is False


# ── filter_by_vad ─────────────────────────────────────────────────────────────

def _make_result(speech_ratio, snr_db, clipping=False):
    return {
        "path": "x.wav", "filename": "x.wav", "speaker": "MBMPS",
        "speech_ratio": speech_ratio, "snr_db": snr_db,
        "has_clipping": clipping, "quality": "good", "issues": [],
    }


def test_filter_keeps_good_files():
    results = [_make_result(0.8, 20.0)]
    kept, rejected = filter_by_vad(results, min_speech_ratio=0.3, min_snr_db=10.0)
    assert len(kept) == 1
    assert len(rejected) == 0


def test_filter_rejects_low_speech():
    results = [_make_result(0.1, 20.0)]
    kept, rejected = filter_by_vad(results, min_speech_ratio=0.3)
    assert len(kept) == 0
    assert len(rejected) == 1


def test_filter_rejects_low_snr():
    results = [_make_result(0.8, 3.0)]
    kept, rejected = filter_by_vad(results, min_snr_db=10.0)
    assert len(kept) == 0
    assert len(rejected) == 1


def test_filter_rejects_clipped():
    results = [_make_result(0.8, 20.0, clipping=True)]
    kept, rejected = filter_by_vad(results, exclude_clipping=True)
    assert len(kept) == 0
    assert len(rejected) == 1


def test_filter_allows_clipped_when_disabled():
    results = [_make_result(0.8, 20.0, clipping=True)]
    kept, rejected = filter_by_vad(results, exclude_clipping=False)
    assert len(kept) == 1
