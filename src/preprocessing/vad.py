# src/preprocessing/vad.py
"""
Voice Activity Detection (VAD) for EIT audio files.

The original pipeline crashed with:
    "cannot reshape array of size N into shape (512)"
because it did audio.reshape(-1, 512) — which only works when
len(audio) is an exact multiple of 512. Audio files never are.

Fix: use librosa.effects.split() which works on any length by using
a sliding window internally with hop_length, not a hard reshape.

This module provides:
  - vad_analyze()  : analyze a single audio file → speech stats
  - vad_batch()    : analyze a folder of files → DataFrame
  - filter_by_vad(): drop files with too little speech
"""

import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# ── Core VAD function ─────────────────────────────────────────────────────────

def vad_analyze(
    audio: np.ndarray,
    sr: int,
    top_db: float = 30,
    frame_length: int = 2048,   # ~128ms at 16kHz — safe for any file length
    hop_length: int  = 512,     # ~32ms hop
) -> dict:
    """
    Analyze voice activity in an audio array.

    Uses librosa.effects.split() which pads internally — no reshape crash.

    Args:
        audio:        float32 numpy array.
        sr:           Sample rate (Hz).
        top_db:       Silence threshold. 30 dB is good for clean recordings;
                      use 20 dB for noisy environments.
        frame_length: FFT window size. 2048 works for any audio length ≥ 1 frame.
        hop_length:   Hop size between frames.

    Returns:
        dict with:
            total_duration_s    — length of the full clip
            speech_duration_s   — estimated speech duration
            speech_ratio        — speech / total (0–1)
            num_speech_segments — number of non-silent intervals
            snr_db              — rough signal-to-noise estimate
            has_clipping        — True if any sample exceeds 0.99
            quality             — "excellent" / "good" / "poor"
            issues              — list of detected problems (empty if clean)
    """
    total_duration_s = len(audio) / sr
    issues = []

    # ── Speech segmentation (the fixed version) ───────────────────────────────
    # librosa.effects.split uses librosa.feature.rms internally with
    # frame_length and hop_length — handles any audio length gracefully.
    try:
        intervals = librosa.effects.split(
            audio,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        speech_samples = sum(end - start for start, end in intervals)
        speech_duration_s = speech_samples / sr
        num_segments = len(intervals)
    except Exception as e:
        logger.warning(f"VAD failed: {e} — treating entire clip as speech")
        speech_duration_s = total_duration_s
        num_segments = 1
        issues.append(f"vad_error: {e}")

    speech_ratio = speech_duration_s / total_duration_s if total_duration_s > 0 else 0.0

    # ── SNR estimate ──────────────────────────────────────────────────────────
    # Simple but effective: compare RMS of speech vs silent regions
    snr_db = _estimate_snr(audio, intervals, sr)

    # ── Clipping detection ────────────────────────────────────────────────────
    has_clipping = bool(np.any(np.abs(audio) > 0.99))
    if has_clipping:
        issues.append("clipping")

    # ── Quality tier ──────────────────────────────────────────────────────────
    if speech_ratio < 0.2:
        quality = "poor"
        issues.append(f"low_speech_ratio ({speech_ratio:.2f})")
    elif snr_db < 10:
        quality = "poor"
        issues.append(f"low_snr ({snr_db:.1f} dB)")
    elif speech_ratio < 0.5 or snr_db < 20:
        quality = "good"
    else:
        quality = "excellent"

    return {
        "total_duration_s":     round(total_duration_s, 3),
        "speech_duration_s":    round(speech_duration_s, 3),
        "speech_ratio":         round(speech_ratio, 4),
        "num_speech_segments":  num_segments,
        "snr_db":               round(snr_db, 2),
        "has_clipping":         has_clipping,
        "quality":              quality,
        "issues":               issues,
    }


def _estimate_snr(
    audio: np.ndarray,
    speech_intervals: np.ndarray,
    sr: int,
) -> float:
    """
    Estimate SNR by comparing speech RMS vs noise RMS.
    Falls back to a simple energy estimate if intervals are empty.
    """
    if len(speech_intervals) == 0 or len(audio) == 0:
        return 0.0

    # Collect speech samples
    speech_mask = np.zeros(len(audio), dtype=bool)
    for start, end in speech_intervals:
        speech_mask[start:end] = True

    speech_audio = audio[speech_mask]
    noise_audio  = audio[~speech_mask]

    speech_rms = np.sqrt(np.mean(speech_audio ** 2)) if len(speech_audio) > 0 else 1e-8
    noise_rms  = np.sqrt(np.mean(noise_audio  ** 2)) if len(noise_audio)  > 0 else 1e-8

    if noise_rms < 1e-8:
        return 40.0   # essentially silent background → excellent SNR

    snr = 20 * np.log10(speech_rms / noise_rms + 1e-8)
    return float(np.clip(snr, -20, 60))


# ── Batch processing ──────────────────────────────────────────────────────────

def vad_batch(
    folder: str | Path,
    sr: int = 16000,
    top_db: float = 30,
    recursive: bool = False,
) -> list[dict]:
    """
    Run VAD analysis on all WAV files in a folder.

    Args:
        folder:     Directory containing WAV files.
        sr:         Sample rate to load at (resamples if needed).
        top_db:     Silence threshold for librosa.effects.split.
        recursive:  If True, scan subdirectories.

    Returns:
        List of dicts — one per file — with VAD stats plus "path" key.
    """
    folder = Path(folder)
    pattern = "**/*.wav" if recursive else "*.wav"
    files = sorted(folder.glob(pattern))

    if not files:
        logger.warning(f"No WAV files found in: {folder}")
        return []

    logger.info(f"Running VAD on {len(files)} files...")
    results = []

    for wav_path in files:
        try:
            audio, file_sr = librosa.load(wav_path, sr=sr, mono=True)
            stats = vad_analyze(audio, sr=sr, top_db=top_db)
            stats["path"]     = str(wav_path)
            stats["filename"] = wav_path.name
            stats["speaker"]  = wav_path.stem.split("_")[0]  # e.g. MBMPS from MBMPS_arctic_a0001
            results.append(stats)
        except Exception as e:
            logger.error(f"Failed to process {wav_path.name}: {e}")
            results.append({
                "path":     str(wav_path),
                "filename": wav_path.name,
                "speaker":  wav_path.stem.split("_")[0],
                "quality":  "error",
                "issues":   [str(e)],
            })

    # Print summary
    quality_counts = {}
    for r in results:
        q = r.get("quality", "error")
        quality_counts[q] = quality_counts.get(q, 0) + 1

    logger.info("VAD complete:")
    for q, count in sorted(quality_counts.items()):
        logger.info(f"  {q:<12}: {count:>5} files")

    return results


def filter_by_vad(
    results: list[dict],
    min_speech_ratio: float = 0.3,
    min_snr_db: float = 5.0,
    exclude_clipping: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Split VAD results into (kept, rejected) based on quality thresholds.

    Args:
        results:            Output of vad_batch().
        min_speech_ratio:   Minimum fraction of clip that must be speech.
        min_snr_db:         Minimum SNR in dB.
        exclude_clipping:   If True, reject clipped files.

    Returns:
        (kept, rejected) — two lists of result dicts.
    """
    kept, rejected = [], []

    for r in results:
        if r.get("quality") == "error":
            rejected.append(r)
            continue

        reasons = []
        if r.get("speech_ratio", 0) < min_speech_ratio:
            reasons.append(f"speech_ratio={r['speech_ratio']:.2f} < {min_speech_ratio}")
        if r.get("snr_db", 0) < min_snr_db:
            reasons.append(f"snr={r['snr_db']:.1f}dB < {min_snr_db}dB")
        if exclude_clipping and r.get("has_clipping"):
            reasons.append("clipping")

        if reasons:
            r["rejection_reasons"] = reasons
            rejected.append(r)
        else:
            kept.append(r)

    logger.info(
        f"VAD filter: kept={len(kept)} | rejected={len(rejected)} "
        f"(speech<{min_speech_ratio}, snr<{min_snr_db}dB)"
    )
    return kept, rejected
