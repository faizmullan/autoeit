# src/preprocessing/cleaner.py
"""
Audio cleaning: noise reduction, silence trimming, loudness normalization.
These steps are critical for learner speech which is often recorded in
non-studio environments with background noise.
"""

import logging

import librosa
import numpy as np
import noisereduce as nr

logger = logging.getLogger(__name__)


def reduce_noise(
    audio: np.ndarray,
    sr: int,
    stationary: bool = True,
    prop_decrease: float = 0.8,
) -> np.ndarray:
    """
    Apply spectral noise reduction.

    For EIT recordings, stationary=True works well for consistent
    background noise (room tone, AC hum). Set stationary=False for
    recordings with intermittent noise.

    Args:
        audio:          float32 numpy array.
        sr:             Sample rate.
        stationary:     If True, assumes noise is constant across the clip.
        prop_decrease:  How much to reduce noise (0 = none, 1 = full).

    Returns:
        Cleaned float32 numpy array.
    """
    logger.debug("Applying noise reduction...")
    cleaned = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=stationary,
        prop_decrease=prop_decrease,
    )
    return cleaned.astype(np.float32)


def trim_silence(
    audio: np.ndarray,
    sr: int,
    top_db: float = 30,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Trim leading and trailing silence.

    A lower top_db is stricter (trims more). For learner speech,
    30–40 dB is a good default — learners often pause before speaking.

    Args:
        audio:          float32 numpy array.
        sr:             Sample rate.
        top_db:         Silence threshold in dB below peak amplitude.
        frame_length:   FFT window size.
        hop_length:     Hop size between frames.

    Returns:
        Trimmed float32 numpy array.
    """
    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    original_dur = len(audio) / sr
    trimmed_dur = len(trimmed) / sr
    logger.debug(
        f"Silence trimmed: {original_dur:.2f}s → {trimmed_dur:.2f}s"
    )
    return trimmed


def normalize_loudness(
    audio: np.ndarray,
    target_lufs: float = -23.0,
) -> np.ndarray:
    """
    Normalize audio to a target loudness level (RMS-based approximation).

    True LUFS normalization requires pyloudnorm; this is a lightweight
    RMS approximation that works well for speech.

    Args:
        audio:          float32 numpy array.
        target_lufs:    Target loudness in LUFS (EBU R128 standard: -23).

    Returns:
        Loudness-normalized float32 numpy array.
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-8:
        logger.warning("Audio is near-silent; skipping normalization.")
        return audio

    # Convert target LUFS to linear RMS (rough approximation)
    target_rms = 10 ** (target_lufs / 20)
    gain = target_rms / rms
    normalized = audio * gain

    # Clip to prevent clipping artifacts
    normalized = np.clip(normalized, -1.0, 1.0)
    return normalized.astype(np.float32)


def clean_audio(
    audio: np.ndarray,
    sr: int,
    noise_reduction: bool = True,
    silence_trimming: bool = True,
    normalization: bool = True,
    noise_reduction_kwargs: dict | None = None,
    trim_kwargs: dict | None = None,
    normalize_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Run the full cleaning pipeline on a single audio array.

    This is the main entry point — call this from the pipeline.

    Args:
        audio:                  Raw float32 numpy array.
        sr:                     Sample rate.
        noise_reduction:        Whether to apply noise reduction.
        silence_trimming:       Whether to trim leading/trailing silence.
        normalization:          Whether to normalize loudness.
        noise_reduction_kwargs: Passed to reduce_noise().
        trim_kwargs:            Passed to trim_silence().
        normalize_kwargs:       Passed to normalize_loudness().

    Returns:
        Cleaned float32 numpy array.
    """
    if noise_reduction:
        audio = reduce_noise(audio, sr, **(noise_reduction_kwargs or {}))

    if silence_trimming:
        audio = trim_silence(audio, sr, **(trim_kwargs or {}))

    if normalization:
        audio = normalize_loudness(audio, **(normalize_kwargs or {}))

    return audio
