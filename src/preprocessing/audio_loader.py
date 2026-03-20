# src/preprocessing/audio_loader.py
"""
Load audio files from disk and convert to a standard format:
16kHz, mono, float32 numpy array.
Supports .wav, .mp3, .m4a, .flac, .ogg
"""

import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}


def load_audio(
    path: str | Path,
    target_sr: int = 16000,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load an audio file and resample to target_sr.

    Args:
        path:       Path to the audio file.
        target_sr:  Target sample rate in Hz. Whisper requires 16000.
        mono:       If True, convert stereo to mono by averaging channels.

    Returns:
        (audio, sample_rate) — float32 numpy array in range [-1, 1], sample rate.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError:        if the file format is unsupported.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{path.suffix}'. Supported: {SUPPORTED_FORMATS}"
        )

    logger.debug(f"Loading audio: {path}")

    # librosa handles resampling and mono conversion natively
    audio, sr = librosa.load(path, sr=target_sr, mono=mono, dtype=np.float32)

    logger.debug(f"Loaded {path.name}: {len(audio)/sr:.2f}s at {sr}Hz")
    return audio, sr


def load_batch(
    folder: str | Path,
    target_sr: int = 16000,
    recursive: bool = False,
) -> list[dict]:
    """
    Load all supported audio files from a folder.

    Returns:
        List of dicts: [{"path": Path, "audio": np.ndarray, "sr": int}, ...]
    """
    folder = Path(folder)
    pattern = "**/*" if recursive else "*"
    files = [
        f for f in folder.glob(pattern)
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    ]

    if not files:
        logger.warning(f"No audio files found in: {folder}")
        return []

    logger.info(f"Found {len(files)} audio files in {folder}")

    results = []
    for f in sorted(files):
        try:
            audio, sr = load_audio(f, target_sr=target_sr)
            results.append({"path": f, "audio": audio, "sr": sr})
        except Exception as e:
            logger.error(f"Failed to load {f.name}: {e}")

    return results


def save_audio(audio: np.ndarray, sr: int, path: str | Path) -> None:
    """Save a float32 numpy array as a WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_16")
    logger.debug(f"Saved audio to {path}")
