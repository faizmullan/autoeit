# src/asr/transcriber.py
"""
ASR inference using OpenAI Whisper (via Hugging Face transformers).
Designed for non-native Spanish EIT responses.
"""

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """
    Wrapper around Whisper for EIT transcription.

    Usage:
        transcriber = WhisperTranscriber(model_name="openai/whisper-small")
        result = transcriber.transcribe(audio_array, sr=16000)
        print(result["text"])
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        language: str = "es",
        task: str = "transcribe",
        device: str | None = None,
    ):
        """
        Args:
            model_name:  HuggingFace model ID or path to fine-tuned checkpoint.
            language:    ISO 639-1 language code. "es" for Spanish.
            task:        "transcribe" (keep language) or "translate" (to English).
            device:      "cuda", "cpu", or None (auto-detect).
        """
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self.model_name = model_name
        self.language = language
        self.task = task
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading model: {model_name} on {self.device}")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device=self.device,
        )

        logger.info("Model loaded successfully.")

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        beam_size: int = 5,
        word_timestamps: bool = True,
    ) -> dict:
        """
        Transcribe a single audio array.

        Args:
            audio:           float32 numpy array at 16kHz.
            sr:              Sample rate (must be 16000 for Whisper).
            beam_size:       Beam search width. Higher = more accurate but slower.
            word_timestamps: If True, return word-level timestamps.

        Returns:
            dict with keys:
                "text"   — transcribed string
                "chunks" — list of {"text": str, "timestamp": [start, end]}
                           (only if word_timestamps=True)
        """
        if sr != 16000:
            raise ValueError(f"Whisper requires 16000Hz audio, got {sr}Hz.")

        generate_kwargs = {
            "language": self.language,
            "task": self.task,
            "num_beams": beam_size,
        }

        result = self.pipe(
            {"raw": audio, "sampling_rate": sr},
            generate_kwargs=generate_kwargs,
            return_timestamps="word" if word_timestamps else False,
        )

        return result

    def transcribe_file(self, path: str | Path, **kwargs) -> dict:
        """
        Convenience method: load and transcribe a file.

        Args:
            path:    Path to a 16kHz mono WAV file.
            **kwargs: Passed to transcribe().
        """
        from src.preprocessing.audio_loader import load_audio

        audio, sr = load_audio(path, target_sr=16000, mono=True)
        return self.transcribe(audio, sr=sr, **kwargs)

    def transcribe_batch(
        self,
        audio_list: list[dict],
        **kwargs,
    ) -> list[dict]:
        """
        Transcribe a batch of audio dicts (output from audio_loader.load_batch).

        Args:
            audio_list: [{"path": Path, "audio": np.ndarray, "sr": int}, ...]

        Returns:
            [{"path": Path, "text": str, "chunks": [...]}, ...]
        """
        results = []
        for item in audio_list:
            logger.info(f"Transcribing: {item['path'].name}")
            try:
                result = self.transcribe(item["audio"], sr=item["sr"], **kwargs)
                results.append({
                    "path": item["path"],
                    "text": result["text"].strip(),
                    "chunks": result.get("chunks", []),
                })
            except Exception as e:
                logger.error(f"Failed to transcribe {item['path'].name}: {e}")
                results.append({
                    "path": item["path"],
                    "text": "",
                    "chunks": [],
                    "error": str(e),
                })
        return results
