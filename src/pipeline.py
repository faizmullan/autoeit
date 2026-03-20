# src/pipeline.py
"""
AutoEIT end-to-end pipeline orchestrator.

Chains:
  1. Load audio
  2. Preprocess (denoise, trim, normalize)
  3. Transcribe (Whisper)
  4. Post-process (filler removal, corrections)
  5. Save output

Usage:
    from src.pipeline import AutoEITPipeline
    pipeline = AutoEITPipeline(config_path="configs/default.yaml")
    pipeline.run(input_dir="data/raw/", output_dir="outputs/")
"""

import csv
import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class AutoEITPipeline:
    """
    Full AutoEIT transcription pipeline.

    Instantiate once, then call run() for each batch of audio files.
    """

    def __init__(self, config_path: str | Path = "configs/default.yaml"):
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.transcriber = None  # Loaded lazily to avoid slow startup

    def _load_config(self, path: str | Path) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        level = self.config.get("pipeline", {}).get("log_level", "INFO")
        logging.basicConfig(
            level=getattr(logging, level),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

    def _get_transcriber(self):
        """Lazy-load the ASR model (slow to initialize)."""
        if self.transcriber is None:
            from src.asr.transcriber import WhisperTranscriber
            asr_cfg = self.config.get("asr", {})
            self.transcriber = WhisperTranscriber(
                model_name=asr_cfg.get("model_name", "openai/whisper-small"),
                language=asr_cfg.get("language", "es"),
                task=asr_cfg.get("task", "transcribe"),
            )
        return self.transcriber

    def preprocess_file(self, audio, sr: int):
        """Run preprocessing on a single audio array."""
        from src.preprocessing.cleaner import clean_audio
        pp_cfg = self.config.get("preprocessing", {})
        return clean_audio(
            audio,
            sr,
            noise_reduction=pp_cfg.get("noise_reduction", {}).get("enabled", True),
            silence_trimming=pp_cfg.get("silence_trimming", {}).get("enabled", True),
            normalization=pp_cfg.get("normalization", {}).get("enabled", True),
        )

    def run(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
    ) -> list[dict]:
        """
        Run the full pipeline on all audio files in input_dir.

        Args:
            input_dir:  Directory with raw audio files.
            output_dir: Directory to write transcriptions.

        Returns:
            List of result dicts with "path" and "text" keys.
        """
        from src.preprocessing.audio_loader import load_batch, save_audio
        from src.postprocessing.corrector import postprocess_batch

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pp_cfg = self.config.get("preprocessing", {})
        asr_cfg = self.config.get("asr", {})
        post_cfg = self.config.get("postprocessing", {})
        save_intermediate = self.config.get("pipeline", {}).get("save_intermediate", True)

        # ── Stage 1: Load ──────────────────────────────────────────
        logger.info(f"[1/4] Loading audio from: {input_dir}")
        audio_batch = load_batch(
            input_dir,
            target_sr=pp_cfg.get("target_sample_rate", 16000),
        )
        if not audio_batch:
            logger.error("No audio files found. Exiting.")
            return []

        # ── Stage 2: Preprocess ────────────────────────────────────
        logger.info(f"[2/4] Preprocessing {len(audio_batch)} files...")
        processed_batch = []
        for item in audio_batch:
            cleaned = self.preprocess_file(item["audio"], item["sr"])
            item["audio"] = cleaned
            processed_batch.append(item)

            if save_intermediate:
                proc_path = output_dir / "processed" / item["path"].stem / ".wav"
                save_audio(cleaned, item["sr"], proc_path)

        # ── Stage 3: Transcribe ────────────────────────────────────
        logger.info("[3/4] Transcribing...")
        transcriber = self._get_transcriber()
        inf_cfg = asr_cfg.get("inference", {})
        results = transcriber.transcribe_batch(
            processed_batch,
            beam_size=inf_cfg.get("beam_size", 5),
            word_timestamps=inf_cfg.get("word_timestamps", True),
        )

        # ── Stage 4: Post-process ──────────────────────────────────
        logger.info("[4/4] Post-processing transcriptions...")
        results = postprocess_batch(
            results,
            lowercase=post_cfg.get("lowercase", True),
            strip_fillers=post_cfg.get("strip_filler_words", True),
            filler_words=set(post_cfg.get("filler_words", [])),
            lexicon_path=post_cfg.get("custom_lexicon_path"),
        )

        # ── Save output ────────────────────────────────────────────
        out_format = self.config.get("pipeline", {}).get("output_format", "csv")
        self._save_results(results, output_dir, fmt=out_format)

        logger.info(f"Pipeline complete. {len(results)} files transcribed.")
        return results

    def _save_results(self, results: list[dict], output_dir: Path, fmt: str = "csv"):
        """Save transcription results to CSV or JSON."""
        if fmt == "csv":
            out_path = output_dir / "transcriptions.csv"
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["filename", "transcript", "raw_transcript"])
                writer.writeheader()
                for r in results:
                    writer.writerow({
                        "filename": Path(r["path"]).name,
                        "transcript": r.get("text", ""),
                        "raw_transcript": r.get("raw_text", ""),
                    })
            logger.info(f"Transcriptions saved to: {out_path}")

        elif fmt == "json":
            out_path = output_dir / "transcriptions.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    [{"filename": Path(r["path"]).name, **r} for r in results],
                    f, indent=2, default=str,
                )
            logger.info(f"Transcriptions saved to: {out_path}")
