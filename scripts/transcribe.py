#!/usr/bin/env python
# scripts/transcribe.py
"""
Run only the ASR transcription stage (expects preprocessed audio).

Usage:
    python scripts/transcribe.py --input data/processed/ --output outputs/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.asr.transcriber import WhisperTranscriber
from src.preprocessing.audio_loader import load_batch
from src.postprocessing.corrector import postprocess_batch
from src.utils.config import load_config, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Transcribe preprocessed EIT audio")
    parser.add_argument("--input",  "-i", required=True)
    parser.add_argument("--output", "-o", default="outputs/")
    parser.add_argument("--config", "-c", default="configs/default.yaml")
    parser.add_argument("--model",  "-m", default=None,
                        help="Override model name from config")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("pipeline", {}).get("log_level", "INFO"))

    asr_cfg  = config.get("asr", {})
    post_cfg = config.get("postprocessing", {})

    model_name = args.model or asr_cfg.get("model_name", "openai/whisper-small")
    transcriber = WhisperTranscriber(
        model_name=model_name,
        language=asr_cfg.get("language", "es"),
        task=asr_cfg.get("task", "transcribe"),
    )

    batch = load_batch(args.input, target_sr=16000)
    results = transcriber.transcribe_batch(batch)
    results = postprocess_batch(
        results,
        lowercase=post_cfg.get("lowercase", True),
        strip_fillers=post_cfg.get("strip_filler_words", True),
        filler_words=set(post_cfg.get("filler_words", [])),
        lexicon_path=post_cfg.get("custom_lexicon_path"),
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "transcriptions.csv"

    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "transcript", "raw_transcript"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "filename": Path(r["path"]).name,
                "transcript": r.get("text", ""),
                "raw_transcript": r.get("raw_text", ""),
            })

    print(f"\nTranscriptions saved to: {out_csv}")
    print(f"Total files: {len(results)}")


if __name__ == "__main__":
    main()
