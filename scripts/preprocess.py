#!/usr/bin/env python
# scripts/preprocess.py
"""
Run only the preprocessing stage.

Usage:
    python scripts/preprocess.py --input data/raw/ --output data/processed/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.audio_loader import load_batch, save_audio
from src.preprocessing.cleaner import clean_audio
from src.utils.config import load_config, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Preprocess EIT audio files")
    parser.add_argument("--input",  "-i", required=True, help="Input audio directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory for processed audio")
    parser.add_argument("--config", "-c", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("pipeline", {}).get("log_level", "INFO"))

    pp = config.get("preprocessing", {})
    batch = load_batch(args.input, target_sr=pp.get("target_sample_rate", 16000))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in batch:
        cleaned = clean_audio(
            item["audio"],
            item["sr"],
            noise_reduction=pp.get("noise_reduction", {}).get("enabled", True),
            silence_trimming=pp.get("silence_trimming", {}).get("enabled", True),
            normalization=pp.get("normalization", {}).get("enabled", True),
        )
        out_path = output_dir / (item["path"].stem + ".wav")
        save_audio(cleaned, item["sr"], out_path)
        print(f"Processed: {item['path'].name} → {out_path.name}")

    print(f"\nDone. {len(batch)} files processed.")


if __name__ == "__main__":
    main()
