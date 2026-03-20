#!/usr/bin/env python
# scripts/run_pipeline.py
"""
CLI entry point to run the full AutoEIT transcription pipeline.

Usage:
    python scripts/run_pipeline.py \
        --input data/raw/ \
        --output outputs/ \
        --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

# Make sure src/ is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import AutoEITPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="AutoEIT: Audio-to-text transcription for L2 learner EIT data"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Directory containing raw audio files (.wav, .mp3, etc.)",
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs/",
        help="Directory to write transcription results (default: outputs/)",
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after transcription (requires human transcripts in config)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = AutoEITPipeline(config_path=args.config)
    results = pipeline.run(
        input_dir=args.input,
        output_dir=args.output,
    )

    if args.evaluate and results:
        from src.evaluation.metrics import evaluate
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        eval_cfg = config.get("evaluation", {})
        refs = eval_cfg.get("human_transcripts_path")
        if not refs:
            print("No human_transcripts_path set in config — skipping evaluation.")
        else:
            report = evaluate(
                predictions=results,
                references_path=refs,
                output_path=eval_cfg.get("output_report", "outputs/evaluation_report.json"),
            )
            print(f"\nAgreement rate: {report['human_agreement']['agreement_rate']:.1%}")
            print(f"WER: {report['wer']['wer']:.1%}")
            print(f"Goal (≥90%) met: {report['goal_met']}")


if __name__ == "__main__":
    main()
