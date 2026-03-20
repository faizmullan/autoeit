#!/usr/bin/env python
# scripts/evaluate.py
"""
Evaluate transcription output against human reference transcripts.

Usage:
    python scripts/evaluate.py \
        --predictions outputs/transcriptions.csv \
        --references data/transcripts/human_reference.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.evaluation.metrics import compute_wer, compute_cer, compute_human_agreement
from src.utils.config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Evaluate AutoEIT transcriptions")
    parser.add_argument("--predictions", "-p", required=True,
                        help="CSV with columns: filename, transcript")
    parser.add_argument("--references",  "-r", required=True,
                        help="CSV with columns: filename, transcript")
    parser.add_argument("--output", "-o", default="outputs/evaluation_report.json")
    args = parser.parse_args()

    setup_logging()

    preds_df = pd.read_csv(args.predictions)
    refs_df  = pd.read_csv(args.references)

    # Normalise filename column (drop directory, keep basename)
    for df in [preds_df, refs_df]:
        df["filename"] = df["filename"].apply(lambda x: Path(x).name)

    merged = preds_df.merge(refs_df, on="filename", suffixes=("_pred", "_ref"))

    if merged.empty:
        print("No matching filenames between predictions and references.")
        sys.exit(1)

    pred_texts = merged["transcript_pred"].fillna("").tolist()
    ref_texts  = merged["transcript_ref"].fillna("").tolist()

    wer_result  = compute_wer(pred_texts, ref_texts)
    cer_score   = compute_cer(pred_texts, ref_texts)
    agreement   = compute_human_agreement(pred_texts, ref_texts)

    print("\n" + "="*50)
    print("  AutoEIT Evaluation Report")
    print("="*50)
    print(f"  Files evaluated : {len(merged)}")
    print(f"  WER             : {wer_result['wer']:.1%}")
    print(f"  CER             : {cer_score:.1%}")
    print(f"  Human agreement : {agreement['agreement_rate']:.1%}")
    print(f"  Goal (≥90%) met : {'YES ✓' if agreement['agreement_rate'] >= 0.9 else 'NO ✗'}")
    print("="*50)

    if agreement["disagreements"]:
        print(f"\nSample disagreements (first 5):")
        for d in agreement["disagreements"][:5]:
            print(f"  PRED: {d['prediction']}")
            print(f"  REF : {d['reference']}")
            print()

    import json
    report = {
        "files_evaluated": len(merged),
        "wer": wer_result,
        "cer": cer_score,
        "human_agreement": agreement,
        "goal_met": agreement["agreement_rate"] >= 0.9,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Full report saved to: {out}")


if __name__ == "__main__":
    main()
