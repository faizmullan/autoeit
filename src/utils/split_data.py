# src/utils/split_data.py
"""
Split labelled EIT data (audio + human transcripts) into
train / validation / test manifests for fine-tuning.

Usage:
    python src/utils/split_data.py \
        --audio_dir data/processed/ \
        --transcripts data/transcripts/human_reference.csv \
        --output_dir data/splits/ \
        --train 0.8 --val 0.1 --test 0.1
"""

import argparse
import random
from pathlib import Path

import pandas as pd


def create_splits(
    audio_dir: str | Path,
    transcripts_csv: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Match processed audio files with human transcripts and create splits.

    The transcripts CSV must have columns:
        filename   — audio filename (e.g. speaker_001_item_01.wav)
        transcript — human reference transcription

    Args:
        audio_dir:       Directory with processed .wav files.
        transcripts_csv: CSV of human reference transcriptions.
        output_dir:      Where to write train.csv, val.csv, test.csv.
        train_ratio:     Fraction for training (default 0.8).
        val_ratio:       Fraction for validation (default 0.1).
        test_ratio:      Fraction for test (default 0.1).
        seed:            Random seed for reproducibility.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    refs = pd.read_csv(transcripts_csv)
    refs["filename"] = refs["filename"].apply(lambda x: Path(x).name)

    # Match audio files to transcripts
    audio_files = {f.name: f for f in audio_dir.glob("*.wav")}
    matched = refs[refs["filename"].isin(audio_files)].copy()
    matched["audio_path"] = matched["filename"].map(
        lambda fn: str(audio_files[fn])
    )

    print(f"Matched {len(matched)} / {len(refs)} transcripts to audio files.")

    # Shuffle and split
    rows = matched.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(rows)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = rows.iloc[:n_train]
    val = rows.iloc[n_train:n_train + n_val]
    test = rows.iloc[n_train + n_val:]

    # Save splits — keep only audio_path and sentence columns (HuggingFace format)
    cols = ["audio_path", "transcript"]
    train[cols].rename(columns={"transcript": "sentence"}).to_csv(
        output_dir / "train.csv", index=False
    )
    val[cols].rename(columns={"transcript": "sentence"}).to_csv(
        output_dir / "val.csv", index=False
    )
    test[cols].rename(columns={"transcript": "sentence"}).to_csv(
        output_dir / "test.csv", index=False
    )

    print(f"Splits saved to {output_dir}")
    print(f"  train: {len(train)}  val: {len(val)}  test: {len(test)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--transcripts", required=True)
    parser.add_argument("--output_dir", default="data/splits/")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val",   type=float, default=0.1)
    parser.add_argument("--test",  type=float, default=0.1)
    parser.add_argument("--seed",  type=int,   default=42)
    args = parser.parse_args()

    create_splits(
        audio_dir=args.audio_dir,
        transcripts_csv=args.transcripts,
        output_dir=args.output_dir,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )
