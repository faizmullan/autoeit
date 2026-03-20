# src/data/l2arctic_loader.py
"""
L2-Arctic dataset loader.

Parses the L2-Arctic directory structure, matches WAV files with their
TXT transcripts, and produces CSV manifests ready for Whisper fine-tuning.

L2-Arctic structure (per speaker):
    <root>/
        <SPEAKER_ID>/
            wav/           ← audio at 44.1kHz
            transcript/    ← one .txt per utterance (the reference text)
            textgrid/      ← forced-aligned phonemes (optional)
            annotation/    ← manual mispronunciation tags (optional)

Usage:
    python src/data/l2arctic_loader.py \
        --dataset_dir /path/to/l2-arctic \
        --output_dir  data/splits/ \
        --l1 spanish  # optional: filter by L1 background
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# All 24 speakers with their L1 and gender
# Source: official L2-Arctic documentation
SPEAKER_METADATA = {
    # Arabic
    "ABA":   {"l1": "arabic",     "gender": "M"},
    "SKA":   {"l1": "arabic",     "gender": "F"},
    "YBAA":  {"l1": "arabic",     "gender": "M"},
    "ZHAA":  {"l1": "arabic",     "gender": "F"},
    # Chinese (Mandarin)
    "BWC":   {"l1": "mandarin",   "gender": "M"},
    "LXC":   {"l1": "mandarin",   "gender": "F"},
    "NCC":   {"l1": "mandarin",   "gender": "F"},
    "TXHC":  {"l1": "mandarin",   "gender": "M"},
    # Hindi
    "ASI":   {"l1": "hindi",      "gender": "M"},
    "RRBI":  {"l1": "hindi",      "gender": "M"},
    "SVBI":  {"l1": "hindi",      "gender": "F"},
    "TNI":   {"l1": "hindi",      "gender": "F"},
    # Korean
    "HJK":   {"l1": "korean",     "gender": "F"},
    "HKK":   {"l1": "korean",     "gender": "M"},
    "YDCK":  {"l1": "korean",     "gender": "F"},
    "YKWK":  {"l1": "korean",     "gender": "M"},
    # Spanish  ← most relevant for the AutoEIT EIT project
    "EBVS":  {"l1": "spanish",    "gender": "M"},
    "ERMS":  {"l1": "spanish",    "gender": "M"},
    "MBMPS": {"l1": "spanish",    "gender": "F"},
    "NJS":   {"l1": "spanish",    "gender": "F"},
    # Vietnamese
    "HQTV":  {"l1": "vietnamese", "gender": "M"},
    "PNV":   {"l1": "vietnamese", "gender": "F"},
    "THV":   {"l1": "vietnamese", "gender": "F"},
    "TLV":   {"l1": "vietnamese", "gender": "M"},
}


def build_manifest(
    dataset_dir: str | Path,
    l1_filter: list[str] | None = None,
    speakers_filter: list[str] | None = None,
    max_duration_s: float | None = None,
    min_duration_s: float = 0.5,
) -> pd.DataFrame:
    """
    Scan the L2-Arctic directory and build a manifest DataFrame.

    Each row is one utterance:
        audio_path   — absolute path to the WAV file
        sentence     — reference transcript text
        speaker      — speaker ID (e.g. "EBVS")
        l1           — speaker's native language
        gender       — M or F
        duration_s   — audio duration in seconds (estimated from filename; exact after loading)

    Args:
        dataset_dir:     Root of the L2-Arctic corpus.
        l1_filter:       If set, keep only speakers with these L1s.
                         e.g. ["spanish"] or ["spanish", "arabic"]
        speakers_filter: If set, keep only these specific speaker IDs.
        max_duration_s:  Discard utterances longer than this (seconds).
        min_duration_s:  Discard utterances shorter than this (seconds).

    Returns:
        DataFrame with columns: audio_path, sentence, speaker, l1, gender
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"L2-Arctic root not found: {dataset_dir}")

    # Normalise filter values to lowercase
    if l1_filter:
        l1_filter = [x.lower() for x in l1_filter]

    rows = []
    speakers_found = []

    for speaker_id, meta in SPEAKER_METADATA.items():
        # Apply filters
        if speakers_filter and speaker_id not in speakers_filter:
            continue
        if l1_filter and meta["l1"] not in l1_filter:
            continue

        speaker_dir = dataset_dir / speaker_id
        if not speaker_dir.exists():
            logger.warning(f"Speaker directory not found, skipping: {speaker_dir}")
            continue

        wav_dir        = speaker_dir / "wav"
        transcript_dir = speaker_dir / "transcript"

        if not wav_dir.exists() or not transcript_dir.exists():
            logger.warning(f"Missing wav/ or transcript/ for {speaker_id}, skipping.")
            continue

        speakers_found.append(speaker_id)
        matched = 0

        for wav_file in sorted(wav_dir.glob("*.wav")):
            txt_file = transcript_dir / (wav_file.stem + ".txt")
            if not txt_file.exists():
                logger.debug(f"No transcript for {wav_file.name}, skipping.")
                continue

            transcript = txt_file.read_text(encoding="utf-8").strip()
            if not transcript:
                continue

            rows.append({
                "audio_path": str(wav_file.resolve()),
                "sentence":   transcript,
                "speaker":    speaker_id,
                "l1":         meta["l1"],
                "gender":     meta["gender"],
            })
            matched += 1

        logger.info(f"  {speaker_id} ({meta['l1']}, {meta['gender']}): {matched} utterances")

    if not rows:
        raise ValueError(
            "No utterances found. Check dataset_dir path and filter settings."
        )

    df = pd.DataFrame(rows)
    logger.info(
        f"\nBuilt manifest: {len(df)} utterances | "
        f"{len(speakers_found)} speakers | "
        f"L1s: {df['l1'].unique().tolist()}"
    )
    return df


def split_manifest(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
    test_ratio:  float = 0.10,
    seed: int = 42,
    speaker_stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the manifest into train / val / test sets.

    With speaker_stratify=True (recommended), each speaker's utterances
    are split independently, so all speakers appear in all three sets.
    This prevents the model from overfitting to a specific speaker's voice.

    Args:
        df:                Full manifest from build_manifest().
        train_ratio:       Fraction for train.
        val_ratio:         Fraction for validation.
        test_ratio:        Fraction for test.
        seed:              Random seed.
        speaker_stratify:  Split within each speaker.

    Returns:
        (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    train_parts, val_parts, test_parts = [], [], []

    groups = df.groupby("speaker") if speaker_stratify else [(None, df)]

    for _, group in groups:
        shuffled = group.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train_parts.append(shuffled.iloc[:n_train])
        val_parts.append(  shuffled.iloc[n_train : n_train + n_val])
        test_parts.append( shuffled.iloc[n_train + n_val :])

    train = pd.concat(train_parts).reset_index(drop=True)
    val   = pd.concat(val_parts).reset_index(drop=True)
    test  = pd.concat(test_parts).reset_index(drop=True)

    logger.info(
        f"Split: train={len(train)} | val={len(val)} | test={len(test)}"
    )
    return train, val, test


def save_splits(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save train/val/test splits to CSV files in output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save only the columns Whisper fine-tuning needs
    cols = ["audio_path", "sentence", "speaker", "l1", "gender"]

    train[cols].to_csv(output_dir / "train.csv", index=False)
    val[cols].to_csv(  output_dir / "val.csv",   index=False)
    test[cols].to_csv( output_dir / "test.csv",  index=False)

    logger.info(f"Splits saved to: {output_dir}")
    logger.info(f"  train.csv : {len(train)} rows")
    logger.info(f"  val.csv   : {len(val)} rows")
    logger.info(f"  test.csv  : {len(test)} rows")


def print_dataset_stats(df: pd.DataFrame) -> None:
    """Print a summary of the manifest."""
    print("\n" + "="*55)
    print("  L2-Arctic manifest statistics")
    print("="*55)
    print(f"  Total utterances : {len(df):,}")
    print(f"  Unique speakers  : {df['speaker'].nunique()}")
    print()
    print("  Utterances per L1:")
    for l1, count in df.groupby("l1").size().items():
        speakers = df[df["l1"] == l1]["speaker"].unique().tolist()
        print(f"    {l1:<12} {count:>5}  ({', '.join(speakers)})")
    print()
    print("  Gender distribution:")
    for g, count in df.groupby("gender").size().items():
        label = "Male" if g == "M" else "Female"
        print(f"    {label:<10} {count:>5}")
    print("="*55)


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Build train/val/test splits from L2-Arctic for Whisper fine-tuning"
    )
    parser.add_argument(
        "--dataset_dir", "-d", required=True,
        help="Root directory of L2-Arctic corpus"
    )
    parser.add_argument(
        "--output_dir", "-o", default="data/splits/",
        help="Where to write train.csv, val.csv, test.csv"
    )
    parser.add_argument(
        "--l1", nargs="+", default=None,
        help="Filter by L1 language(s). Options: arabic hindi korean mandarin spanish vietnamese. "
             "Default: all. Example: --l1 spanish arabic"
    )
    parser.add_argument(
        "--speakers", nargs="+", default=None,
        help="Filter by specific speaker IDs. Example: --speakers EBVS ERMS NJS MBMPS"
    )
    parser.add_argument(
        "--train", type=float, default=0.80
    )
    parser.add_argument(
        "--val", type=float, default=0.10
    )
    parser.add_argument(
        "--test", type=float, default=0.10
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--stats_only", action="store_true",
        help="Print dataset stats and exit without saving splits"
    )
    args = parser.parse_args()

    df = build_manifest(
        dataset_dir=args.dataset_dir,
        l1_filter=args.l1,
        speakers_filter=args.speakers,
    )

    print_dataset_stats(df)

    if args.stats_only:
        print("\nStats-only mode. No splits saved.")
    else:
        train, val, test = split_manifest(
            df,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed,
        )
        save_splits(train, val, test, args.output_dir)
        print(f"\nDone. Splits written to: {args.output_dir}")
