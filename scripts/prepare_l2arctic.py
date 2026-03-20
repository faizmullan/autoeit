#!/usr/bin/env python
# scripts/prepare_l2arctic.py
"""
One-command setup for L2-Arctic data.

Usage — all speakers:
    python scripts/prepare_l2arctic.py --dataset_dir /path/to/l2-arctic

Usage — Spanish L1 speakers only (most relevant for Spanish EIT):
    python scripts/prepare_l2arctic.py \
        --dataset_dir /path/to/l2-arctic \
        --l1 spanish

Usage — all speakers, just print stats:
    python scripts/prepare_l2arctic.py \
        --dataset_dir /path/to/l2-arctic \
        --stats_only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.l2arctic_loader import (
    build_manifest,
    print_dataset_stats,
    save_splits,
    split_manifest,
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", "-d", required=True,
                        help="Root directory of L2-Arctic corpus")
    parser.add_argument("--output_dir", "-o", default="data/splits/",
                        help="Where to write train/val/test CSVs")
    parser.add_argument("--l1", nargs="+", default=None,
                        help="Filter by L1. E.g.: --l1 spanish arabic")
    parser.add_argument("--speakers", nargs="+", default=None,
                        help="Filter by speaker IDs. E.g.: --speakers EBVS ERMS")
    parser.add_argument("--train", type=float, default=0.80)
    parser.add_argument("--val",   type=float, default=0.10)
    parser.add_argument("--test",  type=float, default=0.10)
    parser.add_argument("--seed",  type=int,   default=42)
    parser.add_argument("--stats_only", action="store_true",
                        help="Print stats only, don't write files")
    args = parser.parse_args()

    df = build_manifest(
        dataset_dir=args.dataset_dir,
        l1_filter=args.l1,
        speakers_filter=args.speakers,
    )
    print_dataset_stats(df)

    if args.stats_only:
        return

    train, val, test = split_manifest(
        df,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
        speaker_stratify=True,
    )
    save_splits(train, val, test, args.output_dir)
    print(f"\nAll done! Next step:\n"
          f"  python src/asr/finetune_l2arctic.py "
          f"--train_csv {args.output_dir}/train.csv "
          f"--val_csv {args.output_dir}/val.csv")


if __name__ == "__main__":
    main()
