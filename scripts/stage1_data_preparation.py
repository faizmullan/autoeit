#!/usr/bin/env python3
# scripts/stage1_data_preparation.py
"""
Stage 1: Data preparation and VAD analysis.

Fixes the original crash:
    "cannot reshape array of size N into shape (512)"

The broken code did:
    frames = audio.reshape(-1, 512)   # crashes when len(audio) % 512 != 0

The fix uses librosa.effects.split() which handles any audio length.

Usage:
    python scripts/stage1_data_preparation.py \
        --dataset_dir /path/to/l2-arctic \
        --output_dir  outputs/stage1 \
        --speakers MBMPS ERMS EBVS NJS
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.vad import vad_analyze, filter_by_vad

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Spanish L1 speakers in L2-Arctic
SPANISH_SPEAKERS = ["MBMPS", "ERMS", "EBVS", "NJS"]


def run_stage1(
    dataset_dir: str,
    output_dir: str,
    speakers: list[str] | None = None,
    top_db: float = 30.0,
    min_speech_ratio: float = 0.3,
    min_snr_db: float = 5.0,
) -> dict:
    """
    Run Stage 1: scan L2-Arctic, run VAD, filter bad files, save manifest.

    Args:
        dataset_dir:       Root of L2-Arctic corpus.
        output_dir:        Where to write stage1_results.json and clean_manifest.csv.
        speakers:          Which speakers to include. Default: all Spanish L1.
        top_db:            Silence threshold for VAD.
        min_speech_ratio:  Minimum speech fraction to keep a file.
        min_snr_db:        Minimum SNR to keep a file.

    Returns:
        Results dict (also saved to output_dir/stage1_results.json).
    """
    import librosa

    dataset_dir = Path(dataset_dir)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_speakers = speakers or SPANISH_SPEAKERS

    # ── Collect all WAV files ─────────────────────────────────────────────────
    all_files = []
    speaker_counts = {}

    for speaker in target_speakers:
        # Handle both flat (dataset/SPEAKER/wav) and
        # nested (dataset/SPEAKER/SPEAKER/wav) L2-Arctic structures
        wav_dir = dataset_dir / speaker / "wav"
        if not wav_dir.exists():
            wav_dir = dataset_dir / speaker / speaker / "wav"
        if not wav_dir.exists():
            logger.warning(f"No wav/ dir for speaker {speaker}: {wav_dir}")
            continue
        wavs = sorted(wav_dir.glob("*.wav"))
        all_files.extend(wavs)
        speaker_counts[speaker] = len(wavs)
        logger.info(f"  {speaker}: {len(wavs)} files")

    logger.info(f"Total files to process: {len(all_files)}")

    # ── Run VAD on every file ─────────────────────────────────────────────────
    results = []
    for wav_path in all_files:
        try:
            audio, sr = librosa.load(wav_path, sr=16000, mono=True)
            stats = vad_analyze(audio, sr=sr, top_db=top_db)
            stats["path"]     = str(wav_path)
            stats["filename"] = wav_path.name
            stats["speaker"]  = wav_path.parent.parent.name
            stats["duration_s"] = stats["total_duration_s"]
            results.append(stats)
        except Exception as e:
            logger.error(f"Failed: {wav_path.name}: {e}")
            results.append({
                "path": str(wav_path),
                "filename": wav_path.name,
                "speaker": wav_path.parent.parent.name,
                "quality": "error",
                "issues": [str(e)],
            })

    # ── Filter ────────────────────────────────────────────────────────────────
    kept, rejected = filter_by_vad(
        results,
        min_speech_ratio=min_speech_ratio,
        min_snr_db=min_snr_db,
        exclude_clipping=False,   # clipping is rare in L2-Arctic
    )

    # ── Statistics ────────────────────────────────────────────────────────────
    quality_dist = {}
    for r in results:
        q = r.get("quality", "error")
        quality_dist[q] = quality_dist.get(q, 0) + 1

    durations = [r["total_duration_s"] for r in results if "total_duration_s" in r]
    speech_ds = [r["speech_duration_s"] for r in results if "speech_duration_s" in r]

    stats_summary = {
        "total_files":          len(all_files),
        "files_processed":      len(results),
        "files_kept":           len(kept),
        "files_rejected":       len(rejected),
        "quality_distribution": quality_dist,
        "speaker_breakdown":    speaker_counts,
        "duration_stats": {
            "total_hours":   round(sum(durations) / 3600, 4),
            "avg_duration_s": round(np.mean(durations), 3) if durations else 0,
            "min_duration_s": round(np.min(durations), 3)  if durations else 0,
            "max_duration_s": round(np.max(durations), 3)  if durations else 0,
        },
        "speech_stats": {
            "total_speech_hours":  round(sum(speech_ds) / 3600, 4),
            "avg_speech_ratio":    round(np.mean([r["speech_ratio"] for r in results if "speech_ratio" in r]), 4),
        },
    }

    # ── Save manifest of clean files ──────────────────────────────────────────
    import csv
    manifest_path = output_dir / "clean_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "path", "speaker",
            "duration_s", "speech_ratio", "snr_db", "quality"
        ])
        writer.writeheader()
        for r in kept:
            writer.writerow({
                "filename":     r.get("filename", ""),
                "path":         r.get("path", ""),
                "speaker":      r.get("speaker", ""),
                "duration_s":   r.get("total_duration_s", 0),
                "speech_ratio": r.get("speech_ratio", 0),
                "snr_db":       r.get("snr_db", 0),
                "quality":      r.get("quality", ""),
            })

    # ── Save full JSON report ─────────────────────────────────────────────────
    report = {
        "stage": "stage1_data_preparation",
        "status": "COMPLETE",
        "statistics": stats_summary,
        "sample_results": results[:20],     # first 20 for inspection
        "rejection_summary": [
            {"filename": r["filename"], "reasons": r.get("rejection_reasons", r.get("issues", []))}
            for r in rejected[:20]
        ],
    }

    report_path = output_dir / "stage1_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  Stage 1 complete")
    print("="*55)
    print(f"  Files found      : {stats_summary['total_files']:,}")
    print(f"  Files kept       : {stats_summary['files_kept']:,}")
    print(f"  Files rejected   : {stats_summary['files_rejected']:,}")
    print(f"  Total audio      : {stats_summary['duration_stats']['total_hours']:.2f} hours")
    print(f"  Total speech     : {stats_summary['speech_stats']['total_speech_hours']:.2f} hours")
    print(f"  Quality breakdown:")
    for q, count in sorted(quality_dist.items()):
        print(f"    {q:<12} : {count:>5}")
    print(f"\n  Clean manifest   : {manifest_path}")
    print(f"  Full report      : {report_path}")
    print("="*55)

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1: L2-Arctic VAD analysis (fixed reshape bug)"
    )
    parser.add_argument("--dataset_dir", "-d", required=True,
                        help="Root of L2-Arctic corpus")
    parser.add_argument("--output_dir",  "-o", default="outputs/stage1",
                        help="Output directory")
    parser.add_argument("--speakers", nargs="+", default=None,
                        help=f"Speaker IDs to include (default: {SPANISH_SPEAKERS})")
    parser.add_argument("--top_db",    type=float, default=30.0,
                        help="Silence threshold for VAD (default: 30)")
    parser.add_argument("--min_speech",type=float, default=0.30,
                        help="Min speech ratio to keep file (default: 0.30)")
    parser.add_argument("--min_snr",   type=float, default=5.0,
                        help="Min SNR dB to keep file (default: 5.0)")
    args = parser.parse_args()

    run_stage1(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        speakers=args.speakers,
        top_db=args.top_db,
        min_speech_ratio=args.min_speech,
        min_snr_db=args.min_snr,
    )
