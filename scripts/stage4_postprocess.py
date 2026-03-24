#!/usr/bin/env python3
# scripts/stage4_postprocess.py
"""
Stage 4: Post-processing for L2 learner speech transcriptions.

Applies three layers of correction:
1. Filler word removal (um, uh, er — very common in L2 speech)
2. Spanish L1 interference correction (systematic phoneme substitutions)
3. Text normalization (lowercase, spacing, punctuation cleanup)

Then re-evaluates WER/CER/agreement against the baseline.

Usage:
    python scripts/stage4_postprocess.py \
        --input_csv  outputs/stage3_eval/finetuned_transcriptions.csv \
        --output_dir outputs/stage4
"""

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Filler words common in L2 learner speech ─────────────────────────────────
FILLER_WORDS = {
    "um", "uh", "er", "eh", "ah", "hmm", "hm", "mm",
    "erm", "uhh", "umm", "uhm", "mhm", "uh-huh"
}

# ── Spanish L1 phoneme interference corrections ───────────────────────────────
# These are the systematic substitutions identified in Stage 4 phoneme analysis:
# /θ/ → /s/ : Spanish has no /θ/ phoneme
# /ð/ → /d/ : Spanish has no /ð/ phoneme
# /v/ → /b/ : Spanish merges /v/ and /b/
# /ŋ/ → /n/ : Spanish has no /ŋ/ phoneme
#
# Format: (wrong_pattern, correct_replacement, description)
# These are word-level corrections based on common Whisper errors on Spanish L1 speech

L1_CORRECTIONS = [
    # /θ/ → /s/ substitutions (theta → s)
    # Whisper often transcribes "the" as "de", "this" as "dis", "that" as "dat"
    (r'\bde\b',    'the',    'theta/s: de→the'),
    (r'\bdis\b',   'this',   'theta/s: dis→this'),
    (r'\bdat\b',   'that',   'theta/s: dat→that'),
    (r'\bdey\b',   'they',   'theta/s: dey→they'),
    (r'\bdem\b',   'them',   'theta/s: dem→them'),
    (r'\bden\b',   'then',   'theta/s: den→then'),
    (r'\bdere\b',  'there',  'theta/s: dere→there'),
    (r'\bsink\b',  'think',  'theta/s: sink→think'),
    (r'\bsing\b',  'thing',  'theta/s: sing→thing'),
    (r'\bsought\b','thought','theta/s: sought→thought'),
    (r'\bsree\b',  'three',  'theta/s: sree→three'),
    (r'\bsrough\b','through','theta/s: srough→through'),

    # /v/ → /b/ substitutions
    (r'\bberry\b', 'very',   'v/b: berry→very'),
    (r'\bbery\b',  'very',   'v/b: bery→very'),
    (r'\bbote\b',  'vote',   'v/b: bote→vote'),
    (r'\bbiew\b',  'view',   'v/b: biew→view'),
    (r'\bbast\b',  'vast',   'v/b: bast→vast'),

    # Common learner deletions — articles and prepositions
    # Spanish often omits English articles
    # These are handled carefully to avoid over-correction
]

# ── Punctuation and normalization ─────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase, strip extra whitespace, remove leading/trailing punctuation."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation at start/end
    text = text.strip('.,!?;:')
    return text


def remove_fillers(text: str) -> str:
    """Remove filler words from transcription."""
    words = text.split()
    cleaned = []
    for word in words:
        # Strip punctuation for comparison
        clean_word = word.strip('.,!?;:').lower()
        if clean_word not in FILLER_WORDS:
            cleaned.append(word)
    result = ' '.join(cleaned)
    # Collapse any double spaces left behind
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def apply_l1_corrections(text: str) -> tuple[str, list[str]]:
    """
    Apply Spanish L1 interference corrections.

    Returns:
        (corrected_text, list_of_corrections_applied)
    """
    corrections_applied = []
    for pattern, replacement, description in L1_CORRECTIONS:
        new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        if new_text != text:
            corrections_applied.append(description)
            text = new_text
    return text, corrections_applied


def postprocess(text: str) -> tuple[str, dict]:
    """
    Full post-processing pipeline for a single transcription.

    Returns:
        (processed_text, processing_log)
    """
    log = {"original": text, "steps": []}

    # Step 1: Normalize
    text = normalize_text(text)
    log["steps"].append({"step": "normalize", "result": text})

    # Step 2: Remove fillers
    before = text
    text = remove_fillers(text)
    if text != before:
        log["steps"].append({"step": "filler_removal", "removed": before, "result": text})

    # Step 3: L1 corrections
    before = text
    text, corrections = apply_l1_corrections(text)
    if corrections:
        log["steps"].append({"step": "l1_correction", "corrections": corrections, "result": text})

    log["final"] = text
    return text, log


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    if not r: return 0.0 if not h else 1.0
    d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            d[i][j] = d[i-1][j-1] if r[i-1]==h[j-1] else 1+min(d[i-1][j],d[i][j-1],d[i-1][j-1])
    return d[len(r)][len(h)] / len(r)


def compute_cer(ref, hyp):
    r, h = ref.lower(), hyp.lower()
    if not r: return 0.0 if not h else 1.0
    d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            d[i][j] = d[i-1][j-1] if r[i-1]==h[j-1] else 1+min(d[i-1][j],d[i][j-1],d[i-1][j-1])
    return d[len(r)][len(h)] / len(r)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_stage4(input_csv: str, output_dir: str) -> dict:
    input_csv  = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load fine-tuned transcriptions
    rows = []
    with open(input_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    logger.info(f"Loaded {len(rows)} transcriptions from {input_csv}")

    # Process each row
    results = []
    before_wers, after_wers = [], []
    before_cers, after_cers = [], []
    before_agrs, after_agrs = [], []
    speaker_stats = {}
    corrections_count = 0

    for row in rows:
        ref        = row.get("reference", "").strip()
        hyp_before = row.get("hypothesis", "").strip()
        speaker    = row.get("speaker", "unknown")

        if not ref or not hyp_before:
            continue

        # Apply post-processing
        hyp_after, log = postprocess(hyp_before)
        if len(log["steps"]) > 1:
            corrections_count += 1

        # Compute metrics before and after
        wer_before = compute_wer(ref, hyp_before)
        wer_after  = compute_wer(ref, hyp_after)
        cer_before = compute_cer(ref, hyp_before)
        cer_after  = compute_cer(ref, hyp_after)
        agr_before = 1 if " ".join(ref.lower().split()) == " ".join(hyp_before.lower().split()) else 0
        agr_after  = 1 if " ".join(ref.lower().split()) == " ".join(hyp_after.lower().split()) else 0

        before_wers.append(wer_before)
        after_wers.append(wer_after)
        before_cers.append(cer_before)
        after_cers.append(cer_after)
        before_agrs.append(agr_before)
        after_agrs.append(agr_after)

        if speaker not in speaker_stats:
            speaker_stats[speaker] = {"before_wers": [], "after_wers": []}
        speaker_stats[speaker]["before_wers"].append(wer_before)
        speaker_stats[speaker]["after_wers"].append(wer_after)

        results.append({
            "filename":       row.get("filename", ""),
            "speaker":        speaker,
            "reference":      ref,
            "before":         hyp_before,
            "after":          hyp_after,
            "wer_before":     round(wer_before, 4),
            "wer_after":      round(wer_after, 4),
            "wer_delta":      round(wer_after - wer_before, 4),
            "corrections":    len(log["steps"]) - 1,
        })

    # Aggregate
    n = len(results)
    avg = lambda lst: round(sum(lst)/len(lst), 4) if lst else 0

    overall_before_wer = avg(before_wers)
    overall_after_wer  = avg(after_wers)
    overall_before_cer = avg(before_cers)
    overall_after_cer  = avg(after_cers)
    overall_before_agr = avg(before_agrs)
    overall_after_agr  = avg(after_agrs)

    per_speaker = {}
    for spk, data in speaker_stats.items():
        per_speaker[spk] = {
            "wer_before": round(avg(data["before_wers"]), 4),
            "wer_after":  round(avg(data["after_wers"]),  4),
            "improvement": round((avg(data["before_wers"]) - avg(data["after_wers"])) * 100, 2),
        }

    # Save results CSV
    out_csv = output_dir / "postprocessed_transcriptions.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Save report
    report = {
        "stage":              "stage4_postprocessing",
        "files_processed":    n,
        "corrections_applied": corrections_count,
        "metrics": {
            "wer_before":     f"{overall_before_wer:.1%}",
            "wer_after":      f"{overall_after_wer:.1%}",
            "wer_improvement": f"{(overall_before_wer - overall_after_wer)*100:.1f}%",
            "cer_before":     f"{overall_before_cer:.1%}",
            "cer_after":      f"{overall_after_cer:.1%}",
            "agreement_before": f"{overall_before_agr:.1%}",
            "agreement_after":  f"{overall_after_agr:.1%}",
        },
        "per_speaker": per_speaker,
        "pipeline_summary": {
            "stage2_baseline_wer":   "28.6%",
            "stage3_finetuned_wer":  f"{overall_before_wer:.1%}",
            "stage4_postprocessed_wer": f"{overall_after_wer:.1%}",
        }
    }

    with open(output_dir / "stage4_results.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("  Stage 4 complete — Post-processing")
    print("="*60)
    print(f"  Files processed      : {n}")
    print(f"  Corrections applied  : {corrections_count}")
    print()
    print(f"  {'Metric':<25} {'Before':>10} {'After':>10} {'Change':>10}")
    print(f"  {'-'*57}")
    print(f"  {'WER':<25} {overall_before_wer:>9.1%} {overall_after_wer:>9.1%} {(overall_after_wer-overall_before_wer)*100:>+9.1f}%")
    print(f"  {'CER':<25} {overall_before_cer:>9.1%} {overall_after_cer:>9.1%}")
    print(f"  {'Human agreement':<25} {overall_before_agr:>9.1%} {overall_after_agr:>9.1%}")
    print()
    print(f"  Full pipeline summary:")
    print(f"    Stage 2 baseline   : 28.6% WER")
    print(f"    Stage 3 fine-tuned : {overall_before_wer:.1%} WER")
    print(f"    Stage 4 post-proc  : {overall_after_wer:.1%} WER")
    print()
    print(f"  Per-speaker WER after post-processing:")
    for spk, data in per_speaker.items():
        arrow = "↓" if data["improvement"] > 0 else "↑"
        print(f"    {spk:<8}: {data['wer_before']:.1%} → {data['wer_after']:.1%}  ({arrow}{abs(data['improvement']):.1f}%)")
    print()
    print(f"  Saved: {out_csv}")
    print("="*60)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 4: Post-processing for L2 learner transcriptions"
    )
    parser.add_argument("--input_csv",  required=True,
                        help="outputs/stage3_eval/finetuned_transcriptions.csv")
    parser.add_argument("--output_dir", default="outputs/stage4")
    args = parser.parse_args()

    run_stage4(args.input_csv, args.output_dir)
