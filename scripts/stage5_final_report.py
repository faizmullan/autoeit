#!/usr/bin/env python3
# scripts/stage5_final_report.py
"""
Stage 5: Generate final evaluation report.

Combines baseline and fine-tuned results into a
publication-ready summary report.

Usage:
    python scripts/stage5_final_report.py \
        --baseline_csv   outputs/stage2/transcriptions.csv \
        --finetuned_csv  outputs/stage3_eval/finetuned_transcriptions.csv \
        --output_dir     outputs/stage5
"""

import argparse, json, csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


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


def human_agreement(ref, hyp, tolerance="exact"):
    ref_n = " ".join(ref.lower().split())
    hyp_n = " ".join(hyp.lower().split())
    if tolerance == "exact":
        return ref_n == hyp_n
    elif tolerance == "wer_threshold":
        return compute_wer(ref, hyp) < 0.10


def load_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def analyze(rows, label):
    wers, cers, agreements = [], [], []
    speaker_data = {}

    for row in rows:
        ref = row.get("reference", "")
        hyp = row.get("hypothesis", "")
        spk = row.get("speaker", "unknown")
        if not ref or not hyp:
            continue

        wer = compute_wer(ref, hyp)
        cer = compute_cer(ref, hyp)
        agr = human_agreement(ref, hyp, tolerance="wer_threshold")

        wers.append(wer)
        cers.append(cer)
        agreements.append(agr)

        if spk not in speaker_data:
            speaker_data[spk] = {"wers": [], "agreements": []}
        speaker_data[spk]["wers"].append(wer)
        speaker_data[spk]["agreements"].append(agr)

    n = len(wers)
    overall_wer = sum(wers)/n if n else 0
    overall_cer = sum(cers)/n if n else 0
    overall_agr = sum(agreements)/n if n else 0

    per_speaker = {}
    for spk, data in speaker_data.items():
        sw = data["wers"]
        sa = data["agreements"]
        per_speaker[spk] = {
            "count":          len(sw),
            "wer":            round(sum(sw)/len(sw), 4),
            "agreement_rate": round(sum(sa)/len(sa), 4),
        }

    return {
        "label":            label,
        "files_evaluated":  n,
        "overall_wer":      round(overall_wer, 4),
        "overall_wer_pct":  f"{overall_wer:.1%}",
        "overall_cer":      round(overall_cer, 4),
        "overall_cer_pct":  f"{overall_cer:.1%}",
        "agreement_rate":   round(overall_agr, 4),
        "agreement_pct":    f"{overall_agr:.1%}",
        "goal_met":         overall_agr >= 0.90,
        "per_speaker":      per_speaker,
    }


def run_stage5(baseline_csv, finetuned_csv, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows  = load_csv(baseline_csv)
    finetuned_rows = load_csv(finetuned_csv)

    baseline  = analyze(baseline_rows,  "Whisper-small (baseline, no fine-tuning)")
    finetuned = analyze(finetuned_rows, "Whisper-tiny (fine-tuned on L2-Arctic)")

    wer_improvement = (baseline["overall_wer"] - finetuned["overall_wer"]) * 100
    agr_improvement = (finetuned["agreement_rate"] - baseline["agreement_rate"]) * 100

    report = {
        "project":     "AutoEIT — GSoC 2026",
        "description": "Audio-to-text transcription for L2 Spanish EIT learner data",
        "dataset":     "L2-Arctic v5.0 (Spanish L1 speakers: MBMPS, NJS, ERMS, EBVS)",
        "baseline":    baseline,
        "finetuned":   finetuned,
        "summary": {
            "wer_baseline":       baseline["overall_wer_pct"],
            "wer_finetuned":      finetuned["overall_wer_pct"],
            "wer_improvement":    f"{wer_improvement:.1f}% absolute reduction",
            "agreement_baseline": baseline["agreement_pct"],
            "agreement_finetuned": finetuned["agreement_pct"],
            "agreement_improvement": f"{agr_improvement:.1f}% absolute improvement",
            "gsoc_goal":          "≥90% human agreement",
            "goal_met":           finetuned["goal_met"],
            "next_steps": [
                "Run full fine-tuning (500 samples, 10 epochs) for further WER reduction",
                "Implement Stage 4 post-processing (filler removal, L1 lexicon correction)",
                "Target: <10% WER and ≥90% human agreement after full pipeline",
            ]
        }
    }

    report_path = output_dir / "final_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print report
    print("\n" + "="*60)
    print("  AutoEIT — Final Evaluation Report")
    print("  GSoC 2026 | HumanAI Foundation")
    print("="*60)
    print(f"\n  Dataset   : L2-Arctic v5.0 (Spanish L1 speakers)")
    print(f"  Files     : {finetuned['files_evaluated']} utterances evaluated")
    print()
    print(f"  {'Metric':<25} {'Baseline':>12} {'Fine-tuned':>12} {'Change':>10}")
    print(f"  {'-'*60}")
    print(f"  {'WER':<25} {baseline['overall_wer_pct']:>12} {finetuned['overall_wer_pct']:>12} {'-'+str(round(wer_improvement,1))+'%':>10}")
    print(f"  {'CER':<25} {baseline['overall_cer_pct']:>12} {finetuned['overall_cer_pct']:>12}")
    print(f"  {'Human agreement':<25} {baseline['agreement_pct']:>12} {finetuned['agreement_pct']:>12} {'+'+str(round(agr_improvement,1))+'%':>10}")
    print()
    print(f"  Per-speaker WER (fine-tuned):")
    for spk, data in finetuned["per_speaker"].items():
        print(f"    {spk:<8} : {data['wer']:.1%}  (agreement: {data['agreement_rate']:.1%})")
    print()
    print(f"  GSoC goal (≥90% agreement): {'ACHIEVED' if finetuned['goal_met'] else 'IN PROGRESS'}")
    print(f"\n  Full report saved to: {report_path}")
    print("="*60)

    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Stage 5: Final evaluation report for AutoEIT"
    )
    p.add_argument("--baseline_csv",  required=True,
                   help="outputs/stage2/transcriptions.csv")
    p.add_argument("--finetuned_csv", required=True,
                   help="outputs/stage3_eval/finetuned_transcriptions.csv")
    p.add_argument("--output_dir",    default="outputs/stage5")
    args = p.parse_args()

    run_stage5(args.baseline_csv, args.finetuned_csv, args.output_dir)
