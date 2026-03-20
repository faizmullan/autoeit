#!/usr/bin/env python3
# scripts/evaluate_finetuned.py
"""
Evaluate a HuggingFace fine-tuned Whisper model on L2-Arctic.
Compares WER against the baseline result.

Usage:
    python scripts/evaluate_finetuned.py \
        --manifest outputs/stage1/clean_manifest.csv \
        --model_path outputs/stage3/whisper-small-l2arctic \
        --output_dir outputs/stage3_eval \
        --sample 50
"""

import argparse, csv, json, logging, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def get_transcript(audio_path):
    txt = audio_path.parent.parent / "transcript" / (audio_path.stem + ".txt")
    return txt.read_text(encoding="utf-8").strip() if txt.exists() else None


def compute_wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    if not r: return 0.0 if not h else 1.0
    d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0]=i
    for j in range(len(h)+1): d[0][j]=j
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1):
            d[i][j] = d[i-1][j-1] if r[i-1]==h[j-1] else 1+min(d[i-1][j],d[i][j-1],d[i-1][j-1])
    return d[len(r)][len(h)] / len(r)


def run_eval(manifest_path, model_path, output_dir, sample=50, language="en"):
    import torch
    import pandas as pd
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    manifest_path = Path(manifest_path)
    model_path    = Path(model_path)
    output_dir    = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
    logger.info(f"Evaluating {len(df)} files with model: {model_path}")

    # Load HuggingFace fine-tuned model
    logger.info("Loading fine-tuned model...")
    processor = AutoProcessor.from_pretrained(str(model_path))
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(model_path), torch_dtype=torch.float32
    )
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float32,
        device="cpu",
    )
    logger.info("Model loaded. Transcribing...")

    results, wer_scores, speaker_wers = [], [], {}

    for _, row in df.iterrows():
        audio_path = Path(str(row["path"]))
        filename   = str(row["filename"])
        speaker    = str(row.get("speaker", "unknown"))

        if not audio_path.exists():
            continue

        reference = get_transcript(audio_path)
        if reference is None:
            continue

        t = time.time()
        try:
            result     = pipe(str(audio_path), generate_kwargs={"language": language, "task": "transcribe"})
            hypothesis = result["text"].strip()
            elapsed    = time.time() - t
        except Exception as e:
            logger.error(f"Failed {filename}: {e}")
            continue

        wer = compute_wer(reference, hypothesis)
        wer_scores.append(wer)
        speaker_wers.setdefault(speaker, []).append(wer)
        results.append({
            "filename": filename, "speaker": speaker,
            "reference": reference, "hypothesis": hypothesis,
            "wer": round(wer, 4), "time_s": round(elapsed, 2)
        })

        if len(results) % 10 == 0 or len(results) == 1:
            logger.info(f"[{len(results)}/{len(df)}] avg WER: {sum(wer_scores)/len(wer_scores):.1%} | {filename} WER={wer:.1%}")

    if not wer_scores:
        logger.error("No files evaluated.")
        return {}

    overall_wer = sum(wer_scores) / len(wer_scores)
    per_speaker = {s: {"count": len(v), "avg_wer": round(sum(v)/len(v), 4)} for s,v in speaker_wers.items()}

    # Save CSV
    out_csv = output_dir / "finetuned_transcriptions.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)

    report = {
        "model": str(model_path),
        "files_evaluated": len(results),
        "overall_wer": round(overall_wer, 4),
        "overall_wer_pct": f"{overall_wer:.1%}",
        "baseline_wer": "28.6%",
        "improvement": f"{28.6 - overall_wer*100:.1f}% absolute improvement",
        "per_speaker": per_speaker,
    }
    with open(output_dir / "finetuned_eval.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Fine-tuned model evaluation")
    print(f"{'='*55}")
    print(f"  Model          : {model_path.name}")
    print(f"  Files          : {len(results)}")
    print(f"  Baseline WER   : 28.6%  (whisper-small, no fine-tuning)")
    print(f"  Fine-tuned WER : {overall_wer:.1%}")
    print(f"  Improvement    : {28.6 - overall_wer*100:.1f}% absolute")
    print()
    print("  Per-speaker:")
    for s, d in per_speaker.items():
        print(f"    {s:<8}: {d['avg_wer']:.1%} ({d['count']} files)")
    print(f"\n  Saved to: {out_csv}")
    print(f"{'='*55}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",    required=True)
    p.add_argument("--model_path",  required=True)
    p.add_argument("--output_dir",  default="outputs/stage3_eval")
    p.add_argument("--sample",      type=int, default=50)
    p.add_argument("--language",    default="en")
    args = p.parse_args()
    run_eval(args.manifest, args.model_path, args.output_dir, args.sample, args.language)
