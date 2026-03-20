#!/usr/bin/env python3
# scripts/stage2_transcribe.py
import argparse, csv, json, logging, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
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

def run_stage2(manifest_path, output_dir, model_name="small", sample=50, language="en"):
    import whisper, pandas as pd
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    logger.info(f"Manifest: {len(df)} files, columns: {df.columns.tolist()}")
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled {len(df)} files")

    logger.info(f"Loading Whisper {model_name}...")
    model = whisper.load_model(model_name)
    logger.info("Model loaded. Transcribing...")

    results, wer_scores, speaker_wers = [], [], {}

    for _, row in df.iterrows():
        audio_path = Path(str(row["path"]))
        filename = str(row["filename"])
        speaker = str(row.get("speaker", "unknown"))

        if not audio_path.exists():
            logger.warning(f"Not found: {audio_path}")
            continue

        reference = get_transcript(audio_path)
        if reference is None:
            logger.warning(f"No transcript: {filename}")
            continue

        t = time.time()
        try:
            result = model.transcribe(str(audio_path), language=language, fp16=False)
            hypothesis = result["text"].strip()
            elapsed = time.time() - t
        except Exception as e:
            logger.error(f"Failed {filename}: {e}")
            continue

        wer = compute_wer(reference, hypothesis)
        wer_scores.append(wer)
        speaker_wers.setdefault(speaker, []).append(wer)
        results.append({"filename": filename, "speaker": speaker, "reference": reference, "hypothesis": hypothesis, "wer": round(wer,4), "time_s": round(elapsed,2)})

        if len(results) % 10 == 0 or len(results) == 1:
            logger.info(f"[{len(results)}/{len(df)}] avg WER: {sum(wer_scores)/len(wer_scores):.1%} | {filename} WER={wer:.1%} ({elapsed:.1f}s)")

    if not wer_scores:
        logger.error("No files transcribed.")
        return {}

    overall_wer = sum(wer_scores) / len(wer_scores)
    per_speaker = {s: {"count": len(v), "avg_wer": round(sum(v)/len(v),4)} for s,v in speaker_wers.items()}

    out_csv = output_dir / "transcriptions.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)

    report = {"stage": "stage2", "model": f"whisper-{model_name}", "files_transcribed": len(results), "overall_wer": round(overall_wer,4), "overall_wer_pct": f"{overall_wer:.1%}", "per_speaker": per_speaker, "samples": results[:5]}
    with open(output_dir / "stage2_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*55}\n  Stage 2 complete\n{'='*55}")
    print(f"  Files: {len(results)}  |  Overall WER: {overall_wer:.1%}")
    for s,d in per_speaker.items(): print(f"    {s:<8}: {d['avg_wer']:.1%} ({d['count']} files)")
    print(f"\n  Saved: {out_csv}")
    return report

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--output_dir", default="outputs/stage2")
    p.add_argument("--model", default="small")
    p.add_argument("--sample", type=int, default=50)
    p.add_argument("--language", default="en")
    args = p.parse_args()
    run_stage2(args.manifest, args.output_dir, args.model, args.sample, args.language)
