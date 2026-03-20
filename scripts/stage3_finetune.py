#!/usr/bin/env python3
# scripts/stage3_finetune.py
"""
Stage 3: Fine-tune Whisper on L2-Arctic Spanish speaker data.

Two modes:
  --quick    : 2 epochs, 100 samples — verifies everything works (~20 min CPU)
  (default)  : full training on all 4,402 files — run overnight on CPU

Usage:
    # Test first (always do this first)
    python scripts/stage3_finetune.py \
        --manifest outputs/stage1/clean_manifest.csv \
        --output_dir outputs/stage3 \
        --quick

    # Full training (run overnight)
    python scripts/stage3_finetune.py \
        --manifest outputs/stage1/clean_manifest.csv \
        --output_dir outputs/stage3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_transcript(audio_path: Path) -> str | None:
    """Get reference transcript from sibling transcript/ folder."""
    txt = audio_path.parent.parent / "transcript" / (audio_path.stem + ".txt")
    return txt.read_text(encoding="utf-8").strip() if txt.exists() else None


def build_dataset(manifest_path: Path, sample: int | None = None):
    """
    Load manifest CSV and build a HuggingFace Dataset with
    audio arrays and reference transcripts.
    """
    import pandas as pd
    from datasets import Dataset, Audio

    df = pd.read_csv(manifest_path)
    logger.info(f"Manifest loaded: {len(df)} files")

    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)
        logger.info(f"Sampled {len(df)} files for quick mode")

    # Filter to files that have transcripts
    rows = []
    for _, row in df.iterrows():
        audio_path = Path(str(row["path"]))
        if not audio_path.exists():
            continue
        transcript = get_transcript(audio_path)
        if transcript is None:
            continue
        rows.append({
            "audio": str(audio_path),
            "sentence": transcript,
            "speaker": str(row.get("speaker", "unknown")),
        })

    logger.info(f"Files with transcripts: {len(rows)}")

    # Create HuggingFace dataset and cast audio column
    # This handles the 44.1kHz -> 16kHz resampling automatically
    dataset = Dataset.from_list(rows)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset


def prepare_features(batch, processor):
    """Extract features and tokenize labels for one batch."""
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="np",
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch


from dataclasses import dataclass
from typing import Any


@dataclass
class DataCollator:
    """Pad inputs and labels for Whisper fine-tuning."""
    processor: Any

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def compute_wer_simple(ref, hyp):
    """Simple WER without external dependencies."""
    r, h = ref.lower().split(), hyp.lower().split()
    if not r:
        return 0.0 if not h else 1.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1): d[i][0] = i
    for j in range(len(h) + 1): d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = d[i-1][j-1] if r[i-1] == h[j-1] else 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return d[len(r)][len(h)] / len(r)


def run_stage3(
    manifest_path: str,
    output_dir: str,
    model_name: str = "openai/whisper-small",
    quick: bool = False,
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    warmup_steps: int = 200,
    language: str = "en",
):
    import torch
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    manifest_path = Path(manifest_path)
    output_dir    = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    if quick:
        logger.info("QUICK MODE: 100 samples, 2 epochs — for testing only")
        num_epochs  = 2
        warmup_steps = 10
        sample = 100
    else:
        logger.info("FULL TRAINING MODE — capped at 500 samples for 7.7GB RAM")
        sample = 500

    # ── Load processor and model ──────────────────────────────────────────────
    logger.info(f"Loading {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.set_prefix_tokens(language=language, task="transcribe")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # CPU: must use float32
    )
    model.config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model.config.use_cache          = False  # required for gradient checkpointing

    # ── Build dataset ─────────────────────────────────────────────────────────
    full_dataset = build_dataset(manifest_path, sample=sample)

    # Train/val split (90/10)
    split = full_dataset.train_test_split(test_size=0.10, seed=42)
    train_dataset = split["train"]
    val_dataset   = split["test"]

    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Feature extraction
    logger.info("Extracting features (this may take a few minutes)...")
    train_dataset = train_dataset.map(
        lambda b: prepare_features(b, processor),
        remove_columns=train_dataset.column_names,
        desc="Train features",
        writer_batch_size=50,
    )
    val_dataset = val_dataset.map(
        lambda b: prepare_features(b, processor),
        remove_columns=val_dataset.column_names,
        desc="Val features",
        writer_batch_size=50,
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer_scores = [compute_wer_simple(r, h) for r, h in zip(label_str, pred_str)]
        return {"wer": round(sum(wer_scores) / len(wer_scores), 4)}

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,   # effective batch = 4 * batch_size
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        gradient_checkpointing=True,
        fp16=False,                      # CPU: no fp16
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=10,
        save_total_limit=2,              # keep only 2 checkpoints to save disk space
        report_to=[],                    # disable wandb/tensorboard for simplicity
        use_cpu=device == "cpu",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor.feature_extractor,
        data_collator=DataCollator(processor=processor),
        compute_metrics=compute_metrics,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting fine-tuning...")
    train_result = trainer.train()

    # ── Save final model ──────────────────────────────────────────────────────
    model_output = output_dir / "whisper-small-l2arctic"
    trainer.save_model(str(model_output))
    processor.save_pretrained(str(model_output))
    logger.info(f"Model saved to: {model_output}")

    # ── Save training summary ─────────────────────────────────────────────────
    summary = {
        "stage":        "stage3_finetuning",
        "base_model":   model_name,
        "saved_to":     str(model_output),
        "language":     language,
        "train_samples": len(train_dataset),
        "val_samples":  len(val_dataset),
        "num_epochs":   num_epochs,
        "batch_size":   batch_size,
        "learning_rate": learning_rate,
        "quick_mode":   quick,
        "train_loss":   round(train_result.training_loss, 4),
        "next_step":    "Run stage2_transcribe.py with --model outputs/stage3/whisper-small-l2arctic",
    }
    with open(output_dir / "stage3_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*55)
    print("  Stage 3 complete — Fine-tuning done")
    print("="*55)
    print(f"  Base model   : {model_name}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Epochs       : {num_epochs}")
    print(f"  Train loss   : {train_result.training_loss:.4f}")
    print(f"  Saved to     : {model_output}")
    print()
    print("  Next: run stage2 with fine-tuned model:")
    print(f"  python scripts/stage2_transcribe.py \\")
    print(f"      --manifest outputs/stage1/clean_manifest.csv \\")
    print(f"      --dataset_dir \"D:\\folder 2\\l2arctic_release_v5.0\" \\")
    print(f"      --output_dir outputs/stage3_eval \\")
    print(f"      --model {model_output} \\")
    print(f"      --sample 50")
    print("="*55)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 3: Fine-tune Whisper on L2-Arctic Spanish speakers"
    )
    parser.add_argument("--manifest",      required=True,
                        help="outputs/stage1/clean_manifest.csv")
    parser.add_argument("--output_dir",    default="outputs/stage3")
    parser.add_argument("--model",         default="openai/whisper-small")
    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--lr",            type=float, default=1e-5)
    parser.add_argument("--warmup",        type=int,   default=200)
    parser.add_argument("--language",      default="en")
    parser.add_argument("--quick",         action="store_true",
                        help="Quick test: 100 samples, 2 epochs (~20 min on CPU)")
    args = parser.parse_args()

    run_stage3(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        model_name=args.model,
        quick=args.quick,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        language=args.language,
    )



