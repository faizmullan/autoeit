# src/asr/finetune_l2arctic.py
"""
Fine-tune Whisper on L2-Arctic data.

Key difference from the generic fine_tuner.py:
- L2-Arctic audio is at 44.1kHz — must resample to 16kHz for Whisper
- Handles the HuggingFace datasets Audio cast for resampling automatically
- Supports filtering to Spanish L1 speakers only (most relevant for EIT)

Run with:
    python src/asr/finetune_l2arctic.py \
        --train_csv  data/splits/train.csv \
        --val_csv    data/splits/val.csv \
        --output_dir models/whisper-l2arctic \
        --model      openai/whisper-small \
        --epochs     10
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import evaluate
from datasets import Dataset, DatasetDict, Audio
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: any

    def __call__(self, features: list[dict]) -> dict:
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


def prepare_dataset(batch, processor):
    """Feature extraction + tokenization for a single sample."""
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="np",
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch


def load_splits_as_dataset(train_csv: str, val_csv: str) -> DatasetDict:
    """
    Load the L2-Arctic CSV splits into a HuggingFace DatasetDict.
    Casts the audio_path column to Audio(sampling_rate=16000) so that
    HuggingFace handles the 44.1kHz → 16kHz resampling automatically.
    """
    import pandas as pd

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    # Rename so HuggingFace datasets recognises the audio column
    train_df = train_df.rename(columns={"audio_path": "audio"})
    val_df   = val_df.rename(columns={"audio_path": "audio"})

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "eval":  Dataset.from_pandas(val_df),
    })

    # Cast audio column → HuggingFace loads + resamples to 16kHz on the fly
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    logger.info(
        f"Loaded dataset: {len(dataset['train'])} train | {len(dataset['eval'])} eval"
    )
    return dataset


def finetune_l2arctic(
    train_csv:    str = "data/splits/train.csv",
    val_csv:      str = "data/splits/val.csv",
    model_name:   str = "openai/whisper-small",
    output_dir:   str = "models/whisper-l2arctic",
    language:     str = "en",          # L2-Arctic is English
    num_epochs:   int = 10,
    batch_size:   int = 8,
    learning_rate: float = 1e-5,
    warmup_steps: int = 500,
    fp16:         bool = True,
    num_workers:  int = 4,
) -> None:
    """
    Fine-tune Whisper on L2-Arctic CSV splits.

    Args:
        train_csv:      Path to train.csv from l2arctic_loader.py
        val_csv:        Path to val.csv
        model_name:     Base Whisper checkpoint to start from
        output_dir:     Where to save fine-tuned checkpoints
        language:       "en" for L2-Arctic (English speech)
        num_epochs:     Training epochs
        batch_size:     Per-device batch size
        learning_rate:  AdamW LR
        warmup_steps:   LR warmup
        fp16:           Mixed precision (GPU only)
        num_workers:    Dataloader workers for feature extraction
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device} | Model: {model_name}")

    # ── Processor & model ─────────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.set_prefix_tokens(language=language, task="transcribe")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens    = []
    model.config.use_cache          = False   # required with gradient checkpointing

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = load_splits_as_dataset(train_csv, val_csv)
    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=dataset.column_names["train"],
        num_proc=num_workers,
        desc="Extracting features",
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(wer, 4)}

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 16 // batch_size),
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        gradient_checkpointing=True,
        fp16=fp16 and device == "cuda",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=50,
        report_to=["tensorboard"],
        push_to_hub=False,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting fine-tuning on L2-Arctic...")
    trainer.train()

    # Save final model and processor together
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    logger.info(f"Fine-tuned model saved to: {output_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper on L2-Arctic non-native English speech"
    )
    parser.add_argument("--train_csv",      default="data/splits/train.csv")
    parser.add_argument("--val_csv",        default="data/splits/val.csv")
    parser.add_argument("--model",          default="openai/whisper-small")
    parser.add_argument("--output_dir",     default="models/whisper-l2arctic")
    parser.add_argument("--language",       default="en")
    parser.add_argument("--epochs",  type=int,   default=10)
    parser.add_argument("--batch",   type=int,   default=8)
    parser.add_argument("--lr",      type=float, default=1e-5)
    parser.add_argument("--warmup",  type=int,   default=500)
    parser.add_argument("--no_fp16", action="store_true")
    args = parser.parse_args()

    finetune_l2arctic(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        model_name=args.model,
        output_dir=args.output_dir,
        language=args.language,
        num_epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        fp16=not args.no_fp16,
    )
