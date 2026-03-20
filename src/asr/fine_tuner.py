# src/asr/fine_tuner.py
"""
Fine-tune Whisper on EIT learner speech data.

This script is used once you have a labelled dataset of
(audio, transcript) pairs from human transcribers.

Run with:
    python src/asr/fine_tuner.py --config configs/default.yaml
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import DatasetDict, load_dataset, Audio
import evaluate

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Pad inputs and labels for Whisper fine-tuning.
    Labels are padded with -100 so they are ignored in the loss computation.
    """
    processor: any

    def __call__(self, features: list[dict]) -> dict:
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        # Replace padding token id with -100
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Remove decoder_start_token_id if it's at position 0
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset(batch, processor):
    """
    Feature extraction + tokenization for a single dataset example.
    Expects columns: "audio" (dict with "array" and "sampling_rate") and "sentence".
    """
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="np",
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch


def fine_tune(
    model_name: str = "openai/whisper-small",
    train_manifest: str = "data/splits/train.csv",
    val_manifest: str = "data/splits/val.csv",
    output_dir: str = "models/whisper-eit-finetuned",
    language: str = "es",
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 8,
    learning_rate: float = 1e-5,
    warmup_steps: int = 500,
    fp16: bool = True,
) -> None:
    """
    Fine-tune a Whisper model on EIT transcription data.

    The train/val CSV files must have columns:
        - audio_path: path to the preprocessed WAV file
        - sentence:   human reference transcription

    Args:
        model_name:                  Base Whisper model to fine-tune from.
        train_manifest:              Path to train split CSV.
        val_manifest:                Path to validation split CSV.
        output_dir:                  Where to save checkpoints.
        language:                    Language code for the forced decoder prompt.
        num_train_epochs:            Training epochs.
        per_device_train_batch_size: Batch size per GPU.
        learning_rate:               AdamW learning rate.
        warmup_steps:                LR scheduler warmup.
        fp16:                        Mixed precision training (requires GPU).
    """
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.set_prefix_tokens(language=language, task="transcribe")

    # Load dataset from CSV manifests
    dataset = DatasetDict()
    dataset["train"] = load_dataset("csv", data_files=train_manifest, split="train")
    dataset["eval"] = load_dataset("csv", data_files=val_manifest, split="train")

    # Cast audio column so HuggingFace handles resampling
    dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))
    dataset = dataset.rename_column("audio_path", "audio")

    # Feature extraction
    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=dataset.column_names["train"],
        num_proc=4,
    )

    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # required for gradient checkpointing

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(wer, 4)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        gradient_checkpointing=True,
        fp16=fp16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
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

    logger.info("Starting fine-tuning...")
    trainer.train()
    logger.info(f"Fine-tuning complete. Model saved to: {output_dir}")
