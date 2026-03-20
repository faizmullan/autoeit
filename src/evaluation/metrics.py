# src/evaluation/metrics.py
"""
Evaluation metrics for AutoEIT transcription quality.

Primary metric: human agreement rate (% of transcriptions that match
a human transcriber within acceptable tolerance).

Secondary metrics: WER (Word Error Rate), CER (Character Error Rate).

The GSoC goal is ≥90% agreement with human transcribers.
"""

import json
import logging
from pathlib import Path

import pandas as pd
from jiwer import compute_measures, cer

logger = logging.getLogger(__name__)


def compute_wer(predictions: list[str], references: list[str]) -> dict:
    """
    Compute Word Error Rate and its components.

    WER = (Substitutions + Deletions + Insertions) / Total reference words

    Args:
        predictions: List of ASR output strings.
        references:  List of human reference strings (same order).

    Returns:
        Dict with keys: wer, mer, wil, hits, substitutions, deletions, insertions
    """
    measures = compute_measures(references, predictions)
    return {
        "wer": round(measures["wer"], 4),
        "mer": round(measures["mer"], 4),
        "wil": round(measures["wil"], 4),
        "hits": measures["hits"],
        "substitutions": measures["substitutions"],
        "deletions": measures["deletions"],
        "insertions": measures["insertions"],
    }


def compute_cer(predictions: list[str], references: list[str]) -> float:
    """
    Compute Character Error Rate.

    CER is more forgiving of minor spelling/accent errors than WER
    and is particularly informative for evaluating learner speech where
    the model might get the word slightly wrong but mostly correct.

    Returns:
        CER as a float (0 = perfect, 1 = completely wrong).
    """
    score = cer(references, predictions)
    return round(score, 4)


def compute_human_agreement(
    predictions: list[str],
    references: list[str],
    tolerance: str = "exact",
) -> dict:
    """
    Compute agreement rate between ASR predictions and human transcriptions.

    The primary metric for this project: how often does the model
    agree with what a human transcriber would write?

    Args:
        predictions: List of post-processed ASR strings.
        references:  List of human transcription strings.
        tolerance:   One of:
                     "exact"     — strings must match exactly after normalization
                     "wer_threshold" — agreement if WER < 0.1 per utterance

    Returns:
        Dict with:
            "agreement_rate"  — fraction in [0, 1]
            "agreed_count"    — number of matching pairs
            "total"           — total pairs evaluated
            "disagreements"   — list of (prediction, reference) pairs that differ
    """
    assert len(predictions) == len(references), "Lists must be the same length."

    agreed = 0
    disagreements = []

    for pred, ref in zip(predictions, references):
        pred_norm = _normalize(pred)
        ref_norm = _normalize(ref)

        if tolerance == "exact":
            match = pred_norm == ref_norm
        elif tolerance == "wer_threshold":
            try:
                m = compute_measures([ref_norm], [pred_norm])
                match = m["wer"] < 0.10
            except Exception:
                match = False
        else:
            raise ValueError(f"Unknown tolerance: {tolerance}")

        if match:
            agreed += 1
        else:
            disagreements.append({"prediction": pred, "reference": ref})

    total = len(predictions)
    rate = agreed / total if total > 0 else 0.0

    return {
        "agreement_rate": round(rate, 4),
        "agreed_count": agreed,
        "total": total,
        "disagreements": disagreements,
    }


def _normalize(text: str) -> str:
    """Lowercase and strip extra whitespace for comparison."""
    return " ".join(text.lower().split())


def evaluate(
    predictions: list[dict],
    references_path: str | Path,
    output_path: str | Path | None = None,
) -> dict:
    """
    Run full evaluation against human reference transcripts.

    Args:
        predictions:     List of {"path": Path, "text": str} dicts.
        references_path: CSV file with columns: filename, transcript.
        output_path:     If provided, save the report as JSON here.

    Returns:
        Full evaluation report dict.
    """
    refs_df = pd.read_csv(references_path)
    refs_df["filename"] = refs_df["filename"].apply(lambda x: Path(x).name)

    pred_texts, ref_texts = [], []
    matched_files = []

    for item in predictions:
        fname = Path(item["path"]).name
        match = refs_df[refs_df["filename"] == fname]
        if match.empty:
            logger.warning(f"No reference found for: {fname}")
            continue
        pred_texts.append(item["text"])
        ref_texts.append(match.iloc[0]["transcript"])
        matched_files.append(fname)

    if not pred_texts:
        logger.error("No matched prediction/reference pairs found.")
        return {}

    wer_results = compute_wer(pred_texts, ref_texts)
    cer_score = compute_cer(pred_texts, ref_texts)
    agreement = compute_human_agreement(pred_texts, ref_texts, tolerance="exact")

    report = {
        "num_files_evaluated": len(matched_files),
        "wer": wer_results,
        "cer": cer_score,
        "human_agreement": agreement,
        "goal_met": agreement["agreement_rate"] >= 0.90,
    }

    logger.info(
        f"WER: {wer_results['wer']:.1%} | "
        f"CER: {cer_score:.1%} | "
        f"Agreement: {agreement['agreement_rate']:.1%} "
        f"({'✓ GOAL MET' if report['goal_met'] else '✗ below 90%'})"
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Evaluation report saved to: {output_path}")

    return report
