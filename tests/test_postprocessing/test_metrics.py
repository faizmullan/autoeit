# tests/test_postprocessing/test_metrics.py
"""Tests for WER, CER, and human agreement metrics."""

import pytest
from src.evaluation.metrics import (
    compute_wer,
    compute_cer,
    compute_human_agreement,
)


def test_wer_perfect_match():
    preds = ["yo quiero comer"]
    refs  = ["yo quiero comer"]
    result = compute_wer(preds, refs)
    assert result["wer"] == 0.0


def test_wer_one_substitution():
    preds = ["yo quiero dormir"]
    refs  = ["yo quiero comer"]
    result = compute_wer(preds, refs)
    # 1 substitution out of 3 words = 0.333
    assert 0.30 < result["wer"] < 0.40


def test_cer_perfect_match():
    preds = ["hola"]
    refs  = ["hola"]
    assert compute_cer(preds, refs) == 0.0


def test_cer_partial_match():
    preds = ["holas"]
    refs  = ["hola"]
    score = compute_cer(preds, refs)
    assert 0 < score <= 0.5


def test_human_agreement_all_match():
    preds = ["yo como", "ella duerme"]
    refs  = ["yo como", "ella duerme"]
    result = compute_human_agreement(preds, refs)
    assert result["agreement_rate"] == 1.0
    assert result["agreed_count"] == 2
    assert len(result["disagreements"]) == 0


def test_human_agreement_none_match():
    preds = ["yo como"]
    refs  = ["ella duerme"]
    result = compute_human_agreement(preds, refs)
    assert result["agreement_rate"] == 0.0
    assert len(result["disagreements"]) == 1


def test_human_agreement_partial():
    preds = ["yo como", "ella duerme", "nosotros hablamos"]
    refs  = ["yo como", "ella corre",  "nosotros hablamos"]
    result = compute_human_agreement(preds, refs)
    assert result["agreement_rate"] == pytest.approx(2 / 3, abs=0.01)


def test_human_agreement_case_insensitive():
    preds = ["Yo Como"]
    refs  = ["yo como"]
    result = compute_human_agreement(preds, refs)
    assert result["agreement_rate"] == 1.0
