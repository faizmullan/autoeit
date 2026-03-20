# tests/test_preprocessing/test_l2arctic_loader.py
"""
Unit tests for the L2-Arctic loader.
Uses a fake directory structure — no real dataset required.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os

from src.data.l2arctic_loader import (
    build_manifest,
    split_manifest,
    SPEAKER_METADATA,
)


@pytest.fixture
def fake_l2arctic(tmp_path):
    """
    Create a minimal fake L2-Arctic directory structure:
        tmp_path/
            EBVS/           (Spanish male)
                wav/arctic_a0001.wav  (empty placeholder)
                transcript/arctic_a0001.txt
            NJS/            (Spanish female)
                wav/arctic_a0001.wav
                transcript/arctic_a0001.txt
            HKK/            (Korean male)
                wav/arctic_a0001.wav
                transcript/arctic_a0001.txt
    """
    for speaker, meta in [
        ("EBVS", "spanish"),
        ("NJS",  "spanish"),
        ("HKK",  "korean"),
    ]:
        (tmp_path / speaker / "wav").mkdir(parents=True)
        (tmp_path / speaker / "transcript").mkdir(parents=True)

        # Create 3 fake wav + transcript pairs per speaker
        for i in range(1, 4):
            stem = f"arctic_a{i:04d}"
            # Empty WAV placeholder (not a real audio file — loader only reads path)
            (tmp_path / speaker / "wav" / f"{stem}.wav").touch()
            (tmp_path / speaker / "transcript" / f"{stem}.txt").write_text(
                f"This is sentence {i} spoken by {speaker}."
            )

    return tmp_path


def test_build_manifest_all_speakers(fake_l2arctic):
    df = build_manifest(fake_l2arctic)
    assert len(df) == 9  # 3 speakers × 3 utterances
    assert set(df.columns) >= {"audio_path", "sentence", "speaker", "l1", "gender"}


def test_build_manifest_l1_filter(fake_l2arctic):
    df = build_manifest(fake_l2arctic, l1_filter=["spanish"])
    assert len(df) == 6  # only EBVS and NJS
    assert set(df["l1"].unique()) == {"spanish"}
    assert "korean" not in df["l1"].values


def test_build_manifest_speaker_filter(fake_l2arctic):
    df = build_manifest(fake_l2arctic, speakers_filter=["EBVS"])
    assert len(df) == 3
    assert df["speaker"].unique().tolist() == ["EBVS"]


def test_build_manifest_missing_dir():
    with pytest.raises(FileNotFoundError):
        build_manifest("/nonexistent/path/to/l2arctic")


def test_split_manifest_ratios(fake_l2arctic):
    df = build_manifest(fake_l2arctic)
    train, val, test = split_manifest(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    assert len(train) + len(val) + len(test) == len(df)


def test_split_manifest_no_overlap(fake_l2arctic):
    df = build_manifest(fake_l2arctic)
    train, val, test = split_manifest(df)
    all_paths = pd.concat([train, val, test])["audio_path"].tolist()
    assert len(all_paths) == len(set(all_paths)), "Duplicate paths across splits"


def test_speaker_metadata_complete():
    """All 24 speakers must be in SPEAKER_METADATA with required fields."""
    assert len(SPEAKER_METADATA) == 24
    for speaker, meta in SPEAKER_METADATA.items():
        assert "l1" in meta, f"Missing l1 for {speaker}"
        assert "gender" in meta, f"Missing gender for {speaker}"
        assert meta["gender"] in ("M", "F"), f"Invalid gender for {speaker}"
