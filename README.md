# AutoEIT — Audio-to-Text Transcription for L2 Learner Data

A GSoC 2026 project under HumanAI Foundation.

## Overview
End-to-end pipeline to transcribe audio from non-native Spanish speakers performing the
Elicited Imitation Task (EIT), targeting ≥90% agreement with human transcribers.

## Project Structure
```
autoeit/
├── data/
│   ├── raw/            # Original .wav/.mp3 EIT recordings
│   ├── processed/      # Cleaned, normalized audio (16kHz mono)
│   ├── transcripts/    # Human reference transcripts (.csv / .json)
│   └── splits/         # train/val/test split manifests
├── src/
│   ├── preprocessing/  # Audio cleaning, segmentation, normalization
│   ├── asr/            # Whisper fine-tuning and inference
│   ├── postprocessing/ # Error correction, LM rescoring
│   ├── evaluation/     # WER, CER, agreement metrics
│   └── utils/          # Shared helpers (logging, config, file I/O)
├── configs/            # YAML config files for each stage
├── notebooks/          # Exploration and analysis notebooks
├── tests/              # Unit tests per module
├── scripts/            # CLI entry points (run_pipeline.py etc.)
└── outputs/            # Final transcription outputs
```

## Setup
```bash
git clone <repo-url>
cd autoeit
pip install -r requirements.txt
```

## Quick Start
```bash
# Run full pipeline on a folder of audio files
python scripts/run_pipeline.py --input data/raw/ --output outputs/ --config configs/default.yaml

# Run only preprocessing
python scripts/preprocess.py --input data/raw/ --output data/processed/

# Run only transcription
python scripts/transcribe.py --input data/processed/ --output outputs/

# Evaluate against human transcripts
python scripts/evaluate.py --predictions outputs/ --references data/transcripts/
```

## Pipeline Stages
1. **Preprocessing** — noise reduction, silence trimming, 16kHz mono conversion
2. **ASR Transcription** — fine-tuned Whisper model for learner Spanish
3. **Post-processing** — rule-based + LM error correction
4. **Evaluation** — WER/CER + human agreement scoring

## Requirements
- Python 3.9+
- PyTorch 2.0+
- See `requirements.txt` for full list

## Mentors
- Mandy Faretta-Stutenberg (Northern Illinois University)
- Xabier Granja (University of Alabama)
