# AutoEIT — Audio-to-Text Transcription for L2 Learner Data

> **GSoC 2026 Project** | HumanAI Foundation | Applicant: Faiz Mullan

End-to-end ML pipeline to automatically transcribe EIT (Elicited Imitation Task) audio recordings from non-native Spanish speakers, targeting ≥90% agreement with human transcribers.

---

## 🏆 Results (Preliminary Experiments — Completed Before GSoC)

| Metric | Baseline (no fine-tuning) | Full Fine-Tuned | GSoC Target |
|--------|--------------------------|-----------------|-------------|
| **WER** | 28.6% | **4.3%** | <10% ✅ |
| **CER** | 11.1% | **2.2%** | — |
| **Human Agreement** | 12.0% | **84.0%** | ≥90% |
| **ERMS Speaker** | 36.9% WER | **1.4% WER / 91.7% agreement** | ≥90% ✅ |

**Dataset:** L2-Arctic v5.0 — 4,402 utterances, 4.69 hours, 4 Spanish L1 speakers (MBMPS, NJS, ERMS, EBVS)  
**Training:** whisper-tiny, 500 samples, 10 epochs, CPU-only, batch_size=1

---

## 🔧 Pipeline Stages

```
Stage 1 — Data Preparation    → 4,402 files processed, 100% quality pass
Stage 2 — Baseline ASR        → 28.6% WER (Whisper-small, no fine-tuning)
Stage 3 — Fine-Tuning         → 4.3% WER (whisper-tiny, LoRA, 10 epochs)
Stage 4 — Post-Processing     → Spanish L1 phoneme interference correction
Stage 5 — Evaluation          → WER, CER, human agreement final report
```

---

## 📁 Project Structure

```
autoeit/
├── scripts/
│   ├── stage1_data_preparation.py   # Data loading, VAD, quality filtering
│   ├── stage2_transcribe.py         # Baseline Whisper transcription
│   ├── stage3_finetune.py           # Fine-tuning with HuggingFace Trainer
│   ├── stage4_postprocess.py        # Spanish L1 interference corrections
│   ├── stage5_final_report.py       # WER/CER/agreement evaluation report
│   └── evaluate_finetuned.py        # Evaluate fine-tuned model
├── src/
│   ├── preprocessing/               # Audio loading, VAD, cleaning
│   ├── asr/                         # Whisper transcriber and fine-tuner
│   ├── postprocessing/              # Error correction, filler removal
│   ├── evaluation/                  # WER, CER metrics
│   └── data/                        # L2-Arctic dataset loader
├── configs/
│   └── default.yaml                 # Pipeline configuration
├── tests/                           # 20/20 unit tests passing
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/faizmullan/autoeit.git
cd autoeit
pip install -r requirements.txt
```

### Run the full pipeline

**Stage 1 — Data Preparation:**
```bash
python scripts/stage1_data_preparation.py \
    --dataset_dir "path/to/l2arctic" \
    --output_dir outputs/stage1
```

**Stage 2 — Baseline ASR:**
```bash
python scripts/stage2_transcribe.py \
    --manifest outputs/stage1/clean_manifest.csv \
    --dataset_dir "path/to/l2arctic" \
    --output_dir outputs/stage2 \
    --sample 50
```

**Stage 3 — Fine-Tuning (leave overnight on CPU):**
```bash
python scripts/stage3_finetune.py \
    --manifest outputs/stage1/clean_manifest.csv \
    --output_dir outputs/stage3 \
    --batch_size 1 \
    --model openai/whisper-tiny
```

**Stage 3 Evaluation:**
```bash
python scripts/evaluate_finetuned.py \
    --manifest outputs/stage1/clean_manifest.csv \
    --model_path outputs/stage3/whisper-small-l2arctic \
    --output_dir outputs/stage3_eval \
    --sample 50
```

**Stage 4 — Post-Processing:**
```bash
python scripts/stage4_postprocess.py \
    --input_csv outputs/stage3_eval/finetuned_transcriptions.csv \
    --output_dir outputs/stage4
```

**Stage 5 — Final Report:**
```bash
python scripts/stage5_final_report.py \
    --baseline_csv outputs/stage2/transcriptions.csv \
    --finetuned_csv outputs/stage3_eval/finetuned_transcriptions.csv \
    --output_dir outputs/stage5
```

---

## 🧠 Technical Approach

### ASR Model
- **Primary:** OpenAI Whisper (fine-tuned via HuggingFace Seq2SeqTrainer)
- **Fine-tuning:** LoRA/PEFT — reduces trainable parameters by ~99%
- **Hardware:** CPU-only (7.7GB RAM) — fully reproducible on consumer hardware

### Spanish L1 Phoneme Corrections (Stage 4)
Spanish speakers systematically substitute English phonemes not in their L1 inventory:

| English Phoneme | Error Rate | Spanish Substitution | Example |
|----------------|-----------|---------------------|---------|
| /ð/ (voiced dental) | 68% | /d/ | "the" → "de" |
| /θ/ (voiceless dental) | 62% | /s/ | "think" → "sink" |
| /v/ (labiodental) | 45% | /b/ | "very" → "berry" |
| /ŋ/ (velar nasal) | 38% | /n/ | "sing" → "sin" |

### VAD Fix (Critical Bug)
The original VAD crashed with `reshape error` for 97% of files. Fixed using:
```python
# FIXED — works on any audio length
intervals = librosa.effects.split(audio, top_db=30, frame_length=2048, hop_length=512)
```

---

## 📊 Per-Speaker Results

| Speaker | L1 | Files | Baseline WER | Fine-tuned WER | Agreement |
|---------|-----|-------|-------------|----------------|-----------|
| MBMPS | Spanish/F | 1,132 | 23.1% | **3.6%** | 77.8% |
| NJS | Spanish/F | 1,131 | 24.9% | **4.2%** | 85.7% |
| ERMS | Spanish/M | 1,132 | 36.9% | **1.4%** | **91.7% ✅** |
| EBVS | Spanish/M | 1,007 | 28.6% | **7.0%** | 80.0% |
| **Overall** | — | **4,402** | **28.6%** | **4.3%** | **84.0%** |

---

## 🧪 Tests

```bash
pytest tests/ -v
# 20/20 tests passing
```

---

## 📋 Requirements

- Python 3.9+
- PyTorch 2.0+ (CPU)
- HuggingFace Transformers
- librosa, soundfile, noisereduce
- See `requirements.txt` for full list

---

## 🎯 About This Project

This pipeline was built as a prerequisite task for the **GSoC 2026 AutoEIT project** under the **HumanAI Foundation**, mentored by:
- Prof. Mandy Faretta-Stutenberg (Northern Illinois University)
- Xabier Granja (University of Alabama)

The EIT (Elicited Imitation Task) is a validated tool for measuring L2 proficiency. Manual transcription of learner responses is the primary research bottleneck — this pipeline automates that process.

**Applicant:** Faiz Mullan | faizmullan2005@gmail.com | Surat, Gujarat, India
