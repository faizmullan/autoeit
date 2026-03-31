# AutoEIT — Audio-to-Text Transcription for L2 Learner Data
> **GSoC 2026 Project** | HumanAI Foundation | Applicant: Faiz Mullan
End-to-end ML pipeline to automatically transcribe EIT audio from non-native Spanish speakers, targeting >= 90% agreement with human transcribers.
## Results (Completed Before GSoC Submission)
| Metric | Baseline | Fine-Tuned | GSoC Target |
|--------|----------|------------|-------------|
| WER | 28.6% | 4.3% | <10% |
| CER | 11.1% | 2.2% | --- |
| Human Agreement | 12.0% | 84.0% | >=90% |
| ERMS Speaker | 36.9% WER | 1.4% WER / 91.7% agreement | >=90% |
Dataset: L2-Arctic v5.0 --- 4,402 utterances, 4.69 hours, 4 Spanish L1 speakers
## Pipeline Stages
| Stage | Description | Status |
|-------|-------------|--------|
| Stage 1 | Data preparation, VAD, quality filtering | Done |
| Stage 2 | Baseline Whisper transcription (28.6% WER) | Done |
| Stage 3 | Fine-tuning whisper-tiny, 500 samples, 10 epochs | Done |
| Stage 4 | Spanish L1 phoneme interference correction | Done |
| Stage 5 | Final evaluation report | Done |
## Quick Start
git clone https://github.com/faizmullan/autoeit.git
cd autoeit
pip install -r requirements.txt
## Run the Pipeline
Stage 1 - Data preparation
python scripts/stage1_data_preparation.py --dataset_dir path/to/l2arctic --output_dir outputs/stage1
Stage 2 - Baseline ASR
python scripts/stage2_transcribe.py --manifest outputs/stage1/clean_manifest.csv --dataset_dir path/to/l2arctic --output_dir outputs/stage2 --sample 50
Stage 3 - Fine-tuning
python scripts/stage3_finetune.py --manifest outputs/stage1/clean_manifest.csv --output_dir outputs/stage3 --batch_size 1 --model openai/whisper-tiny
Stage 3 - Evaluate
python scripts/evaluate_finetuned.py --manifest outputs/stage1/clean_manifest.csv --model_path outputs/stage3/whisper-small-l2arctic --output_dir outputs/stage3_eval --sample 50
Stage 4 - Post-processing
python scripts/stage4_postprocess.py --input_csv outputs/stage3_eval/finetuned_transcriptions.csv --output_dir outputs/stage4
Stage 5 - Final report
python scripts/stage5_final_report.py --baseline_csv outputs/stage2/transcriptions.csv --finetuned_csv outputs/stage3_eval/finetuned_transcriptions.csv --output_dir outputs/stage5
## Per-Speaker Results
| Speaker | L1 | Files | Baseline WER | Fine-tuned WER | Agreement |
|---------|-----|-------|-------------|----------------|-----------|
| MBMPS | Spanish/F | 1,132 | 23.1% | 3.6% | 77.8% |
| NJS | Spanish/F | 1,131 | 24.9% | 4.2% | 85.7% |
| ERMS | Spanish/M | 1,132 | 36.9% | 1.4% | 91.7% |
| EBVS | Spanish/M | 1,007 | 28.6% | 7.0% | 80.0% |
| Overall | --- | 4,402 | 28.6% | 4.3% | 84.0% |
## Tests
pytest tests/ -v
20/20 tests passing
## Mentors
Prof. Mandy Faretta-Stutenberg (Northern Illinois University)
Xabier Granja (University of Alabama)
## Author
Faiz Mullan | faizmullan2005@gmail.com | Surat, Gujarat, India
