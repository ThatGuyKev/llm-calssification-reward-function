# falcon-classification

Kaggle competition: https://www.kaggle.com/competitions/llm-classification-finetuning
Predict which of two chatbot responses a human preferred (`winner_model_a`, `winner_model_b`, `winner_tie`). Metric: multi-class log loss (uniform guess = 1.0986).

## Current state
- `llm_classification_starter.ipynb` exists: CPU baseline (hand-crafted text features + TF-IDF similarity/SVD, LightGBM, 5-fold CV, A/B swap augmentation + swap TTA, writes `submission.csv`).
- The notebook was written by hand and has **never been executed or validated**. First step next session: run it (or `DEBUG = True` for a quick pass) and fix any errors.
- No data is downloaded yet. Notebook reads `/kaggle/input/llm-classification-finetuning`, falling back to `./data`.
- Local Python 3.14 had no pandas/sklearn/lightgbm/nbformat installed and no pip/uv on PATH, so running locally needs an env set up first (or run on Kaggle).

## Data notes
- `prompt`, `response_a`, `response_b` are JSON-encoded lists of turns (entries can be null).
- `model_a` / `model_b` exist only in train, so they can't be features.
- Label is symmetric in A/B: swap augmentation maps labels with `[1, 0, 2]`.

## QLoRA Falcon (`train_qlora.py`)
- Falcon3-1B-Base (Llama arch) + 4-bit NF4 + LoRA r=16 on all linear layers, `AutoModelForSequenceClassification` head, fp16 autocast (RTX 2060 6 GB, Turing: no bf16/flash-attn).
- Random A/B swap augmentation in training, swap-TTA at inference. Holdout val split (5%), not CV. Outputs in `outputs/falcon3-1b-qlora/` (adapter, `val_probs.npy`, `submission_falcon.csv`).
- Smoke test (`--smoke`, 256 samples, len 256) passes: ~1.45 s per optimizer step at eff. batch 8. Real run not done yet; record val log loss here once it is.
- Deps installed into `.venv` (torch 2.15 cu132, transformers 5.22, peft 0.21, bitsandbytes 0.50) on Python 3.14.

## Next steps
1. Run the baseline, record CV log loss here.
2. Fine-tune a cross-encoder (DeBERTa-v3) or LoRA a decoder LLM as a 3-way classifier.
3. Stack LLM logits with the LightGBM OOF predictions; calibrate with temperature scaling.
4. Check the competition's Code Requirements (internet, runtime limit) before submitting.
