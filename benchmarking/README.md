# Benchmarking: Technical Guide

This folder contains the **benchmarking stack** for running 3D medical VLMs on CT-SpatialVQA.

## Structure

```
benchmarking/
├── preprocess/                          # preprocessing scripts
├── inference/                           # model inference harnesses (+ submodules)
├── inputs/                              # model-specific JSONL inputs (paths + questions)
├── eval_scripts/
│   └── scripts/                         # scoring, aggregation, plots
└── reports/                             # predictions, LLM eval, metrics, plots
```

## Setup

### 1) Submodules

Initialize submodules (see the repository root instructions).

### 2) Environments

Each model has its own dependencies (often pinned). Recommended workflow:
- create a separate env per model (e.g., conda/venv)
- install requirements from that model’s folder under `benchmarking/inference/<model>/`

## Workflow

### 1) Inputs

Model-specific JSONLs live in `benchmarking/inputs/<model>/` and reference CT‑RATE volumes under:
`dataset/data_volumes/dataset/valid_fixed/...`

### 2) Preprocess

Preprocessing scripts live in `benchmarking/preprocess/` and convert CT volumes into model-specific formats.

### 3) Inference

Inference harnesses are in `benchmarking/inference/`. Outputs are written to `benchmarking/reports/`.

### 4) LLM‑as‑Judge

Evaluation scripts live in `benchmarking/eval_scripts/` and produce per-model judge outputs plus aggregated scores.

### 5) Aggregation & Plots

Aggregation and plotting utilities are under `benchmarking/eval_scripts/` and write figures/tables into `benchmarking/reports/`.

## Commands (Generic Template)

Run from the repository root. Replace placeholders like `<MODEL>` and paths as needed.

### 1) Preprocess

```bash
python benchmarking/preprocess/preprocess_<MODEL>.py \
  --input-jsonl benchmarking/inputs/<MODEL>/INPUT.jsonl \
  --nifti-root dataset/data_volumes/dataset/valid_fixed \
  --output-root benchmarking/preprocess/<MODEL>_outputs \
  --output-jsonl benchmarking/preprocess/<MODEL>_processed.jsonl
```

### 2) Inference

```bash
python benchmarking/inference/<MODEL>/run_custom_eval.py \
  --dataset benchmarking/preprocess/<MODEL>_processed.jsonl \
  --output benchmarking/reports/predictions/<MODEL>_predictions_full.jsonl
```

### 3) LLM-as-Judge

```bash
python benchmarking/eval_scripts/scripts/evaluate_with_gpt.py \
  --predictions benchmarking/reports/predictions/<MODEL>_predictions_full.jsonl \
  --output benchmarking/reports/llm_eval/<MODEL>_gpt_eval.json

python benchmarking/eval_scripts/scripts/evaluate_with_gemini.py \
  --predictions benchmarking/reports/predictions/<MODEL>_predictions_full.jsonl \
  --output benchmarking/reports/llm_eval/<MODEL>_gemini_eval.json

python benchmarking/eval_scripts/scripts/evaluate_with_qwen.py \
  --predictions benchmarking/reports/predictions/<MODEL>_predictions_full.jsonl \
  --output benchmarking/reports/llm_eval/<MODEL>_qwen_eval.json
```

### 4) Aggregate & Plot

```bash
python benchmarking/eval_scripts/scripts/build_correctness_matrix.py \
  --reports-dir benchmarking/reports/llm_eval \
  --pattern "*_eval*.json" \
  --output benchmarking/reports/metrics/correctness_matrix_avg3.csv

python benchmarking/eval_scripts/scripts/plot_answer_length_distributions.py \
  --data-json benchmarking/reports/plots/answer_length_distributions.json \
  --output benchmarking/reports/plots/answer_length_distributions.png
```

## Notes

- Final dataset files live in `dataset/ct_spatialvqa/`.
- QA generation & filtering is documented in `QA_generation/`.
