# Benchmarking: Technical Guide

This folder contains the **full benchmarking stack** for CT‑SpatialVQA:

- **Preprocess** CT volumes per model
- **Run inference** across 3D medical VLMs
- **Evaluate** with LLM‑as‑Judge and text metrics
- **Aggregate** results

## Structure

```
benchmarking/
├── preprocess/               # preprocessing scripts
├── inference/                # model-specific inference (submodules)
├── inputs/                   # model-specific JSONL inputs
├── eval_scripts/             # evaluation + plots + metrics
└── reports/                  # predictions, eval, plots, metrics
```

## 1) Preprocess

Preprocessing scripts live in `benchmarking/preprocess/`.
Example (M3D):

```bash
python benchmarking/preprocess/preprocess_m3d.py \
  --input-jsonl benchmarking/inputs/m3d/spatial_qa_filtered_full_nifti_m3d.jsonl \
  --nifti-root dataset/data_volumes/dataset/valid_fixed \
  --output-root benchmarking/preprocess/m3d_outputs \
  --output-jsonl benchmarking/preprocess/m3d_processed.jsonl
```

## 2) Inference

Model harnesses are in `benchmarking/inference/<model>/`. Example:

```bash
python benchmarking/inference/med3dvlm/run_custom_eval.py \
  --dataset benchmarking/preprocess/med3dvlm_processed.jsonl \
  --output benchmarking/reports/predictions/med3dvlm_predictions_full.jsonl \
  --batch-size 4 \
  --device cuda:0
```

## 3) LLM‑as‑Judge Evaluation

Scripts are in `benchmarking/eval_scripts/`:

```bash
python benchmarking/eval_scripts/evaluate_with_gemini.py \
  --predictions benchmarking/reports/predictions/med3dvlm_predictions_full.jsonl \
  --output benchmarking/reports/llm_eval/med3dvlm_gemini_eval.json \
  --model models/gemini-2.5-flash

python benchmarking/eval_scripts/evaluate_with_gpt.py \
  --predictions benchmarking/reports/predictions/med3dvlm_predictions_full.jsonl \
  --output benchmarking/reports/llm_eval/med3dvlm_gpt_eval.json \
  --model gpt-4o-mini

python benchmarking/eval_scripts/evaluate_with_qwen.py \
  --predictions benchmarking/reports/predictions/med3dvlm_predictions_full.jsonl \
  --output benchmarking/reports/llm_eval/med3dvlm_qwen_eval.json \
  --model qwen-plus
```

## 4) Aggregation & Plots

```bash
python benchmarking/eval_scripts/build_correctness_matrix.py \
  --reports-dir benchmarking/reports/llm_eval \
  --pattern "*_eval*.json" \
  --output benchmarking/reports/metrics/correctness_matrix_avg3.csv

python benchmarking/eval_scripts/plot_answer_length_distributions.py \
  --data-json benchmarking/reports/plots/answer_length_distributions.json \
  --output benchmarking/reports/plots/answer_length_distributions.png
```

## Notes

- **Model‑specific JSONLs** live in `benchmarking/inputs/<model>/`.
- **Final dataset** lives in `dataset/ct_spatialvqa/`.
- For QA generation details, see `QA_generation/README.md`.
