# QA Generation & Filtering Pipeline

This folder contains the pipeline that generates, validates, and filters spatial QA pairs from CT-RATE reports.

## What This Produces

- A raw QA set (before filtering)
- An LLM-validated / filtered QA set (dataset release)
- Generic JSONL exports used for inference

The released dataset is in `dataset/`.

## Pipeline (High-Level)

1. Generate QA pairs from report text
2. Validate with an LLM (spatial + relevance)
3. Filter to keep only spatially grounded QA pairs
4. Export to JSONL for inference

## Requirements

- `OPENAI_API_KEY` (generation)
- `GEMINI_API_KEY` (validation / filtering)

If you want to reproduce the generation and filtering, run the scripts in this folder in the order described above.

## Commands

Run from the repository root.

### 1) Generate raw QA pairs

```bash
python QA_generation/qa_generation.py \
  --input_csv PATH/TO/CT_RATE_REPORTS.csv \
  --output_json spatial_qa_output_full.json
```

### 2) LLM validation (spatial + relevance)

```bash
python QA_generation/llm_judge.py \
  --qa spatial_qa_output_full.json \
  --reports validation_reports_output.json \
  --output gemini_judgments.json
```

### 3) Filter to the final QA set

```bash
python QA_generation/filter_qa_pairs.py \
  --qa-json spatial_qa_output_full.json \
  --judgments gemini_judgments.json \
  --output spatial_qa_filtered_full.json
```

### 4) Export to generic JSONL (for inference)

```bash
python QA_generation/qa_to_jsonl.py \
  --qa-json spatial_qa_filtered_full.json \
  --nifti-root dataset/data_volumes/dataset/valid_fixed \
  --output-jsonl spatial_qa_filtered_full_nifti.jsonl
```

## Human Audit

Human audit artifacts (summary and progress) are stored in `human_eval/`.
