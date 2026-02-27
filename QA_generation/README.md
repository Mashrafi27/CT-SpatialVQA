# QA Generation & Filtering Pipeline

This folder contains the pipeline that generates, validates, and filters spatial QA pairs from CT‑RATE reports.

## Contents

```
QA_generation/
├── qa_generation.py              # Generate raw QA pairs (OpenAI)
├── llm_judge.py                  # LLM validator (spatial + relevance)
├── filter_qa_pairs.py            # Filter using LLM judgments
├── qa_to_jsonl.py                # Convert JSON → JSONL
├── spatial_qa_output_full.json   # Raw QA output (full)
├── spatial_qa_filtered_full.json # Filtered QA output (full)
├── validation_reports_output.json
└── human_eval/                   # Human audit artifacts
```

## Pipeline Steps (Summary)

1. **Generate raw QA pairs** from report text
2. **LLM validation** to tag spatial + relevance
3. **Filter** to obtain the final QA set
4. **Convert to JSONL** for inference

## Typical Usage

### 1) Generate raw QA

`qa_generation.py` expects a CSV with columns: `file_name`, `findings_en`, `impressions_en`.

```bash
python QA_generation/qa_generation.py \
  --input_csv PATH/TO/ct_rate_reports.csv \
  --output_json spatial_qa_output_full.json
```

### 2) Validate with LLM

```bash
python QA_generation/llm_judge.py \
  --input spatial_qa_output_full.json \
  --output gemini_judgments.json \
  --model models/gemini-2.5-flash
```

### 3) Filter

```bash
python QA_generation/filter_qa_pairs.py \
  --qa-json spatial_qa_output_full.json \
  --judgments gemini_judgments.json \
  --output spatial_qa_filtered_full.json
```

### 4) Convert to JSONL

```bash
python QA_generation/qa_to_jsonl.py \
  --qa-json spatial_qa_filtered_full.json \
  --nifti-root dataset/data_volumes/dataset/valid_fixed \
  --output-jsonl spatial_qa_filtered_full_nifti.jsonl
```

## Notes

- The **final dataset** is published under `dataset/ct_spatialvqa/`.
- **Model‑specific JSONLs** are stored in `benchmarking/inputs/`.
