# QA Generation & Augmentation: Technical Guide

> **Comprehensive pipeline for generating, filtering, and augmenting spatial question-answer pairs from CT radiology reports.**

## 📚 Overview

This folder implements the complete QA generation pipeline:

1. **Extract** radiology findings and impressions from structured reports
2. **Generate** spatial QA pair candidates using LLM prompts
3. **Filter** non-spatial and low-quality pairs using Gemini 2.5 Flash
4. **Validate** ground truth answers against imaging content
5. **Augment** dataset with category labels and metadata
6. **Format** into model-specific input formats (JSONL, CSV, etc.)

**Result**: 9,077 high-quality spatial VQA pairs derived from 1,601 CT radiology reports

## 🏗️ Directory Structure

```
QA_generation/
├── README.md                              # This file
├── validation_reports_output.json         # Source: radiology reports (CT-RATE)
├── spatial_qa_output_full.json           # Raw QA output (before filtering)
├── spatial_qa_filtered_full.json         # Final filtered QA dataset
├── spatial_qa_filtered_full_with_categories.json  # With spatial categories
│
├── qa_generation.py                      # Core QA generation (OpenAI API)
├── qa_to_jsonl.py                        # Convert JSON → JSONL
├── filter_qa_pairs.py                    # Filter non-spatial QAs
├── llm_judge.py                          # Gemini-based spatial classification
├── human_eval/                           # Human annotation results
│   ├── inter_rater_agreement.json       # Cohen's kappa per category
│   ├── annotation_samples.json          # Sample annotations
│   └── validation_report.md             # Human eval summary
│
└── [metadata]
    ├── case_mappings.json               # Case ID → file mappings
    ├── qa_statistics.json               # Dataset statistics
    └── validation_splits.json           # Train/val/test splits
```

## 🔄 Complete QA Generation Pipeline

### Step 1: Prepare Input Reports

**Source**: CT-RATE dataset with structured radiology reports

**Format** (`validation_reports_output.json`):
```json
{
  "valid_001_a_1.nii.gz": {
    "Findings_EN": "Trachea and both main bronchi are open. Mediastinal structures...",
    "Impressions_EN": "No acute findings. Lungs are clear."
  },
  "valid_002_a_1.nii.gz": {
    "Findings_EN": "...",
    "Impressions_EN": "..."
  }
}
```

**Data Cleaning**:
```bash
python QA_generation/clean_reports.py \
  --input validation_reports_output.json \
  --output validation_reports_clean.json \
  --remove-nulls \
  --filter-language en \
  --min-length 100 \
  --max-length 5000
```

### Step 2: Generate QA Candidates

**Method**: GPT-3.5/GPT-4 with spatial reasoning prompts

**Prompt Template** (`qa_generation.py`):

```python
BASE_PROMPT = """
You are a medical AI assistant specialized in radiology and 3D spatial reasoning.

Read the 3D CT scan report below and generate 7–10 question–answer (QA) pairs 
that test a vision–language model's understanding of spatial and anatomical relationships 
explicitly described in the report.

Focus only on spatial facts such as:
- Laterality (right vs. left, unilateral vs. bilateral)
- Vertical position (upper/lower, superior/inferior)
- Anterior–posterior relations
- Medial–lateral orientation (central/peripheral/midline)
- Spatial extent or boundaries (confined to, crossing midline, extending into)
- Adjacency or containment (within, posterior to, adjacent to)

Guidelines:
- Use ONLY information from the Findings and Impressions sections.
- Do not include diagnostic, interpretive, or normality statements.
- Questions must emphasize *where*, *which side*, *above/below*, or *extent*.
- Answers must be strictly factual and directly derived from the report.

Output as a valid JSON list:
[
  {"question": "...", "answer": "..."},
  ...
]
"""
```

**Batch Generation**:

```bash
python QA_generation/qa_generation.py \
  --input validation_reports_clean.json \
  --output spatial_qa_output_full.json \
  --model gpt-4 \
  --api-key $OPENAI_API_KEY \
  --batch-size 10 \
  --temperature 0.7 \
  --max-tokens 1000 \
  --num-retries 3
```

**Key Parameters**:
- `batch-size`: 10 reports per API call
- `temperature`: 0.7 (balanced creativity vs. consistency)
- `num-retries`: Handle transient API failures
- `rate-limit`: Wait between batches to avoid throttling

**Output**: `spatial_qa_output_full.json` (~12,000 raw QA pairs)

```json
{
  "valid_001_a_1.nii.gz": {
    "qa_pairs": [
      {
        "question": "Which lung contains the lesion?",
        "answer": "Right lung",
        "source": "findings"
      }
    ]
  }
}
```

### Step 3: Filter Non-Spatial Questions

**Method**: Google Gemini 2.5 Flash with spatial classification

**Filtering Logic** (`llm_judge.py`):

```python
FILTER_PROMPT = """
You are a radiology QA auditor. Label each QA pair with two booleans:

- is_spatial: true only if answering requires spatial reasoning about 
  anatomical location, orientation, or relative position visible in the image
  (e.g., "which lung", "proximity", "comparison of organ sizes").
  Pure presence/absence or textual metadata questions are false.

- is_relevant: true only if the question could be answered from the imaging-
  derived Findings/Impressions (not from general knowledge or wording of text).

Respond as JSON:
[
  {
    "index": <QA index starting at 1>,
    "is_spatial": true/false,
    "is_relevant": true/false,
    "confidence": <0.0-1.0>
  }, ...
]

Avoid additional narration. Return ONLY the JSON list.
"""
```

**Batch Filtering**:

```bash
python QA_generation/llm_judge.py \
  --input spatial_qa_output_full.json \
  --output spatial_qa_judged.json \
  --model models/gemini-2.5-flash \
  --api-key $GOOGLE_API_KEY \
  --batch-size 5 \
  --keep-non-spatial False \
  --keep-non-relevant False \
  --confidence-threshold 0.8
```

**Filtering Metrics**:
```json
{
  "total_qa_pairs": 12487,
  "spatial_and_relevant": 9077,
  "non_spatial": 2145,
  "non_relevant": 265,
  "low_confidence": 95,
  "retention_rate": 0.727
}
```

**Output**: `spatial_qa_filtered_full.json` (9,077 pairs)

### Step 4: Category Annotation

**Spatial Categories** (6 dimensions):

1. **Laterality**: Left/Right/Bilateral/Midline
2. **Vertical**: Superior/Inferior/Upper/Lower
3. **Anterior-Posterior**: Anterior/Posterior/Central
4. **Medial-Lateral**: Medial/Lateral/Central/Peripheral
5. **Adjacency/Containment**: Within/Adjacent/Containing
6. **Extent/Boundaries**: Confined/Crossing/Extending

**Auto-Categorization** (keyword matching + LLM refinement):

```bash
python QA_generation/categorize_spatial_qa.py \
  --input spatial_qa_filtered_full.json \
  --output spatial_qa_filtered_full_with_categories.json \
  --method hybrid \
  --use-llm True \
  --model models/gemini-2.5-flash
```

**Output Format**:
```json
{
  "question": "Which lung contains the larger lesion?",
  "answer": "Right lung",
  "categories": {
    "laterality": "laterality",
    "vertical": "none",
    "anterior_posterior": "none",
    "medial_lateral": "none",
    "adjacency": "none",
    "extent": "none"
  },
  "confidence_scores": {
    "laterality": 0.98,
    "vertical": 0.05
  }
}
```

### Step 5: Format for Models

**Convert to JSONL** (unified input format):

```bash
python QA_generation/qa_to_jsonl.py \
  --input spatial_qa_filtered_full.json \
  --output spatial_qa_filtered_full.jsonl \
  --reports validation_reports_output.json \
  --image-root data_volumes/dataset/valid_fixed \
  --absolute-paths True
```

**Create Model-Specific Datasets**:

```bash
# M3D: Ensure absolute NIfTI paths
python QA_generation/prepare_model_dataset.py \
  --input spatial_qa_filtered_full.jsonl \
  --output spatial_qa_filtered_full_nifti_m3d.jsonl \
  --model m3d \
  --format nifti

# CT-Chat: Multi-view slice preparation
python QA_generation/prepare_model_dataset.py \
  --input spatial_qa_filtered_full.jsonl \
  --output spatial_qa_filtered_full_ct_chat.jsonl \
  --model ct-chat \
  --format multi_view \
  --num-slices 10

# RadFM: 2D radiograph patches
python QA_generation/prepare_model_dataset.py \
  --input spatial_qa_filtered_full.jsonl \
  --output spatial_qa_filtered_full_radfm.jsonl \
  --model radfm \
  --format 2d_patches

# MedGemma: Text embeddings only
python QA_generation/prepare_model_dataset.py \
  --input spatial_qa_filtered_full.jsonl \
  --output spatial_qa_filtered_full_medgemma.jsonl \
  --model medgemma \
  --format text_only
```

**Example Output** (`spatial_qa_filtered_full.jsonl`):
```json
{"idx": 0, "image_path": "/PATH/TO/valid_001_a_1.nii.gz", "question": "Which lung contains the larger lesion?", "answer": "Right lung", "category": "laterality", "confidence": 0.98}
{"idx": 1, "image_path": "/PATH/TO/valid_001_a_1.nii.gz", "question": "At what level is the nodule located?", "answer": "Right lower lobe", "category": "vertical", "confidence": 0.92}
```

### Step 6: Data Augmentation

**Optional**: Generate synthetic variations for underrepresented categories

```bash
python QA_generation/augment_qa_pairs.py \
  --input spatial_qa_filtered_full_with_categories.json \
  --output spatial_qa_augmented.json \
  --augmentation-methods ["synonym_replacement", "back_translation", "paraphrase_generation"] \
  --target-distribution balanced \
  --minority-class-threshold 100
```

**Augmentation Techniques**:
- **Synonym Replacement**: Disease/anatomy synonyms
- **Back Translation**: en→de→en paraphrasing
- **Generative**: GPT-based rephrasings maintaining spatial content

## 🔬 Data Quality Metrics

### Per-Pipeline-Step Metrics

```json
{
  "step_1_reports": {
    "total_cases": 1601,
    "valid_findings": 1598,
    "valid_impressions": 1601,
    "avg_findings_len": 512,
    "avg_impressions_len": 128
  },
  "step_2_generation": {
    "total_qa_pairs_generated": 12487,
    "avg_qa_per_case": 7.8,
    "min_qa_per_case": 0,
    "max_qa_per_case": 25
  },
  "step_3_filtering": {
    "total_qa_pairs": 12487,
    "spatial_and_relevant": 9077,
    "non_spatial": 2145,
    "non_relevant": 265,
    "low_confidence": 95,
    "final_retention_rate": 0.727
  },
  "step_4_categorization": {
    "category_distribution": {
      "laterality": 3215,
      "vertical": 1847,
      "anterior_posterior": 892,
      "medial_lateral": 1456,
      "adjacency": 1103,
      "extent": 564
    },
    "multi_category_qa": 1856
  }
}
```

### Human Evaluation (Subset)

**Setup**: 200 QA pairs manually reviewed by 2 radiologists

**Metrics**:
```
Cohen's Kappa (inter-rater agreement):
├─ Relevance: κ = 0.87 (excellent)
├─ Spatial correctness: κ = 0.82 (very good)
├─ Answer accuracy: κ = 0.80 (very good)
└─ Overall: κ = 0.83

Radiologist Agreement vs. Gemini:
├─ False positives: 3.2% (marked spatial, but aren't)
├─ False negatives: 2.8% (missed spatial)
└─ Overall precision: 96.5%, recall: 95.8%
```

## 📊 Dataset Statistics

**Final Dataset Composition** (`qa_statistics.json`):

```json
{
  "total_qa_pairs": 9077,
  "unique_cases": 1601,
  "avg_qa_per_case": 5.67,
  "qa_length_stats": {
    "avg_question_len": 12,
    "min_question_len": 3,
    "max_question_len": 28,
    "avg_answer_len": 8,
    "min_answer_len": 1,
    "max_answer_len": 45
  },
  "category_distribution": {
    "laterality": {
      "count": 3215,
      "percentage": 35.4
    },
    "vertical": {
      "count": 1847,
      "percentage": 20.3
    },
    "anterior_posterior": {
      "count": 892,
      "percentage": 9.8
    },
    "medial_lateral": {
      "count": 1456,
      "percentage": 16.0
    },
    "adjacency": {
      "count": 1103,
      "percentage": 12.1
    },
    "extent": {
      "count": 564,
      "percentage": 6.2
    }
  },
  "multi_category": 1856,
  "vocabulary_stats": {
    "unique_words_questions": 2147,
    "unique_words_answers": 1534,
    "avg_answer_options": 3.2
  }
}
```

## 🔧 Running the Full Pipeline

**Complete Script** (`generate_full_dataset.sh`):

```bash
#!/bin/bash

set -e

# Configuration
source_reports="validation_reports_output.json"
output_dir="."

echo "=== QA Generation Pipeline ==="

echo "[1/6] Cleaning reports..."
python QA_generation/clean_reports.py \
  --input "$source_reports" \
  --output validation_reports_clean.json

echo "[2/6] Generating QA candidates..."
python QA_generation/qa_generation.py \
  --input validation_reports_clean.json \
  --output spatial_qa_output_full.json \
  --model gpt-4 \
  --batch-size 10

echo "[3/6] Filtering with Gemini..."
python QA_generation/llm_judge.py \
  --input spatial_qa_output_full.json \
  --output spatial_qa_judged.json \
  --model models/gemini-2.5-flash \
  --confidence-threshold 0.8

echo "[4/6] Categorizing spatial dimensions..."
python QA_generation/categorize_spatial_qa.py \
  --input spatial_qa_judged.json \
  --output spatial_qa_filtered_full_with_categories.json \
  --method hybrid

echo "[5/6] Converting to JSONL..."
python QA_generation/qa_to_jsonl.py \
  --input spatial_qa_filtered_full_with_categories.json \
  --output spatial_qa_filtered_full.jsonl \
  --reports validation_reports_output.json

echo "[6/6] Creating model-specific datasets..."
for model in m3d ct-chat radfm medgemma merlin; do
  python QA_generation/prepare_model_dataset.py \
    --input spatial_qa_filtered_full.jsonl \
    --output "spatial_qa_filtered_full_${model}.jsonl" \
    --model "$model"
done

echo "=== Pipeline Complete ==="
python QA_generation/compute_statistics.py \
  --dataset spatial_qa_filtered_full.jsonl \
  --output qa_statistics.json
```

**Run**:

```bash
bash generate_full_dataset.sh
```

## 🔐 API Configuration

### OpenAI (GPT-4/GPT-3.5)

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_ORG_ID="org-..."

# Or in Python:
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

### Google Gemini 2.5 Flash

```bash
export GOOGLE_API_KEY="AIzaSy..."

# Or in Python:
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
```

## 📈 Monitoring & Logging

**Logging Configuration** (`logging_config.json`):

```json
{
  "level": "INFO",
  "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  "handlers": {
    "file": {
      "filename": "qa_generation.log",
      "level": "DEBUG"
    },
    "console": {
      "level": "INFO"
    }
  }
}
```

**Progress Tracking**:

```bash
# Monitor API usage in real-time
tail -f qa_generation.log | grep "BATCH\|ERROR\|COMPLETE"

# Check statistics
python -c "
import json
with open('qa_statistics.json') as f:
    stats = json.load(f)
print(f'Total QA pairs: {stats[\"total_qa_pairs\"]}')
print(f'Unique cases: {stats[\"unique_cases\"]}')
print(f'Category distribution:')
for cat, val in stats['category_distribution'].items():
    print(f'  {cat}: {val[\"count\"]} ({val[\"percentage\"]:.1f}%)')
"
```

## 🛟 Troubleshooting

### API Rate Limits

```bash
# Increase timeout and retry backoff
python QA_generation/qa_generation.py \
  --input validation_reports_clean.json \
  --api-timeout 30 \
  --max-retries 5 \
  --retry-backoff-factor 3.0
```

### Invalid JSON Responses

```bash
# Enable strict JSON validation and error recovery
python QA_generation/qa_generation.py \
  --input validation_reports_clean.json \
  --json-strict True \
  --handle-parse-errors true \
  --fallback-to-gpt35 True
```

### Duplicate/Low Quality Pairs

```bash
# Add deduplication and quality filtering
python QA_generation/filter_qa_pairs.py \
  --input spatial_qa_judged.json \
  --output spatial_qa_filtered.json \
  --remove-duplicates True \
  --min-question-length 4 \
  --min-answer-length 2 \
  --max-answer-length 50
```

## 📚 References

- [OpenAI GPT-4 API](https://platform.openai.com/docs/guides/gpt-4)
- [Google Gemini API](https://ai.google.dev/docs)
- [Radiology Report Standards](https://www.radiologyassistant.nl/)

---

**Last Updated**: February 27, 2026
