# Benchmarking Pipeline: Technical Implementation Guide

> **Complete infrastructure for running 3D medical VLMs on the CT-SpatialVQA benchmark with multi-LLM evaluation.**

## 📚 Overview

This folder contains the model-specific inference harnesses and evaluation infrastructure for benchmarking 3D medical vision-language models on spatial VQA. The benchmarking pipeline:

1. **Preprocesses input data** to model-specific formats (3D volumes, multi-view slices, etc.)
2. **Runs inference** on each model with standardized interfaces
3. **Captures predictions** in a unified JSONL format
4. **Evaluates results** using three independent LLM judges (GPT-4o, Gemini 2.5 Flash, Qwen-Plus)
5. **Aggregates scores** into comprehensive performance metrics

## 🏗️ Directory Structure

```
benchmarking/
├── README.md                          # This file
├── inference/                         # Model-specific inference code
│   ├── med3dvlm/                     # Med3DVLM: 3D CNN + CLIP
│   │   ├── run_custom_eval.py       # Inference harness
│   │   ├── README.md                # Model-specific setup
│   │   ├── env/                     # conda/pip environment files
│   │   └── Med3DVLM.git/            # Git submodule
│   │
│   ├── m3d/                         # M3D: 3D ViT + LLaMA
│   │   ├── run_m3d_eval.py
│   │   ├── README.md
│   │   └── env/requirements.txt
│   │
│   ├── ct-chat/                     # CT-Chat: Multimodal LLM
│   │   ├── run_ctchat_eval.py
│   │   ├── README.md
│   │   └── env/requirements.txt
│   │
│   ├── merlin/                      # Merlin: Multi-task encoder
│   │   ├── run_merlin_eval.py
│   │   ├── README.md
│   │   └── repo/ (Git submodule)
│   │
│   ├── radfm/                       # RadFM: Radiology Foundation Model
│   │   ├── run_radfm_eval.py
│   │   ├── README.md
│   │   └── env/requirements.txt
│   │
│   ├── medgemma/                    # MedGemma: Medical LLM
│   │   ├── run_medgemma_eval.py
│   │   ├── README.md
│   │   └── env/requirements.txt
│   │
│   ├── vila-m3/                     # VILA-M3: Hybrid expert system
│   │   ├── run_vista3d_eval.py
│   │   ├── README.md
│   │   └── env/requirements.txt
│   │
│   ├── medevalkit/                  # MedEvalKit: Multi-task suite
│   │   ├── run_medevalkit_eval.py
│   │   ├── README.md
│   │   └── env/requirements.txt
│   │
│   ├── ct-clip/                     # CT-CLIP: Utility functions
│   │   └── README.md
│   │
│   ├── README.md                    # Model matrix and setup instructions
│   └── models.csv                   # Model metadata
│
└── [evaluation scripts]             # Global evaluation & aggregation
    # Located in 3D_VLM_Spatial/scripts/
```

## 🔄 Workflow

### Phase 1: Data Preparation

**Input**: `3D_VLM_Spatial/QA_generation/spatial_qa_filtered_full.json`

Model-specific preprocessing converts the dataset:

```
Original QA JSON
    ↓
[per-model preprocessing]
    ↓
Model-specific format
    ├─ Med3DVLM: NIfTI volumes (indexed)
    ├─ M3D: 256×256×32 .npy files + JSONL
    ├─ CT-Chat: Multi-view DICOM/NIfTI
    ├─ Merlin: Multi-view slices + metadata
    ├─ RadFM: 2D radiograph patches
    ├─ MedGemma: Text features + image embeddings
    ├─ VILA-M3: VISTA3D segmentation overlays
    └─ MedEvalKit: Multi-view standardized format
```

**Example**: M3D preprocessing (256×256×32 voxels)

```bash
python 3D_VLM_Spatial/preprocess/preprocess_m3d.py \
  --input-jsonl QA_generation/spatial_qa_filtered_full_nifti_m3d.jsonl \
  --nifti-root data_volumes/dataset/valid_fixed \
  --output-root 3D_VLM_Spatial/preprocess/m3d_outputs \
  --output-jsonl 3D_VLM_Spatial/preprocess/m3d_processed.jsonl \
  --depth 32 --height 256 --width 256
```

### Phase 2: Inference

Each model has a unified inference interface:

**Common Input Format** (JSONL):
```json
{
  "image_id": "valid_001_a_1",
  "image_path": "/PATH/TO/volume.nii.gz",
  "question": "Which lung contains the larger lesion?",
  "answer": "Right lung"
}
```

**Common Output Format** (JSONL):
```json
{
  "image_id": "valid_001_a_1",
  "question": "Which lung contains the larger lesion?",
  "answer": "Right lung",
  "prediction": "The right lung has the larger lesion",
  "model": "med3dvlm",
  "timestamp": "2026-02-27T10:30:45"
}
```

**Example**: Run Med3DVLM inference

```bash
python benchmarking/inference/med3dvlm/run_custom_eval.py \
  --dataset 3D_VLM_Spatial/preprocess/med3dvlm_processed.jsonl \
  --model-path checkpoints/med3dvlm-ft-spatial \
  --output-dir 3D_VLM_Spatial/reports/predictions \
  --output-name med3dvlm_predictions_full.jsonl \
  --batch-size 4 \
  --device cuda:0 \
  --num-workers 4
```

### Phase 3: Evaluation

Three independent LLM judges evaluate each prediction:

#### **Step 1: LLM Evaluation**

```bash
# GPT-4o evaluation
python 3D_VLM_Spatial/scripts/evaluate_with_gpt.py \
  --predictions 3D_VLM_Spatial/reports/predictions/med3dvlm_predictions_full.jsonl \
  --output 3D_VLM_Spatial/reports/llm_eval/med3dvlm_gpt_eval.json \
  --model gpt-4o-mini \
  --batch-size 100 \
  --prompt-file prompts/gpt_jury_prompt.txt

# Gemini 2.5 Flash evaluation
python 3D_VLM_Spatial/scripts/evaluate_with_gemini.py \
  --predictions 3D_VLM_Spatial/reports/predictions/med3dvlm_predictions_full.jsonl \
  --output 3D_VLM_Spatial/reports/llm_eval/med3dvlm_gemini_eval.json \
  --model models/gemini-2.5-flash \
  --batch-size 50

# Qwen-Plus evaluation
python 3D_VLM_Spatial/scripts/evaluate_with_qwen.py \
  --predictions 3D_VLM_Spatial/reports/predictions/med3dvlm_predictions_full.jsonl \
  --output 3D_VLM_Spatial/reports/llm_eval/med3dvlm_qwen_eval.json \
  --model qwen-plus \
  --batch-size 50
```

#### **Step 2: Jury Aggregation**

```bash
# Aggregate three judges into final accuracy
python 3D_VLM_Spatial/scripts/jury_aggregator.py \
  --gpt-eval 3D_VLM_Spatial/reports/llm_eval/med3dvlm_gpt_eval.json \
  --gemini-eval 3D_VLM_Spatial/reports/llm_eval/med3dvlm_gemini_eval.json \
  --qwen-eval 3D_VLM_Spatial/reports/llm_eval/med3dvlm_qwen_eval.json \
  --output 3D_VLM_Spatial/reports/llm_eval/med3dvlm_jury_verdict.json \
  --aggregation-method majority_vote
```

**Jury Logic**:
- **Majority Voting**: ≥2/3 judges agree → correct
- **Strict Consensus**: 3/3 judges agree → high-confidence correct
- **Per-Judge Scores**: Individual judge accuracy also tracked

#### **Step 3: Build Matrices**

```bash
# Create cross-model correctness matrix
python 3D_VLM_Spatial/scripts/build_correctness_matrix.py \
  --reports-dir 3D_VLM_Spatial/reports/llm_eval \
  --pattern "*_jury_verdict.json" \
  --output 3D_VLM_Spatial/reports/metrics/correctness_matrix_avg3.csv \
  --include-judges gpt,gemini,qwen

# Per-category performance
python 3D_VLM_Spatial/scripts/analyze_categories.py \
  --dataset 3D_VLM_Spatial/QA_generation/spatial_qa_filtered_full_with_categories.json \
  --results 3D_VLM_Spatial/reports/llm_eval \
  --output 3D_VLM_Spatial/reports/metrics/category_performance.csv
```

### Phase 4: Analysis & Visualization

```bash
# Plot answer length distributions
python 3D_VLM_Spatial/scripts/plot_answer_length_distributions.py \
  --dataset 3D_VLM_Spatial/QA_generation/spatial_qa_filtered_full.json \
  --predictions-dir 3D_VLM_Spatial/reports/predictions \
  --pred-glob "*_predictions_full.jsonl" \
  --output 3D_VLM_Spatial/reports/plots/answer_length_distributions.png

# Generate category radar plot
python 3D_VLM_Spatial/scripts/generate_radar_plot.py \
  --metrics 3D_VLM_Spatial/reports/metrics/category_performance.csv \
  --output 3D_VLM_Spatial/reports/plots/category_radar.pdf

# Model comparison heatmap
python 3D_VLM_Spatial/scripts/generate_heatmap.py \
  --matrix 3D_VLM_Spatial/reports/metrics/correctness_matrix_avg3.csv \
  --output 3D_VLM_Spatial/reports/plots/model_heatmap.png
```

## 🔧 Setting Up Individual Models

### Prerequisites (All Models)

```bash
# Install Python 3.11+
python --version  # Should be 3.11+

# Create environment
conda create -n vl-med-3d python=3.11 -y
conda activate vl-med-3d

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.0 PIL numpy tqdm
```

### Med3DVLM

```bash
# Navigate to model folder
cd benchmarking/inference/med3dvlm

# Create environment snapshot (if cloning from HPC)
conda create --name med3dvlm --file env/environment.yml

# Or install dependencies manually
pip install -r env/requirements.txt

# Download model weights (requires Hugging Face token)
huggingface-cli download mirthai/med3dvlm --local-dir ./weights

# Run evaluation
python run_custom_eval.py \
  --dataset /path/to/dataset.jsonl \
  --model-path ./weights \
  --output-dir /path/to/output \
  --gpu-id 0
```

**Key Details**:
- Input: Stacked multi-view slices or 3D volumes
- Output: Free-form text answers
- Memory: ~16GB VRAM for batch_size=4
- Runtime: ~2 hours for 9,077 samples on single A100

### M3D (3D ViT)

```bash
cd benchmarking/inference/m3d

# Install specialized dependencies
pip install -r env/requirements.txt

# Preprocess data to .npy format (256×256×32)
python 3D_VLM_Spatial/preprocess/preprocess_m3d.py --input-jsonl ...

# Run inference
python run_m3d_eval.py \
  --dataset /path/to/preprocessed.jsonl \
  --npy-root preprocess/m3d_outputs \
  --output-dir /path/to/output \
  --batch-size 8 \
  --gpu-id 0,1  # Can use multiple GPUs
```

**Key Details**:
- Input: 256×256×32 .npy files (float32, min-max normalized)
- Output: Semantic embeddings → fed to decoder LLM
- Memory: ~20GB for batch_size=8
- Runtime: ~1.5 hours on dual A100

### CT-Chat

```bash
cd benchmarking/inference/ct-chat

# Install dependencies
pip install -r env/requirements.txt

# Download model (multimodal adapter required)
git clone https://github.com/ibrahimethemhamamci/CT-CHAT.git
cp -r CT-CHAT/weights ./

# Run evaluation
python run_ctchat_eval.py \
  --dataset /path/to/dataset.jsonl \
  --model-path ./weights \
  --slice-extraction axial,coronal,sagittal \
  --num-slices 10 \
  --output-dir /path/to/output
```

**Key Details**:
- Input: Multi-view slice selection from 3D volume
- Slice extraction: Extracts representative slices per view
- Output: Free-form text
- Memory: ~24GB VRAM
- Runtime: ~3 hours (includes slice extraction)

### Merlin

```bash
cd benchmarking/inference/merlin

# Install via pip (official package)
pip install merlin-vlm

# Or clone repository
pip install -r repo/requirements.txt

# Run evaluation
python run_merlin_eval.py \
  --dataset /path/to/dataset.jsonl \
  --model-name merlin_large \
  --output-dir /path/to/output \
  --report-style spatial
```

**Key Details**:
- Input: Multi-view CT slices + metadata
- Architecture: Separate encoders for images + metadata
- Output: Report-level + QA-level predictions
- Memory: ~32GB VRAM
- Runtime: ~4 hours

### VILA-M3 + VISTA3D

```bash
cd benchmarking/inference/vila-m3

# Install dependencies
pip install -r env/requirements.txt

# VISTA3D automatically generates segmentations
# (Part of the VILA-M3 framework)

python run_vista3d_eval.py \
  --dataset /path/to/dataset.jsonl \
  --nifti-root /path/to/nifti \
  --output-dir /path/to/output \
  --use-segmentation True \
  --segmentation-overlay-alpha 0.3
```

**Key Details**:
- Input: 3D NIfTI volumes
- VISTA3D expert: Auto-generates organ segmentations
- Segmentation overlay: Combined with original image for VLM
- Output: Free-form text
- Memory: ~48GB VRAM (includes segmentation)
- Runtime: ~5 hours

### MedGemma-1.5

```bash
cd benchmarking/inference/medgemma

# Install dependencies
pip install -r env/requirements.txt

# Download model
pip install medgemma

python run_medgemma_eval.py \
  --dataset /path/to/dataset.jsonl \
  --image-embed-model clip-base \
  --text-only False \
  --output-dir /path/to/output \
  --quantization 4bit
```

**Key Details**:
- Input: Compressed image embeddings + text context
- 4-bit quantization: Reduces VRAM footprint
- Output: Free-form text
- Memory: ~12GB VRAM (with quantization)
- Runtime: ~1 hour (fastest model)

## 📊 Evaluation Metrics

### LLM Jury Protocol

Each prediction is judged by three independent LLMs:

```
GPT-4o --------┐
               ├─→ Majority Vote / Strict Consensus
Gemini 2.5 ----┤
               ├─→ Jury Verdict (binary: correct/incorrect)
Qwen-Plus -----┘
```

**Jury Prompt Template**:

```
You are an expert radiologist evaluating a medical QA system.

Question: {question}
Ground Truth Answer: {reference_answer}
Model Prediction: {model_prediction}

Determine if the model's prediction is medically correct and addresses the question.
Respond with only: YES (correct) or NO (incorrect)
```

**Additional Metrics**:

1. **SBERT Similarity**: Semantic embedding similarity (0-1)
2. **BLEU-4**: Lexical overlap (0-1)
3. **ROUGE-L**: Longest common subsequence (0-1)
4. **Exact Match**: Case-insensitive string equality (0-1)

### Output Matrices

**Correctness Matrix** (`correctness_matrix_avg3.csv`):
```
Model,GPT_Acc,Gemini_Acc,Qwen_Acc,Jury_Acc_Majority,Jury_Acc_Strict
Med3DVLM,0.52,0.51,0.49,0.51,0.38
CT-Chat,0.48,0.47,0.46,0.47,0.35
M3D,0.55,0.54,0.53,0.54,0.42
RadFM,0.45,0.44,0.43,0.44,0.31
...
```

**Category Performance** (`category_performance.csv`):
```
Model,Laterality,Vertical,Depth,Centricity,Adjacency,Extent
Med3DVLM,0.68,0.49,0.38,0.55,0.44,0.51
M3D,0.71,0.51,0.42,0.58,0.47,0.54
...
```

## 🚀 Running the Full Benchmarking Pipeline

**Complete End-to-End Command**:

```bash
#!/bin/bash

# Configuration
DATASET_DIR="3D_VLM_Spatial/QA_generation"
PREPROCESS_DIR="3D_VLM_Spatial/preprocess"
OUTPUT_DIR="3D_VLM_Spatial/reports"
MODELS=("med3dvlm" "m3d" "ct-chat" "merlin" "radfm" "medgemma" "vila-m3" "medevalkit")

# Phase 1: Preprocess all models
for model in "${MODELS[@]}"; do
  echo "Preprocessing for $model..."
  python "$PREPROCESS_DIR/preprocess_${model}.py" \
    --input-jsonl "$DATASET_DIR/spatial_qa_filtered_full.json" \
    --output-jsonl "$PREPROCESS_DIR/${model}_processed.jsonl"
done

# Phase 2: Run inference for all models
for model in "${MODELS[@]}"; do
  echo "Running inference for $model..."
  python "benchmarking/inference/${model}/run_${model}_eval.py" \
    --dataset "$PREPROCESS_DIR/${model}_processed.jsonl" \
    --output-dir "$OUTPUT_DIR/predictions"
done

# Phase 3: Run LLM evaluation
for model in "${MODELS[@]}"; do
  echo "Evaluating $model with LLM judges..."
  python 3D_VLM_Spatial/scripts/evaluate_with_gpt.py \
    --predictions "$OUTPUT_DIR/predictions/${model}_predictions_full.jsonl" \
    --output "$OUTPUT_DIR/llm_eval/${model}_gpt_eval.json"
  python 3D_VLM_Spatial/scripts/evaluate_with_gemini.py \
    --predictions "$OUTPUT_DIR/predictions/${model}_predictions_full.jsonl" \
    --output "$OUTPUT_DIR/llm_eval/${model}_gemini_eval.json"
  python 3D_VLM_Spatial/scripts/evaluate_with_qwen.py \
    --predictions "$OUTPUT_DIR/predictions/${model}_predictions_full.jsonl" \
    --output "$OUTPUT_DIR/llm_eval/${model}_qwen_eval.json"
done

# Phase 4: Aggregate and analyze
python 3D_VLM_Spatial/scripts/build_correctness_matrix.py \
  --reports-dir "$OUTPUT_DIR/llm_eval" \
  --pattern "*_jury_verdict.json" \
  --output "$OUTPUT_DIR/metrics/correctness_matrix_avg3.csv"

python 3D_VLM_Spatial/scripts/analyze_categories.py \
  --results "$OUTPUT_DIR/llm_eval" \
  --output "$OUTPUT_DIR/metrics/category_performance.csv"

echo "Benchmarking complete!"
```

## 🔐 API Keys & Credentials

**Required Environment Variables**:

```bash
# OpenAI (GPT-4o)
export OPENAI_API_KEY="sk-..."

# Google Gemini
export GOOGLE_API_KEY="AIzaSy..."

# Alibaba DashScope (Qwen)
export DASHSCOPE_API_KEY="sk-..."

# Hugging Face (model downloads)
export HF_TOKEN="hf_..."

# Optional: Ray cluster configuration
export RAY_ADDRESS="ray://localhost:6379"
```

## 📈 Benchmarking Results

Results are stored in modular outputs:

```
3D_VLM_Spatial/reports/
├── predictions/           # Raw model outputs
│   ├── med3dvlm_predictions_full.jsonl
│   ├── m3d_predictions_full.jsonl
│   └── ...
├── llm_eval/             # LLM judge verdicts
│   ├── med3dvlm_gpt_eval.json
│   ├── med3dvlm_gemini_eval.json
│   ├── med3dvlm_qwen_eval.json
│   └── ...
├── metrics/              # Aggregated metrics
│   ├── correctness_matrix_avg3.csv
│   ├── category_performance.csv
│   └── text_metrics.csv
└── plots/                # Visualizations
    ├── answer_length_dist.png
    ├── category_radar.pdf
    └── model_heatmap.png
```

## 🛟 Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
--batch-size 1  # Instead of 4 or 8

# Use gradient checkpointing (if supported)
--use-checkpointing True

# Quantize model weights
--quantization 4bit  # or 8bit
```

### Slow Inference

```bash
# Increase number of workers
--num-workers 8  # instead of 4

# Use multiple GPUs
--device cuda:0,1,2,3
--distributed-backend nccl

# Enable mixed precision
--mixed-precision True
```

### API Rate Limits (LLM eval)

```bash
# Increase batch timeout
--batch-timeout 60  # seconds

# Add retry logic with exponential backoff
--max-retries 3
--retry-backoff-factor 2.0

# Stagger requests across multiple processes
python 3D_VLM_Spatial/scripts/evaluate_with_gpt.py ... --num-threads 4
```

## 📚 References

- [Med3DVLM](https://arxiv.org/abs/2503.20047)
- [M3D-LaMed](https://arxiv.org/abs/2404.08039)
- [CT-Chat](https://arxiv.org/abs/2405.03841)
- [Merlin](https://arxiv.org/abs/2406.06512)
- [RadFM](https://arxiv.org/abs/2511.18876)
- [VILA-M3](https://openaccess.thecvf.com/content/CVPR2025/papers/Nath_VILA-M3_Enhancing_Vision-Language_Models_with_Medical_Expert_Knowledge_CVPR_2025_paper)

---

**Last Updated**: February 27, 2026
