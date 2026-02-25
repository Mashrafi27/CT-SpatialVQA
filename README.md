# Benchmarking 3D Medical Vision-Language Models for Spatial Understanding

> **A comprehensive evaluation framework for assessing spatial reasoning capabilities in 3D medical vision-language models using volumetric CT imaging.**

![Paper Status](https://img.shields.io/badge/Status-MICCAI%202026-blue)
![Dataset](https://img.shields.io/badge/Dataset-CT--RATE-green)
![Models](https://img.shields.io/badge/Models-8%2B-orange)

## 📋 Overview

This repository contains the evaluation framework, dataset, and benchmarking code for assessing how well 3D medical vision-language models (VLMs) understand spatial relationships in volumetric CT imaging. We introduce a curated **spatial QA benchmark** that evaluates models on their ability to reason about anatomical positions, distances, laterality, and inter-organ relationships—critical capabilities for clinical deployment.

### Key Contributions

- **Spatial QA Benchmark**: A curated dataset of ~1,000 spatial reasoning questions derived from CT radiology reports with LLM-validated ground truth
- **Multi-Model Evaluation**: Comprehensive evaluation of 8+ state-of-the-art 3D medical VLMs including Med3DVLM, CT-Chat, M3D, RadFM, MedGemma, Merlin, VILA-M3, and others
- **Spatial Classification Framework**: Automated spatial vs. non-spatial question filtering using LLM judges (Gemini 2.0)
- **Flexible Benchmarking Pipeline**: Modular evaluation infrastructure supporting multiple models, input formats, and evaluation metrics
- **Reproducibility**: Complete preprocessing, inference, and evaluation scripts with detailed documentation

## 🎯 Research Questions

This work addresses:

1. **Do 3D medical VLMs understand spatial relationships?** How well do they localize findings and reason about anatomical positions?
2. **What types of spatial reasoning are challenging?** Laterality, distance comparison, volumetric assessment, or complex multi-organ reasoning?
3. **How do different model architectures compare** on spatial understanding despite similar overall medical VLM capabilities?
4. **What role does input representation play?** How do different 3D slicing strategies (axial, coronal, sagittal) affect spatial reasoning?

## 📊 Dataset

### CT-RATE Source

We utilize the **CT-RATE dataset** (CT Radiology Assessment Training Exam), a comprehensive collection of CT scans with expert radiologist reports. The dataset provides:

- **~130 CT volumes** from various anatomical regions and pathologies
- **Structured radiology reports** with findings and impressions sections
- **Diverse pathologies** including tumors, infections, structural abnormalities
- **English language reports** with detailed anatomical descriptions

### Spatial QA Pairs

From each case's findings and impressions, we generate spatial question-answer pairs targeting:

- **Laterality reasoning**: "Which lung contains the lesion?"
- **Distance & proximity**: "What is the spatial relationship between the nodule and the pleura?"
- **Volumetric assessment**: "Which lesion is larger in volume?"
- **Anatomical localization**: "In which segment of the liver is the lesion located?"
- **Multi-organ comparisons**: "Is the right kidney larger than the left?"

### Quality Assurance

All QA pairs are validated using **Google Gemini 2.0 Flash** with custom prompts to ensure:

- **Spatial relevance**: Questions genuinely require spatial reasoning about anatomical location, orientation, or relative position
- **Ground truth validity**: Answers are inferable from the imaging findings and radiologist annotations
- **Non-redundancy**: Removal of textual paraphrasing or general knowledge questions

**Final dataset**: ~1,000 high-quality spatial QA pairs with validated ground truth

## 🏗️ Repository Structure

```
MICCAI2026-3DMedVLMS/
├── 3D_VLM_Spatial/                    # Core benchmarking framework
│   ├── dataset/                       # CT-RATE download & validation
│   │   └── download_dataset.py       # Script to fetch CT volumes
│   │
│   ├── preprocess/                    # Model-specific preprocessing
│   │   ├── preprocess_m3d.py         # M3D: 256×256×32 NIfTI → .npy
│   │   ├── preprocess_med3dvlm.py    # Med3DVLM format
│   │   ├── preprocess_merlin.py      # Merlin preprocessing
│   │   ├── preprocess_radfm.py       # RadFM preprocessing
│   │   ├── preprocess_ctchat.py      # CT-Chat preprocessing
│   │   └── ...                       # Additional model preprocessing
│   │
│   ├── qa_generation_v2/             # QA pair generation pipeline
│   │   ├── qa_generation.py          # Generate QA from reports
│   │   ├── prepare_case_split.py     # Manage case sampling
│   │   └── spatial_qa_*.jsonl        # Generated QA datasets
│   │
│   ├── reports/                      # Generated reports & judgments
│   │   ├── gemini_judgments.json     # LLM spatial classification
│   │   ├── *_predictions_*.jsonl    # Model outputs
│   │   └── *_eval.json              # Evaluation metrics
│   │
│   ├── scripts/                      # Analysis & evaluation
│   │   ├── filter_qa_pairs.py       # Remove non-spatial QAs
│   │   ├── to_jsonl.py              # Convert JSON → JSONL
│   │   ├── evaluate_predictions.py  # Exact-match accuracy
│   │   ├── evaluate_with_gemini.py  # LLM-based evaluation
│   │   ├── evaluate_text_metrics.py # BLEU, ROUGE, METEOR
│   │   ├── evaluate_with_alignscore.py # Factual consistency
│   │   ├── build_correctness_matrix.py # Cross-model comparison
│   │   └── plot_answer_length_distributions.py # Analysis
│   │
│   ├── README.md                    # Detailed 3D_VLM_Spatial guide
│   ├── spatial_qa_output.json       # Master QA list
│   ├── spatial_qa_filtered.json     # After spatial filtering
│   └── validation_reports_output.json  # Radiology reports
│
├── benchmarking/                     # Model inference harness
│   ├── inference/
│   │   ├── med3dvlm/                 # Med3DVLM eval
│   │   │   ├── run_custom_eval.py   # Inference script
│   │   │   └── README.md
│   │   ├── m3d/                      # M3D eval
│   │   ├── merlin/                   # Merlin eval
│   │   ├── radfm/                    # RadFM eval
│   │   ├── ctchat/                   # CT-Chat eval
│   │   ├── medgemma/                 # MedGemma eval
│   │   ├── vila-m3/                  # VILA-M3 + VISTA3D eval
│   │   └── [other-models]/
│   └── README.md                     # Benchmarking pipeline overview
│
├── checkpoints/                      # Model weights (downloaded separately)
│   └── alignscore/                   # AlignScore weights
│
├── scripts/                          # Global utilities
│
├── Visualizations/                   # Generated plots & figures
│
└── [CSV files]                       # Model inventory & tracking
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS.git
cd MICCAI2026-3DMedVLMS

# Create conda environment
conda create -n vl-med-3d python=3.11 -y
conda activate vl-med-3d

# Install core dependencies
pip install -r requirements.txt

# Install model-specific dependencies (optional, per model)
cd benchmarking/inference/med3dvlm && pip install -r env/requirements.txt
```

### 2. Download Dataset

```bash
# Download CT-RATE volumes (requires access credentials)
python 3D_VLM_Spatial/dataset/download_dataset.py \
  --output-dir 3D_VLM_Spatial/dataset/data_volumes \
  --split valid_fixed
```

### 3. Generate/Filter QA Pairs

```bash
# Filter out non-spatial questions using Gemini
python 3D_VLM_Spatial/scripts/filter_qa_pairs.py \
  --qa-json 3D_VLM_Spatial/spatial_qa_output.json \
  --judgments 3D_VLM_Spatial/reports/gemini_judgments.json \
  --output 3D_VLM_Spatial/spatial_qa_filtered.json

# Convert to JSONL format for model inference
python 3D_VLM_Spatial/scripts/to_jsonl.py \
  --input 3D_VLM_Spatial/spatial_qa_filtered.json \
  --output 3D_VLM_Spatial/spatial_qa_filtered.jsonl \
  --image-root 3D_VLM_Spatial/dataset/data_volumes/dataset/valid_fixed
```

### 4. Preprocess Data for Model

Example: M3D model (resize to 256×256×32, normalize, save as .npy)

```bash
python 3D_VLM_Spatial/preprocess/preprocess_m3d.py \
  --input-jsonl 3D_VLM_Spatial/spatial_qa_filtered.jsonl \
  --nifti-root 3D_VLM_Spatial/dataset/data_volumes/dataset/valid_fixed \
  --output-root 3D_VLM_Spatial/preprocess/m3d_outputs \
  --output-jsonl 3D_VLM_Spatial/preprocess/m3d_processed.jsonl
```

### 5. Run Model Inference

Example: Med3DVLM inference

```bash
python benchmarking/inference/med3dvlm/run_custom_eval.py \
  --dataset 3D_VLM_Spatial/spatial_qa_filtered.jsonl \
  --model-path checkpoints/med3dvlm \
  --output-dir 3D_VLM_Spatial/reports \
  --batch-size 4 \
  --gpu-id 0
```

### 6. Evaluate Results

```bash
# Exact-match accuracy
python 3D_VLM_Spatial/scripts/evaluate_predictions.py \
  --predictions 3D_VLM_Spatial/reports/med3dvlm_predictions.jsonl \
  --report 3D_VLM_Spatial/reports/med3dvlm_eval.json \
  --show-mismatches

# LLM-based evaluation (clinically-aware scoring)
python 3D_VLM_Spatial/scripts/evaluate_with_gemini.py \
  --predictions 3D_VLM_Spatial/reports/med3dvlm_predictions.jsonl \
  --output 3D_VLM_Spatial/reports/med3dvlm_gemini_eval.json \
  --model models/gemini-2.0-flash

# AlignScore (factual consistency)
python 3D_VLM_Spatial/scripts/evaluate_with_alignscore.py \
  --predictions 3D_VLM_Spatial/reports/med3dvlm_predictions.jsonl \
  --output 3D_VLM_Spatial/reports/med3dvlm_alignscore.json

# Build cross-model comparison matrix
python 3D_VLM_Spatial/scripts/build_correctness_matrix.py \
  --reports-dir 3D_VLM_Spatial/reports \
  --pattern "*_predictions_*.jsonl" \
  --output 3D_VLM_Spatial/reports/correctness_matrix.csv
```

### 7. Visualize Results

```bash
# Generate answer length distributions
python 3D_VLM_Spatial/scripts/plot_answer_length_distributions.py \
  --dataset-json 3D_VLM_Spatial/spatial_qa_filtered.json \
  --predictions-dir 3D_VLM_Spatial/reports \
  --pred-glob "*_predictions_full.jsonl" \
  --output 3D_VLM_Spatial/reports/answer_length_distributions.png
```

## 📈 Evaluated Models

| Model | Architecture | Input | Spatial QA Acc. | Notes |
|-------|--------------|-------|-----------------|-------|
| **Med3DVLM** | CLIP + 3D CNN | 3D volumes | - | State-of-the-art 3D medical VLM |
| **CT-Chat** | Multimodal LLM | Multi-view slices | - | Radiologist-aligned VLM |
| **M3D-LaMed** | 3D ViT + LLaMA | 256×256×32 voxels | - | 3D spatial encoding |
| **RadFM** | Foundation Model | Radiograph patches | - | Radiology-specific pretraining |
| **MedGemma-1.5** | Small LLM | Text + 2D images | - | Efficient medical LLM |
| **Merlin** | Multi-task encoder | Multi-view CT | - | Report generation focused |
| **VILA-M3 + VISTA3D** | Hybrid expert system | VISTA3D segmentation | - | 3D segmentation + 2D vision-language |
| **[Additional models]** | - | - | - | Under evaluation |

> **Note**: Accuracy scores pending final evaluation runs. Check `3D_VLM_Spatial/reports/correctness_matrix.csv` for latest results.

## 🔬 Methodology

### QA Generation & Validation Pipeline

1. **Extract from Reports**: Parse findings and impressions sections from radiologist reports
2. **Generate Candidates**: Use templated prompts + LLM to generate spatial QA candidates
3. **LLM Filtering**: Gemini 2.0 validates:
   - **Spatial relevance**: Requires anatomical reasoning (position, distance, orientation)
   - **Ground truth validity**: Answerable from imaging findings alone
   - **Non-redundancy**: Not just textual paraphrasing
4. **Final Dataset**: ~1,000 validated spatial QA pairs

### Evaluation Metrics

- **Exact-Match Accuracy**: Normalized string matching (lowercase, punctuation-agnostic)
- **LLM-Based Scoring**: Gemini judges clinical correctness accounting for medical synonyms
- **Text Metrics**: BLEU, ROUGE-L, METEOR for answer quality
- **Factual Consistency**: AlignScore measures factual alignment between prediction and image
- **Cross-Model Analysis**: Confusion matrices, per-category performance, failure mode classification

### Input Representations Tested

- **3D Voxels**: Full volumetric encoding (preferred for spatial understanding)
- **Multi-view Slices**: Axial, coronal, sagittal projections (2D-friendly models)
- **Segmentation-Augmented**: VISTA3D expert segmentation overlays (VILA-M3)
- **Slice Grids**: Combined views in canvas format (visual context)

## 🛠️ Detailed Guides

For detailed setup and running instructions per model:

- [3D_VLM_Spatial Guide](3D_VLM_Spatial/README.md) – Dataset, filtering, evaluation
- [QA Generation v2 Guide](3D_VLM_Spatial/qa_generation_v2/README.md) – Generate new QA pairs
- [Benchmarking Guide](benchmarking/inference/README.md) – Model-specific inference setup
- Model-specific READMEs in `benchmarking/inference/<model>/`

## 📝 Data Formats

### Input: JSONL Format

```json
{
  "image_path": "/abs/path/to/case001.nii.gz",
  "question": "Which kidney contains the larger lesion?",
  "answer": "Right kidney"
}
```

### Output: Predictions JSONL

```json
{
  "image_path": "/abs/path/to/case001.nii.gz",
  "question": "Which kidney contains the larger lesion?",
  "answer": "Right kidney",
  "prediction": "The right kidney has a larger lesion",
  "correct": false
}
```

## 🔑 Key Features

### Reproducibility
- ✅ Complete preprocessing pipelines for each model
- ✅ Frozen dataset with validation splits
- ✅ Exact hyperparameters and model versions documented
- ✅ W&B logging for all benchmark runs

### Extensibility
- ✅ Modular model interfaces for easy addition of new models
- ✅ Pluggable evaluation metrics (add custom scorers)
- ✅ Flexible input format handling (JSONL, CSV, folder structures)

### Transparency
- ✅ LLM judge prompts fully documented
- ✅ Failure mode analysis and error categorization
- ✅ Per-question performance tracking

## 📚 Citations

If you use this benchmark in your work, please cite:

```bibtex
@inproceedings{MICCAI2026-3DMedVLMS,
  title={Benchmarking 3D Medical Vision-Language Models for Spatial Understanding},
  author={[Authors]},
  booktitle={Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2026}
}
```

**Key referenced works**:
- Med3DVLM: https://arxiv.org/abs/2503.20047
- DeepTumorVQA: https://arxiv.org/pdf/2505.18915
- Merlin: https://pmc.ncbi.nlm.nih.gov/articles/PMC11230513/
- VILA-M3: https://openaccess.thecvf.com/content/CVPR2025/papers/Nath_VILA-M3_Enhancing_Vision-Language_Models_with_Medical_Expert_Knowledge_CVPR_2025_paper
- RadFM: https://arxiv.org/abs/2511.18876
- AlignScore: https://arxiv.org/abs/2305.05252

## 🔐 Requirements & Dependencies

Primary dependencies:
- Python 3.11+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU inference)
- Google Generative AI SDK (for Gemini evaluation)
- SimpleITK (for NIfTI handling)
- tqdm, numpy, pandas

See `requirements.txt` for the full dependency list.

## 📋 Acknowledgments

- **CT-RATE Dataset** creators and radiologist annotators
- **Model Teams**: Med3DVLM, CT-Chat, M3D-LaMed, RadFM, Merlin, VILA-M3 developers
- **Evaluation Tools**: AlignScore authors, NLTK contributors
- **Infrastructure**: HPC cluster support for large-scale benchmarking

## 📧 Contact & Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact: [author email]
- Lab: [institution]

## 📄 License

This project is released under the [MIT/Apache 2.0] License. Individual model components maintain their original licenses (see respective `benchmarking/inference/<model>/` folders).

---

**Last Updated**: February 2026  
**Paper Status**: Under Review (MICCAI 2026)  
**Dataset Version**: v1.0 (Final)
