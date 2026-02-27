# 3D VLM Spatial Benchmark

This folder contains the CT-SpatialVQA dataset, download scripts, preprocessing utilities, and evaluation tools used to benchmark 3D medical VLMs.

## Dataset Layout

- `dataset/`: scripts + helpers to download CT-RATE volumes. After running `download_dataset.py`, files are stored under `dataset/valid_fixed/`.
- `qa_generation_v2/`: finalized QA datasets used for benchmarking.
  - `spatial_qa_filtered_full.json` (master QA list)
  - `spatial_qa_filtered_full_nifti_*.jsonl` (model-specific inputs)
- `Spatial_categories/`: spatial category assignments for analysis.
- `reports/`: modularized outputs:
  - `predictions/`, `llm_eval/`, `metrics/`, `analysis/`, `plots/`, `human_eval/`

## QA Example

Each QA entry has:

```json
{
  "image_path": "/PATH/TO/PROJECT/data_volumes/dataset/valid_fixed/case001.nii.gz",
  "question": "Which lung shows the larger lesion volume?",
  "answer": "Right lung"
}
```

## Preprocessing

Example: M3D preprocessing (resize to 256×256×32, normalize, save as .npy)

```bash
python 3D_VLM_Spatial/preprocess/preprocess_m3d.py \
  --input-jsonl 3D_VLM_Spatial/qa_generation_v2/spatial_qa_filtered_full_nifti_m3d.jsonl \
  --nifti-root data_volumes/dataset/valid_fixed \
  --output-root 3D_VLM_Spatial/preprocess/m3d_outputs \
  --output-jsonl 3D_VLM_Spatial/preprocess/m3d_processed.jsonl
```

## LLM-Based Evaluation

```bash
python 3D_VLM_Spatial/scripts/evaluate_with_gemini.py \
  --predictions 3D_VLM_Spatial/reports/predictions/med3dvlm_predictions_full.jsonl \
  --output 3D_VLM_Spatial/reports/llm_eval/med3dvlm_gemini2.5_eval_full.json \
  --model models/gemini-2.5-flash

python 3D_VLM_Spatial/scripts/evaluate_with_gpt.py \
  --predictions 3D_VLM_Spatial/reports/predictions/med3dvlm_predictions_full.jsonl \
  --output 3D_VLM_Spatial/reports/llm_eval/med3dvlm_gpt_eval.json \
  --model gpt-4o-mini

python 3D_VLM_Spatial/scripts/evaluate_with_qwen.py \
  --predictions 3D_VLM_Spatial/reports/predictions/med3dvlm_predictions_full.jsonl \
  --output 3D_VLM_Spatial/reports/llm_eval/med3dvlm_qwen_eval.json \
  --model qwen-plus
```

## Cross-Model Aggregation

```bash
python 3D_VLM_Spatial/scripts/build_correctness_matrix.py \
  --reports-dir 3D_VLM_Spatial/reports/llm_eval \
  --pattern "*_eval*.json" \
  --output 3D_VLM_Spatial/reports/metrics/correctness_matrix_avg3.csv
```
