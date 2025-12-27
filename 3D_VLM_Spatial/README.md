# 3D VLM Spatial Benchmark

This folder contains the spatial QA dataset, download scripts, and filtering utilities used to evaluate Med3DVLM and other models.

## Dataset Layout

- `dataset/`: scripts + helpers to download the CT-RATE volumes. After running `download_dataset.py`, NIfTI files are stored under `data_volumes/<split>/<case>/...`.
- `reports/`: JSON/metrics generated during annotation and Gemini validation (e.g., `gemini_judgments.json`, `gemini_human_disagreements.json`).
- `spatial_qa_output.json`: master QA list produced from findings/impressions (contains `image_path`, `question`, `answer`). Answers are ground-truth references and are not passed to the models.

Each QA entry has:

```json
{
  "image_path": "/abs/path/to/train_1_a_1.nii.gz",
  "question": "Which lung shows the larger lesion volume?",
  "answer": "Right lung"
}
```

## Filtering Script

Use `scripts/filter_qa_pairs.py` to remove QA pairs that Gemini judged as non-spatial or non-relevant:

```bash
python 3D_VLM_Spatial/scripts/filter_qa_pairs.py \
  --qa-json 3D_VLM_Spatial/spatial_qa_output.json \
  --judgments 3D_VLM_Spatial/reports/gemini_judgments.json \
  --output 3D_VLM_Spatial/spatial_qa_filtered.json
```

Flags `--keep-nonspatial` or `--keep-nonrelevant` can be used if you only want to filter one category.

The resulting JSON can be fed to `benchmarking/inference/med3dvlm/run_custom_eval.py` (or other model scripts) using the absolute `image_path` references. Keep the answers for evaluation only—they are not supplied to the models during inference.

## Convert to JSONL

Some evaluators expect a flat JSONL (one QA per line). Use `scripts/to_jsonl.py` to transform the filtered JSON into that format and optionally prepend the image root:

```bash
python 3D_VLM_Spatial/scripts/to_jsonl.py \
  --input 3D_VLM_Spatial/spatial_qa_filtered.json \
  --output 3D_VLM_Spatial/spatial_qa_filtered.jsonl \
  --image-root 3D_VLM_Spatial/dataset/data_volumes/dataset/valid_fixed
```

The JSONL records will include the resolved `image_path`, `question`, and `answer`, ready for `run_custom_eval.py --dataset ...jsonl`.

## Evaluate Predictions

After running a model, you can compute simple exact-match accuracy against the ground truth answers using `scripts/evaluate_predictions.py`:

```bash
python 3D_VLM_Spatial/scripts/evaluate_predictions.py \
  --predictions 3D_VLM_Spatial/reports/med3dvlm_predictions.jsonl \
  --report 3D_VLM_Spatial/reports/med3dvlm_eval.json \
  --show-mismatches
```

The script prints `{total, correct, accuracy}` and optionally logs sample mismatches. Adjust the path to point at any other model’s prediction JSONL.
