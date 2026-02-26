# 3D Medical VLM Benchmarking

This folder gathers everything required to run spatial-understanding benchmarks for 3D medical vision-language models. Code lives in this repo for convenience, while heavyweight datasets and checkpoints can remain on external storage/HPC.

## Repository Structure

- `benchmarking/inference/<model>/` – model-specific inference harness.
  - `README.md` – model status, setup notes, and run instructions.
  - `env/` (optional) – conda/env files captured from the HPC once validated.
  - `scripts/` or wrapper files needed to launch inference from the HPC queue.
- `benchmarking/inference/README.md` – this file; contains global workflow, dataset schema, and tracking table.

## Benchmark Workflow

1. **Clone/Prepare Code** – either drop lightweight inference scripts in the model folder or clone the upstream repo in-place if custom pipelines are required. Keep only what is necessary for inference.
2. **Create HPC Environment** – spin up an isolated env per model on the cluster, install dependencies, and record exact versions. After a successful dry-run, export `requirements.txt` (or `environment.yml`) back into the matching folder under `env/`.
3. **Sanity Dataset** – craft a small diagnostic split (≈10 image-question-answer items) to validate the pipeline. Store pointers/format here; the data itself remains on HPC storage.
4. **Run + Log** – execute inference, capture command, config, commit hash, GPU needs, and any quirks in the per-model README.
5. **Freeze** – once stable, lock the env, scripts, and documentation so future benchmarking only requires syncing the dataset/questions and launching the provided script.

## Sample Input Format

All inference scripts should accept a JSONL with the following fields per line:

```json
{
  "image_path": "/PATH/TO/PROJECT/data_volumes/dataset/valid_fixed/case001.nii.gz",
  "question": "Which kidney contains the larger lesion volume?",
  "answer": "Right kidney",
  "meta": {
    "study_type": "CT",
    "notes": "10 mm slice thickness"
  }
}
```

Keep loaders flexible (supporting CSV, JSONL, or folder lists), but JSONL will be the shared canonical format.

## Model Tracking

| Model | Paper | Notes |
| --- | --- | --- |
| Med3DVLM | https://arxiv.org/abs/2503.20047 | VQA + retrieval |
| CT-Chat | https://arxiv.org/abs/2405.03841 | multi-view slices |
| M3D | https://arxiv.org/abs/2404.08039 | 3D ViT + LLaMA |
| RadFM | https://arxiv.org/abs/2511.18876 | foundation model |
| MedGemma-1.5 | https://arxiv.org/abs/2407.02546 | small LLM |
| Merlin | https://pmc.ncbi.nlm.nih.gov/articles/PMC11230513/ | multi-task radiology |
| VILA-M3 | https://openaccess.thecvf.com/content/CVPR2025/papers/Nath_VILA-M3_Enhancing_Vision-Language_Models_with_Medical_Expert_Knowledge_CVPR_2025_paper | segmentation-augmented |
| MedEvalKit | https://arxiv.org/abs/2406.01633 | evaluation suite |

Update the checkboxes once the HPC environments are validated and the 10-sample run succeeds.
