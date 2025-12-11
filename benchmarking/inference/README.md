# 3D Medical VLM Benchmarking

This folder gathers everything required to run spatial-understanding benchmarks for 3D medical vision-language models. Code will live in this repo for convenience, while heavyweight datasets, checkpoints, and all benchmarking runs will execute on the HPC cluster.

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
  "image_path": "/cluster/path/to/case001.nii.gz",
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

| Model | GitHub | Paper | HPC Env | Sample Run | Notes |
| --- | --- | --- | --- | --- | --- |
| Med3DVLM | https://github.com/mirthai/med3dvlm | https://arxiv.org/abs/2503.20047 | ☐ | ☐ | VQA + retrieval |
| MS-VLM | _pending_ | https://arxiv.org/pdf/2412.13558 | ☐ | ☐ | need repo/source |
| Merlin | https://github.com/StanfordMIMI/Merlin | https://pmc.ncbi.nlm.nih.gov/articles/PMC11230513/ | ☐ | ☐ | multi-task radiology |
| BrainMD | https://github.com/YuliWanghust/BrainMD | https://openreview.net/pdf?id=JrJW21IP9p | ☐ | ☐ | MRI-focused |
| DeepTumorVQA | https://github.com/Schuture/DeepTumorVQA | https://arxiv.org/pdf/2505.18915 | ☐ | ☐ | detailed spatial QA |
| VILA-M3 | _pending_ | https://openaccess.thecvf.com/content/CVPR2025/papers/Nath_VILA-M3_Enhancing_Vision-Language_Models_with_Medical_Expert_Knowledge_CVPR_2025_paper | ☐ | ☐ | needs code release |
| E3D-GPT | _pending_ | https://arxiv.org/pdf/2410.14200 | ☐ | ☐ | grounding heavy |
| FILA | https://github.com/alibaba-damo-academy/fvlm | https://openreview.net/pdf?id=nYpPAT4L3D | ☐ | ☐ | aka FVLM |

Update the checkboxes once the HPC environments are validated and the 10-sample run succeeds.
