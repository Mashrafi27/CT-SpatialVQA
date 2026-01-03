# FILA (FVLM)

- **GitHub**: https://github.com/alibaba-damo-academy/fvlm
- **Paper**: https://openreview.net/pdf?id=nYpPAT4L3D
- **Primary Tasks**: (per paper) multimodal reasoning; adapt to our VQA + spatial tasks.

## Repo Snapshot

- `repo/` holds the upstream `alibaba-damo-academy/fvlm` project as a git submodule (requires `git submodule update --init --recursive`).
- Once we pick a working commit/branch, record it here to keep future runs reproducible.

## Setup Checklist

- [ ] Update/sync the FVLM submodule and extract inference components relevant to our dataset.
- [ ] Determine if existing checkpoints cover 3D modalities or if projections are required.
- [ ] Validate on the 10-sample set and capture command lines/configs.
- [ ] Export frozen requirements/environment information post-validation.

## Notes

Alibaba often distributes large checkpoints separately; be ready to document the exact download/placement process (HPC only).

## Plan to integrate with our benchmark

- Use our resampled CTs and QA JSONL. Start with center-slice evaluation to get a baseline; expand to multi-slice/3D if feasible.
- Drop FILA weights (from their Google Drive link) into the HPC; also place the ViT-B MAE weights and BiomedVLP-CXR-BERT-specialized text encoder in `repo/data` as expected by their scripts.
- Implement a custom evaluator (similar to `med3dvlm/run_custom_eval.py`) that:
  - Reads our QA JSONL,
  - Resolves each `case_id` to a NIfTI path and picks a slice (configurable),
  - Runs FILA and writes `3D_VLM_Spatial/reports/fila_predictions.jsonl`.
- Keep this repo clone clean locally; heavy files and checkpoints stay on the HPC and out of git.
