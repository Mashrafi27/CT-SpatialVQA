# BrainMD

- **GitHub**: https://github.com/YuliWanghust/BrainMD
- **Paper**: https://openreview.net/pdf?id=JrJW21IP9p
- **Primary Tasks**: brain tumor diagnosis, MRI VQA, report interpretation.

## Repo Snapshot

- Upstream `YuliWanghust/BrainMD` pulled into `repo/` as a submodule (remember `git submodule update --init --recursive`).
- Capture and document the commit hash we benchmark against.

## Setup Checklist

- [ ] Update `repo/` to the target commit and note pretrained checkpoint requirements (likely large, stored on HPC only).
- [ ] Prepare MRI loaders that match our JSONL schema (probably NIfTI or DICOM volumes).
- [ ] Validate inference with 10 MRI-question pairs covering tumors/anatomy context.
- [ ] Freeze env details after HPC validation.

## Notes

Pay attention to GPU memory consumption; BrainMD may expect 3D convolutions with large volumes.
