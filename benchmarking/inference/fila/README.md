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
