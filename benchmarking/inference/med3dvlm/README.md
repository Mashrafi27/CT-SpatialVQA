# Med3DVLM

- **GitHub**: https://github.com/mirthai/med3dvlm
- **Paper**: https://arxiv.org/abs/2503.20047
- **Primary Tasks**: image-text retrieval, radiology report generation, visual question answering (open/closed).

## Repo Snapshot

- The upstream Med3DVLM repository should be cloned manually on the HPC under `benchmarking/inference/med3dvlm/Med3DVLM` (this repo does not track it as a submodule).
- Record the commit hash + any local patches in this README so others can reproduce the exact checkout.

## To-Do

- [ ] Sync the external `Med3DVLM` clone (ensure the HPC path above is up to date) and decide whether to keep the full project or extract lean inference modules here.
- [ ] Document required checkpoints and where to fetch them on the HPC.
- [ ] Create 10-sample JSONL and verify inference locally on HPC GPUs.
- [ ] Export and store frozen `env/requirements.txt`.

## Notes

- Focus evaluation on spatial metrics such as position, depth, and distance reasoning per the paper.
- Expect multi-modal inputs (3D images + text); ensure loaders handle volumetric formats (e.g., NIfTI).
