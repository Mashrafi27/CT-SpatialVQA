# Med3DVLM

- **GitHub**: https://github.com/mirthai/med3dvlm
- **Paper**: https://arxiv.org/abs/2503.20047
- **Primary Tasks**: image-text retrieval, radiology report generation, visual question answering (open/closed).

## Repo Snapshot

- Upstream repo tracked as a git submodule inside `Med3DVLM/` (pointing to the latest default branch of `mirthai/med3dvlm`).
- Use `git submodule update --init --recursive` after cloning this repo (or syncing on the HPC) to pull the source and note the commit hash here.

## To-Do

- [ ] Sync the `Med3DVLM` submodule and decide whether to keep the full project or extract lean inference modules here.
- [ ] Document required checkpoints and where to fetch them on the HPC.
- [ ] Create 10-sample JSONL and verify inference locally on HPC GPUs.
- [ ] Export and store frozen `env/requirements.txt`.

## Notes

- Focus evaluation on spatial metrics such as position, depth, and distance reasoning per the paper.
- Expect multi-modal inputs (3D images + text); ensure loaders handle volumetric formats (e.g., NIfTI).
