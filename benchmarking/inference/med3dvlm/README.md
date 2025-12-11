# Med3DVLM

- **GitHub**: https://github.com/mirthai/med3dvlm
- **Paper**: https://arxiv.org/abs/2503.20047
- **Primary Tasks**: image-text retrieval, radiology report generation, visual question answering (open/closed).

## Repo Snapshot

- Upstream repo tracked as a git submodule inside `repo/` (currently pointing to the latest default branch of `mirthai/med3dvlm`).
- Use `git submodule update --init --recursive` after cloning this repo to pull the source.

## To-Do

- [ ] Sync `repo/` and decide whether to keep full project or extract lean inference modules here.
- [ ] Document required checkpoints and where to fetch them on the HPC.
- [ ] Create 10-sample JSONL and verify inference locally on HPC GPUs.
- [ ] Export and store frozen `env/requirements.txt`.

## Notes

- Focus evaluation on spatial metrics such as position, depth, and distance reasoning per the paper.
- Expect multi-modal inputs (3D images + text); ensure loaders handle volumetric formats (e.g., NIfTI).
