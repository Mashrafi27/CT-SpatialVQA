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

## Custom Evaluation Script

We provide `run_custom_eval.py` to run Med3DVLM on the spatial benchmark JSONL on either local machines or the HPC. Example usage:

```bash
cd benchmarking/inference/med3dvlm
python run_custom_eval.py \
  --dataset ../../../3D_VLM_Spatial/reports/my_spatial_eval.jsonl \
  --output ../../../3D_VLM_Spatial/reports/med3dvlm_predictions.jsonl \
  --model-path MagicXin/Med3DVLM-Qwen-2.5-7B \
  --image-root ../../../3D_VLM_Spatial/dataset/data_volumes/dataset/valid_fixed \
  --device cuda:0 \
  --dtype bfloat16
```

Inputs must contain `image_path` (either absolute, or relative to `--image-root`) and `question`. The script adds `<im_patch>` tokens automatically and stores predictions as JSONL with the original metadata. `--model-path` can point to a local directory (`Med3DVLM/models/Med3DVLM-Qwen-2.5-7B`) or a Hugging Face repo id (`MagicXin/Med3DVLM-Qwen-2.5-7B`).
