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

Inputs must contain `image_path` (either absolute, or relative to `--image-root`) and `question`. If the path looks like `valid_1_a_1.nii.gz`, the script automatically resolves it to `<root>/valid_1/valid_1_a/valid_1_a_1.nii.gz`. The script adds `<im_patch>` tokens automatically and stores predictions as JSONL with the original metadata. `--model-path` can point to a local directory (`Med3DVLM/models/Med3DVLM-Qwen-2.5-7B`) or a Hugging Face repo id (`MagicXin/Med3DVLM-Qwen-2.5-7B`).

## CT-RATE Preprocessing Workflow

Med3DVLM expects 128×256×256 tensors (like the M3D-Cap preparation). Use the helper scripts in this repo before running inference:

1. **Filter QA pairs** (optional but recommended):
   ```bash
   python 3D_VLM_Spatial/scripts/filter_qa_pairs.py \
     --qa-json 3D_VLM_Spatial/spatial_qa_output.json \
     --judgments 3D_VLM_Spatial/reports/gemini_judgments.json \
     --output 3D_VLM_Spatial/spatial_qa_filtered.json
   ```

2. **Convert nested JSON → JSONL**:
   ```bash
   python 3D_VLM_Spatial/scripts/to_jsonl.py \
     --input 3D_VLM_Spatial/spatial_qa_filtered.json \
     --output 3D_VLM_Spatial/spatial_qa_filtered.jsonl
   ```

3. **Preprocess CT volumes** (resample + normalize):
   ```bash
   cd benchmarking/inference/med3dvlm
   python preprocess_ctrate.py \
     --input-jsonl ../../3D_VLM_Spatial/spatial_qa_filtered.jsonl \
     --image-root ../../3D_VLM_Spatial/dataset/data_volumes/dataset/valid_fixed \
     --output-root ../../3D_VLM_Spatial/processed_volumes \
     --output-jsonl ../../3D_VLM_Spatial/spatial_qa_processed.jsonl
   ```
   This produces `.npy` tensors (128×256×256, clipped to [-1200,600], normalized to [-1,1]) plus a JSONL pointing to the processed files.

4. **Run inference** using the processed JSONL:
   ```bash
   python run_custom_eval.py \
     --dataset ../../3D_VLM_Spatial/spatial_qa_processed.jsonl \
     --output ../../3D_VLM_Spatial/reports/med3dvlm_predictions.jsonl \
     --model-path MagicXin/Med3DVLM-Qwen-2.5-7B \
     --device cuda:0 --dtype bfloat16
   ```

Adjust paths to match the HPC layout. Make sure SimpleITK and numpy are installed in the environment (already part of the upstream requirements).
