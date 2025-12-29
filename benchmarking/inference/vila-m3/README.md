# VILA-M3

- **GitHub**: https://github.com/Project-MONAI/VLM-Radiology-Agent-Framework
- **Paper**: https://openaccess.thecvf.com/content/CVPR2025/papers/Nath_VILA-M3_Enhancing_Vision-Language_Models_with_Medical_Expert_Knowledge_CVPR_2025_paper
- **Hugging Face checkpoints**: `MONAI/Llama3-VILA-M3-{3B,8B,13B}`
- **Primary tasks**: multimodal VQA, radiology report generation, abnormality classification, segmentation-routed reasoning through expert models.

The upstream MONAI team now ships VILA-M3 together with expert-model plugins, evaluation scripts, and a self-contained Gradio demo. We track their repo as a submodule so we can pin a commit for reproducible benchmarking.

## Layout

```
benchmarking/inference/vila-m3/
├── README.md  # this file
└── repo/      # submodule: Project-MONAI/VLM-Radiology-Agent-Framework
```

To materialize or update the code run:

```bash
git submodule update --init --recursive benchmarking/inference/vila-m3/repo
```

## Environment setup (manual, HPC friendly)

```bash
cd benchmarking/inference/vila-m3/repo
conda create -n vila-m3 python=3.10 -y
conda activate vila-m3
# Pull the base VILA dependencies plus MONAI add-ons + expert checkpoints
make demo_m3
```

`make demo_m3` installs the upstream VILA stack (via `thirdparty/VILA/environment_setup.sh`), MONAI extras, TorchXRayVision models, and required segmentation bundles (VISTA3D + BRATS). It expects CUDA 12.2+ and ≥18 GB of GPU memory (8 B checkpoint) or ≥30 GB (13 B).

### Docker alternative

Upstream also publishes a CUDA-ready Dockerfile that bakes all dependencies:

```bash
cd benchmarking/inference/vila-m3/repo
docker build --network=host --progress=plain -t monai-m3:latest -f m3/demo/Dockerfile .
docker run -it --rm --ipc host --gpus all --net host \
  -v /path/to/checkpoints:/data/checkpoints \
  monai-m3:latest bash
```

Once inside the container activate the environment and follow the demo/eval steps below.

## Running the default demo

```bash
cd benchmarking/inference/vila-m3/repo/m3/demo
conda activate vila-m3  # or source .venv if you used python -m venv
python gradio_m3.py \
  --source hf \
  --modelpath MONAI/Llama3-VILA-M3-8B \
  --convmode llama_3
```

The demo automatically downloads the requested Hugging Face checkpoint (must be logged in with a token that has the appropriate usage agreement). Add `--source local --modelpath /path/to/weights` if you already mirrored the checkpoint on the HPC scratch.

## Hooking into our benchmarking dataset

We will reuse the `3D_VLM_Spatial/spatial_qa_processed.jsonl` questions and CT volumes prepared for Med3DVLM. Planned steps:

1. [ ] Implement a slicer/preprocessor that exports CT-RATE volumes into the format expected by `m3` (likely MHA/PNG stacks) while keeping case IDs aligned with our JSONL loader.
2. [ ] Extend `m3/demo/gradio_m3.py` or add a new `benchmarking/inference/vila-m3/run_custom_eval.py` that ingests one QA pair at a time and queries a chosen VILA-M3 checkpoint via the MONAI agent interface.
3. [ ] Capture outputs in `3D_VLM_Spatial/reports/vila-m3_predictions.jsonl`, then score them with the Gemini rubric we already use for Med3DVLM.

## Notes / TODO

- [ ] Document Slurm launcher once the inference harness exists (expect ≥22 GB GPU RAM for the 8 B model, >30 GB for 13 B when expert models fire).
- [ ] Mirror the MONAI bundles (VISTA3D + BRATS) to the HPC shared cache to avoid repeated downloads per user.
- [ ] Track upstream commit hash in this README once we freeze on a version for benchmarking.
