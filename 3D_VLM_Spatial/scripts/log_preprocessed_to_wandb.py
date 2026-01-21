#!/usr/bin/env python3
"""Log preprocessed CT volumes to Weights & Biases as montage images."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log preprocessed CT montages to W&B.")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--run-name", default="preprocessed-ct-logs")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--axis", choices=["axial", "coronal", "sagittal"], default="axial")
    parser.add_argument("--num-slices", type=int, default=16)
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--max-per-model", type=int, default=None, help="Optional cap per model label")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Root of raw NIfTI volumes (valid_fixed). If set, logs raw + preprocessed.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=2,
        metavar=("LABEL", "JSONL"),
        help="Model label and JSONL with image_path entries. Repeat for multiple models.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_paths(path: Path) -> List[Tuple[str, str, str]]:
    items: List[Tuple[str, str]] = []
    for rec in iter_jsonl(path):
        image_path = rec.get("image_path") or rec.get("case_id")
        case_id = rec.get("case_id") or image_path
        if image_path:
            items.append((str(image_path), str(case_id), rec.get("question", "")))
    return items


def load_volume(path: Path) -> np.ndarray:
    suffix = "".join(path.suffixes)
    if suffix in {".nii", ".nii.gz"}:
        try:
            import nibabel as nib  # type: ignore
        except ImportError as exc:
            raise SystemExit("Install nibabel to read NIfTI files.") from exc
        return nib.load(str(path)).get_fdata().astype(np.float32)
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        if "arr" in data:
            return data["arr"]
        return data[list(data.keys())[0]]
    raise SystemExit(f"Unsupported input: {path}")


def squeeze_to_3d(volume: np.ndarray) -> np.ndarray:
    vol = volume
    while vol.ndim > 3:
        vol = vol[0]
    if vol.ndim != 3:
        raise SystemExit(f"Expected 3D volume after squeeze, got shape {vol.shape}")
    return vol


def select_axis(volume: np.ndarray, axis: str) -> np.ndarray:
    if axis == "axial":
        return volume
    if axis == "coronal":
        return np.transpose(volume, (0, 2, 1))
    return np.transpose(volume, (2, 1, 0))


def compute_indices(depth: int, num_slices: int) -> np.ndarray:
    if num_slices >= depth:
        return np.arange(depth)
    return np.linspace(0, depth - 1, num_slices).astype(int)


def window_volume(volume: np.ndarray, vmin: float | None, vmax: float | None, normalize: bool) -> np.ndarray:
    vol = volume.astype(np.float32)
    if vmin is not None or vmax is not None:
        if vmin is None:
            vmin = float(vol.min())
        if vmax is None:
            vmax = float(vol.max())
        vol = np.clip(vol, vmin, vmax)
    if normalize:
        vmin = float(vol.min())
        vmax = float(vol.max())
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)
    return vol


def make_montage(volume: np.ndarray, indices: np.ndarray) -> np.ndarray:
    num = len(indices)
    cols = math.ceil(math.sqrt(num))
    rows = math.ceil(num / cols)
    h, w = volume.shape[0], volume.shape[1]
    canvas = np.zeros((rows * h, cols * w), dtype=np.float32)
    for i, idx in enumerate(indices):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = volume[:, :, idx]
    return canvas


def to_uint8(image: np.ndarray) -> np.ndarray:
    img = image.copy()
    vmin = float(img.min())
    vmax = float(img.max())
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def resolve_raw_nifti(case_id: str, root: Path) -> Path:
    name = Path(case_id).name
    if name.endswith(".npz"):
        name = name.replace(".npz", ".nii.gz")
    if name.endswith(".npy"):
        name = name.replace(".npy", ".nii.gz")
    if not name.endswith(".nii.gz"):
        name = f"{name}.nii.gz"
    stem = name.replace(".nii.gz", "")
    tokens = stem.split("_")
    if len(tokens) >= 3 and tokens[0] == "valid":
        patient = "_".join(tokens[:2])
        series = "_".join(tokens[:3])
        return root / patient / series / name
    return root / name


def main() -> None:
    args = parse_args()
    if not args.dataset:
        raise SystemExit("Provide at least one --dataset LABEL JSONL pair.")

    random.seed(args.seed)
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("Install wandb to log images.") from exc

    run = wandb.init(project=args.project, name=args.run_name)

    for label, jsonl_path in args.dataset:
        entries = load_paths(Path(jsonl_path))
        if not entries:
            print(f"[WARN] No entries in {jsonl_path}")
            continue
        random.shuffle(entries)
        max_items = args.num_samples
        if args.max_per_model is not None:
            max_items = min(max_items, args.max_per_model)
        entries = entries[:max_items]

        images = []
        for image_path, case_id, question in entries:
            path = Path(image_path)
            volume = load_volume(path)
            volume = squeeze_to_3d(volume)
            axes = ["axial", "coronal", "sagittal"]
            for axis in axes:
                vol_axis = select_axis(volume, axis)
                vol_axis = window_volume(vol_axis, args.vmin, args.vmax, args.normalize)
                indices = compute_indices(vol_axis.shape[2], args.num_slices)
                montage = make_montage(vol_axis, indices)
                montage = to_uint8(montage)
                caption = f"{label} | {path.name} | {axis}"
                if question:
                    caption = f"{caption} | Q: {question}"
                images.append(wandb.Image(montage, caption=caption))

            if args.raw_root is not None:
                raw_path = resolve_raw_nifti(case_id, args.raw_root)
                if raw_path.is_file():
                    raw_vol = load_volume(raw_path)
                    raw_vol = squeeze_to_3d(raw_vol)
                    for axis in axes:
                        raw_axis = select_axis(raw_vol, axis)
                        raw_axis = window_volume(raw_axis, args.vmin, args.vmax, args.normalize)
                        indices = compute_indices(raw_axis.shape[2], args.num_slices)
                        montage = make_montage(raw_axis, indices)
                        montage = to_uint8(montage)
                        caption = f"{label} RAW | {raw_path.name} | {axis}"
                        if question:
                            caption = f"{caption} | Q: {question}"
                        images.append(wandb.Image(montage, caption=caption))
                else:
                    print(f"[WARN] Missing raw NIfTI: {raw_path}")

        run.log({f"{label}_montages": images})

    run.finish()


if __name__ == "__main__":
    main()
