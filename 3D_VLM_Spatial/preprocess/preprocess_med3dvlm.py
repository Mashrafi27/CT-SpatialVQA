#!/usr/bin/env python3
"""Preprocess CT NIfTI volumes for Med3DVLM.

Pipeline aligned with M3D/Med3DVLM preprocessing used in their data scripts:
- min-max normalize volume to [0, 1]
- crop foreground (nonzero)
- resize to fixed size (D,H,W) = (128,256,256) by default
- save .npy volumes (D,H,W)
- emit updated JSONL with image_path pointing to .npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess NIfTI for Med3DVLM")
    p.add_argument("--input-jsonl", type=Path, required=True, help="JSONL with image_path/case_id")
    p.add_argument("--image-root", type=Path, default=None, help="Root to resolve relative case_ids")
    p.add_argument("--output-root", type=Path, required=True, help="Where to save .npy volumes")
    p.add_argument("--output-jsonl", type=Path, required=True, help="Updated JSONL path")
    p.add_argument("--depth", type=int, default=128)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--foreground-threshold", type=float, default=1e-6, help="Foreground threshold after normalization")
    return p.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_path(raw_path: str, root: Optional[Path]) -> Path:
    cleaned = str(raw_path).strip().strip('"').strip("'")
    candidate = Path(cleaned)
    if candidate.is_file():
        return candidate
    if root is None:
        return candidate
    name = candidate.name or cleaned
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        stem = name
    tokens = stem.split("_")
    if len(tokens) >= 3 and tokens[0].lower() == "valid":
        patient = "_".join(tokens[:2])
        series = "_".join(tokens[:3])
        preferred = root / patient / series / name
        if preferred.is_file():
            return preferred
        if not name.endswith((".nii.gz", ".nii")):
            alt = root / patient / series / f"{name}.nii.gz"
            if alt.is_file():
                return alt
        return preferred
    fallback = root / name
    if fallback.is_file():
        return fallback
    if not name.endswith((".nii.gz", ".nii")):
        alt = root / f"{name}.nii.gz"
        if alt.is_file():
            return alt
    return fallback


def derive_out_path(output_root: Path, raw_path: str) -> Path:
    name = Path(raw_path).name
    stem = name.replace(".nii.gz", "")
    tokens = stem.split("_")
    if len(tokens) >= 3 and tokens[0] == "valid":
        patient = "_".join(tokens[:2])
        series = "_".join(tokens[:3])
        out_dir = output_root / patient / series
    else:
        out_dir = output_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}.npy"


def normalize_minmax(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    min_val = float(arr.min())
    max_val = float(arr.max())
    if max_val - min_val < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    arr = (arr - min_val) / (max_val - min_val)
    return arr.astype(np.float32)


def crop_foreground(arr: np.ndarray, threshold: float) -> np.ndarray:
    mask = arr > threshold
    if not mask.any():
        return arr
    coords = np.where(mask)
    z_min, y_min, x_min = (int(c.min()) for c in coords)
    z_max, y_max, x_max = (int(c.max()) for c in coords)
    if (z_min, y_min, x_min) == (0, 0, 0) and (
        z_max == arr.shape[0] - 1
        and y_max == arr.shape[1] - 1
        and x_max == arr.shape[2] - 1
    ):
        return arr
    return arr[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]


def resize_array_to_size(arr: np.ndarray, target_dhw: tuple[int, int, int]) -> np.ndarray:
    """Resize array to target (D,H,W) using linear interpolation."""
    target_d, target_h, target_w = target_dhw
    img = sitk.GetImageFromArray(arr)
    in_size = np.array(img.GetSize(), dtype=np.float32)  # (W,H,D)
    out_size = np.array([target_w, target_h, target_d], dtype=np.float32)
    out_spacing = in_size / np.clip(out_size, a_min=1.0, a_max=None)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetSize([int(x) for x in out_size])
    resampler.SetOutputSpacing([float(x) for x in out_spacing])
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetDefaultPixelValue(0.0)
    resized = resampler.Execute(img)
    return sitk.GetArrayFromImage(resized).astype(np.float32)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    processed = []
    for rec in tqdm(list(iter_jsonl(args.input_jsonl)), desc="Preprocess Med3DVLM"):
        raw_path = rec.get("image_path") or rec.get("case_id")
        if not raw_path:
            raise ValueError("Missing image_path/case_id")
        src = resolve_path(str(raw_path), args.image_root)
        if not src.is_file():
            raise FileNotFoundError(f"Missing NIfTI: {src}")

        sitk_img = sitk.ReadImage(str(src))
        vol = sitk.GetArrayFromImage(sitk_img)  # (D,H,W)
        vol = normalize_minmax(vol)
        vol = crop_foreground(vol, args.foreground_threshold)
        vol = resize_array_to_size(vol, (args.depth, args.height, args.width))

        out_path = derive_out_path(args.output_root, str(raw_path))
        np.save(out_path, vol)

        updated = dict(rec)
        updated["image_path"] = str(out_path.resolve())
        if "case_id" not in updated:
            updated["case_id"] = Path(str(raw_path)).name
        processed.append(updated)

    with args.output_jsonl.open("w") as f:
        for rec in processed:
            f.write(json.dumps(rec) + "\n")

    print(f"Saved {len(processed)} volumes to {args.output_root}")


if __name__ == "__main__":
    main()
