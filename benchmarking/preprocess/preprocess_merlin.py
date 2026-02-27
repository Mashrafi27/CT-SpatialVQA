#!/usr/bin/env python3
"""Preprocess CT NIfTI volumes for Merlin.

Pipeline (matches benchmarking/inference/merlin/run_custom_eval.py):
- reorient to RAS
- resample to spacing 1.5 x 1.5 x 3.0
- clip HU to [-1000, 1000]
- normalize to [0,1]
- center pad/crop to (D,H,W) = (160,224,224)
- save .npy volumes (D,H,W)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess NIfTI for Merlin")
    p.add_argument("--input-jsonl", type=Path, required=True, help="JSONL with image_path/case_id")
    p.add_argument("--image-root", type=Path, default=None, help="Root to resolve relative case_ids")
    p.add_argument("--output-root", type=Path, required=True, help="Where to save .npy volumes")
    p.add_argument("--output-jsonl", type=Path, required=True, help="Updated JSONL path")
    p.add_argument("--depth", type=int, default=160)
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    return p.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_path(raw_path: str, root: Optional[Path]) -> Path:
    candidate = Path(raw_path)
    if candidate.is_file():
        return candidate
    if root is None:
        return candidate
    name = candidate.name or raw_path
    stem = name.replace(".nii.gz", "")
    tokens = stem.split("_")
    if len(tokens) >= 3 and tokens[0] == "valid":
        patient = "_".join(tokens[:2])
        series = "_".join(tokens[:3])
        return root / patient / series / name
    return root / name


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


def center_pad_crop(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    depth, height, width = volume.shape
    t_depth, t_height, t_width = target_shape

    pad_d = max(t_depth - depth, 0)
    pad_h = max(t_height - height, 0)
    pad_w = max(t_width - width, 0)

    if pad_d or pad_h or pad_w:
        pad_before = (pad_d // 2, pad_h // 2, pad_w // 2)
        pad_after = (pad_d - pad_before[0], pad_h - pad_before[1], pad_w - pad_before[2])
        volume = np.pad(
            volume,
            ((pad_before[0], pad_after[0]),
             (pad_before[1], pad_after[1]),
             (pad_before[2], pad_after[2])),
            mode="constant",
            constant_values=0,
        )

    depth, height, width = volume.shape
    start_d = max((depth - t_depth) // 2, 0)
    start_h = max((height - t_height) // 2, 0)
    start_w = max((width - t_width) // 2, 0)
    return volume[
        start_d:start_d + t_depth,
        start_h:start_h + t_height,
        start_w:start_w + t_width,
    ]


def merlin_preprocess(nifti_path: Path, target_shape: Tuple[int, int, int]) -> np.ndarray:
    img = sitk.ReadImage(str(nifti_path))
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("RAS")
    img = orienter.Execute(img)

    target_spacing = (1.5, 1.5, 3.0)
    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(orig_size, orig_spacing, target_spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize([int(x) for x in new_size])
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetDefaultPixelValue(img.GetPixelIDValue())
    img = resampler.Execute(img)

    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # z,y,x
    arr = np.clip(arr, -1000, 1000)
    arr = (arr + 1000.0) / 2000.0  # [0,1]
    arr = center_pad_crop(arr, target_shape)
    return arr  # D,H,W


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    target_shape = (args.depth, args.height, args.width)

    processed = []
    for rec in tqdm(list(iter_jsonl(args.input_jsonl)), desc="Preprocess Merlin"):
        raw_path = rec.get("image_path") or rec.get("case_id")
        if not raw_path:
            raise ValueError("Missing image_path/case_id")
        src = resolve_path(str(raw_path), args.image_root)
        if not src.is_file():
            raise FileNotFoundError(f"Missing NIfTI: {src}")

        vol = merlin_preprocess(src, target_shape)
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
