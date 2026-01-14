#!/usr/bin/env python3
"""
Preprocess CT volumes for M3D-LaMed.

Paper spec (DeepTumorVQA Table 2): direct resize to 256x256x32.
M3D README: min-max normalize to [0,1], save as .npy with shape 1x32x256x256.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess CT-RATE volumes for M3D")
    p.add_argument("--input-jsonl", type=Path, required=True, help="JSONL with image_path/question[/answer]")
    p.add_argument("--nifti-root", type=Path, required=True, help="Root folder of original NIfTI files (valid_fixed)")
    p.add_argument("--output-root", type=Path, required=True, help="Where to write .npy volumes")
    p.add_argument("--output-jsonl", type=Path, required=True, help="Path to write updated JSONL")
    p.add_argument("--depth", type=int, default=32, help="Target depth (D)")
    p.add_argument("--height", type=int, default=256, help="Target height (H)")
    p.add_argument("--width", type=int, default=256, help="Target width (W)")
    return p.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_nifti(case_id: str, root: Path) -> Path:
    name = Path(case_id).name
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


def derive_out_path(case_id: str, out_root: Path) -> Path:
    name = Path(case_id).name
    if name.endswith(".nii.gz"):
        name = name.replace(".nii.gz", ".npy")
    elif not name.endswith(".npy"):
        name = f"{name}.npy"
    stem = name.replace(".npy", "")
    tokens = stem.split("_")
    if len(tokens) >= 3 and tokens[0] == "valid":
        patient = "_".join(tokens[:2])
        series = "_".join(tokens[:3])
        out_dir = out_root / patient / series
    else:
        out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / name


def resample_to_size(img: sitk.Image, target_size: tuple[int, int, int]) -> sitk.Image:
    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()
    target_spacing = tuple(
        float(osz) * float(ospc) / float(tsz)
        for osz, ospc, tsz in zip(orig_size, orig_spacing, target_size)
    )
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize([int(x) for x in target_size])
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetDefaultPixelValue(img.GetPixelIDValue())
    return resampler.Execute(img)


def normalize_minmax(arr: np.ndarray) -> np.ndarray:
    min_val = float(arr.min())
    max_val = float(arr.max())
    if max_val - min_val < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    arr = (arr - min_val) / (max_val - min_val)
    return arr.astype(np.float32)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    updated = []
    for rec in tqdm(list(iter_jsonl(args.input_jsonl)), desc="Preprocessing for M3D"):
        case_id = rec.get("case_id") or rec.get("image_path")
        if not case_id:
            raise ValueError("Missing case_id/image_path in record.")
        src = resolve_nifti(str(case_id), args.nifti_root)
        if not src.is_file():
            raise FileNotFoundError(f"Missing source NIfTI: {src}")

        out_path = derive_out_path(str(case_id), args.output_root)
        if not out_path.is_file():
            img = sitk.ReadImage(str(src))
            resampled = resample_to_size(img, (args.width, args.height, args.depth))
            arr = sitk.GetArrayFromImage(resampled).astype("float32")  # z,y,x
            arr = normalize_minmax(arr)
            # Save as 1 x D x H x W
            arr = arr[None, ...]
            np.save(out_path, arr)

        new_rec = dict(rec)
        new_rec["image_path"] = str(out_path.resolve())
        updated.append(new_rec)

    with args.output_jsonl.open("w") as f:
        for rec in updated:
            f.write(json.dumps(rec) + "\n")
    print(f"Preprocessed {len(updated)} volumes to {args.depth}x{args.height}x{args.width}.")


if __name__ == "__main__":
    main()
