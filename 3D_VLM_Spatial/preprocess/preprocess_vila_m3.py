#!/usr/bin/env python3
"""Preprocess CT NIfTI volumes for VILA-M3.

VILA-M3 in this repo consumes the same normalized .npy volumes as Med3DVLM
(clip [-1200, 600] → normalize to [-1, 1], resample to 128x256x256 by default).
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
    p = argparse.ArgumentParser(description="Preprocess NIfTI for VILA-M3")
    p.add_argument("--input-jsonl", type=Path, required=True, help="JSONL with image_path/case_id")
    p.add_argument("--image-root", type=Path, default=None, help="Root to resolve relative case_ids")
    p.add_argument("--output-root", type=Path, required=True, help="Where to save .npy volumes")
    p.add_argument("--output-jsonl", type=Path, required=True, help="Updated JSONL path")
    p.add_argument("--depth", type=int, default=128)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--clip-min", type=int, default=-1200)
    p.add_argument("--clip-max", type=int, default=600)
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


def resample_image(image: sitk.Image, target_size: tuple[int, int, int]) -> sitk.Image:
    orig_size = np.array(list(image.GetSize()), dtype=np.int64)
    orig_spacing = np.array(list(image.GetSpacing()), dtype=np.float32)
    target_size = np.array(target_size, dtype=np.int64)
    new_spacing = orig_spacing * (orig_size / target_size)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetSize([int(x) for x in target_size])
    resampler.SetOutputSpacing([float(x) for x in new_spacing])
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    return resampler.Execute(image)


def normalize_array(arr: np.ndarray, clip_min: int, clip_max: int) -> np.ndarray:
    arr = np.clip(arr, clip_min, clip_max)
    arr = (arr - clip_min) / float(clip_max - clip_min)  # [0,1]
    arr = arr * 2.0 - 1.0  # [-1,1]
    return arr.astype(np.float32)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    processed = []
    for rec in tqdm(list(iter_jsonl(args.input_jsonl)), desc="Preprocess VILA-M3"):
        raw_path = rec.get("image_path") or rec.get("case_id")
        if not raw_path:
            raise ValueError("Missing image_path/case_id")
        src = resolve_path(str(raw_path), args.image_root)
        if not src.is_file():
            raise FileNotFoundError(f"Missing NIfTI: {src}")

        sitk_img = sitk.ReadImage(str(src))
        resampled = resample_image(sitk_img, (args.width, args.height, args.depth))
        vol = sitk.GetArrayFromImage(resampled)
        vol = normalize_array(vol, args.clip_min, args.clip_max)

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
