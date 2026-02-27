#!/usr/bin/env python3
"""Preprocess CT NIfTI volumes for RadFM.

Creates .npy volumes shaped (H,W,D) or (3,H,W,D) compatible with
benchmarking/inference/radfm/run_custom_eval.py.
Pipeline:
- resample to fixed size (W,H,D) by default
- clip HU to [-1024, 1024]
- normalize to [0,1]
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
    p = argparse.ArgumentParser(description="Preprocess NIfTI for RadFM")
    p.add_argument("--input-jsonl", type=Path, required=True, help="JSONL with image_path/case_id")
    p.add_argument("--image-root", type=Path, default=None, help="Root to resolve relative case_ids")
    p.add_argument("--output-root", type=Path, required=True, help="Where to save .npy volumes")
    p.add_argument("--output-jsonl", type=Path, required=True, help="Updated JSONL path")
    p.add_argument("--depth", type=int, default=128)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--clip-min", type=int, default=-1024)
    p.add_argument("--clip-max", type=int, default=1024)
    p.add_argument("--channels", type=int, default=3, choices=[1, 3], help="Output channels (1 or 3)")
    p.add_argument("--no-normalize", action="store_true", help="Skip [0,1] normalization")
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


def preprocess_volume(path: Path, target_size: tuple[int, int, int], clip_min: int, clip_max: int, normalize: bool) -> np.ndarray:
    sitk_img = sitk.ReadImage(str(path))
    resampled = resample_image(sitk_img, target_size)
    vol = sitk.GetArrayFromImage(resampled).astype(np.float32)  # (D,H,W)
    vol = np.clip(vol, clip_min, clip_max)
    if normalize:
        vol = (vol - clip_min) / float(clip_max - clip_min)  # [0,1]
    vol = np.transpose(vol, (1, 2, 0))  # (H,W,D)
    return vol


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    processed = []
    target_size = (args.width, args.height, args.depth)
    for rec in tqdm(list(iter_jsonl(args.input_jsonl)), desc="Preprocess RadFM"):
        raw_path = rec.get("image_path") or rec.get("case_id")
        if not raw_path:
            raise ValueError("Missing image_path/case_id")
        src = resolve_path(str(raw_path), args.image_root)
        if not src.is_file():
            raise FileNotFoundError(f"Missing NIfTI: {src}")

        vol = preprocess_volume(src, target_size, args.clip_min, args.clip_max, not args.no_normalize)
        if args.channels == 3:
            vol = np.stack([vol, vol, vol], axis=0)
        out_path = derive_out_path(args.output_root, str(raw_path))
        np.save(out_path, vol.astype(np.float32))

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
