#!/usr/bin/env python3
"""
Preprocess CT-RATE volumes for Med3DVLM.

Steps:
1. Read raw NIfTI volumes referenced in a JSONL (one record per QA pair).
2. Resample each volume to 128 x 256 x 256 (configurable).
3. Clip intensities to [-1200, 600], normalize to [-1, 1].
4. Save .npy tensors under the specified output root (mirrors case_id hierarchy).
5. Emit a new JSONL pointing to the processed files for downstream inference.
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
    parser = argparse.ArgumentParser(description="Preprocess CT-RATE volumes for Med3DVLM")
    parser.add_argument("--input-jsonl", type=Path, required=True, help="JSONL with image_path/question[/answer]")
    parser.add_argument("--image-root", type=Path, default=None, help="Root dir prepended when image_path is relative")
    parser.add_argument("--output-root", type=Path, required=True, help="Folder to store processed tensors (.npy)")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Path to write updated JSONL")
    parser.add_argument("--depth", type=int, default=128, help="Target slices (default: 128)")
    parser.add_argument("--height", type=int, default=256, help="Target height (default: 256)")
    parser.add_argument("--width", type=int, default=256, help="Target width (default: 256)")
    parser.add_argument("--clip-min", type=int, default=-1200, help="CT intensity min clip")
    parser.add_argument("--clip-max", type=int, default=600, help="CT intensity max clip")
    return parser.parse_args()


def read_image(image_path: Path) -> sitk.Image:
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing volume: {image_path}")
    return sitk.ReadImage(str(image_path))


def resample_image(image: sitk.Image, target_size: tuple[int, int, int]) -> sitk.Image:
    orig_size = np.array(list(image.GetSize()), dtype=np.int64)
    orig_spacing = np.array(list(image.GetSpacing()))
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
    arr = (arr - clip_min) / (clip_max - clip_min)  # [0,1]
    arr = arr * 2 - 1  # [-1,1]
    return arr.astype(np.float32)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_path(raw_path: str, root: Path | None) -> Path:
    candidate = Path(raw_path)
    if candidate.is_file():
        return candidate
    if root is None:
        return candidate
    name = candidate.name or raw_path
    stem = name.replace(".nii.gz", "")
    tokens = stem.split("_")
    if len(tokens) >= 3 and tokens[0] == "valid":
        patient = "_".join(tokens[:2])  # valid_1
        series = "_".join(tokens[:3])   # valid_1_a
        return (root / patient / series / name).resolve()
    return (root / name).resolve()


def derive_case_id(raw_path: str) -> str:
    name = Path(raw_path).stem
    return name or raw_path


def derive_save_path(output_root: Path, raw_path: str) -> Path:
    name = Path(raw_path).name
    stem = name.replace(".nii.gz", "")
    tokens = stem.split("_")
    if len(tokens) >= 3 and tokens[0] == "valid":
        patient = "_".join(tokens[:2])
        series = "_".join(tokens[:3])
        save_dir = output_root / patient / series
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / f"{stem}.npy"
    save_dir = output_root / stem
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir / f"{stem}.npy"


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    processed_records = []
    for record in tqdm(list(iter_jsonl(args.input_jsonl)), desc="Preprocessing volumes"):
        raw_path = record.get("image_path") or record.get("case_id")
        if not raw_path:
            raise ValueError("Each record must contain image_path or case_id")
        image_path = resolve_path(raw_path, args.image_root)

        sitk_img = read_image(image_path)
        resampled_img = resample_image(sitk_img, (args.width, args.height, args.depth))
        volume = sitk.GetArrayFromImage(resampled_img)  # (depth, height, width)
        volume = normalize_array(volume, args.clip_min, args.clip_max)

        save_path = derive_save_path(args.output_root, raw_path)
        np.save(save_path, volume)

        updated = dict(record)
        updated["case_id"] = record.get("case_id") or derive_case_id(raw_path)
        updated["image_path"] = str(save_path.resolve())
        processed_records.append(updated)

    with args.output_jsonl.open("w") as out_f:
        for rec in processed_records:
            out_f.write(json.dumps(rec) + "\n")

    print(f"Processed {len(processed_records)} volumes. Saved tensors under {args.output_root}.")


if __name__ == "__main__":
    main()
