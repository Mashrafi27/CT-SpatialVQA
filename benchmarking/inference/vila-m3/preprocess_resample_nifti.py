#!/usr/bin/env python3
"""
Resample CT-RATE NIfTI volumes to a fixed voxel grid and normalize intensities
to match the Med3DVLM preprocessing (clip [-1200, 600] → normalize to [-1, 1]).

Inputs:
  - JSONL with case_id (or image_path) referencing original NIfTI files
  - Root directory containing the original NIfTI hierarchy (valid_fixed/valid_*/valid_*_*/file.nii.gz)

Outputs:
  - Resampled & normalized NIfTI files saved under an output root, preserving the hierarchy
  - A JSONL mirroring the input but pointing image_path to the resampled files (float32)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import SimpleITK as sitk
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resample CT-RATE NIfTI volumes to isotropic spacing.")
    p.add_argument("--input-jsonl", type=Path, required=True, help="JSONL with case_id/image_path (e.g., spatial_qa_processed.jsonl)")
    p.add_argument("--nifti-root", type=Path, required=True, help="Root folder of original NIfTI files (valid_fixed)")
    p.add_argument("--output-root", type=Path, required=True, help="Where to write resampled NIfTI files")
    p.add_argument("--output-jsonl", type=Path, required=True, help="Path to write updated JSONL with resampled paths")
    p.add_argument("--depth", type=int, default=128, help="Target slices (D, default 128)")
    p.add_argument("--height", type=int, default=256, help="Target height (H, default 256)")
    p.add_argument("--width", type=int, default=256, help="Target width (W, default 256)")
    p.add_argument("--clip-min", type=int, default=-1200, help="CT intensity min clip")
    p.add_argument("--clip-max", type=int, default=600, help="CT intensity max clip")
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
    if name.endswith(".npy"):
        name = name.replace(".npy", ".nii.gz")
    if not name.endswith(".nii.gz"):
        name = f"{name}.nii.gz"
    stem = name.replace(".nii.gz", "")
    tokens = stem.split("_")
    if len(tokens) >= 3 and tokens[0] == "valid":
        patient = "_".join(tokens[:2])
        series = "_".join(tokens[:3])
        out_dir = out_root / patient / series
    else:
        out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / name


def resample_image(img: sitk.Image, target_spacing: tuple[float, float, float]) -> sitk.Image:
    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()
    target_size = [
        int(round(osz * ospc / tspc)) for osz, ospc, tspc in zip(orig_size, orig_spacing, target_spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetDefaultPixelValue(img.GetPixelIDValue())
    return resampler.Execute(img)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    updated = []
    for rec in tqdm(list(iter_jsonl(args.input_jsonl)), desc="Resampling NIfTI volumes"):
        case_id = rec.get("case_id") or Path(rec.get("image_path", "")).name
        if not case_id:
            raise ValueError("Missing case_id/image_path in record.")
        src_path = resolve_nifti(case_id, args.nifti_root)
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing source NIfTI: {src_path}")

        out_path = derive_out_path(case_id, args.output_root)
        if not out_path.is_file():
            img = sitk.ReadImage(str(src_path))

            # Compute target spacing from desired size to preserve physical extent.
            orig_spacing = img.GetSpacing()
            orig_size = img.GetSize()
            target_size = (args.width, args.height, args.depth)
            target_spacing = tuple(
                float(osz) * float(ospc) / float(tsz) for osz, ospc, tsz in zip(orig_size, orig_spacing, target_size)
            )

            # Resample to target size
            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetOutputSpacing(target_spacing)
            resampler.SetSize([int(args.width), int(args.height), int(args.depth)])
            resampler.SetOutputDirection(img.GetDirection())
            resampler.SetOutputOrigin(img.GetOrigin())
            resampler.SetDefaultPixelValue(img.GetPixelIDValue())
            resampled = resampler.Execute(img)

            # Clip and normalize to [-1, 1]
            arr = sitk.GetArrayFromImage(resampled).astype("float32")  # z,y,x
            arr = arr.transpose(2, 1, 0)  # to x,y,z for consistent clipping
            arr = arr.clip(args.clip_min, args.clip_max)
            arr = (arr - args.clip_min) / (args.clip_max - args.clip_min)  # [0,1]
            arr = arr * 2.0 - 1.0  # [-1,1]
            arr = arr.transpose(2, 1, 0)  # back to z,y,x

            out_img = sitk.GetImageFromArray(arr, isVector=False)
            out_img.SetDirection(resampled.GetDirection())
            out_img.SetOrigin(resampled.GetOrigin())
            out_img.SetSpacing(target_spacing)
            sitk.WriteImage(out_img, str(out_path), useCompression=True)

        new_rec = dict(rec)
        new_rec["image_path"] = str(out_path.resolve())
        updated.append(new_rec)

    with args.output_jsonl.open("w") as f:
        for rec in updated:
            f.write(json.dumps(rec) + "\n")
    print(
        f"Resampled {len(updated)} volumes to size (D,H,W)=({args.depth},{args.height},{args.width}). "
        f"Output root: {args.output_root}"
    )


if __name__ == "__main__":
    main()
