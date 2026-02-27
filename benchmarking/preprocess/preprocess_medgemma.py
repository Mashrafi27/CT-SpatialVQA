#!/usr/bin/env python3
"""Preprocess CT NIfTI volumes for MedGemma.

Creates .npz files with pre-rendered RGB slices using CT windows.
Pipeline mirrors benchmarking/inference/medgemma/run_custom_eval.py:
- load volume, transpose to (Z,Y,X)
- sample N slices uniformly (default 85)
- apply 3 CT windows to RGB
- optional resize / resize-longest / pad-square
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import nibabel as nib
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing nibabel. Install with: pip install nibabel") from exc

DEFAULT_WINDOWS: List[Tuple[int, int]] = [
    (-1225, 1025),  # bone/lung (ww=2250, wl=-100)
    (-135, 215),    # soft tissue (ww=350, wl=40)
    (0, 80),        # brain (ww=80, wl=40)
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess NIfTI for MedGemma")
    p.add_argument("--input-jsonl", type=Path, required=True, help="JSONL with image_path/case_id")
    p.add_argument("--nifti-root", type=Path, required=True, help="Root of raw NIfTI volumes (valid_fixed)")
    p.add_argument("--output-root", type=Path, required=True, help="Where to save .npz slices")
    p.add_argument("--output-jsonl", type=Path, required=True, help="Updated JSONL path")
    p.add_argument("--num-slices", type=int, default=85)
    p.add_argument("--resize", type=int, default=896)
    p.add_argument("--resize-longest", type=int, default=0)
    p.add_argument("--pad-square", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Only process first N records (0 = all)")
    return p.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def derive_nifti_path(nifti_root: Path, case_id: str) -> Path:
    stem = case_id.replace(".nii.gz", "")
    subdir = stem.rsplit("_", 1)[0]
    if "_" in subdir:
        base = subdir.rsplit("_", 1)[0]
    else:
        base = subdir
    return nifti_root / base / subdir / case_id


def sample_indices(n_slices: int, num_samples: int) -> List[int]:
    if num_samples <= 0:
        return []
    if n_slices <= num_samples:
        return list(range(n_slices))
    idxs = np.linspace(0, n_slices - 1, num_samples)
    return [int(round(i)) for i in idxs]


def window_slice(slice_2d: np.ndarray, window: Tuple[int, int]) -> np.ndarray:
    lo, hi = window
    x = np.clip(slice_2d, lo, hi).astype(np.float32)
    x -= lo
    x /= (hi - lo)
    x *= 255.0
    return np.round(x, 0).astype(np.uint8)


def _resize_keep_aspect(rgb: np.ndarray, longest: int, pad_square: bool) -> np.ndarray:
    if longest <= 0:
        return rgb
    h, w = rgb.shape[:2]
    scale = longest / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = Image.fromarray(rgb)
    img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    if not pad_square:
        return np.asarray(img)
    canvas = Image.new("RGB", (longest, longest), (0, 0, 0))
    x0 = (longest - new_w) // 2
    y0 = (longest - new_h) // 2
    canvas.paste(img, (x0, y0))
    return np.asarray(canvas)


def make_rgb_window(slice_2d: np.ndarray, resize: int = 0, resize_longest: int = 0, pad_square: bool = False) -> np.ndarray:
    chans = [window_slice(slice_2d, w) for w in DEFAULT_WINDOWS]
    rgb = np.stack(chans, axis=-1)
    if resize and resize > 0:
        img = Image.fromarray(rgb)
        img = img.resize((resize, resize), resample=Image.BILINEAR)
        rgb = np.asarray(img)
    elif resize_longest and resize_longest > 0:
        rgb = _resize_keep_aspect(rgb, resize_longest, pad_square)
    return rgb


def load_volume(nifti_path: Path) -> np.ndarray:
    img = nib.load(str(nifti_path))
    vol = np.asarray(img.get_fdata())
    if vol.ndim == 3:
        vol = np.transpose(vol, (2, 1, 0))  # z,y,x
    return vol


def derive_out_path(output_root: Path, case_id: str) -> Path:
    stem = case_id.replace(".nii.gz", "")
    tokens = stem.split("_")
    if len(tokens) >= 3 and tokens[0] == "valid":
        patient = "_".join(tokens[:2])
        series = "_".join(tokens[:3])
        out_dir = output_root / patient / series
    else:
        out_dir = output_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}.npz"


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    processed = []
    records = list(iter_jsonl(args.input_jsonl))
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    for rec in tqdm(records, desc="Preprocess MedGemma"):
        case_id = rec.get("case_id") or Path(rec.get("image_path", "")).name
        if not case_id:
            raise ValueError("Missing case_id/image_path")
        nifti_path = derive_nifti_path(args.nifti_root, case_id)
        if not nifti_path.exists():
            image_path = rec.get("image_path")
            if image_path and Path(image_path).exists():
                vol = np.load(image_path)
            else:
                raise FileNotFoundError(f"Missing NIfTI: {nifti_path}")
        else:
            vol = load_volume(nifti_path)

        slice_idxs = sample_indices(vol.shape[0], args.num_slices)
        slice_rgbs = [make_rgb_window(vol[i], args.resize, args.resize_longest, args.pad_square) for i in slice_idxs]
        slice_rgbs = np.asarray(slice_rgbs, dtype=np.uint8)

        out_path = derive_out_path(args.output_root, case_id)
        np.savez(out_path, slices=slice_rgbs, slice_indices=np.asarray(slice_idxs, dtype=np.int32))

        updated = dict(rec)
        updated["image_path"] = str(out_path.resolve())
        processed.append(updated)

    with args.output_jsonl.open("w") as f:
        for rec in processed:
            f.write(json.dumps(rec) + "\n")

    print(f"Saved {len(processed)} slice packs to {args.output_root}")


if __name__ == "__main__":
    main()
