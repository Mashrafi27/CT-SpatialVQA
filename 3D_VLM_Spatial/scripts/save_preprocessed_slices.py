#!/usr/bin/env python3
"""Save mid-slice PNGs for raw + preprocessed volumes per model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save mid-slice PNGs for raw + preprocessed volumes.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory to save PNGs.")
    parser.add_argument("--raw-root", type=Path, default=None, help="Root of raw NIfTI volumes (valid_fixed).")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--size", type=int, default=512, help="Output PNG size (square).")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--common-only", action="store_true", help="Use only case_ids present in all datasets.")
    parser.add_argument(
        "--use-raw-for-label",
        action="append",
        default=[],
        help="Labels that should use raw volumes instead of image_path (e.g., CTCHAT).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=2,
        metavar=("LABEL", "JSONL"),
        help="Model label and JSONL with image_path/case_id entries. Repeat for multiple models.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_index(path: Path) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for rec in iter_jsonl(path):
        case_id = rec.get("case_id") or rec.get("image_path")
        image_path = rec.get("image_path") or rec.get("case_id")
        if case_id and image_path:
            index[str(case_id)] = str(image_path)
    return index


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


def window_slice(slice_2d: np.ndarray, vmin: float | None, vmax: float | None, normalize: bool) -> np.ndarray:
    img = slice_2d.astype(np.float32)
    if vmin is not None or vmax is not None:
        if vmin is None:
            vmin = float(img.min())
        if vmax is None:
            vmax = float(img.max())
        img = np.clip(img, vmin, vmax)
    if normalize:
        vmin = float(img.min())
        vmax = float(img.max())
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
    return img


def to_uint8(image: np.ndarray) -> np.ndarray:
    img = image.copy()
    vmin = float(img.min())
    vmax = float(img.max())
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def save_png(slice_2d: np.ndarray, out_path: Path, size: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(slice_2d)
    img = img.resize((size, size), resample=Image.BILINEAR)
    img.save(out_path)


def main() -> None:
    args = parse_args()
    if not args.dataset:
        raise SystemExit("Provide at least one --dataset LABEL JSONL pair.")

    random.seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    label_indices: Dict[str, Dict[str, str]] = {}
    for label, jsonl in args.dataset:
        label_indices[label] = load_index(Path(jsonl))

    case_ids = list({cid for index in label_indices.values() for cid in index})
    if args.common_only:
        case_ids = list(set.intersection(*(set(idx) for idx in label_indices.values())))
    if not case_ids:
        raise SystemExit("No case_ids found.")

    random.shuffle(case_ids)
    case_ids = case_ids[: args.num_samples]

    axes = ["axial", "coronal", "sagittal"]
    for case_id in tqdm(case_ids, desc="Saving slices"):
        raw_vol = None
        if args.raw_root is not None:
            raw_path = resolve_raw_nifti(case_id, args.raw_root)
            if raw_path.is_file():
                raw_vol = squeeze_to_3d(load_volume(raw_path))
        for axis in axes:
            if raw_vol is not None:
                raw_axis = select_axis(raw_vol, axis)
                mid = raw_axis.shape[2] // 2
                raw_slice = window_slice(raw_axis[:, :, mid], args.vmin, args.vmax, args.normalize)
                raw_slice = to_uint8(raw_slice)
                save_png(raw_slice, args.output_root / case_id / axis / "RAW.png", args.size)

            for label, index in label_indices.items():
                if case_id not in index:
                    continue
                image_path = index[case_id]
                if label in args.use_raw_for_label and args.raw_root is not None:
                    src = resolve_raw_nifti(case_id, args.raw_root)
                else:
                    src = Path(image_path)
                if not src.is_file():
                    continue
                vol = squeeze_to_3d(load_volume(src))
                vol_axis = select_axis(vol, axis)
                mid = vol_axis.shape[2] // 2
                slice_2d = window_slice(vol_axis[:, :, mid], args.vmin, args.vmax, args.normalize)
                slice_2d = to_uint8(slice_2d)
                save_png(slice_2d, args.output_root / case_id / axis / f"{label}.png", args.size)

    print(f"Saved slices to {args.output_root}")


if __name__ == "__main__":
    main()
