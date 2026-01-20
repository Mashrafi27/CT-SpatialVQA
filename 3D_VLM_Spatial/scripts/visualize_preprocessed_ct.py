#!/usr/bin/env python3
"""Visualize a preprocessed CT volume by saving a slice montage."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save a montage from a preprocessed CT volume.")
    parser.add_argument("--input", type=Path, required=True, help="Path to .nii/.nii.gz/.npy/.npz volume")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--axis", choices=["axial", "coronal", "sagittal"], default="axial")
    parser.add_argument("--num-slices", type=int, default=16, help="Number of slices in the montage")
    parser.add_argument("--start", type=int, default=None, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--vmin", type=float, default=None, help="Lower display window")
    parser.add_argument("--vmax", type=float, default=None, help="Upper display window")
    parser.add_argument("--normalize", action="store_true", help="Normalize to 0-1 for display")
    return parser.parse_args()


def load_volume(path: Path) -> np.ndarray:
    suffix = "".join(path.suffixes)
    if suffix in {".nii", ".nii.gz"}:
        try:
            import nibabel as nib  # type: ignore
        except ImportError as exc:
            raise SystemExit("Install nibabel to visualize NIfTI files.") from exc
        vol = nib.load(str(path)).get_fdata().astype(np.float32)
        return vol
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        if "arr" in data:
            return data["arr"]
        # fall back to first array
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


def compute_indices(depth: int, num_slices: int, start: int | None, end: int | None) -> np.ndarray:
    if start is None:
        start = 0
    if end is None or end > depth:
        end = depth
    if end <= start:
        raise SystemExit("Invalid slice range.")
    if num_slices >= end - start:
        return np.arange(start, end)
    return np.linspace(start, end - 1, num_slices).astype(int)


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


def save_montage(volume: np.ndarray, indices: np.ndarray, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    num = len(indices)
    cols = math.ceil(math.sqrt(num))
    rows = math.ceil(num / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.atleast_2d(axes)

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")
        if i >= num:
            continue
        idx = indices[i]
        ax.imshow(volume[:, :, idx], cmap="gray")
        ax.set_title(f"{idx}", fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    volume = load_volume(args.input)
    volume = squeeze_to_3d(volume)
    volume = select_axis(volume, args.axis)
    volume = window_volume(volume, args.vmin, args.vmax, args.normalize)

    depth = volume.shape[2]
    indices = compute_indices(depth, args.num_slices, args.start, args.end)
    save_montage(volume, indices, args.output)
    print(f"Saved montage to {args.output}")


if __name__ == "__main__":
    main()
