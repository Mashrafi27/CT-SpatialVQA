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
        "--dhw-labels",
        nargs="+",
        default=[],
        help="Labels whose volumes are stored as D,H,W and should be transposed to H,W,D for viewing.",
    )
    parser.add_argument(
        "--auto-orient",
        action="store_true",
        help="Auto-detect volume orientation (D,H,W vs H,W,D) and transpose to H,W,D when needed.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=2,
        metavar=("LABEL", "JSONL"),
        help="Model label and JSONL with image_path/case_id entries. Repeat for multiple models.",
    )
    parser.add_argument(
        "--ctclip-labels",
        nargs="+",
        default=[],
        help="Labels that should use CT-CLIP preprocessing (from raw NIfTI) instead of image_path.",
    )
    parser.add_argument(
        "--label-mode",
        action="append",
        default=[],
        help=(
            "Per-label preprocessing mode. Format: LABEL=MODE. "
            "Modes: raw, med3dvlm, merlin, vila_m3, m3d, radfm, medevalkit, medgemma, ctclip, generic."
        ),
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


def ctclip_preprocess(nifti_path: Path) -> np.ndarray:
    """Mirror CT-CLIP preprocessing in encode_embeddings.py (spacing, clip, normalize, crop/pad)."""
    try:
        import nibabel as nib  # type: ignore
    except ImportError as exc:
        raise SystemExit("Install nibabel to read NIfTI files.") from exc
    import torch
    import torch.nn.functional as F

    nii_img = nib.load(str(nifti_path))
    img_data = nii_img.get_fdata()
    zooms = nii_img.header.get_zooms()
    if len(zooms) >= 3:
        xy_spacing = float(zooms[0])
        z_spacing = float(zooms[2])
    else:
        xy_spacing = 1.0
        z_spacing = 1.0

    # Mimic encode_embeddings.py
    target_spacing = (1.5, 0.75, 0.75)  # z, y, x
    current = (z_spacing, xy_spacing, xy_spacing)

    img_data = img_data.astype(np.float32)
    img_data = np.clip(img_data, -1000, 1000)
    img_data = img_data.transpose(2, 0, 1)  # z,y,x

    tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0)
    original_shape = tensor.shape[2:]
    scaling_factors = [current[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    resized = F.interpolate(tensor, size=new_shape, mode="trilinear", align_corners=False).cpu().numpy()
    img_data = resized[0][0]
    img_data = np.transpose(img_data, (1, 2, 0))  # h,w,d

    img_data = (img_data / 1000).astype(np.float32)
    tensor = torch.tensor(img_data)

    target_shape = (480, 480, 240)
    h, w, d = tensor.shape
    dh, dw, dd = target_shape

    h_start = max((h - dh) // 2, 0)
    w_start = max((w - dw) // 2, 0)
    d_start = max((d - dd) // 2, 0)
    tensor = tensor[h_start:h_start + min(dh, h), w_start:w_start + min(dw, w), d_start:d_start + min(dd, d)]

    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before
    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before
    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before

    tensor = F.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)
    # Return as H,W,D
    return tensor.numpy()


def center_pad_crop(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
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


def merlin_preprocess(nifti_path: Path) -> np.ndarray:
    """Mirror Merlin preprocessing (RAS, resample spacing, clip, normalize, pad/crop)."""
    try:
        import SimpleITK as sitk  # type: ignore
    except ImportError as exc:
        raise SystemExit("Install SimpleITK to use Merlin preprocessing.") from exc

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
    arr = (arr + 1000) / 2000.0  # [0,1]
    arr = center_pad_crop(arr, (160, 224, 224))
    return np.transpose(arr, (1, 2, 0))  # H,W,D


DEFAULT_WINDOWS: List[Tuple[int, int]] = [
    (-1024, 1024),
    (-135, 215),
    (0, 80),
]


def make_rgb_window(slice_2d: np.ndarray) -> np.ndarray:
    def _window(slice_2d: np.ndarray, window: Tuple[int, int]) -> np.ndarray:
        lo, hi = window
        x = np.clip(slice_2d, lo, hi).astype(np.float32)
        x -= lo
        x /= (hi - lo)
        x *= 255.0
        return np.round(x, 0).astype(np.uint8)

    chans = [_window(slice_2d, w) for w in DEFAULT_WINDOWS]
    return np.stack(chans, axis=-1)


def parse_label_modes(pairs: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            continue
        label, mode = item.split("=", 1)
        mapping[label.strip()] = mode.strip()
    return mapping


def infer_mode_from_label(label: str) -> str:
    key = label.lower()
    if "ct-clip" in key or "ctclip" in key or "ctchat" in key:
        return "ctclip"
    if "medgemma" in key:
        return "medgemma"
    if "medeval" in key or "lingshu" in key:
        return "medevalkit"
    if "merlin" in key:
        return "merlin"
    if "vila" in key:
        return "vila_m3"
    if key == "m3d" or "m3d" in key:
        return "m3d"
    if "radfm" in key:
        return "radfm"
    if "med3dvlm" in key:
        return "med3dvlm"
    return "generic"


def squeeze_to_3d(volume: np.ndarray) -> np.ndarray:
    vol = volume
    while vol.ndim > 3:
        vol = vol[0]
    if vol.ndim != 3:
        raise SystemExit(f"Expected 3D volume after squeeze, got shape {vol.shape}")
    return vol


def _infer_orientation(shape: tuple[int, int, int]) -> str | None:
    """
    Heuristic: CT usually has H,W ~ 512 and D smaller.
    - If axis0 much smaller than axis1/axis2 => D,H,W.
    - If axis2 much smaller than axis0/axis1 => H,W,D.
    """
    d0, d1, d2 = shape
    def is_much_smaller(a, b, c):
        return a <= min(b, c) * 0.75
    if is_much_smaller(d0, d1, d2):
        return "DHW"
    if is_much_smaller(d2, d0, d1):
        return "HWD"
    return None


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

    label_modes = parse_label_modes(args.label_mode)

    case_ids = list({cid for index in label_indices.values() for cid in index})
    if args.common_only:
        case_ids = list(set.intersection(*(set(idx) for idx in label_indices.values())))
    if not case_ids:
        raise SystemExit("No case_ids found.")

    random.shuffle(case_ids)
    case_ids = case_ids[: args.num_samples]

    def to_hwd(vol: np.ndarray, assume_dhw: bool) -> np.ndarray:
        vol = squeeze_to_3d(vol)
        if args.auto_orient:
            orient = _infer_orientation(vol.shape)
            if orient == "DHW":
                return np.transpose(vol, (1, 2, 0))
            if orient == "HWD":
                return vol
        if assume_dhw:
            return np.transpose(vol, (1, 2, 0))
        return vol

    def load_nifti_to_hwd(nifti_path: Path) -> np.ndarray:
        try:
            import nibabel as nib  # type: ignore
        except ImportError as exc:
            raise SystemExit("Install nibabel to read NIfTI files.") from exc
        vol = np.asarray(nib.load(str(nifti_path)).get_fdata())
        # nibabel loads x,y,z; match medgemma/medevalkit: z,y,x
        if vol.ndim == 3:
            vol = np.transpose(vol, (2, 1, 0))  # D,H,W
        return np.transpose(vol, (1, 2, 0))  # H,W,D

    axes = ["axial", "coronal", "sagittal"]
    for case_id in tqdm(case_ids, desc="Saving slices"):
        raw_vol = None
        if args.raw_root is not None:
            raw_path = resolve_raw_nifti(case_id, args.raw_root)
            if raw_path.is_file():
                raw_vol = to_hwd(load_volume(raw_path), assume_dhw=True)

        volumes_by_label: Dict[str, np.ndarray] = {}
        for label, index in label_indices.items():
            if case_id not in index:
                continue
            mode = label_modes.get(label) or infer_mode_from_label(label)
            if label in args.ctclip_labels:
                mode = "ctclip"
            if label in args.use_raw_for_label:
                mode = "raw"

            image_path = Path(index[case_id])
            if mode == "raw":
                if raw_vol is not None:
                    volumes_by_label[label] = raw_vol
                continue
            if mode == "ctclip":
                if args.raw_root is None:
                    continue
                src = resolve_raw_nifti(case_id, args.raw_root)
                if not src.is_file():
                    continue
                volumes_by_label[label] = ctclip_preprocess(src)
                continue
            if mode == "merlin":
                if args.raw_root is None:
                    continue
                src = resolve_raw_nifti(case_id, args.raw_root)
                if not src.is_file():
                    continue
                volumes_by_label[label] = merlin_preprocess(src)
                continue
            if mode in {"medevalkit", "medgemma"}:
                if args.raw_root is not None:
                    src = resolve_raw_nifti(case_id, args.raw_root)
                    if src.is_file():
                        volumes_by_label[label] = load_nifti_to_hwd(src)
                        continue
                if image_path.is_file():
                    volumes_by_label[label] = to_hwd(load_volume(image_path), assume_dhw=True)
                continue
            if not image_path.is_file():
                continue
            if mode == "vila_m3":
                vol = to_hwd(load_volume(image_path), assume_dhw=True)
                vol = (vol + 1.0) / 2.0
                volumes_by_label[label] = np.clip(vol, 0.0, 1.0)
                continue
            if mode == "m3d":
                volumes_by_label[label] = to_hwd(load_volume(image_path), assume_dhw=True)
                continue
            if mode == "radfm":
                vol = load_volume(image_path)
                if vol.ndim == 4:
                    vol = vol[0]
                volumes_by_label[label] = to_hwd(vol, assume_dhw=False)
                continue
            if mode == "med3dvlm":
                volumes_by_label[label] = to_hwd(load_volume(image_path), assume_dhw=True)
                continue
            volumes_by_label[label] = to_hwd(load_volume(image_path), assume_dhw=False)
        for axis in axes:
            if raw_vol is not None:
                raw_axis = select_axis(raw_vol, axis)
                mid = raw_axis.shape[2] // 2
                raw_slice = window_slice(raw_axis[:, :, mid], args.vmin, args.vmax, args.normalize)
                raw_slice = to_uint8(raw_slice)
                save_png(raw_slice, args.output_root / case_id / axis / "RAW.png", args.size)

            for label, vol in volumes_by_label.items():
                vol_axis = select_axis(vol, axis)
                mid = vol_axis.shape[2] // 2
                slice_2d = vol_axis[:, :, mid]
                mode = label_modes.get(label) or infer_mode_from_label(label)
                if label in args.ctclip_labels:
                    mode = "ctclip"
                if label in args.use_raw_for_label:
                    mode = "raw"
                if mode in {"medevalkit", "medgemma"}:
                    rgb = make_rgb_window(slice_2d)
                    save_png(rgb, args.output_root / case_id / axis / f"{label}.png", args.size)
                else:
                    slice_2d = window_slice(slice_2d, args.vmin, args.vmax, args.normalize)
                    slice_2d = to_uint8(slice_2d)
                    save_png(slice_2d, args.output_root / case_id / axis / f"{label}.png", args.size)

    print(f"Saved slices to {args.output_root}")


if __name__ == "__main__":
    main()
