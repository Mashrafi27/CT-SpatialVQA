#!/usr/bin/env python3
"""Preprocess CT NIfTI volumes for CT-CHAT.

Matches benchmarking/inference/ct-chat/encode_embeddings.py:
- resample to target spacing (0.75, 0.75, 1.5)
- clip HU to [-1000, 1000]
- normalize to [-1, 1] by /1000
- center crop/pad to 480x480x240

By default, encodes CT-CLIP embeddings (CTViT) and saves .npz (key: arr).
Use --basic-only to save the preprocessed volume as .npy and skip embeddings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode CT volumes for CT-CHAT")
    p.add_argument("--input-jsonl", type=Path, required=True, help="JSONL with image_path/case_id")
    p.add_argument("--nifti-root", type=Path, required=True, help="Root of raw NIfTI volumes (valid_fixed)")
    p.add_argument("--output-root", type=Path, required=True, help="Directory to save .npz embeddings")
    p.add_argument("--output-jsonl", type=Path, required=True, help="Updated JSONL with image_path pointing to .npz")
    p.add_argument("--encoder-ckpt", type=Path, default=None, help="CT-CLIP encoder checkpoint")
    p.add_argument("--device", default="cuda", help="cuda, cuda:0, cpu")
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    p.add_argument("--basic-only", action="store_true", help="Only preprocess volume; skip CT-CLIP encoding")
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
    if name.endswith(".npz"):
        name = name.replace(".npz", ".nii.gz")
    if not name.endswith(".nii.gz"):
        name = f"{name}.nii.gz"
    stem = name.replace(".nii.gz", "")
    tokens = stem.split("_")
    if len(tokens) >= 3 and tokens[0] == "valid":
        patient = "_".join(tokens[:2])
        series = "_".join(tokens[:3])
        return root / patient / series / name
    return root / name


def derive_out_path(case_id: str, out_root: Path, ext: str = ".npz") -> Path:
    name = Path(case_id).name
    if name.endswith(".nii.gz"):
        name = name.replace(".nii.gz", ext)
    elif not name.endswith(ext):
        name = f"{name}{ext}"
    stem = name.replace(ext, "")
    tokens = stem.split("_")
    if len(tokens) >= 3 and tokens[0] == "valid":
        patient = "_".join(tokens[:2])
        series = "_".join(tokens[:3])
        out_dir = out_root / patient / series
    else:
        out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / name


def resize_array(array: torch.Tensor, current_spacing: tuple[float, float, float], target_spacing: tuple[float, float, float]) -> np.ndarray:
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    resized = F.interpolate(array, size=new_shape, mode="trilinear", align_corners=False).cpu().numpy()
    return resized


def nii_to_tensor(path: Path) -> torch.Tensor:
    nii_img = nib.load(str(path))
    img_data = nii_img.get_fdata()
    zooms = nii_img.header.get_zooms()
    if len(zooms) >= 3:
        xy_spacing = float(zooms[0])
        z_spacing = float(zooms[2])
    else:
        xy_spacing = 1.0
        z_spacing = 1.0

    target_spacing = (1.5, 0.75, 0.75)  # z,y,x
    current = (z_spacing, xy_spacing, xy_spacing)

    img_data = img_data.astype(np.float32)
    img_data = np.clip(img_data, -1000, 1000)
    img_data = img_data.transpose(2, 0, 1)  # z,y,x

    tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0)
    img_data = resize_array(tensor, current, target_spacing)[0][0]
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
    tensor = tensor.permute(2, 0, 1)  # D,H,W
    return tensor.unsqueeze(0)


def load_ctvit(ckpt: Path, device: torch.device):
    # add CT-CLIP repo to path
    repo_root = Path(__file__).resolve().parents[2] / "benchmarking" / "inference" / "ct-clip" / "repo"
    sys.path.insert(0, str(repo_root))
    from transformer_maskgit import CTViT  # type: ignore

    image_encoder = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
    ).to(device=device).eval()

    ckpt_obj = torch.load(str(ckpt), map_location="cpu")
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        ckpt_obj = ckpt_obj["state_dict"]
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj:
        ckpt_obj = ckpt_obj["model"]
    if isinstance(ckpt_obj, dict):
        remapped = {}
        for key, value in ckpt_obj.items():
            if key.startswith("visual_transformer."):
                remapped[key.replace("visual_transformer.", "", 1)] = value
            elif key.startswith("image_encoder."):
                remapped[key.replace("image_encoder.", "", 1)] = value
        if remapped:
            ckpt_obj = remapped
    try:
        image_encoder.load_state_dict(ckpt_obj, strict=False)
    except Exception:
        image_encoder.load(ckpt_obj)
    return image_encoder


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    image_encoder = None
    if not args.basic_only:
        if args.encoder_ckpt is None:
            raise SystemExit("--encoder-ckpt is required unless --basic-only is set.")
        image_encoder = load_ctvit(args.encoder_ckpt, device)

    updated = []
    for idx, rec in enumerate(tqdm(list(iter_jsonl(args.input_jsonl)), desc="Encode CT-CHAT")):
        if args.limit is not None and idx >= args.limit:
            break
        case_id = rec.get("case_id") or rec.get("image_path")
        if not case_id:
            raise ValueError("Missing case_id/image_path")
        src = resolve_nifti(str(case_id), args.nifti_root)
        if not src.is_file():
            raise FileNotFoundError(f"Missing source NIfTI: {src}")

        ext = ".npy" if args.basic_only else ".npz"
        out_path = derive_out_path(str(case_id), args.output_root, ext=ext)
        if not out_path.is_file():
            image = nii_to_tensor(src).to(device=device)
            if args.basic_only:
                vol = image.squeeze(0).cpu().numpy()
                np.save(out_path, vol.astype(np.float32))
            else:
                with torch.no_grad():
                    encoded = image_encoder(image.unsqueeze(0), return_encoded_tokens=True)
                np.savez(out_path, arr=encoded.cpu().detach().numpy())

        new_rec = dict(rec)
        new_rec["image_path"] = str(out_path.resolve())
        updated.append(new_rec)

    with args.output_jsonl.open("w") as f:
        for rec in updated:
            f.write(json.dumps(rec) + "\n")
    print(f"Encoded {len(updated)} volumes. Output root: {args.output_root}")


if __name__ == "__main__":
    main()
