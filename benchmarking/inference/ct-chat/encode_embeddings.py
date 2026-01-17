#!/usr/bin/env python3
"""
Encode CT volumes into CT-CHAT embeddings using the CTViT encoder.

This follows the preprocessing in llava/serve/encode_script.py:
- resample to target spacing (0.75, 0.75, 1.5)
- clip HU to [-1000, 1000]
- normalize to [-1, 1] by dividing by 1000
- center crop/pad to 480x480x240
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode CT volumes for CT-CHAT")
    p.add_argument("--input-jsonl", type=Path, required=True, help="JSONL with image_path/question[/answer]")
    p.add_argument("--nifti-root", type=Path, required=True, help="Root folder of NIfTI files (valid_fixed)")
    p.add_argument("--output-root", type=Path, required=True, help="Directory to save .npz embeddings")
    p.add_argument("--output-jsonl", type=Path, required=True, help="Updated JSONL with image_path set to .npz")
    p.add_argument("--encoder-ckpt", type=Path, required=True, help="Path to CT-CLIP encoder checkpoint (clip_visual_encoder.pth)")
    p.add_argument("--device", default="cuda", help="cuda, cuda:0, cpu")
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples")
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


def derive_out_path(case_id: str, out_root: Path) -> Path:
    name = Path(case_id).name
    if name.endswith(".nii.gz"):
        name = name.replace(".nii.gz", ".npz")
    elif not name.endswith(".npz"):
        name = f"{name}.npz"
    stem = name.replace(".npz", "")
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


def nii_to_tensor(path: Path, slope: float = 1.0, intercept: float = 0.0,
                  xy_spacing: float = 1.0, z_spacing: float = 1.0) -> torch.Tensor:
    nii_img = nib.load(str(path))
    img_data = nii_img.get_fdata()

    target_spacing = (1.5, 0.75, 0.75)
    current = (z_spacing, xy_spacing, xy_spacing)

    img_data = slope * img_data + intercept
    img_data = np.clip(img_data, -1000, 1000)
    img_data = img_data.transpose(2, 0, 1)

    tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0)
    img_data = resize_array(tensor, current, target_spacing)[0][0]
    img_data = np.transpose(img_data, (1, 2, 0))

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


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    from transformer_maskgit import CTViT  # local dependency from CT-CLIP

    device = torch.device(args.device)
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
    image_encoder.load(args.encoder_ckpt)

    updated = []
    for idx, rec in enumerate(tqdm(list(iter_jsonl(args.input_jsonl)), desc="Encoding CT-CHAT")):
        if args.limit is not None and idx >= args.limit:
            break
        case_id = rec.get("case_id") or rec.get("image_path")
        if not case_id:
            raise ValueError("Missing case_id/image_path in record.")
        src = resolve_nifti(str(case_id), args.nifti_root)
        if not src.is_file():
            raise FileNotFoundError(f"Missing source NIfTI: {src}")

        out_path = derive_out_path(str(case_id), args.output_root)
        if not out_path.is_file():
            image = nii_to_tensor(src).to(device=device)
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
