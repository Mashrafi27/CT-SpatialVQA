#!/usr/bin/env python3
"""
Custom evaluation runner for Merlin (report-generation checkpoint) on our VQA JSONL.

This treats Merlin as a conditional generator: we feed the full resampled 3D CT volume
and the question as text, and decode a concise answer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import SimpleITK as sitk
import torch
from merlin import Merlin
from tqdm import tqdm

SYSTEM_PROMPT = (
    "Answer the question concisely using this CT. Output only the answer (no preamble). "
    "Question: {question}\nAnswer:"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Merlin (report gen) on VQA JSONL")
    p.add_argument("--dataset", type=Path, required=True, help="JSONL with case_id/question[/answer]")
    p.add_argument("--nifti-root", type=Path, required=True, help="Root folder containing resampled NIfTI volumes")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL for predictions")
    p.add_argument("--device", type=str, default="cuda:0", help="Device for inference")
    p.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens to generate")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    p.add_argument("--limit", type=int, default=None, help="Optional limit on number of QA pairs")
    p.add_argument("--skip", type=int, default=0, help="Skip this many QA pairs from start")
    p.add_argument("--resume", action="store_true", help="If set, append to output and auto-skip existing lines")
    return p.parse_args()


def load_jsonl(path: Path, skip: int = 0, limit: Optional[int] = None) -> List[Dict]:
    recs: List[Dict] = []
    with path.open() as f:
        for idx, line in enumerate(f):
            if idx < skip:
                continue
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
            if limit is not None and len(recs) >= limit:
                break
    return recs


def resolve_nifti_path(rec: Dict, root: Path) -> Path:
    # Prefer explicit image_path if present
    if "image_path" in rec and rec["image_path"]:
        p = Path(rec["image_path"])
        if p.is_absolute():
            return p
        return root / p
    case_id = str(rec.get("case_id") or "")
    name = Path(case_id).name
    if not name.endswith(".nii") and not name.endswith(".nii.gz"):
        name = f"{name}.nii.gz"
    return root / name


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


def load_volume(nifti_path: Path, device: torch.device) -> torch.Tensor:
    img = sitk.ReadImage(str(nifti_path))
    # Match Merlin preprocessing: RAS orientation + spacing 1.5x1.5x3
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

    arr = sitk.GetArrayFromImage(img).astype("float32")  # z,y,x
    arr = np.clip(arr, -1000, 1000)
    arr = (arr - (-1000)) / (1000 - (-1000))  # [0,1]
    arr = center_pad_crop(arr, (160, 224, 224))
    vol = torch.from_numpy(arr)[None, None]  # [1,1,D,H,W]
    return vol.to(device)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # If resuming, auto-infer skip from existing output lines
    if args.resume and args.output.exists():
        existing = sum(1 for _ in args.output.open())
        if existing > args.skip:
            args.skip = existing

    records = load_jsonl(args.dataset, skip=args.skip, limit=args.limit)
    if not records:
        raise SystemExit("No records loaded.")

    model = Merlin(RadiologyReport=True).to(device)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.resume else "w"
    with args.output.open(mode) as out_f, torch.no_grad():
        for rec in tqdm(records, desc="Running Merlin"):
            nifti_path = resolve_nifti_path(rec, args.nifti_root)
            if not nifti_path.is_file():
                raise FileNotFoundError(f"Missing NIfTI: {nifti_path}")
            vol = load_volume(nifti_path, device)

            question = rec.get("question") or "Describe the findings in this CT."
            prompt = SYSTEM_PROMPT.format(question=question)

            if args.temperature <= 0:
                gen_kwargs = {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": False,
                }
            else:
                gen_kwargs = {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "do_sample": True,
                }

            generated = model.generate(vol, [prompt], **gen_kwargs)
            # model.generate returns a list of strings
            prediction = generated[0] if isinstance(generated, list) else str(generated)
            prediction = prediction.strip()

            out_rec = {
                "case_id": rec.get("case_id"),
                "image_path": str(nifti_path),
                "question": question,
                "prediction": prediction,
            }
            if "answer" in rec:
                out_rec["answer"] = rec["answer"]
            out_f.write(json.dumps(out_rec) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())


if __name__ == "__main__":
    main()
