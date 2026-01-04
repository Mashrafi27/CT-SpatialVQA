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

import nibabel as nib
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


def load_volume(nifti_path: Path, device: torch.device) -> torch.Tensor:
    img = nib.load(str(nifti_path))
    data = img.get_fdata().astype("float32")
    # Ensure shape [D,H,W]; add batch & channel -> [1,1,D,H,W]
    if data.ndim == 4:
        data = data[..., 0]
    vol = torch.from_numpy(data)[None, None]  # [1,1,D,H,W]
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
