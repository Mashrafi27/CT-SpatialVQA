#!/usr/bin/env python3
"""Run MedGemma on CT volumes by sampling 2D slices and prompting with images.

Implements the approach shown in medgemma/notebooks/high_dimensional_ct_hugging_face.ipynb:
- sample N slices uniformly across the volume (default 85)
- apply 3 CT windows and stack to RGB
- encode slices as inline images in the chat prompt
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import nibabel as nib
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing nibabel. Install with: pip install nibabel") from exc

from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_WINDOWS: List[Tuple[int, int]] = [
    (-1024, 1024),  # wide
    (-135, 215),    # mediastinum
    (0, 80),        # brain
]

DEFAULT_INSTRUCTION = (
    "You are an instructor teaching medical students. You are "
    "analyzing a contiguous block of CT slices. Please review the slices provided "
    "below carefully."
)

DEFAULT_QUERY_SUFFIX = (
    "\n\nBased on the visual evidence in the slices provided above, "
    "answer the question below. Provide concise reasoning and conclude with a final answer."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedGemma on CT volumes with slice sampling.")
    parser.add_argument("--dataset", required=True, help="Input JSONL with case_id/question/answer.")
    parser.add_argument("--nifti-root", required=True, help="Root of raw NIfTI volumes.")
    parser.add_argument("--output", required=True, help="Output JSONL for predictions.")
    parser.add_argument("--model-id", default="google/medgemma-1.5-4b-it", help="HF model id.")
    parser.add_argument("--num-slices", type=int, default=85, help="Number of slices sampled uniformly.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--query-suffix", default=DEFAULT_QUERY_SUFFIX)
    return parser.parse_args()


def derive_nifti_path(nifti_root: Path, case_id: str) -> Path:
    """Convert case_id like valid_1_a_1.nii.gz to valid_fixed path structure."""
    stem = case_id.replace(".nii.gz", "")
    subdir = stem.rsplit("_", 1)[0]  # valid_1_a
    base = subdir.split("_a")[0]     # valid_1
    return nifti_root / base / subdir / case_id


def sample_indices(n_slices: int, num_samples: int) -> List[int]:
    if num_samples <= 0:
        return []
    if n_slices <= num_samples:
        return list(range(n_slices))
    # evenly spaced indices across volume
    idxs = np.linspace(0, n_slices - 1, num_samples)
    return [int(round(i)) for i in idxs]


def window_slice(slice_2d: np.ndarray, window: Tuple[int, int]) -> np.ndarray:
    lo, hi = window
    x = np.clip(slice_2d, lo, hi).astype(np.float32)
    x -= lo
    x /= (hi - lo)
    x *= 255.0
    return np.round(x, 0).astype(np.uint8)


def make_rgb_window(slice_2d: np.ndarray) -> np.ndarray:
    chans = [window_slice(slice_2d, w) for w in DEFAULT_WINDOWS]
    return np.stack(chans, axis=-1)


def encode_png(data: np.ndarray) -> str:
    """Encode uint8 HxWx3 to data URI string."""
    with Image.fromarray(data) as img:
        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        encoded = base64.b64encode(buf.getbuffer()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def build_messages(slice_rgbs: List[np.ndarray], instruction: str, question: str, query_suffix: str) -> List[dict]:
    content = [{"type": "text", "text": instruction}]
    for i, rgb in enumerate(slice_rgbs, 1):
        content.append({"type": "image", "image": encode_png(rgb)})
        content.append({"type": "text", "text": f"SLICE {i}"})
    content.append({"type": "text", "text": f"{query_suffix}\n\nQuestion: {question}"})
    return [{"role": "user", "content": content}]


def load_volume(nifti_path: Path) -> np.ndarray:
    img = nib.load(str(nifti_path))
    vol = np.asarray(img.get_fdata())
    # ensure z,y,x ordering; nibabel loads as x,y,z. transpose to z,y,x
    if vol.ndim == 3:
        vol = np.transpose(vol, (2, 1, 0))
    return vol


def main() -> None:
    args = parse_args()
    nifti_root = Path(args.nifti_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = set()
    if args.resume and output_path.exists():
        with output_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    processed.add(json.loads(line)["case_id"])
                except Exception:
                    continue

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    model_kwargs = dict(
        torch_dtype=dtype,
        device_map="auto",
        offload_buffers=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)

    with Path(args.dataset).open() as f_in, output_path.open("a") as f_out:
        for line in tqdm(f_in, total=sum(1 for _ in Path(args.dataset).open()), desc="Running MedGemma"):
            if not line.strip():
                continue
            ex = json.loads(line)
            case_id = ex["case_id"]
            if case_id in processed:
                continue
            question = ex["question"]
            answer = ex.get("answer")
            nifti_path = derive_nifti_path(nifti_root, case_id)
            if not nifti_path.exists():
                raise FileNotFoundError(f"Missing NIfTI: {nifti_path}")

            vol = load_volume(nifti_path)
            idxs = sample_indices(vol.shape[0], args.num_slices)
            slice_rgbs = [make_rgb_window(vol[i]) for i in idxs]

            messages = build_messages(slice_rgbs, args.instruction, question, args.query_suffix)
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                continue_final_message=False,
                return_tensors="pt",
                tokenize=True,
                return_dict=True,
            )
            inputs = inputs.to(model.device, dtype=dtype)

            gen_kwargs = {
                "do_sample": args.temperature > 0,
                "max_new_tokens": args.max_new_tokens,
            }
            if args.temperature > 0:
                gen_kwargs["temperature"] = args.temperature

            with torch.inference_mode():
                generated = model.generate(**inputs, **gen_kwargs)

            decoded = processor.post_process_image_text_to_text(generated, skip_special_tokens=True)[0]
            decoded_inputs = processor.post_process_image_text_to_text(inputs["input_ids"], skip_special_tokens=True)[0]
            if decoded.startswith(decoded_inputs):
                decoded = decoded[len(decoded_inputs):].lstrip()

            out = {
                "case_id": case_id,
                "question": question,
                "answer": answer,
                "prediction": decoded,
                "model_id": args.model_id,
                "num_slices": args.num_slices,
            }
            f_out.write(json.dumps(out) + "\n")
            f_out.flush()


if __name__ == "__main__":
    main()
