#!/usr/bin/env python3
"""Run MedGemma on CT volumes by sampling 2D slices and prompting with images.

Implements the approach shown in medgemma/notebooks/high_dimensional_ct_hugging_face.ipynb:
- sample N slices uniformly across the volume (default 85)
- apply 3 CT windows and stack to RGB
- encode slices inline as base64 images in the chat prompt
"""
from __future__ import annotations

import argparse
import base64
import io
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
    parser.add_argument("--resize", type=int, default=0, help="Resize each slice to NxN (0 = keep original).")
    parser.add_argument("--resize-longest", type=int, default=0, help="Resize longest side to N (keeps aspect).")
    parser.add_argument("--pad-square", action="store_true", help="Pad resized slice to square (for keep-aspect).")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-new-tokens", type=int, default=4, help="Force minimum new tokens to generate.")
    parser.add_argument("--max-input-tokens", type=int, default=0, help="Override max input tokens (0 = auto from model config).")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N records (0 = all).")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--query-suffix", default=DEFAULT_QUERY_SUFFIX)
    return parser.parse_args()


def derive_nifti_path(nifti_root: Path, case_id: str) -> Path:
    """Convert case_id like valid_1_a_1.nii.gz to valid_fixed path structure."""
    stem = case_id.replace(".nii.gz", "")
    subdir = stem.rsplit("_", 1)[0]  # valid_1_a
    # Drop trailing split like _a/_b/_c to get base (valid_1)
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


def _encode_inline(rgb: np.ndarray, fmt: str = "jpeg") -> str:
    with io.BytesIO() as buf:
        with Image.fromarray(rgb) as img:
            img.save(buf, format=fmt)
        buf.seek(0)
        encoded = base64.b64encode(buf.getbuffer()).decode("utf-8")
    return f"data:image/{fmt};base64,{encoded}"


def build_messages(
    slice_rgbs: List[np.ndarray],
    instruction: str,
    question: str,
    query_suffix: str,
    add_slice_labels: bool = True,
) -> List[dict]:
    content = [{"type": "text", "text": instruction}]
    for i, rgb in enumerate(slice_rgbs, 1):
        content.append({"type": "image", "image": _encode_inline(rgb)})
        if add_slice_labels:
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
        dtype=dtype,
        device_map="auto",
        offload_buffers=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)

    # Determine context length for truncation
    max_ctx = None
    for attr in ("max_position_embeddings", "max_sequence_length"):
        max_ctx = getattr(model.config, attr, None)
        if max_ctx:
            break
    if max_ctx is None and hasattr(model.config, "text_config"):
        max_ctx = getattr(model.config.text_config, "max_position_embeddings", None)
    if args.max_input_tokens and args.max_input_tokens > 0:
        max_ctx = args.max_input_tokens

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
            slice_rgbs = [
                make_rgb_window(vol[i], resize=args.resize, resize_longest=args.resize_longest, pad_square=args.pad_square)
                for i in idxs
            ]

            messages = build_messages(
                slice_rgbs,
                args.instruction,
                question,
                args.query_suffix,
            )
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                continue_final_message=False,
                return_tensors="pt",
                tokenize=True,
                return_dict=True,
            )
            # Move tensors to device, keep integer tensors as int64
            inputs = {
                k: (v.to(model.device, dtype=dtype) if torch.is_floating_point(v) else v.to(model.device))
                for k, v in inputs.items()
            }
            # Avoid truncating input_ids: image tokens must align with images.
            if max_ctx:
                input_len = inputs["input_ids"].shape[1]
                if input_len + args.max_new_tokens > max_ctx:
                    print(
                        f"Warning: input length {input_len} + max_new_tokens {args.max_new_tokens} "
                        f"exceeds context {max_ctx}. Generation may be truncated."
                    )

            gen_kwargs = {
                "do_sample": args.temperature > 0,
                "max_new_tokens": args.max_new_tokens,
            }
            if args.min_new_tokens and args.min_new_tokens > 0:
                gen_kwargs["min_new_tokens"] = args.min_new_tokens
            if args.temperature > 0:
                gen_kwargs["temperature"] = args.temperature

            def _generate(batch, force_no_eos: bool = False, override_gen: dict | None = None):
                local_gen = dict(gen_kwargs)
                if override_gen:
                    local_gen.update(override_gen)
                return model.generate(
                    **batch,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    eos_token_id=None if force_no_eos else processor.tokenizer.eos_token_id,
                    **local_gen,
                )

            with torch.inference_mode():
                generated = _generate(inputs)
                # If no new tokens were generated, drop the final token and retry once
                if generated.shape[1] == inputs["input_ids"].shape[1]:
                    retry = dict(inputs)
                    retry["input_ids"] = retry["input_ids"][:, :-1]
                    if "attention_mask" in retry:
                        retry["attention_mask"] = retry["attention_mask"][:, :-1]
                    generated = _generate(retry)
                    inputs = retry
                # If still no new tokens, force a short generation without EOS stopping
                if generated.shape[1] == inputs["input_ids"].shape[1]:
                    forced = {"min_new_tokens": 4}
                    generated = _generate(inputs, force_no_eos=True, override_gen=forced)

            raw_output = processor.post_process_image_text_to_text(generated, skip_special_tokens=True)[0]
            decoded_inputs = processor.post_process_image_text_to_text(inputs["input_ids"], skip_special_tokens=True)[0]
            decoded = raw_output
            idx = decoded.find(decoded_inputs)
            if 0 <= idx <= 2:
                decoded = decoded[idx + len(decoded_inputs):]
            decoded = decoded.strip()

            out = {
                "case_id": case_id,
                "question": question,
                "answer": answer,
                "prediction": decoded,
                "prediction_raw": raw_output,
                "model_id": args.model_id,
                "num_slices": args.num_slices,
            }
            f_out.write(json.dumps(out) + "\n")
            f_out.flush()
            if args.limit and len(processed) + 1 >= args.limit:
                break


if __name__ == "__main__":
    main()
