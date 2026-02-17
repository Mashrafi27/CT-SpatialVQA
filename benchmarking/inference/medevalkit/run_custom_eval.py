#!/usr/bin/env python3
"""Run Lingshu (Qwen2.5-VL) on CT volumes by sampling 2D slices.

This script mirrors the MedGemma slice-based prompting but uses the Lingshu
model (Qwen2.5-VL backbone) from the Lingshu HF collection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import nibabel as nib
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing nibabel. Install with: pip install nibabel") from exc

try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing transformers with Qwen2.5-VL. Install a recent transformers.") from exc

try:
    from qwen_vl_utils import process_vision_info
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing qwen_vl_utils. Install with: pip install qwen-vl-utils") from exc

DEFAULT_WINDOWS: List[Tuple[int, int]] = [
    (-1024, 1024),  # wide
    (-135, 215),    # mediastinum
    (0, 80),        # brain
]

DEFAULT_INSTRUCTION = (
    "You are an instructor teaching medical students. You are analyzing a "
    "contiguous block of CT slices. Please review the slices provided below carefully."
)

DEFAULT_QUERY_SUFFIX = (
    "\n\nBased on the visual evidence in the slices provided above, "
    "answer the question below. Provide concise reasoning and conclude with a final answer."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lingshu on CT volumes with slice sampling.")
    parser.add_argument("--dataset", required=True, help="Input JSONL with case_id/question/answer.")
    parser.add_argument("--nifti-root", required=True, help="Root of raw NIfTI volumes.")
    parser.add_argument("--output", required=True, help="Output JSONL for predictions.")
    parser.add_argument("--model-id", default="lingshu-medical-mllm/Lingshu-7B", help="HF model id.")
    parser.add_argument("--num-slices", type=int, default=32, help="Number of slices sampled uniformly.")
    parser.add_argument("--resize", type=int, default=0, help="Resize each slice to NxN (0 = keep original).")
    parser.add_argument("--resize-longest", type=int, default=0, help="Resize longest side to N (keeps aspect).")
    parser.add_argument("--pad-square", action="store_true", help="Pad resized slice to square (for keep-aspect).")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.001)
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


def build_messages(
    slice_rgbs: List[np.ndarray],
    instruction: str,
    question: str,
    query_suffix: str,
    add_slice_labels: bool = True,
) -> List[dict]:
    content = [{"type": "text", "text": instruction}]
    for i, rgb in enumerate(slice_rgbs, 1):
        content.append({"type": "image", "image": Image.fromarray(rgb)})
        if add_slice_labels:
            content.append({"type": "text", "text": f"SLICE {i}"})
    content.append({"type": "text", "text": f"{query_suffix}\n\nQuestion: {question}"})
    return [{"role": "user", "content": content}]


def load_volume(nifti_path: Path) -> np.ndarray:
    img = nib.load(str(nifti_path))
    vol = np.asarray(img.get_fdata())
    # nib loads x,y,z; transpose to z,y,x
    if vol.ndim == 3:
        vol = np.transpose(vol, (2, 1, 0))
    return vol


def _downsample_slices(slices: List[np.ndarray], keep: int) -> List[np.ndarray]:
    if keep <= 0 or keep >= len(slices):
        return slices
    if keep == 1:
        return [slices[len(slices) // 2]]
    idxs = np.linspace(0, len(slices) - 1, keep)
    return [slices[int(round(i))] for i in idxs]


def load_preprocessed(image_path: Path, keep_slices: int) -> List[np.ndarray] | None:
    if not image_path.exists():
        return None
    if image_path.suffix == ".npz":
        data = np.load(image_path)
        if "slices" not in data:
            raise ValueError(f"Missing 'slices' in {image_path}")
        slices = data["slices"]
        if slices.ndim != 4 or slices.shape[-1] != 3:
            raise ValueError(f"Unexpected slices shape {slices.shape} in {image_path}")
        slice_list = [slices[i] for i in range(slices.shape[0])]
        return _downsample_slices(slice_list, keep_slices)
    if image_path.suffix == ".npy":
        vol = np.load(image_path)
        if vol.ndim != 3:
            raise ValueError(f"Unexpected volume shape {vol.shape} in {image_path}")
        slice_idxs = sample_indices(vol.shape[0], keep_slices)
        return [make_rgb_window(vol[i]) for i in slice_idxs]
    return None


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


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
                    rec = json.loads(line)
                    processed.add((rec.get("case_id"), rec.get("question")))
                except Exception:
                    continue

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=True)

    total = sum(1 for _ in iter_jsonl(Path(args.dataset)))
    if args.limit and args.limit > 0:
        total = min(total, args.limit)

    with Path(args.dataset).open() as f_in, output_path.open("a") as f_out:
        for idx, line in enumerate(tqdm(f_in, total=total, desc="Running Lingshu")):
            if not line.strip():
                continue
            if args.limit and args.limit > 0 and idx >= args.limit:
                break
            ex = json.loads(line)
            case_id = ex.get("case_id")
            question = ex.get("question")
            answer = ex.get("answer")
            if (case_id, question) in processed:
                continue

            slice_rgbs = None
            image_path = ex.get("image_path")
            if image_path:
                slice_rgbs = load_preprocessed(Path(image_path), args.num_slices)

            if slice_rgbs is None:
                nifti_path = derive_nifti_path(nifti_root, case_id)
                if not nifti_path.exists():
                    raise FileNotFoundError(f"Missing NIfTI: {nifti_path}")
                vol = load_volume(nifti_path)
                slice_idxs = sample_indices(vol.shape[0], args.num_slices)
                slice_rgbs = [
                    make_rgb_window(vol[i], args.resize, args.resize_longest, args.pad_square)
                    for i in slice_idxs
                ]

            messages = build_messages(slice_rgbs, args.instruction, question, args.query_suffix)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(args.device) for k, v in inputs.items() if hasattr(v, "to")}

            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            if args.temperature == 0.0:
                gen_kwargs["do_sample"] = False

            generated_ids = model.generate(**inputs, **gen_kwargs)
            generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
            prediction = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            record = dict(
                case_id=case_id,
                question=question,
                answer=answer,
                prediction=prediction,
                model_id=args.model_id,
                num_slices=args.num_slices,
            )
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()


if __name__ == "__main__":
    main()
