#!/usr/bin/env python3
"""Run Med3DVLM inference on the custom spatial benchmark JSONL."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict

import numpy as np
import SimpleITK as sitk
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Med3DVLM custom dataset evaluator")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to JSONL with keys image_path/question[/answer]",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save predictions JSONL",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Local path or Hugging Face repo id for Med3DVLM (e.g., models/Med3DVLM-Qwen-2.5-7B or MagicXin/Med3DVLM-Qwen-2.5-7B)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device identifier (cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional root directory to prepend to relative image paths",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Torch dtype for inference",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per sample",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit number of samples",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip this many samples from the start",
    )
    return parser.parse_args()


def load_jsonl(path: Path, skip: int = 0, limit: int | None = None) -> List[Dict]:
    records: List[Dict] = []
    with path.open() as f:
        for idx, line in enumerate(f):
            if idx < skip:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def load_volume(image_path: Path) -> np.ndarray:
    if image_path.suffix == ".npy":
        return np.load(image_path)
    return sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))


def prepare_image(image_path: Path, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    volume = load_volume(image_path)
    if volume.ndim == 3:
        volume = np.expand_dims(volume, axis=0)  # (1, D, H, W)
    image_pt = torch.from_numpy(volume).unsqueeze(0).to(dtype=dtype, device=device)
    return image_pt


def resolve_image_path(raw_path: str, image_root: Path | None) -> Path:
    """
    Resolve image paths.
    - If raw_path exists, return it.
    - Else, if image_root provided and raw_path looks like 'valid_1_a_1.nii.gz',
      construct <root>/<valid_1>/<valid_1_a>/<valid_1_a_1.nii.gz>.
    """
    candidate = Path(raw_path)
    if candidate.is_file():
        return candidate
    if image_root is None:
        return candidate
    name = candidate.name or raw_path
    stem = name.replace(".nii.gz", "")
    tokens = stem.split("_")
    if len(tokens) >= 3 and tokens[0] == "valid":
        patient = "_".join(tokens[:2])  # valid_1
        series = "_".join(tokens[:3])   # valid_1_a
        derived = image_root / patient / series / name
        if derived.is_file():
            return derived
    # fallback: assume direct child
    fallback = image_root / name
    return fallback


def main() -> None:
    args = parse_args()
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    dataset = load_jsonl(args.dataset, skip=args.skip, limit=args.limit)
    if not dataset:
        raise SystemExit("Dataset is empty after applying skip/limit.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        padding_side="right",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=str(device),
        trust_remote_code=True,
    )
    proj_out_num = (
        model.get_model().config.proj_out_num
        if hasattr(model.get_model().config, "proj_out_num")
        else 256
    )
    prompt_prefix = "<im_patch>" * proj_out_num

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out_f:
        for record in tqdm(dataset, desc="Running Med3DVLM"):
            raw_path = record.get("image_path") or record.get("case_id")
            if not raw_path:
                raise ValueError("Record missing image_path/case_id field")
            image_path = resolve_image_path(raw_path, args.image_root)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            question = record.get("question") or "Describe the findings of the medical image you see."
            input_text = prompt_prefix + question
            inputs = tokenizer(input_text, return_tensors="pt")["input_ids"].to(device=device)
            image_tensor = prepare_image(image_path, dtype=dtype, device=device)
            generation = model.generate(
                images=image_tensor,
                inputs=inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
            )
            prediction = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
            result = {
                "image_path": record.get("image_path"),
                "question": question,
                "prediction": prediction,
            }
            if "answer" in record:
                result["answer"] = record["answer"]
            out_f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
