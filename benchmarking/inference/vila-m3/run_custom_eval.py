#!/usr/bin/env python3
"""
Run VILA-M3 inference on the custom spatial CT-RATE benchmark.

This mirrors the Med3DVLM `run_custom_eval.py` but uses the MONAI VLM
stack (VILA-M3) and the LLaVA-style interface.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

# Make the MONAI VLM repo importable: m3, llava, etc.
REPO_ROOT = Path(__file__).resolve().parent / "repo"
sys.path.insert(0, str(REPO_ROOT))

from llava.constants import IMAGE_TOKEN_INDEX  # type: ignore[import]
from llava.conversation import SeparatorStyle, conv_templates  # type: ignore[import]
from llava.mm_utils import (  # type: ignore[import]
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model  # type: ignore[import]
from llava.utils import disable_torch_init  # type: ignore[import]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VILA-M3 custom CT-RATE evaluator")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="JSONL with image_path/question[/answer] (e.g., spatial_qa_processed.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path for predictions",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="MONAI/Llama3-VILA-M3-8B",
        help="HF repo id or local path for VILA-M3 (e.g., MONAI/Llama3-VILA-M3-8B)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device identifier (cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per sample",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (0.0 for greedy)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling",
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
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Optional fixed axial slice index; default uses middle slice",
    )
    return parser.parse_args()


def load_jsonl(path: Path, skip: int = 0, limit: int | None = None) -> List[Dict]:
    records: List[Dict] = []
    with path.open() as f:
        for idx, line in enumerate(f):
            if idx < skip:
                continue
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def volume_to_pil(volume_path: Path, slice_index: int | None = None) -> Image.Image:
    """
    Load a processed CT volume (.npy: (D,H,W) or (1,D,H,W)) and return a central
    axial slice as a PIL.Image in [0,255] uint8.
    """
    volume = np.load(volume_path)
    if volume.ndim == 4:
        volume = volume[0]
    if volume.ndim != 3:
        raise ValueError(f"Expected volume of shape (D,H,W) or (1,D,H,W), got {volume.shape}")
    depth, _, _ = volume.shape
    idx = slice_index if slice_index is not None else depth // 2
    idx = max(0, min(depth - 1, idx))
    slice_ = volume[idx].astype(np.float32)

    # Map from [-1,1] to [0,255]
    slice_ = (slice_ + 1.0) / 2.0
    slice_ = np.clip(slice_, 0.0, 1.0)
    slice_ = (slice_ * 255).astype(np.uint8)
    return Image.fromarray(slice_)


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.dataset, skip=args.skip, limit=args.limit)
    if not records:
        raise SystemExit("Dataset is empty after applying skip/limit.")

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, model_name)
    device = torch.device(args.device)
    model.to(device)

    conv_mode = "llama_3"
    base_conv = conv_templates[conv_mode]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out_f:
        for record in tqdm(records, desc="Running VILA-M3"):
            image_path = Path(record["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing volume: {image_path}")

            pil_image = volume_to_pil(image_path, slice_index=args.slice_index)
            images = [pil_image]
            images_tensor = process_images(images, image_processor, model.config).to(
                device=model.device, dtype=torch.float16
            )
            images_input = [images_tensor]

            question = record.get("question") or "Describe the findings in this CT volume."

            # Build LLaVA-style conversation with an <image> token
            conv = base_conv.copy()
            user_role, assistant_role = conv.roles
            user_text = f"<image> {question}"
            conv.append_message(user_role, user_text)
            if conv.sep_style == SeparatorStyle.LLAMA_3:
                conv.append_message(assistant_role, "")
            prompt_text = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt_text,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(device=model.device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_input,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping],
                    pad_token_id=tokenizer.eos_token_id,
                    min_new_tokens=2,
                )

            text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if text.endswith(stop_str):
                text = text[: -len(stop_str)].strip()

            result = {
                "case_id": record.get("case_id"),
                "image_path": record.get("image_path"),
                "question": question,
                "prediction": text,
            }
            if "answer" in record:
                result["answer"] = record["answer"]
            out_f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()

