#!/usr/bin/env python3
"""Run M3D-LaMed inference on the spatial QA JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3D-LaMed custom evaluator")
    p.add_argument("--dataset", type=Path, required=True, help="JSONL with image_path/question[/answer]")
    p.add_argument("--output", type=Path, required=True, help="Path to write predictions JSONL")
    p.add_argument(
        "--model-path",
        type=str,
        default="GoodBaiBai88/M3D-LaMed-Phi-3-4B",
        help="HF repo id or local path for M3D-LaMed model",
    )
    p.add_argument("--device", default="cuda", help="cuda, cuda:0, cpu")
    p.add_argument(
        "--dtype",
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Torch dtype for model/image",
    )
    p.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=0.9, help="Top-p for sampling")
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    p.add_argument("--skip", type=int, default=0, help="Skip first N samples")
    p.add_argument("--resume", action="store_true", help="Resume from existing output file")
    return p.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_records(path: Path, skip: int, limit: int | None) -> List[Dict]:
    records: List[Dict] = []
    for idx, rec in enumerate(iter_jsonl(path)):
        if idx < skip:
            continue
        records.append(rec)
        if limit is not None and len(records) >= limit:
            break
    return records


def resolve_resume_skip(output_path: Path) -> int:
    if not output_path.exists():
        return 0
    with output_path.open() as f:
        return sum(1 for _ in f)


def load_image(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected (1,D,H,W) or (D,H,W), got {arr.shape}")
    return arr


def get_proj_out_num(model: torch.nn.Module) -> int:
    for obj in (model, getattr(model, "get_model", lambda: None)()):
        if obj is None:
            continue
        config = getattr(obj, "config", None)
        if config is not None and hasattr(config, "proj_out_num"):
            return int(config.proj_out_num)
    return 256


def main() -> None:
    args = parse_args()
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    resume_skip = 0
    if args.resume:
        resume_skip = resolve_resume_skip(args.output)
    records = load_records(args.dataset, skip=args.skip + resume_skip, limit=args.limit)
    if not records:
        raise SystemExit("No records loaded.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=str(device),
        trust_remote_code=True,
    )
    proj_out_num = get_proj_out_num(model)
    image_tokens = "<im_patch>" * proj_out_num

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(output_mode) as out_f:
        for record in tqdm(records, desc="Running M3D-LaMed"):
            raw_path = record.get("image_path") or record.get("case_id")
            if not raw_path:
                raise ValueError("Record missing image_path/case_id")
            image_path = Path(raw_path)
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing image: {image_path}")

            question = record.get("question") or "Describe this medical image."
            input_text = image_tokens + question
            input_id = tokenizer(input_text, return_tensors="pt")["input_ids"].to(device=device)

            image_np = load_image(image_path)
            image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)

            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
            }
            if args.temperature <= 0:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs.update(
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

            output = model.generate(image_pt, input_id, **gen_kwargs)
            if isinstance(output, tuple):
                output = output[0]
            prediction = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

            result = {
                "image_path": record.get("image_path"),
                "question": question,
                "prediction": prediction,
            }
            if "answer" in record:
                result["answer"] = record["answer"]
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()


if __name__ == "__main__":
    main()
