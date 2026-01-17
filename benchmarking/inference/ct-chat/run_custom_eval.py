#!/usr/bin/env python3
"""Run CT-CHAT inference on encoded CT embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Dict

import numpy as np
import torch
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.utils import disable_torch_init


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CT-CHAT custom evaluator")
    p.add_argument("--dataset", type=Path, required=True, help="JSONL with image_path/question[/answer]")
    p.add_argument("--output", type=Path, required=True, help="Predictions JSONL")
    p.add_argument("--model-path", type=str, required=True, help="Path to CT-CHAT model")
    p.add_argument("--model-base", type=str, required=True, help="Base model path")
    p.add_argument("--device", default="cuda", help="cuda, cuda:0, cpu")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--conv-mode", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--resume", action="store_true")
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


def main() -> None:
    args = parse_args()
    disable_torch_init()

    resume_skip = 0
    if args.resume:
        resume_skip = resolve_resume_skip(args.output)
    records = load_records(args.dataset, skip=args.skip + resume_skip, limit=args.limit)
    if not records:
        raise SystemExit("No records loaded.")

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        load_8bit=False,
        load_4bit=False,
        device=args.device,
    )

    if args.conv_mode is None:
        args.conv_mode = "llama3"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(output_mode) as out_f:
        for rec in tqdm(records, desc="Running CT-CHAT"):
            raw_path = rec.get("image_path") or rec.get("case_id")
            if not raw_path:
                raise ValueError("Record missing image_path/case_id")
            emb_path = Path(raw_path)
            if not emb_path.is_file():
                raise FileNotFoundError(f"Missing embedding file: {emb_path}")

            image = np.load(emb_path)["arr"]
            image_tensor = torch.tensor(image).to(model.device, dtype=torch.float16)
            image_size = image.size

            question = rec.get("question") or "Describe the CT volume."
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(model.device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            output_text = tokenizer.decode(output_ids[0]).strip()
            conv.messages[-1][-1] = output_text

            result = {
                "image_path": rec.get("image_path"),
                "question": question,
                "prediction": output_text,
            }
            if "answer" in rec:
                result["answer"] = rec["answer"]
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()


if __name__ == "__main__":
    main()
