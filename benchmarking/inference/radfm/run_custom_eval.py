#!/usr/bin/env python3
"""Run RadFM inference on spatial QA JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import LlamaTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RadFM custom evaluator")
    p.add_argument("--dataset", type=Path, required=True, help="JSONL with image_path/question[/answer]")
    p.add_argument("--output", type=Path, required=True, help="Path to write predictions JSONL")
    p.add_argument("--model-dir", type=Path, required=True, help="Directory containing RadFM checkpoint files")
    p.add_argument("--tokenizer-dir", type=Path, default=None, help="Tokenizer directory (defaults to model-dir/Language_files)")
    p.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint .bin path (defaults to model-dir/pytorch_model.bin)")
    p.add_argument("--device", default="cuda", help="cuda, cuda:0, cpu")
    p.add_argument(
        "--dtype",
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Torch dtype for inference",
    )
    p.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=0.9, help="Top-p")
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


def get_tokenizer(tokenizer_path: Path, max_img_size: int = 100, image_num: int = 32) -> Tuple[LlamaTokenizer, List[str]]:
    image_padding_tokens: List[str] = []
    text_tokenizer = LlamaTokenizer.from_pretrained(str(tokenizer_path))
    special_token = {"additional_special_tokens": ["<image>", "</image>"]}
    for i in range(max_img_size):
        image_padding_token = ""
        for j in range(image_num):
            image_token = "<image" + str(i * image_num + j) + ">"
            image_padding_token = image_padding_token + image_token
            special_token["additional_special_tokens"].append(image_token)
        image_padding_tokens.append(image_padding_token)
    text_tokenizer.add_special_tokens(special_token)
    text_tokenizer.pad_token_id = 0
    text_tokenizer.bos_token_id = 1
    text_tokenizer.eos_token_id = 2
    return text_tokenizer, image_padding_tokens


def load_image(path: Path) -> torch.Tensor:
    arr = np.load(path).astype("float32")
    if arr.ndim == 3:
        arr = np.stack([arr, arr, arr], axis=0)
    if arr.ndim != 4:
        raise ValueError(f"Expected (3,H,W,D) or (H,W,D), got {arr.shape}")
    return torch.from_numpy(arr)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent / "repo"
    quick_demo = repo_root / "Quick_demo"
    sys.path.insert(0, str(quick_demo))

    try:
        from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM  # type: ignore
    except ImportError as exc:
        raise SystemExit(f"Failed to import RadFM model. Check submodule at {quick_demo}.") from exc

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    tokenizer_dir = args.tokenizer_dir or (args.model_dir / "Language_files")
    checkpoint = args.checkpoint or (args.model_dir / "pytorch_model.bin")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer dir not found: {tokenizer_dir}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    resume_skip = 0
    if args.resume:
        resume_skip = resolve_resume_skip(args.output)
    records = load_records(args.dataset, skip=args.skip + resume_skip, limit=args.limit)
    if not records:
        raise SystemExit("No records loaded.")

    tokenizer, image_padding_tokens = get_tokenizer(tokenizer_dir)
    model = MultiLLaMAForCausalLM(lang_model_path=str(tokenizer_dir))
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    if "embedding_layer.bert_model.embeddings.position_ids" in ckpt:
        ckpt.pop("embedding_layer.bert_model.embeddings.position_ids")
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(output_mode) as out_f:
        for record in tqdm(records, desc="Running RadFM"):
            raw_path = record.get("image_path") or record.get("case_id")
            if not raw_path:
                raise ValueError("Record missing image_path/case_id")
            image_path = Path(raw_path)
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing image: {image_path}")

            question = record.get("question") or "Describe this medical image."
            text = "<image>" + image_padding_tokens[0] + "</image>" + question
            lang_x = tokenizer(text, max_length=2048, truncation=True, return_tensors="pt")["input_ids"].to(device=device)

            vision_x = load_image(image_path).to(device=device, dtype=dtype)
            vision_x = vision_x.unsqueeze(0).unsqueeze(0)

            with torch.inference_mode():
                generation = model.generate(lang_x, vision_x)
            if args.max_new_tokens:
                generation = generation[:, : args.max_new_tokens]
            prediction = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]

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
