#!/usr/bin/env python3
"""Run CT2Rep report-generation model on the spatial QA JSONL.

CT2Rep is a report-generation model (no question conditioning). This script
generates a report per case and reuses it for each question tied to that case.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

# Ensure CT2Rep repo + ctvit package are on sys.path
REPO_ROOT = Path(__file__).resolve().parent / "repo"
CT2REP_ROOT = REPO_ROOT / "CT2Rep"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CT2REP_ROOT))
CTVIT_ROOT = REPO_ROOT / "ctvit"
if (CTVIT_ROOT / "ctvit").exists():
    sys.path.insert(0, str(CTVIT_ROOT))

# CT2Rep repo imports
from ctvit.ctvit import CTViT  # noqa: F401  # required for model init side-effects
from CT2Rep.models.ct2rep import CT2RepModel
from CT2Rep.modules.tokenizers import Tokenizer as CT2RepTokenizer


class JsonTokenizer:
    """Tokenizer wrapper that loads token2idx/idx2token from json files."""

    def __init__(self, token2idx_path: Path, idx2token_path: Path) -> None:
        with token2idx_path.open() as f:
            self.token2idx = json.load(f)
        with idx2token_path.open() as f:
            idx2token = json.load(f)
        # idx2token keys may be strings; normalize to int
        self.idx2token = {int(k): v for k, v in idx2token.items()}

    def get_vocab_size(self) -> int:
        return len(self.token2idx)

    def decode(self, ids: Iterable[int]) -> str:
        words: List[str] = []
        for idx in ids:
            if idx <= 0:
                break
            token = self.idx2token.get(int(idx), "")
            if token:
                words.append(token)
        return " ".join(words)

    def decode_batch(self, ids_batch: Iterable[Iterable[int]]) -> List[str]:
        return [self.decode(ids) for ids in ids_batch]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CT2Rep custom dataset evaluator")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to spatial QA JSONL")
    parser.add_argument("--nifti-root", type=Path, default=None, help="Root for NIfTI volumes")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL")
    parser.add_argument("--ckpt", type=Path, required=True, help="CT2Rep checkpoint (.pth)")

    # Tokenizer options (choose one)
    parser.add_argument("--xlsx-file", type=Path, default=None, help="CT-RATE reports xlsx for tokenizer")
    parser.add_argument("--tokenizer-dir", type=Path, default=None, help="Dir with token2idx.json/idx2token.json")

    # Sampling / model args
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-seq-length", type=int, default=200)
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--sample-method", type=str, default="beam_search")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample-n", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--block-trigrams", type=int, default=1)

    # Preprocessing
    parser.add_argument("--scale-hu", action="store_true", help="Multiply by 1000 before HU clipping")
    parser.add_argument("--input-order", choices=["dhw", "hwd"], default=None,
                        help="Axis order for npy/npz inputs. Default: dhw for npy/npz, hwd for NIfTI.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from existing output JSONL")

    return parser.parse_args()


def resolve_image_path(record: Dict, nifti_root: Path | None) -> Path:
    if "image_path" in record:
        candidate = Path(record["image_path"])
        if candidate.is_file():
            return candidate
    if "image" in record:
        candidate = Path(record["image"])
        if candidate.is_file():
            return candidate
    case_id = record.get("case_id") or record.get("image_path") or ""
    name = Path(case_id).name
    if name.endswith(".nii.gz") and nifti_root is not None:
        stem = name.replace(".nii.gz", "")
        parts = stem.split("_")
        base = "_".join(parts[:2]) if len(parts) >= 2 else stem
        series = "_".join(parts[:3]) if len(parts) >= 3 else stem
        return nifti_root / base / series / name
    return Path(case_id)


def load_array(path: Path) -> Tuple[np.ndarray, str]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        arr = np.load(path)["arr_0"]
        return arr, "dhw"
    if suffix == ".npy":
        arr = np.load(path)
        return arr, "dhw"
    # assume NIfTI
    arr = np.asarray(nib.load(str(path)).get_fdata())
    return arr, "hwd"


def preprocess_ct2rep(arr: np.ndarray, order: str, scale_hu: bool) -> torch.Tensor:
    if order == "dhw":
        arr = np.transpose(arr, (1, 2, 0))
    arr = arr.astype(np.float32)
    if scale_hu:
        arr = arr * 1000.0
    hu_min, hu_max = -1000, 200
    arr = np.clip(arr, hu_min, hu_max)
    arr = ((arr + 400.0) / 600.0).astype(np.float32)

    # center crop / pad to (480, 480, 240) in (H, W, D)
    target_h, target_w, target_d = 480, 480, 240
    h, w, d = arr.shape

    h_start = max((h - target_h) // 2, 0)
    w_start = max((w - target_w) // 2, 0)
    d_start = max((d - target_d) // 2, 0)
    h_end = min(h_start + target_h, h)
    w_end = min(w_start + target_w, w)
    d_end = min(d_start + target_d, d)
    arr = arr[h_start:h_end, w_start:w_end, d_start:d_end]

    pad_h_before = (target_h - arr.shape[0]) // 2
    pad_h_after = target_h - arr.shape[0] - pad_h_before
    pad_w_before = (target_w - arr.shape[1]) // 2
    pad_w_after = target_w - arr.shape[1] - pad_w_before
    pad_d_before = (target_d - arr.shape[2]) // 2
    pad_d_after = target_d - arr.shape[2] - pad_d_before

    arr = np.pad(
        arr,
        ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after), (pad_d_before, pad_d_after)),
        mode="constant",
        constant_values=-1,
    )

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, D, H, W)
    return tensor


def load_tokenizer(args: argparse.Namespace):
    if args.tokenizer_dir:
        token2idx = args.tokenizer_dir / "token2idx.json"
        idx2token = args.tokenizer_dir / "idx2token.json"
        return JsonTokenizer(token2idx, idx2token)
    if args.xlsx_file:
        class ArgsWrap:
            threshold = 3
            xlsxfile = str(args.xlsx_file)
        return CT2RepTokenizer(ArgsWrap())
    raise SystemExit("Provide --tokenizer-dir or --xlsx-file for CT2Rep tokenizer.")


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    # strip DataParallel prefix if needed
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_existing_predictions(path: Path) -> Dict[Tuple[str, str], Dict]:
    existing: Dict[Tuple[str, str], Dict] = {}
    if not path.exists():
        return existing
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (record.get("case_id", ""), record.get("question", ""))
            existing[key] = record
    return existing


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    records = load_jsonl(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]

    existing = load_existing_predictions(args.output) if args.resume else {}
    model_args = argparse.Namespace(
        max_seq_length=args.max_seq_length,
        threshold=3,
        num_workers=0,
        batch_size=1,
        dataset_name="ct2rep_eval",
        d_model=512,
        d_ff=512,
        d_vf=512,
        num_heads=8,
        num_layers=3,
        dropout=0.1,
        logit_layers=1,
        bos_idx=0,
        eos_idx=0,
        pad_idx=0,
        use_bn=0,
        drop_prob_lm=0.5,
        rm_num_slots=3,
        rm_num_heads=8,
        rm_d_model=512,
        sample_method=args.sample_method,
        beam_size=args.beam_size,
        temperature=args.temperature,
        sample_n=args.sample_n,
        group_size=args.group_size,
        output_logsoftmax=1,
        decoding_constraint=0,
        block_trigrams=args.block_trigrams,
        n_gpu=1,
        epochs=1,
        save_dir="",
        record_dir="",
        save_period=1,
        monitor_mode="max",
        monitor_metric="BLEU_4",
        early_stop=50,
        optim="Adam",
        lr_ve=5e-5,
        lr_ed=1e-4,
        weight_decay=5e-5,
        amsgrad=True,
        lr_scheduler="StepLR",
        step_size=50,
        gamma=0.1,
        xlsxfile=str(args.xlsx_file) if args.xlsx_file else "",
        trainfolder="",
        validfolder="",
        resume=None,
    )

    tokenizer = load_tokenizer(args)
    model = CT2RepModel(model_args, tokenizer)
    load_checkpoint(model, args.ckpt)
    model.eval().to(device=device, dtype=dtype)

    case_cache: Dict[str, str] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a") as out_f:
        for record in tqdm(records, desc="Running CT2Rep"):
            key = (record.get("case_id", ""), record.get("question", ""))
            if key in existing:
                continue

            case_id = record.get("case_id", "")
            if case_id in case_cache:
                report = case_cache[case_id]
            else:
                image_path = resolve_image_path(record, args.nifti_root)
                arr, default_order = load_array(image_path)
                order = args.input_order or default_order
                tensor = preprocess_ct2rep(arr, order=order, scale_hu=args.scale_hu)
                images = tensor.unsqueeze(0).to(device=device, dtype=dtype)  # (1, 1, D, H, W)
                with torch.no_grad():
                    output = model(images, mode="sample")
                report = tokenizer.decode_batch(output.cpu().numpy())[0]
                case_cache[case_id] = report

            output_record = {
                "case_id": case_id,
                "question": record.get("question", ""),
                "answer": record.get("answer", ""),
                "prediction": report,
                "prediction_raw": report,
                "model_id": "ct2rep",
            }
            out_f.write(json.dumps(output_record) + "\n")
            out_f.flush()


if __name__ == "__main__":
    main()
