#!/usr/bin/env python3
"""
Run VILA-M3 with the VISTA3D expert over full 3D CT volumes.

This script always runs VISTA3D on each volume (3D NIfTI), then feeds the
expert's text summary + segmentation overlay into VILA-M3 for QA.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import nibabel as nib
from tqdm import tqdm

# Add MONAI VLM repo + demo utilities to the import path.
REPO_ROOT = Path(__file__).resolve().parent / "repo"
DEMO_ROOT = REPO_ROOT / "m3" / "demo"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DEMO_ROOT))

from experts.expert_monai_vista3d import ExpertVista3D  # type: ignore[import]
from experts.utils import get_slice_filenames  # type: ignore[import]
from gradio_m3 import M3Generator  # type: ignore[import]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VILA-M3 + VISTA3D evaluator")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="JSONL with case_id/question[/answer] (e.g., spatial_qa_processed.jsonl)",
    )
    parser.add_argument(
        "--nifti-root",
        type=Path,
        required=True,
        help="Root containing original NIfTI files (valid_fixed)",
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
        help="HF repo id or local path for VILA-M3",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("3D_VLM_Spatial/reports/vista3d_cache"),
        help="Where to cache VISTA3D outputs per case",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Optional fixed axial slice index; default uses middle slice",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per QA",
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
        help="Optionally limit number of QA pairs",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip this many QA pairs from the start",
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


def resolve_nifti_path(case_id: str, root: Path) -> Path:
    name = Path(case_id).name
    if name.endswith(".npy"):
        name = name.replace(".npy", ".nii.gz")
    if not name.endswith(".nii.gz"):
        name = f"{name}.nii.gz"
    stem = name.replace(".nii.gz", "")
    tokens = stem.split("_")
    if len(tokens) >= 3 and tokens[0] == "valid":
        patient = "_".join(tokens[:2])  # valid_1
        series = "_".join(tokens[:3])   # valid_1_a
        return root / patient / series / name
    return root / name


def get_depth(nifti_path: Path) -> int:
    img = nib.load(str(nifti_path))
    shape = img.shape
    if len(shape) < 3:
        return 1
    return int(shape[2])


def run_vista3d(
    expert: ExpertVista3D,
    nifti_path: Path,
    case_dir: Path,
    slice_index: int,
) -> Dict[str, str | int]:
    case_dir.mkdir(parents=True, exist_ok=True)
    summary_path = case_dir / "vista3d_summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text())

    text_output, seg_image, _ = expert.run(
        img_file=str(nifti_path),
        image_url="",
        input="<VISTA3D(everything)>",
        output_dir=str(case_dir),
        slice_index=slice_index,
        prompt="",
    )
    slice_image = case_dir / get_slice_filenames(str(nifti_path), slice_index)

    summary = {
        "text_output": text_output,
        "seg_image": str(seg_image),
        "slice_image": str(slice_image),
        "slice_index": slice_index,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def build_messages(question: str, summary: Dict[str, str | int]) -> List[Dict]:
    instruction = f"Use this result to respond to this prompt:\n{question}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"<image> {question}"},
                {"type": "image_path", "image_path": summary["slice_image"]},
            ],
        },
        {
            "role": "expert",
            "content": [
                {"type": "text", "text": summary["text_output"]},
                {"type": "image_path", "image_path": summary["seg_image"]},
            ],
        },
        {
            "role": "expert",
            "content": [{"type": "text", "text": instruction}],
        },
    ]


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.dataset, skip=args.skip, limit=args.limit)
    if not records:
        raise SystemExit("Dataset is empty after applying skip/limit.")

    # Group QA pairs by case to avoid re-running VISTA3D per question.
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for rec in records:
        case_id = rec.get("case_id") or Path(rec.get("image_path", "")).name
        grouped[str(case_id)].append(rec)

    expert = ExpertVista3D()
    generator = M3Generator(source="huggingface", model_path=args.model_path, conv_mode="llama_3")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    with args.output.open("w") as out_f:
        for case_id, case_records in tqdm(grouped.items(), desc="Running VISTA3D + VILA-M3"):
            nifti_path = resolve_nifti_path(case_id, args.nifti_root)
            if not nifti_path.is_file():
                raise FileNotFoundError(f"Missing NIfTI volume: {nifti_path}")

            depth = get_depth(nifti_path)
            slice_index = args.slice_index if args.slice_index is not None else depth // 2
            case_dir = args.cache_dir / Path(case_id).stem.replace(".nii", "")
            summary = run_vista3d(expert, nifti_path, case_dir, slice_index)

            for record in case_records:
                question = record.get("question") or "Describe the findings in this CT volume."
                messages = build_messages(question, summary)
                response = generator.generate_response(
                    messages=generator.squash_expert_messages_into_user(messages),
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    system_prompt=None,
                )

                result = {
                    "case_id": record.get("case_id"),
                    "image_path": str(nifti_path),
                    "question": question,
                    "prediction": response,
                }
                if "answer" in record:
                    result["answer"] = record["answer"]
                out_f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()

