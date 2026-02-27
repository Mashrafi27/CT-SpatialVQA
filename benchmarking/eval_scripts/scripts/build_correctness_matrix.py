#!/usr/bin/env python3
"""Build a question-by-model correctness matrix from LLM judge outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


EVAL_SUFFIXES = [
    "_gemini2.5_eval_full.json",
    "_gemini2.5_eval_fixed.json",
    "_gemini2.5_eval.json",
    "_gemini_eval.json",
    "_gpt_eval.json",
    "_qwen_eval.json",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create correctness matrix CSV (questions x models).")
    p.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("3D_VLM_Spatial/reports"),
        help="Directory containing *_eval*.json files.",
    )
    p.add_argument(
        "--eval-globs",
        nargs="+",
        default=[
            "*_gemini2.5_eval_full.json",
            "*_gpt_eval.json",
            "*_qwen_eval.json",
        ],
        help="Globs for eval files to include (default: gemini2.5 full + gpt + qwen).",
    )
    p.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("3D_VLM_Spatial/reports"),
        help="Directory containing *_predictions*.jsonl files.",
    )
    p.add_argument(
        "--source-json",
        type=Path,
        default=Path("3D_VLM_Spatial/qa_generation_v2/spatial_qa_filtered_full.json"),
        help="Source JSON used to build stable question ids.",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("3D_VLM_Spatial/reports/correctness_matrix.csv"),
        help="Output CSV path.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output with matrix.",
    )
    p.add_argument(
        "--require-judges",
        type=int,
        default=3,
        help="Require this many judge files per model (default: 3).",
    )
    return p.parse_args()


def load_eval_items(path: Path) -> Tuple[List[dict], str]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        if "jury" in data:
            items = data["jury"]
            key = "jury_is_correct"
        elif "items" in data:
            items = data["items"]
            key = "is_correct"
        else:
            items = []
            key = "is_correct"
    else:
        items = data
        key = "is_correct"
    return items, key


def model_name_from_eval(path: Path) -> str:
    name = path.name
    for suf in EVAL_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return re.sub(r"_eval.*\\.json$", "", name)


def find_predictions_file(pred_dir: Path, model_name: str) -> Path | None:
    candidates = [
        f"{model_name}_predictions_full.jsonl",
        f"{model_name}_predictions.jsonl",
        f"{model_name.replace('-', '_')}_predictions_full.jsonl",
        f"{model_name.replace('-', '_')}_predictions.jsonl",
        f"{model_name.replace('_', '-')}_predictions_full.jsonl",
        f"{model_name.replace('_', '-')}_predictions.jsonl",
    ]
    for cand in candidates:
        p = pred_dir / cand
        if p.exists():
            return p
    return None


def normalize_image_id(raw: str) -> str:
    name = Path(raw).name
    if name.endswith(".nii.gz"):
        return name
    if name.endswith(".npy"):
        return name[:-4] + ".nii.gz"
    if name.endswith(".npz"):
        return name[:-4] + ".nii.gz"
    return name + ".nii.gz"


def build_qid_map(source_json: Path) -> Dict[Tuple[str, str, str], List[str]]:
    data = json.loads(source_json.read_text())
    qid_map: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for image_id, entry in data.items():
        for idx, qa in enumerate(entry.get("qa_pairs", [])):
            question = (qa.get("question") or "").strip()
            answer = (qa.get("answer") or "").strip()
            qid = f"{image_id}::{idx}"
            qid_map[(image_id, question, answer)].append(qid)
    return qid_map


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_qid_queue(
    pred_path: Path, qid_map: Dict[Tuple[str, str, str], List[str]]
) -> Dict[Tuple[str, str], List[str]]:
    """Create a queue of qids per (question, answer) in prediction order."""
    queues: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    counters: Dict[Tuple[str, str, str], int] = defaultdict(int)
    idx = 0
    for rec in iter_jsonl(pred_path):
        image_id = rec.get("case_id") or rec.get("image_path")
        if image_id:
            image_id = normalize_image_id(str(image_id))
        question = (rec.get("question") or "").strip()
        answer = (rec.get("answer") or "").strip()
        key_full = (image_id or "", question, answer)
        if key_full in qid_map and qid_map[key_full]:
            pos = counters[key_full]
            if pos < len(qid_map[key_full]):
                qid = qid_map[key_full][pos]
            else:
                qid = f"{image_id or 'unknown'}::{idx}"
            counters[key_full] += 1
        else:
            qid = f"{image_id or 'unknown'}::{idx}"
        queues[(question, answer)].append(qid)
        idx += 1
    return queues


def main() -> None:
    args = parse_args()
    eval_maps: List[Dict[str, Path]] = []
    for glob in args.eval_globs:
        files = sorted(args.eval_dir.glob(glob))
        if not files:
            print(f"[WARN] No eval files matched: {glob}")
            eval_maps.append({})
            continue
        m: Dict[str, Path] = {}
        for f in files:
            m[model_name_from_eval(f)] = f
        eval_maps.append(m)
    if all(not m for m in eval_maps):
        raise SystemExit("No eval files matched any glob.")

    base_qid_map = build_qid_map(args.source_json)
    matrix: Dict[str, Dict[str, bool | None]] = defaultdict(dict)
    models: List[str] = []

    model_names = sorted({name for m in eval_maps for name in m.keys()})
    for model_name in model_names:
        judge_paths = [m.get(model_name) for m in eval_maps]
        if any(p is None for p in judge_paths):
            print(f"[WARN] Missing one or more judge files for {model_name}; skipping.")
            continue
        if args.require_judges and len(judge_paths) < args.require_judges:
            print(f"[WARN] Not enough judge files for {model_name}; skipping.")
            continue

        pred_path = find_predictions_file(args.predictions_dir, model_name)
        if pred_path is None:
            print(f"[WARN] No predictions file for {model_name}; skipping.")
            continue

        qid_queue = build_qid_queue(pred_path, base_qid_map)
        judge_items: List[List[dict]] = []
        judge_keys: List[str] = []
        for p in judge_paths:
            items, key = load_eval_items(p)
            judge_items.append(items)
            judge_keys.append(key)
        min_len = min([len(j) for j in judge_items])
        if any(len(j) != min_len for j in judge_items):
            print(f"[WARN] Length mismatch across judges for {model_name}: {[len(j) for j in judge_items]}; using min length")

        models.append(model_name)
        matched = 0
        for i in range(min_len):
            votes = []
            for items, key in zip(judge_items, judge_keys):
                val = items[i].get(key)
                votes.append(1.0 if val is True else 0.0)
            avg = sum(votes) / len(votes) if votes else 0.0
            # Floor to 2 decimals to match 0.33/0.66 style.
            avg = math.floor(avg * 100.0) / 100.0
            question = (judge_items[0][i].get("question") or "").strip()
            answer = (judge_items[0][i].get("answer") or "").strip()
            qid_list = qid_queue.get((question, answer)) or []
            if qid_list:
                qid = qid_list.pop(0)
                matched += 1
            else:
                qid = f"unknown::{model_name}::{i}"
            matrix[qid][model_name] = avg
        if matched < min_len:
            print(f"[WARN] Matched {matched}/{min_len} eval items to source questions for {model_name}.")

    if not models:
        raise SystemExit("No models processed. Check eval/prediction filenames.")

    # Stable row order by qid
    row_ids = sorted(matrix.keys())
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id"] + models)
        for qid in row_ids:
            row = [qid]
            for m in models:
                val = matrix[qid].get(m)
                if val is None:
                    row.append("")
                else:
                    row.append(f"{float(val):.2f}")
            writer.writerow(row)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"models": models, "matrix": matrix}, indent=2))

    print(f"Wrote {len(row_ids)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
