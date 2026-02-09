#!/usr/bin/env python3
"""Convert QA JSON (case_id -> qa_pairs) to JSONL for inference.

Supports mapping from either an existing JSONL (case_id -> image_path)
or by scanning a NIfTI root for .nii.gz files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert QA JSON to JSONL with image_path.")
    p.add_argument(
        "--qa-json",
        required=True,
        type=Path,
        help="Path to QA JSON (case_id -> {qa_pairs}).",
    )
    p.add_argument(
        "--image-map-jsonl",
        default=None,
        type=Path,
        help="JSONL file providing case_id -> image_path mapping.",
    )
    p.add_argument(
        "--nifti-root",
        default=None,
        type=Path,
        help="Root folder containing .nii.gz files (searched recursively).",
    )
    p.add_argument(
        "--output-jsonl",
        required=True,
        type=Path,
        help="Output JSONL path.",
    )
    p.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip cases missing from the image map instead of erroring.",
    )
    return p.parse_args()


def build_image_map_from_jsonl(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for rec in iter_jsonl(path):
        cid = rec.get("case_id")
        img = rec.get("image_path")
        if cid and img and cid not in mapping:
            mapping[cid] = img
    return mapping


def build_image_map_from_nifti(root: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for p in root.rglob("*.nii.gz"):
        mapping[p.name] = str(p)
    return mapping


def main() -> None:
    args = parse_args()
    qa_data = json.loads(args.qa_json.read_text(encoding="utf-8"))

    image_map: Dict[str, str] = {}
    if args.image_map_jsonl:
        image_map = build_image_map_from_jsonl(args.image_map_jsonl)
    if args.nifti_root:
        image_map = build_image_map_from_nifti(args.nifti_root)

    if not image_map:
        raise SystemExit("No image mapping provided. Use --image-map-jsonl or --nifti-root.")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing = 0
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for case_id, payload in qa_data.items():
            image_path = image_map.get(case_id)
            if not image_path:
                missing += 1
                if args.skip_missing:
                    continue
                raise SystemExit(f"Missing image_path for case_id: {case_id}")
            for qa in payload.get("qa_pairs", []):
                rec = {
                    "case_id": case_id,
                    "image_path": image_path,
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} QA rows to {args.output_jsonl} (missing cases: {missing})")


if __name__ == "__main__":
    main()
