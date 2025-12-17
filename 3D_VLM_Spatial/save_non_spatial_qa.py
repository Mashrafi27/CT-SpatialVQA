#!/usr/bin/env python3
"""
Extract non-spatial QA pairs (based on progress annotations) into a separate JSON file.

Usage:
    python save_non_spatial_qa.py --progress progress.json --qa spatial_qa_output.json \
        --output reports/non_spatial_qa_output.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save QA pairs labeled non-spatial to a new JSON file.")
    parser.add_argument("--progress", default="progress.json", type=Path, help="Path to progress.json")
    parser.add_argument("--qa", default="spatial_qa_output.json", type=Path, help="Path to spatial_qa_output.json")
    parser.add_argument(
        "--output",
        default=Path("reports/non_spatial_qa_output.json"),
        type=Path,
        help="Output JSON path (same schema as spatial_qa_output).",
    )
    return parser.parse_args()


def collect_non_spatial(progress: Dict[str, dict], qa_data: Dict[str, dict]) -> Dict[str, dict]:
    collected: Dict[str, List[tuple]] = {}
    for user_cases in progress.values():
        for case_id, review in user_cases.items():
            answers = review.get("answers", [])
            if not answers or case_id not in qa_data:
                continue
            qa_pairs = qa_data[case_id].get("qa_pairs", [])
            for idx, entry in enumerate(answers):
                if isinstance(entry, dict) and entry.get("spatial") is False and idx < len(qa_pairs):
                    collected.setdefault(case_id, []).append((idx, qa_pairs[idx]))
    output: Dict[str, dict] = {}
    for case_id, entries in collected.items():
        ordered = [pair for _, pair in sorted(entries, key=lambda item: item[0])]
        output[case_id] = {"qa_pairs": ordered}
    return output


def main() -> None:
    args = parse_args()
    progress = json.loads(args.progress.read_text(encoding="utf-8"))
    qa_data = json.loads(args.qa.read_text(encoding="utf-8"))
    non_spatial = collect_non_spatial(progress, qa_data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(non_spatial, handle, indent=2)
    print(
        f"Wrote {sum(len(v['qa_pairs']) for v in non_spatial.values())} non-spatial QA pairs"
        f" across {len(non_spatial)} volumes to {args.output}"
    )


if __name__ == "__main__":
    main()
