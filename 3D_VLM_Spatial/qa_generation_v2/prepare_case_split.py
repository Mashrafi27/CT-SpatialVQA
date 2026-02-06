#!/usr/bin/env python3
"""Prepare a sanitized case split for new QA generation.

- Reads validation_reports_output.json for available cases.
- Excludes cases already present in existing QA files.
- Samples (target_total - existing) cases from remaining.
- Writes selected_cases.json + manifest.json.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Set


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare case list for new QA generation.")
    p.add_argument(
        "--reports",
        type=Path,
        default=Path("3D_VLM_Spatial/validation_reports_output.json"),
        help="Path to validation_reports_output.json",
    )
    p.add_argument(
        "--existing",
        type=Path,
        action="append",
        default=[],
        help=(
            "Existing QA files to exclude (JSON dict keyed by case_id or JSONL with case_id field). "
            "Repeatable."
        ),
    )
    p.add_argument("--target-total", type=int, default=1000, help="Total target cases (inclusive).")
    p.add_argument("--seed", type=int, default=13, help="Random seed for sampling.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("3D_VLM_Spatial/qa_generation_v2"),
        help="Output folder for selected_cases.json/manifest.json",
    )
    return p.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def collect_case_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    if path.suffix == ".jsonl":
        case_ids = set()
        for rec in iter_jsonl(path):
            cid = rec.get("case_id") or rec.get("image_path")
            if cid:
                case_ids.add(Path(str(cid)).name)
        return case_ids
    # JSON
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return {Path(k).name for k in data.keys()}
    if isinstance(data, list):
        case_ids = set()
        for rec in data:
            if isinstance(rec, dict):
                cid = rec.get("case_id") or rec.get("image_path")
                if cid:
                    case_ids.add(Path(str(cid)).name)
        return case_ids
    return set()


def main() -> None:
    args = parse_args()
    reports = json.loads(args.reports.read_text())
    report_case_ids = {Path(k).name for k in reports.keys()}

    existing_case_ids: Set[str] = set()
    for path in args.existing:
        existing_case_ids |= collect_case_ids(path)

    # Only count existing cases that are actually in reports
    existing_case_ids &= report_case_ids
    remaining_case_ids = sorted(report_case_ids - existing_case_ids)

    random.seed(args.seed)
    target_total = args.target_total
    already = len(existing_case_ids)
    needed = max(0, target_total - already)
    if needed > len(remaining_case_ids):
        needed = len(remaining_case_ids)
    sampled = random.sample(remaining_case_ids, needed) if needed > 0 else []

    selected = sorted(existing_case_ids) + sorted(sampled)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selected_cases.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    (args.output_dir / "existing_case_ids.json").write_text(
        json.dumps(sorted(existing_case_ids), indent=2), encoding="utf-8"
    )
    (args.output_dir / "remaining_case_ids.json").write_text(
        json.dumps(sorted(remaining_case_ids), indent=2), encoding="utf-8"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "reports": str(args.reports),
                "existing_files": [str(p) for p in args.existing],
                "target_total": target_total,
                "existing_count": len(existing_case_ids),
                "sampled_count": len(sampled),
                "total_selected": len(selected),
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Reports: {len(report_case_ids)} cases | Existing: {len(existing_case_ids)} | "
        f"Sampled: {len(sampled)} | Total selected: {len(selected)}"
    )
    print(f"Wrote: {args.output_dir}/selected_cases.json and manifest.json")


if __name__ == "__main__":
    main()
