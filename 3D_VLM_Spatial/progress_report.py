#!/usr/bin/env python3
"""
Quick reporting utility for reviewer progress on spatial QA validation.

Usage:
    python progress_report.py --progress progress.json --assignments assignments.json \
        --qa spatial_qa_output.json --output-dir reports
"""

from __future__ import annotations

import argparse
import json
import sys
from io import StringIO
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class CaseProgress:
    name: str
    answered: int
    total: Optional[int]
    status: str


@dataclass
class UserProgress:
    name: str
    status: str
    cases_completed: int
    cases_partial: int
    cases_not_started: int
    qa_answered: int
    qa_total: int
    cases: List[CaseProgress]


def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        sys.exit(f"File not found: {path}")


def load_question_counts(path: Optional[Path]) -> Dict[str, int]:
    if not path:
        return {}
    if not path.exists():
        print(f"Warning: QA file {path} not found. Falling back to counts in progress.json.", file=sys.stderr)
        return {}
    data = load_json(path)
    counts = {}
    for case_id, record in data.items():
        qa_pairs = record.get("qa_pairs")
        counts[case_id] = len(qa_pairs) if isinstance(qa_pairs, list) else 0
    return counts


def case_status(answered: int, total: Optional[int]) -> str:
    if answered == 0:
        return "not_started"
    if total is None:
        return "in_progress"
    if answered >= total and total > 0:
        return "completed"
    return "partial"


def build_user_progress(assignments_path: Path, progress_path: Path, qa_path: Optional[Path]) -> List[UserProgress]:
    assignments = load_json(assignments_path)
    progress = load_json(progress_path)
    qa_counts = load_question_counts(qa_path)
    users: List[UserProgress] = []
    for user, cases in assignments.items():
        prog_cases = progress.get(user, {})
        case_entries: List[CaseProgress] = []
        qa_answered = 0
        qa_total = 0
        completed = partial = not_started = 0
        for case in cases:
            answers = prog_cases.get(case, {}).get("answers", [])
            filled = sum(1 for item in answers if isinstance(item, dict) and item.get("saved_at"))
            expected = qa_counts.get(case)
            if expected is None:
                expected = len(answers) if isinstance(answers, list) and answers else None
            if expected:
                qa_total += expected
            qa_answered += filled
            status = case_status(filled, expected)
            if status == "completed":
                completed += 1
            elif status == "partial":
                partial += 1
            else:
                not_started += 1
            case_entries.append(CaseProgress(case, filled, expected, status))
        user_status = "not_started"
        if completed == len(cases) and len(cases) > 0:
            user_status = "completed_all"
        elif qa_answered == 0:
            user_status = "not_started"
        elif completed == 0 and partial == 0:
            user_status = "not_started"
        else:
            user_status = "in_progress"
        users.append(
            UserProgress(
                name=user,
                status=user_status,
                cases_completed=completed,
                cases_partial=partial,
                cases_not_started=not_started,
                qa_answered=qa_answered,
                qa_total=qa_total,
                cases=case_entries,
            )
        )
    return users


def calculate_overall(users: Iterable[UserProgress]) -> Dict[str, object]:
    users = list(users)
    total_answered = sum(user.qa_answered for user in users)
    total_expected = sum(user.qa_total for user in users)
    status_counts = Counter(user.status for user in users)
    case_counts = Counter()
    total_cases = 0
    for user in users:
        for case in user.cases:
            case_counts[case.status] += 1
            total_cases += 1
    return {
        "qa_answered": total_answered,
        "qa_total": total_expected,
        "qa_percent": percentage(total_answered, total_expected),
        "status_counts": dict(status_counts),
        "case_counts": dict(case_counts),
        "case_total": total_cases,
    }


def percentage(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return (numerator / max(denominator, 1)) * 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize reviewer progress for spatial QA validation.")
    parser.add_argument("--progress", default="progress.json", type=Path, help="Path to progress.json")
    parser.add_argument("--assignments", default="assignments.json", type=Path, help="Path to assignments.json")
    parser.add_argument(
        "--qa",
        default="spatial_qa_output.json",
        type=Path,
        help="Optional spatial QA lookup file for expected question counts.",
    )
    parser.add_argument("--no-qa", action="store_true", help="Ignore QA file even if present.")
    parser.add_argument(
        "--output-dir",
        default=Path("reports"),
        type=Path,
        help="Directory where JSON/text summaries will be written.",
    )
    parser.add_argument(
        "--case-detail-limit",
        default=5,
        type=int,
        help="Number of users to include with per-case details in the text summary.",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation even if matplotlib is available.",
    )
    return parser.parse_args()


def write_outputs(
    users: List[UserProgress],
    output_dir: Path,
    case_detail_limit: int,
    generate_charts: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = calculate_overall(users)
    json_payload = {
        "overall": overall,
        "users": [user_to_dict(user) for user in sorted(users, key=lambda u: u.name.lower())],
    }
    json_path = output_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, indent=2)
    text_report = format_text_report(users, overall, case_detail_limit)
    text_path = output_dir / "summary.txt"
    with text_path.open("w", encoding="utf-8") as handle:
        handle.write(text_report)
    print(f"Wrote summary JSON to {json_path}")
    print(f"Wrote text report to {text_path}")
    if generate_charts:
        create_charts(overall, users, output_dir)


def user_to_dict(user: UserProgress) -> Dict[str, object]:
    return {
        "name": user.name,
        "status": user.status,
        "cases_completed": user.cases_completed,
        "cases_partial": user.cases_partial,
        "cases_not_started": user.cases_not_started,
        "qa_answered": user.qa_answered,
        "qa_total": user.qa_total,
        "cases": [
            {
                "name": case.name,
                "answered": case.answered,
                "total": case.total,
                "status": case.status,
            }
            for case in user.cases
        ],
    }


def format_text_report(users: List[UserProgress], overall: Dict[str, object], case_detail_limit: int) -> str:
    output = StringIO()
    output.write("Overall QA progress:\n")
    output.write(
        f"  Answered {overall['qa_answered']} / {overall['qa_total']} ({overall['qa_percent']:.1f}%)\n\n"
    )
    output.write("Users by status:\n")
    for status in ("completed_all", "in_progress", "not_started"):
        output.write(f"  {status:>13}: {overall['status_counts'].get(status, 0)}\n")
    output.write("\nCases by status:\n")
    for status in ("completed", "partial", "not_started"):
        output.write(f"  {status:>13}: {overall['case_counts'].get(status, 0)}\n")
    output.write("\nPer-user snapshot:\n")
    header = f"{'User':<22} {'Status':<15} {'Done':>4} {'Partial':>7} {'Not':>4} {'QA':>8}"
    output.write(header + "\n")
    output.write("-" * len(header) + "\n")
    for user in sorted(users, key=lambda u: u.name.lower()):
        qa_progress = f"{user.qa_answered}/{user.qa_total}"
        output.write(
            f"{user.name:<22} {user.status:<15} {user.cases_completed:>4} {user.cases_partial:>7} "
            f"{user.cases_not_started:>4} {qa_progress:>8}\n"
        )
    output.write("\nCase detail (limited view):\n")
    for user in sorted(users, key=lambda u: u.name.lower())[:case_detail_limit]:
        output.write(f"- {user.name} ({user.status})\n")
        for case in user.cases:
            total = case.total if case.total is not None else "?"
            output.write(f"    {case.name}: {case.answered}/{total} -> {case.status}\n")
    return output.getvalue()


def create_charts(overall: Dict[str, object], users: List[UserProgress], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping chart generation.")
        return

    status_order = ["completed_all", "in_progress", "not_started"]
    counts = [overall["status_counts"].get(status, 0) for status in status_order]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(status_order, counts, color=["#2c7bb6", "#fdae61", "#d7191c"])
    ax.set_ylabel("Reviewers")
    ax.set_title("Reviewers by status")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(count), ha="center", va="bottom")
    ax.set_ylim(0, max(counts) + 1 if counts else 1)
    fig.tight_layout()
    status_path = output_dir / "status_counts.png"
    fig.savefig(status_path, dpi=150)
    plt.close(fig)

    answered = overall["qa_answered"]
    remaining = max(overall["qa_total"] - answered, 0)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    bars2 = ax2.bar(["Answered", "Remaining"], [answered, remaining], color=["#1b9e77", "#d95f02"])
    ax2.set_ylabel("QA pairs")
    ax2.set_title("QA progress")
    progress_text = f"{overall['qa_percent']:.1f}% complete"
    ax2.text(
        0.5,
        max(answered, remaining) * 0.9 if (answered or remaining) else 0.5,
        progress_text,
        ha="center",
        va="center",
        fontsize=12,
        color="white",
        fontweight="bold",
    )
    for bar, value in zip(bars2, [answered, remaining]):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + max(overall["qa_total"] * 0.01, 5), f"{value}", ha="center")
    fig2.tight_layout()
    qa_path = output_dir / "qa_progress.png"
    fig2.savefig(qa_path, dpi=150)
    plt.close(fig2)
    case_counts = [overall["case_counts"].get(status, 0) for status in ("completed", "partial", "not_started")]
    if overall["case_total"]:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        bars3 = ax3.bar(["Completed", "Partial", "Not Started"], case_counts, color=["#4daf4a", "#ffcc00", "#e41a1c"])
        ax3.set_ylabel("Cases")
        ax3.set_title("Case progress")
        for bar, count in zip(bars3, case_counts):
            ax3.text(bar.get_x() + bar.get_width() / 2, count + 0.5, str(count), ha="center", va="bottom")
        ax3.set_ylim(0, max(case_counts) + 1 if any(case_counts) else 1)
        fig3.tight_layout()
        case_path = output_dir / "case_status_counts.png"
        fig3.savefig(case_path, dpi=150)
        plt.close(fig3)
        print(f"Wrote charts to {status_path}, {qa_path}, and {case_path}")
    else:
        print(f"Wrote charts to {status_path} and {qa_path}")


def main() -> None:
    args = parse_args()
    qa_path = None if args.no_qa else args.qa
    users = build_user_progress(args.assignments, args.progress, qa_path)
    write_outputs(users, args.output_dir, args.case_detail_limit, generate_charts=not args.no_charts)


if __name__ == "__main__":
    main()
