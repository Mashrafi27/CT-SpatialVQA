#!/usr/bin/env python3
"""Run Gemini judgments on QA pairs to flag spatial + relevant prompts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from textwrap import dedent
from typing import Dict, Iterable, List, Sequence

try:
    import google.generativeai as genai
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "google-generativeai is required. Install via `pip install google-generativeai`."
    ) from exc

PROMPT_TEMPLATE = """You are a radiology QA auditor.
You receive radiology Findings / Impressions plus QA pairs created from them.
Label each QA pair with two booleans:
- is_spatial: true only if answering the question requires spatial reasoning about anatomical location, orientation, relative position, or laterality visible in the image (e.g., which lung, which lobe, proximity, direction, comparison of organ sizes, location of effusion). Pure presence/absence or textual metadata questions are false.
- is_relevant: true only if the question could be answered from the imaging-derived Findings/Impressions (not from general knowledge or the wording of the text itself). Questions that just ask which words appear in the report, cite textual phrases, or hallucinate unseen details are false.

Respond strictly as JSON with the schema:
[
  {
    "index": <QA index starting at 1 in this batch>,
    "is_spatial": true/false,
    "is_relevant": true/false,
    "reasoning": "brief justification referencing image evidence"
  }, ...
]
Avoid additional narration.

Case ID: {case_id}
Findings: {findings}
Impressions: {impressions}
QA pairs:
{qa_block}
"""

GENERATION_CONFIG = {
    "temperature": 0.0,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 1024,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge QA pairs via Gemini for spatial + relevance")
    parser.add_argument("--qa", type=Path, default=Path("spatial_qa_output.json"), help="Path to QA JSON")
    parser.add_argument(
        "--reports",
        type=Path,
        default=Path("validation_reports_output.json"),
        help="Path to findings/impressions JSON",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/gemini_judgments.json"))
    parser.add_argument("--batch-size", type=int, default=5, help="QA pairs per API call")
    parser.add_argument(
        "--model",
        default="models/gemini-1.5-pro-latest",
        help="Gemini model name",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between calls (rate limiting)",
    )
    return parser.parse_args()


def chunk(seq: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def build_prompt(case_id: str, findings: str, impressions: str, batch: Sequence[dict]) -> str:
    qa_lines = []
    for idx, qa in enumerate(batch, start=1):
        qa_lines.append(f"{idx}. Q: {qa['question']}\n   A: {qa['answer']}")
    qa_block = "\n".join(qa_lines)
    return PROMPT_TEMPLATE.format(
        case_id=case_id,
        findings=findings or "(none provided)",
        impressions=impressions or "(none provided)",
        qa_block=qa_block,
    )


def main() -> None:
    args = parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY in the environment before running.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    qa_data: Dict[str, dict] = json.loads(args.qa.read_text())
    reports: Dict[str, dict] = json.loads(args.reports.read_text())
    judgments: Dict[str, List[dict]] = {}

    for case_id, payload in qa_data.items():
        qa_pairs = payload.get("qa_pairs", [])
        if not qa_pairs:
            continue
        report = reports.get(case_id, {})
        findings = report.get("Findings_EN", "")
        impressions = report.get("Impressions_EN", "")
        case_results: List[dict] = []
        for batch in chunk(qa_pairs, args.batch_size):
            prompt = build_prompt(case_id, findings, impressions, batch)
            attempt = 0
            while True:
                attempt += 1
                try:
                    response = model.generate_content(
                        prompt,
                        generation_config=GENERATION_CONFIG,
                    )
                    text = response.text
                    parsed = json.loads(text)
                    if not isinstance(parsed, list):  # pragma: no cover
                        raise ValueError("Response is not a list")
                    case_results.extend(parsed)
                    break
                except Exception as exc:  # pragma: no cover
                    if attempt >= 3:
                        raise
                    time.sleep(2 ** attempt)
            if args.sleep:
                time.sleep(args.sleep)
        judgments[case_id] = case_results

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(judgments, indent=2), encoding="utf-8")
    print(f"Saved judgments for {len(judgments)} cases to {args.output}")


if __name__ == "__main__":
    main()
