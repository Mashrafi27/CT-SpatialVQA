#!/usr/bin/env python3
"""Run Gemini judgments on QA pairs to flag spatial + relevant prompts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, cast

try:
    import google.generativeai as genai
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "google-generativeai is required. Install via `pip install google-generativeai`."
    ) from exc

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover
    raise SystemExit("tqdm is required. Install via `pip install tqdm`.") from exc

PROMPT_TEMPLATE = """You are a radiology QA auditor.
You receive radiology Findings / Impressions plus QA pairs created from them.
Label each QA pair with two booleans:
- is_spatial: true only if answering the question requires spatial reasoning about anatomical location, orientation, relative position, or laterality visible in the image (e.g., which lung, which lobe, proximity, direction, comparison of organ sizes, location of effusion). Pure presence/absence or textual metadata questions are false.
- is_relevant: true only if the question could be answered from the imaging-derived Findings/Impressions (not from general knowledge or the wording of the text itself). Questions that just ask which words appear in the report, cite textual phrases, or hallucinate unseen details are false.

Respond strictly as JSON with the schema:
[
  {{
    "index": <QA index starting at 1 in this batch>,
    "is_spatial": true/false,
    "is_relevant": true/false
  }}, ...
]
Avoid additional narration. Return ONLY the JSON list, no extra text.

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
        default="models/gemini-1.0-pro",
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


def extract_text(response) -> str:
    """Best-effort extraction of plain text from the Gemini response object."""
    if getattr(response, "text", None):
        return response.text
    chunks: List[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`\n")
        if cleaned.startswith("json\n"):
            cleaned = cleaned[5:]
    return cleaned.strip()


def _extract_json_list(text: str) -> str | None:
    """Extract the first JSON list from a blob of text."""
    if not text:
        return None
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _repair_truncated_list(text: str) -> str | None:
    """Best-effort repair when output is cut off before closing bracket."""
    start = text.find("[")
    if start == -1:
        return None
    # grab up to last closing brace
    end = text.rfind("}")
    if end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    # strip trailing commas before closing
    candidate = candidate.rstrip()
    if candidate.endswith(","):
        candidate = candidate[:-1]
    return candidate + "]"


def parse_json_response(text: str) -> list:
    """Best-effort parse of model JSON list output."""
    cleaned = _strip_code_fences(text)
    # 1) direct parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # 2) extract list
    extracted = _extract_json_list(cleaned)
    if extracted:
        try:
            parsed = json.loads(extracted)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    # 3) escape raw newlines and retry
    escaped = cleaned.replace("\r\n", "\n").replace("\n", "\\n")
    extracted = _extract_json_list(escaped) or escaped
    parsed = json.loads(extracted)
    if not isinstance(parsed, list):
        raise ValueError("Response is not a list")
    return parsed


def parse_json_response_lenient(text: str) -> list | None:
    """Lenient parse that attempts to repair truncated JSON."""
    cleaned = _strip_code_fences(text)
    # try repair on raw and cleaned text
    for blob in (cleaned, cleaned.replace("\r\n", "\n")):
        repaired = _repair_truncated_list(blob)
        if repaired:
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                continue
    return None


def main() -> None:
    args = parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY in the environment before running.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    qa_data = json.loads(args.qa.read_text())
    if isinstance(qa_data, list):
        qa_dict: Dict[str, dict] = {}
        for item in qa_data:
            case_id = item.get("case_id") or item.get("file_name") or item.get("case")
            if not case_id:
                continue
            qa_dict[case_id] = item
        qa_data = qa_dict
    qa_data = cast(Dict[str, dict], qa_data)
    reports: Dict[str, dict] = json.loads(args.reports.read_text())
    if args.output.exists():
        try:
            judgments: Dict[str, List[dict]] = json.loads(args.output.read_text())
        except json.JSONDecodeError:
            judgments = {}
    else:
        judgments = {}

    case_items = list(qa_data.items())
    for case_id, payload in tqdm(case_items, desc="Cases", unit="case"):
        qa_pairs = payload.get("qa_pairs", [])
        if not qa_pairs:
            continue
        if case_id in judgments:
            continue
        report = reports.get(case_id, {})
        findings = report.get("Findings_EN", "")
        impressions = report.get("Impressions_EN", "")
        case_results: List[dict] = []
        batches = list(chunk(qa_pairs, args.batch_size))
        for batch in tqdm(batches, desc=f"{case_id}", unit="batch", leave=False):
            prompt = build_prompt(case_id, findings, impressions, batch)
            attempt = 0
            while True:
                attempt += 1
                try:
                    response = model.generate_content(
                        prompt,
                        generation_config=GENERATION_CONFIG,
                    )
                    text = extract_text(response)
                    parsed = parse_json_response(text)
                    case_results.extend(parsed)
                    break
                except Exception as exc:  # pragma: no cover
                    if isinstance(exc, json.JSONDecodeError) or isinstance(exc, ValueError):
                        print(
                            f"[WARN] JSON decode failed for case {case_id} batch starting idx "
                            f"{qa_pairs.index(batch[0])} -- raw response:\\n{text}"
                        )
                        # lenient repair for truncated output
                        repaired = parse_json_response_lenient(text)
                        if repaired is not None:
                            case_results.extend(repaired)
                            break
                    if attempt >= 3:
                        # fall back with parse error placeholders and continue
                        case_results.extend(
                            {
                                "index": i + 1,
                                "is_spatial": None,
                                "is_relevant": None,
                                "reasoning": f"PARSE_ERROR: {text[:500]}",
                            }
                            for i in range(len(batch))
                        )
                        break
                    # tighten prompt on retry
                    if attempt == 2:
                        prompt = (
                            prompt
                            + "\n\nIMPORTANT: Return ONLY a valid JSON list. No extra text. "
                            + "Do NOT include trailing commas. Ensure the list is complete."
                        )
                    time.sleep(2 ** attempt)
            if args.sleep:
                time.sleep(args.sleep)
        judgments[case_id] = case_results
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(judgments, indent=2), encoding="utf-8")
    print(f"Saved judgments for {len(judgments)} cases to {args.output}")


if __name__ == "__main__":
    main()
