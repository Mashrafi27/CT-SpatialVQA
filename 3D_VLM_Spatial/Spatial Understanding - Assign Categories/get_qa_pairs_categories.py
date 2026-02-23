#!/usr/bin/env python3
"""Assign spatial categories to QA pairs in spatial_qa_filtered_full.json using Gemini."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Optional

try:
    import google.generativeai as genai
except ImportError as exc:
    raise SystemExit("Install google-generativeai to use this script.") from exc

try:
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit("Install tqdm to use this script.") from exc

PROMPT_TEMPLATE = """
########################################################
Role:
You are an expert Medical AI Specialist specializing in anatomical spatial reasoning and radiology report analysis.

########################################################
Task:
Analyze the provided Question-Answer (QA) pair derived from a CT scan report. Assign one or more categories from the provided list based on the spatial logic required to answer the question or present in the answer.

########################################################
Question: {question}
Answer: {answer}

########################################################
Categories & Logic:

1. Laterality & Bilateral Symmetry: (Left, right, unilateral, bilateral, hemithorax).

2. Longitudinal (Vertical) Position: (Superior/inferior, upper/lower, apical/basal, or specific vertebral/rib levels).

3. Anterior-Posterior (Depth) Relations: (Anterior, posterior, retrosternal, ventral, dorsal).

4. Medial-Lateral Orientation (Centricity): (Midline, medial, lateral, central, peripheral, hilar, paratracheal).

5. Adjacency & Containment: (Adjacent to, around, neighborhood of, within an organ, intra-parenchymal).

6. Spatial Extent & Boundaries: (Extends to, crossing, confined to, diffuse, reaching a fissure).

########################################################
Instructions:
Evaluate both the Question and the Answer for spatial cues.
Return the result as a Python-formatted list of strings (use the exact category names above, e.g. "Laterality & Bilateral Symmetry").
If multiple categories apply, include all relevant ones.
Do not provide conversational filler; return only the list.

"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign spatial categories to QA pairs in JSON using Gemini"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("spatial_qa_filtered_full.json"),
        help="Input JSON file (image_id -> qa_pairs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spatial_qa_filtered_full_with_categories.json"),
        help="Replica JSON file where categories are saved (after each QA pair); used for resume",
    )
    parser.add_argument(
        "--model",
        default="models/gemini-2.5-flash",
        help="Gemini model name",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Delay in seconds between API calls",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Max retries for rate limit errors",
    )
    parser.add_argument(
        "--initial-backoff",
        type=float,
        default=5.0,
        help="Initial backoff seconds",
    )
    parser.add_argument(
        "--max-backoff",
        type=float,
        default=120.0,
        help="Maximum backoff seconds",
    )
    parser.add_argument(
        "--backoff-mult",
        type=float,
        default=2.0,
        help="Backoff multiplier",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from replica (output) file; skip pairs that already have 'category'",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Save replica every N processed QA pairs (default: 50). Use 1 for safest but slowest.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for API calls (default: 1).",
    )
    return parser.parse_args()


def load_qa_json(path: Path) -> dict[str, Any]:
    """Load JSON: { image_id: { qa_pairs: [ {question, answer}, ... ] }, ... }."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_qa_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_text(response) -> str:
    try:
        if getattr(response, "text", None):
            return response.text
    except Exception:
        pass
    parts = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts)


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`\n")
        if text.startswith("python\n"):
            text = text[7:]
        if text.startswith("json\n"):
            text = text[5:]
    return text


def parse_category_list(raw: str) -> List[str]:
    """Parse Gemini response into a list of category strings."""
    raw = strip_code_fence(raw).strip()
    if not raw:
        return []
    # Try JSON first
    try:
        out = json.loads(raw)
        if isinstance(out, list):
            return [str(x).strip() for x in out if x]
        return []
    except json.JSONDecodeError:
        pass
    # Try Python literal
    try:
        out = ast.literal_eval(raw)
        if isinstance(out, list):
            return [str(x).strip() for x in out if x]
        return []
    except (ValueError, SyntaxError):
        pass
    # Fallback: single line list-like
    m = re.match(r"^\[(.*)\]$", raw, re.DOTALL)
    if m:
        inner = m.group(1)
        parts = re.findall(r'"([^"]*)"', inner) or re.findall(r"'([^']*)'", inner)
        return [p.strip() for p in parts if p.strip()]
    return []


def _should_retry(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    if "resourceexhausted" in name or "quota" in msg or "rate" in msg or "429" in msg:
        return True
    return False


def generate_with_retry(
    model,
    prompt: str,
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
    backoff_mult: float,
):
    attempt = 0
    backoff = initial_backoff
    while True:
        try:
            return model.generate_content(prompt)
        except Exception as exc:
            attempt += 1
            if attempt > max_retries or not _should_retry(exc):
                raise
            jitter = random.uniform(0, 0.25 * backoff)
            sleep_for = min(max_backoff, backoff + jitter)
            print(f"[WARN] Rate limited, retrying in {sleep_for:.1f}s (attempt {attempt}/{max_retries})")
            time.sleep(sleep_for)
            backoff = min(max_backoff, backoff * backoff_mult)


def assign_categories(
    replica: dict[str, Any],
    output_path: Path,
    model_name: str,
    sleep: float,
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
    backoff_mult: float,
    save_every: int,
    workers: int,
) -> dict[str, Any]:
    """Add 'category' to each QA pair in the replica; save replica to output_path after each pair."""
    processed = 0
    skipped = 0
    pending_save = 0
    tasks: List[tuple[str, int, str, str]] = []

    for image_id, entry in tqdm(replica.items(), desc="Images", unit="image"):
        if not isinstance(entry, dict) or "qa_pairs" not in entry:
            continue
        for idx, pair in enumerate(entry["qa_pairs"]):
            if not isinstance(pair, dict) or "question" not in pair or "answer" not in pair:
                continue
            if pair.get("category") is not None:
                skipped += 1
                continue
            question = pair.get("question", "")
            answer = pair.get("answer", "")
            tasks.append((image_id, idx, question, answer))

    if skipped:
        tqdm.write(f"[INFO] Resumed: skipped {skipped} pairs that already had 'category'.")
    if not tasks:
        tqdm.write("[INFO] No uncategorized QA pairs to process.")
        return replica

    def _worker(task: tuple[str, int, str, str]):
        image_id, pair_idx, question, answer = task
        prompt = PROMPT_TEMPLATE.format(question=question, answer=answer)
        # Create a model instance per worker thread.
        model = genai.GenerativeModel(model_name)
        try:
            response = generate_with_retry(
                model,
                prompt,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                backoff_mult=backoff_mult,
            )
            text = extract_text(response)
            categories = parse_category_list(text)
            return image_id, pair_idx, categories, None
        except Exception as exc:  # noqa: BLE001
            return image_id, pair_idx, [], exc

    max_workers = max(1, int(workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker, task) for task in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="QA pairs", unit="pair"):
            image_id, pair_idx, categories, err = fut.result()
            if err is not None:
                tqdm.write(f"[WARN] Gemini failed for {image_id}: {err}")
            # Apply result to replica
            entry = replica.get(image_id)
            if isinstance(entry, dict) and "qa_pairs" in entry:
                pair = entry["qa_pairs"][pair_idx]
                pair["category"] = categories
            processed += 1
            pending_save += 1
            if sleep:
                time.sleep(sleep)
            if pending_save >= save_every:
                save_qa_json(output_path, replica)
                pending_save = 0

    tqdm.write(f"[INFO] Processed {processed} QA pairs.")
    if pending_save:
        save_qa_json(output_path, replica)
    return replica


def main() -> None:
    args = parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY before running.")
    genai.configure(api_key=api_key)

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    input_data = load_qa_json(args.input)

    # Replica: copy of QA pairs where we add and persist categories (saved to output after each pair)
    if args.resume and args.output.exists():
        try:
            replica = load_qa_json(args.output)
            # Ensure replica has same structure as input (e.g. input may have more images/pairs)
            for image_id, entry in input_data.items():
                if image_id not in replica or not isinstance(replica[image_id], dict):
                    replica[image_id] = copy.deepcopy(entry)
                elif "qa_pairs" in entry and "qa_pairs" in replica[image_id]:
                    inp_pairs = entry["qa_pairs"]
                    rep_pairs = replica[image_id]["qa_pairs"]
                    for i in range(len(rep_pairs), len(inp_pairs)):
                        rep_pairs.append(copy.deepcopy(inp_pairs[i]))
            print("[INFO] Resuming from replica (output file); will skip pairs that already have 'category'.")
        except Exception as e:
            print(f"[WARN] Could not load replica for resume: {e}")
            replica = copy.deepcopy(input_data)
    else:
        replica = copy.deepcopy(input_data)

    assign_categories(
        replica,
        args.output,
        args.model,
        args.sleep,
        args.max_retries,
        args.initial_backoff,
        args.max_backoff,
        args.backoff_mult,
        args.save_every,
        args.workers,
    )

    save_qa_json(args.output, replica)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
