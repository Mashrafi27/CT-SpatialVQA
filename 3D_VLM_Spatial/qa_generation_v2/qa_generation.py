import os
import json
import time
import random
from pathlib import Path
from typing import Optional
from openai import OpenAI
from tqdm import tqdm

# ----------------------------
# ✅ CONFIGURATION
# ----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Your refined, concise prompt
BASE_PROMPT = """
You are a medical AI assistant specialized in radiology and 3D spatial reasoning.

Read the 3D CT scan report below and generate 7–10 question–answer (QA) pairs 
that test a vision–language model’s understanding of spatial and anatomical relationships 
explicitly described in the report.

Focus only on spatial facts such as:
- Laterality (right vs. left, unilateral vs. bilateral)
- Vertical position (upper/lower, superior/inferior)
- Anterior–posterior relations
- Medial–lateral orientation (central/peripheral/midline)
- Spatial extent or boundaries (confined to, crossing midline, extending into)
- Adjacency or containment (within, posterior to, adjacent to)

Guidelines:
- Use only information from the Findings and Impressions sections.
- Do not include diagnostic, interpretive, or normality statements.
- Questions must emphasize *where*, *which side*, *above/below*, or *extent*.
- Answers must be strictly factual and directly derived from the report.

Output as a valid JSON list:
[
  {"question": "...", "answer": "..."},
  ...
]
"""

# ----------------------------
# ✅ MAIN FUNCTION
# ----------------------------
def generate_spatial_QA(findings, impressions, model="gpt-4o-mini", max_retries=5, sleep_base=2.0, seed: Optional[int] = None):
    """Generate spatial QA pairs from a CT report using OpenAI API."""
    full_prompt = (
        BASE_PROMPT
        + f"\n\nHere is the 3D CT Scan Report:\n**Findings:** {findings}\n**Impressions:** {impressions}"
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,  # Or "gpt-4o" for higher accuracy
                messages=[
                    {"role": "system", "content": "You are a precise radiology assistant."},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.2,
                seed=seed,
            )
            content = response.choices[0].message.content.strip()
            try:
                qa_pairs = json.loads(content)
                # enforce list output
                if not isinstance(qa_pairs, list):
                    raise json.JSONDecodeError("Not a list", content, 0)
                return qa_pairs
            except json.JSONDecodeError:
                return [{"error": "Invalid JSON output", "raw_output": content}]
        except Exception:
            # naive exponential backoff
            time.sleep(sleep_base * (2 ** attempt) + random.random())
    return [{"error": "Failed after retries"}]


# ----------------------------
# ✅ PROCESS CSV
# ----------------------------
def process_reports(
    reports_json: Path,
    selected_cases: Optional[Path],
    existing_cases: Optional[Path],
    output_json: Path,
    target_total: int,
    seed: int,
    model: str,
    dry_run: bool,
):
    random.seed(seed)
    reports = json.loads(reports_json.read_text(encoding="utf-8"))
    all_case_ids = {Path(k).name for k in reports.keys()}

    existing = set()
    if existing_cases and existing_cases.exists():
        existing = set(json.loads(existing_cases.read_text(encoding="utf-8")))
        existing = {Path(k).name for k in existing}
    existing &= all_case_ids

    if selected_cases and selected_cases.exists():
        selected = [Path(k).name for k in json.loads(selected_cases.read_text(encoding="utf-8"))]
    else:
        remaining = sorted(all_case_ids - existing)
        random.seed(seed)
        need = max(0, target_total - len(existing))
        if need > len(remaining):
            need = len(remaining)
        sampled = random.sample(remaining, need)
        selected = sorted(existing) + sorted(sampled)

    if dry_run:
        selected = selected[:10]

    output_json.parent.mkdir(parents=True, exist_ok=True)
    out_data = {}
    # resume support
    if output_json.exists():
        try:
            out_data = json.loads(output_json.read_text(encoding="utf-8"))
        except Exception:
            out_data = {}

    for case_id in tqdm(selected, desc="Processing CT Reports"):
        if case_id in out_data:
            continue
        entry = reports.get(case_id)
        if not entry:
            continue
        findings = entry.get("Findings_EN", "")
        impressions = entry.get("Impressions_EN", "")
        qa_output = generate_spatial_QA(findings, impressions, model=model, seed=seed)
        out_data[case_id] = {"qa_pairs": qa_output}
        output_json.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ Saved {len(out_data)} QA entries to {output_json}")


# ----------------------------
# ✅ ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate spatial QA pairs for CT reports using OpenAI API.")
    parser.add_argument("--reports_json", required=True, help="Path to validation_reports_output.json.")
    parser.add_argument("--selected_cases", default=None, help="Path to selected_cases.json (optional).")
    parser.add_argument("--existing_cases", default=None, help="Path to existing_case_ids.json (optional).")
    parser.add_argument("--output_json", default="spatial_QA_output.json", help="Path to save generated JSON output.")
    parser.add_argument("--target_total", type=int, default=1000, help="Total target cases (incl. existing).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model id.")
    parser.add_argument("--dry-run", action="store_true", help="Generate only 10 cases for validation.")
    args = parser.parse_args()

    process_reports(
        reports_json=Path(args.reports_json),
        selected_cases=Path(args.selected_cases) if args.selected_cases else None,
        existing_cases=Path(args.existing_cases) if args.existing_cases else None,
        output_json=Path(args.output_json),
        target_total=args.target_total,
        seed=args.seed,
        model=args.model,
        dry_run=args.dry_run,
    )
 
