#!/usr/bin/env python3
"""Render a LaTeX table from text_metrics_summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_MODEL_ORDER = [
    "ctchat",
    "merlin",
    "med3dvlm",
    "m3d",
    "radfm",
]

DEFAULT_LABELS = {
    "ctchat": "CT-Chat",
    "merlin": "MERLIN",
    "med3dvlm": "Med3DVLM",
    "m3d": "M3D",
    "radfm": "RadFM",
    "vila-m3": "VILA-M3",
    "vila-m3_vista3d": "VILA-M3+Vista3D",
    "medgemma": "MedGemma",
    "lingshu7b": "Lingshu-7B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render LaTeX table from metrics JSON.")
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("3D_VLM_Spatial/reports/text_metrics_summary.json"),
        help="Path to metrics JSON list.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model order (use keys from metrics JSON).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output .tex file; otherwise prints to stdout.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Decimal precision for metrics.",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> Dict[str, dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("metrics", [])
    metrics = {}
    for row in data:
        name = row.get("model")
        if name:
            metrics[name] = row
    return metrics


def fmt(val, precision: int) -> str:
    if val is None:
        return "-"
    try:
        return f"{float(val):.{precision}f}"
    except Exception:
        return "-"


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.metrics_json)
    model_order = args.models if args.models else DEFAULT_MODEL_ORDER

    headers = [DEFAULT_LABELS.get(m, m) for m in model_order]

    def get(model: str, key: str):
        return metrics.get(model, {}).get(key)

    rows = [
        ("LLM as Judge", ["-" for _ in model_order]),
        ("LLM as Jury", ["-" for _ in model_order]),
        ("SBERT Cosine Sim.", [fmt(get(m, "sbert_cosine"), args.precision) for m in model_order]),
        ("BLEU", [fmt(get(m, "bleu"), args.precision) for m in model_order]),
        ("ROUGE-L", [fmt(get(m, "rougeL_f"), args.precision) for m in model_order]),
        ("METEOR", [fmt(get(m, "meteor"), args.precision) for m in model_order]),
    ]

    col_spec = "l" + "c" * len(model_order)
    lines: List[str] = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Evaluation Metrics across Medical 3D VLMs}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\renewcommand{\\arraystretch}{1.25}")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    header_row = " & ".join(["\\textbf{Models $\\rightarrow$}"] + [f"\\textbf{{{h}}}" for h in headers]) + " \\\\"
    lines.append(header_row)
    lines.append("\\midrule")
    for i, (name, values) in enumerate(rows):
        line = " & ".join([name] + values) + " \\\\"
        lines.append(line)
        if name in ("LLM as Jury",):
            lines.append("\\addlinespace")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    content = "\n".join(lines)
    if args.output:
        args.output.write_text(content)
    else:
        print(content)


if __name__ == "__main__":
    main()
