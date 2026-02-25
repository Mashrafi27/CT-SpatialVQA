#!/usr/bin/env python3
"""Plot answer-length distributions for GT and model predictions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List


WORD_RE = re.compile(r"\b\w+\b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot answer length distributions (GT + models).")
    p.add_argument(
        "--dataset-json",
        type=Path,
        default=Path("3D_VLM_Spatial/qa_generation_v2/spatial_qa_filtered_full.json"),
        help="CT-SpatialVQA JSON (image_id -> qa_pairs).",
    )
    p.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("3D_VLM_Spatial/reports"),
        help="Directory containing *_predictions*.jsonl files.",
    )
    p.add_argument(
        "--pred-glob",
        default="*_predictions_full.jsonl",
        help="Glob for prediction files within predictions-dir.",
    )
    p.add_argument(
        "--prediction-field",
        default="prediction",
        help="Field name for model output in predictions JSONL.",
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of model names to include (matches file stem before _predictions).",
    )
    p.add_argument(
        "--plot-type",
        choices=["violin", "box"],
        default="violin",
        help="Plot type.",
    )
    p.add_argument(
        "--include-questions",
        action="store_true",
        help="Include question-length distribution above GT answers.",
    )
    p.add_argument(
        "--log-x",
        action="store_true",
        help="Use log scale on x-axis (length).",
    )
    p.add_argument(
        "--bottom-first",
        action="store_true",
        help="Keep first label at the bottom (default places first label at top).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("3D_VLM_Spatial/reports/answer_length_distributions.png"),
        help="Output image path (png/pdf).",
    )
    p.add_argument(
        "--font-scale",
        type=float,
        default=0.85,
        help="Scale factor for font sizes (default: 0.85).",
    )
    return p.parse_args()


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(WORD_RE.findall(text))


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_gt_lengths(dataset_json: Path) -> List[int]:
    data = json.loads(dataset_json.read_text())
    lengths: List[int] = []
    for entry in data.values():
        for qa in entry.get("qa_pairs", []):
            lengths.append(word_count(qa.get("answer", "")))
    return lengths


def load_question_lengths(dataset_json: Path) -> List[int]:
    data = json.loads(dataset_json.read_text())
    lengths: List[int] = []
    for entry in data.values():
        for qa in entry.get("qa_pairs", []):
            lengths.append(word_count(qa.get("question", "")))
    return lengths


def load_pred_lengths(pred_path: Path, prediction_field: str) -> List[int]:
    return [word_count(rec.get(prediction_field, "")) for rec in iter_jsonl(pred_path)]


def main() -> None:
    args = parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    import numpy as np  # noqa: E402
    from matplotlib import colors as mcolors  # noqa: E402

    # Smaller text for compact plots
    base = 10 * args.font_scale
    plt.rcParams.update(
        {
            "font.size": base,
            "axes.titlesize": base,
            "axes.labelsize": base,
            "xtick.labelsize": base * 0.9,
            "ytick.labelsize": base * 0.9,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
        }
    )

    label_map = {
        "vila_m3_vista3d": "VILA-M3",
        "ctchat": "CT-Chat",
        "m3d": "M3D",
        "med3dvlm": "Med3DVLM",
        "medevalkit": "MedEvalKit",
        "medgemma": "MedGemma-1.5",
        "merlin": "Merlin",
        "radfm": "RadFM",
    }

    labels: List[str] = []
    series: List[List[int]] = []

    # Ground-truth (and optionally questions)
    if args.include_questions:
        qlens = load_question_lengths(args.dataset_json)
        labels.append("Questions")
        series.append(qlens)

    gt = load_gt_lengths(args.dataset_json)
    labels.append("GT Answers")
    series.append(gt)

    # Predictions
    files = sorted(args.predictions_dir.glob(args.pred_glob))
    for path in files:
        model_name = path.stem.replace("_predictions_full", "").replace("_predictions", "")
        if args.models and model_name not in args.models:
            continue
        lengths = load_pred_lengths(path, args.prediction_field)
        labels.append(label_map.get(model_name, model_name))
        series.append(lengths)

    if len(series) <= 1:
        raise SystemExit("No prediction files found for plotting.")

    # Plot size tuned for compact output
    height = max(2.0, 0.28 * len(labels))
    fig, ax = plt.subplots(figsize=(7.5, height))

    # Add a small visual gap between GT rows and model rows
    gap_after = 2 if args.include_questions else 1
    gap_size = 0.6
    positions = []
    for i in range(len(labels)):
        pos = i + 1
        if i + 1 > gap_after:
            pos += gap_size
        positions.append(pos)

    # Base colors (questions/GT keep distinct colors, models share light-cyan)
    base_palette = plt.get_cmap("tab20")
    question_base = base_palette(0)
    gt_base = "#1f6fb2"
    model_base = "#3aaed8"

    def make_cmap(base_color: str | tuple, to_white: float = 0.9) -> mcolors.LinearSegmentedColormap:
        base_rgb = np.array(mcolors.to_rgb(base_color))
        # Push low-density color close to white for strong visible gradients.
        light_rgb = base_rgb * (1 - to_white) + np.ones(3) * to_white
        return mcolors.LinearSegmentedColormap.from_list("density_cmap", [light_rgb, base_rgb])

    question_cmap = make_cmap(question_base, 0.9)
    gt_cmap = make_cmap(gt_base, 0.9)
    model_cmap = make_cmap(model_base, 0.9)

    def cmap_for_label(label: str) -> mcolors.LinearSegmentedColormap:
        if label == "Questions":
            return question_cmap
        if label == "GT Answers":
            return gt_cmap
        return model_cmap

    def base_for_label(label: str) -> str | tuple:
        if label == "Questions":
            return question_base
        if label == "GT Answers":
            return gt_base
        return model_base

    if args.plot_type == "violin":
        # Custom gradient violins (higher density -> darker)
        all_vals = [v for seq in series for v in seq]
        if not all_vals:
            raise SystemExit("No data to plot.")
        data_min = min(all_vals)
        data_max = max(all_vals)
        if data_min == data_max:
            data_min = max(0, data_min - 1)
            data_max = data_max + 1
        if args.log_x:
            min_pos = min(v for v in all_vals if v > 0) if any(v > 0 for v in all_vals) else 1
            xs = np.logspace(np.log10(min_pos), np.log10(max(data_max, min_pos + 1)), num=220)
        else:
            xs = np.linspace(data_min, data_max, num=220)

        def kde_density(data: List[int], grid: np.ndarray) -> np.ndarray:
            vals = np.asarray(data, dtype=float)
            if args.log_x:
                vals = vals[vals > 0]
            if vals.size == 0:
                return np.zeros_like(grid)
            std = np.std(vals)
            bw = 1.06 * std * (vals.size ** (-1 / 5)) if std > 0 else 0.4
            bw = max(bw, 0.2)
            diff = (grid[:, None] - vals[None, :]) / bw
            density = np.exp(-0.5 * diff * diff).sum(axis=1)
            density /= (vals.size * bw * np.sqrt(2 * np.pi))
            return density

        max_width = 0.38
        for label, data, y0 in zip(labels, series, positions):
            density = kde_density(data, xs)
            max_d = density.max()
            if max_d <= 0:
                continue
            density_norm = density / max_d
            cmap = cmap_for_label(label)
            for i in range(len(xs) - 1):
                d = density_norm[i]
                if d <= 0:
                    continue
                width = d * max_width
                ax.fill_between(
                    [xs[i], xs[i + 1]],
                    y0 - width,
                    y0 + width,
                    color=cmap(d),
                    linewidth=0,
                )
            median = float(np.median(np.asarray(data, dtype=float))) if data else None
            if median is not None:
                idx = int(np.argmin(np.abs(xs - median)))
                width = density_norm[idx] * max_width
                ax.plot([median, median], [y0 - width, y0 + width], color="black", lw=0.6)
    else:
        bxp = ax.boxplot(series, positions=positions, vert=False, showfliers=False, patch_artist=True)
        for box, label in zip(bxp["boxes"], labels):
            box.set_facecolor(base_for_label(label))

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Text length (words)")
    # Title removed to save space
    if args.log_x:
        ax.set_xscale("log")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    if not args.bottom_first:
        ax.invert_yaxis()
    # separator between GT rows and model rows
    if len(labels) > 2:
        if gap_after < len(positions):
            sep_y = (positions[gap_after - 1] + positions[gap_after]) / 2
            ax.hlines(
                sep_y,
                xmin=-0.05,
                xmax=0.98,
                transform=ax.get_yaxis_transform(),
                color="black",
                linewidth=0.6,
                alpha=0.6,
                linestyles="--",
                clip_on=False,
            )
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
