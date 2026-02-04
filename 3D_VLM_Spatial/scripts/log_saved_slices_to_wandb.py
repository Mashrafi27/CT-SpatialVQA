#!/usr/bin/env python3
"""Log saved slice PNGs to Weights & Biases."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log saved PNGs to W&B.")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--run-name", default="ct-slice-logs")
    parser.add_argument("--images-root", type=Path, required=True, help="Root with case_id/axis/*.png")
    parser.add_argument("--grid-root", type=Path, default=None, help="Root with case_id/<axis>_grid.png")
    parser.add_argument("--labels", nargs="+", required=True, help="Labels to include (order matters)")
    parser.add_argument("--axes", nargs="+", default=["axial", "coronal", "sagittal"])
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def combine_side_by_side(images: List[Image.Image]) -> Image.Image:
    heights = [img.height for img in images]
    widths = [img.width for img in images]
    canvas = Image.new("L", (sum(widths), max(heights)))
    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width
    return canvas


def main() -> None:
    args = parse_args()
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("Install wandb to log images.") from exc

    run = wandb.init(project=args.project, name=args.run_name)

    root_for_cases = args.grid_root if args.grid_root is not None else args.images_root
    cases = [p for p in root_for_cases.iterdir() if p.is_dir()]
    cases.sort()
    if args.limit is not None:
        cases = cases[: args.limit]

    images = []
    for case_dir in tqdm(cases, desc="Logging cases"):
        if args.grid_root is not None:
            grid_case = case_dir
            for axis in args.axes:
                grid_path = grid_case / f"{axis}_grid.png"
                if grid_path.is_file():
                    caption = f"{case_dir.name} | {axis} | grid"
                    images.append(wandb.Image(str(grid_path), caption=caption))
            continue
        for axis in args.axes:
            axis_dir = case_dir / axis
            if not axis_dir.is_dir():
                continue
            panels = []
            captions = []
            raw_path = axis_dir / "RAW.png"
            if raw_path.is_file():
                panels.append(Image.open(raw_path))
                captions.append("RAW")
            for label in args.labels:
                label_path = axis_dir / f"{label}.png"
                if label_path.is_file():
                    panels.append(Image.open(label_path))
                    captions.append(label)
            if not panels:
                continue
            combined = combine_side_by_side(panels)
            caption = f"{case_dir.name} | {axis} | " + " | ".join(captions)
            images.append(wandb.Image(combined, caption=caption))

    key = "grid_slices" if args.grid_root is not None else "combined_slices"
    if not images:
        print("WARNING: no images found to log. Check --grid-root/--images-root paths.")
    run.log({key: images})
    run.finish()


if __name__ == "__main__":
    main()
