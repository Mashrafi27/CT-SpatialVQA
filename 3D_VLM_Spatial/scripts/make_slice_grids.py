import argparse
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--models", default="Med3DVLM,Merlin,VILA-M3,M3D,CT-CLIP,RadFM,MedEvalKit,MedGemma,RAW")
    p.add_argument("--views", default="axial,coronal,sagittal")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--raw-center", action="store_true", help="place RAW in the center tile")
    return p.parse_args()

def normalize_name(name):
    return name.replace(" ", "").replace("_", "-")

def find_image(case_dir, view, model):
    view_dir = Path(case_dir) / view
    if not view_dir.exists():
        return None
    # Try exact filename
    cand = view_dir / f"{model}.png"
    if cand.exists():
        return cand
    # Try normalized variants
    norm = normalize_name(model)
    cand = view_dir / f"{norm}.png"
    if cand.exists():
        return cand
    # Try case-insensitive match
    for f in view_dir.glob("*.png"):
        if normalize_name(f.stem).lower() == norm.lower():
            return f
    return None

def load_tile(path, label, tile_size):
    img = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
    if path and Path(path).exists():
        im = Image.open(path).convert("RGB")
        im.thumbnail((tile_size, tile_size), Image.BICUBIC)
        x = (tile_size - im.size[0]) // 2
        y = (tile_size - im.size[1]) // 2
        img.paste(im, (x, y))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = label
    # Draw a small label bar
    tw, th = draw.textsize(text, font=font)
    pad = 2
    draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return img

def make_grid(tiles, tile_size):
    grid = Image.new("RGB", (tile_size * 3, tile_size * 3), (0, 0, 0))
    for i, tile in enumerate(tiles):
        r = i // 3
        c = i % 3
        grid.paste(tile, (c * tile_size, r * tile_size))
    return grid

def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.raw_center and "RAW" in models:
        models = [m for m in models if m != "RAW"]
        # Insert RAW at center (index 4 in 0-based list for 3x3 grid)
        models.insert(4, "RAW")
    views = [v.strip() for v in args.views.split(",") if v.strip()]

    cases = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    for case_dir in cases:
        case_out = output_root / case_dir.name
        case_out.mkdir(parents=True, exist_ok=True)
        for view in views:
            tiles = []
            for model in models:
                img_path = find_image(case_dir, view, model)
                label = model if img_path else f"{model} (missing)"
                tiles.append(load_tile(img_path, label, args.tile_size))
            # Fill 9th empty tile
            if len(tiles) < 9:
                tiles.append(load_tile(None, "", args.tile_size))
            grid = make_grid(tiles[:9], args.tile_size)
            out_path = case_out / f"{view}_grid.png"
            grid.save(out_path)

    print(f"done: {output_root}")

if __name__ == "__main__":
    main()
