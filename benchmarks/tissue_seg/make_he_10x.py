from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def _load_pyvips():
    try:
        import pyvips
    except ModuleNotFoundError as exc:
        raise SystemExit("pyvips is required. Run this in the conda env `aligner`.") from exc
    return pyvips


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize an HE TIFF to exactly match a 10x tissue GT PNG.")
    parser.add_argument("--he_path", required=True)
    parser.add_argument("--gt_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--quality", type=int, default=90)
    args = parser.parse_args()

    pyvips = _load_pyvips()
    gt = Image.open(args.gt_path)
    target_w, target_h = gt.size

    img = pyvips.Image.new_from_file(args.he_path, access="sequential")
    xscale = target_w / img.width
    yscale = target_h / img.height
    resized = img.resize(xscale, vscale=yscale)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        resized.tiffsave(
            str(output_path),
            compression="jpeg",
            Q=args.quality,
            tile=True,
            tile_width=512,
            tile_height=512,
            bigtiff=True,
        )
    elif suffix in {".jpg", ".jpeg"}:
        resized.jpegsave(str(output_path), Q=args.quality)
    elif suffix == ".png":
        resized.pngsave(str(output_path))
    else:
        raise ValueError(f"Unsupported output suffix: {output_path.suffix}")

    print(f"{output_path} {target_w}x{target_h}")


if __name__ == "__main__":
    main()
