from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dominant RGB colors in tissue GT PNGs.")
    parser.add_argument("--gt", nargs="+", required=True, help="GT PNG path(s)")
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    counts: dict[tuple[int, int, int], int] = {}
    for path in args.gt:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).reshape(-1, 3)
        colors, color_counts = np.unique(arr, axis=0, return_counts=True)
        for color, count in zip(colors, color_counts, strict=False):
            key = tuple(int(v) for v in color)
            counts[key] = counts.get(key, 0) + int(count)

    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)[: args.top_k]
    result = [{"rgb": list(rgb), "count": count} for rgb, count in ranked]

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    else:
        for item in result:
            print(f"{item['rgb']}\t{item['count']}")


if __name__ == "__main__":
    main()
