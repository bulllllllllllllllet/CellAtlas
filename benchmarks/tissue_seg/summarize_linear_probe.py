from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize linear-probe metrics for multiple heads.")
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--heads", nargs="+", default=["reg", "proj"])
    args = parser.parse_args()

    rows: list[dict[str, str | float]] = []
    for head in args.heads:
        metrics_path = Path(args.results_root) / f"linear_{head}" / "metrics.json"
        if not metrics_path.exists():
            print(f"[skip] missing {metrics_path}")
            continue
        metrics = _read_metrics(metrics_path)
        rows.append(
            {
                "head": head,
                "cell_accuracy": metrics.get("cell_accuracy", ""),
                "cell_mean_iou": metrics.get("cell_mean_iou", ""),
                "cell_macro_f1": metrics.get("cell_macro_f1", ""),
                "pixel_accuracy": metrics.get("pixel_accuracy", ""),
                "pixel_mean_iou": metrics.get("pixel_mean_iou", ""),
                "pixel_mean_dice": metrics.get("pixel_mean_dice", ""),
                "train_images": len(metrics.get("train_images", [])),
                "test_images": len(metrics.get("test_images", [])),
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "head",
        "cell_accuracy",
        "cell_mean_iou",
        "cell_macro_f1",
        "pixel_accuracy",
        "pixel_mean_iou",
        "pixel_mean_dice",
        "train_images",
        "test_images",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(output_csv)


if __name__ == "__main__":
    main()
