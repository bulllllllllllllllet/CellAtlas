from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _he_stem_from_tiff(path: Path) -> str | None:
    stem = path.stem
    if stem.endswith("-class") or stem.endswith("-instance"):
        return None
    return stem


def build_pairs(he_root: Path, tissue_root: Path) -> tuple[list[dict[str, str]], list[str], list[str]]:
    he_by_stem: dict[str, Path] = {}
    for path in he_root.rglob("*.tiff"):
        stem = _he_stem_from_tiff(path)
        if stem is None:
            continue
        he_by_stem[stem] = path

    png_by_stem = {path.stem: path for path in tissue_root.rglob("*.png")}

    rows: list[dict[str, str]] = []
    for stem in sorted(set(he_by_stem) & set(png_by_stem)):
        he_path = he_by_stem[stem]
        class_path = he_path.with_name(f"{stem}-class.tiff")
        instance_path = he_path.with_name(f"{stem}-instance.tiff")
        rows.append(
            {
                "image_id": stem,
                "he_path": str(he_path),
                "tissue_gt_path": str(png_by_stem[stem]),
                "nucleus_class_path": str(class_path if class_path.exists() else ""),
                "nucleus_instance_path": str(instance_path if instance_path.exists() else ""),
            }
        )

    missing_png = sorted(set(he_by_stem) - set(png_by_stem))
    missing_he = sorted(set(png_by_stem) - set(he_by_stem))
    return rows, missing_png, missing_he


def main() -> None:
    parser = argparse.ArgumentParser(description="Match DP500 HE TIFF files to 10x tissue PNG labels.")
    parser.add_argument("--he_root", required=True, help="DP500-COAD_READ-GDPH_P01 root with part folders")
    parser.add_argument("--tissue_root", required=True, help="tissue_seg@10x root or nested PNG folder")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--missing_report", default=None)
    args = parser.parse_args()

    rows, missing_png, missing_he = build_pairs(Path(args.he_root), Path(args.tissue_root))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "he_path",
                "tissue_gt_path",
                "nucleus_class_path",
                "nucleus_instance_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if args.missing_report:
        report = Path(args.missing_report)
        report.parent.mkdir(parents=True, exist_ok=True)
        with open(report, "w", encoding="utf-8") as f:
            f.write(f"matched={len(rows)}\n")
            f.write(f"missing_png={len(missing_png)}\n")
            for item in missing_png:
                f.write(f"  he_without_png: {item}\n")
            f.write(f"missing_he={len(missing_he)}\n")
            for item in missing_he:
                f.write(f"  png_without_he: {item}\n")

    print(f"matched={len(rows)} missing_png={len(missing_png)} missing_he={len(missing_he)}")
    print(output_csv)


if __name__ == "__main__":
    main()
