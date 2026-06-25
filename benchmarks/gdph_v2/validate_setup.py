from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT, sha256_file


Image.MAX_IMAGE_PIXELS = None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def validate_setup(output_root: str | Path) -> dict:
    root = Path(output_root)
    with open(root / "config" / "experiment.json", "r", encoding="utf-8") as file:
        config = json.load(file)

    model_checks = {}
    for name, record in config["models"].items():
        path = Path(record["path"])
        exists = path.is_file()
        actual_hash = sha256_file(path) if exists else None
        model_checks[name] = {
            "exists": exists,
            "bytes_match": exists and path.stat().st_size == record["bytes"],
            "sha256_match": actual_hash == record["sha256"],
        }

    mapping_rows = _read_csv(Path(config["dataset"]["mapping_csv"]))
    pilot_rows = _read_csv(root / "manifests" / "pilot_5.csv")
    pilot_checks = []
    for row in pilot_rows:
        he_path = Path(row["he_path"])
        gt_path = Path(row["tissue_gt_path"])
        nucleus_instance_path = Path(row["nucleus_instance_path"])
        nucleus_class_path = Path(row["nucleus_class_path"])
        old_resized_cells_path = Path(
            "/nfs-medical3/zyh/cellatlas_tissue_linear_probe_v1/prepared/cell_labels"
        ) / f"{row['image_id']}_cell_labels.csv"
        with (
            Image.open(he_path) as he_image,
            Image.open(gt_path) as gt_image,
            Image.open(nucleus_instance_path) as nucleus_instance_image,
            Image.open(nucleus_class_path) as nucleus_class_image,
        ):
            x_scale = gt_image.width / he_image.width
            y_scale = gt_image.height / he_image.height
            he_size = [he_image.width, he_image.height]
            nucleus_instance_size = [
                nucleus_instance_image.width,
                nucleus_instance_image.height,
            ]
            nucleus_class_size = [nucleus_class_image.width, nucleus_class_image.height]
            pilot_checks.append(
                {
                    "image_id": row["image_id"],
                    "part": row["part"],
                    "selection_reason": row["selection_reason"],
                    "he_exists": he_path.is_file(),
                    "gt_exists": gt_path.is_file(),
                    "nucleus_instance_exists": nucleus_instance_path.is_file(),
                    "nucleus_class_exists": nucleus_class_path.is_file(),
                    "old_resized_cells_exists": old_resized_cells_path.is_file(),
                    "he_size": he_size,
                    "gt_size": [gt_image.width, gt_image.height],
                    "nucleus_instance_size": nucleus_instance_size,
                    "nucleus_class_size": nucleus_class_size,
                    "nucleus_instance_matches_he": nucleus_instance_size == he_size,
                    "nucleus_class_matches_he": nucleus_class_size == he_size,
                    "x_scale": x_scale,
                    "y_scale": y_scale,
                    "gt_is_approximately_half_scale": (
                        abs(x_scale - 0.5) < 1e-3 and abs(y_scale - 0.5) < 1e-3
                    ),
                }
            )

    checks = {
        "mapping_rows": len(mapping_rows),
        "mapping_expected_963": len(mapping_rows) == 963,
        "pilot_rows": len(pilot_rows),
        "pilot_expected_5": len(pilot_rows) == 5,
        "pilot_unique_images": len({row["image_id"] for row in pilot_rows}) == 5,
        "pilot_unique_parts": len({row["part"] for row in pilot_rows}) == 5,
        "pilot_unique_reasons": len({row["selection_reason"] for row in pilot_rows}) == 5,
        "model_checks": model_checks,
        "pilots": pilot_checks,
    }
    checks["passed"] = all(
        [
            checks["mapping_expected_963"],
            checks["pilot_expected_5"],
            checks["pilot_unique_images"],
            checks["pilot_unique_parts"],
            checks["pilot_unique_reasons"],
            all(all(values.values()) for values in model_checks.values()),
            all(
                pilot["he_exists"]
                and pilot["gt_exists"]
                and pilot["nucleus_instance_exists"]
                and pilot["nucleus_class_exists"]
                and pilot["old_resized_cells_exists"]
                and pilot["nucleus_instance_matches_he"]
                and pilot["nucleus_class_matches_he"]
                and pilot["gt_is_approximately_half_scale"]
                for pilot in pilot_checks
            ),
        ]
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GDPH v2 setup and pilot selection.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    report = validate_setup(args.output_root)
    report_path = Path(args.output_root) / "config" / "setup_validation.json"
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    temporary.replace(report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit("GDPH v2 setup validation failed")


if __name__ == "__main__":
    main()
