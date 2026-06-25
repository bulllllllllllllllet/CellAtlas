from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None


def _fullres_valid(item: Any) -> bool:
    return bool(
        isinstance(item, dict)
        and item.get("passed")
        and item.get("features_finite") is True
        and item.get("cross_tile_near_duplicates_within_2px") == 0
        and item.get("cache_schema_version") == 2
        and item.get("xcell_token_order") == "cellpose_label_ascending_reference"
    )


def _label_valid(item: Any) -> bool:
    return bool(
        isinstance(item, dict)
        and item.get("passed")
        and item.get("indices_equal") is True
        and item.get("purity_definition") == "majority_class_pixels / all_polygon_pixels"
    )
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def audit_experiment(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    pilot = _read_csv(root / "manifests" / "pilot_5.csv")
    main = _read_csv(root / "manifests" / "main_20.csv")
    setup = _json(root / "config" / "setup_validation.json")
    experiment_config = _json(root / "config" / "experiment.json")
    model_compatibility = _json(root / "config" / "model_compatibility_validation.json")
    smoke = _json(root / "config" / "fullres_smoke_validation.json")
    main_selection = _json(root / "config" / "main_selection_validation.json")
    scale_validation = _json(
        root / "cell_classification" / "scale_comparison" / "validation.json"
    )
    query_validation = _json(root / "region_retrieval" / "queries_validation.json")
    retrieval_validation = _json(
        root / "region_retrieval" / "cell_retrieval_validation.json"
    )
    coverage_validation = _json(
        root / "dense_evaluation" / "cell_coverage_validation.json"
    )
    hybrid_validation = _json(
        root / "dense_evaluation" / "patch_hybrid_validation.json"
    )
    probe_metrics = {
        head: _json(
            root / "cell_classification" / f"{head}_balanced_valid_only" / "metrics.json"
        )
        for head in ("raw", "reg", "proj")
    }

    pilot_fullres = [
        _json(root / "cells" / row["image_id"] / "validation.json") for row in pilot
    ]
    pilot_labels = [
        _json(root / "cells" / row["image_id"] / "label_validation.json") for row in pilot
    ]
    main_fullres = [
        _json(root / "cells" / row["image_id"] / "validation.json") for row in main
    ]
    main_labels = [
        _json(root / "cells" / row["image_id"] / "label_validation.json") for row in main
    ]
    patch_validations = [
        _json(root / "patches" / row["image_id"] / "validation.json") for row in main
    ]

    stages = [
        {
            "stage": 1,
            "name": "experiment_setup",
            "passed": bool(
                isinstance(setup, dict)
                and setup.get("passed")
                and isinstance(model_compatibility, dict)
                and model_compatibility.get("passed")
                and isinstance(experiment_config, dict)
                and experiment_config.get("dataset", {}).get(
                    "external_to_xcellformer_training"
                )
                is True
                and experiment_config.get("inference", {}).get("image_scale")
                == "original_level_0"
                and experiment_config.get("inference", {}).get("xcell_token_order")
                == "cellpose_label_ascending_reference"
                and all(
                    item.get("old_resized_cells_exists")
                    and item.get("nucleus_instance_matches_he")
                    and item.get("nucleus_class_matches_he")
                    and item.get("gt_is_approximately_half_scale")
                    for item in setup.get("pilots", [])
                )
                and len(setup.get("pilots", [])) == 5
            ),
            "evidence": str(root / "config" / "setup_validation.json"),
        },
        {
            "stage": 2,
            "name": "pilot_selection",
            "passed": len(pilot) == 5
            and len({row.get("part") for row in pilot}) == 5
            and len({row.get("selection_reason") for row in pilot}) == 5,
            "evidence": str(root / "manifests" / "pilot_5.csv"),
        },
        {
            "stage": 3,
            "name": "full_resolution_inference",
            "passed": bool(
                isinstance(smoke, dict)
                and smoke.get("passed")
                and smoke.get("xcell_token_order") == "cellpose_label_ascending_reference"
                and smoke.get("labels_strictly_increasing") is True
            )
            and len(pilot_fullres) == 5
            and all(_fullres_valid(item) for item in pilot_fullres),
            "completed_pilot_slides": sum(
                _fullres_valid(item) for item in pilot_fullres
            ),
            "evidence": str(root / "cells"),
        },
        {
            "stage": 4,
            "name": "polygon_majority_labels",
            "passed": len(pilot_labels) == 5
            and all(_label_valid(item) for item in pilot_labels),
            "completed_pilot_slides": sum(
                _label_valid(item) for item in pilot_labels
            ),
            "evidence": str(root / "cells"),
        },
        {
            "stage": 5,
            "name": "paired_scale_validation",
            "passed": bool(
                isinstance(scale_validation, dict) and scale_validation.get("passed")
            ),
            "evidence": str(
                root / "cell_classification" / "scale_comparison" / "validation.json"
            ),
        },
        {
            "stage": 6,
            "name": "main_20_five_fold",
            "passed": len(main) == 20
            and len({row.get("fold") for row in main}) == 5
            and isinstance(main_selection, dict)
            and main_selection.get("passed")
            and all(_fullres_valid(item) for item in main_fullres)
            and all(_label_valid(item) for item in main_labels)
            and all(
                isinstance(probe_metrics[head], dict)
                and isinstance(probe_metrics[head].get("validation"), dict)
                and probe_metrics[head]["validation"].get("passed")
                for head in ("raw", "reg", "proj")
            ),
            "selected_images": len(main),
            "completed_fullres": sum(
                _fullres_valid(item) for item in main_fullres
            ),
            "completed_labels": sum(
                _label_valid(item) for item in main_labels
            ),
            "evidence": str(root / "cell_classification"),
        },
        {
            "stage": 7,
            "name": "query_region_retrieval",
            "passed": bool(
                isinstance(query_validation, dict)
                and query_validation.get("passed")
                and isinstance(retrieval_validation, dict)
                and retrieval_validation.get("passed")
            ),
            "evidence": str(root / "region_retrieval"),
        },
        {
            "stage": 8,
            "name": "sparse_coverage",
            "passed": bool(
                isinstance(coverage_validation, dict) and coverage_validation.get("passed")
            ),
            "evidence": str(root / "dense_evaluation" / "cell_coverage_validation.json"),
        },
        {
            "stage": 9,
            "name": "patch_and_hybrid",
            "passed": len(main) == 20
            and len(patch_validations) == 20
            and all(
                isinstance(item, dict)
                and item.get("passed")
                and item.get("blank_ratio") == 0.995
                and item.get("purity_definition")
                == "majority_class_pixels / all_patch_gt_pixels"
                for item in patch_validations
            )
            and isinstance(hybrid_validation, dict)
            and hybrid_validation.get("passed")
            and (root / "FINAL_REPORT.md").is_file()
            and (root / "FINAL_REPORT.md").stat().st_size > 0,
            "completed_patch_slides": sum(
                isinstance(item, dict) and bool(item.get("passed")) for item in patch_validations
            ),
            "evidence": str(root / "dense_evaluation"),
        },
    ]
    report = {
        "output_root": str(root),
        "complete": all(stage["passed"] for stage in stages),
        "passed_stages": sum(bool(stage["passed"]) for stage in stages),
        "total_stages": 9,
        "stages": stages,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit all nine GDPH v2 experiment stages.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_root)
    report = audit_experiment(root)
    output_path = root / "config" / "experiment_audit.json"
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    temporary.replace(output_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.require_complete and not report["complete"]:
        raise SystemExit("experiment audit is incomplete")


if __name__ == "__main__":
    main()
