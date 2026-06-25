from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT
from benchmarks.gdph_v2.select_pilot import sampled_class_fractions


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_profiles(path: Path, profiles: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_id", "fractions"])
        writer.writeheader()
        for image_id in sorted(profiles):
            writer.writerow(
                {"image_id": image_id, "fractions": json.dumps(profiles[image_id].tolist())}
            )
    temporary.replace(path)


def assign_image_folds(rows: list[dict[str, str]], n_folds: int = 5) -> list[dict[str, str]]:
    by_part: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_part.setdefault(row["part"], []).append(row)
    result = []
    for part_index, part in enumerate(sorted(by_part)):
        part_rows = sorted(by_part[part], key=lambda item: item["image_id"])
        for index, row in enumerate(part_rows):
            item = dict(row)
            item["fold"] = str((index + part_index) % n_folds)
            result.append(item)
    fold_counts = np.bincount([int(row["fold"]) for row in result], minlength=n_folds)
    if fold_counts.max() - fold_counts.min() > 1:
        raise RuntimeError(f"unbalanced fold assignment: {fold_counts.tolist()}")
    return sorted(result, key=lambda item: (int(item["fold"]), item["part"], item["image_id"]))


def choose_main_rows(
    candidates: list[dict[str, str]],
    profiles: dict[str, np.ndarray],
    pilot_ids: set[str],
    per_part: int = 4,
) -> list[dict[str, str]]:
    by_part: dict[str, list[dict[str, str]]] = {}
    for row in candidates:
        by_part.setdefault(row["part"], []).append(row)
    global_fraction = np.mean([profiles[row["image_id"]] for row in candidates], axis=0)
    weights = np.minimum(10.0, 1.0 / np.sqrt(global_fraction + 1e-4))
    selected: list[dict[str, str]] = []
    for part in sorted(by_part):
        pool = sorted(by_part[part], key=lambda item: item["image_id"])
        chosen = [row for row in pool if row["image_id"] in pilot_ids]
        if len(chosen) > per_part:
            raise ValueError(f"{part} has too many required pilot rows")
        coverage = sum((profiles[row["image_id"]] for row in chosen), np.zeros(12))
        while len(chosen) < per_part:
            remaining = [row for row in pool if row not in chosen]
            if not remaining:
                raise ValueError(f"{part} has fewer than {per_part} candidates")
            def gain(row: dict[str, str]) -> tuple[float, str]:
                updated = coverage + profiles[row["image_id"]]
                value = float(np.sum(weights * (np.sqrt(updated) - np.sqrt(coverage))))
                return value, row["image_id"]
            best = max(remaining, key=gain)
            chosen.append(best)
            coverage += profiles[best["image_id"]]
        selected.extend(chosen)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Select 20 class-diverse GDPH main slides.")
    parser.add_argument(
        "--selected_40",
        default="/nfs-medical3/zyh/cellatlas_tissue_linear_probe_v1/splits/selected_40.csv",
    )
    parser.add_argument(
        "--prepared_manifest",
        default="/nfs-medical3/zyh/cellatlas_tissue_linear_probe_v1/prepared/prepared_manifest.csv",
    )
    parser.add_argument(
        "--pilot_csv", default=str(DEFAULT_OUTPUT_ROOT / "manifests" / "pilot_5.csv")
    )
    parser.add_argument(
        "--profiles_csv",
        default=str(DEFAULT_OUTPUT_ROOT / "manifests" / "candidate_class_profiles.csv"),
    )
    parser.add_argument(
        "--output_csv", default=str(DEFAULT_OUTPUT_ROOT / "manifests" / "main_20.csv")
    )
    parser.add_argument(
        "--validation_json",
        default=str(DEFAULT_OUTPUT_ROOT / "config" / "main_selection_validation.json"),
    )
    parser.add_argument("--sample_stride", type=int, default=64)
    args = parser.parse_args()

    candidates = _read_csv(args.selected_40)
    prepared = {row["image_id"]: row for row in _read_csv(args.prepared_manifest)}
    profiles_path = Path(args.profiles_csv)
    profiles: dict[str, np.ndarray] = {}
    if profiles_path.exists():
        profiles.update({
            row["image_id"]: np.asarray(json.loads(row["fractions"]), dtype=np.float64)
            for row in _read_csv(profiles_path)
        })
    candidate_ids = {row["image_id"] for row in candidates}
    profiles = {image_id: value for image_id, value in profiles.items() if image_id in candidate_ids}
    for row in candidates:
        image_id = row["image_id"]
        if image_id in profiles:
            continue
        profiles[image_id] = sampled_class_fractions(
            prepared[image_id]["gt_mask_path"], args.sample_stride
        )
        _write_profiles(profiles_path, profiles)
        print(f"profile {len(profiles)}/{len(candidates)} {image_id}", flush=True)

    pilot_ids = {row["image_id"] for row in _read_csv(args.pilot_csv)}
    selected = choose_main_rows(candidates, profiles, pilot_ids, per_part=4)
    selected = assign_image_folds(selected, n_folds=5)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)

    part_counts = {
        part: sum(row["part"] == part for row in selected)
        for part in sorted({row["part"] for row in selected})
    }
    fold_counts = {
        str(fold): sum(row["fold"] == str(fold) for row in selected)
        for fold in range(5)
    }
    selected_ids = {row["image_id"] for row in selected}
    report = {
        "candidate_profile_count": len(profiles),
        "selected_count": len(selected),
        "unique_selected_count": len(selected_ids),
        "part_counts": part_counts,
        "fold_counts": fold_counts,
        "pilot_ids": sorted(pilot_ids),
        "all_pilots_retained": pilot_ids <= selected_ids,
    }
    report["passed"] = bool(
        report["candidate_profile_count"] == len(candidates) == 40
        and report["selected_count"] == report["unique_selected_count"] == 20
        and len(part_counts) == 5
        and set(part_counts.values()) == {4}
        and set(fold_counts.values()) == {4}
        and report["all_pilots_retained"]
    )
    validation_path = Path(args.validation_json)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = validation_path.with_suffix(validation_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(validation_path)
    if not report["passed"]:
        raise RuntimeError(f"main selection validation failed: {report}")
    print(output_path)
    print(validation_path)
    for row in selected:
        print(row["fold"], row["part"], row["image_id"], "pilot" if row["image_id"] in pilot_ids else "")


if __name__ == "__main__":
    main()
