from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT


TARGET_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("tumor_epithelium", (0,)),
    ("tumor_stroma", (1,)),
    ("muscle_or_submucosa", (7, 6)),
    ("immune_or_normal_gland", (8, 4)),
    ("acellular_fat_mucus_necrosis", (10, 9, 3)),
)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def sampled_class_fractions(mask_path: str | Path, stride: int = 64) -> np.ndarray:
    # Sequentially load the array before strided sampling.  Striding directly
    # over an NFS-backed mmap turns each sampled row into many random page
    # faults and is substantially slower than one sequential read.
    mask = np.load(mask_path)
    sampled = np.asarray(mask[::stride, ::stride]).reshape(-1)
    valid = sampled[(sampled >= 0) & (sampled < 12)]
    if valid.size == 0:
        return np.zeros(12, dtype=np.float64)
    counts = np.bincount(valid.astype(np.int64), minlength=12)
    return counts / counts.sum()


def choose_pilot_rows(
    selected_rows: list[dict[str, str]],
    prepared_rows: list[dict[str, str]],
    stride: int = 64,
) -> list[dict[str, str]]:
    selected_by_id = {row["image_id"]: row for row in selected_rows}
    candidates: list[dict] = []
    for prepared in prepared_rows:
        image_id = prepared["image_id"]
        selected = selected_by_id.get(image_id)
        if selected is None:
            continue
        fractions = sampled_class_fractions(prepared["gt_mask_path"], stride=stride)
        candidates.append({"selected": selected, "prepared": prepared, "fractions": fractions})

    parts = sorted({candidate["selected"].get("part", "unknown") for candidate in candidates})
    if len(parts) < len(TARGET_GROUPS):
        raise ValueError(f"Need at least {len(TARGET_GROUPS)} parts, found {parts}")

    best_by_target_part: dict[tuple[int, str], tuple[float, dict]] = {}
    for target_index, (_, class_ids) in enumerate(TARGET_GROUPS):
        for part in parts:
            eligible = [candidate for candidate in candidates if candidate["selected"].get("part") == part]
            if not eligible:
                continue
            scored = [
                (float(candidate["fractions"][list(class_ids)].sum()), candidate)
                for candidate in eligible
            ]
            best_by_target_part[(target_index, part)] = max(
                scored, key=lambda item: (item[0], item[1]["selected"]["image_id"])
            )

    best_assignment: tuple[float, tuple[str, ...]] | None = None
    for assigned_parts in itertools.permutations(parts, len(TARGET_GROUPS)):
        score = sum(
            best_by_target_part[(target_index, part)][0]
            for target_index, part in enumerate(assigned_parts)
        )
        if best_assignment is None or score > best_assignment[0]:
            best_assignment = (score, assigned_parts)
    if best_assignment is None:
        raise RuntimeError("Could not assign pilot targets to parts")

    result: list[dict[str, str]] = []
    for target_index, part in enumerate(best_assignment[1]):
        target_name, _ = TARGET_GROUPS[target_index]
        score, candidate = best_by_target_part[(target_index, part)]
        row = dict(candidate["selected"])
        row.update(
            {
                "selection_reason": target_name,
                "target_fraction": f"{score:.8f}",
                "sampled_class_fractions": json.dumps(
                    candidate["fractions"].round(8).tolist(), separators=(",", ":")
                ),
                "prepared_gt_mask_path": candidate["prepared"]["gt_mask_path"],
            }
        )
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Select five representative GDPH pilot slides.")
    parser.add_argument(
        "--selected_csv",
        default="/nfs-medical3/zyh/cellatlas_tissue_linear_probe_v1/splits/selected_40.csv",
    )
    parser.add_argument(
        "--prepared_manifest",
        default="/nfs-medical3/zyh/cellatlas_tissue_linear_probe_v1/prepared/prepared_manifest.csv",
    )
    parser.add_argument(
        "--output_csv", default=str(DEFAULT_OUTPUT_ROOT / "manifests" / "pilot_5.csv")
    )
    parser.add_argument("--sample_stride", type=int, default=64)
    args = parser.parse_args()

    rows = choose_pilot_rows(
        _read_csv(args.selected_csv),
        _read_csv(args.prepared_manifest),
        stride=args.sample_stride,
    )
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output_path)
    for row in rows:
        print(row["part"], row["selection_reason"], row["image_id"], row["target_fraction"])


if __name__ == "__main__":
    main()
