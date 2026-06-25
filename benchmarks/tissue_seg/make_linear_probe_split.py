from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def _part_name(he_path: str) -> str:
    parts = Path(he_path).parts
    for part in reversed(parts):
        if part.startswith("part"):
            return part
    return "unknown"


def _read_mapping(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"image_id", "he_path", "tissue_gt_path"}
    if rows:
        missing = required - set(rows[0])
        if missing:
            raise ValueError(f"mapping CSV missing columns: {sorted(missing)}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a balanced image-level split for tissue linear-probe training."
    )
    parser.add_argument("--mapping_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples_per_part", type=int, default=8)
    parser.add_argument("--train_per_part", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260622)
    args = parser.parse_args()

    if args.train_per_part >= args.samples_per_part:
        raise ValueError("--train_per_part must be smaller than --samples_per_part")

    rows = _read_mapping(args.mapping_csv)
    by_part: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        row = dict(row)
        row["part"] = _part_name(row["he_path"])
        by_part[row["part"]].append(row)

    rng = random.Random(args.seed)
    selected: list[dict[str, str]] = []
    split_rows: list[dict[str, str]] = []
    for part in sorted(by_part):
        candidates = sorted(by_part[part], key=lambda r: r["image_id"])
        if len(candidates) < args.samples_per_part:
            raise ValueError(
                f"{part} has only {len(candidates)} images, "
                f"but {args.samples_per_part} were requested."
            )
        sampled = rng.sample(candidates, args.samples_per_part)
        train_ids = {r["image_id"] for r in rng.sample(sampled, args.train_per_part)}
        sampled = sorted(sampled, key=lambda r: r["image_id"])
        selected.extend(sampled)

        for row in sampled:
            split_rows.append(
                {
                    "image_id": row["image_id"],
                    "split": "train" if row["image_id"] in train_ids else "test",
                    "part": part,
                }
            )

    output_dir = Path(args.output_dir)
    selected_path = output_dir / f"selected_{len(selected)}.csv"
    split_path = output_dir / f"split_{sum(r['split'] == 'train' for r in split_rows)}_{sum(r['split'] == 'test' for r in split_rows)}.csv"

    selected_fields = list(rows[0].keys()) + ["part"] if rows else ["part"]
    _write_csv(selected_path, selected, selected_fields)
    _write_csv(split_path, split_rows, ["image_id", "split", "part"])

    print(f"selected={len(selected)}")
    print(f"train={sum(r['split'] == 'train' for r in split_rows)}")
    print(f"test={sum(r['split'] == 'test' for r in split_rows)}")
    print(selected_path)
    print(split_path)


if __name__ == "__main__":
    main()
