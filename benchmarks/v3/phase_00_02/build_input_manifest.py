# 作用：构建 v3 实验目录和统一输入 manifest，并输出阶段 0-2 的验证报告。

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


DEFAULT_MAIN_MANIFEST = Path("/nfs-medical3/zyh/cellatlas_gdph_benchmark_v2/manifests/main_20.csv")
DEFAULT_CELLS_ROOT = Path("/nfs-medical3/zyh/cellatlas_gdph_benchmark_v2/cells")
DEFAULT_EXISTING_SUPERPIXEL_ROOT = Path(
    "/nfs-medical3/zyh/cellatlas_pret_superpixel_main20_full10x_auto_physical/pret_superpixel"
)
DEFAULT_PROMPT_CSV = Path("/nfs-medical3/zyh/pret_eval_prompt_class_fix_20260706_gt_pure_prompts/prompts.csv")
DEFAULT_OUTPUT_ROOT = Path("/nfs-medical3/zyh/v3/cellatlas_pret_superpixel_aaai_v3_multiscale_refiner")
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent

OUTPUT_SUBDIRS = [
    "multiscale_tokens",
    "prompt_tasks",
    "scale_selection",
    "mask_decoder",
    "graph_refiner",
    "evaluations",
    "visualizations",
    "reports",
]

MANIFEST_FIELDS = [
    "image_id",
    "he_path",
    "gt_mask_path",
    "nucleus_class_path",
    "nucleus_instance_path",
    "part",
    "fold",
    "cell_dir",
    "cell_feature_raw_path",
    "cell_feature_reg_path",
    "cell_feature_proj_path",
    "cell_coord_path",
    "cell_polygon_path",
    "existing_superpixel_dir",
    "existing_superpixel_path",
    "existing_superpixel_csv_path",
    "existing_validation_path",
    "base_token_path",
    "enhanced_token_path",
    "prompt_csv_path",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_row(
    row: dict[str, str],
    cells_root: Path,
    existing_superpixel_root: Path,
    prompt_csv: Path,
) -> dict[str, str]:
    image_id = row["image_id"]
    cell_dir = cells_root / image_id
    existing_superpixel_dir = existing_superpixel_root / image_id
    return {
        "image_id": image_id,
        "he_path": row["he_path"],
        "gt_mask_path": row["tissue_gt_path"],
        "nucleus_class_path": row["nucleus_class_path"],
        "nucleus_instance_path": row["nucleus_instance_path"],
        "part": row.get("part", ""),
        "fold": row.get("fold", ""),
        "cell_dir": str(cell_dir),
        "cell_feature_raw_path": str(cell_dir / "raw.npy"),
        "cell_feature_reg_path": str(cell_dir / "reg.npy"),
        "cell_feature_proj_path": str(cell_dir / "proj.npy"),
        "cell_coord_path": str(cell_dir / "cells.csv"),
        "cell_polygon_path": str(cell_dir / "polygons.jsonl"),
        "existing_superpixel_dir": str(existing_superpixel_dir),
        "existing_superpixel_path": str(existing_superpixel_dir / "superpixels.npy"),
        "existing_superpixel_csv_path": str(existing_superpixel_dir / "superpixels.csv"),
        "existing_validation_path": str(existing_superpixel_dir / "validation.json"),
        "base_token_path": str(existing_superpixel_dir / "tokens_image_cell_reg_cellw0p5.npy"),
        "enhanced_token_path": str(existing_superpixel_dir / "tokens_image_cell_reg_texture_cellstats.npy"),
        "prompt_csv_path": str(prompt_csv),
    }


def path_fields() -> list[str]:
    return [field for field in MANIFEST_FIELDS if field.endswith("_path") or field.endswith("_dir")]


def validate_rows(rows: list[dict[str, str]]) -> tuple[Counter[str], list[dict[str, str]]]:
    present = Counter()
    missing: list[dict[str, str]] = []
    for row in rows:
        for field in path_fields():
            value = row.get(field, "")
            if not value:
                missing.append({"image_id": row["image_id"], "field": field, "path": value})
                continue
            if Path(value).exists():
                present[field] += 1
            else:
                missing.append({"image_id": row["image_id"], "field": field, "path": value})
    return present, missing


def prompt_summary(prompt_csv: Path) -> dict[str, object]:
    rows = read_csv(prompt_csv)
    sources = sorted({row.get("prompt_source", "") for row in rows})
    images = sorted({row.get("image_id", "") for row in rows})
    shots = sorted({row.get("shot", "") for row in rows})
    return {
        "prompt_csv": str(prompt_csv),
        "prompt_rows": len(rows),
        "prompt_sources": sources,
        "prompt_image_count": len(images),
        "shots": shots,
    }


def write_report(
    report_path: Path,
    output_root: Path,
    manifest_path: Path,
    main_manifest: Path,
    rows: list[dict[str, str]],
    present: Counter[str],
    missing: list[dict[str, str]],
    prompt: dict[str, object],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 00-02 输入数据报告",
        "",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- output_root: `{output_root}`",
        f"- input_manifest: `{main_manifest}`",
        f"- data_manifest_v3: `{manifest_path}`",
        f"- image_rows: {len(rows)}",
        f"- prompt_csv: `{prompt['prompt_csv']}`",
        f"- prompt_rows: {prompt['prompt_rows']}",
        f"- prompt_sources: {', '.join(prompt['prompt_sources'])}",
        f"- prompt_image_count: {prompt['prompt_image_count']}",
        f"- prompt_shots: {', '.join(prompt['shots'])}",
        "",
        "## 目录",
        "",
    ]
    for subdir in OUTPUT_SUBDIRS:
        lines.append(f"- `{output_root / 'pret_superpixel' / subdir}`")
    lines.extend(["", "## 文件存在性统计", ""])
    for field in path_fields():
        lines.append(f"- {field}: {present[field]}/{len(rows)}")
    lines.extend(["", "## 缺失文件", ""])
    if missing:
        for item in missing:
            lines.append(f"- {item['image_id']} {item['field']}: `{item['path']}`")
    else:
        lines.append("- 无")
    lines.extend(
        [
            "",
            "## 验证结论",
            "",
            f"- passed: {str(not missing).lower()}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validation_json(
    path: Path,
    rows: list[dict[str, str]],
    present: Counter[str],
    missing: list[dict[str, str]],
    prompt: dict[str, object],
) -> None:
    payload = {
        "passed": not missing,
        "image_rows": len(rows),
        "file_presence": {field: present[field] for field in path_fields()},
        "missing_files": missing,
        "prompt_summary": prompt,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v3 input manifest through goal section #2.")
    parser.add_argument("--main_manifest", type=Path, default=DEFAULT_MAIN_MANIFEST)
    parser.add_argument("--cells_root", type=Path, default=DEFAULT_CELLS_ROOT)
    parser.add_argument("--existing_superpixel_root", type=Path, default=DEFAULT_EXISTING_SUPERPIXEL_ROOT)
    parser.add_argument("--prompt_csv", type=Path, default=DEFAULT_PROMPT_CSV)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for required in [args.main_manifest, args.prompt_csv, args.cells_root, args.existing_superpixel_root]:
        if not required.exists():
            raise FileNotFoundError(required)

    pret_root = args.output_root / "pret_superpixel"
    for subdir in OUTPUT_SUBDIRS:
        (pret_root / subdir).mkdir(parents=True, exist_ok=True)

    source_rows = read_csv(args.main_manifest)
    rows = [
        build_row(row, args.cells_root, args.existing_superpixel_root, args.prompt_csv)
        for row in source_rows
    ]
    present, missing = validate_rows(rows)
    prompt = prompt_summary(args.prompt_csv)

    manifest_path = pret_root / "data_manifest_v3.csv"
    report_path = args.report_dir / "report.md"
    validation_path = pret_root / "reports" / "phase_00_02_validation.json"

    if missing:
        write_report(report_path, args.output_root, manifest_path, args.main_manifest, rows, present, missing, prompt)
        write_validation_json(validation_path, rows, present, missing, prompt)
        raise RuntimeError(f"missing required files: {len(missing)}")

    write_csv(manifest_path, rows, MANIFEST_FIELDS)
    write_report(report_path, args.output_root, manifest_path, args.main_manifest, rows, present, missing, prompt)
    write_validation_json(validation_path, rows, present, missing, prompt)
    print(f"wrote {manifest_path}")
    print(f"wrote {report_path}")
    print(f"wrote {validation_path}")


if __name__ == "__main__":
    main()
