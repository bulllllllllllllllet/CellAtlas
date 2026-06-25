from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path("/nfs-medical3/zyh/cellatlas_tissue_linear_probe_v1")
DEFAULT_XCELL_CHECKPOINT = (
    "/home/zyh/NewMedLabel/CellAtlas/experiments/"
    "crc012_holdout3/20260515_160711/he_model_best.pth"
)


def _read_selected(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"image_id", "he_path", "tissue_gt_path"}
    if rows:
        missing = required - set(rows[0])
        if missing:
            raise ValueError(f"selected CSV missing columns: {sorted(missing)}")
    return rows


def _run(
    cmd: list[str],
    log_path: Path,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    allow_timeout: bool = False,
) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(cmd))
    print(f"[log] {log_path}")
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log.write(f"\n[TIMEOUT] command exceeded {timeout} seconds\n")
            if allow_timeout:
                print(f"[timeout] {log_path}")
                return False
            raise
    if proc.returncode != 0:
        raise SystemExit(f"Command failed with exit code {proc.returncode}. See {log_path}")
    return True


def _stem(image_id: str) -> str:
    return f"{image_id}-10x"


def _paths(output_root: Path, image_id: str) -> dict[str, Path]:
    stem = _stem(image_id)
    infer_root = output_root / "inference"
    return {
        "resized_he": output_root / "resized_he_10x" / f"{stem}.tiff",
        "reg": infer_root / "temp_reg" / f"features_{stem}_reg.npy",
        "proj": infer_root / "temp_proj" / f"features_{stem}_proj.npy",
        "mask": infer_root / "temp_masks" / f"masks_{stem}.npy",
        "aligned_mask": infer_root / "temp_masks" / f"masks_{stem}_aligned.npy",
    }


def _resize_images(rows: list[dict[str, str]], output_root: Path, quality: int) -> None:
    for row in rows:
        paths = _paths(output_root, row["image_id"])
        if paths["resized_he"].exists():
            print(f"[skip resize] {row['image_id']}")
            continue
        _run(
            [
                sys.executable,
                "-m",
                "benchmarks.tissue_seg.make_he_10x",
                "--he_path",
                row["he_path"],
                "--gt_path",
                row["tissue_gt_path"],
                "--output_path",
                str(paths["resized_he"]),
                "--quality",
                str(quality),
            ],
            output_root / "logs" / f"{row['image_id']}_resize.log",
        )


def _row_outputs_complete(row: dict[str, str], output_root: Path) -> bool:
    paths = _paths(output_root, row["image_id"])
    return paths["reg"].exists() and paths["proj"].exists() and paths["mask"].exists()


def _completed_rows(rows: list[dict[str, str]], output_root: Path) -> list[dict[str, str]]:
    return [row for row in rows if _row_outputs_complete(row, output_root)]


def _run_inference(args: argparse.Namespace, rows: list[dict[str, str]], output_root: Path) -> None:
    if len(_completed_rows(rows, output_root)) == len(rows) and not args.force_inference:
        print("[skip inference] all feature and mask outputs already exist")
        return

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    timeout = args.inference_timeout_minutes * 60 if args.inference_timeout_minutes else None
    for row in rows:
        if _row_outputs_complete(row, output_root) and not args.force_inference:
            print(f"[skip inference] {row['image_id']}")
            continue
        paths = _paths(output_root, row["image_id"])
        ok = _run(
            [
                sys.executable,
                "new_inference_stream/inference.py",
                "--input_folder",
                str(paths["resized_he"]),
                "--output_folder",
                str(output_root / "inference"),
                "--model_path",
                args.model_path,
                "--ctranspath_checkpoint",
                args.ctranspath_checkpoint,
                "--cellpose_model",
                args.cellpose_model,
                "--k",
                str(args.k),
                "--max_workers",
                str(args.max_workers),
                "--batch_size",
                str(args.batch_size),
                "--device",
                args.device,
            ],
            output_root / "logs" / f"{row['image_id']}_inference.log",
            env=env,
            timeout=timeout,
            allow_timeout=args.skip_inference_timeouts,
        )
        if not ok:
            print(f"[skip timed out image] {row['image_id']}")


def _align_masks(rows: list[dict[str, str]], output_root: Path) -> None:
    for row in rows:
        if not _row_outputs_complete(row, output_root):
            print(f"[skip align missing inference] {row['image_id']}")
            continue
        paths = _paths(output_root, row["image_id"])
        if paths["aligned_mask"].exists():
            print(f"[skip align] {row['image_id']}")
            continue
        _run(
            [
                sys.executable,
                "-m",
                "benchmarks.tissue_seg.align_tiled_masks",
                "--masks_path",
                str(paths["mask"]),
                "--features_path",
                str(paths["reg"]),
                "--output_path",
                str(paths["aligned_mask"]),
            ],
            output_root / "logs" / f"{row['image_id']}_align.log",
        )


def _write_prepare_mapping(rows: list[dict[str, str]], output_root: Path) -> Path:
    mapping_path = output_root / "mapping_for_prepare.csv"
    rows = [
        row for row in rows
        if _row_outputs_complete(row, output_root) and _paths(output_root, row["image_id"])["aligned_mask"].exists()
    ]
    if not rows:
        raise SystemExit("No completed rows with aligned masks; cannot prepare labels.")
    with open(mapping_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "gt_path",
                "masks_path",
                "reg_path",
                "proj_path",
                "raw_path",
                "he_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            paths = _paths(output_root, row["image_id"])
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "gt_path": row["tissue_gt_path"],
                    "masks_path": str(paths["aligned_mask"]),
                    "reg_path": str(paths["reg"]),
                    "proj_path": str(paths["proj"]),
                    "raw_path": "",
                    "he_path": str(paths["resized_he"]),
                }
            )
    print(f"[mapping] {mapping_path}")
    return mapping_path


def _prepare_labels(args: argparse.Namespace, output_root: Path, mapping_path: Path) -> Path:
    prepared_dir = output_root / "prepared"
    manifest_path = prepared_dir / "prepared_manifest.csv"
    if manifest_path.exists() and not args.force_prepare:
        print(f"[skip prepare] {manifest_path}")
        return manifest_path

    cmd = [
        sys.executable,
        "-m",
        "benchmarks.tissue_seg.prepare_cell_labels",
        "--mapping_csv",
        str(mapping_path),
        "--output_dir",
        str(prepared_dir),
        "--color_tolerance",
        str(args.color_tolerance),
    ]
    if args.palette_json:
        cmd.extend(["--palette_json", args.palette_json])
    _run(cmd, output_root / "logs" / "prepare_cell_labels.log")
    return manifest_path


def _run_linear_probe(args: argparse.Namespace, output_root: Path, manifest_path: Path) -> None:
    if not args.split_csv:
        raise ValueError("--run_eval requires --split_csv")
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
        manifest_ids = {row["image_id"] for row in csv.DictReader(f)}
    split_counts = {"train": 0, "test": 0, "val": 0}
    with open(args.split_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["image_id"] in manifest_ids:
                split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    if split_counts.get("train", 0) == 0 or (split_counts.get("test", 0) + split_counts.get("val", 0)) == 0:
        print(f"[skip eval] insufficient completed split images: {split_counts}")
        return

    for head in args.heads:
        result_dir = output_root / "results" / f"linear_{head}"
        metrics_path = result_dir / "metrics.json"
        if metrics_path.exists() and not args.force_eval:
            print(f"[skip eval] {head}: {metrics_path}")
            continue

        cmd = [
            sys.executable,
            "-m",
            "benchmarks.tissue_seg.eval_linear_probe",
            "--prepared_manifest",
            str(manifest_path),
            "--output_dir",
            str(result_dir),
            "--head",
            head,
            "--split_csv",
            args.split_csv,
            "--max_iter",
            str(args.max_iter),
        ]
        if args.include_pixel_miou:
            cmd.append("--include_pixel_miou")
        if args.palette_json:
            cmd.extend(["--palette_json", args.palette_json])
        _run(cmd, output_root / "logs" / f"linear_{head}.log")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a tissue linear-probe dataset and optionally train/evaluate linear probes."
    )
    parser.add_argument("--selected_csv", required=True)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split_csv", default=None)
    parser.add_argument("--palette_json", default=None)
    parser.add_argument("--model_path", default=DEFAULT_XCELL_CHECKPOINT)
    parser.add_argument("--ctranspath_checkpoint", default="module/checkpoint/ctranspath.pth")
    parser.add_argument("--cellpose_model", default="/home/zyh/.cellpose/models/cpsam")
    parser.add_argument("--cuda_visible_devices", default="1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--max_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--inference_timeout_minutes", type=float, default=15.0)
    parser.add_argument("--resize_quality", type=int, default=90)
    parser.add_argument("--color_tolerance", type=float, default=35.0)
    parser.add_argument("--heads", nargs="+", choices=["reg", "proj", "raw"], default=["reg", "proj"])
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--include_pixel_miou", action="store_true")
    parser.add_argument("--run_eval", action="store_true")
    parser.add_argument("--force_inference", action="store_true")
    parser.add_argument("--force_prepare", action="store_true")
    parser.add_argument("--force_eval", action="store_true")
    parser.add_argument("--skip_inference_timeouts", action="store_true", default=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = _read_selected(args.selected_csv)
    if not rows:
        raise SystemExit("No rows in selected CSV.")

    _resize_images(rows, output_root, args.resize_quality)
    _run_inference(args, rows, output_root)
    _align_masks(rows, output_root)
    mapping_path = _write_prepare_mapping(rows, output_root)
    manifest_path = _prepare_labels(args, output_root, mapping_path)
    if args.run_eval:
        _run_linear_probe(args, output_root, manifest_path)

    print(f"[done] output_root={output_root}")
    print(f"[done] prepared_manifest={manifest_path}")


if __name__ == "__main__":
    main()
