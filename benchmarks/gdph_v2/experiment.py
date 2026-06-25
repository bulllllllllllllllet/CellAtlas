from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


EXPERIMENT_NAME = "cellatlas_gdph_benchmark_v2"
DEFAULT_OUTPUT_ROOT = Path("/nfs-medical3/zyh") / EXPERIMENT_NAME
OUTPUT_DIRS = (
    "config",
    "manifests",
    "logs",
    "cells",
    "patches",
    "masks",
    "overlays",
    "cell_classification",
    "region_retrieval",
    "dense_evaluation",
)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repo_root: str | Path) -> dict[str, str | bool]:
    root = Path(repo_root)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unknown", "dirty": True}
    return {"revision": revision, "dirty": dirty}


def initialize_experiment(
    output_root: str | Path,
    mapping_csv: str | Path,
    palette_json: str | Path,
    xcell_checkpoint: str | Path,
    ctranspath_checkpoint: str | Path,
    cellpose_checkpoint: str | Path,
    repo_root: str | Path,
) -> dict:
    output_root = Path(output_root)
    for relative in OUTPUT_DIRS:
        (output_root / relative).mkdir(parents=True, exist_ok=True)

    inputs = {
        "dataset_mapping": Path(mapping_csv),
        "tissue_palette": Path(palette_json),
    }
    copied_inputs: dict[str, str] = {}
    for name, source in inputs.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output_root / "manifests" / source.name
        shutil.copy2(source, destination)
        copied_inputs[name] = str(destination)

    checkpoints = {
        "xcellformer": Path(xcell_checkpoint),
        "ctranspath": Path(ctranspath_checkpoint),
        "cellpose": Path(cellpose_checkpoint),
    }
    model_records: dict[str, dict[str, str | int]] = {}
    for name, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        model_records[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    config = {
        "experiment_name": EXPERIMENT_NAME,
        "created_at": datetime.now().astimezone().isoformat(),
        "output_root": str(output_root),
        "dataset": {
            "name": "GDPH tissue segmentation",
            "external_to_xcellformer_training": True,
            "mapping_csv": copied_inputs["dataset_mapping"],
            "palette_json": copied_inputs["tissue_palette"],
        },
        "models": model_records,
        "inference": {
            "image_scale": "original_level_0",
            "tile_size": 4096,
            "halo": 256,
            "cellpose_diameter": 18,
            "flow_threshold": 0.4,
            "cellprob_threshold": 0.0,
            "stain_normalization": "none",
            "xcell_max_cells_per_batch": 255,
            "xcell_token_order": "cellpose_label_ascending_reference",
            "tile_cache_schema_version": 2,
            "keep_rule": "cell centroid inside non-overlapping core tile",
        },
        "labeling": {
            "method": "polygon_majority",
            "purity_denominator": "all_covered_gt_pixels_including_ignore",
            "valid_purity": 0.7,
            "boundary_purity": [0.5, 0.7],
            "ignore_below_purity": 0.5,
        },
        "evaluation": {
            "pilot_images": 5,
            "main_images": 20,
            "main_folds": 5,
            "linear_probe_max_train_cells_per_slide": 25000,
            "linear_probe_test_cells": "all_valid_cells",
            "heads": ["raw", "reg", "proj"],
            "patch_size_original": 1024,
            "patch_stride_original": 1024,
            "patch_white_threshold": 240,
            "patch_blank_ratio": 0.995,
            "region_query_sizes_original": [512, 1024, 2048],
            "region_query_min_gt_purity": 0.8,
            "region_query_min_gt_valid_fraction": 0.8,
            "region_retrieval_exclusion_buffer_original": 256,
            "region_retrieval_query_similarity_quantile": 0.1,
        },
        "code": git_revision(repo_root),
    }

    config_path = output_root / "config" / "experiment.json"
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    with open(output_root / "config" / "environment.json", "w", encoding="utf-8") as file:
        json.dump(environment, file, indent=2, ensure_ascii=False)

    return config
