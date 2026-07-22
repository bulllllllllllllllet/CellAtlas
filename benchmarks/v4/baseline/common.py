"""Shared immutable-I/O and metric helpers for baseline tools."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import re

import numpy as np
import pandas as pd
import torch

from benchmarks.v4.phase_6_mask_decoder.src.evaluation import boundary_f1, dice_from_counts


TIMESTAMP_PATTERN = re.compile(r"^\d{8}_\d{6}$")
REQUIRED_EPISODE_COLUMNS = (
    "occurrence_id", "occurrence_order", "episode_index", "split", "patch_id", "wsi_id",
    "patient_id", "target_class", "prompt_size", "positive_points_10x",
    "negative_points_10x", "positive_box_10x", "x_10x", "y_10x", "width_10x",
    "height_10x", "x_level0", "y_level0", "width_level0", "height_level0",
    "wsi_path", "gt_path", "source_region_ids",
)


def timestamp(value: str | None = None) -> str:
    result = value or datetime.now().strftime("%Y%m%d_%H%M%S")
    if not TIMESTAMP_PATTERN.fullmatch(result):
        raise ValueError("timestamp must use YYYYMMDD_HHMMSS")
    return result


def new_output_directory(root: Path, prefix: str, stamp: str) -> Path:
    output = root / f"{prefix}_{timestamp(stamp)}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    return output


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_json_array(value: Any, columns: int | None = None, dtype=np.float32) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        array = np.empty((0, columns or 0), dtype=dtype) if columns else np.empty(0, dtype=dtype)
    else:
        array = np.asarray(value, dtype=dtype)
    if columns is not None and (array.ndim != 2 or array.shape[1] != columns):
        raise ValueError(f"expected [N,{columns}] JSON array, got {array.shape}")
    return array


def validate_episode_manifest(frame: pd.DataFrame, expected_split: str | None = None) -> dict[str, Any]:
    missing = set(REQUIRED_EPISODE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"episode manifest misses columns {sorted(missing)}")
    if frame.empty:
        raise ValueError("episode manifest is empty")
    if frame["occurrence_id"].isna().any() or frame["occurrence_id"].duplicated().any():
        raise ValueError("occurrence_id must be non-null and unique")
    order = frame["occurrence_order"].to_numpy(dtype=np.int64)
    if not np.array_equal(order, np.arange(len(frame))):
        raise ValueError("occurrence_order must be contiguous and already sorted")
    splits = sorted(frame["split"].astype(str).unique())
    if len(splits) != 1 or (expected_split is not None and splits != [expected_split]):
        raise ValueError(f"manifest split mismatch: {splits}, expected={expected_split}")
    if frame[["patch_id", "wsi_id", "patient_id", "wsi_path", "gt_path"]].isna().any().any():
        raise ValueError("manifest has null identity/source fields")
    for row in frame.itertuples(index=False):
        positive = parse_json_array(row.positive_points_10x, 2)
        negative = parse_json_array(row.negative_points_10x, 2)
        width, height = int(row.width_10x), int(row.height_10x)
        if width <= 0 or height <= 0 or len(positive) == 0:
            raise ValueError(f"invalid geometry for occurrence {row.occurrence_id}")
        for points in (positive, negative):
            if not np.isfinite(points).all() or ((points[:, 0] < 0) | (points[:, 0] >= width) | (points[:, 1] < 0) | (points[:, 1] >= height)).any():
                raise ValueError(f"out-of-bounds prompt for occurrence {row.occurrence_id}")
        if str(row.prompt_size) != "point":
            box = np.asarray(json.loads(row.positive_box_10x), dtype=np.float32)
            if box.shape != (4,) or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                raise ValueError(f"invalid frozen box for occurrence {row.occurrence_id}")
        elif row.positive_box_10x not in (None, "", "null") and not pd.isna(row.positive_box_10x):
            box = np.asarray(json.loads(row.positive_box_10x), dtype=np.float32)
            if box.shape != (4,):
                raise ValueError(f"malformed optional point box for occurrence {row.occurrence_id}")
    return {
        "rows": len(frame), "split": splits[0], "unique_episode_indices": int(frame["episode_index"].nunique()),
        "unique_patches": int(frame["patch_id"].nunique()), "unique_wsi": int(frame["wsi_id"].nunique()),
        "unique_patients": int(frame["patient_id"].nunique()),
        "prompt_sizes": frame["prompt_size"].value_counts().sort_index().to_dict(),
        "target_classes": frame["target_class"].value_counts().sort_index().to_dict(),
    }


def binary_metric_row(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray, tolerance: int = 2) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if prediction.shape != truth.shape or truth.shape != valid.shape:
        raise ValueError("prediction/truth/valid shapes differ")
    tp = int((prediction & truth & valid).sum())
    fp = int((prediction & ~truth & valid).sum())
    fn = int((~prediction & truth & valid).sum())
    tn = int((~prediction & ~truth & valid).sum())
    boundary = boundary_f1(
        torch.from_numpy(prediction[None]), torch.from_numpy(truth[None]),
        torch.from_numpy(valid[None]), int(tolerance),
    )
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "positive": int((truth & valid).sum()),
        "valid": int(valid.sum()), "episode_dice": float(dice_from_counts(tp, fp, fn)),
        "boundary_f1": float(boundary["boundary_f1"][0]),
        "boundary_evaluable": bool(boundary["boundary_evaluable"][0]),
    }


def summarize_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    required = {"tp", "fp", "fn", "episode_dice", "boundary_f1", "status", "prompt_size", "target_class"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"metric rows miss {sorted(missing)}")
    tp, fp, fn = (int(frame[name].sum()) for name in ("tp", "fp", "fn"))
    completed = frame["status"] == "completed"
    result = {
        "episode_count": len(frame), "completed": int(completed.sum()),
        "abstained": int((frame["status"] == "abstained").sum()),
        "failed": int((frame["status"] == "failed").sum()),
        "coverage": float(completed.mean()), "pooled_pixel_dice": float(dice_from_counts(tp, fp, fn)),
        "macro_episode_dice": float(frame["episode_dice"].mean()),
        "boundary_f1_2px": float(frame["boundary_f1"].mean(skipna=True)),
        "tp": tp, "fp": fp, "fn": fn,
        "latency_ms_mean": float(frame["latency_ms"].mean()),
        "latency_ms_median": float(frame["latency_ms"].median()),
        "peak_memory_mb_max": float(frame["peak_memory_mb"].max()),
    }
    for group in ("prompt_size", "target_class"):
        result[f"by_{group}"] = {
            str(key): {
                "episodes": len(part), "coverage": float((part["status"] == "completed").mean()),
                "macro_episode_dice": float(part["episode_dice"].mean()),
                "pooled_pixel_dice": float(dice_from_counts(part["tp"].sum(), part["fp"].sum(), part["fn"].sum())),
                "boundary_f1_2px": float(part["boundary_f1"].mean(skipna=True)),
            }
            for key, part in frame.groupby(group, sort=True)
        }
    return result


def exact_occurrence_alignment(frames: Iterable[pd.DataFrame]) -> list[str]:
    frames = list(frames)
    if not frames:
        raise ValueError("no frames supplied")
    reference = frames[0].sort_values("occurrence_order")["occurrence_id"].astype(str).tolist()
    for index, frame in enumerate(frames[1:], start=1):
        candidate = frame.sort_values("occurrence_order")["occurrence_id"].astype(str).tolist()
        if candidate != reference:
            raise ValueError(f"occurrence alignment mismatch in frame {index}")
    return reference

