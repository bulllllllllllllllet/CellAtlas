from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def sliding_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Return deterministic starts that cover an axis exactly through its end."""
    length = int(length); tile_size = int(tile_size); stride = int(stride)
    if length < tile_size:
        raise ValueError(f"axis length {length} is smaller than tile size {tile_size}")
    if stride < 1 or stride > tile_size:
        raise ValueError("stride must be in [1, tile_size]")
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def build_tile_rows(
    wsi_id: str,
    wsi_path: str,
    width_level0: int,
    height_level0: int,
    level0_downsample: int,
    tile_size: int,
    stride: int,
    split: str,
) -> list[dict]:
    """Build a complete Cartesian sliding grid in the model's 10x space."""
    downsample = int(level0_downsample)
    if downsample < 1:
        raise ValueError("level0_downsample must be a positive integer")
    width_10x = int(width_level0) // downsample
    height_10x = int(height_level0) // downsample
    xs = sliding_starts(width_10x, tile_size, stride)
    ys = sliding_starts(height_10x, tile_size, stride)
    rows = []
    for source_index, (y_10x, x_10x) in enumerate((y, x) for y in ys for x in xs):
        rows.append({
            "source_index": source_index,
            "patch_id": f"{wsi_id}__x{x_10x}_y{y_10x}",
            "wsi_id": str(wsi_id),
            "split": str(split),
            "x_10x": int(x_10x), "y_10x": int(y_10x),
            "width_10x": int(tile_size), "height_10x": int(tile_size),
            "x_level0": int(x_10x * downsample), "y_level0": int(y_10x * downsample),
            "width_level0": int(tile_size * downsample),
            "height_level0": int(tile_size * downsample),
            "level0_downsample": downsample,
            "wsi_width_level0": int(width_level0), "wsi_height_level0": int(height_level0),
            "output_width_10x": width_10x, "output_height_10x": height_10x,
            "wsi_path": str(wsi_path),
        })
    return rows


def validate_tile_rows(rows: list[dict]) -> tuple[int, int, int]:
    if not rows:
        raise ValueError("tile index is empty")
    required = {
        "source_index", "patch_id", "wsi_id", "x_10x", "y_10x", "width_10x",
        "height_10x", "x_level0", "y_level0", "width_level0", "height_level0",
        "level0_downsample", "output_width_10x", "output_height_10x", "wsi_path",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"tile index misses columns: {sorted(missing)}")
    source = np.asarray([int(row["source_index"]) for row in rows])
    if not np.array_equal(source, np.arange(len(rows))):
        raise ValueError("source_index must be contiguous and sorted from zero")
    singleton_fields = (
        "wsi_id", "wsi_path", "split", "level0_downsample",
        "output_width_10x", "output_height_10x",
    )
    for field in singleton_fields:
        if len({str(row[field]) for row in rows}) != 1:
            raise ValueError(f"tile index contains multiple values for {field}")
    width = int(rows[0]["output_width_10x"]); height = int(rows[0]["output_height_10x"])
    downsample = int(rows[0]["level0_downsample"])
    for row in rows:
        if (
            int(row["x_level0"]) != int(row["x_10x"]) * downsample
            or int(row["y_level0"]) != int(row["y_10x"]) * downsample
            or int(row["width_level0"]) != int(row["width_10x"]) * downsample
            or int(row["height_level0"]) != int(row["height_10x"]) * downsample
        ):
            raise ValueError(f"tile coordinate/downsample mismatch: {row['patch_id']}")
    x_intervals = sorted({(int(row["x_10x"]), int(row["x_10x"]) + int(row["width_10x"])) for row in rows})
    y_intervals = sorted({(int(row["y_10x"]), int(row["y_10x"]) + int(row["height_10x"])) for row in rows})
    for intervals, length, name in ((x_intervals, width, "x"), (y_intervals, height, "y")):
        if intervals[0][0] != 0 or intervals[-1][1] != length:
            raise ValueError(f"{name}-axis tiles do not reach both slide boundaries")
        if any(right < next_left for (_, right), (next_left, _) in zip(intervals, intervals[1:])):
            raise ValueError(f"{name}-axis tile grid contains an uncovered gap")
    expected = len(x_intervals) * len(y_intervals)
    pairs = {(int(row["x_10x"]), int(row["y_10x"])) for row in rows}
    if len(rows) != expected or len(pairs) != expected:
        raise ValueError("tile index is not a complete, duplicate-free Cartesian grid")
    return width, height, downsample


def blend_window(height: int, width: int, floor: float = 1e-3) -> np.ndarray:
    """Positive separable Hann window; the floor keeps slide borders covered."""
    if height < 2 or width < 2 or not 0.0 < floor < 1.0:
        raise ValueError("invalid blend window dimensions or floor")
    wy = np.maximum(np.hanning(height), floor)
    wx = np.maximum(np.hanning(width), floor)
    return np.outer(wy, wx).astype(np.float32)


@dataclass
class OverlapAccumulator:
    root: Path
    height: int
    width: int
    threshold: float

    def __post_init__(self) -> None:
        if not 0.0 < float(self.threshold) < 1.0:
            raise ValueError("threshold must lie strictly between zero and one")
        self.root = Path(self.root)
        self.sum_path = self.root / "probability_weighted_sum_float32.dat"
        self.weight_path = self.root / "blend_weight_sum_float32.dat"
        self.probability_path = self.root / "probability_float32.dat"
        self.mask_path = self.root / "mask_uint8.dat"
        for path in (self.sum_path, self.weight_path, self.probability_path, self.mask_path):
            if path.exists():
                raise FileExistsError(path)
        shape = (int(self.height), int(self.width))
        self.weighted_sum = np.memmap(self.sum_path, mode="w+", dtype=np.float32, shape=shape)
        self.weight_sum = np.memmap(self.weight_path, mode="w+", dtype=np.float32, shape=shape)
        self.weighted_sum[:] = 0; self.weight_sum[:] = 0

    def add(self, probability: np.ndarray, x: int, y: int, window: np.ndarray) -> None:
        probability = np.asarray(probability, dtype=np.float32)
        if probability.shape != window.shape or not np.isfinite(probability).all():
            raise ValueError("tile probability/window mismatch or non-finite probability")
        h, w = probability.shape; x = int(x); y = int(y)
        if x < 0 or y < 0 or x + w > self.width or y + h > self.height:
            raise ValueError("tile output lies outside the whole-slide canvas")
        self.weighted_sum[y:y + h, x:x + w] += probability * window
        self.weight_sum[y:y + h, x:x + w] += window

    def finalize(self, chunk_rows: int = 1024) -> tuple[np.memmap, np.memmap]:
        self.weighted_sum.flush(); self.weight_sum.flush()
        probability = np.memmap(
            self.probability_path, mode="w+", dtype=np.float32, shape=(self.height, self.width)
        )
        mask = np.memmap(self.mask_path, mode="w+", dtype=np.uint8, shape=(self.height, self.width))
        for start in range(0, self.height, int(chunk_rows)):
            stop = min(start + int(chunk_rows), self.height)
            weights = self.weight_sum[start:stop]
            if np.any(weights <= 0) or not np.isfinite(weights).all():
                raise RuntimeError(f"uncovered or non-finite pixels in rows {start}:{stop}")
            values = self.weighted_sum[start:stop] / weights
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise RuntimeError(f"invalid fused probability in rows {start}:{stop}")
            probability[start:stop] = values
            mask[start:stop] = (values >= float(self.threshold)).astype(np.uint8) * 255
        probability.flush(); mask.flush()
        return probability, mask
