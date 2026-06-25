from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoreTile:
    tile_index: int
    core_x0: int
    core_y0: int
    core_x1: int
    core_y1: int
    read_x0: int
    read_y0: int
    read_x1: int
    read_y1: int

    def owns(self, x: float, y: float) -> bool:
        return self.core_x0 <= x < self.core_x1 and self.core_y0 <= y < self.core_y1


def generate_core_tiles(width: int, height: int, tile_size: int, halo: int) -> list[CoreTile]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if tile_size <= 0 or halo < 0:
        raise ValueError("tile_size must be positive and halo non-negative")
    tiles: list[CoreTile] = []
    index = 0
    for y0 in range(0, height, tile_size):
        for x0 in range(0, width, tile_size):
            x1 = min(x0 + tile_size, width)
            y1 = min(y0 + tile_size, height)
            tiles.append(
                CoreTile(
                    tile_index=index,
                    core_x0=x0,
                    core_y0=y0,
                    core_x1=x1,
                    core_y1=y1,
                    read_x0=max(0, x0 - halo),
                    read_y0=max(0, y0 - halo),
                    read_x1=min(width, x1 + halo),
                    read_y1=min(height, y1 + halo),
                )
            )
            index += 1
    return tiles


def map_point_to_gt(
    x: float, y: float, original_size: tuple[int, int], gt_size: tuple[int, int]
) -> tuple[float, float]:
    original_width, original_height = original_size
    gt_width, gt_height = gt_size
    if min(original_width, original_height, gt_width, gt_height) <= 0:
        raise ValueError("image dimensions must be positive")
    return x * gt_width / original_width, y * gt_height / original_height

