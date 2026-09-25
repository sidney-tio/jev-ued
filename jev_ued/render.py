"""Renders a binary drawing as a maze image with plain numpy (no MiniGrid).

Filled pixels (1) become grey wall tiles inside a one-tile grey border; empty
pixels are black floor, with thin grey grid lines, as in MiniGrid.
"""
import numpy as np

WALL = np.array([100, 100, 100], dtype=np.uint8)
FLOOR = np.array([0, 0, 0], dtype=np.uint8)
GRID_LINE = np.array([100, 100, 100], dtype=np.uint8)


def render_maze(canvas, tile_size=32):
  """Returns an (H, W, 3) uint8 RGB image of canvas ([row][col] of 0/1)."""
  walls = np.pad(np.asarray(canvas, dtype=bool), 1, constant_values=True)
  image = np.where(walls[..., None], WALL, FLOOR).astype(np.uint8)
  image = image.repeat(tile_size, axis=0).repeat(tile_size, axis=1)

  # Grid lines along each tile's top and left edges
  width = max(1, round(tile_size * 0.031))
  for offset in range(width):
    image[offset::tile_size, :] = GRID_LINE
    image[:, offset::tile_size] = GRID_LINE
  return image
