from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PatternSpec:
    """Fixed-length cosine-basis genome for a renderable pattern.

    The genome is a flat vector of per-channel 2D cosine-basis amplitudes.
    `render()` evaluates the basis on a small internal grid (independent of
    the eventual output size) and resizes to fit -- this keeps the black-box
    search space small regardless of the target region's pixel size.
    """

    n_basis_x: int = 8
    n_basis_y: int = 8
    n_channels: int = 3
    grid_size: tuple[int, int] = (64, 64)  # (width, height), matches PIL's Image.size order

    @property
    def genome_length(self) -> int:
        return self.n_basis_x * self.n_basis_y * self.n_channels

    def bounds(self) -> list[tuple[float, float]]:
        return [(-1.0, 1.0)] * self.genome_length

    def random_genome(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=self.genome_length).astype(np.float64)

    def render(self, genome: np.ndarray, size: tuple[int, int]) -> Image.Image:
        """Render `genome` to an image resized to `size` = (width, height)."""
        genome_arr = np.asarray(genome, dtype=np.float64)
        if genome_arr.shape != (self.genome_length,):
            raise ValueError(
                f"Expected genome of length {self.genome_length}, got shape {genome_arr.shape}"
            )

        grid_w, grid_h = self.grid_size
        amplitudes = genome_arr.reshape(self.n_channels, self.n_basis_y, self.n_basis_x)

        x_idx = np.arange(grid_w)
        y_idx = np.arange(grid_h)
        i_idx = np.arange(self.n_basis_x)
        j_idx = np.arange(self.n_basis_y)

        basis_x = np.cos(np.pi * i_idx[:, None] * (x_idx[None, :] + 0.5) / grid_w)
        basis_y = np.cos(np.pi * j_idx[:, None] * (y_idx[None, :] + 0.5) / grid_h)

        norm = float(self.n_basis_x * self.n_basis_y)
        channels = np.empty((grid_h, grid_w, self.n_channels), dtype=np.float64)
        for c in range(self.n_channels):
            raw = basis_y.T @ amplitudes[c] @ basis_x  # (grid_h, grid_w)
            channels[:, :, c] = raw / norm

        pixels = np.clip((channels * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        if self.n_channels == 1:
            image = Image.fromarray(pixels[:, :, 0], mode="L")
        elif self.n_channels == 3:
            image = Image.fromarray(pixels, mode="RGB")
        elif self.n_channels == 4:
            image = Image.fromarray(pixels, mode="RGBA")
        else:
            raise ValueError(f"Unsupported n_channels: {self.n_channels} (expected 1, 3, or 4)")

        return image.resize(size, Image.Resampling.BICUBIC)


def blend_pattern(
    base: Image.Image, pattern: Image.Image, region: tuple[int, int, int, int], strength: float
) -> Image.Image:
    """Alpha-blend `pattern` into `base` within `region`=(x,y,w,h) at `strength` opacity.

    Preserves `base`'s original PIL mode via the same numpy round-trip
    convention the other perturbations use, so it composes generically
    with L/RGB/RGBA inputs.
    """
    x, y, w, h = region
    strength = max(0.0, min(1.0, strength))

    arr = np.array(base)
    region_slice = arr[y : y + h, x : x + w]

    pattern_resized = pattern.resize((w, h), Image.Resampling.BICUBIC)
    if pattern_resized.mode != base.mode:
        pattern_resized = pattern_resized.convert(base.mode)
    pattern_arr = np.array(pattern_resized)

    blended = (
        region_slice.astype(np.float64) * (1 - strength) + pattern_arr.astype(np.float64) * strength
    )
    arr[y : y + h, x : x + w] = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)
