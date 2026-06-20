from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from plateshapez.optim.pattern import PatternSpec, blend_pattern
from plateshapez.perturbations.base import Perturbation, register


@register
class LearnedPerturbation(Perturbation):
    """Applies a previously-optimized adversarial pattern (see `advplate optimize`).

    Loads a saved genome (.npy) plus the PatternSpec it was rendered with,
    renders it to the target region's size, and blends it in. Cheap and
    synchronous -- never runs optimization inline, and never imports
    `alprg`, so it stays testable with zero optional dependencies.
    """

    name = "learned"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        genome_path = self.params.get("genome_path")
        if not genome_path:
            raise ValueError("LearnedPerturbation requires a 'genome_path' parameter.")
        self._genome_path = Path(genome_path)
        self._genome: np.ndarray = np.load(self._genome_path)
        self._spec = PatternSpec(
            n_basis_x=int(self.params.get("n_basis_x", 8)),
            n_basis_y=int(self.params.get("n_basis_y", 8)),
            n_channels=int(self.params.get("n_channels", 3)),
            grid_size=tuple(self.params.get("grid_size", (64, 64))),
        )
        self._strength = float(self.params.get("strength", 0.6))

    def apply(self, img: Image.Image, region: tuple[int, int, int, int]) -> Image.Image:
        x, y, w, h = region
        pattern_img = self._spec.render(self._genome, (w, h))
        return blend_pattern(img, pattern_img, region, self._strength)

    def serialize(self) -> dict[str, Any]:
        params = dict(self.params)
        params["genome_source"] = str(self._genome_path)
        params.pop("genome_path", None)
        return {"type": self.name, "params": params}
