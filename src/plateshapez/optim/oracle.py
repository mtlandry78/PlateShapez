from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from plateshapez.optim.base import OracleQuery, OracleRecord, OracleResponse, QueryLogger
from plateshapez.optim.pattern import PatternSpec, blend_pattern
from plateshapez.utils.overlay import ensure_rgb, ensure_rgba

if TYPE_CHECKING:
    from alprg import MultiEngineHarness

# Lower is better for the adversary: not-detected beats misread beats correct.
_PLATE_CLASS_SCORES: dict[str, float] = {"A": 0.0, "B": 0.4, "C": 1.0}


class ALPRGOracle:
    """Wraps a live MultiEngineHarness as a black-box scoring function for
    PatternOptimizer.optimize.

    `alprg` is only imported under TYPE_CHECKING here -- this module never
    hard-imports it at runtime, keeping the optional dependency boundary as
    narrow as possible.
    """

    def __init__(
        self,
        harness: "MultiEngineHarness",
        background: Image.Image,
        overlay: Image.Image,
        region: tuple[int, int, int, int],
        pattern_spec: PatternSpec,
        expected_text: str | None,
        background_name: str,
        overlay_name: str,
        strength: float = 0.6,
        log_path: str | Path | None = None,
    ) -> None:
        self.harness = harness
        self.background = ensure_rgb(background)
        self.overlay = ensure_rgba(overlay)
        self.region = region
        self.pattern_spec = pattern_spec
        self.expected_text = expected_text
        self.background_name = background_name
        self.overlay_name = overlay_name
        self.strength = strength
        self._logger = QueryLogger(log_path) if log_path is not None else None
        self._query_index = 0

    def __call__(self, genome: np.ndarray) -> OracleResponse:
        x, y, w, h = self.region
        pattern_img = self.pattern_spec.render(np.asarray(genome), (w, h))

        composite = self.background.copy()
        composite.paste(self.overlay, (x, y), self.overlay)
        composite = blend_pattern(composite, pattern_img, self.region, self.strength)

        bgr_array = np.array(ensure_rgb(composite))[:, :, ::-1].copy()
        result = self.harness.classify_array(bgr_array, expected_text=self.expected_text)

        response = OracleResponse(
            score=_PLATE_CLASS_SCORES.get(result.plate_class.value, 1.0),
            plate_class=result.plate_class.value,
            detail=dict(result.detail),
        )

        if self._logger is not None:
            query = OracleQuery(genome=np.asarray(genome).tolist(), query_index=self._query_index)
            self._logger.log(OracleRecord(query=query, response=response))
        self._query_index += 1
        return response

    def score(self, genome: np.ndarray) -> float:
        """Float-returning adapter for PatternOptimizer.optimize's `evaluate`."""
        return self(genome).score

    def close(self) -> None:
        if self._logger is not None:
            self._logger.close()
