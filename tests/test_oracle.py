import json
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("alprg")

from alprg.engines.fake_engine import FakeEngine  # noqa: E402
from alprg.harness import MultiEngineHarness  # noqa: E402
from alprg.types import BoundingBox, EngineResult, OCRReading, PlateDetection  # noqa: E402

from plateshapez.optim.es_optimizer import EvolutionStrategyOptimizer  # noqa: E402
from plateshapez.optim.oracle import ALPRGOracle  # noqa: E402
from plateshapez.optim.pattern import PatternSpec  # noqa: E402

pytestmark = pytest.mark.requires_optimize


def _responder_for(text: str | None) -> Callable[[np.ndarray], EngineResult]:
    def responder(image: np.ndarray) -> EngineResult:
        if text is None:
            return EngineResult(engine_name="fake", detections=[])
        return EngineResult(
            engine_name="fake",
            detections=[PlateDetection(BoundingBox(0, 0, 10, 10), OCRReading(text, 0.95))],
        )

    return responder


def _build_oracle(
    responder_text: str | None,
    expected_text: str | None,
    log_path: Path | None = None,
) -> tuple[ALPRGOracle, MultiEngineHarness, PatternSpec]:
    background = Image.new("RGB", (200, 100), color="gray")
    overlay = Image.new("RGBA", (60, 30), color=(255, 255, 255, 255))
    region = (70, 35, 60, 30)
    spec = PatternSpec(n_basis_x=2, n_basis_y=2, n_channels=3)

    harness = MultiEngineHarness(engines=[FakeEngine(responder=_responder_for(responder_text))])
    harness.load()
    oracle = ALPRGOracle(
        harness=harness,
        background=background,
        overlay=overlay,
        region=region,
        pattern_spec=spec,
        expected_text=expected_text,
        background_name="bg.jpg",
        overlay_name="ov.png",
        log_path=log_path,
    )
    return oracle, harness, spec


class TestALPRGOracle:
    def test_correct_match_scores_as_class_c(self):
        oracle, harness, spec = _build_oracle("ABC123", expected_text="ABC123")
        try:
            response = oracle(np.zeros(spec.genome_length))
        finally:
            oracle.close()
            harness.close()

        assert response.plate_class == "C"
        assert response.score == 1.0

    def test_misread_scores_as_class_b(self):
        oracle, harness, spec = _build_oracle("ZZZ999", expected_text="ABC123")
        try:
            response = oracle(np.zeros(spec.genome_length))
        finally:
            oracle.close()
            harness.close()

        assert response.plate_class == "B"
        assert response.score == 0.4

    def test_no_detection_scores_as_class_a(self):
        oracle, harness, spec = _build_oracle(None, expected_text="ABC123")
        try:
            response = oracle(np.zeros(spec.genome_length))
        finally:
            oracle.close()
            harness.close()

        assert response.plate_class == "A"
        assert response.score == 0.0

    def test_score_method_matches_call_score(self):
        oracle, harness, spec = _build_oracle("ABC123", expected_text="ABC123")
        try:
            genome = np.zeros(spec.genome_length)
            assert oracle.score(genome) == oracle(genome).score
        finally:
            oracle.close()
            harness.close()

    def test_logs_each_query_to_jsonl(self, tmp_path):
        log_path = tmp_path / "query_log.jsonl"
        oracle, harness, spec = _build_oracle("ABC123", expected_text="ABC123", log_path=log_path)
        try:
            oracle(np.zeros(spec.genome_length))
            oracle(np.ones(spec.genome_length))
        finally:
            oracle.close()
            harness.close()

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        assert first["query"]["query_index"] == 0
        assert second["query"]["query_index"] == 1
        assert first["response"]["plate_class"] == "C"

    def test_no_log_path_skips_logging_without_error(self):
        oracle, harness, spec = _build_oracle("ABC123", expected_text="ABC123", log_path=None)
        try:
            oracle(np.zeros(spec.genome_length))
        finally:
            oracle.close()  # must not raise even though nothing was opened
            harness.close()

    def test_usable_as_optimizer_evaluate_callable(self):
        oracle, harness, spec = _build_oracle("ABC123", expected_text="ABC123")
        try:
            optimizer = EvolutionStrategyOptimizer(population_size=4, seed=0)
            result = optimizer.optimize(
                evaluate=oracle.score,
                genome_length=spec.genome_length,
                bounds=spec.bounds(),
                budget=8,
            )
        finally:
            oracle.close()
            harness.close()

        assert result.n_queries == 8
        # Every individual reads back as the correct plate -> all scores are 1.0.
        assert result.best_score == 1.0
