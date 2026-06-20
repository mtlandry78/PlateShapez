import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from plateshapez.optim.base import OptimizationResult, PatternOptimizer  # noqa: E402
from plateshapez.optim.es_optimizer import EvolutionStrategyOptimizer  # noqa: E402
from plateshapez.optim.surrogate_optimizer import SurrogatePatternOptimizer  # noqa: E402

pytestmark = pytest.mark.requires_surrogate


class _CallCountingEvaluator:
    """Scores genomes by distance to a fixed target, counting calls."""

    def __init__(self, target: np.ndarray) -> None:
        self.target = target
        self.n_calls = 0

    def __call__(self, genome: np.ndarray) -> float:
        self.n_calls += 1
        return float(np.sum((genome - self.target) ** 2))


def _fast_optimizer(
    *,
    proposals_per_round: int = 4,
    exploration_sigma: float = 0.1,
    warm_start_logs: list[str | Path] | None = None,
    seed: int | None = 0,
) -> SurrogatePatternOptimizer:
    """A surrogate optimizer with cheap training settings for fast tests."""
    return SurrogatePatternOptimizer(
        bootstrap_fraction=0.4,
        proposals_per_round=proposals_per_round,
        hidden_sizes=(32, 32),
        epochs=60,
        proposal_steps=40,
        exploration_sigma=exploration_sigma,
        warm_start_logs=warm_start_logs,
        seed=seed,
    )


class TestSurrogatePatternOptimizer:
    def test_is_pattern_optimizer(self):
        optimizer = SurrogatePatternOptimizer()
        assert isinstance(optimizer, PatternOptimizer)

    def test_never_exceeds_budget(self):
        bounds = [(-1.0, 1.0)] * 5
        evaluator = _CallCountingEvaluator(target=np.zeros(5))
        optimizer = _fast_optimizer(proposals_per_round=8)

        result = optimizer.optimize(evaluator, genome_length=5, bounds=bounds, budget=20)

        assert evaluator.n_calls <= 20
        assert result.n_queries == evaluator.n_calls
        assert isinstance(result, OptimizationResult)

    def test_budget_smaller_than_proposal_batch(self):
        bounds = [(-1.0, 1.0)] * 3
        evaluator = _CallCountingEvaluator(target=np.zeros(3))
        optimizer = _fast_optimizer(proposals_per_round=8)

        result = optimizer.optimize(evaluator, genome_length=3, bounds=bounds, budget=5)

        assert evaluator.n_calls == 5
        assert result.n_queries == 5
        assert len(result.history) == 5

    def test_deterministic_with_seed(self):
        bounds = [(-1.0, 1.0)] * 4
        target = np.array([0.2, -0.3, 0.5, 0.0])

        first = _fast_optimizer(seed=42).optimize(
            _CallCountingEvaluator(target), genome_length=4, bounds=bounds, budget=24
        )
        second = _fast_optimizer(seed=42).optimize(
            _CallCountingEvaluator(target), genome_length=4, bounds=bounds, budget=24
        )

        assert first.best_genome == second.best_genome
        assert first.best_score == second.best_score
        assert first.history == second.history

    def test_minimizes_better_than_random_baseline(self):
        bounds = [(-1.0, 1.0)] * 6
        target = np.array([0.5, -0.5, 0.25, -0.25, 0.1, -0.1])
        budget = 80

        result = _fast_optimizer(seed=1).optimize(
            _CallCountingEvaluator(target), genome_length=6, bounds=bounds, budget=budget
        )

        # Random baseline: best of `budget` uniform draws in bounds.
        rng = np.random.default_rng(123)
        random_best = min(
            float(np.sum((rng.uniform(-1.0, 1.0, size=6) - target) ** 2)) for _ in range(budget)
        )

        assert result.best_score < random_best
        # The surrogate should also improve on its own bootstrap phase.
        assert result.best_score < result.history[0]
        # And land close to the optimum (score 0 at the target).
        assert result.best_score < 0.1

    def test_respects_bounds(self):
        bounds = [(-0.1, 0.1)] * 4
        evaluator = _CallCountingEvaluator(target=np.full(4, 5.0))
        optimizer = _fast_optimizer(exploration_sigma=1.0, seed=3)

        result = optimizer.optimize(evaluator, genome_length=4, bounds=bounds, budget=24)

        assert all(-0.1 - 1e-6 <= gene <= 0.1 + 1e-6 for gene in result.best_genome)

    def test_warm_start_from_query_log(self, tmp_path):
        genome_length = 4
        target = np.zeros(genome_length)
        log_path = tmp_path / "query_log.jsonl"
        rng = np.random.default_rng(7)
        with log_path.open("w") as handle:
            for index in range(12):
                genome = rng.uniform(-1.0, 1.0, size=genome_length)
                score = float(np.sum((genome - target) ** 2))
                record = {
                    "query": {"genome": genome.tolist(), "query_index": index},
                    "response": {"score": score, "plate_class": "C", "detail": {}},
                }
                handle.write(json.dumps(record) + "\n")

        bounds = [(-1.0, 1.0)] * genome_length
        evaluator = _CallCountingEvaluator(target)
        optimizer = _fast_optimizer(warm_start_logs=[log_path], seed=5)

        result = optimizer.optimize(
            evaluator, genome_length=genome_length, bounds=bounds, budget=16
        )

        # Warm-start records must not count against the live query budget.
        assert evaluator.n_calls <= 16
        assert result.n_queries == evaluator.n_calls

    def test_drop_in_for_es_optimizer(self):
        """Same call signature and result shape as the ES optimizer."""
        bounds = [(-1.0, 1.0)] * 4
        target = np.array([0.1, 0.2, -0.3, 0.4])

        es_result = EvolutionStrategyOptimizer(seed=0).optimize(
            _CallCountingEvaluator(target), genome_length=4, bounds=bounds, budget=24
        )
        surrogate_result = _fast_optimizer(seed=0).optimize(
            _CallCountingEvaluator(target), genome_length=4, bounds=bounds, budget=24
        )

        for result in (es_result, surrogate_result):
            assert isinstance(result, OptimizationResult)
            assert len(result.best_genome) == 4
            assert isinstance(result.best_score, float)
            assert len(result.history) == result.n_queries
