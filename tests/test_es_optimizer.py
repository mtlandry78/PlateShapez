import numpy as np

from plateshapez.optim.es_optimizer import EvolutionStrategyOptimizer


class _CallCountingEvaluator:
    """Scores genomes by distance to a fixed target, counting calls."""

    def __init__(self, target: np.ndarray) -> None:
        self.target = target
        self.n_calls = 0

    def __call__(self, genome: np.ndarray) -> float:
        self.n_calls += 1
        return float(np.sum((genome - self.target) ** 2))


class TestEvolutionStrategyOptimizer:
    def test_never_exceeds_budget(self):
        bounds = [(-1.0, 1.0)] * 5
        evaluator = _CallCountingEvaluator(target=np.zeros(5))
        optimizer = EvolutionStrategyOptimizer(population_size=8, seed=0)

        result = optimizer.optimize(evaluator, genome_length=5, bounds=bounds, budget=20)

        assert evaluator.n_calls <= 20
        assert result.n_queries == evaluator.n_calls

    def test_budget_smaller_than_population_size(self):
        bounds = [(-1.0, 1.0)] * 3
        evaluator = _CallCountingEvaluator(target=np.zeros(3))
        optimizer = EvolutionStrategyOptimizer(population_size=16, seed=0)

        result = optimizer.optimize(evaluator, genome_length=3, bounds=bounds, budget=5)

        assert evaluator.n_calls == 5
        assert result.n_queries == 5

    def test_deterministic_with_seed(self):
        bounds = [(-1.0, 1.0)] * 4
        target = np.array([0.2, -0.3, 0.5, 0.0])

        first = EvolutionStrategyOptimizer(population_size=8, seed=42).optimize(
            _CallCountingEvaluator(target), genome_length=4, bounds=bounds, budget=24
        )
        second = EvolutionStrategyOptimizer(population_size=8, seed=42).optimize(
            _CallCountingEvaluator(target), genome_length=4, bounds=bounds, budget=24
        )

        assert first.best_genome == second.best_genome
        assert first.best_score == second.best_score
        assert first.history == second.history

    def test_converges_toward_target(self):
        bounds = [(-1.0, 1.0)] * 6
        target = np.array([0.5, -0.5, 0.25, -0.25, 0.1, -0.1])
        evaluator = _CallCountingEvaluator(target)
        optimizer = EvolutionStrategyOptimizer(population_size=16, seed=1)

        result = optimizer.optimize(evaluator, genome_length=6, bounds=bounds, budget=200)

        # Score should improve substantially from a random first-generation guess.
        assert result.best_score < result.history[0]

    def test_respects_bounds(self):
        bounds = [(-0.1, 0.1)] * 4
        evaluator = _CallCountingEvaluator(target=np.full(4, 5.0))
        optimizer = EvolutionStrategyOptimizer(population_size=8, initial_sigma=2.0, seed=3)

        result = optimizer.optimize(evaluator, genome_length=4, bounds=bounds, budget=40)

        assert all(-0.1 - 1e-9 <= g <= 0.1 + 1e-9 for g in result.best_genome)
