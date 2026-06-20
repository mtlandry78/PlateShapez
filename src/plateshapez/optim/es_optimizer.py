from __future__ import annotations

from typing import Callable

import numpy as np

from plateshapez.optim.base import OptimizationResult, PatternOptimizer


class EvolutionStrategyOptimizer(PatternOptimizer):
    """Pure-numpy (mu/mu_w, lambda)-ES-lite black-box optimizer.

    Each generation samples a population around a running mean, clips to
    bounds, scores every individual via `evaluate`, keeps the elite
    fraction, re-estimates the mean from the elites, and decays the search
    radius (sigma). Lower scores are better (a pure minimizer); never
    exceeds `budget` calls to `evaluate`.
    """

    def __init__(
        self,
        population_size: int = 16,
        elite_fraction: float = 0.25,
        initial_sigma: float = 0.5,
        sigma_decay: float = 0.98,
        seed: int | None = None,
    ) -> None:
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.initial_sigma = initial_sigma
        self.sigma_decay = sigma_decay
        self.seed = seed

    def optimize(
        self,
        evaluate: Callable[[np.ndarray], float],
        genome_length: int,
        bounds: list[tuple[float, float]],
        budget: int,
    ) -> OptimizationResult:
        rng = np.random.default_rng(self.seed)
        lower = np.array([b[0] for b in bounds], dtype=np.float64)
        upper = np.array([b[1] for b in bounds], dtype=np.float64)

        mean = rng.uniform(lower, upper)
        sigma = self.initial_sigma
        n_elite = max(1, round(self.population_size * self.elite_fraction))

        best_genome = mean.copy()
        best_score = float("inf")
        history: list[float] = []
        n_queries = 0

        while n_queries < budget:
            batch_size = min(self.population_size, budget - n_queries)
            population = rng.normal(loc=mean, scale=sigma, size=(batch_size, genome_length))
            population = np.clip(population, lower, upper)

            scores = np.empty(batch_size, dtype=np.float64)
            for i in range(batch_size):
                scores[i] = evaluate(population[i])
                n_queries += 1
                if scores[i] < best_score:
                    best_score = float(scores[i])
                    best_genome = population[i].copy()
                history.append(float(scores[i]))

            elite_count = min(n_elite, batch_size)
            elite_idx = np.argsort(scores)[:elite_count]
            mean = population[elite_idx].mean(axis=0)
            sigma *= self.sigma_decay

        return OptimizationResult(
            best_genome=best_genome.tolist(),
            best_score=best_score,
            n_queries=n_queries,
            history=history,
        )
