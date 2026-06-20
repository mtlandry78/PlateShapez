from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from plateshapez.optim.base import OptimizationResult, PatternOptimizer

# torch is an optional, heavy dependency: it is imported lazily inside the
# methods that need it (mirroring how runner.py imports `alprg`), so this module
# imports with zero torch installed and the `es` optimizer / generate paths keep
# working. Anything that actually trains the surrogate calls `_require_torch()`.


def _require_torch() -> Any:
    """Import torch on demand, raising an actionable error if it is missing."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise RuntimeError(
            "SurrogatePatternOptimizer requires PyTorch. Install the optional "
            "extra with: uv sync --extra surrogate"
        ) from exc
    return torch


class SurrogatePatternOptimizer(PatternOptimizer):
    """Differentiable-surrogate (Bayesian-opt-lite) black-box optimizer (Phase B).

    Instead of searching the genome space directly, this fits a small PyTorch
    MLP ``f_theta: R^genome_length -> R`` to the observed ``(genome, score)``
    pairs, then descends the *surrogate's* gradient (the real oracle is not
    differentiable) to propose lower-scoring genomes. Each round:

    1. fit the surrogate on every observation so far,
    2. propose new genomes by gradient descent on the surrogate from the
       best-known genome plus exploration noise,
    3. evaluate the proposals on the *real* ``evaluate`` (counted against budget).

    It is a pure drop-in for :class:`PatternOptimizer`: lower scores are better,
    it never exceeds ``budget`` calls to ``evaluate``, clips every proposal to
    ``bounds``, and is deterministic given ``seed``. Surrogate training stays on
    CPU by default (this runs on machines without CUDA).
    """

    def __init__(
        self,
        bootstrap_fraction: float = 0.25,
        proposals_per_round: int = 8,
        hidden_sizes: tuple[int, ...] = (64, 64),
        epochs: int = 300,
        learning_rate: float = 1e-2,
        weight_decay: float = 1e-4,
        proposal_steps: int = 100,
        proposal_lr: float = 0.05,
        exploration_sigma: float = 0.1,
        warm_start_logs: list[str | Path] | None = None,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.bootstrap_fraction = bootstrap_fraction
        self.proposals_per_round = proposals_per_round
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.proposal_steps = proposal_steps
        self.proposal_lr = proposal_lr
        self.exploration_sigma = exploration_sigma
        self.warm_start_logs = warm_start_logs
        self.device = device
        self.seed = seed

    def optimize(
        self,
        evaluate: Callable[[np.ndarray], float],
        genome_length: int,
        bounds: list[tuple[float, float]],
        budget: int,
    ) -> OptimizationResult:
        torch = _require_torch()
        # Deterministic given a seed: seed both numpy (sampling/noise) and torch
        # (weight init + Adam) once, up front, so a fixed RNG-consumption order
        # reproduces exactly.
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(0 if self.seed is None else self.seed)

        lower = np.array([bound[0] for bound in bounds], dtype=np.float64)
        upper = np.array([bound[1] for bound in bounds], dtype=np.float64)

        # Pre-existing oracle logs augment the surrogate's training set without
        # counting against this run's budget (they were already paid for).
        warm_genomes, warm_scores = self._load_warm_start(genome_length)

        observed_genomes: list[np.ndarray] = list(warm_genomes)
        observed_scores: list[float] = list(warm_scores)

        best_genome = (
            np.clip((lower + upper) / 2.0, lower, upper)
            if not observed_genomes
            else observed_genomes[int(np.argmin(observed_scores))].copy()
        )
        best_score = float("inf")
        history: list[float] = []
        n_queries = 0

        def run_query(genome: np.ndarray) -> None:
            nonlocal best_genome, best_score, n_queries
            clipped = np.clip(genome, lower, upper)
            score = float(evaluate(clipped))
            observed_genomes.append(clipped)
            observed_scores.append(score)
            history.append(score)
            n_queries += 1
            if score < best_score:
                best_score = score
                best_genome = clipped.copy()

        # 1. Bootstrap: spend a fraction of the budget on random genomes to seed
        #    a training set (at least one, never more than the whole budget).
        n_bootstrap = max(1, round(budget * self.bootstrap_fraction))
        n_bootstrap = min(n_bootstrap, budget)
        for _ in range(n_bootstrap):
            run_query(rng.uniform(lower, upper))

        # 2-5. Surrogate-guided rounds until the budget is exhausted.
        while n_queries < budget:
            model = self._fit_surrogate(
                torch, np.asarray(observed_genomes), np.asarray(observed_scores), genome_length
            )
            remaining = budget - n_queries
            n_proposals = min(self.proposals_per_round, remaining)
            proposals = self._propose(torch, model, best_genome, lower, upper, rng, n_proposals)
            for proposal in proposals:
                run_query(proposal)

        return OptimizationResult(
            best_genome=best_genome.tolist(),
            best_score=best_score,
            n_queries=n_queries,
            history=history,
        )

    def _load_warm_start(self, genome_length: int) -> tuple[list[np.ndarray], list[float]]:
        """Read `(genome, score)` pairs from prior query_log.jsonl files.

        Only entries whose genome matches `genome_length` are kept, so logs from
        an incompatible PatternSpec are silently ignored rather than corrupting
        the training set.
        """
        if not self.warm_start_logs:
            return [], []

        genomes: list[np.ndarray] = []
        scores: list[float] = []
        for log_path in self.warm_start_logs:
            path = Path(log_path)
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                genome = record.get("query", {}).get("genome")
                score = record.get("response", {}).get("score")
                if genome is None or score is None or len(genome) != genome_length:
                    continue
                genomes.append(np.asarray(genome, dtype=np.float64))
                scores.append(float(score))
        return genomes, scores

    def _fit_surrogate(
        self,
        torch: Any,
        genomes: np.ndarray,
        scores: np.ndarray,
        genome_length: int,
    ) -> Any:
        """Train a fresh MLP to regress observed scores from genomes (MSE)."""
        layers: list[Any] = []
        in_features = genome_length
        for hidden in self.hidden_sizes:
            layers.append(torch.nn.Linear(in_features, hidden))
            layers.append(torch.nn.ReLU())
            in_features = hidden
        layers.append(torch.nn.Linear(in_features, 1))
        model = torch.nn.Sequential(*layers).to(self.device)

        inputs = torch.tensor(genomes, dtype=torch.float32, device=self.device)
        targets = torch.tensor(scores, dtype=torch.float32, device=self.device).reshape(-1, 1)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = torch.nn.MSELoss()
        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
        model.eval()
        return model

    def _propose(
        self,
        torch: Any,
        model: Any,
        best_genome: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        rng: np.random.Generator,
        n_proposals: int,
    ) -> list[np.ndarray]:
        """Gradient-descend the surrogate from noisy copies of the best genome.

        The first proposal starts at the best genome itself (greedy); the rest
        add exploration noise so the search does not collapse to a single point.
        Each proposal is projected back into `bounds` after every step.
        """
        lower_t = torch.tensor(lower, dtype=torch.float32, device=self.device)
        upper_t = torch.tensor(upper, dtype=torch.float32, device=self.device)

        proposals: list[np.ndarray] = []
        for index in range(n_proposals):
            start = best_genome.copy()
            if index > 0:
                start = start + rng.normal(0.0, self.exploration_sigma, size=best_genome.shape)
            start = np.clip(start, lower, upper)

            candidate = torch.tensor(
                start, dtype=torch.float32, device=self.device, requires_grad=True
            )
            inner = torch.optim.Adam([candidate], lr=self.proposal_lr)
            for _ in range(self.proposal_steps):
                inner.zero_grad()
                predicted = model(candidate.reshape(1, -1))
                predicted.backward()
                inner.step()
                with torch.no_grad():
                    candidate.clamp_(lower_t, upper_t)

            proposals.append(candidate.detach().cpu().numpy().astype(np.float64))
        return proposals
