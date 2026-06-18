from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, TextIO

import numpy as np


@dataclass(frozen=True)
class OracleQuery:
    genome: list[float]
    query_index: int


@dataclass(frozen=True)
class OracleResponse:
    score: float
    plate_class: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OracleRecord:
    query: OracleQuery
    response: OracleResponse

    def to_json_line(self) -> str:
        return json.dumps({"query": asdict(self.query), "response": asdict(self.response)})


class QueryLogger:
    """Append-mode JSONL writer for OracleRecord entries.

    Mirrors utils/io.py's save_metadata mkdir(parents=True, exist_ok=True)
    convention. This is the literal training-data schema a future
    differentiable-surrogate optimizer (Phase B) would read back via
    `glob("optimization_runs/*/query_log.jsonl")` -- no migration needed.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO | None = self.path.open("a")

    def log(self, record: OracleRecord) -> None:
        if self._file is None:
            raise RuntimeError("QueryLogger is closed.")
        self._file.write(record.to_json_line() + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "QueryLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class OptimizationResult:
    best_genome: list[float]
    best_score: float
    n_queries: int
    history: list[float]


class PatternOptimizer(ABC):
    """Black-box optimizer seam: repeatedly query `evaluate` (at most
    `budget` times) to minimize its return value over genome vectors.

    The single abstract method is intentionally narrow so a future
    differentiable-surrogate optimizer (Phase B) is a pure drop-in
    replacement wherever a `PatternOptimizer` is expected (e.g.
    `runner.run_optimization`'s `optimizer` parameter).
    """

    @abstractmethod
    def optimize(
        self,
        evaluate: Callable[[np.ndarray], float],
        genome_length: int,
        bounds: list[tuple[float, float]],
        budget: int,
    ) -> OptimizationResult:
        """Minimize `evaluate` over genome vectors of length `genome_length`."""
