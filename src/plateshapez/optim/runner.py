from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from plateshapez.optim.base import OptimizationResult, PatternOptimizer
from plateshapez.optim.es_optimizer import EvolutionStrategyOptimizer
from plateshapez.optim.oracle import ALPRGOracle
from plateshapez.optim.pattern import PatternSpec
from plateshapez.utils.io import save_image
from plateshapez.utils.overlay import calculate_center_position, ensure_rgb, ensure_rgba


def run_optimization(
    background_path: str | Path,
    overlay_path: str | Path,
    out_dir: str | Path,
    expected_text: str | None = None,
    budget: int = 500,
    strength: float = 0.6,
    pattern_spec: PatternSpec | None = None,
    optimizer: PatternOptimizer | None = None,
    engines: list[str] | None = None,
    seed: int | None = None,
) -> OptimizationResult:
    """Optimize a near-invisible adversarial pattern against a live
    multi-engine ALPR oracle, using a black-box (gradient-free) optimizer.

    This is the one place (besides oracle.py's TYPE_CHECKING-only import)
    that imports `alprg` at runtime -- callers without the `optimize` extra
    installed will get an ImportError here, not at module import time.
    """
    from alprg import MultiEngineHarness
    from alprg.engines import ENGINE_REGISTRY

    pattern_spec = pattern_spec or PatternSpec()
    optimizer = optimizer or EvolutionStrategyOptimizer(seed=seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    background = ensure_rgb(Image.open(background_path))
    overlay = ensure_rgba(Image.open(overlay_path))
    bx, by = calculate_center_position(background, overlay)
    ow, oh = overlay.size
    region = (bx, by, ow, oh)

    engine_instances = (
        [ENGINE_REGISTRY[name]() for name in engines] if engines is not None else None
    )

    with MultiEngineHarness(engines=engine_instances, verbose=False) as harness:
        oracle = ALPRGOracle(
            harness=harness,
            background=background,
            overlay=overlay,
            region=region,
            pattern_spec=pattern_spec,
            expected_text=expected_text,
            background_name=Path(background_path).name,
            overlay_name=Path(overlay_path).name,
            strength=strength,
            log_path=out_dir / "query_log.jsonl",
        )
        try:
            result = optimizer.optimize(
                evaluate=oracle.score,
                genome_length=pattern_spec.genome_length,
                bounds=pattern_spec.bounds(),
                budget=budget,
            )
        finally:
            oracle.close()

    best_genome = np.array(result.best_genome, dtype=np.float64)
    best_pattern = pattern_spec.render(best_genome, (ow, oh))
    save_image(best_pattern, out_dir / "best_pattern.png")
    np.save(out_dir / "best_genome.npy", best_genome)

    result_payload = {
        "best_score": result.best_score,
        "n_queries": result.n_queries,
        "background": Path(background_path).name,
        "overlay": Path(overlay_path).name,
        "region": list(region),
        "strength": strength,
        "pattern_spec": {
            "n_basis_x": pattern_spec.n_basis_x,
            "n_basis_y": pattern_spec.n_basis_y,
            "n_channels": pattern_spec.n_channels,
            "grid_size": list(pattern_spec.grid_size),
        },
    }
    (out_dir / "result.json").write_text(json.dumps(result_payload, indent=2))

    return result
