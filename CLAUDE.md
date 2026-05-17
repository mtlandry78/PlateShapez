# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

PlateShapez is a Python research tool for generating adversarially perturbed license plate datasets. It composites license plate overlays onto vehicle background images, applies configurable image perturbations, and writes structured outputs (PNG images + JSON metadata) for training adversarial robustness models.

## Commands

```bash
# One-time setup
uv venv && source .venv/bin/activate
uv sync --group dev    # installs all deps including dev tools

# Run the CLI
uv run advplate generate
uv run advplate generate --config my_config.yaml --seed 42 --dry-run
uv run advplate list       # show available perturbations
uv run advplate info --as yaml
uv run advplate demo       # run end-to-end demo with synthetic images

# Tests
uv run pytest
uv run pytest tests/test_pipeline.py         # single file
uv run pytest --cov=plateshapez              # with coverage

# Code quality (all three must pass before committing)
uv run dev format   # ruff format .
uv run dev lint     # ruff check . --fix
uv run dev type     # mypy .
uv run dev check    # runs all three in sequence (same as CI)

# Pre-commit hooks
uv run dev hooks install   # install pre-commit and pre-push hooks
uv run dev pre-commit      # run hooks on all files manually

# Cleanup generated files
uv run dev cleanup         # remove demo outputs and datasets
uv run dev cleanup all     # also removes .venv and build artifacts
```

CI runs `./scripts/check.sh`, which is identical to `uv run dev check`.

## Architecture

### Package layout (`src/plateshapez/`)

The source lives under `src/` with `mypy_path = "src"` and `explicit_package_bases = true` in `pyproject.toml`.

```
src/plateshapez/
├── __main__.py          # Typer CLI (advplate entry point)
├── config.py            # Config loading and merging
├── pipeline.py          # DatasetGenerator — main orchestrator
├── dev.py               # Dev utility commands (dev entry point)
├── perturbations/
│   ├── base.py          # Perturbation base class + PERTURBATION_REGISTRY + @register
│   ├── noise.py         # NoisePerturbation
│   ├── shapes.py        # ShapesPerturbation
│   ├── texture.py       # TexturePerturbation
│   └── warp.py          # WarpPerturbation
└── utils/
    ├── io.py            # iter_backgrounds, iter_overlays, save_image, save_metadata
    └── overlay.py       # calculate_center_position, ensure_rgb, ensure_rgba
```

### Config hierarchy (`config.py`)

`load_config()` merges three layers in order of increasing priority:
1. **DEFAULTS** (baked into `config.py`)
2. **File** — YAML or JSON, if `--config PATH` is provided
3. **CLI overrides** — `--n_variants`, `--seed`, `--verbose`, `--debug`

The `_deep_merge()` helper merges nested dicts without clobbering unrelated keys, and ignores `None` CLI values so unset flags don't overwrite file config.

### Perturbation registry pattern (`perturbations/base.py`)

New perturbations are registered via the `@register` class decorator:

```python
from plateshapez.perturbations.base import Perturbation, register

@register
class MyPerturbation(Perturbation):
    name = "my_perturbation"   # key used in config YAML

    def apply(self, img: Image.Image, region: tuple[int, int, int, int]) -> Image.Image:
        # region = (x, y, w, h) of the overlay on the background
        ...
        return img
```

`PERTURBATION_REGISTRY` is a module-level dict populated at import time. `perturbations/__init__.py` imports all submodules so every perturbation is registered before `pipeline.py` looks anything up. Duplicate `name` values raise `ValueError` at import time.

All perturbations support a `scope` param (`"region"` default, or `"global"`) that controls whether the effect applies only to the plate region or the full image.

### Pipeline flow (`pipeline.py`)

`DatasetGenerator.run(n_variants)` does the following for every `(background, overlay)` pair × `n_variants`:
1. Opens background (JPG → RGB) and overlay (PNG → RGBA)
2. Calculates center position for the overlay using `calculate_center_position`
3. Pastes the overlay onto a copy of the background
4. Iterates `self.perturbations`, instantiating each class from the registry and calling `pert.apply(img, (x, y, w, h))`
5. Saves the composite image to `out_dir/images/<bg_stem>_<ov_stem>_<NNN>.png`
6. Saves JSON metadata to `out_dir/labels/<same_name>.json` (if `save_metadata=True`)

Seeding (`random.seed` + `np.random.seed`) is applied once at the start of `run()`.

### CLI (`__main__.py`)

Built with [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/). `app = typer.Typer(no_args_is_help=True)`. On any unhandled exception the CLI prints the relevant command help panel before exiting with code 1 — this is intentional UX per the project spec.

## Code Conventions

- **Python version**: ≥3.10, targeting 3.12/3.13 for type checking.
- **Type annotations**: Use PEP 604 unions (`str | None`, not `Optional[str]`) and builtin generics (`list[str]`, `dict[str, int]`, not `List`, `Dict`).
- **Naming**: Descriptive names — `image_height`, `num_shapes`, `overlay_path`. No single-letter variables.
- **Line length**: 100 characters (`ruff` enforced).
- **Imports**: All at file top; no mid-function imports except inside `if TYPE_CHECKING` blocks.
- **Formatting**: `ruff format` is canonical (double quotes, space indent).
- **Mypy**: Strict-ish — `check_untyped_defs`, `disallow_incomplete_defs`, `no_implicit_optional`, `warn_unused_ignores`.
