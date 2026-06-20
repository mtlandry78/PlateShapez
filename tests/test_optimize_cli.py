import subprocess
import tempfile
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from plateshapez.__main__ import app

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_assets():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        background_path = temp_path / "bg.jpg"
        overlay_path = temp_path / "overlay.png"

        Image.new("RGB", (200, 100), color="gray").save(background_path)
        Image.new("RGBA", (60, 30), color=(255, 255, 255, 255)).save(overlay_path)

        yield {
            "background": background_path,
            "overlay": overlay_path,
            "out": temp_path / "opt_run",
        }


class TestOptimizeCLI:
    @pytest.mark.requires_optimize
    def test_optimize_with_fake_engine(self, sample_assets):
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "plateshapez",
                "optimize",
                "--background",
                str(sample_assets["background"]),
                "--overlay",
                str(sample_assets["overlay"]),
                "--out",
                str(sample_assets["out"]),
                "--engines",
                "fake",
                "--budget",
                "5",
                "--expected-text",
                "ABC123",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(REPO_ROOT),
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert "Optimization complete" in result.stdout
        assert (sample_assets["out"] / "query_log.jsonl").exists()
        assert (sample_assets["out"] / "best_pattern.png").exists()
        assert (sample_assets["out"] / "best_genome.npy").exists()
        assert (sample_assets["out"] / "result.json").exists()

    def test_optimize_help(self):
        result = subprocess.run(
            ["uv", "run", "python", "-m", "plateshapez", "optimize", "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        assert "--background" in result.stdout


class TestOptimizeMissingDependency:
    """Simulates the alprg-not-installed path via monkeypatching, so this
    test is deterministic regardless of whether the 'optimize' extra
    happens to be installed in the current environment."""

    def test_missing_alprg_dependency_gives_friendly_error(self, sample_assets, monkeypatch):
        def _raise_import_error(*args: object, **kwargs: object) -> None:
            raise ImportError("No module named 'alprg'")

        monkeypatch.setattr("plateshapez.optim.runner.run_optimization", _raise_import_error)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "optimize",
                "--background",
                str(sample_assets["background"]),
                "--overlay",
                str(sample_assets["overlay"]),
                "--out",
                str(sample_assets["out"]),
                "--engines",
                "fake",
                "--budget",
                "5",
            ],
        )

        assert result.exit_code == 1
        assert "Missing dependency" in result.stdout
        assert "Traceback" not in result.stdout
