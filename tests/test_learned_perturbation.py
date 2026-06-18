from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from plateshapez.perturbations.base import PERTURBATION_REGISTRY
from plateshapez.perturbations.learned import LearnedPerturbation


@pytest.fixture
def zero_genome_path(tmp_path: Path) -> Path:
    """A flat (all-zero) genome renders to a uniform mid-gray pattern."""
    genome = np.zeros(8 * 8 * 3)
    path = tmp_path / "genome.npy"
    np.save(path, genome)
    return path


class TestLearnedPerturbationRegistry:
    def test_registered_under_learned(self):
        assert PERTURBATION_REGISTRY["learned"] is LearnedPerturbation
        assert LearnedPerturbation.name == "learned"

    def test_missing_genome_path_raises(self):
        with pytest.raises(ValueError, match="genome_path"):
            LearnedPerturbation()


class TestLearnedPerturbationApply:
    def test_region_bounds_respected(self, zero_genome_path):
        img = Image.new("RGB", (100, 100), color="white")
        region = (20, 20, 40, 40)

        perturbation = LearnedPerturbation(genome_path=str(zero_genome_path), strength=1.0)
        result = perturbation.apply(img, region)
        result_array = np.array(result)
        original_array = np.array(img)

        assert isinstance(result, Image.Image)
        assert result.size == img.size
        # Areas outside the region should be unchanged.
        assert np.array_equal(original_array[0:10, 0:10], result_array[0:10, 0:10])
        assert np.array_equal(original_array[90:100, 90:100], result_array[90:100, 90:100])

    def test_strength_affects_blend_amount(self, zero_genome_path):
        img = Image.new("RGB", (50, 50), color=(255, 255, 255))
        region = (0, 0, 50, 50)
        original_array = np.array(img).astype(float)

        low = LearnedPerturbation(genome_path=str(zero_genome_path), strength=0.1)
        low_array = np.array(low.apply(img, region)).astype(float)

        high = LearnedPerturbation(genome_path=str(zero_genome_path), strength=0.9)
        high_array = np.array(high.apply(img, region)).astype(float)

        low_diff = np.mean(np.abs(original_array - low_array))
        high_diff = np.mean(np.abs(original_array - high_array))
        assert high_diff > low_diff

    def test_serialization_replaces_genome_path_with_source(self, zero_genome_path):
        perturbation = LearnedPerturbation(genome_path=str(zero_genome_path), strength=0.5)
        serialized = perturbation.serialize()

        assert serialized["type"] == "learned"
        assert "genome_path" not in serialized["params"]
        assert serialized["params"]["genome_source"] == str(zero_genome_path)
        assert serialized["params"]["strength"] == 0.5


class TestLearnedPerturbationChannelCompatibility:
    @pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
    def test_works_with_different_channel_counts(self, zero_genome_path, mode):
        from typing import Union

        color: Union[int, tuple[int, int, int], tuple[int, int, int, int]]
        if mode == "L":
            color = 128
        elif mode == "RGB":
            color = (128, 128, 128)
        else:
            color = (128, 128, 128, 255)
        img = Image.new(mode, (50, 50), color=color)
        region = (10, 10, 30, 30)

        perturbation = LearnedPerturbation(genome_path=str(zero_genome_path))
        result = perturbation.apply(img, region)

        assert isinstance(result, Image.Image)
        assert result.mode == mode
        assert result.size == img.size
