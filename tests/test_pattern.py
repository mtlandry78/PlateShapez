import numpy as np
import pytest
from PIL import Image

from plateshapez.optim.pattern import PatternSpec, blend_pattern


class TestPatternSpec:
    def test_genome_length_default(self):
        spec = PatternSpec()
        assert spec.genome_length == 8 * 8 * 3

    def test_genome_length_custom(self):
        spec = PatternSpec(n_basis_x=4, n_basis_y=2, n_channels=1)
        assert spec.genome_length == 8

    def test_bounds_length_and_range(self):
        spec = PatternSpec(n_basis_x=4, n_basis_y=4, n_channels=3)
        bounds = spec.bounds()
        assert len(bounds) == spec.genome_length
        assert all(b == (-1.0, 1.0) for b in bounds)

    def test_random_genome_within_bounds(self):
        spec = PatternSpec(n_basis_x=4, n_basis_y=4, n_channels=3)
        rng = np.random.default_rng(0)
        genome = spec.random_genome(rng)
        assert genome.shape == (spec.genome_length,)
        assert np.all(genome >= -1.0)
        assert np.all(genome <= 1.0)

    def test_render_wrong_length_raises(self):
        spec = PatternSpec(n_basis_x=2, n_basis_y=2, n_channels=3)
        with pytest.raises(ValueError, match="Expected genome of length"):
            spec.render(np.zeros(3), (10, 10))

    @pytest.mark.parametrize("n_channels,expected_mode", [(1, "L"), (3, "RGB"), (4, "RGBA")])
    def test_render_produces_expected_mode_and_size(self, n_channels, expected_mode):
        spec = PatternSpec(n_basis_x=3, n_basis_y=3, n_channels=n_channels, grid_size=(16, 16))
        rng = np.random.default_rng(1)
        genome = spec.random_genome(rng)

        image = spec.render(genome, (40, 25))

        assert isinstance(image, Image.Image)
        assert image.mode == expected_mode
        assert image.size == (40, 25)

    def test_render_unsupported_channels_raises(self):
        spec = PatternSpec(n_basis_x=2, n_basis_y=2, n_channels=2)
        with pytest.raises(ValueError, match="Unsupported n_channels"):
            spec.render(np.zeros(spec.genome_length), (10, 10))

    def test_render_is_deterministic_for_same_genome(self):
        spec = PatternSpec(n_basis_x=4, n_basis_y=4, n_channels=3)
        genome = np.linspace(-1, 1, spec.genome_length)

        first = np.array(spec.render(genome, (20, 20)))
        second = np.array(spec.render(genome, (20, 20)))

        assert np.array_equal(first, second)


class TestBlendPattern:
    def test_region_bounds_respected(self):
        base = Image.new("RGB", (100, 100), color="white")
        pattern = Image.new("RGB", (64, 64), color="black")
        region = (20, 20, 40, 40)

        result = blend_pattern(base, pattern, region, strength=1.0)
        result_array = np.array(result)
        base_array = np.array(base)

        # Outside the region should be untouched.
        assert np.array_equal(base_array[0:10, 0:10], result_array[0:10, 0:10])
        assert np.array_equal(base_array[90:100, 90:100], result_array[90:100, 90:100])

    def test_strength_zero_leaves_base_unchanged(self):
        base = Image.new("RGB", (50, 50), color=(200, 100, 50))
        pattern = Image.new("RGB", (50, 50), color="black")
        region = (0, 0, 50, 50)

        result = blend_pattern(base, pattern, region, strength=0.0)

        assert np.array_equal(np.array(result), np.array(base))

    def test_strength_one_matches_pattern_in_region(self):
        base = Image.new("RGB", (50, 50), color=(200, 100, 50))
        pattern = Image.new("RGB", (50, 50), color=(10, 20, 30))
        region = (0, 0, 50, 50)

        result = blend_pattern(base, pattern, region, strength=1.0)

        assert np.array_equal(np.array(result), np.array(pattern))

    def test_strength_is_clamped(self):
        base = Image.new("RGB", (20, 20), color=(0, 0, 0))
        pattern = Image.new("RGB", (20, 20), color=(255, 255, 255))
        region = (0, 0, 20, 20)

        result = blend_pattern(base, pattern, region, strength=5.0)

        assert np.array_equal(np.array(result), np.array(pattern))

    @pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
    def test_preserves_base_mode(self, mode):
        from typing import Union

        color: Union[int, tuple[int, int, int], tuple[int, int, int, int]]
        if mode == "L":
            color = 128
        elif mode == "RGB":
            color = (128, 128, 128)
        else:
            color = (128, 128, 128, 255)
        base = Image.new(mode, (40, 40), color=color)
        pattern = Image.new("RGB", (40, 40), color=(0, 0, 0))
        region = (5, 5, 20, 20)

        result = blend_pattern(base, pattern, region, strength=0.5)

        assert result.mode == mode
        assert result.size == base.size
