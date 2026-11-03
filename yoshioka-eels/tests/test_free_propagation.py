"""Independent analytical checks and regression coverage for Benchmark 1."""

import unittest

import numpy as np
from scipy.constants import c, elementary_charge, hbar, m_e

from benchmarks.free_electron import BenchmarkConfig, TOLERANCES, run_benchmark
from benchmarks.reference_solutions import (
    constant_wave,
    gaussian_wave,
    periodic_mode,
    radial_second_moment,
    relative_wave_error,
)
from yoshioka import FreePropagator, TransverseGrid, electron_wavenumber, grid_norm_squared


class FreePropagationTests(unittest.TestCase):
    def setUp(self):
        self.grid = TransverseGrid(10e-9, 10e-9, 256, 256)
        self.width = 0.2e-9
        self.energy = 60e3 * elementary_charge
        self.k = electron_wavenumber(self.energy)
        self.zd = self.k * self.width**2
        self.propagator = FreePropagator(self.grid, self.k)

    def assertWaveClose(self, numerical, exact, grid=None, tolerance=1e-10):
        self.assertLess(relative_wave_error(numerical, exact, grid or self.grid), tolerance)

    def test_relativistic_energy_and_si_units(self):
        # Independent energy-momentum invariant using the returned momentum.
        momentum = hbar * self.k
        recovered_energy = np.hypot(momentum * c, m_e * c**2) - m_e * c**2
        self.assertLess(abs(recovered_energy / self.energy - 1), 2e-15)
        self.assertGreater(self.k, 1.29e12)
        self.assertLess(self.k, 1.30e12)

    def test_constant_plane_wave(self):
        initial = constant_wave(self.grid)
        self.assertAlmostEqual(float(grid_norm_squared(initial, self.grid)), 1, places=14)
        for s in (0, 0.25, 1, 2):
            with self.subTest(s=s):
                propagated = self.propagator.propagate(initial, s * self.zd)
                self.assertWaveClose(propagated, initial)
                self.assertLess(abs(float(grid_norm_squared(propagated, self.grid)) - 1), 1e-10)

    def test_fourier_mode_complex_phase(self):
        initial = periodic_mode(self.grid, 3, -5, self.k)
        self.assertAlmostEqual(float(grid_norm_squared(initial, self.grid)), 1, places=14)
        for s in (0, 0.25, 1, 2):
            with self.subTest(s=s):
                expected = periodic_mode(self.grid, 3, -5, self.k, s * self.zd)
                self.assertWaveClose(self.propagator.propagate(initial, s * self.zd), expected)

    def test_rectangular_and_odd_grids_axis_order(self):
        for nx, ny in ((96, 80), (95, 79)):
            grid = TransverseGrid(8.4e-9, 11.2e-9, nx, ny)
            initial = periodic_mode(grid, 7, -11, self.k)
            z = 1.7 * self.zd
            expected = periodic_mode(grid, 7, -11, self.k, z)
            with self.subTest(nx=nx, ny=ny):
                self.assertWaveClose(FreePropagator(grid, self.k).propagate(initial, z), expected, grid)

    def test_gaussian_wave_and_radial_moment(self):
        initial = gaussian_wave(self.grid, self.width, self.k)
        initial_norm = float(grid_norm_squared(initial, self.grid))
        self.assertAlmostEqual(initial_norm, 1, places=14)
        for s in (0, 0.25, 1, 2):
            with self.subTest(s=s):
                numerical = self.propagator.propagate(initial, s * self.zd)
                exact = gaussian_wave(self.grid, self.width, self.k, s * self.zd)
                self.assertWaveClose(numerical, exact, tolerance=1e-8)
                expected_moment = self.width**2 * (1 + s**2)
                self.assertLess(abs(radial_second_moment(numerical, self.grid) / expected_moment - 1), 1e-8)
                self.assertLess(abs(float(grid_norm_squared(numerical, self.grid)) / initial_norm - 1), 1e-10)

    def test_composition_and_reversibility_on_all_represented_frequencies(self):
        # A seeded broadband wave exercises high frequencies, unlike a smooth Gaussian.
        grid = TransverseGrid(10e-9, 8e-9, 48, 32)
        rng = np.random.default_rng(137)
        initial = rng.standard_normal(grid.shape) + 1j * rng.standard_normal(grid.shape)
        initial /= np.sqrt(grid_norm_squared(initial, grid))
        propagator = FreePropagator(grid, self.k)
        for z in (0.25 * self.zd, 2 * self.zd, -self.zd):
            with self.subTest(z=z):
                direct = propagator.propagate(initial, z)
                stepped = initial.copy()
                for _ in range(100):
                    stepped = propagator.propagate(stepped, z / 100)
                self.assertWaveClose(stepped, direct, grid)
                self.assertWaveClose(propagator.propagate(direct, -z), initial, grid)
                self.assertLess(abs(float(grid_norm_squared(stepped, grid)) - 1), 1e-10)

    def test_uncoupled_channels_stay_empty_and_conserve_total_norm(self):
        initial = np.zeros((3, *self.grid.shape), dtype=np.complex128)
        initial[0] = gaussian_wave(self.grid, self.width, self.k)
        for s in (0, 0.25, 1, 2):
            with self.subTest(s=s):
                propagated = self.propagator.propagate(initial, s * self.zd)
                self.assertTrue(np.all(propagated[1:] == 0))
                populations = grid_norm_squared(propagated, self.grid)
                self.assertEqual(float(np.sum(populations[1:])), 0)
                self.assertLess(abs(float(np.sum(populations)) / float(np.sum(grid_norm_squared(initial, self.grid))) - 1), 1e-10)
                self.assertWaveClose(propagated[0], self.propagator.propagate(initial[0], s * self.zd))

    def test_error_metric_keeps_global_phase(self):
        initial = constant_wave(self.grid)
        self.assertAlmostEqual(relative_wave_error(initial * np.exp(0.3j), initial, self.grid), 2 * np.sin(0.15), places=14)

    def test_dtype_and_input_preservation(self):
        initial = gaussian_wave(self.grid, self.width, self.k)
        saved = initial.copy()
        output = self.propagator.propagate(initial, self.zd)
        np.testing.assert_array_equal(initial, saved)
        self.assertEqual(output.dtype, np.dtype(np.complex128))
        zero_distance = self.propagator.propagate(initial, 0)
        self.assertFalse(np.shares_memory(zero_distance, initial))
        real_input = np.ones(self.grid.shape, dtype=np.float32)
        self.assertEqual(self.propagator.propagate(real_input, self.zd).dtype, np.dtype(np.complex128))

    def test_invalid_physical_parameters_and_wave_shape(self):
        for value in (0, -1, np.nan, np.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    electron_wavenumber(value)
                with self.assertRaises(ValueError):
                    FreePropagator(self.grid, value)
                with self.assertRaises(ValueError):
                    TransverseGrid(value, 10e-9, 16, 16)
        for nx in (1, 2.5, True):
            with self.assertRaises(ValueError):
                TransverseGrid(10e-9, 10e-9, nx, 16)
        with self.assertRaises(ValueError):
            self.propagator.propagate(np.ones((32, 64)), self.zd)
        with self.assertRaises(ValueError):
            self.propagator.propagate(np.ones(self.grid.shape), np.nan)


class BenchmarkIntegrationTests(unittest.TestCase):
    def test_all_default_acceptance_checks_and_grid_refinement(self):
        results = run_benchmark(BenchmarkConfig())
        self.assertEqual(results["status"], "PASS", results["summary"])
        for key, passed in results["checks"].items():
            with self.subTest(check=key):
                self.assertTrue(passed)
        self.assertEqual([entry["nx"] for entry in results["grid_refinement"]], [64, 128, 256])
        for entry in results["grid_refinement"]:
            maximum = max(case["relative_wave_error"] for case in entry["cases"])
            if entry["nx"] == 64:
                self.assertGreater(maximum, TOLERANCES["gaussian_relative_wave_error"])
            else:
                self.assertLess(maximum, TOLERANCES["gaussian_relative_wave_error"])
        # No monotonicity requirement at the roundoff floor.

    def test_underresolved_grid_reports_failure_without_relaxing_tolerances(self):
        results = run_benchmark(BenchmarkConfig(nx=64, ny=64, refinement_grids=(64,)))
        self.assertEqual(results["status"], "FAIL")
        self.assertFalse(results["checks"]["gaussian_relative_wave_error"])
        self.assertFalse(results["checks"]["gaussian_spectral_tail_negligible"])
        self.assertEqual(results["tolerances"], TOLERANCES)


if __name__ == "__main__":
    unittest.main()
