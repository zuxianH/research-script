"""Elastic analytical regressions and independent-reference adapter checks."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.constants import c, elementary_charge, hbar, m_e

from benchmarks.elastic_observables import (
    angular_grid_mrad, angular_masks, compare_observables,
    diffraction_probabilities, intensity_observables,
)
from benchmarks.reference_solutions import constant_wave, periodic_mode, relative_wave_error
from yoshioka import (
    ElasticPropagator, FreePropagator, TransverseGrid, electron_speed,
    electron_wavenumber, grid_norm_squared, interaction_coefficient,
)
from yoshioka.elastic_propagation import antialias_aperture


class ElasticAnalyticalTests(unittest.TestCase):
    def setUp(self):
        self.grid = TransverseGrid(5.43053e-10, 3.83996e-10, 32, 24)
        self.energy = 60000 * elementary_charge
        self.initial = constant_wave(self.grid)

    def test_uniform_potential_phase_sign_units_and_integrated_slices(self):
        # Derive v independently from p/(gamma*m), not from the solver helper.
        gamma = 1 + self.energy / (m_e * c**2)
        momentum = np.sqrt((self.energy / c)**2 + 2 * m_e * self.energy)
        velocity = momentum / (gamma * m_e)
        thicknesses = np.array([0.31, 0.73, 1.17]) * 1e-10
        for voltage in (0, 13.7, -13.7):
            for filtered in (False, True):
                with self.subTest(voltage=voltage, filtered=filtered):
                    solver = ElasticPropagator(self.grid, self.energy, antialias_aperture(self.grid) if filtered else None)
                    potentials = voltage * thicknesses[:, None, None] * np.ones((3, *self.grid.shape))
                    result = solver.propagate(self.initial, potentials, thicknesses, repetitions=7)
                    expected = self.initial * np.exp(1j * elementary_charge * voltage * np.sum(thicknesses) * 7 / (hbar * velocity))
                    self.assertLess(relative_wave_error(result.wave, expected, self.grid), 1e-12)
                    self.assertLess(np.max(np.abs(result.norm_history - 1)), 1e-12)

    def test_zero_potential_reduces_to_benchmark1_for_nonconstant_wave(self):
        rng = np.random.default_rng(331)
        initial = rng.standard_normal(self.grid.shape) + 1j * rng.standard_normal(self.grid.shape)
        initial /= np.sqrt(grid_norm_squared(initial, self.grid))
        thicknesses = np.array([0.31, 0.73, 1.17]) * 1e-10
        zero = np.zeros((3, *self.grid.shape))
        result = ElasticPropagator(self.grid, self.energy).propagate(initial, zero, thicknesses)
        expected = FreePropagator(self.grid, electron_wavenumber(self.energy)).propagate(initial, float(np.sum(thicknesses)))
        self.assertLess(relative_wave_error(result.wave, expected, self.grid), 1e-10)

    def test_filtered_zero_potential_preserves_incident_plane_wave(self):
        result = ElasticPropagator(self.grid, self.energy, antialias_aperture(self.grid)).propagate(
            self.initial, np.zeros((8, *self.grid.shape)), np.full(8, 0.48e-10), repetitions=13,
        )
        self.assertLess(relative_wave_error(result.wave, self.initial, self.grid), 1e-12)

    def test_real_potential_unitarity_and_no_channel_generation(self):
        rng = np.random.default_rng(515)
        projected = rng.uniform(0, 100e-10, (7, *self.grid.shape))
        thicknesses = np.linspace(0.2e-10, 0.8e-10, 7)
        channels = np.zeros((3, *self.grid.shape), dtype=np.complex128)
        channels[0] = self.initial
        saved = channels.copy()
        for filtered in (False, True):
            solver = ElasticPropagator(self.grid, self.energy, antialias_aperture(self.grid) if filtered else None)
            result = solver.propagate(channels, projected, thicknesses, repetitions=4)
            self.assertTrue(np.all(result.wave[1:] == 0))
            self.assertTrue(np.all(result.norm_history[:, 1:] == 0))
            self.assertEqual(result.wave.dtype, np.dtype(np.complex128))
            if not filtered:
                self.assertLess(np.max(np.abs(np.sum(result.norm_history, axis=-1) - 1)), 1e-12)
        np.testing.assert_array_equal(channels, saved)

    def test_relativistic_velocity_and_coefficient(self):
        self.assertGreater(electron_speed(self.energy), 0.44 * c)
        self.assertLess(electron_speed(self.energy), 0.45 * c)
        self.assertAlmostEqual(interaction_coefficient(self.energy) * hbar * electron_speed(self.energy) / elementary_charge, 1)

    def test_parseval_and_no_independent_exit_normalization(self):
        wave = periodic_mode(self.grid, 2, -3, electron_wavenumber(self.energy)) * 0.7
        diffraction = diffraction_probabilities(wave, self.grid)
        self.assertAlmostEqual(float(diffraction.sum()), 0.49, places=13)
        self.assertAlmostEqual(float(diffraction[-3, 2]), 0.49, places=13)
        self.assertAlmostEqual(float(diffraction.sum()), float(grid_norm_squared(wave, self.grid)), places=13)

    def test_angular_units_axes_and_fixed_masks(self):
        wavelength = 2 * np.pi / electron_wavenumber(self.energy)
        tx, ty = angular_grid_mrad(self.grid, wavelength)
        self.assertAlmostEqual(tx[0, 1], wavelength / self.grid.lx_m * 1e3)
        self.assertAlmostEqual(ty[1, 0], wavelength / self.grid.ly_m * 1e3)
        masks = angular_masks(self.grid, wavelength)
        total = sum(mask.astype(int) for name, mask in masks.items() if name.startswith("annulus"))
        np.testing.assert_array_equal(total, masks["region_up_to_50_mrad"].astype(int))
        intensity = np.zeros(self.grid.shape)
        intensity[0, 0], intensity[-1, 1] = 0.4, 0.6
        values = intensity_observables(intensity, self.grid, wavelength, [{"h": 1, "k": -1}])
        self.assertEqual(values["transmitted_0_0"], 0.4)
        self.assertEqual(values["beam_1_-1"], 0.6)
        self.assertEqual(values["annulus_10_30_mrad"], 0.6)

    def test_weak_reference_uses_absolute_errors_and_strict_fixed_tolerances(self):
        compared = compare_observables({"strong": 0.202, "zero": 1e-10}, {"strong": 0.2, "zero": 0.0})
        self.assertIsNone(compared["observables"]["zero"]["relative_error"])
        self.assertTrue(compared["observables"]["zero"]["passed"])
        self.assertFalse(compare_observables({"strong": 0.21}, {"strong": 0.2})["passed"])

    def test_reject_invalid_energy_integrated_potential_and_metadata(self):
        for energy in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                ElasticPropagator(self.grid, energy)
        solver = ElasticPropagator(self.grid, self.energy)
        with self.assertRaises(ValueError):
            solver.propagate(self.initial, np.zeros((2, *self.grid.shape)), np.array([1e-10]))
        with self.assertRaises(ValueError):
            solver.propagate(self.initial, np.zeros((2, *self.grid.shape)), np.array([-1e-10, 1e-10]))
        with self.assertRaises(ValueError):
            solver.transmission(np.ones(self.grid.shape, dtype=complex))
        with self.assertRaises(ValueError):
            ElasticPropagator(self.grid, self.energy, np.ones(self.grid.shape) * 2)


class ConvergenceReportingTests(unittest.TestCase):
    def _cases(self, scale):
        from benchmarks.elastic_si110 import ElasticConfig
        config = ElasticConfig(repeats=(13,), extra_refinement=False)
        cases = []
        for n in config.grids:
            for slices in config.slices_per_repeat:
                value = scale(n, slices)
                cases.append({
                    "key": f"n{n}_s{slices}_r13", "thickness_nm": 4.991948,
                    "observables_ours": {"transmitted_0_0": value},
                    "observables_abtem": {"transmitted_0_0": value},
                })
        return config, cases

    def test_agreement_on_every_input_does_not_imply_convergence(self):
        from benchmarks.elastic_si110 import calculate_convergence
        config, cases = self._cases(lambda n, slices: 0.2 + 20 / n + 0.5 / slices)
        convergence = calculate_convergence(cases, config)
        self.assertFalse(convergence["passed"])
        self.assertTrue(convergence["evaluated"])

    def test_finest_pair_determines_convergence_without_hiding_coarse_failure(self):
        from benchmarks.elastic_si110 import calculate_convergence
        config, cases = self._cases(lambda n, slices: 0.2 + ({64: 0.1, 128: 0.0002, 256: 0}[n]) + ({8: 0.1, 16: 0.0002, 32: 0}[slices]))
        convergence = calculate_convergence(cases, config)
        self.assertTrue(convergence["passed"])
        self.assertTrue(any(not item["passed"] for item in convergence["grid"] if not item["finest_requested_pair"]))


@unittest.skipUnless(importlib.util.find_spec("abtem") and importlib.util.find_spec("ase"), "install the elastic-reference dependencies to run abTEM integration tests")
class AbTEMReferenceTests(unittest.TestCase):
    def test_benchmark_writes_shared_inputs_and_reports_unevaluated_convergence(self):
        from benchmarks.elastic_si110 import ElasticConfig, run_benchmark
        config = ElasticConfig(grids=(32,), slices_per_repeat=(8,), repeats=(1,), initial_grid=32, initial_slices=8, extra_refinement=False)
        with tempfile.TemporaryDirectory(prefix="yoshioka-elastic-test-") as directory:
            output = Path(directory)
            results = run_benchmark(config, output)
            self.assertEqual(results["status"], "FAIL")
            self.assertFalse(results["convergence"]["evaluated"])
            self.assertTrue(results["checks"]["implementation_agreement_all_requested_configurations"])
            self.assertTrue((output / "results.json").is_file())
            with np.load(output / "shared_inputs/n32_s8.npz") as saved:
                np.testing.assert_array_equal(saved["slice_unit_indices_r1"], np.arange(8))
                self.assertAlmostEqual(float(saved["slice_thickness_m_r1"].sum()), 3.83996e-10)
                self.assertEqual(saved["potential_v_angstrom_sxy"].dtype, np.dtype(np.float64))
                self.assertIn("native_antialias_aperture_yx", saved.files)

    def test_structure_and_original_file_unchanged(self):
        from benchmarks.abtem_reference import load_structure
        path = Path(__file__).resolve().parents[1] / "Si_110_temp300.lmp"
        original = path.read_bytes()
        atoms, verification = load_structure(path)
        self.assertEqual(len(atoms), 4)
        self.assertEqual(verification["coordination_numbers"], [4, 4, 4, 4])
        self.assertTrue(verification["projected_arrangement_verified"])
        self.assertFalse(verification["thermal_displacements_added"])
        self.assertEqual(path.read_bytes(), original)

    def test_installed_potential_array_is_integrated_not_slice_averaged(self):
        from abtem.potentials.iam import PotentialArray
        from benchmarks.abtem_reference import reference_configuration
        with reference_configuration():
            array = np.full((1, 16, 12), 9.7, dtype=np.float64)  # V Å
            a = PotentialArray(array, slice_thickness=0.2, extent=(5.43053, 3.83996)).transmission_function(60000).array
            b = PotentialArray(array, slice_thickness=0.8, extent=(5.43053, 3.83996)).transmission_function(60000).array
            np.testing.assert_array_equal(a, b)  # thickness is not applied a second time
            expected = np.exp(1j * interaction_coefficient(60000 * elementary_charge) * 9.7e-10)
            self.assertLess(float(np.max(np.abs(a - expected))), 1e-9)

    def test_shared_reference_phase_axis_filter_normalization_and_native_driver(self):
        from abtem.antialias import antialias_aperture as native_aperture
        from benchmarks.abtem_reference import load_structure, build_shared_slices, propagate_reference, reference_configuration, verify_direct_repeated_build
        atoms, _ = load_structure(Path(__file__).resolve().parents[1] / "Si_110_temp300.lmp")
        energy = 60000 * elementary_charge
        with reference_configuration():
            shared = build_shared_slices(atoms, 32, 8, 60000)
            mask = antialias_aperture(shared.grid)
            self.assertLess(float(np.max(np.abs(mask - native_aperture(shared.native.gpts, shared.native.sampling, xp=np).T))), 1e-12)
            reference = propagate_reference(shared, 2, check_driver=True)
            ours = ElasticPropagator(shared.grid, energy, mask).propagate(shared.incident_si, shared.projected_potential_v_m, shared.thicknesses_m, repetitions=2)
            self.assertLess(relative_wave_error(ours.wave, reference.wave_si, shared.grid), 1e-6)
            self.assertLess(float(np.max(np.abs(diffraction_probabilities(reference.wave_si, shared.grid) - reference.diffraction))), 1e-12)
            self.assertLess(float(np.max(np.abs(ours.norm_history - reference.norm_history))), 1e-7)
            self.assertEqual(reference.metadata["exit_wave_dtype"], "complex128")
            self.assertTrue(verify_direct_repeated_build(shared, atoms, 2)["passed"])


if __name__ == "__main__":
    unittest.main()
