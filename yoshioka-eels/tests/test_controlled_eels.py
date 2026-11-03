"""Physical-model/cache separation and paired-batch uncertainty controls."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import constants

from benchmarks.controlled_eels import validate_covariances, check_source_contract, PREVIOUS
from benchmarks.controlled_eels_analysis import paired_estimate, window_sum
from benchmarks.approximate_eels import (PRIMITIVE, covariance_path, run_source_case,
    prepare_background)
from benchmarks.workflow_io import ResourceLimits, sha256
from yoshioka.bulk_thermal_source import covariance_integral, mode_group, occupations
from yoshioka.lattice_dynamics import RealSpaceForceConstants


class ControlledEelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.force = PREVIOUS/'ehm/force_constants_T300.npz'
        cls.model = RealSpaceForceConstants.load(cls.force)
        cls.covariance, _ = covariance_integral(cls.model, 300, 4)

    def config(self):
        return {'temperatures_k': [300], 'covariance_meshes': [4], 'selected_covariance_mesh': 4,
            'selected_repeats': 2, 'padding_angstrom': 12,
            'primitive_force_constants_by_temperature': {'300': str(self.force)}}

    def write_covariance(self, folder, force=None, value=None):
        path = covariance_path(folder, 300, 4); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'covariance_m2': (self.covariance if value is None else value).tolist(),
            'force_constants_sha256': sha256(self.force if force is None else force)}))

    def test_covariance_reuse_rejects_other_phonon_model(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); self.write_covariance(folder, force=PRIMITIVE)
            with self.assertRaisesRegex(ValueError, 'force-model identity'): validate_covariances(self.config(), folder)

    def test_covariance_is_positive_and_temperature_model_specific(self):
        fixed, _ = covariance_integral(RealSpaceForceConstants.load(PRIMITIVE), 300, 4)
        self.assertGreater(np.trace(self.covariance), np.trace(fixed))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); self.write_covariance(folder); validate_covariances(self.config(), folder)
            self.write_covariance(folder, value=-self.covariance)
            with self.assertRaisesRegex(ValueError, 'Invalid thermal'): validate_covariances(self.config(), folder)

    def test_background_cache_separates_covariance_and_geometry(self):
        settings = {'temperature_k': 300, 'grid_unit': 5, 'slices': 17}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); c = self.config(); self.write_covariance(folder)
            prepare_background(c, settings, folder, ResourceLimits(60, 2))
            original = set((folder/'background').iterdir())
            self.write_covariance(folder, value=self.covariance*1.1)
            prepare_background(c, settings, folder, ResourceLimits(60, 2))
            self.assertEqual(len(set((folder/'background').iterdir())-original), 1)
            prepare_background({**c, 'selected_repeats': 3}, settings, folder, ResourceLimits(60, 2))
            self.assertEqual(len(list((folder/'background').iterdir())), 3)

    def test_fitted_source_uses_actual_modes_and_resumes_identically(self):
        settings = {'name': 'fitted', 'temperature_k': 300, 'grid_unit': 5, 'slices': 17, 'power': 1, 'seed': 71903}
        config = self.config()
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a, b = Path(first), Path(second)
            for path in (a, b): self.write_covariance(path)
            run_source_case(config, settings, a, ResourceLimits(60, 2), 1)
            run_source_case(config, settings, a, ResourceLimits(60, 2))
            run_source_case(config, settings, b, ResourceLimits(60, 2))
            for name in ('energies_mev', 'weights', 'born_weights'):
                np.testing.assert_array_equal(np.load(a/'source/fitted'/f'{name}.npy'), np.load(b/'source/fitted'/f'{name}.npy'))
            with np.load(a/'source/fitted/spectrum.npz') as saved:
                q = saved['q_fractional'][0]
            omega, _, _, _ = mode_group(self.model, self.model.positions_m, [1e-9, 1e-9], q)
            energies = np.load(a/'source/fitted/energies_mev.npy')[0]
            np.testing.assert_allclose(energies[:6], omega*constants.hbar/(1e-3*constants.elementary_charge), rtol=1e-14)
            np.testing.assert_array_equal(energies[6:], -energies[:6])
            self.assertEqual(check_source_contract(config, settings, a), 2)
            from benchmarks.controlled_eels_run import verify_complete
            snapshot = {p.name: sha256(p) for p in (a/'source/fitted').iterdir() if p.is_file()}
            verify_complete({**config, 'source_cases': [settings]}, a, 'source')
            self.assertEqual(snapshot, {p.name: sha256(p) for p in (a/'source/fitted').iterdir() if p.is_file()})
            with self.assertRaisesRegex(ValueError, 'physical contract'):
                check_source_contract({**config, 'primitive_force_constants_by_temperature': {}}, settings, a)
            with self.assertRaisesRegex(ValueError, 'physical contract'):
                check_source_contract({**config, 'selected_repeats': 3}, settings, a)
            self.write_covariance(a, value=self.covariance*1.01)
            with self.assertRaisesRegex(ValueError, 'physical contract'):
                check_source_contract(config, settings, a)

    def test_paired_identical_estimators_have_zero_difference_uncertainty(self):
        rng = np.random.default_rng(19); source = rng.normal(size=(8, 40, 4)); gamma = rng.normal(size=(8, 40, 4))
        mean, sem, half = paired_estimate(source, source, gamma, gamma)
        for a in (mean, sem, half): np.testing.assert_array_equal(a, 0)

    def test_nested_sampling_prefix_refinement_matches_fresh_source(self):
        from benchmarks.controlled_eels_sampling import promote_prefix
        config = self.config()
        case = {'name': 'nested', 'temperature_k': 300, 'grid_unit': 5, 'slices': 17, 'power': 1, 'seed': 71299}
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a, b = Path(first), Path(second)
            for folder in (a, b): self.write_covariance(folder)
            run_source_case(config, case, a, ResourceLimits(60, 2))
            old = {p.name: sha256(p) for p in (a/'source/nested').iterdir() if p.is_file()}
            refined = promote_prefix(config, case, a, 2)
            self.assertEqual(check_source_contract(config, refined, a), 2)
            run_source_case(config, refined, a, ResourceLimits(60, 2))
            run_source_case(config, refined, b, ResourceLimits(60, 2))
            for name in ('energies_mev', 'weights', 'born_weights'):
                np.testing.assert_array_equal(np.load(a/'source'/refined['name']/(name+'.npy')),
                                              np.load(b/'source'/refined['name']/(name+'.npy')))
            self.assertEqual(old, {p.name: sha256(p) for p in (a/'source/nested').iterdir() if p.is_file()})
            with self.assertRaisesRegex(ValueError, 'physical contract'):
                promote_prefix(config, {**case, 'seed': case['seed']+1}, a, 3)

    def test_paired_correlation_and_independent_gamma_variance(self):
        rng = np.random.default_rng(10); common = 100*rng.normal(size=(8, 5, 4))
        delta = rng.normal(size=common.shape); ga = rng.normal(size=common.shape); gd = rng.normal(size=common.shape)
        mean, sem, _ = paired_estimate(common, common+delta, ga, ga+gd)
        expected = np.var(delta, axis=0, ddof=1)/8+np.var(gd, axis=0, ddof=1)/8
        np.testing.assert_allclose(sem**2, expected, rtol=2e-14, atol=1e-15)
        np.testing.assert_allclose(mean, np.mean(delta+gd, axis=0), atol=1e-14)

    def test_paired_estimates_keep_signed_absolute_scale(self):
        a = np.zeros((8, 2, 4)); b = np.ones_like(a)*-.2; g = np.zeros_like(a)
        mean, sem, _ = paired_estimate(a, b, g, g)
        np.testing.assert_allclose(mean, -.2); np.testing.assert_allclose(sem, 0, atol=1e-16)
        with self.assertRaises(ValueError): paired_estimate(a, b[:4], g, g)

    def test_raw_window_counts_signed_transitions_without_broadening(self):
        energy = np.array([[0., 39., 40., 65., -39., -65.]])
        weights = np.ones((1, 6, 4))*.2
        np.testing.assert_allclose(window_sum(energy, weights, 0, 40), .4)
        np.testing.assert_allclose(window_sum(energy, weights, -65, 0), .4)
        np.testing.assert_allclose(window_sum(energy, weights, -np.inf, np.inf), 1.2)

    def test_thermal_factors_for_fitted_modes_match_explicit_trace(self):
        omega = self.model.modes(np.array([[.071, -.113, .193]]))[0][0]
        for temperature in (300, 1000):
            for frequency, nbar in zip(omega, occupations(omega, temperature)):
                r = np.exp(-constants.hbar*frequency/(constants.Boltzmann*temperature)); n = np.arange(4096)
                p = (1-r)*r**n
                self.assertAlmostEqual(float(np.sum(p*n)), nbar, places=12)
                self.assertAlmostEqual(float(np.sum(p*(n+1))), nbar+1, places=12)


if __name__ == '__main__': unittest.main()
