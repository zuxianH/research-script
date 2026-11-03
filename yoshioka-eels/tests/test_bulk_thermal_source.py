"""Independent normalization, thermal, stochastic and resume controls."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import constants

from yoshioka.bulk_thermal_source import (occupations, importance_nodes, mode_group,
    covariance_integral, born_thermal_batch)
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.electron_phonon import full_space_born, LobatoSilicon
from yoshioka.bloch_source import bath_groups
from yoshioka.bloch_covariance import covariance_from_groups
from yoshioka.free_propagation import TransverseGrid, electron_wavenumber
from yoshioka.dynamical_source import full_transverse_vectors
from yoshioka.fourier_depth_fields import complex_depth_field, fourier_depth_provider
from benchmarks.approximate_eels import (geometry, run_source_case, covariance_path,
    PRIMITIVE, ORIENTED)
from benchmarks.born_quadrature import born_batch
from benchmarks.workflow_io import ResourceLimits


class BulkThermalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primitive = RealSpaceForceConstants.load(PRIMITIVE)
        cls.oriented = RealSpaceForceConstants.load(ORIENTED)

    def test_born_extension_matches_existing_B7_at_zero_temperature(self):
        _, positions, length = geometry(self.oriented, 4)
        covariance, _ = covariance_integral(self.primitive, 0, 4)
        q = np.array([[.17, -.21, .13], [-.31, .08, .23]])
        weights = np.array([.3, .7]); lengths = np.diag(self.oriented.cell_m)[:2]
        e, amplitude, _, angle, _ = born_batch(self.primitive, positions, covariance, lengths,
            length, 12e-10, q, weights, 60000*constants.elementary_charge)
        exact = np.stack([np.sum(abs(amplitude)**2*m[:, None], axis=-1) for m in (
            angle <= 20+1e-12, angle < 5, (angle >= 5)&(angle < 10), (angle >= 10)&(angle <= 20+1e-12))], axis=-1)
        energies, actual, _ = born_thermal_batch(self.primitive, positions, covariance, lengths,
            length, 12e-10, q, weights, 0, 60000*constants.elementary_charge)
        np.testing.assert_allclose(actual[:, :6], exact, rtol=2e-12, atol=1e-20)
        np.testing.assert_allclose(energies[:, :6], e)
        np.testing.assert_array_equal(actual[:, 6:], 0)

    def test_signed_thermal_born_against_independent_atomic_sum(self):
        _, positions, length = geometry(self.oriented, 4)
        covariance, _ = covariance_integral(self.primitive, 300, 4)
        q = np.array([.17, -.21, .13]); weight = .4
        grid = TransverseGrid(*np.diag(self.oriented.cell_m)[:2], 11, 11)
        omega, d, shift, _ = mode_group(self.primitive, positions, [grid.lx_m, grid.ly_m], q, weight)
        _, expected, _ = born_thermal_batch(self.primitive, positions, covariance,
            [grid.lx_m, grid.ly_m], length, 12e-10, q[None], [weight], 300, 60000*constants.elementary_charge)
        nbar = occupations(omega, 300)
        for half, sign in enumerate((1, -1)):
            Q = full_transverse_vectors(grid)+sign*shift
            angle = np.linalg.norm(Q, axis=1)/electron_wavenumber(60000*constants.elementary_charge)*1000
            a = full_space_born(LobatoSilicon.from_abtem_table(), positions, d if sign == 1 else d.conj(),
                covariance, Q, sign*omega, 60000*constants.elementary_charge, grid.lx_m*grid.ly_m, length)
            probability = abs(a)**2*(nbar+1 if sign == 1 else nbar)[:, None]
            actual = np.stack([np.sum(probability[:, mask], axis=1) for mask in (
                angle <= 20+1e-12, angle < 5, (angle >= 5)&(angle < 10), (angle >= 10)&(angle <= 20+1e-12))], axis=-1)
            np.testing.assert_allclose(actual, expected[0, half*6:(half+1)*6], rtol=3e-12, atol=1e-20)

    def test_primitive_and_folded_finite_bath_normalization(self):
        positions, groups = bath_groups(self.oriented, (2, 2, 2), 2)
        covariance, _ = covariance_from_groups(groups, 4)
        relation = self.oriented.reciprocal_m_inv @ np.linalg.inv(self.primitive.reciprocal_m_inv)
        coset = next(row for row in relation if np.linalg.norm(row-np.rint(row)) > 1e-8)
        lengths = np.diag(self.oriented.cell_m)[:2]; length = 2*self.oriented.cell_m[2, 2]+24e-10
        primitive_total = np.zeros(4); folded_total = np.zeros(4)
        grid = TransverseGrid(*lengths, 11, 11)
        for group in groups:
            q0 = group['q_fractional']@relation
            q = np.array([q0, q0+coset]); q = (q+.5)%1-.5
            _, w, _ = born_thermal_batch(self.primitive, positions, covariance, lengths,
                length, 12e-10, q, np.full(2, 1/16), 0, 60000*constants.elementary_charge)
            primitive_total += np.sum(w, axis=(0, 1))
            Q = full_transverse_vectors(grid)+group['bloch_shift_m_inv']
            angle = np.linalg.norm(Q, axis=1)/electron_wavenumber(60000*constants.elementary_charge)*1000
            a = full_space_born(LobatoSilicon.from_abtem_table(), positions, group['displacements_m'], covariance,
                Q, group['omega_rad_s'], 60000*constants.elementary_charge, np.prod(lengths), length)
            for j, mask in enumerate((angle <= 20+1e-12, angle < 5, (angle >= 5)&(angle < 10), (angle >= 10)&(angle <= 20+1e-12))):
                folded_total[j] += np.sum(abs(a[:, mask])**2)
        np.testing.assert_allclose(primitive_total, folded_total, rtol=1e-9, atol=1e-18)

    def test_radial_importance_weights_known_integrals_and_reproducibility(self):
        values = []
        for seed in range(8):
            q, weight = importance_nodes(10, 59031+seed)
            r = np.linalg.norm(q, axis=1)
            values.append([np.mean(weight), np.mean(np.sum(q*q, axis=1)*weight),
                           np.mean(np.where(r < .35, 1/r**2, 0)*weight)])
            a, b = importance_nodes(10, 59031+seed)
            np.testing.assert_array_equal(a, q); np.testing.assert_array_equal(b, weight)
        np.testing.assert_allclose(np.mean(values, axis=0), [1, .25, 4*np.pi*.35], rtol=.012, atol=.005)

    def test_bose_factors_against_explicit_thermal_trace(self):
        omega = np.array([30e-3*constants.elementary_charge/constants.hbar])
        nbar = occupations(omega, 300)[0]
        ratio = np.exp(-constants.hbar*omega[0]/(constants.Boltzmann*300))
        n = np.arange(1000); probability = (1-ratio)*ratio**n
        self.assertAlmostEqual(float(np.sum(probability*(n+1))), nbar+1, places=13)
        self.assertAlmostEqual(float(np.sum(probability*n)), nbar, places=13)
        np.testing.assert_array_equal(occupations(omega, 0), [0])

    def test_conditional_feature_matching_preserves_unmatched_peaks(self):
        from benchmarks.approximate_eels_compare import match_features
        a = {'region': {'all_resolved_peaks_mev': [23., 45., 68.]}}
        b = {'region': {'all_resolved_peaks_mev': [23.2, 56., 70.5]}}
        result = match_features(a, b)
        self.assertEqual(len(result['matches']), 2)
        self.assertEqual(result['unmatched_reference_mev'], [45.])
        self.assertEqual(result['unmatched_model_mev'], [56.])

    def test_cached_physics_rejects_changed_geometry_identity(self):
        from benchmarks.approximate_eels import verify_input_baseline
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'protected_before.json'
            path.write_text(json.dumps({'files': {'benchmark_results/si_lattice_dynamics/hiphive_model.npz': {'sha256': '0'*64}}}))
            with self.assertRaisesRegex(ValueError, 'Physical input changed'):
                verify_input_baseline(Path(directory))

    def test_absorption_source_is_adjoint_with_opposite_bloch_shift(self):
        positions, reference, length = geometry(self.oriented, 2)
        covariance, _ = covariance_integral(self.primitive, 300, 4)
        grid = TransverseGrid(*np.diag(self.oriented.cell_m)[:2], 7, 7)
        omega, d, shift, _ = mode_group(self.primitive, positions, [grid.lx_m, grid.ly_m],
                                       [.17, -.21, .13], reference_positions=reference)
        with tempfile.TemporaryDirectory() as directory:
            emission = complex_depth_field(positions, covariance, grid, length, 33, Path(directory)/'emission.npy', d, shift)
            absorption = complex_depth_field(positions, covariance, grid, length, 33, Path(directory)/'absorption.npy', d.conj(), -shift)
            a = fourier_depth_provider(emission, grid); b = fourier_depth_provider(absorption, grid)
            for index in (0, 7, 16, 22, 32):
                self.assertLess(np.linalg.norm(a(index).conj()-b(index))/np.linalg.norm(a(index)), 2e-13)

    def test_source_prefix_resume_matches_fresh_run(self):
        config = {'selected_repeats': 2, 'padding_angstrom': 12, 'selected_covariance_mesh': 4}
        settings = {'name': 'resume', 'temperature_k': 300, 'grid_unit': 5, 'slices': 17, 'power': 1, 'seed': 71903}
        covariance, _ = covariance_integral(self.primitive, 300, 4)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory in (first, second):
                path = covariance_path(Path(directory), 300, 4); path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'covariance_m2': covariance.tolist()}))
            partial = run_source_case(config, settings, Path(first), ResourceLimits(60, 2), 1)
            self.assertEqual(partial['completed_q'], 1)
            resumed = run_source_case(config, settings, Path(first), ResourceLimits(60, 2))
            fresh = run_source_case(config, settings, Path(second), ResourceLimits(60, 2))
            self.assertEqual(resumed['status'], 'COMPLETE'); self.assertEqual(fresh['status'], 'COMPLETE')
            for name in ('weights', 'born_weights', 'energies_mev'):
                np.testing.assert_array_equal(np.load(Path(first)/'source/resume'/f'{name}.npy'),
                                              np.load(Path(second)/'source/resume'/f'{name}.npy'))


if __name__ == '__main__': unittest.main()
