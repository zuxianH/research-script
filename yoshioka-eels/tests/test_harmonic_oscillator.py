"""Analytical checks of oscillator amplitudes, cutoffs, and correlations."""

import unittest

import numpy as np
from scipy.constants import atomic_mass, elementary_charge, h, hbar

from references.oscillator_reference import (
    adaptive_cutoff, exact_correlation, omitted_inelastic_fraction,
    omitted_probability, poisson_weights, vacuum_amplitudes,
)
from yoshioka.harmonic_oscillator import (
    HarmonicOscillator, fock_scattering_matrix, ladder_operators,
    normalized_hermite_polynomials, quadrature_scattering_block,
    quadrature_vacuum_amplitudes, resolve_coupling, spectral_correlation,
)


class OscillatorNumericalTests(unittest.TestCase):
    def setUp(self):
        self.oscillator = HarmonicOscillator.from_amu_mev(28.085, 60)

    def test_si_conversions_and_oscillator_length(self):
        oscillator = self.oscillator
        self.assertEqual(oscillator.mass_kg, 28.085 * atomic_mass)
        self.assertEqual(oscillator.quantum_energy_j, 60e-3 * elementary_charge)
        self.assertAlmostEqual(oscillator.angular_frequency_rad_s / oscillator.frequency_hz, 2 * np.pi)
        self.assertAlmostEqual(oscillator.frequency_hz / (oscillator.quantum_energy_j / h), 1)
        self.assertAlmostEqual(oscillator.zero_point_displacement_m**2 / (hbar / (2 * oscillator.mass_kg * oscillator.angular_frequency_rad_s)), 1)
        self.assertAlmostEqual(oscillator.oscillator_length_m / oscillator.zero_point_displacement_m, np.sqrt(2))

    def test_signed_Q_and_eta_are_consistent_without_two_pi(self):
        q, xi, eta = resolve_coupling(self.oscillator)
        self.assertEqual(eta, 0.1)
        self.assertGreater(q, 0)
        self.assertAlmostEqual(q * self.oscillator.zero_point_displacement_m, np.sqrt(0.1))
        negative = resolve_coupling(self.oscillator, -q, eta)
        self.assertLess(negative[1], 0)
        self.assertAlmostEqual(negative[2], eta)
        for q_bad, eta_bad in ((q, 0.5), (0, 1e-30), (q * 1e-10, 0.1)):
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                resolve_coupling(self.oscillator, q_bad, eta_bad)
        self.assertEqual(resolve_coupling(self.oscillator, 0, 0), (0.0, 0.0, 0.0))

    def test_finite_ladder_commutator_and_ground_variance(self):
        dimension = 16
        lower, raise_ = ladder_operators(dimension)
        expected = np.eye(dimension)
        expected[-1, -1] = 1 - dimension
        np.testing.assert_allclose(lower @ raise_ - raise_ @ lower, expected, atol=1e-13)
        coordinate = self.oscillator.zero_point_displacement_m * (lower + raise_)
        self.assertAlmostEqual((coordinate @ coordinate)[0, 0].real / self.oscillator.zero_point_displacement_m**2, 1)

    def test_explicit_complex_phase_cycle(self):
        xi = np.sqrt(0.1)
        ground = np.exp(-xi**2 / 2)
        expected = ground * np.array([1, -1j * xi, -xi**2 / np.sqrt(2), 1j * xi**3 / np.sqrt(6)])
        for numerical in (fock_scattering_matrix(xi, 32)[:4, 0], quadrature_vacuum_amplitudes(xi, 3, 64)):
            np.testing.assert_allclose(numerical, expected, atol=1e-12, rtol=0)

    def test_Q_zero_identity_without_phase_alignment(self):
        np.testing.assert_array_equal(fock_scattering_matrix(0, 32), np.eye(32, dtype=complex))
        block = quadrature_scattering_block(0, 6, 128)
        np.testing.assert_allclose(block, np.eye(7), atol=1e-12, rtol=0)
        amplitudes = quadrature_vacuum_amplitudes(0, 32, 128)
        self.assertLess(np.linalg.norm(amplitudes - vacuum_amplitudes(0, 32)), 1e-10)

    def test_required_coupling_sweep_amplitudes_probabilities_and_DW(self):
        for eta in (0, 0.01, 0.1, 0.5, 1, 4):
            xi = np.sqrt(eta)
            exact = vacuum_amplitudes(xi, 32)
            probability = poisson_weights(eta, 32)
            for method, numerical in (
                ("fock", fock_scattering_matrix(xi, 64)[:33, 0]),
                ("quadrature", quadrature_vacuum_amplitudes(xi, 32, 128)),
            ):
                with self.subTest(eta=eta, method=method):
                    self.assertEqual(numerical.dtype, np.dtype(np.complex128))
                    self.assertLess(np.linalg.norm(numerical - exact), 1e-10)
                    self.assertLess(np.max(np.abs(np.abs(numerical)**2 - probability)), 1e-10)
                    self.assertLess(abs(numerical[0] - np.exp(-eta / 2)), 1e-10)
                    self.assertLess(abs(abs(numerical[0])**2 - np.exp(-eta)), 1e-10)

    def test_general_matrix_elements_and_opposite_Q_adjoint(self):
        for eta in (0, 0.1, 4):
            xi = np.sqrt(eta)
            fock = fock_scattering_matrix(xi, 64)[:7, :7]
            quad = quadrature_scattering_block(xi, 6, 128)
            np.testing.assert_allclose(fock, quad, atol=1e-10, rtol=0)
            for negative in (fock_scattering_matrix(-xi, 64)[:7, :7], quadrature_scattering_block(-xi, 6, 128)):
                self.assertLess(np.linalg.norm(negative - fock.conj().T), 1e-10)
            if eta == 4:
                self.assertGreater(np.linalg.norm(fock.conj().T @ fock - np.eye(7)), 0.1)

    def test_premature_truncation_is_unitary_but_wrong(self):
        p, xi = 2, 2.0
        small = fock_scattering_matrix(xi, p + 1)
        correct = vacuum_amplitudes(xi, p)
        self.assertAlmostEqual(float(np.sum(np.abs(small[:, 0])**2)), 1)
        self.assertLess(np.linalg.norm(small.conj().T @ small - np.eye(p + 1)), 1e-12)
        self.assertLess(float(np.sum(np.abs(correct)**2)), 0.25)
        self.assertGreater(np.linalg.norm(small[:, 0] - correct), 0.5)
        large = fock_scattering_matrix(xi, 64)[:p+1, :p+1]
        self.assertGreater(np.linalg.norm(large - small), 1)

    def test_fixed_p_converges_with_working_D(self):
        p, xi = 2, 2.0
        exact = vacuum_amplitudes(xi, p)
        self.assertGreater(np.linalg.norm(fock_scattering_matrix(xi, 3)[:p+1, 0] - exact), 0.1)
        for dimension in (32, 64, 96, 128):
            self.assertLess(np.linalg.norm(fock_scattering_matrix(xi, dimension)[:p+1, 0] - exact), 1e-10)

    def test_quadrature_refinement_reveals_underresolution(self):
        exact = vacuum_amplitudes(2, 32)
        self.assertGreater(np.linalg.norm(quadrature_vacuum_amplitudes(2, 32, 32) - exact), 1e-10)
        for order in (64, 128, 256):
            self.assertLess(np.linalg.norm(quadrature_vacuum_amplitudes(2, 32, order) - exact), 1e-10)

    def test_stable_tails_and_zero_inelastic_branch(self):
        tail = omitted_probability(0.01, 32)
        self.assertGreater(tail, 0)
        self.assertLess(tail, 1e-90)
        self.assertAlmostEqual(omitted_probability(0.1, 0), -np.expm1(-0.1), places=14)
        self.assertEqual(omitted_inelastic_fraction(0, 4), 0)
        self.assertIsNone(omitted_inelastic_fraction(0.1, 0))
        self.assertAlmostEqual(omitted_inelastic_fraction(0.1, 1), omitted_probability(0.1, 1) / -np.expm1(-0.1))


    def test_energy_loss_origin_and_recoil_identity(self):
        losses = self.oscillator.energy_losses_j(6)
        self.assertEqual(losses[0], 0)
        self.assertTrue(np.all(losses >= 0))
        np.testing.assert_allclose(losses / (1e-3 * elementary_charge), np.arange(7) * 60, atol=1e-12)
        for eta in (0, 0.1, 4):
            q, _, _ = resolve_coupling(self.oscillator, eta=eta)
            recoil = hbar**2 * q**2 / (2 * self.oscillator.mass_kg)
            self.assertLess(abs(recoil - eta * self.oscillator.quantum_energy_j) / self.oscillator.quantum_energy_j, 1e-12)

    def test_correlation_one_period_and_finite_p_deficit(self):
        eta, xi = 4, 2
        times = np.linspace(0, 1 / self.oscillator.frequency_hz, 257)
        exact = exact_correlation(eta, self.oscillator.angular_frequency_rad_s, times)
        p = adaptive_cutoff(eta, 1e-12)
        for amplitudes in (fock_scattering_matrix(xi, 64)[:p+1, 0], quadrature_vacuum_amplitudes(xi, p, 128)):
            correlation = spectral_correlation(amplitudes, self.oscillator.quantum_energy_j, times)
            self.assertLess(np.max(np.abs(correlation - exact)), 1e-10)
            self.assertLess(abs(correlation[0] - 1), 1e-10)
        low = spectral_correlation(vacuum_amplitudes(xi, 2), self.oscillator.quantum_energy_j, times)
        self.assertAlmostEqual(1 - low[0].real, omitted_probability(eta, 2), places=13)
        self.assertLess(abs(exact[-1] - 1), 1e-12)


    def test_reject_invalid_physical_and_numerical_inputs(self):
        for value in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                HarmonicOscillator(value, 1e-20)
            with self.assertRaises(ValueError):
                HarmonicOscillator(1e-26, value)
        for dimension in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                fock_scattering_matrix(0.1, dimension)
        with self.assertRaises(ValueError):
            quadrature_vacuum_amplitudes(0.1, 4, 0)
        with self.assertRaises(ValueError):
            normalized_hermite_polynomials(np.array([np.nan]), 4)




if __name__ == "__main__":
    unittest.main()
