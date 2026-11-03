"""Benchmark 4 thermodynamics, signed response, independent reference, and limits."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.constants import Boltzmann, elementary_charge
from scipy.integrate import trapezoid
from scipy.stats import poisson, skellam

from benchmarks.thermal_harmonic_response import ThermalConfig, detailed_balance, run_benchmark
from benchmarks.thermal_reference import (
    elastic_and_leading_terms, energy_window, exact_log_weights, exact_weights,
    exact_window_tail, reference_parameters, thermal_correlation,
)
from yoshioka.harmonic_oscillator import HarmonicOscillator, fock_scattering_matrix, quadrature_scattering_block
from yoshioka.thermal_response import (
    ThermalOscillatorState, broaden_signed_spectrum, final_state_cutoff,
    line_correlation, line_statistics, restrict_lines, transition_sum,
)


class ThermalResponseTests(unittest.TestCase):
    def setUp(self):
        self.oscillator = HarmonicOscillator.from_amu_mev(28.085, 60)
        self.energy = self.oscillator.quantum_energy_j

    def test_stable_geometric_and_bose_probabilities(self):
        zero = ThermalOscillatorState(self.energy, 0)
        np.testing.assert_array_equal(zero.probabilities(4), [1, 0, 0, 0, 0])
        self.assertEqual(zero.mean_occupation, 0)
        self.assertEqual(zero.omitted_probability(0), 0)
        for temperature in (0.001, 10, 100, 300, 1000, 1e8):
            state = ThermalOscillatorState(self.energy, temperature)
            self.assertTrue(np.isfinite(state.mean_occupation))
            self.assertGreaterEqual(state.mean_occupation, 0)
            self.assertTrue(np.all(np.isfinite(state.probabilities(8))))
            self.assertAlmostEqual(float(np.sum(state.probabilities(8))) + state.omitted_probability(8), 1, places=14)
        hot = ThermalOscillatorState(self.energy, 1e8)
        self.assertLess(abs(hot.mean_occupation / (Boltzmann * 1e8 / self.energy) - 1), 1e-5)

    def test_independent_initial_cutoffs_and_tiny_positive_temperature_gain(self):
        for temperature, expected in zip((0, 10, 100, 300, 600, 1000), (0, 1, 4, 12, 25, 42)):
            state = ThermalOscillatorState(self.energy, temperature)
            cutoff = state.initial_cutoff()
            self.assertEqual(cutoff, expected)
            self.assertLess(state.omitted_probability(cutoff), 1e-13)
            if cutoff > 1:
                self.assertGreaterEqual(state.omitted_probability(cutoff - 1), 1e-13)
        state = ThermalOscillatorState(self.energy, 10)
        response = transition_sum(fock_scattering_matrix(np.sqrt(0.1), 64), state.probabilities(state.initial_cutoff()), 16)
        gain = float(np.sum(response.weights[response.ell < 0]))
        self.assertGreater(gain, 0)
        self.assertLess(gain, 1e-30)
        reference = exact_weights(np.array([-1]), 0.1, self.energy, 10)[0]
        self.assertLess(abs(gain / reference - 1), 1e-12)

    def test_average_probabilities_and_signed_energy_grouping(self):
        matrix = np.array([[1, 1j, 0], [1j, -1, 1], [0, 1, 1j]], dtype=complex) / np.sqrt(3)
        probabilities = np.array([0.6, 0.3, 0.05])  # Intentionally not normalized.
        response = transition_sum(matrix, probabilities, 2)
        expected = np.array([0, (0.3 + 0.05) / 3, 0.95 / 3, (0.6 + 0.3) / 3, 0])
        np.testing.assert_array_equal(response.ell, [-2, -1, 0, 1, 2])
        np.testing.assert_allclose(response.weights, expected, atol=1e-16, rtol=0)
        self.assertAlmostEqual(response.retained_initial_weight, 0.95)
        self.assertAlmostEqual(response.coherent_amplitude.real, 0.3 / np.sqrt(3))
        self.assertNotAlmostEqual(response.weights[2], abs(response.coherent_amplitude)**2)

    def test_eta_zero_and_zero_temperature_limits(self):
        for temperature in (0, 300, 1000):
            state = ThermalOscillatorState(self.energy, temperature)
            m_max = state.initial_cutoff()
            response = transition_sum(fock_scattering_matrix(0, 64), state.probabilities(m_max), 63)
            self.assertEqual(np.sum(response.weights[response.ell != 0]), 0)
            self.assertLess(abs(np.sum(response.weights) - 1), 1e-12)
            np.testing.assert_array_equal(exact_weights(np.arange(-2, 3), 0, self.energy, temperature), [0, 0, 1, 0, 0])
        for eta in (0.01, 0.1, 1, 4):
            changes = np.arange(-8, 33)
            exact = exact_weights(changes, eta, self.energy, 0)
            np.testing.assert_allclose(exact, poisson.pmf(changes, eta), atol=1e-15, rtol=0)
            self.assertEqual(np.sum(exact[changes < 0]), 0)

    def test_bessel_reference_against_skellam_and_tiny_argument_fallback(self):
        changes = np.arange(-48, 49)
        for temperature in (100, 300, 1000):
            parameters = reference_parameters(4, self.energy, temperature)
            expected = skellam.pmf(changes, parameters['lambda_plus'], parameters['lambda_minus'])
            np.testing.assert_allclose(exact_weights(changes, 4, self.energy, temperature), expected, atol=1e-14, rtol=0)
        logs = exact_log_weights(np.array([-32, -1, 0, 1, 32]), 0.1, self.energy, 10)
        self.assertTrue(np.all(np.isfinite(logs)))
        low_loss = exact_weights(np.arange(33), 0.1, self.energy, 10)
        np.testing.assert_allclose(low_loss, poisson.pmf(np.arange(33), 0.1), atol=1e-15, rtol=0)
        self.assertGreater(exact_weights(np.array([-1]), 0.1, self.energy, 10)[0], 0)

    def test_stable_signed_window_tails(self):
        tail = exact_window_tail(32, 0, 0.01, self.energy, 0)
        self.assertGreater(tail['omitted_weight'], 0)
        self.assertLess(tail['omitted_weight'], 1e-90)
        self.assertEqual(tail['omitted_weight'], poisson.sf(32, 0.01))
        for eta in (0, 0.01, 0.1, 4):
            loss, gain, tail = energy_window(eta, self.energy, 1000)
            self.assertLess(tail['omitted_weight_upper_bound'], 1e-12)
            outside = np.r_[np.arange(-128, -gain), np.arange(loss + 1, 129)]
            direct = float(np.sum(exact_weights(outside, eta, self.energy, 1000)))
            self.assertLess(abs(direct - tail['omitted_weight']), 1e-24)

    def test_detailed_balance_mixed_residual_and_resolution_policy(self):
        changes = np.arange(-32, 33)
        exact = exact_weights(changes, 4, self.energy, 1000)
        x = self.energy / (Boltzmann * 1000)
        balance = detailed_balance(changes, exact, x, 1e-10, 1e-8, 1e-12)
        self.assertLess(balance['maximum_mixed_residual'], 1)
        self.assertLess(balance['maximum_resolved_log_ratio_error'], 1e-12)
        self.assertGreater(balance['resolved_pair_count'], 0)
        altered = exact.copy()
        altered[changes == -1] += 0.001
        self.assertGreater(detailed_balance(changes, altered, x, 1e-10, 1e-8, 1e-12)['maximum_mixed_residual'], 1)
        tiny = detailed_balance(np.array([-1, 0, 1]), np.array([0, 0.9, 0.1]), 1000, 1e-10, 1e-8, 1e-12)
        self.assertEqual(tiny['resolved_pair_count'], 0)
        self.assertIsNone(tiny['maximum_resolved_log_ratio_error'])

    def test_thermal_coherent_amplitude_differs_from_zero_energy_line(self):
        state = ThermalOscillatorState(self.energy, 300)
        p = state.probabilities(state.initial_cutoff())
        response = transition_sum(fock_scattering_matrix(2, 96), p, 64)
        expected = elastic_and_leading_terms(4, self.energy, 300)
        self.assertLess(abs(response.coherent_amplitude - expected['coherent_amplitude']), 1e-12)
        zero = response.weights[response.ell == 0][0]
        self.assertLess(abs(zero - expected['zero_energy_weight']), 1e-12)
        self.assertGreater(zero - abs(response.coherent_amplitude)**2, 0.01)
        tiny = elastic_and_leading_terms(0.1, self.energy, 10)
        self.assertGreater(tiny['zero_energy_minus_coherent_weight_stable'], 0)

    def test_net_one_quantum_lines_approach_leading_terms_at_weak_eta(self):
        state = ThermalOscillatorState(self.energy, 300)
        p = state.probabilities(state.initial_cutoff())
        corrections = []
        for eta in (1e-4, 1e-3, 1e-2, 0.1):
            response = transition_sum(fock_scattering_matrix(np.sqrt(eta), 64), p, 40)
            leading = elastic_and_leading_terms(eta, self.energy, 300)['leading_loss_one_phonon']
            numerical = response.weights[response.ell == 1][0]
            exact = exact_weights(np.array([1]), eta, self.energy, 300)[0]
            self.assertLess(abs(numerical - exact), 1e-12)
            corrections.append((exact - leading) / exact)
        self.assertTrue(np.all(np.diff(corrections) > 0))
        self.assertLess(corrections[0], 1e-8)
        self.assertGreater(corrections[-1], 1e-4)

    def test_signed_moments_and_one_period_correlation(self):
        eta, temperature = 4, 1000
        state = ThermalOscillatorState(self.energy, temperature)
        p = state.probabilities(state.initial_cutoff())
        matrix = fock_scattering_matrix(2, 128)
        n_max = final_state_cutoff(matrix, p)
        self.assertGreater(n_max, len(p) - 1)
        response = transition_sum(matrix, p, n_max)
        stats = line_statistics(response.ell, response.weights, eta, state.mean_occupation, self.energy)
        self.assertLess(stats['mean_error_scaled'], 1e-8)
        self.assertLess(stats['variance_error_scaled'], 1e-8)
        self.assertLess(abs(stats['raw_mean_ell'] - 4), 1e-8)
        times = np.linspace(0, 1 / self.oscillator.frequency_hz, 257)
        numerical = line_correlation(response.ell, response.weights, self.energy, times)
        exact = thermal_correlation(times, self.oscillator.angular_frequency_rad_s, eta, self.energy, temperature)
        self.assertLess(np.max(abs(numerical - exact)), 1e-10)
        self.assertLess(abs(numerical[0] - 1), 1e-10)

    def test_moment_error_scaling_uses_small_nonzero_exact_moments(self):
        statistics = line_statistics(np.array([0, 1]), np.array([0.99, 0.009]), 0.01, 0, self.energy)
        self.assertAlmostEqual(statistics['mean_error_scaled'], 0.1)
        expected_variance = 0.009 - 0.009**2
        self.assertAlmostEqual(statistics['variance_error_scaled'], abs(expected_variance - 0.01) / 0.01)
        zero = line_statistics(np.array([0]), np.array([1.0]), 0, 0, self.energy)
        self.assertEqual(zero['mean_error_scaled'], 0)
        self.assertEqual(zero['variance_error_scaled'], 0)

    def test_independent_initial_and_final_truncation_deficits(self):
        state = ThermalOscillatorState(self.energy, 1000)
        matrix = fock_scattering_matrix(2, 128)
        p = state.probabilities(state.initial_cutoff())
        complete = transition_sum(matrix, p, final_state_cutoff(matrix, p))
        low_initial = transition_sum(matrix, state.probabilities(1), 96)
        low_final = transition_sum(matrix, p, 1)
        self.assertLess(abs(1 - sum(complete.weights)), 1e-10)
        self.assertGreater(1 - sum(low_initial.weights), 0.2)
        self.assertGreater(low_final.final_omitted_weight_residual, 0.5)
        self.assertAlmostEqual(sum(low_initial.weights), sum(state.probabilities(1)), places=12)

    def test_finite_unitary_matrix_can_give_wrong_thermal_lines(self):
        state = ThermalOscillatorState(self.energy, 100)
        probabilities = state.probabilities(2)
        small = transition_sum(fock_scattering_matrix(2, 3), probabilities, 2)
        self.assertAlmostEqual(sum(small.weights), sum(probabilities), places=14)
        reference = exact_weights(small.ell, 4, self.energy, 100)
        self.assertGreater(np.max(abs(small.weights - reference)), 0.1)
        large = transition_sum(fock_scattering_matrix(2, 96), probabilities, 32)
        self.assertLess(np.max(abs(large.weights - exact_weights(large.ell, 4, self.energy, 100))), 1e-9)

    def test_quadrature_thermal_resolution_and_signed_Q(self):
        state = ThermalOscillatorState(self.energy, 1000)
        p = state.probabilities(state.initial_cutoff())
        fock = fock_scattering_matrix(2, 128)
        for nodes in (128, 256):
            quad = quadrature_scattering_block(2, 65, nodes)
            self.assertLess(np.linalg.norm((quad[:66, :len(p)] - fock[:66, :len(p)]) * np.sqrt(p)), 1e-10)
        positive = transition_sum(fock, p, 65)
        negative = transition_sum(fock_scattering_matrix(-2, 128), p, 65)
        np.testing.assert_allclose(positive.weights, negative.weights, atol=1e-15, rtol=0)

    def test_gaussian_area_signed_components_and_no_normalization(self):
        ell, weights = np.array([-1, 0, 1]), np.array([0.1, 0.3, 0.2])
        arrays, metrics = broaden_signed_spectrum(ell, weights, self.energy, 2e-3 * elementary_charge)
        self.assertAlmostEqual(metrics['integrated_weight'], 0.6, places=12)
        self.assertLess(metrics['area_error_including_window_bound'], 1e-12)
        self.assertFalse(metrics['normalization_applied'])
        conversion = 1e-3 * elementary_charge
        self.assertAlmostEqual(trapezoid(arrays['total_density_per_j'] * conversion, arrays['energy_grid_j'] / conversion), 0.6, places=12)
        for name, weight in (('gain', 0.1), ('elastic', 0.3), ('loss', 0.2)):
            self.assertAlmostEqual(trapezoid(arrays[name + '_density_per_j'], arrays['energy_grid_j']), weight, places=12)

    def test_invalid_parameters(self):
        for temperature in (-1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                ThermalOscillatorState(self.energy, temperature)
        for configuration in ({'temperature_k': -1}, {'initial_tail_tolerance': 0}, {'quadrature_orders': (64,)}, {'eta': -1}, {'q_inv_angstrom': 1, 'eta': 4}):
            with self.assertRaises(ValueError):
                ThermalConfig(**configuration)
        with self.assertRaises(ValueError):
            transition_sum(np.eye(2), np.ones(3), 1)


class ThermalBenchmarkTests(unittest.TestCase):
    def test_default_sweep_acceptance_and_matrix_cache(self):
        with patch('benchmarks.thermal_harmonic_response.fock_scattering_matrix', wraps=fock_scattering_matrix) as exponential:
            results = run_benchmark(ThermalConfig())
        self.assertEqual(results['status'], 'PASS', results['checks'])
        self.assertEqual(len(results['cases']), 36)
        self.assertEqual(exponential.call_count, 6 * 4)  # Reused for all six T.
        self.assertEqual(results['optional_propagation']['status'], 'NOT_PERFORMED')
        self.assertEqual(results['prior_benchmark_statuses_preserved']['elastic_si110']['status'], 'FAIL')
        self.assertTrue(all(all(case['checks'].values()) for case in results['cases']))
        json.dumps(results, allow_nan=False)

    def test_underresolved_quadrature_reports_failure_without_looser_tolerances(self):
        results = run_benchmark(ThermalConfig(sweep_etas=(4,), sweep_temperatures_k=(300,), quadrature_orders=(8, 16), working_dimensions=(64, 96)))
        self.assertEqual(results['status'], 'FAIL')
        self.assertFalse(results['checks']['quadrature_convergence'])
        self.assertEqual(results['parameters']['line_tolerance'], 1e-10)

    def test_output_data_and_preservation_guard(self):
        with tempfile.TemporaryDirectory(prefix='yoshioka-thermal-output-') as directory:
            output = Path(directory)
            previous = b'{"benchmark":"Benchmark 3","status":"PASS"}\n'
            (output / 'results.json').write_bytes(previous)
            with self.assertRaisesRegex(ValueError, 'another benchmark'):
                run_benchmark(ThermalConfig(), output)
            self.assertEqual((output / 'results.json').read_bytes(), previous)
            self.assertFalse((output / 'arrays').exists())
        with tempfile.TemporaryDirectory(prefix='yoshioka-thermal-data-') as directory:
            output = Path(directory)
            results = run_benchmark(ThermalConfig(sweep_etas=(0.1,), sweep_temperatures_k=(300,), working_dimensions=(64, 96), quadrature_orders=(64, 128)), output)
            self.assertEqual(results['status'], 'PASS')
            case = results['cases'][0]
            with np.load(output / case['artifact']) as data:
                self.assertTrue(np.any(data['energy_losses_j'] < 0))
                self.assertEqual(data['energy_losses_j'][case['gain_max']], 0)
                self.assertLess(abs(sum(data['fock_weights']) - 1), 1e-10)


if __name__ == '__main__':
    unittest.main()
