import unittest

import numpy as np
from scipy.constants import elementary_charge, hbar
from scipy.special import gammaln

from yoshioka.coupled_channels import (ChannelSegment, SplitChannelPropagator, channel_basis,
    fourier_populations, linear_displacement_block, oscillator_phase_block, thermal_channel_weights)
from yoshioka.elastic_propagation import electron_speed
from yoshioka.free_propagation import TransverseGrid
from references.channel_reference import dense_generators, dense_propagate, distorted_first_order
from yoshioka.distorted_channels import distorted_wave_orders


class CoupledChannelTests(unittest.TestCase):
    def test_channel_count_and_preallocation_guard(self):
        self.assertEqual(len(channel_basis(2, 8)), 45)
        self.assertEqual(len(channel_basis(3, 5)), 56)
        with self.assertRaises(MemoryError):
            channel_basis(100, 10)

    def test_independent_nonlinear_phase_blocks(self):
        basis = channel_basis(2, 6)
        numerical = oscillator_phase_block(basis, [.3, -.21], working_dimension=32)
        quadrature = oscillator_phase_block(basis, [.3, -.21], working_dimension=32, quadrature_order=64)
        self.assertLess(np.linalg.norm(numerical-quadrature), 1e-12)
        reverse = oscillator_phase_block(basis, [-.3, .21], working_dimension=32)
        np.testing.assert_allclose(reverse, numerical.conj().T, atol=1e-14)

    def test_solvable_driven_oscillator_complex_phase_regression_3B(self):
        grid = TransverseGrid(1e-9, 1e-9, 2, 2)
        basis = channel_basis(1, 24)
        energy = 60000*elementary_charge
        quantum = .06*elementary_charge
        force = .047*elementary_charge
        omega = quantum/hbar
        theta = 1.3
        length = theta*electron_speed(energy)/omega
        coupling = linear_displacement_block(basis, [force])
        solver = SplitChannelPropagator(grid, energy, basis[:, 0]*quantum)
        initial = np.zeros((len(basis), 2, 2), dtype=complex)
        initial[0] = 1/np.sqrt(grid.lx_m*grid.ly_m)
        wave, record = solver.propagate(initial, [ChannelSegment(length, coupling)], steps_per_segment=4)
        lam = force/quantum
        alpha = lam*(np.exp(-1j*theta)-1)
        n = basis[:, 0]
        amplitudes = np.exp(1j*lam**2*(theta-np.sin(theta))-.5*abs(alpha)**2)*alpha**n*np.exp(-.5*gammaln(n+1))
        exact = amplitudes[:, None, None]*np.ones((1, 2, 2))/np.sqrt(grid.lx_m*grid.ly_m)
        self.assertLess(np.linalg.norm(wave-exact)/np.linalg.norm(exact), 2e-13)
        self.assertLess(record['max_relative_norm_drift'], 1e-12)
        self.assertAlmostEqual(np.sum(fourier_populations(wave, grid)), 1., places=12)

    def test_spatial_solver_vs_dense_and_distorted_wave_weak_limit(self):
        grid = TransverseGrid(8e-10, 7e-10, 4, 4)
        basis = channel_basis(2, 2)
        energy = 60000*elementary_charge
        quanta = np.array([.04, .07])*elementary_charge
        xx, yy = np.meshgrid(2*np.pi*np.arange(4)/4, 2*np.pi*np.arange(4)/4)
        first = linear_displacement_block(basis, [.15, .08])*elementary_charge
        second = linear_displacement_block(basis, [-.06, .17])*elementary_charge
        segments = [ChannelSegment(5e-9, np.cos(xx)[..., None, None]*first),
                    ChannelSegment(4e-9, np.sin(yy)[..., None, None]*second)]
        initial = np.zeros((len(basis), 4, 4), dtype=complex)
        initial[0] = 1/np.sqrt(grid.lx_m*grid.ly_m)
        solver = SplitChannelPropagator(grid, energy, basis@quanta)
        generators = dense_generators(grid, energy, basis@quanta, segments)
        wave, record = solver.propagate(initial, segments, steps_per_segment=8)
        exact = dense_propagate(initial, generators)
        self.assertLess(np.linalg.norm(wave-exact)/np.linalg.norm(exact), 1e-7)
        zero, first_order = distorted_first_order(initial, generators)
        scaled_zero, scaled_first = distorted_wave_orders(grid, energy, basis@quanta, initial, segments, 16)
        self.assertLess(np.linalg.norm(scaled_zero-zero)/np.linalg.norm(initial), 1e-8)
        self.assertLess(np.linalg.norm(scaled_first-first_order)/np.linalg.norm(initial), 1e-8)
        weak, _ = solver.propagate(initial, segments, steps_per_segment=8, g=1e-3)
        error = np.linalg.norm(weak-zero-1e-3*first_order)/np.linalg.norm(initial)
        self.assertLess(error, 1e-9)
        self.assertLess(record['max_relative_norm_drift'], 1e-12)
        unchanged, _ = solver.propagate(initial, segments, g=0.)
        self.assertEqual(np.count_nonzero(unchanged[1:]), 0)

    def test_thermal_weights_keep_omitted_probability(self):
        basis = channel_basis(2, 2)
        weights = thermal_channel_weights(basis, np.array([.04, .07])*elementary_charge, 1000)
        self.assertGreater(np.sum(weights), 0)
        self.assertLess(np.sum(weights), .9)
        np.testing.assert_array_equal(thermal_channel_weights(basis, [.04, .07], 0), [1, 0, 0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
