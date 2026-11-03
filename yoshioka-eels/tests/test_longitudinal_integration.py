"""Unit, normalization, and independent numerical-source integration checks."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import constants
from scipy.integrate import quad

from yoshioka.electron_phonon import (FirstOrderSourcePropagator, LobatoSilicon,
                                     numerical_transition_source, transverse_modes)
from yoshioka.free_propagation import TransverseGrid
from yoshioka.longitudinal_integration import (finite_window_angular,
    finite_window_cycles, longitudinal_nodes, anisotropic_nodes,
    numerical_source_contraction, distorted_source_exit, _depth_fourier)


class LongitudinalIntegrationTests(unittest.TestCase):
    def test_window_complex_phase_against_direct_spatial_integral(self):
        length = 19.9621902691195e-9
        kappa = np.array([-.73, 0., 1.17])*2*np.pi/length
        z = (np.arange(100001)+.5)*length/100001
        expected = np.mean(np.exp(1j*kappa[:, None]*z), axis=1)*length
        np.testing.assert_allclose(finite_window_angular(kappa, length), expected,
                                   rtol=3e-10, atol=1e-22)
        np.testing.assert_allclose(finite_window_angular(-kappa, length), expected.conj(),
                                   rtol=3e-10, atol=1e-22)

    def test_angular_and_cycles_first_zero_are_distinct_and_consistent(self):
        length = 19.9621902691195e-9
        self.assertLess(abs(finite_window_angular(2*np.pi/length, length))/length, 1e-15)
        self.assertLess(abs(finite_window_cycles(1/length, length))/length, 1e-15)
        self.assertGreater(abs(finite_window_angular(1/length, length))/length, .95)
        kappa = np.array([-.4, .2, .8])*2*np.pi/length
        np.testing.assert_allclose(finite_window_cycles(kappa/(2*np.pi), length),
                                   finite_window_angular(kappa, length), rtol=2e-15)

    def test_window_parseval_with_explicit_finite_range_tail_bound(self):
        # Integrate x=kappa*L/2. The tail outside |x|<A is bounded by
        # (2/pi)/A because sin(x)^2 <= 1, so this tests the continuum measure.
        endpoint = 200*np.pi
        integral = 2/np.pi*quad(lambda x: np.sinc(x/np.pi)**2, 0, endpoint,
            points=np.arange(1, 200)*np.pi, limit=1000, epsabs=1e-12)[0]
        self.assertLessEqual(integral, 1+1e-12)
        self.assertLess(1-integral, 2/(np.pi*endpoint))

    def test_independent_axis_mesh_weights_and_random_stratum_support(self):
        q, weights = anisotropic_nodes((2, 3, 8), (.5, 0., .5))
        self.assertEqual(q.shape, (48, 3))
        self.assertAlmostEqual(float(np.sum(weights)), 1.)
        for axis in range(3):
            self.assertLess(abs(np.sum(weights*np.exp(2j*np.pi*q[:, axis]))), 1e-15)
        z, weight = longitudinal_nodes(32, seed=8102)
        np.testing.assert_array_equal(z, longitudinal_nodes(32, seed=8102)[0])
        self.assertTrue(np.all(np.floor((z+.5)*32) == np.arange(32)))
        self.assertAlmostEqual(float(np.sum(weight)), 1.)
        self.assertFalse(np.array_equal(z, longitudinal_nodes(32, seed=8103)[0]))
        with self.assertRaises(ValueError): longitudinal_nodes(32, shift=.5, seed=8102)

    def test_numerical_kernel_contraction_matches_stepper_and_scan(self):
        grid = TransverseGrid(1.1e-9, .8e-9, 64, 64)
        energy = 60000*constants.elementary_charge
        shift = np.array([.21e9, -.18e9])
        Q, indices, _ = transverse_modes(grid, energy, 6, shift)
        positions = np.array([[.21e-9, .17e-9, .8e-9], [.67e-9, .43e-9, 1.2e-9]])
        rng = np.random.default_rng(135)
        d = (rng.normal(size=(2, 3, 3))+1j*rng.normal(size=(2, 3, 3)))*1e-12
        omega = np.array([20., 45., 70.])*1e-3*constants.elementary_charge/constants.hbar
        cov = np.diag([1.8e-23, 1.9e-23, 2.e-23])
        propagator = FirstOrderSourcePropagator(grid, energy, omega, indices, shift)
        potential = LobatoSilicon.from_abtem_table()
        _, source = numerical_transition_source(potential, positions, d, cov, Q, grid, 2e-9, 256)
        step, _ = propagator.propagate(source, 2e-9)
        scan = propagator.propagate_scan(source, 2e-9)
        contracted = numerical_source_contraction(potential, positions, d, cov, Q,
            propagator, 2e-9, 256)
        np.testing.assert_allclose(contracted, step, rtol=5e-13, atol=1e-18)
        np.testing.assert_allclose(contracted, scan, rtol=5e-13, atol=1e-18)

    def test_bounded_fft_source_matches_existing_depth_dependent_bloch_action(self):
        from yoshioka.bloch_source_windowed import source_exit_bloch
        from yoshioka.fourier_depth_fields import complex_depth_field, fourier_depth_provider
        grid = TransverseGrid(1.1e-9, .8e-9, 7, 7)
        energy = 60000*constants.elementary_charge; length = 2e-9; steps = 19
        positions = np.array([[.21e-9, .17e-9, .8e-9], [.67e-9, .43e-9, 1.2e-9]])
        d = np.array([[[1e-12], [2e-12j], [.7e-12]], [[-2e-12], [.4e-12j], [1e-12]]])
        omega = np.array([.045*constants.elementary_charge/constants.hbar]); shift = [.21e9, -.18e9]
        cov = np.eye(3)*1.9e-23
        rng = np.random.default_rng(529)
        incident = (1+.1*rng.normal(size=(steps, *grid.shape)))*np.exp(.1j*rng.normal(size=(steps, *grid.shape)))/np.sqrt(grid.lx_m*grid.ly_m)
        background = rng.normal(size=(steps, 1, *grid.shape))*.1*constants.elementary_charge
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            old = complex_depth_field(positions, cov, grid, length, steps, folder/'old.npy', d, shift)
            new = _depth_fourier(positions, cov, grid, length, steps, folder/'new.npy', d, shift)
            np.testing.assert_allclose(new, old, rtol=2e-14, atol=1e-34)
            provider = fourier_depth_provider(old, grid)
            expected = source_exit_bloch(grid, energy, omega, length, steps, provider,
                lambda i: incident[i], lambda i: background[i, 0], shift)
            actual = distorted_source_exit(grid, energy, omega, length, steps,
                new, incident, background, shift)
            np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=1e-18)


if __name__ == '__main__':
    unittest.main()
