import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy import constants
from yoshioka.thermal_dyson_words import (words_through_two, label, local_step,
    thermal_moment, gram_matrix)
from benchmarks.approximate_eels_repeated import dense_thermal_control


class ThermalDysonTests(unittest.TestCase):
    def test_local_two_insertions_against_independent_dense_exponential(self):
        words = words_through_two(2); lookup = {w: i for i, w in enumerate(words)}
        rng = np.random.default_rng(59173)
        waves = rng.normal(size=(len(words), 2, 2))+1j*rng.normal(size=(len(words), 2, 2))
        source = rng.normal(size=(2, 2, 2))+1j*rng.normal(size=(2, 2, 2))
        source = np.r_[source, source.conj()]; background = rng.normal(size=(2, 2))
        signed = np.array([.7, 1.1, -.7, -1.1]); energies = np.array([sum(signed[i] for i in w) for w in words])
        actual = local_step(waves, source, words, energies, background, .3, 1.)
        for iy, ix in np.ndindex(2, 2):
            matrix = np.diag(energies+background[iy, ix]).astype(complex)
            for j, w in enumerate(words[1:], 1): matrix[j, lookup[w[1:]]] += source[w[0], iy, ix]
            reference = expm(-.3j*matrix)@waves[:, iy, ix]
            np.testing.assert_allclose(actual[:, iy, ix], reference, rtol=3e-13, atol=3e-13)

    def test_identical_mode_factors_and_mixed_processes(self):
        nbar = np.array([.37, 1.2])
        self.assertAlmostEqual(thermal_moment(((0, -1), (0, -1), (0, 1), (0, 1)), nbar), 2*(nbar[0]+1)**2)
        self.assertAlmostEqual(thermal_moment(((0, 1), (0, -1), (0, 1), (0, -1)), nbar), 2*nbar[0]**2+nbar[0])
        self.assertAlmostEqual(thermal_moment(((0, -1), (0, 1), (0, 1), (0, -1)), nbar), 2*nbar[0]*(nbar[0]+1))
        self.assertAlmostEqual(thermal_moment(((0, -1), (1, -1), (0, 1), (1, 1)), nbar), (nbar[0]+1)*(nbar[1]+1))

    def test_thermal_gram_is_positive_and_distinct_net_changes_orthogonal(self):
        words = words_through_two(2); matrix = gram_matrix(words, 2, [.37, 1.2])
        self.assertGreater(np.min(np.linalg.eigvalsh(matrix)), -2e-13)
        # Two time orderings of distinct emissions have the same final state.
        a = words.index((0, 1)); b = words.index((1, 0))
        self.assertGreater(matrix[a, b], 0)
        self.assertEqual(matrix[words.index((0,)), words.index((1,))], 0)

    def test_finite_temperature_words_against_dense_coupled_channels(self):
        for temperature in (300, 1000):
            result = dense_thermal_control(temperature)
            self.assertTrue(result['passed'], result)
            json.dumps(result, allow_nan=False)
            errors = [r['thermal_complex_state_error'] for r in result['order_scaling']]
            self.assertGreater(errors[2]/errors[1], 7)
            self.assertGreater(errors[1]/errors[0], 7)

    def test_first_word_matches_existing_streamed_source_for_both_signs(self):
        from benchmarks.approximate_eels import prepare_background, covariance_path, PRIMITIVE
        from benchmarks.workflow_io import ResourceLimits
        from yoshioka.lattice_dynamics import RealSpaceForceConstants
        from yoshioka.bulk_thermal_source import covariance_integral, mode_group
        from yoshioka.fourier_depth_fields import complex_depth_field, fourier_depth_provider
        from yoshioka.bloch_source_windowed import source_exit_bloch
        from yoshioka.thermal_dyson_words import propagate_words
        model = RealSpaceForceConstants.load(PRIMITIVE)
        covariance, _ = covariance_integral(model, 300, 4)
        config = {'selected_repeats': 2, 'padding_angstrom': 12, 'selected_covariance_mesh': 4}
        settings = {'temperature_k': 300, 'grid_unit': 5, 'slices': 33}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory); path = covariance_path(directory, 300, 4); path.parent.mkdir()
            path.write_text(json.dumps({'covariance_m2': covariance.tolist()}))
            positions, reference, length, covariance, grid, bg, inc, _ = prepare_background(config, settings, directory, ResourceLimits(60, 2))
            omega, d, shift, _ = mode_group(model, positions, [grid.lx_m, grid.ly_m], [.17, -.21, .13], .5, reference)
            field = complex_depth_field(positions, covariance, grid, length, 33, directory/'source.npy', d, shift)
            provider = fourier_depth_provider(field, grid)
            words, coefficients, _, _ = propagate_words(grid, omega, np.tile(shift, (len(omega), 1)), length, 33,
                                                       provider, lambda i: bg[i, 0])
            for sign in (1, -1):
                at = provider if sign == 1 else lambda i: provider(i).conj()
                expected = source_exit_bloch(grid, 60000*constants.elementary_charge, sign*omega, length, 33,
                                             at, lambda i: inc[i], lambda i: bg[i, 0], sign*shift)
                indices = [words.index((j,)) for j in range(0 if sign == 1 else len(omega), len(omega) if sign == 1 else 2*len(omega))]
                np.testing.assert_allclose(coefficients[indices], expected, rtol=1e-12, atol=1e-16)

    def test_two_insertions_nonzero_bloch_shift_against_dense_fock_splitting(self):
        from yoshioka.thermal_dyson_words import propagate_words
        from yoshioka.free_propagation import TransverseGrid, electron_wavenumber
        from yoshioka.dynamical_source import full_transverse_vectors
        from yoshioka.elastic_propagation import electron_speed
        grid = TransverseGrid(1e-9, 1.3e-9, 3, 3)
        length = 1.5e-9; steps = 3; dz = length/steps
        energy = 60000*constants.elementary_charge; hv = constants.hbar*electron_speed(energy)
        quantum = .047*constants.elementary_charge; omega = np.array([quantum/constants.hbar])
        shift = np.array([[.37*2*np.pi/grid.lx_m, -.19*2*np.pi/grid.ly_m]])
        rng = np.random.default_rng(97351)
        source = (rng.normal(size=(steps, 1, 3, 3))+1j*rng.normal(size=(steps, 1, 3, 3)))*constants.elementary_charge
        background = rng.normal(size=(steps, 3, 3))*constants.elementary_charge
        words, actual, _, _ = propagate_words(grid, omega, shift, length, steps, lambda i: source[i], lambda i: background[i])
        dim = 3*9; a = np.diag(np.sqrt(np.arange(1, 3)), 1); ad = a.T
        reference = np.zeros((3, 3, 3, 3), complex); reference[0, 0] = 1/np.sqrt(grid.lx_m*grid.ly_m)
        q = full_transverse_vectors(grid)[None]+np.arange(3)[:, None, None]*shift[:, None]
        free = np.exp(-.5j*dz*np.sum(q*q, axis=-1).reshape(3, 3, 3)/(2*electron_wavenumber(energy)))
        for i in range(steps):
            reference = np.fft.ifft2(np.fft.fft2(reference, axes=(-2, -1))*free[None], axes=(-2, -1))
            diagonal = np.diag(np.repeat(np.arange(3)*quantum, 9)+np.tile(background[i].ravel(), 3))
            interaction = np.kron(ad, np.diag(source[i, 0].ravel()))+np.kron(a, np.diag(source[i, 0].conj().ravel()))
            augmented = np.kron(np.eye(3), diagonal)+np.kron(np.diag(np.ones(2), -1), interaction)
            reference = (expm(-1j*dz/hv*augmented)@reference.ravel()).reshape(3, 3, 3, 3)
            reference = np.fft.ifft2(np.fft.fft2(reference, axes=(-2, -1))*free[None], axes=(-2, -1))
        expected = np.fft.fft2(reference, axes=(-2, -1))*np.sqrt(grid.lx_m*grid.ly_m)/9
        reconstructed = np.zeros_like(expected)
        for w, amplitude in zip(words, actual):
            bath = np.eye(3)
            for letter in w: bath = bath@(ad if letter == 0 else a)
            reconstructed[len(w)] += bath[:, 0, None, None]*amplitude[None]
        np.testing.assert_allclose(reconstructed, expected, rtol=1e-10, atol=2e-15)


if __name__ == '__main__': unittest.main()
