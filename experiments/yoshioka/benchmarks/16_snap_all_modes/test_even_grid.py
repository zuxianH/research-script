"""Even electron grids checked against independent finite-beam references."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy import constants
from scipy.linalg import expm

from run import FullDisplacementAction, single_occupation_phase_blocks
from yoshioka.coupled_channels import channel_basis
from yoshioka.leading_reference import DenseDisplacedMaterial
from yoshioka.projected_material_action import phase_blocks


class EvenElectronGridTests(unittest.TestCase):
    def bath(self, modes):
        rng = np.random.default_rng(51342)
        return dict(
            cell_m=np.diag([5.43e-10, 3.84e-10, 3.84e-10]),
            positions_m=np.array([[.68, .96, .96], [4.75, .96, 2.88],
                                  [2.04, 2.88, .96], [3.39, 2.88, 2.88]]) * 1e-10,
            displacements_m=rng.normal(scale=2e-12, size=(4, 3, modes)),
            omega_rad_s=np.linspace(25, 65, modes) * 1e-3 * constants.e / constants.hbar)

    def config(self, n, steps, fields):
        return dict(n=n, steps=steps, field_points=fields, working_dimension=32,
                    p=1, padding_m=1.2e-9, electron_energy_ev=60000.)

    def test_factorized_single_occupation_blocks_match_generic_working_fock_product(self):
        rng = np.random.default_rng(62583)
        for modes, scales in ((9, [0., 1e-8, .01, .2, 1., 4., 9.]),
                              (93, [0., 1e-8, .01, .05, .2])):
            with self.subTest(modes=modes):
                basis = channel_basis(modes, 1)
                xi = rng.normal(size=(len(scales), modes)) * np.asarray(scales)[:, None]
                expected = phase_blocks(basis, xi, dimension=32)
                actual = single_occupation_phase_blocks(basis, xi, dimension=32)
                self.assertEqual(actual.shape, (len(scales), modes + 1, modes + 1))
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
                self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), 2e-12)

    def test_threaded_even_grid_actions_and_propagation_match_serial(self):
        bath = self.bath(3)
        cfg = self.config(n=2, steps=5, fields=5)
        rng = np.random.default_rng(27463)
        wave = rng.normal(size=(4, 2, 2)) + 1j * rng.normal(size=(4, 2, 2))
        wave /= np.linalg.norm(wave)
        with tempfile.TemporaryDirectory() as folder:
            serial = FullDisplacementAction(bath, cfg, Path(folder) / 'serial')
            expected_actions = [serial.apply(wave, step) for step in range(5)]
            expected_waves = [serial.exponential_action(wave, step) for step in range(5)]
            for fft_threads in (1, 2):
                with self.subTest(hamiltonian_threads=2, fft_threads=fft_threads):
                    threaded = FullDisplacementAction(
                        bath, dict(cfg, hamiltonian_threads=2, fft_threads=fft_threads),
                        Path(folder) / f'threaded_fft_{fft_threads}')
                    try:
                        for step in range(5):
                            actual = threaded.apply(wave, step)
                            self.assertLess(np.linalg.norm(actual - expected_actions[step]) /
                                            np.linalg.norm(expected_actions[step]), 2e-12)
                            propagated = threaded.exponential_action(wave, step)
                            self.assertLess(np.linalg.norm(propagated - expected_waves[step]), 2e-12)
                            self.assertLess(abs(np.linalg.norm(propagated)**2 - 1), 2e-12)
                    finally:
                        threaded.thread_pool.shutdown()

    def test_two_beam_grid_matches_restricted_independent_dense_hamiltonians(self):
        bath = self.bath(3)
        cfg = self.config(n=2, steps=5, fields=5)
        declared = SimpleNamespace(
            cell_m=bath['cell_m'], positions_m=bath['positions_m'],
            displacement_m=bath['displacements_m'], omega_rad_s=bath['omega_rad_s'],
            inactive_covariance_m2=np.zeros((4, 3, 3)))
        reference = DenseDisplacedMaterial(declared, p=1, n=3, steps=5)
        full = reference.hamiltonians(method='fock', working_dimension=32)
        # The dense reference uses beams {-1, 0, 1}. Selecting its first two
        # rows and columns independently defines the even grid {-1, 0}².
        spatial = np.array([0, 1, 3, 4])
        retained = (np.arange(len(reference.basis))[:, None] * 9 + spatial).ravel()
        matrices = full[:, retained][:, :, retained]
        rng = np.random.default_rng(91837)
        wave = rng.normal(size=(4, 2, 2)) + 1j * rng.normal(size=(4, 2, 2))
        wave /= np.linalg.norm(wave)
        with tempfile.TemporaryDirectory() as folder:
            action = FullDisplacementAction(bath, cfg, Path(folder))
            self.assertEqual(action.shape, wave.shape)
            self.assertEqual(action.pad, 3)
            for step, matrix in enumerate(matrices):
                with self.subTest(step=step):
                    expected = matrix @ wave.ravel()
                    actual = action.apply(wave, step).ravel()
                    self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), 2e-12)
                    propagated = action.exponential_action(wave, step)
                    expected_propagated = expm(-1j * action.dz / action.hv * matrix) @ wave.ravel()
                    self.assertLess(np.linalg.norm(propagated.ravel() - expected_propagated), 2e-12)
                    self.assertLess(abs(np.linalg.norm(propagated)**2 - 1), 2e-12)

    def test_fifty_beam_grid_preserves_zero_beam_and_unwrapped_edge_transfers(self):
        bath = self.bath(2)
        cfg = self.config(n=50, steps=1, fields=1)
        cfg['working_dimension'] = 8
        with tempfile.TemporaryDirectory() as folder:
            # A single longitudinal sample and three channels keep this
            # requested-size transverse check below 2 MiB of stored fields.
            action = FullDisplacementAction(bath, cfg, Path(folder), tile=256)
            self.assertEqual(action.shape, (3, 50, 50))
            self.assertEqual(action.pad, 99)
            self.assertEqual(action.fields.shape, (1, 99**2, 3, 3))
            zero_beams = np.argwhere(np.all(action.qxy == 0, axis=-1))
            np.testing.assert_array_equal(zero_beams, [[25, 25]])
            np.testing.assert_allclose(action.qxy[0, :, 0] * bath['cell_m'][0, 0] / (2 * np.pi),
                                       np.arange(-25, 25), rtol=1e-15, atol=1e-14)
            np.testing.assert_allclose(action.qxy[:, 0, 1] * bath['cell_m'][1, 1] / (2 * np.pi),
                                       np.arange(-25, 25), rtol=1e-15, atol=1e-14)

            # Start at opposite corner beams in turn. The direct Galerkin
            # matrix column uses physical beam differences through ±49,
            # independently of FFT embedding or reciprocal wraparound.
            field = np.asarray(action.fields[0]).reshape(99, 99, 3, 3)
            rows, columns = np.indices((50, 50))
            for source_y, source_x in ((0, 0), (49, 49)):
                with self.subTest(source=(source_y, source_x)):
                    wave = np.zeros(action.shape, complex)
                    wave[0, source_y, source_x] = 1.
                    expected = field[(rows - source_y) % 99,
                                     (columns - source_x) % 99, :, 0].transpose(2, 0, 1).copy()
                    expected[0, source_y, source_x] += action.kinetic_j[source_y, source_x]
                    actual = action.apply(wave, 0)
                    self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), 2e-12)


if __name__ == '__main__':
    unittest.main()
