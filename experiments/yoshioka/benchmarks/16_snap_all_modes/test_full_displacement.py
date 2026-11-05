"""Independent full-3D dense checks for the external scientific runner."""
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from scipy import constants
from scipy.linalg import expm

from run import FullDisplacementAction
from yoshioka.leading_reference import DenseDisplacedMaterial


class FullDisplacementTests(unittest.TestCase):
    def bath(self, modes):
        rng = np.random.default_rng(51342)
        return dict(cell_m=np.diag([5.43e-10, 3.84e-10, 3.84e-10]),
                    positions_m=np.array([[.68, .96, .96], [4.75, .96, 2.88],
                                          [2.04, 2.88, .96], [3.39, 2.88, 2.88]]) * 1e-10,
                    displacements_m=rng.normal(scale=2e-12, size=(4, 3, modes)),
                    omega_rad_s=np.linspace(25, 65, modes) * 1e-3 * constants.e / constants.hbar)

    def compare(self, modes, fields, steps):
        bath = self.bath(modes)
        cfg = dict(n=3, steps=steps, field_points=fields, working_dimension=32,
                   p=2, padding_m=1.2e-9, electron_energy_ev=60000.)
        declared = SimpleNamespace(cell_m=bath['cell_m'], positions_m=bath['positions_m'],
                                   displacement_m=bath['displacements_m'], omega_rad_s=bath['omega_rad_s'],
                                   inactive_covariance_m2=np.zeros((4, 3, 3)))
        reference = DenseDisplacedMaterial(declared, p=2, n=3, steps=fields)
        matrices = reference.hamiltonians(method='fock', working_dimension=32)
        # Independently DFT the dense z samples and evaluate at the requested
        # midpoints. This tests folding when the field grid exceeds slice count.
        length = reference.length
        frequency = np.rint(np.fft.fftfreq(fields) * fields).astype(int)
        sample_z = (np.arange(fields) + .5) * length / fields
        target_z = (np.arange(steps) + .5) * length / steps
        sample_fourier = np.exp(-2j*np.pi*frequency[:, None]*sample_z[None, :]/length) / fields
        target_fourier = np.exp(2j*np.pi*target_z[:, None]*frequency[None, :]/length)
        independent = np.einsum('ts,sij->tij', (target_fourier @ sample_fourier).real, matrices)
        rng = np.random.default_rng(781)
        wave = rng.normal(size=reference.shape) + 1j*rng.normal(size=reference.shape)
        wave /= np.linalg.norm(wave)
        with tempfile.TemporaryDirectory() as folder:
            h = FullDisplacementAction(bath, cfg, Path(folder))
            for step in range(steps):
                actual = h.apply(wave, step)
                expected = (independent[step] @ wave.ravel()).reshape(h.shape)
                error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
                self.assertLess(error, 2e-12)
            if modes == 2:
                expected = expm(-1j*h.dz/h.hv*independent[0]) @ wave.ravel()
                actual = h.exponential_action(wave, 0)
                self.assertLess(np.linalg.norm(actual.ravel() - expected), 2e-12)
                self.assertLess(abs(np.linalg.norm(actual)**2 - 1), 2e-12)

    def test_all_nine_modes_against_independent_dense_fock_reference(self):
        self.compare(9, 5, 5)

    def test_longitudinal_fields_fold_to_independent_midpoints(self):
        self.compare(2, 9, 3)


if __name__ == '__main__':
    unittest.main()
