from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.constants import elementary_charge, hbar

from yoshioka.dynamical_source import plane_field, propagate_incident, propagate_source_block
from yoshioka.elastic_propagation import electron_speed
from yoshioka.free_propagation import TransverseGrid


class DynamicalSourceTests(unittest.TestCase):
    def test_zero_displacement_recovers_independent_spatial_elastic_limit(self):
        from scipy.linalg import expm
        from scipy.constants import hbar
        from yoshioka.dynamical_source import full_transverse_vectors
        from yoshioka.free_propagation import electron_wavenumber
        grid = TransverseGrid(1e-9, 1.2e-9, 3, 3)
        energy, length = 60000*elementary_charge, 23e-9
        velocity = electron_speed(energy)
        field = (np.array([[.2, -.7, .3], [.1, .4, -.2], [-.5, .6, -.1]])*elementary_charge)
        eye = np.eye(9).reshape(9, 3, 3)
        kinetic_values = np.sum(full_transverse_vectors(grid)**2, axis=1).reshape(grid.shape)/(2*electron_wavenumber(energy))
        kinetic_matrix = np.fft.ifft2(np.fft.fft2(eye)*kinetic_values).reshape(9, 9).T
        generator = kinetic_matrix+np.diag(field.ravel()/(hbar*velocity))
        exact = expm(-1j*length*generator) @ (np.ones(9)/np.sqrt(grid.lx_m*grid.ly_m))
        errors = []
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for steps in (37, 111):
                background = np.lib.format.open_memmap(folder/f'bg{steps}.npy', mode='w+', dtype=complex, shape=(1, 9, steps))
                background[0] = (np.fft.fft2(field)/9).ravel()[:, None]
                wave, _, diagnostic = propagate_incident(grid, energy, background, length, folder/f'h{steps}.npy')
                errors.append(np.linalg.norm(wave.ravel()-exact)/np.linalg.norm(exact))
                self.assertLess(diagnostic['max_incident_norm_drift'], 1e-12)
            self.assertLess(errors[1], errors[0]/8)
            self.assertLess(errors[1], 1e-6)
            zero_sources = plane_field(np.array([[1e-10, 2e-10, 1e-9]]), np.zeros((3, 3)),
                grid, 2e-9, 9, folder/'zero.npy', np.zeros((1, 3, 1)))
            self.assertEqual(np.count_nonzero(zero_sources), 0)

    def test_uniform_elastic_and_source_complex_phase_with_signed_energy(self):
        grid = TransverseGrid(1e-9, 1e-9, 3, 3)
        length, steps = 17e-9, 9
        energy = 60000*elementary_charge
        potential = -.7*elementary_charge
        coupling = .031*elementary_charge
        velocity = electron_speed(energy)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            background = np.lib.format.open_memmap(directory/'bg.npy', mode='w+', dtype=complex, shape=(1, 9, steps))
            background[:] = 0
            background[:, 0] = potential
            sources = np.lib.format.open_memmap(directory/'source.npy', mode='w+', dtype=complex, shape=(1, 9, steps))
            sources[:] = 0
            sources[:, 0] = coupling
            exit_wave, history, diagnostic = propagate_incident(grid, energy, background, length, directory/'history.npy')
            expected_incident = np.exp(-1j*potential*length/(hbar*velocity))/1e-9
            np.testing.assert_allclose(exit_wave, expected_incident, rtol=1e-13)
            self.assertLess(diagnostic['max_incident_norm_drift'], 1e-13)
            for sign in (-1, 0, 1):
                omega = sign*.06*elementary_charge/hbar
                result, _ = propagate_source_block(grid, energy, [omega], background, sources, history, length)
                expected = -1j*coupling*length/(hbar*velocity)*np.exp(-1j*potential*length/(hbar*velocity)-.5j*omega*length/velocity)*np.sinc(omega*length/(2*np.pi*velocity))
                self.assertLess(abs(result[0, 0, 0]-expected), 1e-14)
                self.assertLess(np.linalg.norm(result[0].ravel()[1:]), 1e-14)
            sources[:] = 0
            result, _ = propagate_source_block(grid, energy, [.06*elementary_charge/hbar], background, sources, history, length)
            self.assertEqual(np.count_nonzero(result), 0)

    def test_actual_atomic_field_has_checked_real_pairs(self):
        from yoshioka.dynamical_source import real_slice
        grid = TransverseGrid(5.4e-10, 3.8e-10, 9, 9)
        positions = np.array([[1.234e-10, 2.345e-10, 9e-10]])
        displacement = np.array([[[.01e-10], [.02e-10], [.03e-10]]])
        with tempfile.TemporaryDirectory() as folder:
            values = plane_field(positions, np.eye(3)*1.8e-23, grid, 2e-9, 65, Path(folder)/'source.npy', displacement)
            for index in (0, 13, 29, 64):
                _, residual = real_slice(values, index, grid)
                self.assertLess(residual, 1e-13)

    def test_fixed_periodic_bath_transverse_replication_preserves_mode_weights(self):
        from yoshioka.electron_phonon import full_space_born, LobatoSilicon
        # Enlarge only the electron representation of the *same* periodic
        # displacement field. This is distinct from adding new bath q modes.
        lx, ly = 1.1e-9, .8e-9
        positions = np.array([[.23e-9, .31e-9, 1.1e-9], [.71e-9, .52e-9, 1.3e-9]])
        displacement = np.array([[[.005e-10], [.007e-10], [.003e-10]], [[-.003e-10], [.004e-10], [.006e-10]]])
        covariance = np.eye(3)*2e-23
        omega = np.array([.06*elementary_charge/hbar])
        q = np.array([[0., 0.], [2*np.pi/lx, 0.], [0., 2*np.pi/ly], [2*np.pi/lx, -2*np.pi/ly]])
        potential = LobatoSilicon.from_abtem_table()
        original = full_space_born(potential, positions, displacement, covariance, q, omega,
            60000*elementary_charge, lx*ly, 3e-9)
        shifts = np.array([[0,0,0], [lx,0,0], [0,ly,0], [lx,ly,0]])
        repeated = (positions[None]+shifts[:,None]).reshape(-1,3)
        extended = full_space_born(potential, repeated, np.tile(displacement,(4,1,1)), covariance, q, omega,
            60000*elementary_charge, 4*lx*ly, 3e-9)
        self.assertLess(np.linalg.norm(original-extended)/np.linalg.norm(original), 1e-13)
        new_q = np.array([[np.pi/lx,0.], [0.,np.pi/ly], [np.pi/lx,np.pi/ly]])
        forbidden = full_space_born(potential, repeated, np.tile(displacement,(4,1,1)), covariance, new_q, omega,
            60000*elementary_charge, 4*lx*ly, 3e-9)
        self.assertLess(np.linalg.norm(forbidden)/np.linalg.norm(original), 1e-13)


if __name__ == '__main__':
    unittest.main()
