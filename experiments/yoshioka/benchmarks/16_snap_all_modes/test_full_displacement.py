"""Independent full-3D dense checks for the external scientific runner."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from scipy import constants
from scipy.linalg import expm

from run import FullDisplacementAction, preparation_contract
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

    def test_process_workers_reproduce_complete_fields_and_actions(self):
        bath = self.bath(9)
        cfg = dict(n=3, steps=3, field_points=9, working_dimension=8,
                   p=2, padding_m=1.2e-9, electron_energy_ev=60000.)
        rng = np.random.default_rng(9182)
        shape = (55, 3, 3)
        wave = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        wave /= np.linalg.norm(wave)
        # Thirteen canonical transfers require four tiles. Every worker count
        # therefore exercises assembly across tiles and conjugate partners.
        with tempfile.TemporaryDirectory() as folder:
            reference_fields, reference_actions = None, None
            for workers in (1, 2, 4):
                with self.subTest(field_workers=workers):
                    action = FullDisplacementAction(
                        bath, dict(cfg, field_workers=workers),
                        Path(folder) / str(workers))
                    fields = np.array(action.fields)
                    actions = np.array([action.apply(wave, step)
                                        for step in range(cfg['steps'])])
                    self.assertEqual(fields.shape, (3, 25, 55, 55))
                    self.assertTrue(np.isfinite(fields).all())
                    # An omitted transfer group must not pass through a loose
                    # absolute tolerance: each stored group has nonzero fields.
                    self.assertTrue((np.linalg.norm(fields, axis=(-2, -1)) > 0).all())
                    if workers == 1:
                        reference_fields, reference_actions = fields, actions
                    else:
                        field_error = np.linalg.norm(fields - reference_fields) / np.linalg.norm(reference_fields)
                        action_error = np.linalg.norm(actions - reference_actions) / np.linalg.norm(reference_actions)
                        self.assertLess(field_error, 1e-13)
                        self.assertLess(action_error, 1e-13)

    def test_completed_tiles_are_checked_and_reused_during_reassembly(self):
        bath = self.bath(2)
        cfg = dict(n=3, steps=3, field_points=9, working_dimension=8,
                   p=2, padding_m=1.2e-9, electron_energy_ev=60000.,
                   field_workers=1)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            action = FullDisplacementAction(bath, cfg, folder)
            expected = np.array(action.fields)
            del action
            chunks = sorted((folder / 'chunks').iterdir())
            self.assertTrue(chunks)
            before = {file.name: (file.read_bytes(), file.stat().st_mtime_ns)
                      for file in chunks}
            (folder / 'channel_fields.npy').unlink()
            (folder / 'manifest.json').unlink()
            # Reassembly must load the checked completed chunks without
            # recomputing oscillator blocks or rewriting any chunk artifacts.
            with patch('run.phase_blocks', side_effect=AssertionError('completed tile was recomputed')):
                action = FullDisplacementAction(bath, cfg, folder)
            np.testing.assert_array_equal(np.asarray(action.fields), expected)
            after = {file.name: (file.read_bytes(), file.stat().st_mtime_ns)
                     for file in chunks}
            self.assertEqual(before, after)
            del action
            (folder / 'channel_fields.npy').unlink()
            (folder / 'manifest.json').unlink()
            damaged = next((folder / 'chunks').glob('*.npy'))
            with damaged.open('r+b') as stream:
                stream.seek(-1, 2)
                byte = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([byte[0] ^ 1]))
            with self.assertRaisesRegex(ValueError, r'field tile identity/content mismatch'):
                FullDisplacementAction(bath, cfg, folder)
            self.assertFalse((folder / 'manifest.json').exists())
            self.assertFalse((folder / 'channel_fields.npy').exists())

    def test_all_nine_modes_against_independent_dense_fock_reference(self):
        self.compare(9, 5, 5)

    def test_longitudinal_fields_fold_to_independent_midpoints(self):
        self.compare(2, 9, 3)


class PreparationContractTests(unittest.TestCase):
    def config(self):
        return dict(
            system='Si[110]', specimen_repetitions=[1, 1, 1], mass_u=28.085,
            fit_supercell=[4, 6, 6], harmonic_cutoff_angstrom=9.85,
            fit_sigma_angstrom=0.001, training_pairs=8, validation_pairs=2,
            fit_seed=20261105,
            input_sha256={'Si_110_temp300.lmp': 'specimen-digest',
                          'SNAPotential.snapcoeff': 'coeff-digest',
                          'SNAPotential.snapparam': 'param-digest'},
            n=97, padded_interaction_mesh=193, full_grid=97,
            grid_resolution='confirmed electron grid',
            field_points=769, steps=11, working_dimension=32, p=2,
            padding_m=1.2e-9, electron_energy_ev=60000.,
            incident_beam='exact [110] plane wave',
            electron_potential='Lobato Si independent atom rigid ion',
            temperature_k=10., detector_mrad=[0., 30.],
            energy_axis_mev=[-180., 180., 0.05], fwhm_mev=2.,
            initial_weight_threshold=1e-14,
            maximum_omitted_thermal_probability=1e-12,
            maximum_rss_bytes=64 * 1024**3, maximum_seconds=14400,
            field_workers=1, resource_request={'cpus': 16, 'memory': '64G'})

    def test_electron_and_worker_changes_reuse_the_phonon_contract(self):
        cfg = self.config()
        original = deepcopy(cfg)
        changed = deepcopy(cfg)
        changed.update(
            n=11, padded_interaction_mesh=21, full_grid=11,
            grid_resolution='updated electron grid', field_points=193,
            steps=3, working_dimension=16, p=1, padding_m=2e-9,
            electron_energy_ev=80000., incident_beam='another plane wave',
            electron_potential='another electron potential', temperature_k=20.,
            detector_mrad=[1., 20.], energy_axis_mev=[-100., 100., 0.1],
            fwhm_mev=3., initial_weight_threshold=1e-12,
            maximum_omitted_thermal_probability=1e-10,
            maximum_rss_bytes=8 * 1024**3, maximum_seconds=3600,
            field_workers=4, resource_request={'cpus': 4, 'memory': '8G'})
        self.assertEqual(preparation_contract(cfg), preparation_contract(changed))
        self.assertEqual(cfg, original, 'constructing the contract must not mutate inputs')

    def test_fit_material_mass_and_specimen_changes_invalidate_the_contract(self):
        cfg = self.config()
        contract = preparation_contract(cfg)
        changes = dict(
            system='Ge[110]', specimen_repetitions=[2, 1, 1], mass_u=29.,
            fit_supercell=[5, 6, 6], harmonic_cutoff_angstrom=9.,
            fit_sigma_angstrom=0.002, training_pairs=9, validation_pairs=3,
            fit_seed=20261106)
        for key, value in changes.items():
            with self.subTest(setting=key):
                changed = deepcopy(cfg)
                changed[key] = value
                self.assertNotEqual(contract, preparation_contract(changed))
        for name in cfg['input_sha256']:
            with self.subTest(input_file=name):
                changed = deepcopy(cfg)
                changed['input_sha256'][name] = 'changed-digest'
                self.assertNotEqual(contract, preparation_contract(changed))


if __name__ == '__main__':
    unittest.main()
