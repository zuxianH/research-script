"""Independent full-3D dense checks for the external scientific runner."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from scipy import constants
from scipy.linalg import expm

from run import (FullDisplacementAction, preparation_contract,
                 prune_completed_checkpoints, save_exit_probabilities)
from yoshioka.channel_spectra import broaden_transition_lines, transition_lines
from yoshioka.leading_reference import DenseDisplacedMaterial
from yoshioka.provenance import sha256


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


class ProbabilityOutputTests(unittest.TestCase):
    def test_saved_probabilities_reproduce_thermal_detector_spectra(self):
        rng = np.random.default_rng(60291)
        basis = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]])
        initial_indices = np.array([0, 2])
        waves = rng.normal(size=(2, len(basis), 3, 3)) + 1j * rng.normal(size=(2, len(basis), 3, 3))
        waves /= np.linalg.norm(waves.reshape(2, -1), axis=1)[:, None, None, None]
        mode_quanta_j = np.array([20., 35.]) * 1e-3 * constants.e
        thermal_weights = np.array([.6, .12, .2, .03, .03, .01])
        detector_masks = np.array([np.ones((3, 3), dtype=bool),
                                   [[False, True, False], [True, True, True], [False, True, False]]])
        metadata = dict(initial_indices=initial_indices, basis=basis,
                        mode_quanta_j=mode_quanta_j, detector_masks=detector_masks,
                        thermal_weights=thermal_weights,
                        executed_initial_weights=thermal_weights[initial_indices])
        expected_lines = transition_lines(waves, initial_indices, basis, mode_quanta_j,
                                          thermal_weights[initial_indices], detector_masks)
        axis = np.linspace(-180., 180., 7201)
        expected_spectra = broaden_transition_lines(expected_lines, axis, 2.)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'exit_probabilities.npz'
            save_exit_probabilities(path, waves, **metadata)
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), {'exit_probabilities', *metadata})
                probability = saved['exit_probabilities']
                self.assertEqual(probability.dtype, np.dtype('float64'))
                self.assertEqual(probability.shape, waves.shape)
                self.assertEqual(probability.nbytes * 2, waves.nbytes)
                self.assertTrue(np.isfinite(probability).all())
                self.assertTrue((probability >= 0).all())
                np.testing.assert_array_equal(probability, abs(waves)**2)
                np.testing.assert_allclose(probability.sum(axis=(1, 2, 3)), 1., rtol=0., atol=5e-16)
                for name, value in metadata.items():
                    np.testing.assert_array_equal(saved[name], value)
                self.assertFalse(any(np.iscomplexobj(saved[name]) for name in saved.files))
                detector_probability = np.einsum('ifyx,dyx->ifd', probability,
                                                  saved['detector_masks']) * saved['executed_initial_weights'][:, None, None]
                np.testing.assert_allclose(detector_probability, expected_lines['probability'], rtol=1e-15, atol=0.)
                # The full detector preserves each unnormalized thermal weight;
                # a second thermal multiplication would fail this check.
                np.testing.assert_allclose(detector_probability[:, :, 0].sum(axis=1),
                                           saved['thermal_weights'][saved['initial_indices']], rtol=1e-15, atol=0.)
                energy_mev = saved['basis'] @ saved['mode_quanta_j'] / (1e-3 * constants.e)
                lines = dict(probability=detector_probability,
                             energy_loss_mev=energy_mev[None, :] - energy_mev[saved['initial_indices'], None],
                             returning_state=saved['initial_indices'][:, None] == np.arange(len(saved['basis']))[None, :])
                np.testing.assert_array_equal(lines['energy_loss_mev'], expected_lines['energy_loss_mev'])
                np.testing.assert_array_equal(lines['returning_state'], expected_lines['returning_state'])
                self.assertTrue(lines['returning_state'].any())
                self.assertTrue((~lines['returning_state']).any())
                spectra = broaden_transition_lines(lines, axis, 2.)
                for name in ('components_per_mev', 'inclusive_per_mev', 'changed_per_mev', 'returning_per_mev'):
                    np.testing.assert_allclose(spectra[name], expected_spectra[name], rtol=1e-15, atol=0.)

    def checkpoint(self, folder, index=2, **overrides):
        name = f'initial_{index}_slice_3.npy'
        wave_path = folder / name
        np.save(wave_path, np.array([1.+2.j]))
        state = dict(contract='unchanged-contract', completed_slices=3, total_slices=3,
                     wave_file=name, wave_sha256=sha256(wave_path), max_norm_drift=1e-15)
        state.update(overrides)
        state_path = folder / f'initial_{index}.json'
        state_path.write_text(json.dumps(state))
        return wave_path, state_path, state

    def test_cleanup_retains_metadata_and_removes_only_completed_owned_wave(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            folder = output / 'checkpoints'
            folder.mkdir()
            probability_file = output / 'exit_probabilities.npz'
            np.savez_compressed(probability_file, exit_probabilities=np.array([1.]))
            wave_path, state_path, before = self.checkpoint(folder)
            unrelated = folder / 'unrelated.npy'
            np.save(unrelated, np.array([3.+4.j]))
            prune_completed_checkpoints(folder, [2], probability_file)
            self.assertFalse(wave_path.exists())
            self.assertTrue(unrelated.exists())
            after = json.loads(state_path.read_text())
            for name in ('contract', 'completed_slices', 'total_slices', 'max_norm_drift'):
                self.assertEqual(after[name], before[name])
            self.assertIsNone(after['wave_file'])
            self.assertIsNone(after['wave_sha256'])
            self.assertFalse(after['wave_retained'])
            self.assertEqual(after['final_probability_file'], '../exit_probabilities.npz')

    def test_cleanup_rejects_invalid_checkpoints_before_removing_any_wave(self):
        cases = dict(incomplete=dict(completed_slices=2),
                     corrupt=dict(wave_sha256='incorrect-digest'),
                     foreign_name=dict(wave_file='../foreign.npy'))
        for case, overrides in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                folder = output / 'checkpoints'
                folder.mkdir()
                probability_file = output / 'exit_probabilities.npz'
                np.savez_compressed(probability_file, exit_probabilities=np.array([1.]))
                first_wave, first_state, _ = self.checkpoint(folder, index=0)
                wave_path, state_path, _ = self.checkpoint(folder, **overrides)
                foreign_path = output / 'foreign.npy'
                np.save(foreign_path, np.array([5.+6.j]))
                before = {path: path.read_bytes() for path in (first_wave, first_state, wave_path, state_path, foreign_path)}
                with self.assertRaises((ValueError, FileNotFoundError)):
                    prune_completed_checkpoints(folder, [0, 2], probability_file)
                for path, content in before.items():
                    self.assertEqual(path.read_bytes(), content)

    def test_cleanup_requires_saved_probabilities_before_removing_wave(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'checkpoints'
            folder.mkdir()
            wave_path, state_path, _ = self.checkpoint(folder)
            before = state_path.read_bytes()
            with self.assertRaises((ValueError, FileNotFoundError)):
                prune_completed_checkpoints(folder, [2], folder.parent / 'exit_probabilities.npz')
            self.assertTrue(wave_path.exists())
            self.assertEqual(state_path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
