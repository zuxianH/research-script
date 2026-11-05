"""Small unitary checks for process scheduling and exact checkpoint reuse."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from parallel_propagation import propagate_initial_states
from yoshioka.provenance import json_write, sha256


class TinyHamiltonian:
    """Picklable unitary action with channel phases and spatial translation."""
    calls = []

    def __init__(self, bath, cfg, folder, validated_field=None):
        if validated_field != {'digest': 'checked-field'}:
            raise ValueError('unvalidated test field')
        self.n = cfg['n']
        self.steps = cfg['steps']
        self.shape = (3, self.n, self.n)
        self.fail_at = cfg.get('fail_at')
        self.thread_count = cfg.get('hamiltonian_threads', 1)

    def exponential_action(self, wave, index):
        self.calls.append(index)
        if self.fail_at == index:
            raise RuntimeError('intentional interruption')
        phase = np.exp(1j * (index + 1) * np.array([.2, .3, .7]))
        result = np.roll(wave, 1, axis=1) * phase[:, None, None]
        angle = .17 * (index + 1)
        left, right = result[0].copy(), result[1].copy()
        result[0] = np.cos(angle) * left + 1j * np.sin(angle) * right
        result[1] = 1j * np.sin(angle) * left + np.cos(angle) * right
        return result


class ParallelPropagationTests(unittest.TestCase):
    def setUp(self):
        TinyHamiltonian.calls = []
        self.bath = {'omega_rad_s': np.array([1., 2.])}
        self.cfg = dict(n=2, steps=4, p=1, propagation_workers=1,
                        hamiltonian_threads=2)
        self.identity = {'digest': 'checked-field'}
        self.contract = {'config': 'numerical-config', 'source': 'source-sha',
                         'bath': 'bath-sha', 'helper_source': 'helper-sha'}

    def run_states(self, folder, indices=(2, 0, 1), cfg=None, contract=None):
        return propagate_initial_states(
            TinyHamiltonian, self.bath, self.cfg if cfg is None else cfg,
            Path(folder) / 'fields', Path(folder) / 'checkpoints',
            np.array(indices, dtype=int), self.identity,
            self.contract if contract is None else contract)

    def state(self, folder, index=0):
        path = Path(folder) / 'checkpoints' / f'initial_{index}.json'
        return path, json.loads(path.read_text())

    def test_spawned_workers_match_serial_and_preserve_requested_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            reference, reference_drifts = self.run_states(folder / 'serial')
            actual, drifts = self.run_states(
                folder / 'parallel', cfg=dict(self.cfg, propagation_workers=2))
            for left, right in zip(actual, reference):
                np.testing.assert_array_equal(left, right)
            np.testing.assert_array_equal(drifts, reference_drifts)
            self.assertFalse(np.array_equal(actual[0], actual[1]))
            for index in (2, 0, 1):
                _, state = self.state(folder / 'parallel', index)
                self.assertEqual(state['completed_slices'], self.cfg['steps'])
                waves = list((folder / 'parallel' / 'checkpoints').glob(
                    f'initial_{index}_slice_*.npy'))
                self.assertEqual(len(waves), 1)

    def test_interrupted_wave_resumes_exactly_and_completed_wave_skips_actions(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, 'intentional interruption'):
                self.run_states(folder / 'resumed', indices=(0,),
                                cfg=dict(self.cfg, fail_at=1))
            _, saved = self.state(folder / 'resumed')
            self.assertEqual(saved['completed_slices'], 1)
            TinyHamiltonian.calls = []
            actual, actual_drifts = self.run_states(folder / 'resumed', indices=(0,))
            self.assertEqual(TinyHamiltonian.calls, [1, 2, 3])
            expected, expected_drifts = self.run_states(folder / 'reference', indices=(0,))
            np.testing.assert_array_equal(actual[0], expected[0])
            self.assertEqual(actual_drifts, expected_drifts)
            TinyHamiltonian.calls = []
            repeated, repeated_drifts = self.run_states(folder / 'resumed', indices=(0,))
            self.assertEqual(TinyHamiltonian.calls, [])
            np.testing.assert_array_equal(repeated[0], actual[0])
            self.assertEqual(repeated_drifts, actual_drifts)

    def test_checkpoint_contract_and_checksum_corruption_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.run_states(folder, indices=(0,))
            with self.assertRaisesRegex(ValueError, 'configuration/source/bath'):
                self.run_states(folder, indices=(0,),
                                contract=dict(self.contract, helper_source='changed'))
            _, saved = self.state(folder)
            wave_path = folder / 'checkpoints' / saved['wave_file']
            content = bytearray(wave_path.read_bytes())
            content[-1] ^= 1
            wave_path.write_bytes(content)
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                self.run_states(folder, indices=(0,))

    def test_checkpoint_wave_shape_dtype_finiteness_and_norm_are_checked(self):
        mutations = {
            'shape': lambda wave: wave[:, :1],
            'real dtype': lambda wave: wave.real.copy(),
            'reduced precision': lambda wave: wave.astype(np.complex64),
            'non-finite': lambda wave: wave * np.nan,
            'norm': lambda wave: wave * 2,
        }
        for name, mutate in mutations.items():
            with self.subTest(corruption=name), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                self.run_states(folder, indices=(0,))
                state_path, saved = self.state(folder)
                wave_path = folder / 'checkpoints' / saved['wave_file']
                wave = np.load(wave_path, allow_pickle=False)
                np.save(wave_path, mutate(wave), allow_pickle=False)
                saved['wave_sha256'] = sha256(wave_path)
                json_write(state_path, saved)
                with self.assertRaises(ValueError):
                    self.run_states(folder, indices=(0,))

    def test_checkpoint_metadata_cannot_select_another_wave_or_slice(self):
        mutations = {
            'other filename': {'wave_file': '../elsewhere.npy'},
            'wrong total': {'total_slices': 99},
            'negative slice': {'completed_slices': -1},
            'fractional slice': {'completed_slices': 1.5},
            'invalid drift': {'max_norm_drift': -1.},
            'excessive drift': {'max_norm_drift': 1e-3},
        }
        for name, update in mutations.items():
            with self.subTest(corruption=name), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                self.run_states(folder, indices=(0,))
                state_path, saved = self.state(folder)
                saved.update(update)
                json_write(state_path, saved)
                with self.assertRaises(ValueError):
                    self.run_states(folder, indices=(0,))

    def test_previous_wave_survives_failed_metadata_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with self.assertRaises(RuntimeError):
                self.run_states(folder, indices=(0,), cfg=dict(self.cfg, fail_at=1))
            state_path, saved = self.state(folder)
            original_metadata = state_path.read_bytes()
            previous_wave = folder / 'checkpoints' / saved['wave_file']
            original_wave = previous_wave.read_bytes()
            with patch('parallel_propagation.json_write',
                       side_effect=RuntimeError('metadata interruption')):
                with self.assertRaisesRegex(RuntimeError, 'metadata interruption'):
                    self.run_states(folder, indices=(0,))
            self.assertEqual(state_path.read_bytes(), original_metadata)
            self.assertEqual(previous_wave.read_bytes(), original_wave)
            # A later resume may replace its own uncommitted next-slice wave.
            self.run_states(folder, indices=(0,))
            self.assertFalse(previous_wave.exists())

    def test_duplicate_or_out_of_range_initial_states_and_worker_counts_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            for indices in ((0, 0), (-1,), (3,)):
                with self.subTest(indices=indices), self.assertRaises(ValueError):
                    self.run_states(temporary, indices=indices)
            for workers in (0, -1, 1.5, True):
                with self.subTest(workers=workers), self.assertRaises(ValueError):
                    self.run_states(temporary, cfg=dict(self.cfg, propagation_workers=workers))
            self.assertFalse((Path(temporary) / 'checkpoints').exists())


if __name__ == '__main__':
    unittest.main()
