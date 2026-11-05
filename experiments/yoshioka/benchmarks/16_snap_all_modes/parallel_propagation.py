"""Independent thermal-state propagation with atomic, checked checkpoints.

Each process owns one Hamiltonian and writes only its current initial state's
checkpoint files. Final waves stay in memory for the caller's probability save.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from yoshioka.provenance import fingerprint, json_write, sha256


_PROPAGATION_CONTEXT = None
_NORM_TOLERANCE = 1e-9


def _initialize_worker(hamiltonian_class, bath, cfg, fields_folder,
                       checkpoint_folder, field_identity, contract_context):
    global _PROPAGATION_CONTEXT
    from threadpoolctl import threadpool_limits

    # Retain the controller for this worker's lifetime. Explicit Hamiltonian
    # threads operate independently of the BLAS thread count.
    thread_limit = threadpool_limits(limits=1)
    hamiltonian = hamiltonian_class(
        bath, cfg, Path(fields_folder), validated_field=field_identity)
    _PROPAGATION_CONTEXT = dict(
        hamiltonian=hamiltonian, checkpoint=Path(checkpoint_folder),
        contract_context=contract_context, thread_limit=thread_limit)


def _log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def _check_wave(wave, shape, description):
    if wave.shape != shape or wave.dtype != np.dtype(np.complex128):
        raise ValueError(f'{description} shape/complex dtype mismatch')
    if not np.all(np.isfinite(wave)):
        raise ValueError(f'{description} has non-finite amplitudes')
    norm = float(np.sum(np.abs(wave)**2))
    if not np.isfinite(norm) or abs(norm - 1.) > _NORM_TOLERANCE:
        raise ValueError(f'{description} probability norm drift exceeds 1e-9')
    return abs(norm - 1.)


def _read_checkpoint(path, initial_index, contract, hamiltonian):
    """Accept only the exact current wave owned by this initial state."""
    try:
        state = json.loads(path.read_text())
        if not isinstance(state, dict) or state['contract'] != contract:
            raise ValueError('checkpoint configuration/source/bath mismatch')
        completed = state['completed_slices']
        total = state['total_slices']
        if (type(completed) is not int or type(total) is not int
                or total != hamiltonian.steps or not 1 <= completed <= total):
            raise ValueError('checkpoint slice metadata mismatch')
        expected = f'initial_{initial_index}_slice_{completed}.npy'
        if state['wave_file'] != expected:
            raise ValueError('checkpoint wave filename mismatch')
        wave_path = path.parent / expected
        if sha256(wave_path) != state['wave_sha256']:
            raise ValueError('checkpoint wave checksum mismatch')
        drift = float(state['max_norm_drift'])
        if not np.isfinite(drift) or not 0. <= drift <= _NORM_TOLERANCE:
            raise ValueError('checkpoint norm drift metadata is invalid')
        wave = np.load(wave_path, allow_pickle=False)
        measured_drift = _check_wave(wave, hamiltonian.shape, 'checkpoint wave')
        # Validate the recorded bound as well as the actual resumed wave.
        if measured_drift > drift + 8 * np.finfo(float).eps:
            raise ValueError('checkpoint norm drift exceeds recorded bound')
    except (OSError, KeyError, TypeError, OverflowError, json.JSONDecodeError) as error:
        raise ValueError(f'invalid checkpoint {path.name}: {error}') from error
    return wave, completed, drift, state


def _propagate_initial(initial_index):
    context = _PROPAGATION_CONTEXT
    h = context['hamiltonian']
    checkpoint = context['checkpoint']
    initial_index = int(initial_index)
    contract = fingerprint(dict(context['contract_context'], initial=initial_index))
    state_path = checkpoint / f'initial_{initial_index}.json'
    state = None
    if state_path.exists():
        wave, completed, drift, state = _read_checkpoint(
            state_path, initial_index, contract, h)
        _log(f'Initial {initial_index}: resuming after slice {completed}/{h.steps}')
    else:
        wave = np.zeros(h.shape, dtype=np.complex128)
        wave[initial_index, h.n // 2, h.n // 2] = 1.
        completed, drift = 0, 0.

    for index in range(completed, h.steps):
        wave = np.asarray(h.exponential_action(wave, index))
        drift = max(drift, _check_wave(wave, h.shape, 'propagated wave'))
        filename = f'initial_{initial_index}_slice_{index + 1}.npy'
        temporary = checkpoint / (filename + '.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, wave, allow_pickle=False)
        wave_path = checkpoint / filename
        temporary.replace(wave_path)
        previous = state['wave_file'] if state is not None else None
        state = dict(
            contract=contract, completed_slices=index + 1, total_slices=h.steps,
            wave_file=filename, wave_sha256=sha256(wave_path), max_norm_drift=drift)
        json_write(state_path, state)
        if previous is not None and previous != filename:
            (checkpoint / previous).unlink()
        _log(f'Initial {initial_index}: slice {index + 1}/{h.steps}; norm drift {drift:.3g}')
    return wave, drift


def propagate_initial_states(hamiltonian_class, bath, cfg, fields_folder,
                             checkpoint_folder, initial_indices, field_identity,
                             contract_context):
    """Propagate separate thermal states, returning waves in the input order.

    ``contract_context`` must contain every numerical/source/bath identity that
    affects propagation. Execution-only process/thread counts can be omitted.
    ``hamiltonian_class`` must be importable by spawned Python processes.
    """
    workers = cfg.get('propagation_workers', 1)
    if type(workers) is not int or workers < 1:
        raise ValueError('propagation_workers must be a positive integer')
    if not isinstance(contract_context, dict) or 'initial' in contract_context:
        raise ValueError('contract_context must be a dictionary without an initial key')
    indices = np.asarray(initial_indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('initial_indices must be a one-dimensional integer array')
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0):
        raise ValueError('initial_indices must be unique and nonnegative')
    # The channel count is inexpensive to determine from this declared basis.
    from yoshioka.coupled_channels import channel_basis
    channels = len(channel_basis(len(bath['omega_rad_s']), cfg['p']))
    if np.any(indices >= channels):
        raise ValueError('initial channel index exceeds the declared basis')
    if not len(indices):
        return [], []
    checkpoint_folder = Path(checkpoint_folder)
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    arguments = (hamiltonian_class, bath, cfg, str(fields_folder),
                 str(checkpoint_folder), field_identity, contract_context)
    workers = min(workers, len(indices))
    if workers == 1:
        _initialize_worker(*arguments)
        results = [_propagate_initial(int(index)) for index in indices]
    else:
        with ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                initializer=_initialize_worker, initargs=arguments) as pool:
            # executor.map preserves caller order despite differing runtimes.
            results = list(pool.map(_propagate_initial, indices.tolist()))
    return [item[0] for item in results], [item[1] for item in results]
