"""Accelerated full-space Born BZ cubature, independently checked against sources.

This evaluates the same finite atom sum, common-v phases and DW covariance as
Benchmark 7. It does not invoke the numerical gradient/source propagator or the
old full_space_born implementation. Arrays retain absolute per-q line weights.
"""

import json
import hashlib
from pathlib import Path
import time

import numpy as np
from scipy import constants
from scipy.signal import fftconvolve

from yoshioka.elastic_propagation import electron_speed
from yoshioka.electron_phonon import LobatoSilicon
from yoshioka.free_propagation import electron_wavenumber
from yoshioka.integration_measures import uniform_bz_nodes
from .bulk_eels import primitive_atom_mapping
from .eels_observables import DETECTORS, detector_masks
from .workflow_io import fingerprint, json_write, sha256


def bloch_beams(primitive, lengths_m, q_fractional, energy_j, angle_mrad=20):
    spacing = 2*np.pi / np.asarray(lengths_m)
    relation = primitive.reciprocal_m_inv[:, :2] / spacing
    if not np.allclose(relation, np.rint(relation), atol=1e-8, rtol=0):
        raise ValueError('electron cell and primitive lattice are not commensurate')
    residue = (-q_fractional @ np.rint(relation)) % 1.
    residue[residue > .5] -= 1.
    shift = residue * spacing
    radius = electron_wavenumber(energy_j)*angle_mrad/1000
    bounds = np.ceil(radius/spacing+.5).astype(int)
    hx, hy = np.meshgrid(np.arange(-bounds[0], bounds[0]+1), np.arange(-bounds[1], bounds[1]+1))
    regular = np.column_stack((hx.ravel(), hy.ravel()))*spacing
    qxy = regular[None] + shift[:, None]
    angles = np.linalg.norm(qxy, axis=-1)/electron_wavenumber(energy_j)*1000
    return qxy, angles, angles <= angle_mrad+1e-12


def born_batch(primitive, positions_m, covariance_m2, lengths_m, length_m,
               padding_m, q_fractional, q_weights, energy_j, max_angle_mrad=20):
    """Vectorized atomic sum; returned amplitudes include sqrt(BZ weight) once."""
    q_fractional = np.asarray(q_fractional)
    omega, eigenvectors = primitive.modes(q_fractional)
    physical = np.ones(omega.shape, dtype=bool)
    gamma = np.flatnonzero(np.linalg.norm(q_fractional, axis=1) < 1e-14)
    translations = np.tile(np.eye(3), (len(primitive.positions_m), 1))
    translations *= np.repeat(np.sqrt(primitive.masses_kg), 3)[:, None]
    translations /= np.linalg.norm(translations, axis=0)
    for iq in gamma:
        overlaps = np.sum(abs(eigenvectors[iq].conj().T @ translations)**2, axis=-1)
        physical[iq, np.argsort(overlaps)[-3:]] = False
    if np.any(omega[physical] <= 0):
        raise ValueError('unstable physical phonon branch')
    safe_omega = np.where(physical, omega, 1.)
    qxy, angles, inside = bloch_beams(primitive, lengths_m, q_fractional, energy_j, max_angle_mrad)
    v = electron_speed(energy_j)
    k = electron_wavenumber(energy_j)
    kappa = np.sum(qxy**2, axis=-1)[:, None]/(2*k) + safe_omega[:, :, None]/v
    qz = -kappa
    basis, _ = primitive_atom_mapping(primitive, positions_m)
    eigen = eigenvectors.reshape(len(q_fractional), len(primitive.positions_m), 3, -1)[:, basis].conj()
    # q,mode,beam,atom. The q-dependent atomic phase and Q-dependent electron
    # phase are kept separately from the common outgoing propagation phase.
    projection = np.einsum('qbc,qicm->qmbi', qxy, eigen[:, :, :2])
    projection += qz[:, :, :, None]*eigen[:, :, 2].transpose(0, 2, 1)[:, :, None]
    mass_factor = np.sqrt(constants.hbar*q_weights[:, None, None] /
                          (2*safe_omega[:, :, None]*primitive.masses_kg[basis][None, None]))
    qcart = q_fractional @ primitive.reciprocal_m_inv
    xy_phase = np.einsum('qbc,ic->qbi', qxy+qcart[:, None, :2], positions_m[:, :2])
    z_phase = (qz+qcart[:, None, None, 2])[:, :, :, None]*positions_m[None, None, None, :, 2]
    atomic_sum = np.sum(projection*mass_factor[:, :, None]*np.exp(-1j*(xy_phase[:, None]+z_phase)), axis=-1)
    squared = np.sum(qxy**2, axis=-1)[:, None]+qz**2
    potential = LobatoSilicon.from_abtem_table().fourier_energy_j_m3(squared)
    dw = np.exp(-.5*(np.einsum('qba,ac,qbc->qb', qxy, covariance_m2[:2, :2], qxy)[:, None]
                       + covariance_m2[2, 2]*qz**2))
    if np.any(abs(covariance_m2[:2, 2]) > 1e-35):
        raise ValueError('this B7 material branch requires the verified diagonal covariance')
    amplitudes = -potential*dw*atomic_sum*np.exp(-1j*kappa*(length_m-padding_m)) / (constants.hbar*v*np.prod(lengths_m))
    amplitudes *= physical[:, :, None]*inside[:, None]
    return omega*constants.hbar/(1e-3*constants.elementary_charge), amplitudes, qxy, angles, physical


def line_spectra(energies_mev, weights, grid_mev, fwhm_mev=2., deposition_step_mev=.0025):
    """Absolute Gaussian spectra using conservative linear line deposition.

    No spectrum normalization. Kernel samples are the actual normalized Gaussian
    density (per meV), without a fitted discrete normalization. For each line the
    L1 interpolation error is bounded by h²||G''||_1/8 < h²/(4 sigma²).
    Gaussian truncation is 12 sigma. Both the bound and an exact-route validation
    are reported by the benchmark. Unbroadened lines remain the acceptance input.
    """
    e = np.asarray(energies_mev).ravel()
    weights = np.asarray(weights).reshape(len(e), -1)
    sigma = fwhm_mev/(2*np.sqrt(2*np.log(2)))
    h = deposition_step_mev
    lower = np.floor((min(np.min(e), np.min(grid_mev))-12*sigma)/h)*h
    upper = max(np.max(e), np.max(grid_mev))+12*sigma
    size = int(np.ceil((upper-lower)/h))+2
    coordinate = (e-lower)/h
    indices = np.floor(coordinate).astype(int)
    fraction = coordinate-indices
    lines = np.zeros((size, weights.shape[1]))
    for j in range(weights.shape[1]):
        lines[:, j] = np.bincount(indices, weights=weights[:, j]*(1-fraction), minlength=size)
        lines[:, j] += np.bincount(indices+1, weights=weights[:, j]*fraction, minlength=size)
    reach = int(np.ceil(12*sigma/h))
    kernel = np.exp(-.5*(np.arange(-reach, reach+1)*h/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
    values = fftconvolve(lines, kernel[:, None], mode='same', axes=0)
    axis = lower+np.arange(size)*h
    return np.stack([np.interp(grid_mev, axis, values[:, j]) for j in range(weights.shape[1])], axis=-1)


def checkpointed_born(primitive, spec, mesh, shift, output, limits, batch_size=128, max_new_q=None):
    """Atomic prefix checkpoint. Configuration/source/shape changes reject reuse."""
    output = Path(output)
    primitive_hashes = {name: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
                        for name, value in primitive.__dict__.items()}
    contract = {'mesh': mesh, 'shift': list(shift), 'spec': spec,
                'algorithm_sha256': sha256(__file__), 'primitive': primitive_hashes}
    digest = fingerprint(contract)
    directory = output/'cache'/digest[:20]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'state.json'
    count = mesh**3
    state = json.loads(path.read_text()) if path.exists() else {
        'digest': digest, 'completed_q': 0, 'total_q': count, 'elapsed_seconds': 0.,
        'status': 'INCOMPLETE', 'contract': contract}
    if state['digest'] != digest or state['total_q'] != count or not 0 <= state['completed_q'] <= count:
        raise ValueError('checkpoint identity or completed prefix is invalid')
    shapes = {'energies': (count, 6), 'weights': (count, 6, 4), 'momentum': (count, 8, 8),
              'halves': (count, 2), 'forward': (count,), 'physical': (count, 6)}
    arrays = {}
    for name, shape in shapes.items():
        target = directory/f'{name}.npy'
        if target.exists():
            array = np.load(target, mmap_mode='r+', allow_pickle=False)
            if array.shape != shape or array.dtype != np.dtype('float64'):
                raise ValueError('checkpoint array shape/dtype mismatch')
        else:
            if state['completed_q']:
                raise ValueError('checkpoint array missing for a committed prefix')
            array = np.lib.format.open_memmap(target, mode='w+', dtype=np.float64, shape=shape)
        arrays[name] = array
    invocation_started = time.monotonic()
    last_commit = invocation_started
    resumed = state['completed_q']
    try:
        for start in range(resumed, count, batch_size):
            limits.check(estimated_bytes=batch_size*6*128*32*16*5+count*6*4*8)
            if max_new_q is not None and start-resumed >= max_new_q:
                break
            stop = min(start+batch_size, count)
            if max_new_q is not None:
                stop = min(stop, resumed+max_new_q)
            q, qw = uniform_bz_nodes(mesh, start, stop, shift)
            padding = 12e-10
            energies, amplitudes, qxy, angles, physical = born_batch(
                primitive, np.array(spec['positions_m']), np.array(spec['covariance_m2']),
                spec['transverse_lengths_m'], spec['specimen_thickness_m']+2*padding,
                padding, q, qw, spec['incident_energy_ev']*constants.elementary_charge)
            probability = abs(amplitudes)**2
            arrays['energies'][start:stop] = energies
            arrays['physical'][start:stop] = physical
            masks = detector_masks(angles)
            for j, mask in enumerate(masks.values()):
                arrays['weights'][start:stop, :, j] = np.sum(probability*mask[:, None], axis=-1)
            xy = qxy/electron_wavenumber(spec['incident_energy_ev']*constants.elementary_charge)*1000
            for iq in range(stop-start):
                arrays['momentum'][start+iq] = np.histogram2d(xy[iq, :, 0], xy[iq, :, 1], bins=np.linspace(-20, 20, 9), weights=np.sum(probability[iq], axis=0))[0]
            for side, selected in enumerate((xy[:, :, 0] >= 0, xy[:, :, 0] < 0)):
                arrays['halves'][start:stop, side] = np.sum(probability*selected[:, None], axis=(1, 2))
            arrays['forward'][start:stop] = np.sum(probability*(np.linalg.norm(qxy, axis=-1) < 1e-5)[:, None], axis=(1, 2))
            for array in arrays.values():
                array.flush()
            state['completed_q'] = stop
            state['elapsed_seconds'] += time.monotonic()-last_commit
            json_write(path, state)
            last_commit = time.monotonic()
    except (TimeoutError, MemoryError) as exc:
        state['stop_reason'] = str(exc)
    state['status'] = 'COMPLETE' if state['completed_q'] == count else 'INCOMPLETE'
    state['latest_invocation_seconds'] = time.monotonic()-invocation_started
    state['resumed_from_q'] = resumed
    json_write(path, state)
    return directory, state, arrays
