"""Benchmark 7 finite-volume detector integration with unchanged physical BZ.

Each source coefficient already includes 1/sqrt(Nq). Exact detector/cell area
fractions replace binary indicators; there is no per-spectrum or per-bin fitted
rescaling. Halo nodes just outside the detector are evaluated explicitly.
"""

import argparse
import hashlib
import json
import mmap
from pathlib import Path
import time

import numpy as np
from scipy import constants

from yoshioka.detector_cells import angular_cell_spacing, cell_fractions
from yoshioka.free_propagation import electron_wavenumber
from yoshioka.integration_measures import uniform_bz_nodes
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from .born_quadrature import born_batch
from .eels_observables import detector_masks
from .repair_convergence import compare_cases, summarize_arrays
from .workflow_io import ROOT, ResourceLimits, fingerprint, json_write, sha256


def geometry_table(primitive, spec, mesh, shift):
    k = electron_wavenumber(spec['incident_energy_ev']*constants.elementary_charge)
    spacing = angular_cell_spacing(primitive.reciprocal_m_inv, spec['transverse_lengths_m'], k, mesh)
    beam = 2*np.pi/np.array(spec['transverse_lengths_m'])
    integer = np.rint(primitive.reciprocal_m_inv[:, :2]/beam).astype(int)
    gcds = np.array([np.gcd.reduce(np.r_[mesh, abs(integer[:, axis])]) for axis in range(2)])
    offset = (-np.asarray(shift) @ integer/gcds) % 1.
    bounds = np.ceil((20+np.linalg.norm(spacing))/spacing).astype(int)+1
    x = (np.arange(-bounds[0], bounds[0]+1)+offset[0])*spacing[0]
    y = (np.arange(-bounds[1], bounds[1]+1)+offset[1])*spacing[1]
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    detectors, pixels, halves = cell_fractions(points, spacing)
    # Positivity is enforced only for bounded roundoff in analytic geometry.
    minimum = min(np.min(detectors), np.min(pixels), np.min(halves))
    if minimum < -2e-9:
        raise ValueError('negative geometric overlap exceeds roundoff allowance')
    detectors, pixels, halves = [np.maximum(value, 0) for value in (detectors, pixels, halves)]
    area = np.prod(spacing)
    diagnostic = {'cell_spacing_mrad': spacing.tolist(), 'cell_area_mrad2': float(area),
        'constant_full_disk_area': float(np.sum(detectors[:, 0])*area), 'exact_full_disk_area': float(np.pi*400),
        'maximum_partition_residual': float(np.max(abs(np.sum(pixels, axis=-1)-detectors[:, 0]))),
        'roundoff_minimum_before_clipping': float(minimum), 'offset_in_cells': offset.tolist()}
    return spacing, offset, bounds, len(x), detectors, pixels, halves, diagnostic


def checkpointed_cells(primitive, spec, mesh, shift, output, limits, batch=128):
    sources = {name: sha256(ROOT/name) for name in ('benchmarks/detector_cubature.py', 'benchmarks/born_quadrature.py',
        'yoshioka/detector_cells.py', 'yoshioka/integration_measures.py', 'yoshioka/lattice_dynamics.py')}
    contract = {'spec': spec, 'mesh': mesh, 'shift': shift, 'sources': sources,
        'force_constants': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in primitive.__dict__.items()}}
    digest = fingerprint(contract)
    folder = output/'cache'/digest[:20]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'state.json'
    count = mesh**3
    state = json.loads(path.read_text()) if path.exists() else {'digest': digest, 'contract': contract,
        'completed_q': 0, 'total_q': count, 'cumulative_seconds': 0., 'status': 'INCOMPLETE'}
    if state['digest'] != digest or not 0 <= state['completed_q'] <= count or state['total_q'] != count:
        raise ValueError('invalid cubature checkpoint identity or prefix')
    shapes = {'energies': (count, 6), 'weights': (count, 6, 4), 'old_weights': (count, 6, 4),
        'momentum': (count, 8, 8), 'old_momentum': (count, 8, 8), 'halves': (count, 2),
        'old_halves': (count, 2), 'forward': (count,), 'physical': (count, 6)}
    arrays = {}
    for name, shape in shapes.items():
        target = folder/f'{name}.npy'
        if target.exists():
            value = np.load(target, mmap_mode='r+', allow_pickle=False)
            if value.shape != shape or value.dtype != np.dtype('float64'):
                raise ValueError('cubature checkpoint array mismatch')
        else:
            if state['completed_q']:
                raise ValueError('missing committed cubature array')
            value = np.lib.format.open_memmap(target, mode='w+', dtype=np.float64, shape=shape)
        arrays[name] = value
    if state['completed_q'] == count:
        state['latest_invocation_seconds'] = 0.
        return folder, state, arrays
    spacing, offset, bounds, nx, det, pixels, halves, diagnostic = geometry_table(primitive, spec, mesh, shift)
    state['geometric_checks'] = diagnostic
    max_angle = 20+np.linalg.norm(spacing)/2+1e-9
    positions = np.asarray(spec['positions_m'])
    covariance = np.asarray(spec['covariance_m2'])
    energy = spec['incident_energy_ev']*constants.elementary_charge
    k = electron_wavenumber(energy)
    started = time.monotonic()
    last_commit = started
    state['resumed_from_q'] = state['completed_q']
    try:
        for start in range(state['completed_q'], count, batch):
            limits.check(estimated_bytes=500*1024**2)
            stop = min(start+batch, count)
            q, qw = uniform_bz_nodes(mesh, start, stop, shift)
            energies, amplitudes, qxy, angles, physical = born_batch(primitive, positions, covariance,
                spec['transverse_lengths_m'], spec['specimen_thickness_m']+24e-10, 12e-10, q, qw, energy, max_angle)
            probability = abs(amplitudes)**2
            theta = qxy/k*1000
            coordinate = theta/spacing-offset
            nearest = np.rint(coordinate).astype(int)
            if np.max(abs(coordinate-nearest)) > 1e-8:
                raise ValueError('physical Bloch beams do not lie on the derived angular cell lattice')
            ix, iy = nearest[..., 0]+bounds[0], nearest[..., 1]+bounds[1]
            outside = (ix < 0) | (ix >= nx) | (iy < 0) | (iy >= 2*bounds[1]+1)
            if np.any(probability*np.any(outside[:, None], axis=1)[:, None]):
                # Test the actual beam-specific mask below; square FFT candidate
                # corners lie outside the circular halo and have exact zero P.
                if np.any(probability*outside[:, None]):
                    raise ValueError('nonzero source beam outside geometry table')
            index = np.clip(iy, 0, 2*bounds[1])*nx+np.clip(ix, 0, nx-1)
            arrays['energies'][start:stop] = energies
            arrays['physical'][start:stop] = physical
            arrays['weights'][start:stop] = np.einsum('qmb,qbd->qmd', probability, det[index])
            arrays['momentum'][start:stop] = np.einsum('qb,qbc->qc', np.sum(probability, axis=1), pixels[index]).reshape(stop-start, 8, 8)
            arrays['halves'][start:stop] = np.einsum('qb,qbc->qc', np.sum(probability, axis=1), halves[index])
            arrays['forward'][start:stop] = np.sum(probability*(np.linalg.norm(qxy, axis=-1) < 1e-5)[:, None], axis=(1, 2))
            for j, mask in enumerate(detector_masks(angles).values()):
                arrays['old_weights'][start:stop, :, j] = np.sum(probability*mask[:, None], axis=-1)
            selected = angles <= 20+1e-12
            for iq in range(stop-start):
                arrays['old_momentum'][start+iq] = np.histogram2d(theta[iq, :, 0], theta[iq, :, 1], bins=np.linspace(-20, 20, 9),
                    weights=np.sum(probability[iq], axis=0)*selected[iq])[0]
            for j, mask in enumerate((theta[:, :, 0] >= 0, theta[:, :, 0] < 0)):
                arrays['old_halves'][start:stop, j] = np.sum(probability*(mask & selected)[:, None], axis=(1, 2))
            for value in arrays.values():
                value.flush()
            state['completed_q'] = stop
            state['cumulative_seconds'] += time.monotonic()-last_commit
            json_write(path, state)
            for value in arrays.values():
                value._mmap.madvise(mmap.MADV_DONTNEED)
            last_commit = time.monotonic()
    except (TimeoutError, MemoryError) as exc:
        state['stop_reason'] = str(exc)
    state['status'] = 'COMPLETE' if state['completed_q'] == count else 'INCOMPLETE'
    state['latest_invocation_seconds'] = time.monotonic()-started
    json_write(path, state)
    return folder, state, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='benchmarks/configs/detector_cells_20261102_v1.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = ROOT/config['output']
    output.mkdir(parents=True, exist_ok=True)
    old = json.loads((ROOT/'benchmark_results/si_eels_convergence/results.json').read_text())
    spec = old['model_specification']
    primitive = RealSpaceForceConstants.load(ROOT/'benchmark_results/si_eels_convergence/primitive_force_constants.npz')
    limits = ResourceLimits(config['runtime_limit_seconds'], config['memory_limit_gib'], 100)
    results = {'benchmark': 'Benchmark 7 exact angular-cell integration', 'status': 'INCOMPLETE',
        'parameters': config, 'cases': [], 'comparisons': [], 'old_definition_cases': [],
        'physical_measure': 'same normalized bulk BZ measure, piecewise-constant angular-density interpolation on rectangular image cells; exact circle/pixel overlaps replace nodal masks; angular area not multiplied into BZ weights again',
        'source_validation': 'benchmark_results/repair_20261102_v1/source_validation.json', 'accepted_settings': None}
    for mesh in config['meshes']:
        folder, state, arrays = checkpointed_cells(primitive, spec, mesh, [0, 0, 0], output, limits)
        if state['status'] != 'COMPLETE':
            results['resource_stop'] = {k: state.get(k) for k in ('status', 'completed_q', 'total_q', 'stop_reason', 'latest_invocation_seconds', 'cumulative_seconds')}
            break
        case, spectra = summarize_arrays(arrays)
        case.update(settings={'mesh': mesh, 'shift': [0, 0, 0], 'route': 'Born with exact cell-detector overlaps'}, artifact=str(folder/'summary.npz'), checkpoint=state)
        np.savez_compressed(case['artifact'], energy_grid_mev=np.arange(-10, 100.025, .05), **{f'spectrum_{name}': spectra[:, j] for j, name in enumerate(('full_0_20_mrad', 'central_0_5_mrad', 'annulus_5_10_mrad', 'annulus_10_20_mrad'))})
        original_arrays = {**arrays, 'weights': arrays['old_weights'], 'momentum': arrays['old_momentum'], 'halves': arrays['old_halves']}
        original, _ = summarize_arrays(original_arrays)
        original['settings'] = case['settings']
        results['old_definition_cases'].append(original)
        results['cases'].append(case)
        if len(results['cases']) > 1:
            comparison = compare_cases(results['cases'][-2], case, output)
            results['comparisons'].append(comparison)
            print('cell cubature', mesh, 'L1', comparison['spectral_changes']['maximum_relative_L1'], 'finite max', comparison['physical_detector_changes']['maximum_acceptance_error'], 'peak', comparison['maximum_peak_shift_mev'], flush=True)
        json_write(output/'checkpoint.json', results)
    results['physical_quadrature_status'] = 'PASS' if len(results['comparisons']) >= 2 and all(r['physical_passed'] for r in results['comparisons'][-2:]) else 'INCOMPLETE'
    results['joint_source_status'] = 'INCOMPLETE: final jointly refined numerical-source control remains a separate gate'
    results['resource_costs'] = limits.costs()
    json_write(output/'results.json', results)


if __name__ == '__main__':
    main()
