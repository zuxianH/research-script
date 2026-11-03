"""Resumable affordable bulk-q Si EELS, separate from strict benchmark gates."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import constants

from yoshioka.bulk_thermal_source import (covariance_integral, importance_nodes,
    born_thermal_batch, streamed_q)
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.free_propagation import TransverseGrid
from yoshioka.fourier_depth_fields import depth_field
from yoshioka.depth_fields import incident_depth
from .born_quadrature import line_spectra
from .workflow_io import ROOT, ResourceLimits, environment, fingerprint, json_write, sha256


DETECTORS = ('full_0_20_mrad', 'central_0_5_mrad', 'annulus_5_10_mrad', 'annulus_10_20_mrad')
AXIS = np.arange(-100, 100.025, .05)
PRIMITIVE = ROOT/'benchmark_results/si_eels_convergence/primitive_force_constants.npz'
ORIENTED = ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz'


def primitive_path(config, temperature):
    mapping = config.get('primitive_force_constants_by_temperature', {})
    return ROOT/mapping[str(temperature)] if str(temperature) in mapping else PRIMITIVE


def verify_input_baseline(output):
    """Use the actual pre-run snapshot to reject stale physical input reuse.

    Early inner contracts predate this additional geometry check; their creation
    records are preserved. The before snapshot proves the fixed geometry and
    original force inputs have not changed. This is an explicit current audit,
    not a backdated assertion that an absent hash was in those early contracts.
    """
    for directory in (Path(output), *Path(output).parents):
        path = directory/'protected_before.json'
        if path.exists():
            entries = json.loads(path.read_text())['files']
            for name in ('benchmark_results/si_lattice_dynamics/hiphive_model.npz',
                         'benchmark_results/si_eels_convergence/primitive_force_constants.npz',
                         'Si.tersoff.mod', 'benchmark_results/repair_20261102_v2/B11_refit/T300_n128.fcp',
                         'benchmark_results/repair_20261102_v2/B11_refit/T1000_n128.fcp'):
                if name in entries and sha256(ROOT/name) != entries[name]['sha256']:
                    raise ValueError('Physical input changed since the before snapshot; preserve existing caches and use a new result directory')
            return
        if directory == ROOT: return


def geometry(oriented, repeats, padding_angstrom=12):
    cells = np.zeros((repeats, 3)); cells[:, 2] = np.arange(repeats)
    unpadded = ((cells@oriented.cell_m)[:, None]+oriented.positions_m[None]).reshape(-1, 3)
    positions = unpadded.copy(); positions[:, 2] += padding_angstrom*1e-10
    length = repeats*oriented.cell_m[2, 2]+2*padding_angstrom*1e-10
    return positions, unpadded, length


def covariance_path(output, temperature, mesh):
    return output/'covariance'/f'T{temperature:g}_mesh{mesh}.json'


def run_covariance(config, output, limits):
    for temperature in config['temperatures_k']:
        force_path = primitive_path(config, temperature)
        model = RealSpaceForceConstants.load(force_path)
        for mesh in config['covariance_meshes']:
            path = covariance_path(output, temperature, mesh)
            if path.exists(): continue
            limits.check(); start = time.monotonic()
            covariance, diagnostic = covariance_integral(model, temperature, mesh)
            json_write(path, {'covariance_m2': covariance, 'diagnostic': diagnostic,
                'seconds': time.monotonic()-start, 'force_constants_sha256': sha256(force_path)})
            print('covariance', temperature, mesh, diagnostic['mean_square_displacement_m2'], flush=True)


def selected_covariance(config, output, temperature):
    path = covariance_path(output, temperature, config['selected_covariance_mesh'])
    return np.array(json.loads(path.read_text())['covariance_m2'])


def prepare_background(config, settings, output, limits):
    oriented = RealSpaceForceConstants.load(ORIENTED)
    repeats = settings.get('repeats', config['selected_repeats'])
    positions, unpadded, length = geometry(oriented, repeats, config['padding_angstrom'])
    covariance = selected_covariance(config, output, settings['temperature_k'])
    grid = TransverseGrid(oriented.cell_m[0, 0], oriented.cell_m[1, 1], settings['grid_unit'], settings['grid_unit'])
    contract = {'grid': grid.__dict__, 'steps': settings['slices'], 'repeats': repeats, 'elastic': settings.get('elastic', True),
                'temperature_k': settings['temperature_k'], 'covariance_m2': covariance.tolist(),
                'padding_angstrom': config['padding_angstrom'], 'force_sha256': sha256(ORIENTED),
                'field_sha256': sha256(ROOT/'yoshioka/fourier_depth_fields.py'),
                'incident_sha256': sha256(ROOT/'yoshioka/depth_fields.py')}
    directory = output/'background'/fingerprint(contract)[:20]; directory.mkdir(parents=True, exist_ok=True)
    path = directory/'state.json'
    state = json.loads(path.read_text()) if path.exists() else {'contract': contract}
    limits.check(estimated_bytes=settings['slices']*grid.nx*grid.ny*32+300*1024**2)
    start = time.monotonic()
    if not state.get('background_complete'):
        background, diagnostic = depth_field(positions, covariance, grid, length, settings['slices'], directory/'background.npy')
        state.update(background_complete=True, background_diagnostic=diagnostic)
        json_write(path, state)
    else: background = np.load(directory/'background.npy', mmap_mode='r')
    if not state.get('incident_complete'):
        wave, incident, diagnostic = incident_depth(grid, 60000*constants.elementary_charge,
            background, length, directory/'incident.npy', settings.get('elastic', True))
        np.save(directory/'incident_exit.npy', wave)
        state.update(incident_complete=True, incident_diagnostic=diagnostic,
                     elastic=settings.get('elastic', True))
        json_write(path, state)
    else:
        if state.get('elastic') != settings.get('elastic', True):
            raise ValueError('elastic/undistorted incident cache mismatch')
        incident = np.load(directory/'incident.npy', mmap_mode='r')
    state['latest_setup_seconds'] = time.monotonic()-start
    json_write(path, state)
    return positions, unpadded, length, covariance, grid, background, incident, state


def run_pilot(config, output, limits):
    primitive = RealSpaceForceConstants.load(PRIMITIVE)
    pilot_path = output/'pilot.json'
    rows = json.loads(pilot_path.read_text())['rows'] if pilot_path.exists() else []
    nodes, inverse_pdf = importance_nodes(2, config['pilot_seed'])
    for repeats in config['pilot_repeats']:
        if any(row['repeats'] == repeats for row in rows): continue
        settings = {'temperature_k': 300, 'grid_unit': 33, 'slices': 257, 'repeats': repeats}
        start = time.monotonic()
        positions, unpadded, length, covariance, grid, bg, inc, state = prepare_background(config, settings, output, limits)
        mode_seconds = []
        for i in range(2):
            limits.check(); before = time.monotonic()
            e, weights, born, _ = streamed_q(primitive, positions, covariance, grid, length, settings['slices'],
                nodes[i], 300, inc, bg, output/'pilot_source.npy', weight=inverse_pdf[i], reference_positions=unpadded)
            mode_seconds.append(time.monotonic()-before)
        row = {'repeats': repeats, 'thickness_nm': float(repeats*RealSpaceForceConstants.load(ORIENTED).cell_m[2, 2]*1e9),
               'settings': settings, 'q_seconds': mode_seconds, 'seconds': time.monotonic()-start,
               'incident_norm_drift': state['incident_diagnostic']['max_incident_norm_drift'],
               'predicted_seconds_512_q_at_pilot_settings': float(np.mean(mode_seconds)*512)}
        rows.append(row)
        json_write(pilot_path, {'rows': rows, 'selected_repeats': config['selected_repeats'],
            'selection': 'The shorter pilot specimen is selected for cost, before opening TACAW spectral values; no thickness fitted to spectral agreement.',
            'resources': limits.costs()})
        print('pilot', row, flush=True)


def run_born(config, output, limits):
    verify_input_baseline(output)
    oriented = RealSpaceForceConstants.load(ORIENTED)
    positions, unpadded, length = geometry(oriented, config['selected_repeats'], config['padding_angstrom'])
    for temperature in config['temperatures_k']:
        force_path = primitive_path(config, temperature)
        primitive = RealSpaceForceConstants.load(force_path)
        covariance = selected_covariance(config, output, temperature)
        for mesh in config['born_meshes']:
            count = mesh**3
            contract = {'mesh': mesh, 'temperature_k': temperature, 'covariance': covariance.tolist(),
                'repeats': config['selected_repeats'], 'padding_angstrom': config['padding_angstrom'],
                'algorithm': sha256(ROOT/'yoshioka/bulk_thermal_source.py'), 'force': sha256(force_path)}
            folder = output/'born'/f'T{temperature:g}_mesh{mesh}'; folder.mkdir(parents=True, exist_ok=True)
            path = folder/'state.json'
            state = json.loads(path.read_text()) if path.exists() else {'contract': contract, 'completed_q': 0, 'total_q': count, 'cumulative_seconds': 0.}
            if state['contract'] != contract: raise ValueError('Born checkpoint contract changed')
            if state.get('status') == 'COMPLETE': continue
            arrays = {}
            for name, shape in [('energies_mev', (count, 12)), ('weights', (count, 12, 4))]:
                p = folder/(name+'.npy')
                arrays[name] = np.load(p, mmap_mode='r+') if p.exists() else np.lib.format.open_memmap(p, mode='w+', dtype=float, shape=shape)
            start = time.monotonic(); last = start
            try:
                for lo in range(state['completed_q'], count, config['born_batch']):
                    limits.check(); hi = min(count, lo+config['born_batch'])
                    q = np.array(np.unravel_index(np.arange(lo, hi), (mesh,)*3)).T/mesh
                    q[q > .5] -= 1
                    e, w, _ = born_thermal_batch(primitive, unpadded, covariance,
                        np.diag(oriented.cell_m)[:2], length, config['padding_angstrom']*1e-10,
                        q, np.full(hi-lo, 1/count), temperature, 60000*constants.elementary_charge)
                    arrays['energies_mev'][lo:hi] = e; arrays['weights'][lo:hi] = w
                    for array in arrays.values(): array.flush()
                    state['completed_q'] = hi; state['cumulative_seconds'] += time.monotonic()-last
                    json_write(path, state); last = time.monotonic()
                spectra = line_spectra(arrays['energies_mev'], arrays['weights'], AXIS)
                np.savez_compressed(folder/'spectrum.npz', energy_mev=AXIS, spectra=spectra,
                                    retained_weights=np.sum(arrays['weights'], axis=(0, 1)))
                state['status'] = 'COMPLETE'
            except (TimeoutError, MemoryError) as exc:
                state['status'] = 'INCOMPLETE'; state['stop_reason'] = str(exc)
            state['latest_invocation_seconds'] = time.monotonic()-start
            state['resources'] = limits.costs(); json_write(path, state)
            print('Born', temperature, mesh, state['status'], state['completed_q'], flush=True)
            if state['status'] != 'COMPLETE': return


def run_source_case(config, settings, output, limits, max_new_q=None):
    verify_input_baseline(output)
    force_path = primitive_path(config, settings['temperature_k'])
    primitive = RealSpaceForceConstants.load(force_path)
    nodes, inverse_pdf = importance_nodes(settings['power'], settings['seed'], uniform_fraction=settings.get('uniform_fraction', .5))
    count = len(nodes)
    contract = {'settings': settings, 'repeats': config['selected_repeats'],
        'padding_angstrom': config['padding_angstrom'], 'force': sha256(force_path),
        'covariance': selected_covariance(config, output, settings['temperature_k']).tolist(),
        'algorithm': sha256(ROOT/'yoshioka/bulk_thermal_source.py'),
        'distribution': 'scrambled Sobol 50% uniform plus 50% Gamma radial; explicit inverse mixture density'}
    folder = output/'source'/settings['name']; folder.mkdir(parents=True, exist_ok=True)
    path = folder/'state.json'
    state = json.loads(path.read_text()) if path.exists() else {'contract': contract, 'completed_q': 0, 'total_q': count, 'cumulative_seconds': 0.}
    if state['contract'] != contract: raise ValueError('source checkpoint contract changed')
    if state.get('status') == 'COMPLETE': return state
    positions, unpadded, length, covariance, grid, background, incident, bg_state = prepare_background(config, settings, output, limits)
    arrays = {}
    for name, shape in [('energies_mev', (count, 12)), ('weights', (count, 12, 4)), ('born_weights', (count, 12, 4))]:
        p = folder/(name+'.npy')
        arrays[name] = np.load(p, mmap_mode='r+') if p.exists() else np.lib.format.open_memmap(p, mode='w+', dtype=float, shape=shape)
    start = time.monotonic(); last = start; resumed = state['completed_q']
    try:
        for iq in range(resumed, count):
            limits.check()
            if max_new_q is not None and iq-resumed >= max_new_q: break
            e, w, b, _ = streamed_q(primitive, positions, covariance, grid, length, settings['slices'],
                nodes[iq], settings['temperature_k'], incident, background, folder/'working_source.npy',
                weight=inverse_pdf[iq], elastic=settings.get('elastic', True), reference_positions=unpadded)
            arrays['energies_mev'][iq] = e; arrays['weights'][iq] = w; arrays['born_weights'][iq] = b
            for a in arrays.values(): a.flush()
            state['completed_q'] = iq+1; state['cumulative_seconds'] += time.monotonic()-last
            json_write(path, state); last = time.monotonic()
            if (iq+1)%8 == 0: print('source', settings['name'], iq+1, '/', count, flush=True)
    except (TimeoutError, MemoryError) as exc: state['stop_reason'] = str(exc)
    state['status'] = 'COMPLETE' if state['completed_q'] == count else 'INCOMPLETE'
    if state['status'] == 'COMPLETE':
        direct = line_spectra(arrays['energies_mev'], arrays['weights']/count, AXIS)
        born = line_spectra(arrays['energies_mev'], arrays['born_weights']/count, AXIS)
        np.savez_compressed(folder/'spectrum.npz', energy_mev=AXIS, direct=direct, born_sample=born,
                            correction=direct-born, q_fractional=nodes, inverse_sampling_density=inverse_pdf)
    state.update(latest_invocation_seconds=time.monotonic()-start, resumed_from_q=resumed,
        incident_norm_drift=bg_state['incident_diagnostic']['max_incident_norm_drift'], resources=limits.costs())
    json_write(path, state); return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='benchmarks/configs/approximate_eels_20261102_v1.json')
    parser.add_argument('--stage', choices=('covariance', 'pilot', 'born', 'source', 'all'), default='all')
    parser.add_argument('--case')
    parser.add_argument('--max-new-q', type=int)
    args = parser.parse_args(); config = json.loads(Path(args.config).read_text())
    output = ROOT/config['output']; output.mkdir(parents=True, exist_ok=True)
    limits = ResourceLimits(config['runtime_limit_seconds'], config['memory_limit_gib'], 10000)
    json_write(output/f'environment_{Path(args.config).stem}.json', {'environment': environment(), 'constants': {
        'hbar_j_s': constants.hbar, 'atomic_mass_kg': constants.atomic_mass,
        'elementary_charge_c': constants.elementary_charge, 'Boltzmann_j_k': constants.Boltzmann,
        'source': 'existing project scipy.constants CODATA 2022'}, 'parameters': config})
    if args.stage in ('all', 'covariance'): run_covariance(config, output, limits)
    if args.stage in ('all', 'pilot'): run_pilot(config, output, limits)
    if args.stage in ('all', 'born'): run_born(config, output, limits)
    if args.stage in ('all', 'source'):
        for settings in config['source_cases']:
            if args.case and settings['name'] != args.case: continue
            try: state = run_source_case(config, settings, output, limits, args.max_new_q)
            except (TimeoutError, MemoryError) as exc:
                print('bounded stop', str(exc), flush=True); break
            if state['status'] != 'COMPLETE': break
    json_write(output/'latest_invocation.json', {'stage': args.stage, 'case': args.case, 'resources': limits.costs()})


if __name__ == '__main__': main()
