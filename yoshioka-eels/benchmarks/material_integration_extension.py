"""Fixed-geometry material longitudinal integration: bounded and resumable.

The original ~20-nm B9A target remains incomplete unless its unchanged full
electron/bath/observable gates are passed. Conditional q_perp scans below are
diagnostics, not full-BZ integrals. Thin calculations are separate deliverables.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
from scipy import constants
from scipy.stats import t as student_t

from yoshioka.bulk_thermal_source import born_thermal_batch
from yoshioka.free_propagation import TransverseGrid
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.longitudinal_integration import (finite_window_angular,
    longitudinal_nodes, anisotropic_nodes, prepare_distorted_background,
    fixed_q_distorted_weights)
from .approximate_eels import geometry
from .born_quadrature import line_spectra
from .workflow_io import ROOT, ResourceLimits, environment, fingerprint, json_write, sha256


AXIS = np.arange(-10., 100.025, .05)
DETECTORS = ('full_0_20_mrad', 'central_0_5_mrad', 'annulus_5_10_mrad', 'annulus_10_20_mrad')
WINDOWS = {'acoustic_0_40_mev': (0., 40.), 'middle_40_65_mev': (40., 65.),
           'optical_65_90_mev': (65., 90.)}
MODEL = ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz'
SPECIFICATION = ROOT/'benchmark_results/si_eels_convergence/results.json'


def nodes_for_case(settings):
    if 'shape' in settings:
        return anisotropic_nodes(settings['shape'], settings.get('shifts', (0., 0., 0.)))
    scheme = settings.get('scheme', 'mesh')
    if scheme in ('mesh', 'stratified'):
        z, w = longitudinal_nodes(settings['nz'], settings.get('shift', 0.),
            settings['seed'] if scheme == 'stratified' else None)
    elif scheme == 'guided':
        # Guide chosen before inspecting the material integrand: three first
        # zeros on either side of qz=0, plus full support in both outer regions.
        radius = min(.25, 3/settings['repeats'])
        intervals = ((-.5, -radius), (-radius, radius), (radius, .5))
        z = []; w = []
        for count, (lo, hi) in zip(settings['counts'], intervals):
            x, weight = np.polynomial.legendre.leggauss(count)
            z.extend((lo+hi)/2+(hi-lo)*x/2)
            w.extend(weight*(hi-lo)/2)
        z = np.array(z); w = np.array(w)
    else:
        raise ValueError('unknown longitudinal sampling scheme')
    qxy = np.asarray(settings['qxy'])
    return np.column_stack((np.broadcast_to(qxy, (len(z), 2)), z)), w


def contract_for_case(config, settings, covariance):
    dependencies = ('benchmarks/material_integration_extension.py',
        'yoshioka/longitudinal_integration.py', 'yoshioka/bulk_thermal_source.py',
        'yoshioka/electron_phonon.py', 'yoshioka/depth_fields.py',
        'yoshioka/lattice_dynamics.py', 'yoshioka/free_propagation.py')
    return {'settings': settings, 'covariance_m2': covariance.tolist(),
        'covariance_source_sha256': sha256(SPECIFICATION), 'force_constants_sha256': sha256(MODEL),
        'sources': {name: sha256(ROOT/name) for name in dependencies},
        'energy_ev': 60000, 'padding_angstrom': config['padding_angstrom'],
        'temperature_k': 0, 'FWHM_mev': 2.,
        'measure': 'normalized oriented reciprocal cell, all twelve folded branches; only exact Gamma translations omitted',
        'normalization': 'one incident electron in the fixed oriented unit-cell area; q weights enter displacement as sqrt(weight) once',
        'scope': 'full three-dimensional coarse bath' if 'shape' in settings else 'conditional fixed transverse phonon momentum; integrates only qz',
        'electron_source': 'full depth-dependent distorted incident/source; original split-step rule, one FFT worker'}


def run_case(config, settings, output, limits, max_new_q=None, born_only=False):
    model = RealSpaceForceConstants.load(MODEL)
    spec = json.loads(SPECIFICATION.read_text())['model_specification']
    covariance = np.array(spec['covariance_m2'])
    contract = contract_for_case(config, settings, covariance)
    contract['born_only'] = bool(born_only)
    folder = output/('born' if born_only else 'source')/settings['name']
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'state.json'
    state = json.loads(path.read_text()) if path.exists() else {
        'contract': contract, 'digest': fingerprint(contract), 'completed_q': 0,
        'cumulative_seconds': 0., 'status': 'INCOMPLETE'}
    if state['contract'] != contract:
        raise ValueError('source/input identity changed; preserve old evidence and choose a new output version: '+str(path))
    nodes, weights = nodes_for_case(settings)
    if not np.isclose(np.sum(weights), 1, rtol=0, atol=5e-15) or np.any(weights <= 0):
        raise ValueError('quadrature measure must be positive and integrate one reciprocal cell')
    count = len(nodes); state['total_q'] = count
    arrays = {}
    for name, shape in (('energies_mev', (count, 12)), ('weights', (count, 12, 4)),
                        ('born_weights', (count, 12, 4)), ('q_seconds', (count,))):
        file = folder/(name+'.npy')
        arrays[name] = np.load(file, mmap_mode='r+') if file.exists() else np.lib.format.open_memmap(
            file, mode='w+', dtype=float, shape=shape)
        if arrays[name].shape != shape or arrays[name].dtype != np.float64:
            raise ValueError('checkpoint array shape/type mismatch')
    np.savez(folder/'nodes.npz', q_fractional=nodes, normalized_cell_weights=weights)
    if state['completed_q'] == count:
        return state
    positions, unpadded, length = geometry(model, settings['repeats'], config['padding_angstrom'])
    grid = TransverseGrid(*np.diag(model.cell_m)[:2], settings['grid'], settings['grid'])
    energy = 60000*constants.elementary_charge
    setup_started = time.monotonic()
    if not born_only:
        background_contract = {'positions_m': positions.tolist(), 'grid': grid.__dict__,
            'covariance_m2': covariance.tolist(), 'length_m': length, 'steps': settings['steps'],
            'model': sha256(MODEL), 'algorithm': sha256(ROOT/'yoshioka/longitudinal_integration.py')}
        bgfolder = output/'background'/fingerprint(background_contract)[:20]
        bgfolder.mkdir(parents=True, exist_ok=True)
        json_write(bgfolder/'contract.json', background_contract)
        background, incident, diagnostic = prepare_distorted_background(positions,
            covariance, grid, length, settings['steps'], bgfolder, energy, config.get('fft_workers', 1))
        if diagnostic is not None:
            json_write(bgfolder/'diagnostic.json', diagnostic)
        state['background_checkpoint'] = str(bgfolder)
    state['setup_seconds'] = time.monotonic()-setup_started
    resumed = state['completed_q']; started = time.monotonic()
    try:
        for iq in range(resumed, count):
            limits.check()
            if max_new_q is not None and iq-resumed >= max_new_q:
                break
            before = time.monotonic()
            if born_only:
                e, w, _ = born_thermal_batch(model, unpadded, covariance,
                    (grid.lx_m, grid.ly_m), length, config['padding_angstrom']*1e-10,
                    nodes[iq:iq+1], weights[iq:iq+1], 0, energy)
                e = e[0, :12]; w = w[0, :12]; b = w; branches = np.arange(12)
            else:
                e, w, b, branches = fixed_q_distorted_weights(model, positions,
                    unpadded, covariance, grid, length, settings['steps'], nodes[iq],
                    incident, background, folder/'working_source.npy', energy,
                    weights[iq], config.get('fft_workers', 1))
            arrays['energies_mev'][iq] = 0
            arrays['weights'][iq] = arrays['born_weights'][iq] = 0
            arrays['energies_mev'][iq, branches] = e
            arrays['weights'][iq, branches] = w
            arrays['born_weights'][iq, branches] = b
            arrays['q_seconds'][iq] = time.monotonic()-before
            for array in arrays.values(): array.flush()
            state['completed_q'] = iq+1
            state['cumulative_seconds'] += time.monotonic()-before
            state['status'] = 'COMPLETE' if iq+1 == count else 'INCOMPLETE'
            json_write(path, state)
            if (iq+1) % 8 == 0:
                print(settings['name'], iq+1, '/', count, 'q', flush=True)
    except (TimeoutError, MemoryError) as exc:
        state['stop_reason'] = str(exc)
    state.update(resumed_from_q=resumed, latest_invocation_seconds=time.monotonic()-started,
        thickness_nm=float(settings['repeats']*model.cell_m[2, 2]*1e9),
        physical_spacings_angstrom={'dx': grid.dx_m*1e10, 'dy': grid.dy_m*1e10,
            'dz': length/settings['steps']*1e10}, resources=limits.costs())
    if state['completed_q'] == count:
        spectrum = line_spectra(arrays['energies_mev'], arrays['weights'], AXIS)
        born_spectrum = line_spectra(arrays['energies_mev'], arrays['born_weights'], AXIS)
        observables = {}
        for detector, name in enumerate(DETECTORS):
            observables[name] = float(np.sum(arrays['weights'][:, :, detector]))
            for window, (lo, hi) in WINDOWS.items():
                selected = (arrays['energies_mev'] >= lo)&(arrays['energies_mev'] < hi)
                observables[name+'/'+window] = float(np.sum(arrays['weights'][:, :, detector][selected]))
        artifact = folder/'spectrum.npz'
        np.savez_compressed(artifact, energy_mev=AXIS, spectra=spectrum,
            born_spectra=born_spectrum, correction=spectrum-born_spectrum,
            integrated_weights=np.sum(arrays['weights'], axis=(0, 1)))
        state.update(artifact=str(artifact), observables=observables,
            partial_prefix_renormalized=False, global_B9A_acceptance_evaluated=False,
            mean_seconds_per_q=float(np.mean(arrays['q_seconds'])))
    json_write(path, state)
    return state


def window_diagnostic(config, output, limits):
    model = RealSpaceForceConstants.load(MODEL)
    rows = []
    for repeats in (4, 52):
        L = repeats*model.cell_m[2, 2]
        kappa = np.linspace(-8*2*np.pi/L, 8*2*np.pi/L, 3201)
        amplitude = finite_window_angular(kappa, L)
        target = output/f'window_repeats{repeats}.npz'
        np.savez(target, kappa_angular_rad_m=kappa, reciprocal_cycles_m=kappa/(2*np.pi),
            window_m=amplitude, squared_window_m2=abs(amplitude)**2)
        rows.append({'repeats': repeats, 'thickness_nm': L*1e9,
            'first_zero_angular_rad_m': 2*np.pi/L, 'first_zero_cycles_m': 1/L,
            'oriented_qz_fractional_first_zero': 1/repeats,
            'diagnostic_identity': 'integral d(kappa)/(2*pi)*|W|^2 = L',
            'artifact': str(target)})
    json_write(output/'finite_window_diagnostic.json', {'rows': rows,
        'warning': 'finite window guides reciprocal sampling; actual source keeps its full depth dependence and is not replaced by W_L',
        'units': 'angular wavevectors in rad/m; fractional oriented qz times 2*pi/c is angular, times 1/c is cycles/m'})


def compare_states(coarse, fine):
    if coarse['status'] != 'COMPLETE' or fine['status'] != 'COMPLETE':
        return {'status': 'INCOMPLETE'}
    a = np.load(coarse['artifact']); b = np.load(fine['artifact'])
    absolute_L1 = np.trapezoid(abs(a['spectra']-b['spectra']), AXIS, axis=0)
    denominator = np.trapezoid(abs(b['spectra']), AXIS, axis=0)
    relative_L1 = absolute_L1/np.maximum(denominator, 1e-300)
    changes = {name: abs(coarse['observables'][name]-value)/max(abs(value), 1e-18)
               for name, value in fine['observables'].items()}
    return {'status': 'DIAGNOSTIC_ONLY', 'relative_spectral_L1_by_detector': dict(zip(DETECTORS, relative_L1)),
        'relative_observable_changes': changes, 'maximum_relative_observable_change': max(changes.values()),
        'deterministic_error_interpretation': 'observed quadrature/refinement difference, not a rigorous continuum error bound',
        'intensity_and_spectral_1percent_diagnostic_pass': max(changes.values()) < .01 and np.max(relative_L1) < .01,
        'full_B9A_acceptance': False}


def analyze(config, output):
    cases = {}
    for name in ('born_cases', 'source_cases'):
        group = 'born' if name == 'born_cases' else 'source'
        for settings in config.get(name, []):
            path = output/group/settings['name']/'state.json'
            if path.exists(): cases[settings['name']] = json.loads(path.read_text())
    comparisons = {}
    for coarse, fine in config.get('comparisons', []):
        if coarse in cases and fine in cases:
            comparisons[coarse+' -> '+fine] = compare_states(cases[coarse], cases[fine])
    sampling = {}
    for label, names in config.get('replicate_groups', {}).items():
        chosen = [cases[name] for name in names if name in cases and cases[name]['status'] == 'COMPLETE']
        if len(chosen) != len(names) or len(chosen) < 2:
            sampling[label] = {'status': 'INCOMPLETE', 'completed_replicates': len(chosen), 'required': len(names)}
            continue
        values = np.array([[r['observables'][d] for d in DETECTORS] for r in chosen])
        se = np.std(values, axis=0, ddof=1)/np.sqrt(len(values))
        half = student_t.ppf(.975, len(values)-1)*se
        sampling[label] = {'status': 'ESTIMATED', 'independent_replicates': len(values),
            'mean_by_detector': dict(zip(DETECTORS, np.mean(values, axis=0))),
            'standard_error_by_detector': dict(zip(DETECTORS, se)),
            'student_t_95_half_width_by_detector': dict(zip(DETECTORS, half)),
            'raw_replicate_values': values.tolist(),
            'error_scope': 'randomized longitudinal quadrature only at the fixed declared qxy/electron model; excludes deterministic electron/transverse/model error'}
    result = {'status': 'INCOMPLETE', 'B7': 'INCOMPLETE', 'B9A_original_thick': 'INCOMPLETE',
        'cases': {name: {'status': row['status'], 'completed_q': row['completed_q'], 'total_q': row['total_q'],
            'thickness_nm': row.get('thickness_nm'), 'observables': row.get('observables'),
            'mean_seconds_per_q': row.get('mean_seconds_per_q'), 'checkpoint': str(output/('born' if row['contract']['born_only'] else 'source')/name/'state.json')}
            for name, row in cases.items()}, 'comparisons': comparisons,
        'statistical_longitudinal_error': sampling,
        'acceptance': {'unchanged_B9A_full_target': 'electron, longitudinal/transverse bath and joint refinements at52 repeats; all declared finite detector/window/spectral criteria; no diagnostic closes target',
            'original_B7': 'complete128^3 independent numerical source plus unchanged1% observables/L1,.2meV peaks and1e-4 amplitude'},
        'scope': 'fixed actual thickness and incident normalization; bath refinement never changes fixed elastic covariance or electron model; thin reference is distinct'}
    json_write(output/'integration_results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=('window', 'born', 'source', 'analyze', 'all'), default='all')
    parser.add_argument('--case')
    parser.add_argument('--max-new-q', type=int)
    args = parser.parse_args(); config = json.loads(Path(args.config).read_text())
    output = ROOT/config['output']; output.mkdir(parents=True, exist_ok=True)
    # Preserve the assigned maximum of two CPU cores, regardless of FFT/BLAS
    # defaults in a parent shell. Each numerical FFT here explicitly uses one.
    if hasattr(os, 'sched_setaffinity'):
        available = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, available[:2])
    limits = ResourceLimits(config.get('runtime_seconds', 600), config.get('memory_gib', 2), 100000)
    invocation = {'command': [sys.executable, *sys.argv], 'environment': environment(),
        'parameters': config, 'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'source_sha256': sha256(Path(__file__)), 'start_unix_seconds': time.time(), 'status': 'RUNNING'}
    invocation_path = output/'invocations'/f'{time.time_ns()}_{os.getpid()}.json'
    json_write(invocation_path, invocation)
    if args.stage in ('window', 'all'): window_diagnostic(config, output, limits)
    for group, stage in (('born_cases', 'born'), ('source_cases', 'source')):
        if args.stage not in (stage, 'all'): continue
        for settings in config.get(group, []):
            if args.case and settings['name'] != args.case: continue
            try:
                row = run_case(config, settings, output, limits, args.max_new_q, stage == 'born')
                print(settings['name'], row['status'], row['completed_q'], '/', row['total_q'], flush=True)
                if row['status'] != 'COMPLETE': break
            except (TimeoutError, MemoryError) as exc:
                invocation['stop_reason'] = str(exc); break
    result = analyze(config, output)
    invocation.update(status='STOPPED', resources=limits.costs(),
        completion_scope=result['status'], finish_unix_seconds=time.time())
    json_write(invocation_path, invocation)


if __name__ == '__main__':
    main()
