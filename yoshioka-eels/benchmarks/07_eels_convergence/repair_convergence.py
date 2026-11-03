"""Versioned B7 measure audit and bounded accelerated phonon refinements."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import constants
from scipy.integrate import trapezoid
from scipy.signal import find_peaks

from yoshioka.electron_phonon import (FirstOrderSourcePropagator, LobatoSilicon,
    numerical_transition_source, transverse_modes)
from yoshioka.free_propagation import TransverseGrid
from yoshioka.integration_measures import physical_observables
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from .born_quadrature import born_batch, checkpointed_born, line_spectra
from .bulk_eels import displacement_for_q, primitive_atom_mapping
from .eels_observables import (DETECTORS, PEAK_GROUPS, absolute_spectrum,
    compare_observables, spectrum_change)
from .workflow_io import ROOT, ResourceLimits, environment, json_write, sha256


def summarize_arrays(arrays, grid=np.arange(-10, 100.025, .05)):
    energies = np.asarray(arrays['energies'])
    weights = np.asarray(arrays['weights'])
    spectra = line_spectra(energies, weights, grid)
    values = {}
    peaks = {}
    features = {}
    for j, name in enumerate(DETECTORS):
        values[name] = float(np.sum(weights[:, :, j]))
        for group, (lo, hi) in PEAK_GROUPS.items():
            key = f'{name}/{group}'
            select = (energies >= lo) & (energies < hi)
            values[key] = float(np.sum(weights[:, :, j][select]))
            window = (grid >= lo) & (grid < hi)
            peaks[key] = float(grid[window][np.argmax(spectra[window, j])]) if np.any(window) and values[key] >= 1e-18 else None
            candidates, _ = find_peaks(spectra[:, j])
            candidates = candidates[window[candidates]]
            features[key] = [{'energy_mev': float(grid[k]), 'density_per_mev': float(spectra[k, j])}
                             for k in candidates[np.argsort(spectra[candidates, j])[::-1]][:8]]
    histogram = np.sum(arrays['momentum'], axis=0)
    for ix, iy in np.ndindex(8, 8):
        values[f'momentum_pixel_{ix}_{iy}'] = float(histogram[ix, iy])
    values['positive_qx_half'], values['negative_qx_half'] = map(float, np.sum(arrays['halves'], axis=0))
    values['exact_forward_q_zero'] = float(np.sum(arrays['forward']))
    return {'observables': values, 'peak_positions_mev': peaks, 'peak_features': features}, spectra


def compare_cases(coarse, fine, output):
    with np.load(coarse['artifact']) as a, np.load(fine['artifact']) as b:
        grid = b['energy_grid_mev']
        first = {name: a[f'spectrum_{name}'] for name in DETECTORS}
        second = {name: b[f'spectrum_{name}'] for name in DETECTORS}
    l1 = spectrum_change(first, second, grid)
    strict = compare_observables(coarse['observables'], fine['observables'])
    physical = compare_observables(physical_observables(coarse['observables']), physical_observables(fine['observables']))
    peaks = []
    for name, energy in fine['peak_positions_mev'].items():
        before = coarse['peak_positions_mev'][name]
        candidates = fine['peak_features'][name]
        nearest = min(candidates, key=lambda f: abs(f['energy_mev']-before)) if candidates and before is not None else None
        peaks.append({'peak': name, 'coarse_mev': before, 'fine_mev': energy,
                      'dominant_maximum_change_mev': abs(energy-before) if energy is not None and before is not None else None,
                      'nearest_fine_local_maximum': nearest,
                      'fine_local_maxima': candidates,
                      'missing_significant_peak': (before is None) != (energy is None)})
    maximum_shift = max((r['dominant_maximum_change_mev'] or 0 for r in peaks), default=0)
    common = l1['maximum_relative_L1'] < .01 and maximum_shift < .2 and not any(r['missing_significant_peak'] for r in peaks)
    result = {'coarse': coarse['settings'], 'fine': fine['settings'], 'spectral_changes': l1,
              'original_strict_observable_changes': strict, 'physical_detector_changes': physical,
              'peak_changes': peaks, 'maximum_peak_shift_mev': maximum_shift,
              'original_strict_passed': common and strict['maximum_acceptance_error'] < .01,
              'physical_passed': common and physical['maximum_acceptance_error'] < .01}
    return result


def source_validation(primitive, spec, output):
    """Independent inverse-FFT/FD potential and actual longitudinal source steps."""
    old = json.loads((ROOT/'benchmark_results/si_eels_convergence/results.json').read_text())
    final = old['cases'][-1]
    cache = Path(final['cache_directory'])
    weights = np.load(cache/'weights.npy', mmap_mode='r')
    energies = np.load(cache/'energy_mev.npy', mmap_mode='r')
    chosen = {0, 1, 32, 1024, 16, 512, 16384, 1057, 16912}
    for detector in range(4):
        chosen.add(int(np.argmax(np.sum(weights[:, :, detector], axis=1))))
    mid = (energies >= 40) & (energies < 65)
    chosen.add(int(np.argmax(np.sum(weights[:, :, 1]*mid, axis=1))))
    q = np.array(np.unravel_index(sorted(chosen), (32,)*3)).T/32
    q[q > .5] -= 1
    qw = np.full(len(q), 1/32**3)
    energy = spec['incident_energy_ev']*constants.elementary_charge
    pos = np.array(spec['positions_m'])
    covariance = np.array(spec['covariance_m2'])
    basis, _ = primitive_atom_mapping(primitive, pos)
    omega, eigen = primitive.modes(q)
    rows = []
    started = time.monotonic()
    for grid_size, samples, pad in ((192, 2048, 10.), (256, 4096, 12.), (320, 8192, 14.)):
        grid = TransverseGrid(*spec['transverse_lengths_m'], grid_size, grid_size)
        lengths = spec['transverse_lengths_m']
        length = spec['specimen_thickness_m']+2*pad*1e-10
        energies_fast, fast, fast_q, angles, physical = born_batch(primitive, pos, covariance, lengths, length, pad*1e-10, q, qw, energy)
        reciprocal = primitive.reciprocal_m_inv[:, :2]/(2*np.pi/np.array(lengths))
        kernels = {}
        for iq in range(len(q)):
            shift = (-q[iq] @ np.rint(reciprocal)) % 1.
            shift[shift > .5] -= 1.
            shift *= 2*np.pi/np.array(lengths)
            qxy, indices, theta = transverse_modes(grid, energy, 20, shift)
            selected = physical[iq]
            d = displacement_for_q(primitive, pos, basis, q[iq], omega[iq, selected], eigen[iq][:, selected], 32**3)
            padded = pos.copy()
            padded[:, 2] += pad*1e-10
            _, source = numerical_transition_source(LobatoSilicon.from_abtem_table(), padded, d, covariance, qxy, grid, length, samples, kernels)
            numerical = FirstOrderSourcePropagator(grid, energy, omega[iq, selected], indices, shift).propagate_scan(source, length)
            nearest = np.argmin(np.linalg.norm(qxy[:, None]-fast_q[iq, None], axis=-1), axis=1)
            if np.max(np.linalg.norm(qxy-fast_q[iq, nearest], axis=1)) > 1e-3:
                raise ValueError('Bloch beam representations disagree')
            reference = fast[iq, selected][:, nearest]
            error = numerical-reference
            for branch, branch_error, ref in zip(np.flatnonzero(selected), error, reference):
                norm = np.linalg.norm(ref)
                rows.append({'q_fractional': q[iq].tolist(), 'branch': int(branch), 'energy_mev': float(energies_fast[iq, branch]),
                             'grid': grid_size, 'samples': samples, 'padding_angstrom': pad,
                             'reference_weight': float(norm**2), 'complex_error_norm': float(np.linalg.norm(branch_error)),
                             'complex_relative_error': float(np.linalg.norm(branch_error)/norm) if norm else None,
                             'max_absolute_amplitude_error': float(np.max(abs(branch_error))),
                             'forward_amplitude_error': float(np.linalg.norm(branch_error[theta < 1e-8])),
                             'nearest_detector_boundary_mrad': float(np.min(abs(theta[:, None]-np.array([0, 5, 10, 20]))))})
    record = {'method': 'inverse FFT atomic potentials, fourth-order Cartesian FD, free/Simpson source propagation; accelerated path is a separate atomic sum',
              'rows': rows, 'elapsed_seconds': time.monotonic()-started,
              'selected_sectors': len(q), 'selection': 'Gamma, axial near-forward, zone faces, mixed q, strongest of each detector, strongest central middle-window q',
              'global_fine_relative_error': float(np.sqrt(sum(r['complex_error_norm']**2 for r in rows if r['grid'] == 320)/sum(r['reference_weight'] for r in rows if r['grid'] == 320)))}
    record['passed'] = record['global_fine_relative_error'] < 1e-4
    json_write(output/'source_validation.json', record)
    return record


def historical_audit(output):
    old_path = ROOT/'benchmark_results/si_eels_convergence/results.json'
    old = json.loads(old_path.read_text())
    failures = []
    for category, comparisons in old['comparisons'].items():
        for comparison in comparisons:
            if not comparison['passed']:
                failures.append({'category': category, **comparison})
    for key in ('joint_at_fixed_phonon_sampling', 'joint_electron_and_phonon_comparison'):
        if key in old and not old[key]['passed']:
            failures.append({'category': key, **old[key]})
    before = old['comparisons']['phonon_sampling'][-1]
    finite = [r for r in before['observable_changes']['rows'] if r['observable'] != 'exact_forward_q_zero']
    record = {'old_status': old['status'], 'old_results_sha256': sha256(old_path),
              'all_failed_comparisons': failures,
              'last_finite_observable_worst': max(finite, key=lambda r: r['acceptance_error']),
              'old_peak_features': [],
              'measure': {'BZ': 'normalized Haar measure d^3q/V_BZ, weights 1/Nq; no extra Gamma sample',
                          'phonon_displacement': 'sqrt(hbar/(2 M omega)) e exp(i q.R)/sqrt(Nq)',
                          'incident': '1/sqrt(A), electron Fourier amplitudes dimensionless; atomic h Fourier in J m^3; amplitude prefactor 1/(hbar v A)',
                          'electron_Q': 'Qperp=G_e-qperp (modulo Bloch sector), Qz=-(Qperp^2/(2k)+omega/v)',
                          'detector': 'sum_q sum_lambda sum_G |c_lambda,G(q)|^2 1_D(Qperp); q weight already in c; theta=Qperp/k; no second angular-bin factor',
                          'spectrum_units': 'incident-normalized inelastic probability per meV after normalized 2 meV FWHM Gaussian; unbroadened line weights dimensionless',
                          'genuine_point_masses': 'none in this declared bulk-BZ, one-phonon inelastic response; elastic Bragg point masses are outside this calculation',
                          'forward_diagnostic': 'subset of BZ nodes with represented Qperp exactly zero. Included once in central detector and momentum bins. It is not an additional detector probability or a density. Its transverse locus has measure zero in the continuous BZ; individual discrete-node weights shrink as 1/Nq.',
                          'forward_not_total_zero': 'Qz=-omega/v at Qperp=0; longitudinal polarization is retained',
                          'physical_acceptance': 'all original finite-area/window/bin/half-plane tests unchanged at 1%; exact-forward sample retained as diagnostic only; no weakened absolute criterion'},
              'constants': old['model_specification']['constants'], 'environment': environment()}
    for comparison in failures:
        if comparison['category'] != 'phonon_sampling':
            continue
        for settings in (comparison['coarse_settings'], comparison['fine_settings']):
            case = next(c for c in old['cases'] if c['settings'] == settings)
            with np.load(case['artifact']) as data:
                axis = data['energy_grid_mev']
                spectrum = data['spectrum_central_0_5_mrad']
                maxima, _ = find_peaks(spectrum)
                selected = maxima[(axis[maxima] >= 0) & (axis[maxima] < 40)]
                record['old_peak_features'].append({'settings': settings, 'central_acoustic_local_maxima':
                    [{'energy_mev': float(axis[k]), 'density_per_mev': float(spectrum[k])} for k in selected[np.argsort(spectrum[selected])[::-1]][:8]]})
    json_write(output/'historical_audit.json', record)
    return old, record


def write_outputs(results, old, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases = results['cases']
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), constrained_layout=True)
    for j, name in enumerate(DETECTORS):
        previous = None
        for case in cases:
            with np.load(case['artifact']) as d:
                axis, spectrum = d['energy_grid_mev'], d[f'spectrum_{name}']
            axes[j, 0].plot(axis, spectrum, label=f"{case['settings']['mesh']}³ shift {case['settings']['shift']}", lw=1)
            if previous is not None:
                axes[j, 1].plot(axis, abs(spectrum-previous), lw=1)
            previous = spectrum
        axes[j, 0].set_title(name)
        axes[j, 0].set_ylabel('probability / meV')
        axes[j, 1].set_title('Absolute successive spectral differences')
        axes[j, 1].set_ylabel('absolute probability difference / meV')
        for ax in axes[j]:
            ax.set_xlim(-5, 95)
            ax.set_xlabel('electron energy loss (meV)')
    axes[0, 0].legend(fontsize=7)
    fig.savefig(output/'detector_spectra_and_differences.png', dpi=160)
    plt.close(fig)
    # Historical spectra on identical axes; no independently normalized curves.
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), constrained_layout=True)
    nominal = [c for c in old['cases'] if c.get('category') == 'phonon_sampling']
    for j, name in enumerate(DETECTORS):
        previous = None
        for case in nominal:
            with np.load(case['artifact']) as d:
                axis, spectrum = d['energy_grid_mev'], d[f'spectrum_{name}']
            axes[j, 0].plot(axis, spectrum, label=f"{case['settings']['mesh']}³", lw=1)
            if previous is not None:
                axes[j, 1].plot(axis, abs(spectrum-previous), lw=1)
            previous = spectrum
        axes[j, 0].set_title(name)
        for ax in axes[j]:
            ax.set_xlabel('electron energy loss (meV)')
            ax.set_ylabel('absolute probability / meV')
    axes[0, 0].legend()
    fig.savefig(output/'historical_failed_spectra.png', dpi=160)
    plt.close(fig)
    text = ['# Benchmark 7: versioned momentum-measure audit and refinements', '',
            f"Status: **{results['status']}**. Historical results remain unchanged.", '',
            'The bulk phonon measure is d³q/VBZ. Each displacement contains sqrt(1/Nq), so squared amplitudes already contain the cubature weight. The electron Fourier coefficient is a population in a fixed-q Bloch sector; summing sectors integrates the bulk continuum. No second angular area factor is applied. The exact-forward sample is retained inside finite detectors and separately as a diagnostic. It is not a bulk point mass. Optical Gamma modes remain; only three rigid translations are omitted.', '',
            'All finite detector, window, momentum-bin and half-plane criteria retain the original 1% target. The original strict forward-sample diagnostic is reported alongside the corrected physical-detector decision. Gaussian FWHM stays 2 meV, peak target 0.2 meV, and no spectrum is rescaled.', '',
            '| Mesh comparison | max spectrum L1 | max finite-observable change | max peak shift meV | physical pass | old strict pass |',
            '|---|---:|---:|---:|---|---|']
    for row in results['comparisons']:
        text.append(f"| {row['coarse']['mesh']}→{row['fine']['mesh']} | {row['spectral_changes']['maximum_relative_L1']:.6g} | {row['physical_detector_changes']['maximum_acceptance_error']:.6g} | {row['maximum_peak_shift_mev']:.4g} | {row['physical_passed']} | {row['original_strict_passed']} |")
    text += ['', f"Independent numerical-source validation: relative complex error {results['source_validation']['global_fine_relative_error']:.8g}; per-mode and absolute errors are in source_validation.json.", '',
             'Each atomic checkpoint commits a complete q-index prefix after flushing all arrays. Compatibility includes every physical input, primitive force-constant bytes, mesh shift and algorithm hash. Timing limits do not enter this identity. Resume uses the same command; no partial sum is renormalized.', '',
             'The accelerated calculation uses the converged full-space atomic Born expression for the same finite specimen. Existing electron-grid/padding checks are preserved, and representative source refinements at 192/2048/10 Å, 256/4096/12 Å and 320/8192/14 Å are reported separately. A final complete joint material calculation is not inferred from this validation.', '',
             'Unbroadened absolute per-q branch/detector weights are in cache arrays. Spectrum deposition uses 0.0025 meV linear interpolation with L1 kernel-error bound <2.17e-6 of total weight; direct Gaussian checks are recorded. This numerical plotting error is separate from phonon integration.', '',
             '## Reproduction', '', '```bash', '.venv/bin/python -m benchmarks.repair_convergence --config benchmarks/configs/repair_20261102_v1.json', '```', '',
             '![spectra](detector_spectra_and_differences.png)', '', '![historical audit](historical_failed_spectra.png)']
    (output/'report.md').write_text('\n'.join(text)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='benchmarks/configs/repair_20261102_v1.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = ROOT/config['output']
    output.mkdir(parents=True, exist_ok=True)
    old, audit = historical_audit(output)
    spec = old['model_specification']
    primitive = RealSpaceForceConstants.load(ROOT/'benchmark_results/si_eels_convergence/primitive_force_constants.npz')
    validation_path = output/'source_validation.json'
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else source_validation(primitive, spec, output)
    if not validation['passed']:
        raise RuntimeError('accelerated Born route failed independent source validation')
    limits = ResourceLimits(config['runtime_limit_seconds'], config['memory_limit_gib'], 100)
    results = {'benchmark': 'Benchmark 7 repair', 'status': 'INCOMPLETE', 'parameters': config,
               'source_validation': validation, 'cases': [], 'comparisons': [], 'accepted_settings': None,
               'physical_measure_audited': True, 'historical_status': old['status']}
    for mesh, shift in [(m, [0, 0, 0]) for m in config['meshes']] + [(config['shifted_mesh'], [.5, .5, .5])]:
        directory, state, arrays = checkpointed_born(primitive, spec, mesh, shift, output, limits, batch_size=config['batch_size'])
        if state['status'] != 'COMPLETE':
            results['resource_stop'] = state
            break
        case, spectra = summarize_arrays(arrays)
        case.update(settings={'mesh': mesh, 'shift': shift, 'route': 'validated full-space Born cubature'},
                    checkpoint=state, artifact=str(directory/'summary.npz'))
        np.savez_compressed(case['artifact'], energy_grid_mev=np.arange(-10, 100.025, .05),
                            **{f'spectrum_{name}': spectra[:, j] for j, name in enumerate(DETECTORS)})
        results['cases'].append(case)
        if shift == [0, 0, 0] and len(results['cases']) > 1:
            results['comparisons'].append(compare_cases(results['cases'][-2], case, output))
        print(f"mesh {mesh} shift {shift}: P={case['observables']['full_0_20_mrad']:.10e}, cumulative source {state['elapsed_seconds']:.3f}s, invocation {state['latest_invocation_seconds']:.3f}s", flush=True)
        if results['comparisons']:
            row = results['comparisons'][-1]
            print(f"  L1 {row['spectral_changes']['maximum_relative_L1']:.6g}; finite max {row['physical_detector_changes']['maximum_acceptance_error']:.6g} at {row['physical_detector_changes']['worst_observable']}; peak {row['maximum_peak_shift_mev']:.4g} meV", flush=True)
        json_write(output/'checkpoint.json', results)
    results['phonon_physical_status'] = 'PASS' if len(results['comparisons']) >= 2 and all(r['physical_passed'] for r in results['comparisons'][-2:]) else 'INCOMPLETE'
    results['joint_refinement_status'] = 'INCOMPLETE: requires final joint source calculation after phonon acceptance'
    results['resource_costs'] = limits.costs()
    json_write(output/'results.json', results)
    write_outputs(results, old, output)
    json_write(output/'manifest.json', {'status': results['status'], 'source_sha256': sha256(__file__), 'config_sha256': sha256(args.config),
                                      'results_sha256': sha256(output/'results.json'), 'historical_results_sha256': audit['old_results_sha256']})


if __name__ == '__main__':
    main()
