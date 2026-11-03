"""Postprocess immutable B7/B8/B11 data; never normalize EELS spectra to fit."""

import json
from pathlib import Path

import numpy as np
from scipy import constants
from scipy.integrate import trapezoid

from .workflow_io import ROOT, json_write, sha256


def read_run(name):
    return json.loads((ROOT/'benchmark_results'/name/'results.json').read_text())


def integration_audit(output, plt):
    cells = read_run('detector_cells_20261102_v1')
    nodal = read_run('repair_20261102_v1')
    result = {'status': 'INCOMPLETE', 'definitions': {
        'q': 'phonon angular wavevector, primitive reciprocal coordinates, normalized BZ measure d^3q/V_BZ',
        'Q': 'electron angular wavevector transfer G-q_perp; Qz=-(Q_perp^2/(2k)+omega/v), including at Q_perp=0',
        'angle': 'theta=1000 Q_perp/k in mrad, paraxial',
        'nodal_detector': 'sum_q w_q sum_branch,beam |unweighted amplitude|^2 indicator_D(theta)',
        'cell_detector': 'same weighted coefficients times area(C intersect D)/area(C); no second q or bin-area weight',
        'forward': 'the forward transverse image of BZ fibers, already part of the central detector; not an additional bulk delta mass',
        'normalization': 'incident plane wave 1/sqrt(A); mode displacement sqrt(hbar/(2 M omega)) e exp(iqR) sqrt(w_q); probabilities dimensionless',
        'point_density': 'forward fiber weight / associated angular cubature cell area, units probability/mrad^2; diagnostic estimate, not a detector probability',
        'genuine_point': 'a physical discrete point weight is summed with indicator_D and receives no angular area factor',
        'Gamma': 'unique BZ mesh; only three mass-weighted rigid translations removed; Gamma optical modes retained'},
        'comparisons': [], 'branch_and_q_attribution': [], 'forward_density': [],
        'energy_edges_mev': [0, 30, 38, 39, 39.5, 40, 40.5, 41, 42, 50, 60, 64, 64.5, 65, 65.5, 66, 75, 90],
        'energy_edge_interpretation': 'Unbroadened window indicators remain discontinuous at 40/65 meV; shell contributions diagnose this quadrature sensitivity, without changing the observable or acceptance tolerance.',
        'source_validation': cells['source_validation'], 'shifted_mesh_comparisons': []}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    previous = None
    for case in cells['cases']:
        n = case['settings']['mesh']
        folder = Path(case['artifact']).parent
        state = json.loads((folder/'state.json').read_text())
        spacing = state['geometric_checks']['cell_spacing_mrad']
        f = case['observables']['exact_forward_q_zero']
        result['forward_density'].append({'mesh': n, 'raw_forward_weight': f,
            'cell_area_mrad2': float(np.prod(spacing)), 'density_per_mrad2': f/np.prod(spacing)})
        # This cache is the explicitly labelled meV observable boundary of the
        # SI Born evaluator, not the SI transition array used by B9.
        energy = np.load(folder/'energies.npy', mmap_mode='r')
        weights = np.load(folder/'weights.npy', mmap_mode='r')
        branch = np.zeros(6)
        energy_bins = np.zeros(len(result['energy_edges_mev'])-1)
        shells = np.zeros(4)
        # bounded chunk processing; do not materialize all mode/beam arrays
        for start in range(0, n**3, 4096):
            stop = min(start+4096, n**3)
            e = energy[start:stop]
            w = weights[start:stop, :, 1]
            retained = (e >= 40) & (e < 65)
            branch += np.sum(w*retained, axis=0)
            energy_bins += np.histogram(e.ravel(), result['energy_edges_mev'], weights=w.ravel())[0]
            q = np.column_stack(np.unravel_index(np.arange(start, stop), (n, n, n)))/n
            radius = np.linalg.norm(q-np.rint(q), axis=1)
            shells += np.histogram(radius, [0, .1, .25, .5, 1], weights=np.sum(w*retained, axis=1))[0]
        result['branch_and_q_attribution'].append({'mesh': n, 'central_40_65_branch_weights': branch.tolist(),
            'central_energy_bin_weights': energy_bins.tolist(), 'central_40_65_q_shell_weights': shells.tolist(),
            'q_shell_edges_fractional_norm': [0, .1, .25, .5, 1],
            'branch_convention': 'local sorted eigenvalue index; not a band identity through degeneracies'})
        with np.load(case['artifact']) as data:
            axis = data['energy_grid_mev']
            for col, detector in enumerate(('central_0_5_mrad', 'annulus_5_10_mrad', 'annulus_10_20_mrad')):
                value = data['spectrum_'+detector]
                axes[0, col].plot(axis, value, label=f'cell N={n}')
                if previous is not None:
                    axes[1, col].plot(axis, value-previous[detector], label=f'{previous_n} to {n}')
            previous = {d: data['spectrum_'+d].copy() for d in ('central_0_5_mrad', 'annulus_5_10_mrad', 'annulus_10_20_mrad')}
            previous_n = n
    for comp in cells['comparisons']:
        change = comp['physical_detector_changes']
        worst = next(row for row in change['rows'] if row['observable'] == change['worst_observable'])
        result['comparisons'].append({'coarse': comp['coarse']['mesh'], 'fine': comp['fine']['mesh'],
            'L1': comp['spectral_changes'], 'worst_physical_observable': worst,
            'peak_shift_mev': comp['maximum_peak_shift_mev'], 'passed': comp['physical_passed']})
    # Existing frozen-source nodal results contain the independently computed shifted mesh.
    result['nodal_case_summaries'] = [{'settings': c.get('settings'),
        'central_40_65': c.get('observables', {}).get('central_0_5_mrad/middle_40_65_mev')}
        for c in nodal.get('cases', [])]
    unshifted = next((c for c in nodal.get('cases', []) if c.get('settings', {}).get('mesh') == 48
                      and not any(c.get('settings', {}).get('shift', []))), None)
    shifted = next((c for c in nodal.get('cases', []) if any(c.get('settings', {}).get('shift', []))), None)
    if unshifted and shifted:
        from .repair_convergence import compare_cases
        result['shifted_mesh_comparisons'].append(compare_cases(unshifted, shifted, nodal['parameters']))
    for col, title in enumerate(('central 0–5 mrad', 'annulus 5–10 mrad', 'annulus 10–20 mrad')):
        axes[0, col].set_title(title)
        for row in range(2):
            axes[row, col].set(xlim=(0, 90), xlabel='electron energy loss (meV)',
                               ylabel='probability / meV' if row == 0 else 'signed difference / meV')
            axes[row, col].legend(fontsize=7)
    fig.savefig(output/'B7_cells_and_absolute_differences.png', dpi=160)
    plt.close(fig)
    result['peak_interpretation'] = ('The central 40–65 meV group has its largest window value at the lower boundary, '
        'while distinct local maxima near 49 and 54.5 meV remain inside it. A boundary maximum is not an isolated mode. '
        'The original dominant-maximum diagnostic and local-maximum lists are retained in both run results. '
        'The final cell comparison shifts no dominant maximum; no change of feature is silently fitted away.')
    result['acceptance'] = ('64 to 96 passes physical intensity, L1 and peak targets, but 48 to 64 fails the '
        'central 40–65 meV intensity. Two successive passes and a subsequent joint numerical-source refinement '
        'are not demonstrated. No accepted full settings; old strict forward diagnostic still fails by construction.')
    json_write(output/'B7_diagnostics.json', result)
    return result


def repeated_audit(output, plt):
    data = read_run('repeated_20261102_v2')
    previous = read_run('repeated_20261102_v1')
    result = {'scope': 'synthetic two-mode spatial model', 'weak_scaling': [],
        'full_closed_norm_maximum_error': max(abs(c['observables']['norm']-1) for c in previous['cases']),
        'independent_controls': previous['independent_controls'], 'convergence': {
            k: previous[k] for k in ('channel_refinement', 'electron_refinement', 'longitudinal_refinement')},
        'single_transition_note': 'Hd retains channel-diagonal elastic dynamics; first order in K is not pure Born',
        'Dyson_note': 'Same-channel amplitude orders must be summed before squaring. Different final crystal states are added incoherently.'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for linear in (False, True):
        for length in (20, 80):
            rows = [r for r in data['spectra'] if r['linear'] == linear and r['thickness_nm'] == length]
            small, large = [next(r for r in rows if r['g'] == g) for g in (.01, .1)]
            exponent = np.log(large['full_inelastic_weight']/small['full_inelastic_weight'])/np.log(10)
            result['weak_scaling'].append({'linear': linear, 'thickness_nm': length,
                'g_values': [.01, .1], 'inelastic_log_slope': float(exponent),
                'first_order_probability_per_g_squared': [r['first_inelastic_weight']/r['g']**2 for r in (small, large)],
                'full_first_relative_spectrum_L1': [r['full_minus_first_spectral_L1_relative_to_full_inelastic'] for r in (small, large)]})
            label = f"{'linear' if linear else 'nonlinear'}, {length} nm"
            g = [r['g'] for r in rows]
            axes[0, 0].plot(g, [r['full_observables']['ground_population'] for r in rows], 'o-', label=label)
            axes[0, 1].plot(g, [r['full_observables']['two_or_more_population'] for r in rows], 'o-', label=label)
            axes[1, 0].plot(g, [r['full_observables']['offaxis_1_0/inelastic']-r['first_observables']['offaxis_1_0/inelastic'] for r in rows], 'o-', label=label)
            axes[1, 1].plot(g, [r['full_observables']['zero_beam/inelastic']-r['first_observables']['zero_beam/inelastic'] for r in rows], 'o-', label=label)
    for ax, title in zip(axes.flat, ('incident channel population', 'two or more phonons', 'off-axis (1,0): full minus first', 'forward beam: full minus first')):
        ax.set(xlabel='off-diagonal multiplier g', ylabel='absolute population / probability', title=title)
        ax.legend(fontsize=7)
    fig.savefig(output/'B8_populations_and_detector_departures.png', dpi=160)
    plt.close(fig)
    json_write(output/'B8_diagnostics.json', result)
    return result


def linewidth_audit(output, plt):
    data = read_run('anharmonic_validation_20261102_v1')
    source = read_run('anharmonic_20261102_v1')
    followup = read_run('anharmonic_followup_20261102_v1')
    result = {'status': 'INCOMPLETE', 'intrinsic_linewidth_status': 'NOT_RESOLVED',
        'trajectory_parameters': source['parameters'], 'actual_convergence': data['convergence'],
        'mode_matching': 'mass-weighted eigenvector assignment and degenerate subspace projectors, not sorted index alone',
        'thermostat': 'NVT equilibration, then unthermostatted NVE production; no production thermostat linewidth',
        'volume': data['volume_policy'], 'linewidth_convention': data['linewidth_convention'],
        'recurrence_indicators': [], 'sampling': {'cadence_fs': 5, 'production_duration_ps': 20,
            'Nyquist_mev': constants.h/(2*5e-15)/(1e-3*constants.elementary_charge),
            'rectangular_bin_spacing_mev': constants.h/20e-12/(1e-3*constants.elementary_charge)},
        'frequency_shift_uncertainty': followup['blocked_fit_uncertainty'],
        'force_model_limit': followup['failure_investigation'],
        'interpretation': 'Late coherence can indicate unresolved damping or finite-size recurrence; it is not assigned a unique lifetime. Existing frequency/linewidth gates are unchanged.'}
    fits = data['MD_mode_fits']
    result['setting_sensitivity'] = []
    for reference in data['frequency_rows']:
        temperature, q, group = reference['temperature_k'], reference['q_fractional'], reference['reference_mode_group']
        rows = [r for r in fits if r['temperature_k'] == temperature and r['q_fractional'] == q and r['group'] == group]
        groups = {
            'large_dt1_20ps': [r for r in rows if r['repetitions'] == [4, 6, 6] and r['block'] == 0],
            'small_dt1_20ps': [r for r in rows if r['repetitions'] == [3, 4, 4] and r['timestep_fs'] == 1 and r['block'] == 0],
            'small_dt0p5_20ps': [r for r in rows if r['timestep_fs'] == .5 and r['block'] == 0],
            'large_dt1_10ps_blocks': [r for r in rows if r['repetitions'] == [4, 6, 6] and r['block'] != 0]}
        summary = {name: {'count': len(values),
            'mean_center_mev': float(np.mean([v['center_mev'] for v in values])),
            'fitted_FWHM_range_mev': [float(min(v['fitted_intrinsic_FWHM_mev'] for v in values)), float(max(v['fitted_intrinsic_FWHM_mev'] for v in values))],
            'actual_temperature_range_k': [float(min(v['actual_mean_temperature_k'] for v in values)), float(max(v['actual_mean_temperature_k'] for v in values))],
            'window_FWHM_mev': float(values[0]['Hann_power_window_FWHM_mev'])}
            for name, values in groups.items() if values}
        result['setting_sensitivity'].append({'temperature_k': temperature, 'q': q, 'mode_group': group,
            'groups': summary, 'limitation': 'supercell and timestep ensembles have different actual NVE temperatures; these are combined sensitivity measurements, not pure equal-temperature convergence errors. Width fits below the declared resolution/uncertainty gate remain NOT_RESOLVED.'})
    folder = ROOT/'benchmark_results/anharmonic_validation_20261102_v1'
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for row, temperature in enumerate((300, 1000)):
        files = sorted(folder.glob(f'PSD_T{temperature}_q0_g3_traj*.npz'))
        for file in files:
            with np.load(file) as d:
                velocity = d['projected_velocities']
                n = len(velocity)
                transformed = np.fft.fft(velocity, n=2*n, axis=0)
                correlation = np.sum(np.fft.ifft(abs(transformed)**2, axis=0)[:n], axis=1)/(n-np.arange(n))
                correlation /= correlation[0].real
                stop = n//2
                late = float(np.max(abs(correlation[n//4:stop])))
                result['recurrence_indicators'].append({'artifact': str(file), 'sha256': sha256(file),
                    'late_normalized_coherence_max_5_10ps': late})
                axes[row, 0].plot(np.arange(stop)*.005, correlation[:stop].real, alpha=.6, label=file.stem.split('_')[-1])
                order = np.argsort(d['energy_mev'])
                energy, power = d['energy_mev'][order], d['velocity_PSD'][order]
                axes[row, 1].plot(energy, power/trapezoid(power, energy), alpha=.6)
        axes[row, 0].set(xlabel='time lag (ps)', ylabel='normalized mode velocity correlation', title=f'{temperature} K, optical Gamma subspace')
        axes[row, 1].set(xlim=(60, 85), xlabel='mode energy (meV)', ylabel='display-normalized mode power / meV', title='independent MD mode spectra')
        axes[row, 0].legend(fontsize=7)
    fig.savefig(output/'B11_mode_correlations_and_spectra.png', dpi=160)
    plt.close(fig)
    json_write(output/'B11_sampling_audit.json', result)
    return result


def main():
    output = ROOT/'benchmark_results/evidence_20261102_v1'
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = {'B7': integration_audit(output, plt), 'B8': repeated_audit(output, plt),
              'B11': linewidth_audit(output, plt)}
    json_write(output/'diagnostics.json', result)


if __name__ == '__main__':
    main()
