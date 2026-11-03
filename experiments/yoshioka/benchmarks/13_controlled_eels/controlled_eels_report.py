"""Figures and measured interpretation of the controlled EELS comparison."""
import argparse
import json
from pathlib import Path
import platform
import numpy as np
from scipy.integrate import trapezoid

from .controlled_eels import DEFAULT, PREVIOUS, JOBS
from .controlled_eels_analysis import summarize, load_tacaw
from .approximate_eels_analysis import shape
from .approximate_eels import ORIENTED
from .workflow_io import ROOT, json_write, environment, sha256
from yoshioka.lattice_dynamics import RealSpaceForceConstants

COLORS = {'fixed': '#1766ac', 'fitted': '#d35c17'}


def weak_feature_diagnostics(output):
    """Preserve undefined maxima and candidate-dominance uncertainty explicitly."""
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    from .approximate_eels_analysis import features
    entries = []
    for label, path in [('fixed_r8', output/'fixed_r8_T300_spectra.npz'),
                        ('fitted_r8', output/'source64_evidence/fitted_r8_T300_spectra.npz'),
                        ('fitted_r8', output/'fitted_r8_T300_spectra.npz')]:
        if not path.exists(): continue
        with np.load(path) as saved:
            d = {key: saved[key].copy() for key in ('energy_mev', 'Born', 'DW', 'DW_pointwise95_halfwidth',
                'source_corrections', 'Gamma_corrections', 'q_fractional')}
        name = label+f'_q{d["q_fractional"].shape[1]}'
        if any(name == item[0] for item in entries): continue
        entries.append((name, d))
    if not entries: return []
    fig, axes = plt.subplots(2, len(entries), figsize=(5*len(entries), 7.5), squeeze=False, layout='constrained')
    rows = []
    for col, (name, d) in enumerate(entries):
        x = d['energy_mev']; rng = np.random.default_rng(789133)
        for row, (detector, feature, lo, hi) in enumerate(((2, 'acoustic', 10, 40), (3, 'middle', 40, 60))):
            selected = (x >= lo)&(x < hi); y = d['DW'][selected, detector]
            nominal = features(x, d['DW'][:, detector])[feature]
            found = []; undefined = 0
            for _ in range(500):
                sample = d['Born']+np.mean(d['source_corrections'][rng.integers(8, size=8)], axis=0)
                sample += np.mean(d['Gamma_corrections'][rng.integers(8, size=8)], axis=0)-np.mean(d['Gamma_corrections'], axis=0)
                f = features(x, sample[:, detector])[feature]
                if f['status'] == 'resolved_local_maximum': found.append(f['peak_mev'])
                else: undefined += 1
            peaks, props = find_peaks(y, prominence=0)
            bins, counts = np.unique(np.round(found, 1), return_counts=True)
            maxima = [{'energy_mev': float(x[selected][k]), 'height': float(y[k]),
                'prominence_fraction_of_window_max': float(p/max(y))} for k, p in zip(peaks, props['prominences'])]
            rows.append({'spectrum': name, 'detector': ('annulus_5_10_mrad' if detector == 2 else 'annulus_10_20_mrad'),
                'feature': feature, 'q_per_batch': d['q_fractional'].shape[1], 'nominal': nominal,
                'bootstrap_replicates': 500, 'bootstrap_seed': 789133, 'unresolved_replicates': undefined,
                'resolved_replicates': len(found), 'resolved_peak_bins_0_1mev': dict(zip(map(str, bins.tolist()), counts.tolist())),
                'unthresholded_local_maxima': maxima, 'unchanged_prominence_threshold': .05,
                'scope': 'Bootstrap peak identification/dominance sensitivity; not physical mode tracking or a calibrated frequency confidence interval. An undefined boundary fallback is not a physical peak.'})
            ax = axes[row, col]; h = d['DW_pointwise95_halfwidth'][selected, detector]
            ax.plot(x[selected], y, label='mean'); ax.fill_between(x[selected], y-h, y+h, alpha=.2, label='pointwise 95% band')
            for k, p in zip(peaks, props['prominences']):
                ax.plot(x[selected][k], y[k], 'o' if p >= .05*max(y) else 'x', color='black' if p >= .05*max(y) else '.5')
            ax.set(xlabel='Energy loss (meV)', ylabel='Raw model weight / meV',
                title=f'{name}: {feature}\nUndefined in {undefined}/500 resamples')
            ax.grid(alpha=.15); ax.legend(fontsize=7)
    fig.suptitle('Weak off-axis feature diagnosis; existing 5% prominence rule retained\nCircles pass the prominence rule; crosses do not. No additional smoothing or energy-window change.', fontsize=10)
    fig.savefig(output/'weak_feature_diagnosis.png', dpi=160); plt.close(fig)
    json_write(output/'weak_feature_diagnosis.json', {'rows': rows})
    return rows


def figures(output, result, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for temperature in (300, 1000):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
        fig.subplots_adjust(left=.075, right=.98, bottom=.13, top=.77, wspace=.25)
        for model in ('fixed', 'fitted'):
            key = f'{model}_r4_T{temperature}'
            if key not in data: continue
            d = data[key]; axis = d['axis']; selected = (axis >= 5)&(axis < 90)
            for method, style in [('born', '--'), ('mean', '-')]:
                label = ('Fixed harmonic' if model == 'fixed' else 'Temperature-fitted, diagnostic')+' / '+('Born' if method == 'born' else 'distorted wave')
                y = d[method]
                axes[0].plot(axis, y[:, 0], style, color=COLORS[model], lw=1.4, label=label)
                conditional, _ = shape(axis, y[:, 0]); axes[1].plot(axis[selected], conditional[selected], style, color=COLORS[model], lw=1.4, label=label)
                if method == 'mean': axes[0].fill_between(axis, y[:, 0]-d['halfwidth'][:, 0], y[:, 0]+d['halfwidth'][:, 0], color=COLORS[model], alpha=.13)
        if f'fixed_r4_T{temperature}' in data:
            axis = data[f'fixed_r4_T{temperature}']['axis']; selected = (axis >= 5)&(axis < 90)
            for name, values in load_tacaw(temperature, axis).items():
                y, _ = shape(axis, values)
                axes[1].plot(axis[selected], y[selected], color='black' if name == 'all_recorded_momenta' else '.5',
                    ls='-' if name == 'all_recorded_momenta' else ':', label='TACAW: '+('all recorded momenta' if name == 'all_recorded_momenta' else '20-pixel disk'))
        axes[0].set(xlim=(-90, 90), xlabel='Electron energy loss (meV)', ylabel='Raw model inelastic weight / meV', title='Raw signed spectra + pointwise 95% bands')
        axes[1].set(xlabel='Energy loss (meV), TACAW axis conditional', ylabel='Conditional shape / meV', title='Conditional loss shapes (5–90 meV)')
        for ax in axes: ax.legend(fontsize=6); ax.grid(alpha=.15)
        fig.suptitle(f'Si [110], {temperature} K'+(' — exploratory' if temperature == 1000 else '')+'\n1.536 nm · 60 keV · Yoshioka 0–20 mrad · 2 meV FWHM · elastic line excluded\nTACAW energy calibration, acceptance and native resolution remain unmatched.', fontsize=10, y=.98)
        fig.savefig(output/f'four_models_T{temperature}.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for col, model in enumerate(('fixed', 'fitted')):
        for repeats in (4, 8):
            key = f'{model}_r{repeats}_T300'
            if key not in data: continue
            d = data[key]; axis = d['axis']; selected = (axis >= 5)&(axis < 90); color = '#1766ac' if repeats == 4 else '#bb3b35'
            for method, style in [('born', '--'), ('mean', '-')]:
                label = f'{1.5355530976*repeats/4:.3f} nm / '+('Born' if method == 'born' else 'distorted wave')
                axes[0, col].plot(axis, d[method][:, 0], style, color=color, label=label)
                y, _ = shape(axis, d[method][:, 0]); axes[1, col].plot(axis[selected], y[selected], style, color=color, label=label)
        axes[0, col].set(xlim=(-90, 90), ylabel='Raw model weight / meV', title=model+' phonons')
        axes[1, col].set(xlabel='Electron energy loss (meV)', ylabel='Conditional 5–90 meV shape / meV')
    for ax in axes.flat: ax.legend(fontsize=7); ax.grid(alpha=.15)
    fig.suptitle('300 K physical thickness sensitivity; no thickness renormalization of raw weights\nFixed transverse cell, force model, detectors and resolution; this is not a numerical convergence test.', fontsize=11)
    fig.savefig(output/'thickness_T300.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for j, ax in enumerate(axes.flat):
        for model in ('fixed', 'fitted'):
            for repeats in (4, 8):
                key = f'{model}_r{repeats}_T300'
                if key not in data: continue
                d = data[key]; ax.plot(d['axis'], d['mean'][:, j], '-' if repeats == 4 else '--', color=COLORS[model], label=f'{model}, {1.5355530976*repeats/4:.3f} nm')
        ax.set(xlim=(-90, 90), xlabel='Energy loss (meV)', ylabel='Raw model weight / meV', title=result['curves'][0]['detectors'][j]['detector'].replace('_', ' '))
        ax.legend(fontsize=7); ax.grid(alpha=.15)
    fig.suptitle('300 K distorted-wave detector sensitivity; physical angular boundaries unchanged', fontsize=11)
    fig.savefig(output/'detectors_T300.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout='constrained')
    for name, rows in result['numerical_checks'].items():
        if not rows: continue
        q = [r for r in rows if r['kind'].startswith('Born')]
        axes[0].semilogy([r['fine_mesh'] for r in q], [r['maximum_shape_L1'] for r in q], 'o-', label=name)
        s = [r for r in rows if r['kind'].startswith('independent')]
        axes[1].semilogy([r['fine_q_per_batch'] for r in s], [r['maximum_shape_L1'] for r in s], 'o-', label=name)
        r = [r for r in rows if r['kind'] in ('electron', 'slices', 'joint', 'covariance64')]
        axes[2].semilogy([r0['kind'] for r0 in r], [max(r0['maximum_shape_L1'], 1e-16) for r0 in r], 'o-', label=name)
    for ax in axes:
        ax.axhline(.1, ls=':', color='red', label='10% target' if ax is axes[0] else None)
        ax.set(ylabel='Maximum conditional shape L1 across detectors'); ax.grid(alpha=.15)
    axes[0].set(xlabel='Fine Born q mesh per axis', title='Deterministic q integration')
    axes[1].set(xlabel='Fine q count per independent batch', title='Nested source counts')
    axes[2].set(title='Matched production-q sensitivities')
    axes[0].legend(fontsize=6); fig.savefig(output/'numerical_convergence.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), layout='constrained')
    pairs = [('fixed_r4_T300', 'fitted_r4_T300'), ('fixed_r4_T1000', 'fitted_r4_T1000'),
        ('fixed_r4_T300', 'fixed_r8_T300'), ('fitted_r4_T300', 'fitted_r8_T300')]
    for ax, (a, b) in zip(axes.flat, pairs):
        p = output/(a+'_to_'+b+'_paired.npz')
        if not p.exists(): continue
        with np.load(p) as d:
            x = d['energy_mev']; delta = d['signed_DW_difference'][:, 0]; h = d['pointwise95_halfwidth'][:, 0]
            ax.plot(x, delta); ax.fill_between(x, delta-h, delta+h, alpha=.2)
        ax.axhline(0, color='.5', lw=.5); ax.set(xlim=(-90, 90), xlabel='Energy loss (meV)', ylabel='Signed raw difference / meV', title=b+' minus '+a); ax.grid(alpha=.15)
    fig.suptitle('Paired common-q differences; pointwise 95% bands\nSource and Gamma variances treated separately; model and deterministic quadrature biases excluded.', fontsize=10)
    fig.savefig(output/'paired_differences.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for col, temperature in enumerate((300, 1000)):
        for model in ('fixed', 'fitted'):
            for repeats in ((4, 8) if temperature == 300 else (4,)):
                key = f'{model}_r{repeats}_T{temperature}'
                if key not in data: continue
                d = data[key]; x = d['axis']; delta = np.mean(d['corrections'], axis=0)[:, 0]
                from scipy.stats import t as student_t
                half = student_t.ppf(.975, 7)*np.std(d['corrections'], axis=0, ddof=1)[:, 0]/np.sqrt(8)
                axes[col].plot(x, delta, '-' if repeats == 4 else '--', color=COLORS[model], label=f'{model}, {1.5355530976*repeats/4:.3f} nm')
                axes[col].fill_between(x, delta-half, delta+half, color=COLORS[model], alpha=.1)
        axes[col].axhline(0, color='.5', lw=.5); axes[col].set(xlim=(-90, 90), xlabel='Energy loss (meV)', ylabel='Signed DW minus Born / meV', title=f'{temperature} K'+(' exploratory' if temperature == 1000 else ''))
        axes[col].legend(fontsize=7); axes[col].grid(alpha=.15)
    fig.suptitle('Elastic diffraction correction at fixed phonon model; pointwise 95% source bands\nShared Born and Gamma uncertainty cancels; this signed correction is not a repeated-inelastic probability.', fontsize=10)
    fig.savefig(output/'electron_diffraction_correction.png', dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    for name, rows in result['numerical_checks'].items():
        values = [r for r in rows if r['kind'].startswith('independent')]
        if not values: continue
        values = [values[0]['coarse_statistics']]+[r['fine_statistics'] for r in values]
        ax.plot([r['q_per_batch'] for r in values], [r['loss_weight_5_90_student95_halfwidth'][0]/r['loss_weight_5_90'][0] for r in values], 'o-', label=name)
    ax.set(xlabel='q per independent source batch (8 batches)', ylabel='95% half-width / integrated 5–90 meV weight', title='Statistical source-count sensitivity, full 0–20 mrad detector')
    ax.legend(fontsize=7); ax.grid(alpha=.15); fig.savefig(output/'statistical_convergence.png', dpi=160); plt.close(fig)


def report(output):
    result, data = summarize(output); figures(output, result, data)
    weak = weak_feature_diagnostics(output)
    num_rows = []
    for name, rows in result['numerical_checks'].items():
        if not rows: continue
        selected = [r for r in rows if (r['kind'].startswith('Born') and r['fine_mesh'] == 64)
            or (r['kind'].startswith('independent') and r['fine_q_per_batch'] == result['uncertainty'][name]['q_per_batch'])
            or r['kind'] in ('electron', 'slices', 'joint', 'covariance64')]
        for r in selected:
            resolved_shifts = [d['feature_shifts_mev'][key] for d in r['rows'] for key in d['feature_shifts_mev']
                if d['coarse_features'][key]['status'] == d['fine_features'][key]['status'] == 'resolved_local_maximum']
            num_rows.append({'job': name, 'kind': r['kind'], 'maximum_shape_L1': r['maximum_shape_L1'],
                'maximum_feature_change_mev': r['maximum_peak_shift_mev'],
                'maximum_resolved_pair_shift_mev': max(resolved_shifts, default=None),
                'all_identifiable_pairs_within_1mev': all(value <= 1 for value in resolved_shifts),
                'original_all_detector_peak_gate_pass': all(d['peak_pass'] for d in r['rows']),
                'primary_shape_pass': r['rows'][0]['shape_pass'], 'primary_peak_pass': r['rows'][0]['peak_pass'],
                'all_detectors_shape_pass': all(d['shape_pass'] for d in r['rows']),
                'unresolved_features': {d['detector']: d['unresolved_features'] for d in r['rows'] if d['unresolved_features']},
                'detector_feature_sensitivities': {d['detector']: d['feature_shifts_mev'] for d in r['rows']}})
    json_write(output/'finest_numerical_checks.json', num_rows)
    costs = {'new_job_invocations': {}, 'pipeline_invocations': {}, 'previous_compatible_results': str(PREVIOUS.relative_to(ROOT)),
        'scope': 'Per-invocation wall and peak process RSS; some independent jobs overlap. Imported cache work is separated from new invocation work. TACAW generation cost unknown.'}
    for name in JOBS:
        p = output/name/'invocations.json'
        if p.exists(): costs['new_job_invocations'][name] = json.loads(p.read_text())
        p = output/f'pipeline_{name}.json'
        if p.exists(): costs['pipeline_invocations'][name] = json.loads(p.read_text())
    costs['pilot_setup_and_q_work'] = {}
    costs['cached_work'] = {}; costs['independent_source_controls'] = {}
    for name in JOBS:
        p = output/name/'pilot.json'
        if p.exists():
            pilot = json.loads(p.read_text())
            costs['pilot_setup_and_q_work'][name] = {'cold_pilot_wall_seconds': pilot['seconds'],
                'committed_q_work_seconds': pilot['committed_q_seconds'],
                'setup_and_other_overhead_seconds': pilot['seconds']-pilot['committed_q_seconds'],
                'estimated512q_seconds_at_pilot_load': pilot['estimated_512_q_seconds'],
                'scope': 'Residual setup time includes input/array setup and overhead; committed q work includes field construction, propagation and output commits.'}
        config = json.loads((output/'configs'/f'{name}.json').read_text())
        active = []
        for case in config['source_cases']:
            state = json.loads((output/name/'source'/case['name']/'state.json').read_text())
            active.append({'case': case['name'], 'completed_q': state['completed_q'],
                'new_state_committed_seconds': state['cumulative_seconds'],
                'imported_prefix_seconds': state.get('prefix_import', {}).get('reused_computation_seconds', 0.)})
        costs['cached_work'][name] = {'active_source_cases': active,
            'total_source_work_including_imported_prefix_seconds': sum(v['new_state_committed_seconds']+v['imported_prefix_seconds'] for v in active),
            'Born_states': {str(p.parent.name): json.loads(p.read_text())['cumulative_seconds'] for p in (output/name/'born').glob('*/state.json')},
            'note': 'Imported prior Born work is identified by reuse_provenance.json; these cumulative values are not added to new invocation wall times.'}
        p = output/(name+'_independent_source.json')
        if p.exists(): costs['independent_source_controls'][name] = json.loads(p.read_text())['resources']
    records = [r for rows in costs['new_job_invocations'].values() for r in rows]
    costs['sum_new_stage_invocation_seconds'] = sum(r['seconds'] for r in records)
    costs['maximum_recorded_process_rss_mib'] = max([r['resources']['peak_process_rss_mib'] for r in records]
        +[r['peak_process_rss_mib'] for r in costs['independent_source_controls'].values()], default=0)
    json_write(output/'cost_vs_scope.json', costs)
    json_write(output/'environment.json', {'environment': environment(), 'platform': platform.platform(),
        'inherited_hardware': json.loads((PREVIOUS/'hardware.json').read_text()),
        'constants_and_physics_reference': str(PREVIOUS/'constants_and_physics.json'),
        'constants_and_physics_reference_sha256': sha256(PREVIOUS/'constants_and_physics.json')})
    tests = json.loads((output/'regression.json').read_text()) if (output/'regression.json').exists() else {'status': 'PENDING'}
    preservation = json.loads((output/'protected_after.json').read_text()) if (output/'protected_after.json').exists() else {'status': 'PENDING'}
    primary_ok = all(r['primary_shape_pass'] and r['primary_peak_pass'] for r in num_rows)
    shape_ok = all(r['all_detectors_shape_pass'] for r in num_rows)
    resolved_ok = all(r['all_identifiable_pairs_within_1mev'] for r in num_rows)
    undefined = {}
    for r in num_rows:
        for detector, keys in r['unresolved_features'].items():
            target = undefined.setdefault(r['job'], {}).setdefault(detector, [])
            target[:] = sorted(set(target) | set(keys))
    result['finest_acceptance'] = {'primary_all_checks_pass': primary_ok, 'all_detector_shape_checks_pass': shape_ok,
        'all_identifiable_feature_pairs_within_1mev': resolved_ok,
        'original_all_detector_peak_gates_all_pass': all(r['original_all_detector_peak_gate_pass'] for r in num_rows),
        'unresolved_features_by_job': undefined,
        'scope': 'Original per-observable gates retained; undefined peaks do not become a PASS. The primary0–20mrad spectrum is reported separately.'}
    json_write(output/'comparison.json', result)
    lines = ['# Controlled silicon EELS comparison', '',
        f'Computation status: **{result["status"]}**. Only complete production sources are displayed; incomplete numerical refinements are not accepted.', '',
        f'Finest checks for the new jobs: primary0–20mrad shape/peak targets met={primary_ok}; all detector shape targets met={shape_ok}; all identifiable feature pairs within1meV={resolved_ok}. Full all-detector peak acceptance remains unestablished wherever a window has no identifiable maximum. Its original peak-pass flag is not changed.', '',
        'The four-model comparison separates the phonon model from electron diffraction. All models retain bulk-q scattering matrix elements, all six primitive branches, Bose emission/absorption and one linear inelastic source. Distorted-wave propagation includes repeated elastic diffraction; it does not add repeated inelastic events.', '',
        '300 K is primary. The fitted harmonic candidates retain Benchmark 11A FAIL; 1000 K remains exploratory. All calculations use the same fixed equilibrium volume, 60 keV, transverse unit cell, physical detectors and 2 meV display FWHM.', '',
        '## Four models and physical thickness', '',
        '| T (K) | Repeats | Phonons | Electron model | Optical feature (meV) | Conditional TACAW shape L1 | Raw loss weight, 5–90 meV | Gain/loss, 5–90 meV |',
        '|---:|---:|---|---|---:|---:|---:|---:|']
    for r in result['curves']:
        d = r['detectors'][0]; f = d['features']['optical']
        lines.append(f'| {r["temperature_k"]} | {r["oriented_repeats"]} | {r["phonon_model"]} | {r["method"]} | {f["peak_mev"]:.2f} | {d["conditional_TACAW_shape_L1"]:.4f} | {d["weight_5_90"]:.7g} | {d["gain_over_loss_5_90"]:.4f} |')
    lines += ['', 'Primary-detector raw, unbroadened transition-window weights:', '',
        '| Curve | 0–40 meV | 40–65 meV | 65–90 meV |', '|---|---:|---:|---:|']
    for r in result['curves']:
        raw = r['detectors'][0]['raw_unbroadened_window_weights']
        lines.append(f'| {r["id"]} | {raw["loss_0_40"]:.5g} | {raw["loss_40_65"]:.5g} | {raw["loss_65_90"]:.5g} |')
    lines += ['', 'Identifiable feature maxima in the primary detector (full matched/unmatched lists and undefined statuses are retained in comparison.json):', '',
        '| Curve | Acoustic (meV) | Middle (meV) | Optical (meV) |', '|---|---:|---:|---:|']
    for r in result['curves']:
        f = r['detectors'][0]['features']
        text = [f'{f[k]["peak_mev"]:.2f}' if f[k]['status'] == 'resolved_local_maximum' else 'unresolved' for k in ('acoustic', 'middle', 'optical')]
        lines.append('| '+r['id']+' | '+' | '.join(text)+' |')
    lines += ['', 'Contributions to the global conditional TACAW shape L1, using the same global 5–90 meV conditioning (regions are not independently normalized):', '',
        '| Curve | 5–40 meV | 40–65 meV | 65–90 meV |', '|---|---:|---:|---:|']
    for r in result['curves']:
        p = r['detectors'][0]['conditional_TACAW_shape_L1_region_contributions']
        lines.append(f'| {r["id"]} | {p["acoustic_5_40"]:.4f} | {p["middle_40_65"]:.4f} | {p["optical_65_90"]:.4f} |')
    primary = {r['id']: r['detectors'][0] for r in result['curves']}
    if all(name in primary for name in ('fixed_r4_T300_DW', 'fitted_r4_T300_DW', 'fixed_r8_T300_DW', 'fitted_r8_T300_DW')):
        a, b = primary['fixed_r4_T300_DW'], primary['fitted_r4_T300_DW']
        lines += ['', f'The missing fitted300K distorted-wave model has optical feature {b["features"]["optical"]["peak_mev"]:.2f}meV and conditional TACAW shape L1={b["conditional_TACAW_shape_L1"]:.4f}, versus {a["features"]["optical"]["peak_mev"]:.2f}meV and {a["conditional_TACAW_shape_L1"]:.4f} for fixed-harmonic distorted waves. Completing the electron treatment does not reverse the optical-position versus whole-shape trade-off.', '',
            'Measured300K physical thickness changes:', '',
            '| Phonons | Conditional shape L1, thin versus thick | Thick/thin raw loss weight | Thin→thick optical maximum(meV) |', '|---|---:|---:|---|']
        for model in ('fixed', 'fitted'):
            first, second = primary[model+'_r4_T300_DW'], primary[model+'_r8_T300_DW']
            physical = next(r for r in result['physical_comparisons'] if r['first'] == model+'_r4_T300' and r['second'] == model+'_r8_T300' and r['method'] == 'mean')
            lines.append(f'| {model} | {physical["rows"][0]["shape_L1"]:.4f} | {second["weight_5_90"]/first["weight_5_90"]:.4f} | {first["features"]["optical"]["peak_mev"]:.2f}→{second["features"]["optical"]["peak_mev"]:.2f} |')
        lines += ['', 'The1.536nm specimen remains the cheaper qualitative reference: the tested doubling changes the primary conditional shape by only a few percent and optical maxima by about0.2meV, while raw weights change substantially. This two-thickness comparison does not establish an infinite-thickness limit. Thickness reweights mode contributions; it does not change the fixed model frequencies.']
    oriented = RealSpaceForceConstants.load(ORIENTED)
    lines += ['', f'Four/eight repeats are {4*oriented.cell_m[2,2]*1e9:.9f}/{8*oriented.cell_m[2,2]*1e9:.9f} nm, with 16/32 explicit atoms. Both include 12 Å vacuum on each side. The primary grid is 49². Steps are 385/545, chosen to keep physical dz approximately fixed; each has independent slice and joint tests. Thickness is a change of physical specimen, not numerical convergence.', '',
        'The thin fixed-harmonic results are reconstructed from immutable compatible caches. The new fitted thin baseline is refined through 64³; its Born result can differ from the older 32³ diagnostic. Historical values remain in their original report.', '',
        '## Numerical acceptance and statistical uncertainty', '',
        '| Job | Finest check | Max detector shape L1 | Max candidate/edge change (meV) | Max resolved-pair shift (meV) | Primary shape / identifiable peaks |', '|---|---|---:|---:|---:|---|']
    for r in num_rows:
        resolved_text = f'{r["maximum_resolved_pair_shift_mev"]:.3g}' if r['maximum_resolved_pair_shift_mev'] is not None else 'unresolved'
        lines.append(f'| {r["job"]} | {r["kind"]} | {r["maximum_shape_L1"]:.5g} | {r["maximum_feature_change_mev"]:.3g} | {resolved_text} | {r["primary_shape_pass"]} / {r["primary_peak_pass"]} |')
    lines += ['', 'All detector rows, historical/coarser numerical comparisons, unresolved windows and competing maxima remain in comparison.json. A missing local maximum does not establish a stable peak. Matching source subsets measure numerical sensitivity; they are not full fine-grid production spectra or rigorous BZ-wide error bounds.', '',
        'Inherited fixed-harmonic thin1000K covariance sensitivity is preserved explicitly: the weak10–20mrad middle-window dominant maximum switches by3.1meV despite only0.052% shape change. Its primary0–20mrad features remain stable. The original fixed-model records are under inherited_fixed_r4_numerical_evidence; new fitted-model checks do not erase this flag.', '',
        'The numerical targets remain 1 meV stability of identifiable dominant features and 10% conditional shape L1 over 5–90 meV. No detector/window or smoothing change is used. Raw transition windows 0–40, 40–65 and 65–90 meV are separate from feature-search windows 10–40, 40–60 and 60–85 meV.', '',
        '| Job | Independent source batches × q | Full-detector integrated pointwise 95% half-width / signal |', '|---|---:|---:|']
    for name, u in result['uncertainty'].items(): lines.append(f'| {name} | 8 × {u["q_per_batch"]} | {u["integrated_pointwise95_halfwidth_over_weight_by_detector"][0]:.3%} |')
    lines += ['', '| Job | DW−Born loss weight, 5–90 meV | 95% half-width of that integral |', '|---|---:|---:|']
    for name, u in result['uncertainty'].items():
        lines.append(f'| {name} | {u["DW_minus_Born_weight_5_90"][0]:.7g} | {u["DW_minus_Born_weight_5_90_pointwise_student95_halfwidth"][0]:.4g} |')
    lines += ['', 'Common q nodes and inverse sampling densities are verified before model/thickness pairing. Source and local-Gamma scrambles are independent, while matched pairs preserve cross-model covariance. Pointwise Student-t intervals and 500 batch-bootstrap replicates describe finite-eight-batch sensitivity. They exclude force-model error, deterministic Born bias and reference uncertainty. No confidence interval is invented for the averaged TACAW arrays.', '',
        'The affected thicker fitted model receives an additional64→128q/batch check. Its64q arrays/reports remain in source64_evidence and the original source folders. New raw integrand arrays import the identical nested64q prefix without a normalization change; only the additional q sectors are propagated. Model/thickness pairs with unequal final q counts use the verified common prefix and differences of the eight matched batch integrals.', '',
        '## Weak off-axis features', '',
        'The original fitted eight-repeat32→64q comparison changes its10–20mrad middle-window strongest maximum from57.80 to55.75meV. Its5–10mrad acoustic maximum becomes unidentifiable; the numerical boundary fallback at40meV is not an acoustic transition. This justifies the targeted128q refinement. Original peak-pass flags are preserved, including failures/undefined windows; full shapes can converge while a weak dominant-peak label remains unstable.', '',
        '| Spectrum | Detector / feature | Nominal status | Unresolved bootstrap resamples / 500 |', '|---|---|---|---:|']
    for row in weak: lines.append(f'| {row["spectrum"]} | {row["detector"]} / {row["feature"]} | {row["nominal"]["status"]} | {row["unresolved_replicates"]} |')
    lines += ['', 'The bootstrap diagnoses identification/dominance sensitivity, not a physical phonon-frequency shift. Every unthresholded local maximum, prominence, and resolved-candidate histogram is in weak_feature_diagnosis.json.', '',
        '## Physical interpretation', '',
        'Electron diffraction and phonon modelling answer different questions. Compare Born versus distorted waves within each phonon model, then fixed versus fitted phonons within each electron treatment. Thickness comparisons retain their physical intensity changes; raw spectra are not divided by thickness.', '',
        'The original fitted Born curves improved the optical feature while worsening conditional whole-spectrum agreement. The new complete matrix, acoustic/middle/optical windows and matched/unmatched feature lists determine whether distorted propagation changes that trade-off. Matching one optical maximum does not identify the same mode or validate all force constants.', '',
        'There is no established overall accuracy winner from these conditional TACAW comparisons. Use the fixed-harmonic distorted-wave result as the reproducible reference and the temperature-fitted distorted-wave result as a temperature-dependent model-sensitivity candidate. Both should accompany a 300 K interpretation until independent phonon/reference validation supports a stronger choice.', '',
        '## Reference and phonon-model limits', '',
        'TACAW retains the user-supplied 60 keV, 208 slices, unit-integral normalization, already-applied Kubo correction and tentative supplied Tersoff identification. The 7.5 fs energy axis, physical angular acceptance, native resolution and geometry remain unverified. All-momentum and 20-pixel-disk reductions are shown separately. The assumed bin spacing is about 0.919 meV; sub-bin displayed agreement does not establish that physical precision.', '',
        'Each conditional shape is divided by its own 5–90 meV integral, with raw model weights and original recorded TACAW fractions retained separately. No fitted energy shift, intensity scale, elastic subtraction, second Kubo correction or claim of matched absolute intensities is made.', '',
        'Corrected-MD fits are reused without fitting to EELS. Their actual training/validation temperatures and fixed volume remain recorded in B11_refit. The latest models pass 9/10 frequency groups at 300 K and 6/10 at 1000 K, with largest discrepancies 0.546/2.257 meV. The original force/frequency acceptance criteria and FAIL status remain unchanged. Intrinsic widths and lifetimes are not inferred.', '',
        'A justified follow-up, if quantitative 1000 K modelling becomes necessary, is a bounded higher-order-force pilot on independent existing displacement–force blocks, validating both held-out forces and the failing mode-resolved MD subspace. An improvement in force fit alone would not validate renormalized frequencies. Thermal expansion should be a separate physical branch. None of these extensions is launched here.', '',
        '## Resources, tests and preservation', '',
        f'Sum of new stage invocation work: {costs["sum_new_stage_invocation_seconds"]/60:.2f} minutes. Maximum recorded process RSS: {costs["maximum_recorded_process_rss_mib"]:.2f} MiB. These component times may overlap and are not end-to-end turnaround. Each calculation invocation uses the existing 600 s / 2 GiB budget, with committed q checkpoints. Detailed pilots, latest invocations and imported work are separate in cost_vs_scope.json.', '',
        f'Regression: {json.dumps(tests)}', '', f'Protected-file after-check: {json.dumps(preservation)}', '',
        'Original references 1–6, 3B and 8 remain PASS; B7 integration PASS and independent joint INCOMPLETE; strict full-material 9A/9B remain INCOMPLETE; calibrated TACAW comparison remains blocked by metadata; 11A FAIL and 11B NOT_RESOLVED remain. Neither the 72-hour B7 joint calculation nor full nonlinear material 9B is launched.', '',
        '## Figures and reproduction', '',
        'Exact bounded commands and compatible resume selections are in reproduce_and_resume.sh. Source/input hashes, copied-cache provenance, and raw signed energies/weights are retained. Spectral-area checks are in comparison.json; every result is archived by manifest.json.', '',
        '![300 K models](four_models_T300.png)', '', '![1000 K models](four_models_T1000.png)', '',
        '![Thickness](thickness_T300.png)', '', '![Detectors](detectors_T300.png)', '',
        '![Numerics](numerical_convergence.png)', '', '![Statistics](statistical_convergence.png)', '',
        '![Paired differences](paired_differences.png)', '', '![Elastic correction](electron_diffraction_correction.png)', '',
        '![Weak feature diagnosis](weak_feature_diagnosis.png)']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', default=DEFAULT)
    args = p.parse_args(); result = report(ROOT/args.output)
    print(result['status'], 'curves:', len(result['curves']), flush=True)


if __name__ == '__main__': main()
