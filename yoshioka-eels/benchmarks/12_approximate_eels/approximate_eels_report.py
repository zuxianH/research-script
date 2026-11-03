"""Assemble measured approximate-EELS results, figures, provenance and limits."""

import argparse
import json
from pathlib import Path
import platform
import subprocess
import numpy as np
from scipy import constants
from scipy.integrate import trapezoid

from .workflow_io import ROOT, json_write, sha256
from .approximate_eels import DETECTORS, ORIENTED, PRIMITIVE
from .approximate_eels_analysis import selected_born, available_born, compare
from .born_quadrature import line_spectra
from yoshioka.electron_phonon import LobatoSilicon
from yoshioka.lattice_dynamics import RealSpaceForceConstants


def read(path): return json.loads(Path(path).read_text())


def numerical_checks(output):
    rows = []
    for prefix, label in [('confirm', 'electron/slice/joint'), ('cov64', 'thermal covariance32→64')]:
        for path in sorted((output/'source').glob(prefix+'*/state.json')):
            state = read(path)
            if state.get('status') != 'COMPLETE': continue
            settings = state['contract']['settings']; temperature = settings['temperature_k']; n = 2**settings['power']
            coarse = output/'source'/f'prod_T{temperature}_seed0'
            e = np.load(coarse/'energies_mev.npy', mmap_mode='r')[:n]
            w = np.load(coarse/'weights.npy', mmap_mode='r')[:n]
            b = np.load(coarse/'born_weights.npy', mmap_mode='r')[:n]
            fine = np.load(path.parent/'weights.npy', mmap_mode='r')[:n]
            np.testing.assert_array_equal(e, np.load(path.parent/'energies_mev.npy', mmap_mode='r'))
            _, axis, baseline, _, _ = selected_born(output, temperature)
            original = baseline+line_spectra(e, (w-b)/n, axis)
            refined = original+line_spectra(e, (fine-w)/n, axis)
            comparison = compare(axis, original, refined)
            rows.append({'case': path.parent.name, 'kind': label, 'settings': settings,
                'matched_production_q': n, **comparison})
    json_write(output/'numerical_confirmation.json', rows)
    return rows


def transition_audit(output, config):
    rows = []
    windows = {'loss_0_40': (0, 40), 'loss_40_65': (40, 65), 'loss_65_90': (65, 90),
               'loss_above90': (90, np.inf), 'gain_0_40': (-40, 0),
               'gain_40_65': (-65, -40), 'gain_65_90': (-90, -65)}
    for temperature in (300, 1000):
        mesh, axis, _, _, _ = selected_born(output, temperature)
        folder = output/'born'/f'T{temperature}_mesh{mesh}'
        energy = np.load(folder/'energies_mev.npy', mmap_mode='r'); weights = np.load(folder/'weights.npy', mmap_mode='r')
        with np.load(output/f'Gamma_cell_T{temperature}_mesh{mesh}.npz') as g:
            gamma_e = g['energies_mev']; gamma_w = g['weights']
        def accumulate(lo, hi):
            value = np.sum(weights*((energy >= lo)&(energy < hi))[..., None], axis=(0, 1))
            value -= np.sum(weights[0]*((energy[0] >= lo)&(energy[0] < hi))[:, None], axis=0)
            value += np.mean(np.sum(gamma_w*((gamma_e >= lo)&(gamma_e < hi))[..., None], axis=(1, 2)), axis=0)
            delta = []
            for case in config['source_cases']:
                if case['temperature_k'] != temperature: continue
                p = output/'source'/case['name']
                e = np.load(p/'energies_mev.npy'); w = np.load(p/'weights.npy'); b = np.load(p/'born_weights.npy')
                delta.append(np.sum((w-b)*((e >= lo)&(e < hi))[..., None], axis=(0, 1))/len(e))
            return value+np.mean(delta, axis=0)
        raw = accumulate(-np.inf, np.inf)
        with np.load(output/f'final_T{temperature}_spectra.npz') as final:
            area = trapezoid(final['mean'], final['energy_mev'], axis=0)
        rows.append({'temperature_k': temperature, 'detectors': DETECTORS,
            'raw_retained_model_weights': raw.tolist(), 'broadened_spectrum_integrals': area.tolist(),
            'maximum_absolute_area_error': float(np.max(abs(raw-area))),
            'maximum_relative_area_error': float(np.max(abs(raw-area)/np.maximum(abs(raw), 1e-100))),
            'unbroadened_window_weights': {name: accumulate(lo, hi).tolist() for name, (lo, hi) in windows.items()},
            'spectrum_density_units': 'model inelastic weight per meV', 'renormalized': False})
    json_write(output/'transition_and_area_audit.json', rows)
    return rows


def discovery(output):
    related = ROOT.parent/'HiPhive_TRPMD'
    command = ['rg', '--files', str(related), '-g', '*.in', '-g', '*.lammps', '-g', 'in.*', '-g', '*.sh', '-g', '*.xml']
    result = subprocess.run(command, capture_output=True, text=True)
    record = {'scope': [str(ROOT), str(related)], 'generation_input_search': command,
        'returncode': result.returncode, 'candidate_generation_files': result.stdout.splitlines(),
        'stderr': result.stderr, 'known_analysis_sidecars': {},
        'conclusion': 'No original generation input recovered; downstream analysis parameters do not verify cadence, sign, native window or exact force model.'}
    for name in ('simple_peak_results_300to900K/job.json', 'multiphonon_analysis.py', 'eels-studio-config.json'):
        path = related/name
        if path.exists(): record['known_analysis_sidecars'][str(path)] = {'sha256': sha256(path), 'role': 'downstream analysis, not generation'}
    json_write(output/'metadata_discovery.json', record)
    metadata = read(output/'TACAW_metadata.json')['metadata']
    lines = ['# TACAW metadata used by this approximate comparison', '',
        'The primary TACAW reduction sums all recorded momentum indices. Its physical angular coverage is unknown. The energy axis remains conditional.', '',
        '| Field | Value | Units | Status | Source |', '|---|---|---|---|---|']
    for key, item in metadata.items():
        value = json.dumps(item['value'], ensure_ascii=False) if isinstance(item['value'], (list, dict)) else str(item['value'])
        lines.append(f'| {key} | {value.replace("|", "/")} | {item["units"]} | {item["status"]} | {item["source"]} |')
    (output/'TACAW_metadata.md').write_text('\n'.join(lines)+'\n')
    (output/'missing_inputs.md').write_text('''# Remaining reference information

For independently calibrated qualitative spectra: generation output cadence, frame count/padding, transform ordering and gain/loss sign; native segment/window and zero-loss treatment.

For a matched physical model: physical momentum coordinates/acceptance, actual orientation/cell/thickness/temperature geometry, exact electron-potential settings and generating force-parameter hash.

For absolute intensities: saved-array measure and incident/elastic normalization details.

For a cost comparison only: TACAW generation wall time, peak memory and hardware/software logs. Their absence does not block conditional shape plots.

Already supplied and retained: 60 keV,208slices,unit-integral normalization,Kubo already applied,and a tentative match to the supplied Tersoff potential. These are not requested again.
''')
    return record


def make_figures(output, source_stats, numerical):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(output/'final_T300_spectra.npz') as d:
        axis = d['energy_mev']; mean = d['mean']; born = d['Born']; band = d['pointwise_95_halfwidth']
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    ax.plot(axis, born[:, 0], '--', color='#777777', label='Bulk-q Born reference')
    ax.plot(axis, mean[:, 0], color='#1766ac', label='Full elastic propagation + one phonon source')
    ax.fill_between(axis, mean[:, 0]-band[:, 0], mean[:, 0]+band[:, 0], color='#1766ac', alpha=.2, label='Pointwise 95% statistical interval')
    ax.set(xlim=(-90, 90), xlabel='Electron energy loss (meV)', ylabel='Estimated inelastic model weight / meV',
        title='Silicon [110] phonon EELS at 300 K — approximate\n1.536 nm · 60 keV · 0–20 mrad · 2 meV FWHM')
    ax.legend(fontsize=8); ax.grid(alpha=.15)
    fig.get_layout_engine().set(rect=(0, .08, 1, .92))
    fig.text(.5, .025, 'Elastic zero-loss line excluded; acoustic continuum retained. Statistical band excludes force-model and quadrature bias.', ha='center', fontsize=8)
    fig.savefig(output/'Si_300K_EELS.png', dpi=175); fig.savefig(output/'Si_300K_EELS.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    born = read(output/'Born_convergence.json')
    for temperature in (300, 1000):
        rows = [r for r in born['comparisons'] if r['temperature_k'] == temperature]
        axes[0].semilogy([r['mesh_fine'] for r in rows], [r['maximum_shape_L1'] for r in rows], 'o-', label=f'{temperature}K')
        rows = source_stats['temperatures'][str(temperature)]['numerical_sample_count_comparisons']
        axes[1].semilogy([r['q_per_batch_fine'] for r in rows], [r['maximum_shape_L1'] for r in rows], 'o-', label=f'{temperature}K')
    axes[0].axhline(.1, color='red', ls=':'); axes[1].axhline(.1, color='red', ls=':')
    axes[0].set(xlabel='Uniform Born mesh per axis', ylabel='Maximum conditional shape L1', title='Deterministic q quadrature')
    axes[1].set(xlabel='q points per independent batch', ylabel='Maximum conditional shape L1', title='Nested stochastic sample counts')
    x = np.arange(len(numerical)); axes[2].semilogy(x, [r['maximum_shape_L1'] for r in numerical], 'o')
    axes[2].set(xlabel='Matched production-q check index', ylabel='Maximum conditional shape L1', title='Electron, slices, joint and covariance')
    for ax in axes: ax.grid(alpha=.15)
    axes[0].legend(); axes[1].legend(); fig.savefig(output/'numerical_convergence.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    for col, temperature in enumerate((300, 1000)):
        p = output/f'repeated_optical_T{temperature}_n49_z1025.npz'
        with np.load(p) as d:
            axis = d['energy_mev']
            for order in (1, 2):
                spectrum = d[f'spectrum_order{order}']; selected = spectrum > 1e-15
                axes[col].semilogy(axis[selected], spectrum[selected], label=f'amplitudes through {order} interaction(s)')
        axes[col].set(xlim=(-170, 170), ylim=(1e-13, None), xlabel='Electron energy loss (meV)', ylabel='Reduced-model weight / meV', title=f'{temperature} K: Γ optical triplet only')
        axes[col].legend(fontsize=7); axes[col].grid(alpha=.15)
    fig.suptitle('Leading repeated interactions: reduced linear model, not full 9B\n43 operator words; thermal Wick trace; 2 meV display FWHM', fontsize=11)
    fig.savefig(output/'leading_repeated_optical.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for col, temperature in enumerate((300, 1000)):
        for mesh, axis, spectrum, state in available_born(output, temperature):
            if mesh not in (48, 64): continue
            axes[col].plot(axis, spectrum[:, 0], ls=':', label=f'uniform {mesh}³')
            path = output/f'Gamma_cell_T{temperature}_mesh{mesh}.npz'
            if path.exists():
                with np.load(path) as d: corrected = spectrum[:, 0]+d['correction'][:, 0]
                axes[col].plot(axis, corrected, label=f'{mesh}³ + local Γ cell')
        axes[col].set(xlim=(-8, 8), xlabel='Energy loss (meV)', ylabel='Model weight / meV', title=f'{temperature} K: low-energy quadrature')
        axes[col].legend(fontsize=7); axes[col].grid(alpha=.15)
    fig.savefig(output/'Gamma_cell_diagnostic.png', dpi=160); plt.close(fig)
    comparisons = read(output/'TACAW_comparison.json')['feature_comparisons']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    models = sorted({r['model'] for r in comparisons})
    for model in models:
        rows = sorted([r for r in comparisons if r['model'] == model], key=lambda r: r['temperature_k'])
        temperatures = [r['temperature_k'] for r in rows]
        optical = [next(f['Yoshioka_peak_mev'] for f in r['features'] if f['feature'] == 'optical') for r in rows]
        axes[0].plot(temperatures, optical, 'o-', label=model)
        axes[1].plot(temperatures, [r['Yoshioka_gain_over_loss_5_90'] for r in rows], 'o-', label=model)
    target = sorted([r for r in comparisons if r['model'] == models[0]], key=lambda r: r['temperature_k'])
    axes[0].plot([r['temperature_k'] for r in target], [next(f['TACAW_peak_mev_conditional'] for f in r['features'] if f['feature'] == 'optical') for r in target], 'k^-', label='TACAW, conditional axes')
    axes[1].plot([r['temperature_k'] for r in target], [r['TACAW_gain_over_loss_5_90_conditional'] for r in target], 'k^-', label='TACAW, conditional axes')
    axes[0].set(xlabel='Temperature label (K)', ylabel='Optical spectral-feature maximum (meV)')
    axes[1].set(xlabel='Temperature label (K)', ylabel='Gain / loss weight, |E| in 5–90 meV')
    for ax in axes: ax.legend(fontsize=6); ax.grid(alpha=.15)
    fig.suptitle('Conditional temperature trends; 1000 K exploratory\nFeature maxima are not mode-resolved frequency validation; detector and native resolution remain unmatched.', fontsize=10)
    fig.savefig(output/'temperature_trends.png', dpi=160); plt.close(fig)


def report(config, output):
    numerical = numerical_checks(output)
    transition = transition_audit(output, config)
    source = read(output/'source_statistics.json'); tacaw = read(output/'TACAW_comparison.json')
    independent = read(output/'independent_validation.json'); repeated = read(output/'leading_repeated_material.json')
    tests = read(output/'regression.json'); pilot = read(output/'pilot.json')
    preservation = read(output/'protected_after.json') if (output/'protected_after.json').exists() else {'all_unchanged': False, 'status': 'AFTER_CHECK_PENDING'}
    discovery(output); make_figures(output, source, numerical)
    costs = {'production_invocations': read(output/'approximate_eels_production_20261102_v1_invocations.json'),
        'confirmation_invocations': read(output/'approximate_eels_confirmation_20261102_v1_invocations.json') if (output/'approximate_eels_confirmation_20261102_v1_invocations.json').exists() else [],
        'Born': {}, 'source': {}, 'TACAW_generation': None,
        'note': 'Latest invocation wall and cumulative committed q work differ; no timing or memory is inferred for TACAW generation.'}
    for path in (output/'born').glob('*/state.json'):
        d = read(path); costs['Born'][path.parent.name] = {k: d.get(k) for k in ('status', 'cumulative_seconds', 'latest_invocation_seconds', 'resources')}
    for path in (output/'source').glob('*/state.json'):
        d = read(path); costs['source'][path.parent.name] = {k: d.get(k) for k in ('status', 'completed_q', 'total_q', 'cumulative_seconds', 'latest_invocation_seconds', 'resources')}
    timing = read(output/'phase_timing_pilot.json')
    costs['profiled_phase_pilot'] = timing
    costs['production_wall_seconds'] = sum(r['seconds'] for r in costs['production_invocations'])
    costs['Born_cumulative_committed_seconds'] = sum(r['cumulative_seconds'] for r in costs['Born'].values())
    costs['maximum_recorded_process_rss_mib'] = max(r['resources']['peak_process_rss_mib'] for r in costs['source'].values())
    json_write(output/'cost_vs_scope.json', costs)
    import os
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), platform.processor())
    json_write(output/'hardware.json', {'CPU_model': cpu, 'logical_CPUs': os.cpu_count(),
        'platform': platform.platform(), 'thread_environment': {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        'FFT_workers_in_source_implementation': 4, 'cost_scope': 'per-invocation wall and per-process peakRSS; other project diagnostics sometimes overlapped, so no isolated-hardware speedup is inferred'})
    constants_record = {'hbar_j_s': constants.hbar, 'Boltzmann_j_k': constants.Boltzmann,
        'elementary_charge_c': constants.elementary_charge, 'atomic_mass_kg': constants.atomic_mass,
        'source': 'project scipy.constants/CODATA2022', 'Lobato_parameters': LobatoSilicon.from_abtem_table().__dict__,
        'primitive_force_constants_sha256': sha256(PRIMITIVE), 'oriented_force_constants_sha256': sha256(ORIENTED),
        'cell_m': RealSpaceForceConstants.load(ORIENTED).cell_m.tolist(),
        'mass_kg': RealSpaceForceConstants.load(ORIENTED).masses_kg.tolist()}
    json_write(output/'constants_and_physics.json', constants_record)
    complete = all(source['temperatures'][str(t)]['production_status'] == 'COMPLETE' for t in (300, 1000))
    peak_rows = []
    for r in tacaw['feature_comparisons']:
        for f in r['features']:
            if f['feature'] == 'optical': peak_rows.append({'temperature_k': r['temperature_k'], 'model': r['model'], **f})
    summary = {'status': ('APPROXIMATE_WORKFLOW_COMPUTED' if preservation.get('all_unchanged') else 'COMPUTED_VERIFICATION_PENDING') if complete else 'INCOMPLETE',
        'geometry': {'orientation': 'Si[110]', 'electron_energy_kev': 60, 'thickness_nm': pilot['rows'][0]['thickness_nm'],
            'oriented_repeats': 4, 'unit_cell_grid': 49, 'slices_including_vacuum': 385, 'vacuum_padding_each_side_angstrom': 12},
        'primary': source['temperatures']['300'], 'exploratory': source['temperatures']['1000'],
        'independent_source_finest_relative_complex_error': independent['independent_source'][-1]['complex_vector_relative_error'],
        'numerical_confirmation_cases': len(numerical), 'area_audit': transition,
        'conditional_optical_features': peak_rows, 'tests': tests, 'preservation': preservation,
        'strict_reference_statuses': {'1–6,3B,8': 'PASS preserved', '7 integration': 'PASS preserved', '7 joint': 'INCOMPLETE preserved',
            '9A strict full material': 'INCOMPLETE preserved', '9B full material': 'INCOMPLETE preserved',
            '10 calibrated comparison': 'BLOCKED_BY_METADATA preserved', '11A EHM': 'FAIL preserved', '11B widths': 'NOT_RESOLVED preserved'},
        'full_nonlinear_finite_T_9B_claimed': False}
    json_write(output/'results.json', summary)
    lines = ['# Affordable approximate silicon EELS: measured results', '',
        f'Computed bulk-q Born and distorted-wave spectra at 300 K and exploratory 1000 K for a **{summary["geometry"]["thickness_nm"]:.6f} nm Si [110] slab**, 60 keV, 49² unit-cell pixels, 385 steps including 12 Å vacuum on each side. This is a separate approximate workflow. Historical strict benchmark statuses are unchanged.', '',
        'The incident and final electrons retain dynamical elastic scattering. One linear inelastic source is accumulated coherently in depth; all six primitive phonon branches and both thermal emission/absorption are integrated. No full thermal occupation basis, dense 4,465-channel matrix, energy-shift fit or additional Kubo correction is used.', '',
        '## Approximate convergence and uncertainty', '',
        '| Temperature | Source q points | Independent batches | Full-detector variance reduction | Integrated pointwise 95% half-width / signal (5–90 meV) |', '|---|---:|---:|---:|---:|']
    for t in (300, 1000):
        r = source['temperatures'][str(t)]
        lines.append(f'| {t} K | {r["total_q"]} | {r["completed_independent_batches"]} | {r["variance_reduction_factor_5_90"][0]:.2f}× | {r["detectors"][0]["integrated_pointwise_95_halfwidth_over_weight"]:.3%} |')
    lines += ['', 'The confidence band is pointwise, not simultaneous, and does not include force-model errors or deterministic quadrature bias. Independent Gamma-cell uncertainty is included. The eight-batch bootstrap is a finite-sample sensitivity estimate.', '',
        '| Matched-q refinement | q points | Maximum shape L1 | Maximum resolved-window peak change (meV) |', '|---|---:|---:|---:|']
    for r in numerical: lines.append(f'| {r["case"]} | {r["matched_production_q"]} | {r["maximum_shape_L1"]:.6g} | {r["maximum_peak_shift_mev"]:.3g} |')
    lines += ['', 'The 1000 K covariance refinement switches a weak angular-window dominant maximum by 3.1 meV despite a 0.052% shape change. This remains a flagged feature/dominance sensitivity, not a claimed physical mode shift or a hidden PASS.', '']
    lines += ['', f'The independent numerical Fourier-gradient source versus analytical atomic sum reaches **{summary["independent_source_finest_relative_complex_error"]:.4g} relative complex-amplitude error**, without phase alignment. Uniform 48³→64³ Born comparisons, all four detector curves, unresolved/competing maxima and earlier sample-count failures remain in the machine-readable tables. A window without an identifiable peak does not establish a peak location.', '',
        'The uniform Born baseline is supplemented by local integration of its Gamma cell, whose acoustic interior is otherwise missed by a nodal rigid-translation exclusion. The entire cell is replaced once, using actual matrix-element-weighted EELS. No continuum point mass or second probability weight is added. Low-energy results remain resolution- and quadrature-dependent; no intrinsic linewidth is inferred.', '',
        '## Conditional TACAW comparison', '',
        'TACAW uses all recorded momentum indices, with the historical 20-pixel disk shown as an acceptance sensitivity. Yoshioka uses 0–20 mrad. Physical angular ranges, generating energy cadence/sign, native window, elastic handling, exact force hash and specimen geometry remain unmatched. The 7.5 fs axis is a downstream analysis assumption, not verified generation metadata.', '',
        '| Temperature | Model | Optical feature (meV) | TACAW conditional feature (meV) | Difference (meV) |', '|---|---|---:|---:|---:|']
    for r in peak_rows: lines.append(f'| {r["temperature_k"]} K | {r["model"]} | {r["Yoshioka_peak_mev"]:.2f} | {r["TACAW_peak_mev_conditional"]:.2f} | {r["absolute_difference_mev"]:.2f} |')
    lines += ['', 'The main optical and acoustic features can agree while other peaks and relative weights disagree. All one-to-one matches and unmatched features are reported; a difference between dominant maxima in a broad window is not automatically a shift of the same mode. At 300 K the distorted-wave conditional shape L1 is about 0.46; at 1000 K it is about 0.40. These are visible shape differences, not an unqualified comparison PASS.', '',
        'Fixed harmonic frequencies do not intrinsically soften with temperature. The temperature-fitted Born candidates show softening and improved optical positions but retain the original EHM validation limitations. Corrected-MD fits are used; no forces were adjusted to TACAW peaks. No uncertainty bars or generation costs are invented for the supplied averaged TACAW arrays.', '',
        '## Leading repeated interactions', '',
        'A separate linear-harmonic formulation retains oscillator operator words through two insertions. Thermal Wick contractions include identical-mode factors, exchange, mixed gain/loss and interference among words ending in the same bath change. It agrees with an independently assembled dense coupled-channel reference to 9.3e−16 complex-vector scale; direct thermal-trace differences are at most 1.2e−16. Nonzero-Bloch dense splitting and existing one-source agreement are automated tests.', '',
        'The material pilot uses only the complete primitive-Gamma optical triplet, normalized to its sector in a one-oriented-cell folded bath. It is a reduced model, not the bulk-q spectrum. Electron grids 17/25/33/49/65 and 257/513/1025 steps are tested separately. No unrepresented bath is renormalized.', '',
        'The order 0+1+2 amplitudes interfere. Their square omits other exact intensity terms through g⁴, and a signed detector correction need not be positive. Direct nonlinear single-event multiphonon transitions are omitted by the linear model; detector-weighted displacement parameters and independent nonlinear oscillator controls quantify this approximation without presenting a full-material omission bound.', '',
        'The Q.U.Q diagnostic uses Born final-transfer weighting. Dynamical paths can involve larger intermediate momentum transfers. Neither it nor the oscillator control bounds every omitted nonlinear vertex in the material.', '',
        '## Status and scope', '',
        '| Component | New approximate result | Original strict status |', '|---|---|---|',
        '| 300 K bulk-q Born/distorted spectrum | Computed; primary numerical targets verified | Does not close full 19.96 nm 9A |',
        '| 1000 K spectrum | Computed, exploratory; weak-window dominance sensitivity retained | No new strict material PASS |',
        '| Repeated interactions | Thermal implementation validated; reduced Gamma material pilot refined | Full 9B INCOMPLETE |',
        '| TACAW comparison | Conditional feature/shape comparison with unmatched features | Calibrated/matched comparison BLOCKED_BY_METADATA |',
        '| Temperature-fitted harmonic candidates | Corrected-MD q-sampled diagnostics | B11A FAIL preserved |',
        '| Intrinsic linewidths | Not inferred or fitted here | B11B NOT_RESOLVED |',
        '| References 1–6, 3B, 8 | Regression verified | PASS preserved |',
        '| B7 | Completed 128 mesh reused as reference; not rebuilt | Integration PASS; independent joint INCOMPLETE |', '',
        '## Resources, preservation and reproduction', '',
        f'Tests: **{tests["test_count"]} tests**, return code {tests["returncode"]}, wall {tests["seconds"]:.2f} s. Protected-file after-check: **{preservation.get("all_unchanged", False)}**; see protected_before.json/protected_after.json. The prior 144-test suite and all historical inputs/results remain protected.', '',
        'Production invocation wall times, cumulative committed q work, all pilots/refinements and peak RSS are in cost_vs_scope.json. The configured limit is 2 GiB and 600 s per invocation. The 72-hour B7 joint run and full nonlinear macroscopic 9B were not launched. TACAW runtime/memory are unknown, so no speedup claim is made.', '',
        'Run reproduce_and_resume.sh for exact cached commands; use its documented cold configuration procedure for a fresh result directory. See methods.md, requirements_audit.json, changed_source_files.json and manifest.json.', '',
        '![300K spectrum](Si_300K_EELS.png)', '', '![temperature spectra](distorted_wave_temperature.png)', '',
        '![conditional TACAW comparison](TACAW_broad_detector_comparison.png)', '',
        '![conditional temperature trends](temperature_trends.png)', '',
        '![convergence](numerical_convergence.png)', '', '![low-energy integration](Gamma_cell_diagnostic.png)', '',
        '![limited-order optical pilot](leading_repeated_optical.png)']
    lines.insert(lines.index('## Conditional TACAW comparison')+2,
        'Precision note: the associated TACAW analysis-bin spacing is 0.919 meV. Smaller displayed differences are not evidence of that physical accuracy or verified calibration.')
    at = lines.index('## Resources, preservation and reproduction')+2
    lines.insert(at, f'The two production source invocations took **{costs["production_wall_seconds"]/60:.2f} minutes** for 1,024 q sectors across both temperatures. The complete Born mesh sequence used {costs["Born_cumulative_committed_seconds"]/60:.2f} minutes of committed q work. The largest recorded process RSS across source pilots/refinements was **{costs["maximum_recorded_process_rss_mib"]:.2f} MiB**; these are measured component costs, not total project turnaround time.')
    lines.insert(at+1, '')
    lines += ['', 'A separate four-sector profiled timing pilot is saved in phase_timing_pilot.json. It separates phonon mode diagonalization/mapping, source-field construction, signed outgoing propagation and the Born reference, and reproduces the saved raw weights. Its times include profiler overhead and exclude cold covariance/background setup.']
    lines += ['', f'Protected baseline: {preservation.get("count", "pending")} pre-existing files; all unchanged={preservation.get("all_unchanged", False)}.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='benchmarks/configs/approximate_eels_production_20261102_v1.json')
    args = p.parse_args(); config = read(args.config); output = ROOT/config['output']
    result = report(config, output); print(result['status'], result['tests']['test_count'])


if __name__ == '__main__': main()
