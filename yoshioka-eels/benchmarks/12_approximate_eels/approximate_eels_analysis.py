"""Plots and declared approximate gates; no modification of strict benchmarks."""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
from scipy.stats import t as student_t

from .workflow_io import ROOT, json_write
from .approximate_eels import DETECTORS
from .born_quadrature import line_spectra


FEATURE_WINDOWS = {'acoustic': (10., 40.), 'middle': (40., 60.), 'optical': (60., 85.)}


def shape(axis, spectrum, window=(5, 90)):
    selected = (axis >= window[0]) & (axis < window[1])
    area = float(trapezoid(spectrum[selected], axis[selected]))
    if area <= 0: raise ValueError('nonpositive comparison-window weight')
    return spectrum/area, area


def features(axis, spectrum):
    result = {}
    for name, (lo, hi) in FEATURE_WINDOWS.items():
        selected = (axis >= lo)&(axis < hi)
        x = axis[selected]; y = spectrum[selected]
        peaks, properties = find_peaks(y, prominence=max(float(np.max(y))*.05, 0), distance=max(1, int(2/(x[1]-x[0]))))
        if len(peaks):
            j = peaks[np.argmax(y[peaks])]
            center = float(x[j]); status = 'resolved_local_maximum'
        else:
            j = np.argmax(y); center = float(x[j]); status = 'NO_RESOLVED_LOCAL_MAXIMUM'
        area = float(trapezoid(y, x))
        result[name] = {'peak_mev': center, 'status': status,
            'centroid_mev': float(trapezoid(x*y, x)/area) if area > 0 else None,
            'weight': area, 'all_resolved_peaks_mev': x[peaks].tolist()}
    return result


def compare(axis, coarse, fine):
    selected = (axis >= 5)&(axis < 90)
    rows = []
    for j, detector in enumerate(DETECTORS):
        a, ca = shape(axis, coarse[:, j]); b, fa = shape(axis, fine[:, j])
        change = float(trapezoid(abs(a[selected]-b[selected]), axis[selected]))
        before = features(axis, coarse[:, j]); after = features(axis, fine[:, j])
        shifts = {key: abs(before[key]['peak_mev']-after[key]['peak_mev']) for key in before}
        unresolved = [key for key in before if before[key]['status'] != 'resolved_local_maximum' or after[key]['status'] != 'resolved_local_maximum']
        rows.append({'detector': detector, 'shape_L1': change,
            'coarse_weight_5_90': ca, 'fine_weight_5_90': fa,
            'relative_weight_change': abs(fa-ca)/fa, 'feature_shifts_mev': shifts,
            'coarse_features': before, 'fine_features': after, 'unresolved_features': unresolved,
            'shape_pass': change < .1, 'peak_pass': max(shifts.values()) <= 1 and not unresolved})
    return {'rows': rows, 'maximum_shape_L1': max(r['shape_L1'] for r in rows),
        'maximum_peak_shift_mev': max(max(r['feature_shifts_mev'].values()) for r in rows),
        'approximate_pass': all(r['shape_pass'] and r['peak_pass'] for r in rows),
        'definition': '5–90 meV conditional shapes; separate absolute weights; strongest resolved local maximum in predeclared 10–40,40–60,60–85 meV regions. No smoothing beyond 2 meV FWHM.'}


def available_born(output, temperature):
    rows = []
    for path in (output/'born').glob(f'T{temperature:g}_mesh*/state.json'):
        state = json.loads(path.read_text())
        if state.get('status') == 'COMPLETE':
            with np.load(path.parent/'spectrum.npz') as data:
                rows.append((state['contract']['mesh'], data['energy_mev'].copy(), data['spectra'].copy(), state))
    return sorted(rows, key=lambda row: row[0])


def selected_born(output, temperature):
    """Finest uniform result plus a separately recorded Gamma-cell replacement."""
    mesh, axis, spectrum, state = available_born(output, temperature)[-1]
    path = output/f'Gamma_cell_T{temperature}_mesh{mesh}.npz'
    standard_error = np.zeros_like(spectrum)
    if path.exists():
        with np.load(path) as data:
            spectrum = spectrum+data['correction']
            standard_error = data['standard_error'].copy()
        state = {**state, 'adaptive_Gamma_cell': str(path)}
    return mesh, axis, spectrum, state, standard_error


def plot_baseline(output, temperatures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = {'scope': 'Approximate finite-slab bulk-q harmonic first Born; strict benchmark statuses unchanged.', 'temperatures': {}, 'comparisons': []}
    fig, axes = plt.subplots(1, len(temperatures), figsize=(6*len(temperatures), 4.3), squeeze=False, layout='constrained')
    for panel, temperature in enumerate(temperatures):
        rows = available_born(output, temperature)
        if not rows: continue
        for mesh, axis, spectra, state in rows:
            axes[0, panel].plot(axis, spectra[:, 0], label=f'{mesh}³ q points')
        mesh, axis, spectra, state = rows[-1]
        result['temperatures'][str(temperature)] = {'finest_mesh': mesh, 'features_full_20mrad': features(axis, spectra[:, 0]),
            'integrated_signed_weights': trapezoid(spectra, axis, axis=0).tolist(),
            'negative_plot_density_minimum': float(np.min(spectra)),
            'mean_loss_gain_ratio': float(trapezoid(spectra[axis > 5, 0], axis[axis > 5])/trapezoid(spectra[axis < -5, 0], axis[axis < -5]))}
        for a, b in zip(rows[:-1], rows[1:]):
            result['comparisons'].append({'temperature_k': temperature, 'mesh_coarse': a[0], 'mesh_fine': b[0], **compare(axis, a[2], b[2])})
        axes[0, panel].set(xlim=(-90, 90), ylim=(0, None), xlabel='Electron energy loss (meV)',
            ylabel='Model inelastic weight / meV', title=f'{temperature} K: q-sampled first Born')
        axes[0, panel].legend(fontsize=8); axes[0, panel].grid(alpha=.15)
    fig.suptitle('Si [110], 60 keV, 1.536 nm slab, 0–20 mrad\nThermal covariance and Bose occupations; 2 meV FWHM display resolution', fontsize=11)
    fig.savefig(output/'Born_temperature_convergence.png', dpi=160); plt.close(fig)
    rows = available_born(output, 300)
    if rows:
        mesh, axis, spectra, _, _ = selected_born(output, 300)
        fig, ax = plt.subplots(figsize=(8, 4.4), layout='constrained')
        for j, detector in enumerate(DETECTORS[:2]): ax.plot(axis, spectra[:, j], label=detector.replace('_', ' '))
        ax.set(xlim=(-90, 90), ylim=(0, None), xlabel='Electron energy loss (meV)', ylabel='Model inelastic weight / meV',
            title=f'Silicon phonon EELS at 300 K: approximate first Born\n{mesh}³ bulk q mesh + local Γ refinement; 1.536 nm, 60 keV, 2 meV FWHM')
        ax.legend(); ax.grid(alpha=.15)
        fig.savefig(output/'early_Born_300K.png', dpi=170); plt.close(fig)
    json_write(output/'Born_convergence.json', result)
    return result


def source_batches(output, cases, temperature, names=None):
    baseline = available_born(output, temperature)
    if not baseline: return None
    mesh, axis, dense, _, baseline_sem = selected_born(output, temperature)
    spectra = []; corrections = []; direct = []; used = []
    for case in cases:
        if case['temperature_k'] != temperature or (names is not None and case['name'] not in names): continue
        path = output/'source'/case['name']
        if not (path/'spectrum.npz').exists(): continue
        state = json.loads((path/'state.json').read_text())
        if state.get('status') != 'COMPLETE': continue
        with np.load(path/'spectrum.npz') as d:
            correction = d['correction'].copy(); sample = d['direct'].copy()
        spectra.append(dense+correction); corrections.append(correction); direct.append(sample); used.append(case)
    if not used: return None
    return {'axis': axis, 'dense': dense, 'spectra': np.array(spectra), 'corrections': np.array(corrections),
            'direct': np.array(direct), 'cases': used, 'born_mesh': mesh, 'baseline_sem': baseline_sem}


def summarize_sources(output, cases, temperatures):
    """Independent-scramble confidence estimates; the dense Born bias is separate."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = {'estimator': 'dense Born + mean_q[(DW(q)-Born(q))/sampling_density(q)]',
        'uncertainty': 'Student-t pointwise intervals across independent scrambled Sobol batches; independent Gamma-cell variance is added in quadrature with conservative df=min(batchdf,7). These are not simultaneous spectral confidence bands and exclude the separately measured uniform Born quadrature bias.',
        'temperatures': {}}
    fig, axes = plt.subplots(1, len(temperatures), figsize=(6*len(temperatures), 4.5), squeeze=False, layout='constrained')
    for col, temperature in enumerate(temperatures):
        data = source_batches(output, cases, temperature)
        if data is None: continue
        axis = data['axis']; selected = (axis >= 5)&(axis < 90)
        samples = data['spectra']; count = len(samples)
        mean = np.mean(samples, axis=0)
        sem = np.std(samples, axis=0, ddof=1)/np.sqrt(count) if count > 1 else np.full_like(mean, np.nan)
        sem = np.sqrt(sem**2+data['baseline_sem']**2)
        halfwidth = student_t.ppf(.975, min(count-1, 7))*sem if count > 1 else np.full_like(mean, np.nan)
        direct_integrals = trapezoid(data['direct'][:, selected], axis[selected], axis=1)
        correction_integrals = trapezoid(data['corrections'][:, selected], axis[selected], axis=1)
        ratios = np.var(direct_integrals, axis=0, ddof=1)/np.maximum(np.var(correction_integrals, axis=0, ddof=1), 1e-100) if count > 1 else np.full(4, np.nan)
        row = {'completed_independent_batches': count, 'total_q': sum(2**s['power'] for s in data['cases']),
            'requested_independent_batches': sum(s['temperature_k'] == temperature for s in cases),
            'production_status': 'COMPLETE' if count == sum(s['temperature_k'] == temperature for s in cases) else 'INCOMPLETE',
            'cases': data['cases'], 'Born_mesh': data['born_mesh'], 'features_full_20mrad': features(axis, mean[:, 0]),
            'variance_reduction_factor_5_90': ratios.tolist() if count > 1 else [None]*4,
            'minimum_estimated_density': float(np.min(mean)),
            'temperature_scope': 'primary' if temperature == 300 else 'exploratory',
            'numerical_sample_count_comparisons': []}
        # Nested prefixes use raw importance-weighted integrands: 1/N is applied
        # only here, so increasing sample count never double-counts q weights.
        prefixes = []
        max_power = min(s['power'] for s in data['cases'])
        for power in range(max(2, max_power-2), max_power+1):
            spectra = []
            for settings in data['cases']:
                folder = output/'source'/settings['name']; n = 2**power
                energy = np.load(folder/'energies_mev.npy', mmap_mode='r')[:n]
                dw = np.load(folder/'weights.npy', mmap_mode='r')[:n]
                born = np.load(folder/'born_weights.npy', mmap_mode='r')[:n]
                spectra.append(data['dense']+line_spectra(energy, (dw-born)/n, axis))
            prefixes.append((power, np.mean(spectra, axis=0)))
        for a, b in zip(prefixes[:-1], prefixes[1:]):
            row['numerical_sample_count_comparisons'].append({'q_per_batch_coarse': 2**a[0], 'q_per_batch_fine': 2**b[0], **compare(axis, a[1], b[1])})
        detector_rows = []
        for j, detector in enumerate(DETECTORS):
            absolute = float(trapezoid(mean[selected, j], axis[selected]))
            estimated_ci_L1 = float(trapezoid(halfwidth[selected, j], axis[selected])) if count > 1 else None
            detector_rows.append({'detector': detector, 'weight_5_90': absolute,
                'integrated_pointwise_95_halfwidth_over_weight': estimated_ci_L1/absolute if count > 1 else None,
                'Born_weight_5_90': float(trapezoid(data['dense'][selected, j], axis[selected]))})
        row['detectors'] = detector_rows
        if count > 1:
            # Independent scrambled sequences, not individual correlated Sobol
            # points, are resampled. This reports finite-batch bootstrap
            # sensitivity, not a proof of simultaneous confidence coverage.
            rng = np.random.default_rng(749031)
            boot_l1 = []; boot_peaks = {key: [] for key in FEATURE_WINDOWS}
            reference_area = trapezoid(mean[selected], axis[selected], axis=0)
            reference_shape = mean/reference_area[None]
            gamma_path = output/f'Gamma_cell_T{temperature}_mesh{data["born_mesh"]}.npz'
            gamma_samples = None
            if gamma_path.exists():
                with np.load(gamma_path) as g: gamma_samples = g['independent_corrections'].copy()
            for _ in range(500):
                bootstrap = data['dense']+np.mean(data['corrections'][rng.integers(count, size=count)], axis=0)
                if gamma_samples is not None:
                    bootstrap += np.mean(gamma_samples[rng.integers(len(gamma_samples), size=len(gamma_samples))], axis=0)-np.mean(gamma_samples, axis=0)
                area = trapezoid(bootstrap[selected], axis[selected], axis=0)
                boot_l1.append(trapezoid(abs(bootstrap[selected]/area[None]-reference_shape[selected]), axis[selected], axis=0))
                found = features(axis, bootstrap[:, 0])
                for key in boot_peaks: boot_peaks[key].append(found[key]['peak_mev'])
            row['batch_bootstrap'] = {'replicates': 500, 'seed': 749031, 'shape_L1_95_percentile_by_detector': np.percentile(boot_l1, 95, axis=0).tolist(),
                'primary_feature_2_5_97_5_percentile_mev': {key: np.percentile(values, [2.5, 97.5]).tolist() for key, values in boot_peaks.items()},
                'scope': 'Sensitivity from only8independent source batches and8independentGamma batches; deterministic quadrature and model errors remain separate.'}
        np.savez_compressed(output/f'final_T{temperature:g}_spectra.npz', energy_mev=axis, mean=mean,
            pointwise_95_halfwidth=halfwidth, standard_error=sem, Born=data['dense'], independent_batch_spectra=samples)
        ax = axes[0, col]
        ax.plot(axis, data['dense'][:, 0], label=f'Born, {data["born_mesh"]}³ q mesh', color='#777777', ls='--')
        ax.plot(axis, mean[:, 0], label=f'Distorted wave, {row["total_q"]} sampled q', color='#1766ac')
        if count > 1: ax.fill_between(axis, mean[:, 0]-halfwidth[:, 0], mean[:, 0]+halfwidth[:, 0], color='#1766ac', alpha=.2, label='pointwise 95% interval')
        ax.set(xlim=(-90, 90), xlabel='Electron energy loss (meV)', ylabel='Estimated model weight / meV', title=f'{temperature} K; {count} independent batches')
        ax.legend(fontsize=8); ax.grid(alpha=.15)
        result['temperatures'][str(temperature)] = row
    fig.suptitle('Bulk-q silicon EELS: full elastic propagation, one linear inelastic source\n1.536 nm, 60 keV, 0–20 mrad, 2 meV FWHM; approximate workflow', fontsize=11)
    fig.savefig(output/'distorted_wave_temperature.png', dpi=160); plt.close(fig)
    json_write(output/'source_statistics.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='benchmarks/configs/approximate_eels_20261102_v1.json')
    args = p.parse_args(); config = json.loads(Path(args.config).read_text()); output = ROOT/config['output']
    print(json.dumps(plot_baseline(output, config['temperatures_k']), default=str)[:300])
    if any((output/'source'/s['name']/'spectrum.npz').exists() for s in config.get('source_cases', [])):
        summarize_sources(output, config['source_cases'], config['temperatures_k'])


if __name__ == '__main__': main()
