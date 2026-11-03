"""Conditional broad-detector comparison; no fitted axes or absolute matching."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy import constants
from scipy.integrate import trapezoid
from scipy.optimize import linear_sum_assignment
from yoshioka.spectral_data import gaussian_broaden_bin_weights
from .approximate_eels_analysis import available_born, selected_born, shape, features
from .workflow_io import ROOT, ResourceLimits, json_write


def match_features(reference, numerical, maximum_distance=5.):
    """Minimum-cost one-to-one spectral-feature matches, with dummies.

    This tracks resolved spectral maxima, not phonon-mode identities. A large
    difference between the dominant maxima in a broad window is not automatically
    a frequency shift of the same physical branch.
    """
    x = sorted({p for item in reference.values() for p in item['all_resolved_peaks_mev']})
    y = sorted({p for item in numerical.values() for p in item['all_resolved_peaks_mev']})
    if not x or not y:
        return {'matches': [], 'unmatched_reference_mev': x, 'unmatched_model_mev': y}
    nx, ny = len(x), len(y); cost = np.zeros((nx+ny, ny+nx))
    distance = abs(np.array(x)[:, None]-np.array(y)[None])
    cost[:nx, :ny] = np.where(distance <= maximum_distance, distance, 10000.)
    cost[:nx, ny:] = maximum_distance+.001; cost[nx:, :ny] = maximum_distance+.001
    rows, cols = linear_sum_assignment(cost)
    matches = []; used_x = set(); used_y = set()
    for i, j in zip(rows, cols):
        if i < nx and j < ny and distance[i, j] <= maximum_distance:
            used_x.add(i); used_y.add(j)
            matches.append({'TACAW_mev_conditional': x[i], 'Yoshioka_mev': y[j],
                'absolute_difference_mev': float(distance[i, j]),
                'criterion': 'preferred' if distance[i, j] <= 3 else 'exploratory'})
    return {'matches': matches, 'unmatched_reference_mev': [v for i, v in enumerate(x) if i not in used_x],
            'unmatched_model_mev': [v for i, v in enumerate(y) if i not in used_y],
            'method': 'one-to-one minimum-distance assignment with dummy unmatched features; maximum5meV. All detected maxima are reported.'}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''): h.update(chunk)
    return h.hexdigest()


def raw_tacaw(temperature, output):
    path = ROOT/f'Si_EELS_{temperature}K_Tersoff.npy'
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    full = np.zeros(array.shape[0]); disk = np.zeros_like(full)
    yy, xx = np.ogrid[:array.shape[1], :array.shape[2]]
    mask = (yy-array.shape[1]//2)**2+(xx-array.shape[2]//2)**2 <= 20**2
    for lo in range(0, len(full), 16):
        block = np.asarray(array[lo:lo+16])
        full[lo:lo+16] = np.sum(block, axis=(1, 2)); disk[lo:lo+16] = np.sum(block[:, mask], axis=1)
    energy = np.fft.fftshift(np.fft.fftfreq(len(full), 7.5e-15))*constants.h/(1e-3*constants.elementary_charge)
    step = float(energy[1]-energy[0]); total = float(np.sum(full))
    payload = {'analysis_energy_mev': energy, 'raw_all_momentum_sum': full, 'raw_pixel_disk_sum': disk,
               'recorded_array_total': total}
    spectra = {}
    for name, values in [('all_recorded_momenta', full), ('pixel_disk_radius20', disk)]:
        broad, reach = gaussian_broaden_bin_weights(values/total, step, 2.)
        axis = energy[0]+np.arange(-reach, len(energy)+reach)*step
        spectra[name] = (axis, broad/step)
        payload[name+'_broadened_weights'] = broad
        payload['broadened_axis_mev'] = axis
    np.savez_compressed(output/f'TACAW_T{temperature}_raw_reductions.npz', **payload)
    return spectra, {'path': str(path), 'sha256': file_hash(path), 'shape': list(array.shape),
        'energy_bin_spacing_mev_assumed': step, 'cadence_fs_assumed': 7.5,
        'duration_ps_only_if_unpadded': len(full)*7.5e-3,
        'normalization': 'raw all-momentum and pixel-disk sums saved; shape-only divide by full recorded-array sum',
        'native_window': 'UNKNOWN; added2meV display convolution does not establish a matched total resolution',
        'axis_status': 'CONDITIONAL_ANALYSIS_ASSUMPTION; generation sign/cadence not verified',
        'angular_acceptance': 'all array indices or a20-pixel disk; physical angles UNKNOWN',
        'quantum_processing': 'user confirms Kubo already applied; no additional correction'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='benchmarks/configs/approximate_eels_production_20261102_v1.json')
    args = p.parse_args(); config = json.loads(Path(args.config).read_text()); output = ROOT/config['output']
    limits = ResourceLimits(600, 2); start = time.monotonic()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    results = {'status': 'CONDITIONAL_QUALITATIVE_COMPARISON', 'datasets': [], 'feature_comparisons': [],
        'unmatched_scope': 'Yoshioka0–20mrad/1.536nm versus all TACAW recorded momenta/unknown physical thickness and native window. Also show the historical20-pixel disk as sensitivity.',
        'conditioning': 'Each curve divided by its own5–90meV integral. This disclosed conditioning is not an absolute-intensity comparison.',
        'thresholds_mev': {'preferred': 3, 'exploratory': 5},
        'force_provenance': 'TACAW is hash-matched to Tersoff-labelled arrays; user tentatively identifies suppliedMOD. Exact generation pair style/parameter hash remains unknown. SNAP is not assumed.',
        'calibrated_status': 'BLOCKED_BY_METADATA', 'cost_comparison': 'NOT_ASSESSED; no TACAW generation logs'}
    for row, temperature in enumerate((300, 1000)):
        limits.check(); tacaw, record = raw_tacaw(temperature, output); results['datasets'].append(record)
        _, axis, baseline, _, _ = selected_born(output, temperature)
        display = (axis >= 5)&(axis <= 90)
        candidates = [('q-sampled Born, fixed harmonic', baseline[:, 0])]
        final = output/f'final_T{temperature}_spectra.npz'
        if final.exists():
            with np.load(final) as d: candidates.append(('q-sampled distorted wave', d['mean'][:, 0].copy()))
        ehm = available_born(output/'ehm', temperature)
        if ehm: candidates.append(('temperature-fitted Born (unaccepted diagnostic)', selected_born(output/'ehm', temperature)[2][:, 0]))
        primary_tacaw = None
        for name, (x, y) in tacaw.items():
            spectrum = np.interp(axis, x, y); normalized, area = shape(axis, spectrum)
            axes[row, 0].plot(axis[display], normalized[display], color='black' if name == 'all_recorded_momenta' else '#888888',
                              ls='-' if name == 'all_recorded_momenta' else ':', label='TACAW '+name.replace('_', ' '))
            if name == 'all_recorded_momenta': primary_tacaw = (spectrum, normalized, features(axis, spectrum))
        for name, spectrum in candidates:
            normalized, area = shape(axis, spectrum)
            axes[row, 0].plot(axis[display], normalized[display], label=name)
            axes[row, 1].plot(axis[display], (normalized-primary_tacaw[1])[display], label=name)
            extracted = features(axis, spectrum)
            features_record = []
            for feature, target in primary_tacaw[2].items():
                item = extracted[feature]
                resolved = target['status'] == item['status'] == 'resolved_local_maximum'
                difference = abs(target['peak_mev']-item['peak_mev']) if resolved else None
                verdict = ('PREFERRED_CONDITIONAL_AGREEMENT' if difference <= 3 else 'EXPLORATORY_CONDITIONAL_AGREEMENT' if difference <= 5 else 'DISAGREEMENT_CONDITIONAL') if resolved else 'NOT_IDENTIFIABLE'
                features_record.append({'feature': feature, 'TACAW_peak_mev_conditional': target['peak_mev'],
                    'Yoshioka_peak_mev': item['peak_mev'], 'absolute_difference_mev': difference,
                    'status': verdict, 'TACAW_extraction': target, 'Yoshioka_extraction': item})
            selected = (axis >= 5)&(axis < 90)
            gain = (axis <= -5)&(axis > -90)
            results['feature_comparisons'].append({'temperature_k': temperature, 'model': name,
                'features': features_record, 'conditional_shape_L1': float(trapezoid(abs(normalized[selected]-primary_tacaw[1][selected]), axis[selected])),
                'one_to_one_feature_matching': match_features(primary_tacaw[2], extracted),
                'window_maxima_caveat': 'features[] compares strongest maxima inside fixed broad windows; mismatched or competing maxima do not establish a mode-frequency error.',
                'Yoshioka_5_90_model_weight': area,
                'Yoshioka_gain_over_loss_5_90': float(trapezoid(spectrum[gain], axis[gain])/trapezoid(spectrum[selected], axis[selected])),
                'TACAW_gain_over_loss_5_90_conditional': float(trapezoid(primary_tacaw[0][gain], axis[gain])/trapezoid(primary_tacaw[0][selected], axis[selected]))})
        for col in (0, 1):
            axes[row, col].set(xlim=(5, 90), xlabel='Energy loss (meV); TACAW calibration conditional',
                ylabel='Conditional shape / meV' if col == 0 else 'Signed shape residual / meV',
                title=f'{temperature} K; broad unmatched detector acceptances')
            axes[row, col].legend(fontsize=6); axes[row, col].grid(alpha=.15)
    fig.suptitle('Approximate Yoshioka versus supplied TACAW: qualitative, conditional axes\nNo energy shift, fitted intensity scale, elastic subtraction or second Kubo correction', fontsize=11)
    fig.savefig(output/'TACAW_broad_detector_comparison.png', dpi=160); plt.close(fig)
    results['processing_seconds'] = time.monotonic()-start; results['resources'] = limits.costs()
    json_write(output/'TACAW_comparison.json', results)
    inherited = json.loads((ROOT/'benchmark_results/repair_20261102_v2/B10/results.json').read_text())
    inherited['new_comparison'] = {'TACAW_detector': 'all recorded momenta, with20-pixel disk sensitivity',
        'Yoshioka_detector': '0–20mrad', 'axis_verified': False, 'native_resolution_verified': False,
        'generation_inputs_found_in_repeated_scoped_search': False,
        'note': 'Scoped related-project search found downstream2.5fs/stride3 analysis settings, not originalTRPMD generation inputs. Some sidecars selectSNAP files, so their settings do not verify the suppliedTersoff generation.'}
    json_write(output/'TACAW_metadata.json', inherited)
    print('Conditional TACAW comparison saved', output/'TACAW_comparison.json')


if __name__ == '__main__': main()
