"""Conditional qualitative TACAW/material comparison, without fitted axes/scales."""

import json
from pathlib import Path
import time

import numpy as np
from scipy import constants

from yoshioka.spectral_data import conservative_rebin, gaussian_broaden_bin_weights
from .workflow_io import ROOT, json_write, sha256


def main():
    started = time.monotonic()
    output = ROOT/'benchmark_results/tacaw_material_20261102_v1'
    output.mkdir(parents=True, exist_ok=True)
    audit_path = ROOT/'benchmark_results/tacaw_20261102_v2/results.json'
    audit = json.loads(audit_path.read_text())
    material = json.loads((ROOT/'benchmark_results/dynamical_si_20261102_v1/results.json').read_text())
    effective = json.loads((ROOT/'benchmark_results/anharmonic_followup_20261102_v1/results.json').read_text())
    result = {'benchmark': '10 qualitative material comparison', 'status': 'BLOCKED_BY_METADATA',
        'qualitative_processing_status': 'PASS', 'quantitative_comparison_status': 'BLOCKED_BY_METADATA',
        'metadata_audit_sha256': sha256(audit_path), 'user_metadata': audit['user_metadata'],
        'normalization': 'Shape-only curves divided by each declared 5–90 meV detector-window integral. This is a disclosed conditioning, not an absolute-intensity or matched-detector test. Unnormalized raw/bin weights are saved.',
        'calibration': 'Associated analysis convention only: fftshift(fftfreq(600,7.5fs))*h in meV. Generation sign/cadence not independently verified; no energy shift fitted.',
        'TACAW_detector': 'radius 20 array pixels; angular calibration unavailable',
        'Yoshioka_detector': 'central 0–5 mrad; not asserted equal to the TACAW pixel disk',
        'resolution': '2 meV FWHM added to both minimally processed curves. TACAW native window is unknown, so total resolution is unmatched; no quantitative linewidth test.',
        'quantum_processing': 'User confirms Kubo already applied; no further Kubo correction.',
        'physics': '9A: DW elastic plus one linear source, fixed harmonic MOD. EHM: finite-bath first Born candidates, not accepted because selected independent mode-frequency checks fail. TACAW: anharmonic/auxiliary-wave scope and original force hash incompletely verified.',
        'uncertainty': 'No independent TACAW trajectories/blocks; statistical uncertainty unavailable. No error bars invented.',
        'missing_matching': audit['missing_metadata'], 'comparisons': [], 'cost_status': 'NOT_ASSESSED',
        'TACAW_cost': 'UNAVAILABLE: arrays and current processing time do not establish generation cost',
        'width_status': 'NOT_ASSESSED: native spectral resolution and elastic handling unknown',
        'cross_method_momentum_residual_status': 'BLOCKED_BY_METADATA: no verified TACAW momentum coordinates; index maps are not physically overlaid with angles.'}
    # Conservative common binning, specified independently of the observed peaks.
    edges = np.arange(-300, 301., 1.)
    center = .5*(edges[:-1]+edges[1:])
    maps = []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, temperature in enumerate((300, 1000)):
        with np.load(ROOT/f'benchmark_results/tacaw_20261102_v2/T{temperature}_qualitative_spectra.npz') as data:
            energy = data['analysis_energy_mev']
            step = energy[1]-energy[0]
            source_edges = np.r_[energy-step/2, energy[-1]+step/2]
            tacaw = conservative_rebin(data['shape_detector_bin_weights'], source_edges, edges)
            original_detector_weight = float(np.sum(data['shape_detector_bin_weights']))
        candidates = []
        completed = [c for c in material['cases'] if c['status'] == 'COMPLETE'
            and c['settings']['temperature_k'] == temperature and c['settings']['elastic_background']]
        if completed:
            selected = max(completed, key=lambda c:c['thickness_nm'])
            with np.load(selected['artifact']) as data:
                radius = np.linalg.norm(data['qxy_m_inv'], axis=1)
                from yoshioka.free_propagation import electron_wavenumber
                chosen = radius/electron_wavenumber(60000*constants.elementary_charge)*1000 < 5
                weights = np.sum(data['mode_momentum_weights'][:, chosen], axis=1)
                bins = np.histogram(data['transfers_j']/(1e-3*constants.elementary_charge), edges, weights=weights)[0]
                candidates.append(('9A fixed harmonic, one source', bins, selected['artifact'], selected['thickness_nm']))
        for candidate in effective['EELS_candidates']:
            if candidate['temperature_k'] != temperature:
                continue
            with np.load(candidate['artifact']) as data:
                from yoshioka.free_propagation import electron_wavenumber
                angle = np.linalg.norm(data['qxy_m_inv'], axis=1)/electron_wavenumber(60000*constants.elementary_charge)*1000
                weights = np.sum(data['probabilities'][:, angle < 5], axis=1)
                bins = np.histogram(data['transfers_j']/(1e-3*constants.elementary_charge), edges, weights=weights)[0]
                name = 'EHM candidate (unaccepted)' if 'effective_candidate' in candidate['artifact'] else 'fixed harmonic finite-bath Born'
                candidates.append((name, bins, candidate['artifact'], .76778))
        broadened_t, reach = gaussian_broaden_bin_weights(tacaw, 1., 2.)
        axis = center[0]+np.arange(-reach, len(center)+reach)
        window = (axis >= 5)&(axis < 90)
        display = (axis >= 5)&(axis <= 120)
        t_integral = float(np.sum(broadened_t[window]))
        shape_t = broadened_t/t_integral
        axes[row, 0].plot(axis[display], shape_t[display], 'k', label='TACAW, conditional analysis axis')
        payload = {'energy_edges_mev': edges, 'broadened_energy_mev': axis,
            'TACAW_rebinned_raw_shape_weights': tacaw, 'TACAW_broadened_weights': broadened_t}
        for index, (name, bins, artifact, thickness) in enumerate(candidates):
            broadened_y, _ = gaussian_broaden_bin_weights(bins, 1., 2.)
            y_integral = float(np.sum(broadened_y[window]))
            shape_y = broadened_y/y_integral
            l1 = float(np.sum(abs(shape_t[window]-shape_y[window])))
            beta_energy = .5*constants.Boltzmann*temperature/(1e-3*constants.elementary_charge)
            classical = (axis >= 5)&(axis < min(90, beta_energy))
            metrics = {'temperature_k': temperature, 'method': name, 'Yoshioka_artifact': artifact,
                'Yoshioka_thickness_nm': thickness, 'TACAW_recorded_detector_fraction': original_detector_weight,
                'TACAW_5_90_fraction_of_full_recorded_array': t_integral,
                'Yoshioka_5_90_absolute_detector_weight': y_integral,
                'conditional_shape_L1_5_90': l1,
                'TACAW_peak_5_90_mev': float(axis[window][np.argmax(shape_t[window])]),
                'peak_identity_limit': 'a maximum at the window edge can be an unresolved zero-loss tail; no mode identity is inferred without verified elastic subtraction',
                'Yoshioka_peak_5_90_mev': float(axis[window][np.argmax(shape_y[window])]),
                'classical_region_definition': 'beta E <= 0.5; restricted to E >=5 meV',
                'classical_region_upper_mev': float(min(90, beta_energy)),
                'conditional_shape_L1_classical_region': float(np.sum(abs(shape_t[classical]-shape_y[classical]))),
                'added_broadening_absolute_area_error': float(abs(np.sum(broadened_y)-np.sum(bins))),
                'quantitative_acceptance': 'BLOCKED_BY_METADATA and unmatched model/detector scope'}
            result['comparisons'].append(metrics)
            payload[f'Yoshioka_{index}_unbroadened_weights'] = bins
            payload[f'Yoshioka_{index}_broadened_weights'] = broadened_y
            axes[row, 0].plot(axis[display], shape_y[display], label=name)
            axes[row, 1].plot(axis[display], (shape_y-shape_t)[display], label=name)
        np.savez_compressed(output/f'conditional_spectra_T{temperature}.npz', **payload)
        for col in range(2):
            axes[row, col].set(xlim=(5, 120), xlabel='energy (meV), conditional calibration',
                ylabel='conditional shape / meV' if col == 0 else 'signed shape residual / meV',
                title=f'{temperature} K; unmatched detector and native resolution')
            axes[row, col].legend(fontsize=6)
        array = np.load(ROOT/f'Si_EELS_{temperature}K_Tersoff.npy', mmap_mode='r', allow_pickle=False)
        # Same array indices only; no conversion to invented physical momentum.
        calibrated = np.fft.fftshift(np.fft.fftfreq(len(array), 7.5e-15))*constants.h/(1e-3*constants.elementary_charge)
        total = float(np.sum(array))
        maps.append({label: np.sum(array[(calibrated >= lo)&(calibrated < hi)], axis=0)/total
            for label, lo, hi in (('single_region_5_90', 5, 90), ('higher_loss_90_180', 90, 180))})
    fig.savefig(output/'conditional_material_spectra_and_residuals.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    payload = {}
    for row, label in enumerate(maps[0]):
        commonmax = max(np.max(maps[0][label]), np.max(maps[1][label]))
        for col in range(2):
            im = axes[row, col].imshow(maps[col][label], origin='lower', vmin=0, vmax=commonmax)
            axes[row, col].set(title=f'{(300,1000)[col]} K: {label}', xlabel='array spatial index 2', ylabel='array spatial index 1')
            fig.colorbar(im, ax=axes[row, col], label='fraction of recorded-array sum / pixel')
            payload[f'T{(300,1000)[col]}_{label}'] = maps[col][label]
        residual = maps[1][label]-maps[0][label]
        im = axes[row, 2].imshow(residual, origin='lower', cmap='coolwarm', vmin=-np.max(abs(residual)), vmax=np.max(abs(residual)))
        axes[row, 2].set(title='1000 minus 300 K, same indices', xlabel='array index 2', ylabel='array index 1')
        fig.colorbar(im, ax=axes[row, 2], label='signed shape fraction / pixel')
        payload['temperature_difference_'+label] = residual
    fig.savefig(output/'TACAW_index_window_maps_temperature_residuals.png', dpi=150)
    plt.close(fig)
    np.savez_compressed(output/'TACAW_index_maps.npz', **payload)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for col, label in enumerate(maps[0]):
        for index, temperature in enumerate((300, 1000)):
            cut = np.sum(maps[index][label], axis=0)
            axes[col].plot(np.arange(len(cut)), cut, label=f'{temperature} K')
            payload[f'T{temperature}_{label}_index_cut'] = cut
        axes[col].set(xlabel='array spatial index 2; index 1 summed', ylabel='fraction of recorded-array sum / index bin', title=label)
        axes[col].legend()
    fig.savefig(output/'TACAW_index_momentum_cuts.png', dpi=150)
    plt.close(fig)
    result['processing_seconds'] = time.monotonic()-started
    json_write(output/'results.json', result)
    (output/'report.md').write_text('# Benchmark 10: conditional qualitative material comparison\n\n'
        'Quantitative status: **BLOCKED_BY_METADATA**. Qualitative processing completed. The user authorizes this scope and confirms that Kubo correction is already applied. No correction is applied again.\n\n'
        'The associated analysis convention supplies a conditional energy axis. Momentum calibration, original geometry, force hash, illumination, native window and elastic treatment remain unverified. Shapes are conditioned on the disclosed 5–90 meV interval, without fitting energy shifts or scale parameters. Absolute Yoshioka weights and TACAW recorded-array fractions remain separate in results.json. Their ratio is not a physical cross-section ratio.\n\n'
        'The curves compare a radius-20-pixel TACAW disk to a 0–5 mrad Yoshioka detector under explicitly unmatched assumptions. The additional 2 meV convolution is a display control; unknown native TACAW broadening prevents a known common total resolution or linewidth comparison. Higher-loss TACAW signal is not assumed proof of repeated events or exact multiphonon weights. The EHM curves are unaccepted diagnostic candidates.\n\n'
        'Only index-space TACAW temperature residual maps are formed. Cross-method angular residual maps and quantitative physics acceptance remain blocked. Averaged arrays provide no independent statistical replicates or generation-cost evidence.\n\n'
        '```bash\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.tacaw_material_comparison\n```\n\n'
        '![spectra](conditional_material_spectra_and_residuals.png)\n\n![index maps](TACAW_index_window_maps_temperature_residuals.png)\n')


if __name__ == '__main__':
    main()
