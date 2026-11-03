"""User-authorized qualitative TACAW audit; calibration assumptions stay explicit."""

import json
from pathlib import Path

import numpy as np
from scipy import constants

from yoshioka.spectral_data import gaussian_broaden_bin_weights
from .workflow_io import ROOT, json_write, sha256


def main():
    output = ROOT/'benchmark_results/tacaw_20261102_v2'
    output.mkdir(parents=True, exist_ok=True)
    previous_path = ROOT/'benchmark_results/tacaw_20261102_v1/results.json'
    previous = json.loads(previous_path.read_text())
    user_path = ROOT/'benchmarks/configs/tacaw_user_metadata_20261102.json'
    supplied = json.loads(user_path.read_text())
    results = {**previous, 'status': 'INCOMPLETE', 'user_metadata': supplied,
        'previous_audit_sha256': sha256(previous_path), 'category_statuses': {
            'metadata_completeness': 'INCOMPLETE', 'physical_comparability': 'INCOMPLETE',
            'spectral_comparison': 'INCOMPLETE', 'runtime_memory_comparison': 'NOT_ASSESSED'},
        'qualitative_data_extraction_status': 'PASS', 'qualitative_curves': [],
        'comparison_scope': 'qualitative only, as explicitly authorized; TACAW is not an exact EELS reference',
        'quantum_processing': 'Kubo correction already applied according to user; no second correction',
        'force_model_matching': 'user tentatively identifies the same supplied potential; exact generation parameter hash remains unverified'}
    datafiles = ('Si_EELS_300K_Tersoff.npy', 'Si_EELS_1000K_Tersoff.npy')
    curves = []
    for temperature, filename, record in zip((300, 1000), datafiles, results['datasets']):
        data = np.load(ROOT/filename, mmap_mode='r', allow_pickle=False)
        record['metadata']['electron'] = {'status': 'USER_SUPPLIED', 'beam_energy_kev': 60, 'slices': 208,
            'missing': 'actual grid, potential projection/filtering, slice spacing and illumination generation record'}
        record['metadata']['absolute_normalization'] = supplied['intensity_normalization']
        record['metadata']['quantum_treatment'] = supplied['Kubo_correction']
        record['metadata']['force_model'] = supplied['force_model']
        # Calibration comes only from the associated processing helper. It is
        # not promoted to verified generation metadata by the array's shape.
        cadence_s = 2.5*3*1e-15
        energy = np.fft.fftshift(np.fft.fftfreq(len(data), cadence_s))*constants.h/(1e-3*constants.elementary_charge)
        step = float(energy[1]-energy[0])
        yy, xx = np.ogrid[:data.shape[1], :data.shape[2]]
        mask = (yy-data.shape[1]//2)**2+(xx-data.shape[2]//2)**2 <= 20**2
        full = np.sum(data, axis=(1, 2))
        detector = np.sum(data[:, mask], axis=1)
        # Explicit shape-only processing. Raw arrays and unnormalized diagnostic
        # sums remain in v1. No fitted scale or independently shifted energy axis.
        normalization = np.sum(full)
        full_weights, detector_weights = full/normalization, detector/normalization
        broadened, reach = gaussian_broaden_bin_weights(detector_weights, step, 2.)
        broad_energy = energy[0]+np.arange(-reach, len(energy)+reach)*step
        target = output/f'T{temperature}_qualitative_spectra.npz'
        np.savez_compressed(target, analysis_energy_mev=energy, raw_full_sum=full, raw_detector_sum=detector,
            shape_full_bin_weights=full_weights, shape_detector_bin_weights=detector_weights,
            broadened_energy_mev=broad_energy, broadened_detector_bin_weights=broadened,
            detector_mask_array_indices=mask)
        row = {'temperature_label_k': temperature, 'artifact': str(target), 'analysis_energy_bin_mev': step,
            'analysis_cadence_fs': 7.5, 'analysis_segment_duration_ps': len(data)*cadence_s*1e12,
            'calibration_status': 'USER_SUPPLIED_ANALYSIS_CONVENTION; generation cadence/sign not independently verified',
            'detector_definition': 'radius 20 array pixels centered at the two spatial-axis midpoints; angular calibration unverified',
            'raw_sum': float(normalization), 'shape_detector_weight': float(np.sum(detector_weights)),
            'broadening_area_error': float(abs(np.sum(broadened)-np.sum(detector_weights))),
            'added_display_Gaussian_FWHM_mev': 2, 'native_resolution': 'MISSING; 2 meV is additional display convolution, not known common total resolution',
            'normalization_scope': 'explicit shape-only divide by full recorded-array sum; no absolute-intensity inference',
            'uncertainty': 'unavailable from averaged arrays; none invented'}
        results['qualitative_curves'].append(row)
        curves.append((temperature, energy, detector_weights/step, broad_energy, broadened/step))
    results['temperature_shape_difference_L1_full_recorded_detector'] = float(np.sum(abs(curves[1][2]-curves[0][2]))*(curves[0][1][1]-curves[0][1][0]))
    results['missing_metadata'] = {key: value for key, value in previous['missing_metadata'].items() if key not in ('quantum_treatment',)}
    results['cost_vs_accuracy'] = [
        {'method': 'supplied TACAW arrays', 'scientific_scope': 'user-labelled 300/1000 K, 60 keV, 208 slices, Kubo corrected', 'CPU_GPU_seconds': None, 'peak_memory': None, 'numerical_accuracy': 'not independently established', 'reason': 'generation logs unavailable'},
        {'method': 'B8 synthetic full channels', 'scientific_scope': 'two-mode synthetic spatial model; incomparable physical scope', 'numerical_accuracy': 'independent solver and truncation checks recorded', 'cost_source': 'benchmark_results/repeated_20261102_v1/results.json', 'speedup_claim': None}]
    json_write(output/'results.json', results)
    json_write(output/'metadata_manifest.json', {'datasets': results['datasets'], 'user_metadata': supplied, 'missing': results['missing_metadata']})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for temperature, energy, raw, broad_energy, broad in curves:
        axes[0].semilogy(energy, raw, label=f'{temperature} K label')
        axes[1].plot(broad_energy, broad, label=f'{temperature} K label')
    axes[0].set(xlim=(-150, 200), xlabel='energy (meV), associated-analysis calibration', ylabel='shape-only detector density / meV')
    axes[1].set(xlim=(20, 200), xlabel='energy (meV), associated-analysis calibration', ylabel='shape density / meV, additional 2 meV FWHM')
    for ax in axes:
        ax.legend()
    fig.savefig(output/'qualitative_T300_T1000.png', dpi=160)
    plt.close(fig)
    (output/'report.md').write_text('\n'.join([
        '# Benchmark 10: user-authorized qualitative TACAW comparison', '',
        'The user confirms 60 keV, 208 slices, unit-integral intensity normalization and an already-applied Kubo correction. These are recorded as USER_SUPPLIED metadata. The potential is a tentative match to the supplied MOD file; its generation hash remains unverified. No further Kubo correction is applied.', '',
        'The associated local processing helper uses a 2.5 fs timestep and stride 3. The plotted energy axis follows that documented analysis convention explicitly; missing generation records prevent independent verification. Momentum is integrated over a radius-20-pixel disk, with no invented angular calibration. Shape-only normalization divides by the full recorded-array sum, and preserves the unnormalized sums in saved data. No arbitrary energy shift or fitted scale is used.', '',
        'The 2 meV FWHM curve is an additional display convolution. The native window/resolution remains unknown, so it is not a common physical resolution and cannot establish linewidth agreement. All Gaussian tails are retained on an extended grid, without restoring cropped weight.', '',
        'TACAW accuracy is not assumed exact. Classical-region one-phonon comparisons and multiphonon comparisons require separate scope statements. The material Yoshioka comparison remains pending matching computed spectra. Generation costs and statistical uncertainties are unavailable from these averaged arrays; no speedup or uncertainty bars are invented.', '',
        '```bash', '.venv/bin/python -m benchmarks.tacaw_comparison', '.venv/bin/python -m benchmarks.tacaw_qualitative', '```', '',
        '![qualitative curves](qualitative_T300_T1000.png)'])+'\n')


if __name__ == '__main__':
    main()
