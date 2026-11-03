"""Benchmark 10: immutable TACAW audit and metadata-gated comparison tools.

No axes are inferred from array shape or spectral peaks. Present local analysis
settings are recorded as analysis assumptions until generation provenance is
available. This runner never silently reapplies a quantum correction.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from yoshioka.spectral_data import inspect_numeric_array
from .workflow_io import ROOT, environment, json_write, sha256


REQUIRED = {
    'momentum_axes': 'physical coordinate arrays, units, angular/cycles convention, origin and FFT ordering',
    'energy_axis': 'generation sampling cadence, FFT/sign/zero calibration and bin versus density convention',
    'absolute_normalization': 'incident flux and all FFT, trajectory, bead, energy and momentum normalization factors',
    'specimen': 'actual structure, orientation, lattice constants, repeats, thickness, surfaces and periodicity',
    'electron': 'beam energy, illumination, potential parametrization, grid, projection, slice spacing and filters',
    'force_model': 'exact pair style, parameter file and hash, masses and units',
    'thermal_dynamics': 'actual temperature, bead representation, ensemble, thermostat, timestep, equilibration and production settings',
    'spectral_processing': 'segment duration, window, overlap, averaging, zero padding and prior broadening',
    'quantum_treatment': 'dynamics and estimator, and evidence whether Kubo correction has already been applied',
    'elastic_signal': 'included, coherent mean removed, subtracted or zero-loss masked',
    'uncertainty': 'independent trajectories or source statistical blocks',
    'runtime_memory': 'trustworthy generation timings, hardware, software, thread counts and peak memory'
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='benchmarks/configs/tacaw_20261102_v1.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = ROOT/config['output']
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    datasets = []
    diagnostics = {}
    for item in config['datasets']:
        data, record = inspect_numeric_array(ROOT/item['file'])
        counterpart = Path(item['local_counterpart'])
        identical = counterpart.is_file() and sha256(counterpart) == record['sha256']
        record['exact_local_counterpart'] = {'path': str(counterpart), 'identical_sha256': identical}
        record['metadata'] = {key: {'status': 'MISSING', 'value': None, 'needed': value} for key, value in REQUIRED.items()}
        record['metadata']['temperature_label'] = {'status': 'USER_SUPPLIED', 'value_k': item['temperature_k'], 'source': 'user dataset description; actual dynamics logs not available'}
        record['analysis_assumptions'] = {'axis_order': 'energy,y,x in associated analysis helper; generation order not independently confirmed',
            'timestep_fs': 2.5, 'stride': 3, 'analysis_energy_step_mev': 4.13566769692386/(600*2.5*3/1000),
            'analysis_segment_duration_ps': 4.5, 'analysis_source': '/home/zuxian/Documents/PhD/HiPhive_TRPMD/multiphonon_analysis.py',
            'interpretation': 'these are verified analysis defaults applied to the hash-matched files, not verified generation metadata; they do not calibrate the primary physical comparison'}
        # Only raw index diagnostics are valid without physical coordinate/flux
        # metadata. Preserve every raw total and label the lack of calibration.
        axis0 = np.sum(data, axis=tuple(range(1, data.ndim)))
        remaining = np.sum(data, axis=0)
        diagnostics[item['file']] = (axis0, remaining)
        record['raw_sum_uninterpreted_units'] = float(np.sum(axis0))
        record['processing_applied'] = ['read-only array inspection', 'raw sums over array axes for index diagnostics']
        record['no_correction_or_normalization_applied'] = True
        datasets.append(record)
    sources = {}
    for name in config['associated_analysis_files']:
        path = Path(name)
        sources[name] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    results = {'benchmark': 'Benchmark 10: TACAW comparison', 'status': 'BLOCKED_BY_METADATA',
               'category_statuses': {'metadata_completeness': 'INCOMPLETE', 'physical_comparability': 'BLOCKED_BY_METADATA',
                                    'spectral_comparison': 'BLOCKED_BY_METADATA', 'runtime_memory_comparison': 'NOT_ASSESSED'},
               'datasets': datasets, 'associated_analysis_sources': sources, 'missing_metadata': REQUIRED,
               'force_model_matching': 'NOT_ASSESSED: Tersoff filename does not establish ordinary Tersoff versus MOD; no matched-model branch invented',
               'uncertainty': 'unavailable from averaged arrays; no statistical bars fabricated',
               'TACAW_computation_cost': 'UNAVAILABLE; inspection timing is not TACAW generation cost',
               'environment': environment(), 'inspection_seconds': time.monotonic()-started,
               'method_reference': 'https://doi.org/10.1103/PhysRevLett.134.036402',
               'physical_differences': ['auxiliary-wave configurational scattering approximates channel dynamics',
                   'full Yoshioka retains channel-energy phases during traversal; correction depends on energy and thickness',
                   'harmonic occupations, effective harmonic modes, anharmonic classical MD and TRPMD are distinct approximations',
                   'Kubo factor relates quantum and Kubo correlations exactly; MD/TRPMD approximation to the Kubo correlation is not exact for arbitrary nonlinear operators']}
    json_write(output/'results.json', results)
    json_write(output/'metadata_manifest.json', {'datasets': datasets, 'associated_analysis_sources': sources, 'missing': REQUIRED})
    np.savez_compressed(output/'raw_index_diagnostics.npz', **{
        f'{i}_{kind}': value for i, (_, pair) in enumerate(diagnostics.items()) for kind, value in zip(('axis0_sum', 'axes12_sum'), pair)})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for item, (filename, (line, image)) in zip(config['datasets'], diagnostics.items()):
        axes[0].semilogy(np.arange(len(line)), line, label=f"{item['temperature_k']} K label")
    axes[0].set_xlabel('array axis 0 index (physical calibration unverified)')
    axes[0].set_ylabel('raw sum, dataset units')
    axes[0].legend()
    for ax, item, pair in zip(axes[1:], config['datasets'], diagnostics.values()):
        im = ax.imshow(np.log10(np.maximum(pair[1], np.finfo(float).tiny)), origin='lower')
        ax.set(title=f"{item['temperature_k']} K label: log10 raw axis-0 sum", xlabel='array axis 2 index', ylabel='array axis 1 index')
        fig.colorbar(im, ax=ax)
    fig.savefig(output/'uncalibrated_array_diagnostics.png', dpi=160)
    plt.close(fig)
    text = ['# Benchmark 10: TACAW provenance and comparison audit', '',
            '**Primary physical/spectral comparison: BLOCKED_BY_METADATA.**', '',
            'Both immutable input arrays have exact SHA-256 matches in the local HiPhive_TRPMD directory. The matches establish file identity. They do not establish the interatomic parameterization from the word Tersoff. Associated filenames specify 9 beads at 300 K and 4 at 1000 K; actual dynamics require the generation records.', '',
            'The associated analysis helper explicitly processes these matching filenames and uses a 2.5 fs timestep, stride 3, and an FFT-centered energy axis. These are preserved as analysis assumptions, separate from verified generation metadata. The plotting notebooks describe some other datasets as already Kubo corrected, but no generation record proves this for the two supplied arrays. No correction is applied here.', '',
            '| Dataset label | Shape | Finite | Nonfinite | Negative | Maximum raw value |', '|---|---|---:|---:|---:|---:|']
    for record in datasets:
        text.append(f"| {record['metadata']['temperature_label']['value_k']} K | {record['shape']} | {record['finite']} | {record['nonfinite']} | {record['negative_real']} | {record['maximum_real']:.8g} |")
    text += ['', 'Required missing generation evidence:', '']+[f'- **{key}:** {value}.' for key, value in REQUIRED.items()]
    text += ['', 'Raw figures use array indices and uninterpreted intensity units. They are inspection diagnostics, not an absolute EELS validation. No scale/energy fitting, per-array normalization, Kubo multiplication, background subtraction, broadening or uncertainty estimates were applied.', '',
             'No TACAW runtime/memory estimate or speedup follows from a saved array. A matched cost-versus-accuracy comparison remains unavailable until trustworthy generation logs or a runnable workflow are recovered.', '',
             'The reusable loader, conservative bin-overlap rebinning and explicit metadata gate are implemented in yoshioka/spectral_data.py. The primary comparison remains gated until actual axes, normalization, force model, specimen and beam conventions can be matched. A small auxiliary-wave/channel comparison will be recorded separately with the solver benchmark.', '',
             'Method: [Castellanos-Reyes et al., PRL 134, 036402](https://doi.org/10.1103/PhysRevLett.134.036402). Quantum/Kubo conversion is exact between the corresponding quantum correlations; replacing that Kubo correlation by a nuclear-trajectory estimate introduces an approximation for nonlinear operators.', '',
             '```bash', '.venv/bin/python -m benchmarks.tacaw_comparison --config benchmarks/configs/tacaw_20261102_v1.json', '```', '',
             '![raw diagnostics](uncalibrated_array_diagnostics.png)']
    (output/'report.md').write_text('\n'.join(text)+'\n')


if __name__ == '__main__':
    main()
