"""Versioned Benchmark 2: identical constants and targeted static refinements."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import constants

from yoshioka.elastic_propagation import ElasticPropagator, antialias_aperture
from .abtem_reference import (SharedSlices, build_shared_slices, export_shared_inputs,
    load_structure, reference_configuration, source_audit)
from .elastic_observables import (angular_masks, compare_observables,
    diffraction_probabilities, intensity_observables)
from .reference_solutions import relative_wave_error
from .workflow_io import ROOT, ResourceLimits, environment, fingerprint, json_write, sha256


def shared_slices(atoms, n, slices, old, output):
    """Reuse immutable original slices when available; otherwise build anew."""
    import abtem
    from yoshioka.free_propagation import TransverseGrid
    original = ROOT/f'benchmark_results/elastic_si110/shared_inputs/n{n}_s{slices}.npz'
    if not original.exists():
        shared = build_shared_slices(atoms, n, slices, 60000)
        export_shared_inputs(shared, output/f'n{n}_s{slices}.npz', (13, 26, 52))
        shared.metadata['reused_original_shared_input'] = False
        return shared
    with np.load(original, allow_pickle=False) as data:
        array = data['potential_v_angstrom_sxy'].copy()
        thickness = data['unit_slice_thickness_angstrom'].copy()
        extent = data['extent_angstrom'].copy()
        incident_si = data['incident_si_yx'].copy()
        factor = float(data['native_to_si_wave_factor'])
    native = abtem.PotentialArray(array=array, slice_thickness=thickness, extent=extent)
    incident = abtem.PlaneWave(energy=60000, gpts=n, extent=extent, normalize=True, device='cpu').build(lazy=False)
    metadata = next(r for r in old['shared_configurations'] if r['grid'] == [n, n] and r['slices_per_repeat'] == slices).copy()
    metadata.update(reused_original_shared_input=True, original_shared_input_sha256=sha256(original))
    grid = TransverseGrid(extent[0]*1e-10, extent[1]*1e-10, n, n)
    return SharedSlices(native, array.swapaxes(-2, -1)*1e-10, thickness*1e-10, grid, incident, incident_si, factor, metadata)


def matched_series(shared, n, slices, old, output, limits, audit):
    """Checkpoint both independent waves at each complete lattice repeat.

    Production uses the existing SI transmission and free multipliers, cached
    once per distinct slice. Reference calls abTEM's own step and FFTW routines.
    No global-phase adjustment, wave normalization or intensity scaling occurs.
    """
    from abtem.multislice import FresnelPropagator, conventional_multislice_step
    from abtem.antialias import AntialiasAperture
    contract = {'grid': n, 'slices': slices, 'energy_ev': 60000,
                'structure_sha256': sha256(ROOT/'Si_110_temp300.lmp'),
                'potential_array_sha256': hashlib.sha256(shared.native.array.tobytes()).hexdigest(),
                'native_sigma': audit['native_sigma_per_v_angstrom'], 'native_wavelength': audit['native_wavelength_angstrom'],
                'code_sha256': sha256(__file__), 'solver_sha256': sha256(ROOT/'yoshioka/elastic_propagation.py'),
                'reference_source_sha256': audit['source_sha256']}
    digest = fingerprint(contract)
    folder = output/'cache'/digest[:20]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'state.json'
    state = json.loads(path.read_text()) if path.exists() else {
        'digest': digest, 'completed_repeats': 0, 'cumulative_seconds': 0., 'cases': [], 'contract': contract}
    if state['digest'] != digest or not 0 <= state['completed_repeats'] <= 52:
        raise ValueError('incompatible elastic checkpoint')
    solver = ElasticPropagator(shared.grid, 60000*constants.elementary_charge, antialias_aperture(shared.grid))
    # These are independently reported coefficients from the reference's actual
    # constants table. They are shared inputs, not fitted effective parameters.
    solver.sigma = audit['native_sigma_per_v_angstrom']*1e10
    solver.free.wavenumber_m_inv = 2*np.pi/(audit['native_wavelength_angstrom']*1e-10)
    wavelength = audit['native_wavelength_angstrom']*1e-10
    transmissions = solver.transmission(shared.projected_potential_v_m)
    multipliers = np.array([solver.free.multiplier(float(dz))*solver.reciprocal_filter for dz in shared.thicknesses_m])
    own = shared.incident_si.copy()
    waves = shared.incident_native.copy()
    if state['completed_repeats']:
        committed = folder/state['wave_checkpoint']
        if sha256(committed) != state['wave_checkpoint_sha256']:
            raise ValueError('elastic checkpoint wave checksum mismatch')
        with np.load(committed, allow_pickle=False) as d:
            own = d['own'].copy()
            waves.array[:] = d['reference_native']
    propagator, aperture = FresnelPropagator(), AntialiasAperture()
    slices_native = list(shared.native.generate_slices())
    started = time.monotonic()
    last_commit = started
    for repeat in range(state['completed_repeats']+1, 53):
        try:
            limits.check(estimated_bytes=n*n*slices*32+n*n*16*12)
        except (TimeoutError, MemoryError) as exc:
            state['stop_reason'] = str(exc)
            break
        for i, potential_slice in enumerate(slices_native):
            own = np.fft.ifft2(np.fft.fft2(own*transmissions[i])*multipliers[i])
            waves = conventional_multislice_step(waves, potential_slice, propagator, aperture,
                                                conjugate=False, transpose=False, order=1)
        if repeat in (13, 26, 52):
            reference = waves.array.T*shared.native_to_si_wave_factor
            own_diff = diffraction_probabilities(own, shared.grid)
            ref_diff = diffraction_probabilities(reference, shared.grid)
            region = angular_masks(shared.grid, wavelength)['region_up_to_50_mrad']
            values = intensity_observables(own_diff, shared.grid, wavelength, old['selected_beams'])
            ref_values = intensity_observables(ref_diff, shared.grid, wavelength, old['selected_beams'])
            wave_error = relative_wave_error(own, reference, shared.grid)
            intensity_error = float(np.sum(abs(own_diff-ref_diff)[region])/np.sum(ref_diff[region]))
            case = {'grid': n, 'slices_per_repeat': slices, 'z_repeats': repeat,
                    'thickness_nm': float(np.sum(shared.thicknesses_m)*repeat*1e9),
                    'relative_exit_wave_error': wave_error, 'diffraction_L1_error': intensity_error,
                    'observables': values, 'reference_observables': ref_values,
                    'implementation_passed': wave_error < 1e-6 and intensity_error < 1e-5 and compare_observables(values, ref_values)['passed'],
                    'wave_artifact': str(folder/f'r{repeat}.npz'),
                    'grid_spacing_angstrom': [shared.grid.dx_m*1e10, shared.grid.dy_m*1e10],
                    'slice_spacing_angstrom': (shared.thicknesses_m*1e10).tolist()}
            np.savez_compressed(case['wave_artifact'], own=own, reference=reference, diffraction=own_diff, reference_diffraction=ref_diff)
            state['cases'].append(case)
            print(f"elastic n{n} s{slices} r{repeat}: matched-constants complex error={wave_error:.3e}; I00={values['transmitted_0_0']:.9g}", flush=True)
        # Unique checkpoint generations: a crash between wave write and state
        # commit leaves the preceding committed wave available and unchanged.
        temporary = folder/f'checkpoint_r{repeat}.tmp.npz'
        committed = folder/f'checkpoint_r{repeat}.npz'
        np.savez(temporary, own=own, reference_native=waves.array)
        temporary.replace(committed)
        previous = state.get('wave_checkpoint')
        state['completed_repeats'] = repeat
        state['wave_checkpoint'] = committed.name
        state['wave_checkpoint_sha256'] = sha256(committed)
        state['cumulative_seconds'] += time.monotonic()-last_commit
        json_write(path, state)
        last_commit = time.monotonic()
        if previous and previous != committed.name:
            (folder/previous).unlink(missing_ok=True)
    state['latest_invocation_seconds'] = time.monotonic()-started
    state['status'] = 'COMPLETE' if state['completed_repeats'] == 52 else 'INCOMPLETE'
    json_write(path, state)
    return state


def assess(cases, groups):
    lookup = {(r['grid'], r['slices_per_repeat'], r['z_repeats']): r for r in cases}
    comparisons = []
    for kind, first, second in groups:
        for repeat in (13, 26, 52):
            a, b = lookup.get((*first, repeat)), lookup.get((*second, repeat))
            if a is None or b is None:
                continue
            detail = compare_observables(a['observables'], b['observables'])
            comparisons.append({'kind': kind, 'coarse': first, 'fine': second, 'repeat': repeat,
                                'thickness_nm': b['thickness_nm'], **detail})
    return comparisons


def outputs(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    names = list(results['cases'][0]['observables']) if results['cases'] else []
    for case in results['cases']:
        if case['z_repeats'] == 52:
            axes[0].plot(range(len(names)), [case['observables'][name] for name in names], '.-', label=f"{case['grid']}², {case['slices_per_repeat']} slices/repeat")
    axes[0].set_xticks(range(len(names)), names, rotation=80, fontsize=7)
    axes[0].set_ylabel('absolute probability')
    axes[0].legend(fontsize=7)
    for kind in ('grid', 'slice', 'joint'):
        values = [c['max_significant_relative_error'] for c in results['comparisons'] if c['kind'] == kind]
        axes[1].plot(range(len(values)), values, '.-', label=kind)
    axes[1].axhline(.01, color='black', ls='--', label='1% target')
    axes[1].set_ylabel('maximum significant intensity relative change')
    axes[1].set_xlabel('comparison index (details in results.json)')
    axes[1].legend()
    fig.savefig(output/'elastic_refinements.png', dpi=160)
    plt.close(fig)
    text = ['# Benchmark 2: matched-constants targeted refinements', '', f"Status: **{results['status']}**.", '',
            'The original static [110] geometry, 60 keV plane wave, Lobato finite projected potential, periodic bulk slice convention and abTEM antialias filter are retained. Both propagators use the exact same independently reported wavelength and interaction coefficient from abTEM/ASE. Historical different-constants results remain an immutable sensitivity study. There is no phase alignment, intensity fitting, potential-mean removal or output normalization.', '',
            '| Grid | Slices / repeat | Thickness nm | Complex wave error | Implementation pass |', '|---:|---:|---:|---:|---|']
    for r in results['cases']:
        text.append(f"| {r['grid']} | {r['slices_per_repeat']} | {r['thickness_nm']:.6f} | {r['relative_exit_wave_error']:.5g} | {r['implementation_passed']} |")
    text += ['', '| Kind | Coarse → fine | Repeat | Worst relative intensity change | Pass |', '|---|---|---:|---:|---|']
    for r in results['comparisons']:
        text.append(f"| {r['kind']} | {r['coarse']} → {r['fine']} | {r['repeat']} | {r['max_significant_relative_error']:.5g} | {r['passed']} |")
    text += ['', 'All individual beam/detector absolute intensities, absolute changes and relative changes are retained in results.json. Weak references retain the original 1e-6 significance threshold and 1e-8 absolute criterion; significant intensities retain the 1% criterion. Independent solver agreement does not establish physical discretization convergence.', '',
             'Checkpoints save both independently propagated waves after a full lattice repeat, using immutable wave generations and an atomic state pointer. Source/input/constants checksums guard reuse. Repeat the command to resume:', '',
             '```bash', 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.elastic_refinement --config benchmarks/configs/elastic_repair_20261102_v1.json', '```', '', '![refinements](elastic_refinements.png)']
    (output/'report.md').write_text('\n'.join(text)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='benchmarks/configs/elastic_repair_20261102_v1.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = ROOT/config['output']
    output.mkdir(parents=True, exist_ok=True)
    old = json.loads((ROOT/'benchmark_results/elastic_si110/results.json').read_text())
    atoms, structure = load_structure(ROOT/'Si_110_temp300.lmp')
    limits = ResourceLimits(config['runtime_limit_seconds'], config['memory_limit_gib'], 100)
    audit = source_audit(60000)
    from ase import units
    audit['actual_ASE_constants'] = {name: getattr(units, name) for name in ('_e', '_hplanck', '_me', '_c')}
    results = {'benchmark': 'Benchmark 2 versioned repair', 'status': 'INCOMPLETE', 'parameters': config,
               'source_audit': audit, 'structure': structure, 'environment': environment(),
               'historical_result_sha256': sha256(ROOT/'benchmark_results/elastic_si110/results.json'),
               'historical_failed_convergence': old['convergence'], 'historical_constant_sensitivity': old['constant_diagnostic'],
               'cases': [], 'series': [], 'comparisons': [], 'accepted_settings': None}
    with reference_configuration():
        for n, slices in config['settings']:
            try:
                limits.check(estimated_bytes=n*n*slices*32+n*n*16*12+200*1024**2)
            except (TimeoutError, MemoryError) as exc:
                results['resource_stop'] = str(exc)
                break
            shared = shared_slices(atoms, n, slices, old, output)
            started = time.monotonic()
            series = matched_series(shared, n, slices, old, output, limits, audit)
            series['shared_potential_metadata'] = shared.metadata
            series['latest_series_total_seconds'] = time.monotonic()-started
            results['series'].append(series)
            results['cases'].extend(series['cases'])
            results['comparisons'] = assess(results['cases'], config['comparisons'])
            json_write(output/'results.json', results)
            if series['status'] != 'COMPLETE':
                break
    results['implementation_status'] = 'PASS' if results['cases'] and all(r['implementation_passed'] for r in results['cases']) else 'INCOMPLETE'
    required = assess(results['cases'], config['required_final_comparisons'])
    results['physical_convergence_status'] = 'PASS' if len(required) == 3*len(config['required_final_comparisons']) and all(r['passed'] for r in required) else 'INCOMPLETE'
    results['status'] = 'PASS' if results['implementation_status'] == results['physical_convergence_status'] == 'PASS' else 'INCOMPLETE'
    if results['status'] == 'PASS':
        results['accepted_settings'] = config['settings'][-1]
    results['resource_costs'] = limits.costs()
    json_write(output/'results.json', results)
    outputs(results, output)
    json_write(output/'manifest.json', {'status': results['status'], 'source_sha256': sha256(__file__), 'config_sha256': sha256(args.config), 'results_sha256': sha256(output/'results.json')})


if __name__ == '__main__':
    main()
