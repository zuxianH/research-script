"""Memory-bounded elastic refinements; exact slice operators streamed from disk."""

import argparse
import hashlib
import json
import mmap
from pathlib import Path
import time

import numpy as np
from scipy import constants

from yoshioka.elastic_propagation import ElasticPropagator, antialias_aperture
from yoshioka.free_propagation import TransverseGrid
from .abtem_reference import load_structure, reference_configuration, source_audit
from .elastic_observables import angular_masks, compare_observables, diffraction_probabilities, intensity_observables
from .elastic_refinement import assess, outputs
from .reference_solutions import relative_wave_error
from .workflow_io import ROOT, ResourceLimits, environment, fingerprint, json_write, sha256


def stream_series(atoms, n, slices, old, output, limits, audit):
    import abtem
    from abtem.multislice import FresnelPropagator, conventional_multislice_step
    from abtem.antialias import AntialiasAperture
    folder = output/f'n{n}_s{slices}'
    folder.mkdir(parents=True, exist_ok=True)
    contract = {'n': n, 'slices': slices, 'energy_ev': 60000, 'structure': sha256(ROOT/'Si_110_temp300.lmp'),
                'native_sigma': audit['native_sigma_per_v_angstrom'], 'native_wavelength': audit['native_wavelength_angstrom'],
                'source': sha256(__file__), 'reference_sources': audit['source_sha256']}
    digest = fingerprint(contract)
    state_path = folder/'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'digest': digest, 'contract': contract, 'completed_repeats': 0, 'cases': [], 'cumulative_seconds': 0.}
    if state['digest'] != digest or not 0 <= state['completed_repeats'] <= 52:
        raise ValueError('incompatible streamed elastic checkpoint')
    if state['completed_repeats'] == 52:
        return state
    # Only one real native unit stack is resident. No full SI duplicate and no
    # full complex transmission/multiplier stacks are allocated in memory.
    limits.check(estimated_bytes=n*n*slices*8+n*n*16*16+350*1024**2)
    started = time.monotonic()
    length = atoms.cell.lengths()
    dz_a = float(length[2])/slices
    builder = abtem.Potential(atoms, gpts=(n, n), slice_thickness=(dz_a,)*slices,
        parametrization='lobato', projection='finite', plane='xy', origin=(0, 0, 0), periodic=True, device='cpu')
    potential_path = folder/'native_potential.npy'
    if potential_path.exists():
        native_array = np.load(potential_path, mmap_mode='r', allow_pickle=False)
        if native_array.shape != (slices, n, n) or native_array.dtype != np.dtype('float64'):
            raise ValueError('native potential checkpoint shape/dtype mismatch')
        if sha256(potential_path) != state.get('native_potential_sha256'):
            raise ValueError('native potential checkpoint checksum mismatch')
    else:
        native = builder.build(lazy=False)
        temporary = folder/'native_potential.tmp.npy'
        np.save(temporary, native.array)
        temporary.replace(potential_path)
        del native
        native_array = np.load(potential_path, mmap_mode='r', allow_pickle=False)
        state['native_potential_sha256'] = sha256(potential_path)
        json_write(state_path, state)
    native = abtem.PotentialArray(array=native_array, slice_thickness=(dz_a,)*slices, extent=tuple(length[:2]))
    grid = TransverseGrid(length[0]*1e-10, length[1]*1e-10, n, n)
    solver = ElasticPropagator(grid, 60000*constants.elementary_charge, antialias_aperture(grid))
    solver.sigma = audit['native_sigma_per_v_angstrom']*1e10
    wavelength = audit['native_wavelength_angstrom']*1e-10
    solver.free.wavenumber_m_inv = 2*np.pi/wavelength
    transmission_path = folder/'transmissions.npy'
    if not transmission_path.exists():
        temporary = folder/'transmissions.tmp.npy'
        transmission = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.complex128, shape=(slices, n, n))
        for i in range(slices):
            limits.check()
            transmission[i] = solver.transmission(native_array[i].T*1e-10)
            if (i+1) % 16 == 0:
                transmission.flush()
                transmission._mmap.madvise(mmap.MADV_DONTNEED)
        transmission.flush()
        del transmission
        temporary.replace(transmission_path)
        state['transmission_sha256'] = sha256(transmission_path)
        json_write(state_path, state)
    elif sha256(transmission_path) != state.get('transmission_sha256'):
        raise ValueError('transmission cache checksum mismatch')
    transmission = np.load(transmission_path, mmap_mode='r', allow_pickle=False)
    native_array._mmap.madvise(mmap.MADV_DONTNEED)
    transmission._mmap.madvise(mmap.MADV_DONTNEED)
    multiplier = solver.free.multiplier(dz_a*1e-10)*solver.reciprocal_filter
    waves = abtem.PlaneWave(energy=60000, gpts=n, extent=tuple(length[:2]), normalize=True, device='cpu').build(lazy=False)
    factor = n*n/np.sqrt(grid.lx_m*grid.ly_m)
    own = waves.array.T.copy()*factor
    if state['completed_repeats']:
        wave_path = folder/state['wave_checkpoint']
        if sha256(wave_path) != state['wave_checkpoint_sha256']:
            raise ValueError('wave checkpoint checksum mismatch')
        with np.load(wave_path, allow_pickle=False) as d:
            own = d['own'].copy()
            waves.array[:] = d['reference_native']
    propagator, aperture = FresnelPropagator(), AntialiasAperture()
    last_commit = started
    for repeat in range(state['completed_repeats']+1, 53):
        try:
            limits.check()
        except (TimeoutError, MemoryError) as exc:
            state['stop_reason'] = str(exc)
            break
        for i, potential_slice in enumerate(native.generate_slices()):
            own = np.fft.ifft2(np.fft.fft2(own*transmission[i])*multiplier)
            waves = conventional_multislice_step(waves, potential_slice, propagator, aperture,
                conjugate=False, transpose=False, order=1)
            if (i+1) % 16 == 0:
                transmission._mmap.madvise(mmap.MADV_DONTNEED)
                native_array._mmap.madvise(mmap.MADV_DONTNEED)
        if repeat in (13, 26, 52):
            reference = waves.array.T*factor
            diff = diffraction_probabilities(own, grid)
            ref_diff = diffraction_probabilities(reference, grid)
            values = intensity_observables(diff, grid, wavelength, old['selected_beams'])
            ref_values = intensity_observables(ref_diff, grid, wavelength, old['selected_beams'])
            region = angular_masks(grid, wavelength)['region_up_to_50_mrad']
            error = relative_wave_error(own, reference, grid)
            intensity_error = float(np.sum(abs(diff-ref_diff)[region])/np.sum(ref_diff[region]))
            target = folder/f'r{repeat}.npz'
            np.savez_compressed(target, own=own, reference=reference, diffraction=diff, reference_diffraction=ref_diff)
            case = {'grid': n, 'slices_per_repeat': slices, 'z_repeats': repeat,
                    'thickness_nm': float(length[2]*repeat*.1), 'relative_exit_wave_error': error,
                    'diffraction_L1_error': intensity_error, 'observables': values, 'reference_observables': ref_values,
                    'implementation_passed': error < 1e-6 and intensity_error < 1e-5 and compare_observables(values, ref_values)['passed'],
                    'wave_artifact': str(target), 'grid_spacing_angstrom': [length[0]/n, length[1]/n],
                    'slice_spacing_angstrom': [dz_a]*slices}
            state['cases'].append(case)
            print(f"streamed elastic n{n} s{slices} r{repeat}: complex error {error:.3e}; I00={values['transmitted_0_0']:.10g}", flush=True)
        temporary = folder/f'checkpoint_r{repeat}.tmp.npz'
        committed = folder/f'checkpoint_r{repeat}.npz'
        np.savez(temporary, own=own, reference_native=waves.array)
        temporary.replace(committed)
        previous = state.get('wave_checkpoint')
        state.update(completed_repeats=repeat, wave_checkpoint=committed.name, wave_checkpoint_sha256=sha256(committed))
        state['cumulative_seconds'] += time.monotonic()-last_commit
        json_write(state_path, state)
        last_commit = time.monotonic()
        if previous and previous != committed.name:
            (folder/previous).unlink(missing_ok=True)
    state['latest_invocation_seconds'] = time.monotonic()-started
    state['status'] = 'COMPLETE' if state['completed_repeats'] == 52 else 'INCOMPLETE'
    state['shared_potential_metadata'] = {'projection': 'finite', 'parametrization': 'lobato', 'periodic': True,
        'unit_slice_thickness_angstrom': dz_a, 'native_units': 'V angstrom', 'conversion': 'per-slice transpose, times 1e-10 to V m',
        'native_potential_sha256': state['native_potential_sha256'], 'memory_policy': 'one real stack mapped read-only; complex transmissions mapped; one free multiplier; discard read-only resident pages in bounded batches',
        'antialias': audit['reference_settings'], 'atomic_cutoff_angstrom': float(builder.integrator.cutoff('Si')),
        'integration_step_angstrom': builder.integrator.integration_step, 'quadrature_order': builder.integrator.quad_order}
    json_write(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='benchmarks/configs/elastic_repair_20261102_v2.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = ROOT/config['output']
    output.mkdir(parents=True, exist_ok=True)
    old = json.loads((ROOT/'benchmark_results/elastic_si110/results.json').read_text())
    previous_path = ROOT/'benchmark_results/elastic_repair_20261102_v1/results.json'
    previous = json.loads(previous_path.read_text())
    if sha256(ROOT/'benchmarks/elastic_refinement.py') != json.loads(previous_path.with_name('manifest.json').read_text())['source_sha256']:
        raise ValueError('prior matched-constants source record changed')
    # Reuse completed measured waves as historical data, not as checkpoints under
    # a changed algorithm identity. All new propagation has a separate contract.
    cases = [dict(c, reused_from=str(previous_path), reused_results_sha256=sha256(previous_path)) for c in previous['cases']]
    results = {**previous, 'parameters': config, 'cases': cases, 'series': [], 'status': 'INCOMPLETE',
               'previous_resource_limitation': 'v1 allocated all complex slice transmissions and free multipliers before checking RSS; observed peak 6180 MiB exceeded 3 GiB. v2 streams those arrays and preflights allocations.',
               'environment': environment()}
    limits = ResourceLimits(config['runtime_limit_seconds'], config['memory_limit_gib'], 100)
    atoms, _ = load_structure(ROOT/'Si_110_temp300.lmp')
    with reference_configuration():
        for n, slices in config['settings']:
            try:
                series = stream_series(atoms, n, slices, old, output, limits, previous['source_audit'])
            except (TimeoutError, MemoryError) as exc:
                results['resource_stop'] = str(exc)
                break
            results['series'].append(series)
            results['cases'].extend(series['cases'])
            results['comparisons'] = assess(results['cases'], config['comparisons'])
            json_write(output/'results.json', results)
            if series['status'] != 'COMPLETE':
                results['resource_stop'] = series.get('stop_reason')
                break
    required = assess(results['cases'], config['required_final_comparisons'])
    results['implementation_status'] = 'PASS' if all(c['implementation_passed'] for c in results['cases']) else 'FAIL'
    results['physical_convergence_status'] = 'PASS' if len(required) == 3*len(config['required_final_comparisons']) and all(c['passed'] for c in required) else 'INCOMPLETE'
    results['status'] = 'PASS' if results['implementation_status'] == results['physical_convergence_status'] == 'PASS' else 'INCOMPLETE'
    results['accepted_settings'] = config['settings'][-1] if results['status'] == 'PASS' else None
    results['resource_costs'] = limits.costs()
    json_write(output/'results.json', results)
    outputs(results, output)
    with (output/'report.md').open('a') as stream:
        stream.write('\nThe previous version reached a 6180 MiB transient RSS because it cached every complex slice multiplier in RAM. This version stores transmissions in a mapped file and uses one free multiplier; the physical slice operators are unchanged. The original v1 resource record remains preserved. Resume with `.venv/bin/python -m benchmarks.elastic_streaming`.\n')
    json_write(output/'manifest.json', {'status': results['status'], 'source_sha256': sha256(__file__), 'config_sha256': sha256(args.config), 'results_sha256': sha256(output/'results.json')})


if __name__ == '__main__':
    main()
