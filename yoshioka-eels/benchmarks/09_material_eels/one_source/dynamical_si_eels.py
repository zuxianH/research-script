"""Benchmark 9A dynamical one-source Si EELS; 9B feasibility kept separate.

The finite periodic phonon model is explicitly reduced and remains distinct
from the bulk-BZ convergence study. Actual electron thickness is fixed for each
phonon refinement. Real finite-mode sources have periodic transverse boundaries.
"""

import argparse
import json
from math import comb
from pathlib import Path
import time

import numpy as np
from scipy import constants

from yoshioka.dynamical_source import (full_transverse_vectors, plane_field,
    propagate_incident, propagate_source_block)
from yoshioka.electron_phonon import LobatoSilicon, common_diagonal_covariance, full_space_born
from yoshioka.free_propagation import TransverseGrid, electron_wavenumber
from yoshioka.lattice_dynamics import RealSpaceForceConstants, finite_periodic_modes
from .eels_observables import DETECTORS, PEAK_GROUPS, absolute_spectrum, detector_masks
from .workflow_io import ROOT, ResourceLimits, fingerprint, json_write, sha256


def source_specimen(model, thickness_repeats, phonon_z, temperature):
    modes = finite_periodic_modes(model, (2, 2, phonon_z))
    cells = np.array(list(np.ndindex(2, 2, thickness_repeats)))
    positions = ((cells @ model.cell_m)[:, None]+model.positions_m[None]).reshape(-1, 3)
    translated = cells % np.array([2, 2, phonon_z])
    index = np.ravel_multi_index(translated.T, (2, 2, phonon_z))[:, None]*4+np.arange(4)[None]
    displacement = modes['displacements_m'][index.ravel()]
    omega = modes['omega_rad_s']
    nbar = np.zeros(len(omega)) if temperature == 0 else 1/np.expm1(constants.hbar*omega/(constants.Boltzmann*temperature))
    covariance, diagnostic = common_diagonal_covariance(modes['displacements_m']*np.sqrt(2*nbar+1)[None, None])
    return positions, displacement, omega, nbar, covariance, diagnostic


def calculate_case(model, settings, output, limits, block_size=4):
    n, steps, repeats, phonon_z, temperature = [settings[key] for key in ('grid', 'slices', 'thickness_repeats', 'phonon_z', 'temperature_k')]
    positions, d, omega, nbar, covariance, cov_diagnostic = source_specimen(model, repeats, phonon_z, temperature)
    padding = settings['padding_angstrom']*1e-10
    positions[:, 2] += padding
    thickness = repeats*model.cell_m[2, 2]
    length = thickness+2*padding
    grid = TransverseGrid(2*model.cell_m[0, 0], 2*model.cell_m[1, 1], n, n)
    allowed = int((limits.memory_bytes-350*1024**2)//(n*n*steps*16))-3
    if allowed < 1:
        raise MemoryError('one source mode exceeds configured memory preflight')
    block_size = min(block_size, allowed)
    energy = 60000*constants.elementary_charge
    code = {name: sha256(ROOT/name) for name in ('benchmarks/dynamical_si_eels.py', 'yoshioka/dynamical_source.py', 'yoshioka/electron_phonon.py')}
    contract = {'settings': settings, 'source': code, 'positions_m': positions.tolist(), 'covariance_m2': covariance.tolist(),
        'mode_energies_sha256': fingerprint(omega.tolist()), 'mode_displacements_sha256': fingerprint(d.tolist()),
        'force_constant_file': sha256(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')}
    digest = fingerprint(contract)
    folder = output/'cache'/digest[:20]
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder/'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'digest': digest, 'contract': contract,
        'completed_modes': 0, 'mode_count': len(omega), 'cumulative_seconds': 0., 'status': 'INCOMPLETE'}
    if state['digest'] != digest or not 0 <= state['completed_modes'] <= len(omega):
        raise ValueError('invalid dynamical-source checkpoint')
    shapes = {'loss': (len(omega), n, n), 'gain': (len(omega), n, n),
              'born_loss': (len(omega), n, n), 'born_gain': (len(omega), n, n)}
    arrays = {}
    for name, shape in shapes.items():
        target = folder/f'{name}.npy'
        if target.exists():
            arrays[name] = np.load(target, mmap_mode='r+', allow_pickle=False)
            if arrays[name].shape != shape or arrays[name].dtype != np.dtype('complex128'):
                raise ValueError('dynamical-source checkpoint shape mismatch')
        else:
            if state['completed_modes']:
                raise ValueError('missing committed source-mode array')
            arrays[name] = np.lib.format.open_memmap(target, mode='w+', dtype=np.complex128, shape=shape)
            arrays[name][:] = 0
    if state['status'] == 'COMPLETE':
        return summarize_case(settings, folder, state, arrays, omega, grid), state
    limits.check(estimated_bytes=(block_size+3)*n*n*steps*16+350*1024**2)
    started = time.monotonic()
    background_path = folder/'background_fourier.npy'
    if not state.get('background_complete'):
        background = plane_field(positions, covariance, grid, length, steps, background_path)
        state.update(background_complete=True, background_sha256=sha256(background_path))
        json_write(state_path, state)
    else:
        if sha256(background_path) != state['background_sha256']:
            raise ValueError('background checksum mismatch')
        background = np.load(background_path, mmap_mode='r', allow_pickle=False)
    incident_path = folder/'incident_before_local.npy'
    elastic = settings['elastic_background']
    if not state.get('incident_complete'):
        incident_exit, incident, diagnostic = propagate_incident(grid, energy, background, length, incident_path, elastic)
        np.save(folder/'incident_exit.npy', incident_exit)
        state.update(incident_complete=True, incident_diagnostic=diagnostic, incident_sha256=sha256(incident_path))
        json_write(state_path, state)
    else:
        if sha256(incident_path) != state['incident_sha256']:
            raise ValueError('incoming-wave checkpoint checksum mismatch')
        incident = np.load(incident_path, mmap_mode='r', allow_pickle=False)
    qxy = full_transverse_vectors(grid)
    selected = np.linalg.norm(qxy, axis=-1)/electron_wavenumber(energy)*1000 <= 20+1e-12
    last_commit = started
    resumed = state['completed_modes']
    try:
        for start in range(resumed, len(omega), block_size):
            limits.check()
            stop = min(start+block_size, len(omega))
            source_path = folder/'working_source.npy'
            source = plane_field(positions, covariance, grid, length, steps, source_path, d[:, :, start:stop])
            loss, diagnostic = propagate_source_block(grid, energy, omega[start:stop], background, source, incident, length, elastic)
            arrays['loss'][start:stop] = loss*np.sqrt(nbar[start:stop]+1)[:, None, None]
            if temperature:
                gain, _ = propagate_source_block(grid, energy, -omega[start:stop], background, source, incident, length, elastic)
                arrays['gain'][start:stop] = gain*np.sqrt(nbar[start:stop])[:, None, None]
            born = full_space_born(LobatoSilicon.from_abtem_table(), positions, d[:, :, start:stop], covariance,
                qxy[selected], omega[start:stop], energy, grid.lx_m*grid.ly_m, length)
            target = np.zeros((stop-start, n*n), dtype=np.complex128)
            target[:, selected] = born*np.sqrt(nbar[start:stop]+1)[:, None]
            arrays['born_loss'][start:stop] = target.reshape(stop-start, n, n)
            if temperature:
                born_gain = full_space_born(LobatoSilicon.from_abtem_table(), positions, d[:, :, start:stop], covariance,
                    qxy[selected], -omega[start:stop], energy, grid.lx_m*grid.ly_m, length)
                target[:, selected] = born_gain*np.sqrt(nbar[start:stop])[:, None]
                arrays['born_gain'][start:stop] = target.reshape(stop-start, n, n)
            for array in arrays.values():
                array.flush()
            state['completed_modes'] = stop
            state['maximum_imaginary_source'] = max(state.get('maximum_imaginary_source', 0), diagnostic['maximum_relative_imaginary_source'])
            state['cumulative_seconds'] += time.monotonic()-last_commit
            json_write(state_path, state)
            last_commit = time.monotonic()
            del source
            source_path.unlink(missing_ok=True)
    except (TimeoutError, MemoryError) as exc:
        state['stop_reason'] = str(exc)
    state['status'] = 'COMPLETE' if state['completed_modes'] == len(omega) else 'INCOMPLETE'
    state['resumed_from_modes'] = resumed
    state['latest_invocation_seconds'] = time.monotonic()-started
    state['specimen_thickness_nm'] = float(thickness*1e9)
    state['covariance_diagnostic'] = cov_diagnostic
    json_write(state_path, state)
    if state['status'] != 'COMPLETE':
        return {'settings': settings, 'status': 'INCOMPLETE', 'checkpoint': str(state_path), 'completed_modes': state['completed_modes']}, state
    return summarize_case(settings, folder, state, arrays, omega, grid), state


def summarize_case(settings, folder, state, arrays, omega, grid):
    energy = 60000*constants.elementary_charge
    qxy = full_transverse_vectors(grid)
    theta = np.linalg.norm(qxy, axis=-1)/electron_wavenumber(energy)*1000
    selected = theta <= 20+1e-12
    masks = detector_masks(theta)
    amplitudes = np.concatenate((np.asarray(arrays['loss']), np.asarray(arrays['gain']))).reshape(2*len(omega), -1)
    reference = np.concatenate((np.asarray(arrays['born_loss']), np.asarray(arrays['born_gain']))).reshape(2*len(omega), -1)
    transfer = np.r_[omega, -omega]*constants.hbar
    probability = abs(amplitudes)**2
    ref_probability = abs(reference)**2
    axis = np.arange(-120, 120.025, .05)
    spectra = {}
    values = {}
    for name, mask in masks.items():
        weights = np.sum(probability[:, mask], axis=1)
        born_weights = np.sum(ref_probability[:, mask], axis=1)
        spectra[name] = absolute_spectrum(transfer, weights, axis)
        spectra['born_'+name] = absolute_spectrum(transfer, born_weights, axis)
        values[name] = float(np.sum(weights))
        for group, (lo, hi) in PEAK_GROUPS.items():
            e = transfer/(1e-3*constants.elementary_charge)
            values[name+'/'+group] = float(np.sum(weights[(e >= lo) & (e < hi)]))
        values[name+'/gain'] = float(np.sum(weights[transfer < 0]))
    pure_error = float(np.linalg.norm((amplitudes-reference)[:, selected])/np.linalg.norm(reference[:, selected])) if not settings['elastic_background'] else None
    artifact = folder/'summary.npz'
    np.savez_compressed(artifact, energy_grid_mev=axis, **{f'spectrum_{name}': value for name, value in spectra.items()},
        transfers_j=transfer, mode_momentum_weights=probability, born_mode_momentum_weights=ref_probability, qxy_m_inv=qxy,
        elastic_exit=np.load(folder/'incident_exit.npy'), omega_rad_s=omega)
    return {'settings': settings, 'status': 'COMPLETE', 'checkpoint': str(folder/'state.json'), 'observables': values,
        'mode_count': len(omega), 'thickness_nm': state['specimen_thickness_nm'], 'artifact': str(artifact),
        'pure_Born_limit_complex_relative_error': pure_error, 'max_incident_norm_drift': state['incident_diagnostic']['max_incident_norm_drift'],
        'total_inelastic_weight_all_represented_momenta': float(np.sum(probability)),
        'Born_weight_within_20_mrad': float(np.sum(ref_probability[:, selected])),
        'scope': 'coarse finite periodic phonon model and finite atom slab; Fourier-mode populations are dimensionless, summed without angular-bin-area factors; exact forward is a represented discrete mode here'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='benchmarks/configs/dynamical_si_20261102_v1.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = ROOT/config['output']
    output.mkdir(parents=True, exist_ok=True)
    model = RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
    limits = ResourceLimits(config['runtime_limit_seconds'], config['memory_limit_gib'], 100)
    results = {'benchmark': 'Benchmark 9A/9B dynamical Si EELS', 'status': 'INCOMPLETE', 'parameters': config,
        '9A_status': 'INCOMPLETE', '9B_status': 'INCOMPLETE', 'cases': [],
        'model': 'validated MOD equilibrium geometry, separate from historical original static geometry; Lobato h=-eV electron potential; finite atom slab with explicit vacuum; transverse periodic electron cell 2x2',
        'thermal_trace': 'linear-source effective model: exact incoherent harmonic first moments nB+1 and nB, with DW covariance included once; not the full nonlinear finite-T Fock response',
        'initial_state': 'plane wave 1/sqrt(A); outgoing channels initially zero; common-v Hermitian diagonal propagators; undepleted source approximation'}
    for settings in config['cases']:
        try:
            row, state = calculate_case(model, settings, output, limits, config['mode_block_size'])
        except (TimeoutError, MemoryError) as exc:
            results['resource_stop'] = str(exc)
            break
        results['cases'].append(row)
        json_write(output/'checkpoint.json', results)
        print('9A', settings, row['status'], row.get('observables', {}).get('full_0_20_mrad'), 'Born control', row.get('pure_Born_limit_complex_relative_error'), flush=True)
        if row['status'] != 'COMPLETE':
            results['resource_stop'] = state.get('stop_reason')
            break
    # This is a representation feasibility measurement, not a substitute 9B.
    # B8 measured timings and dimensions are retained rather than extrapolating
    # a scope-unmatched speedup or silently using one-source waves as full waves.
    b8 = json.loads((ROOT/'benchmark_results/repeated_20261102_v1/results.json').read_text())
    material_channels = [{'modes': modes, 'p': p, 'channels': comb(modes+p, p),
        'local_dense_matrix_bytes_at_65_squared': comb(modes+p, p)**2*65**2*16,
        'state_complex128_bytes_at_65_squared': comb(modes+p, p)*65**2*16} for modes in (93, 189) for p in (1, 2, 3)]
    results['9B_feasibility'] = {'status': 'INCOMPLETE', 'dense_representation_estimates': material_channels,
        'measured_B8_case_scaling': [{'channels': c['channel_count'], 'grid': c['settings']['n'], 'seconds': c['elapsed_seconds']} for c in b8['cases'] if c['settings']['thickness_nm'] == 80 and c['settings']['g'] == 2 and not c['settings']['linear']],
        'missing_calculation': 'full repeated-inelastic material propagation and macroscopic thermal trace; a sparse/structured channel representation with converged phonon/channel/thermal truncations is required',
        'limitation': 'the present dense local-matrix representation is infeasible for required material channels; no claim that every possible algorithm is infeasible'}
    results['prerequisites'] = {'elastic_new_status': json.loads((ROOT/'benchmark_results/elastic_repair_20261102_v2/results.json').read_text())['status'],
        'phonon_convergence_status': json.loads((ROOT/'benchmark_results/repair_20261102_v1/results.json').read_text())['status']}
    results['resource_costs'] = limits.costs()
    json_write(output/'results.json', results)
    write_outputs(results, output)


def write_outputs(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    complete = [r for r in results['cases'] if r['status'] == 'COMPLETE' and r['settings']['elastic_background']]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for row in complete:
        with np.load(row['artifact']) as d:
            axis = d['energy_grid_mev']
            label = f"{row['thickness_nm']:.3g} nm, {row['settings']['temperature_k']} K, n={row['settings']['grid']}"
            axes[0].plot(axis, d['spectrum_full_0_20_mrad'], label=label)
            axes[1].plot(axis, d['spectrum_full_0_20_mrad']-d['spectrum_born_full_0_20_mrad'], label=label)
    axes[0].set_title('Dynamical one-source material spectra')
    axes[1].set_title('Dynamical minus pure Born, same input model')
    for ax in axes:
        ax.set(xlim=(-100, 100), xlabel='electron energy loss (meV)', ylabel='absolute inelastic probability / meV')
        ax.legend(fontsize=6)
    fig.savefig(output/'dynamical_vs_Born_spectra.png', dpi=160)
    plt.close(fig)
    text = ['# Benchmark 9A and 9B: distinct material milestones', '',
        '**9A: INCOMPLETE material convergence. 9B: INCOMPLETE, full material repeated-channel propagation not performed.**', '',
        '9A propagates both the incoming and outgoing waves through the same real, thermally DW-averaged finite-atom Lobato potential and integrates one linear normal-mode source. Kinetic propagation, channel-energy phases and source insertion are retained. The Fourier potential/derivative is independent of the final Born atomic-sum reference. Odd spatial/longitudinal grids retain conjugate Fourier pairs. Padding, grids, slices and the finite phonon bath are separate settings.', '',
        'The geometry is the explicitly separate validated MOD equilibrium geometry, with finite-z atom slabs and vacuum. It does not overwrite the original structure or the historical bulk-periodic elastic convention. Phonons have a declared 2×2×Nz periodic normalization, embedded at fixed electron thickness. This coarse bath is not a converged bulk q integral. Its discrete electron Fourier populations are summed as weights; exact forward is a represented mode, not a continuum density.', '',
        'At finite T the linear effective source admits an exact incoherent harmonic occupation trace giving nB+1 and nB. The DW covariance is included once in the effective potential. This is not a full nonlinear finite-T matrix-element calculation. The undepleted source approximation has unit-norm incident propagation plus outgoing intensity; their sum is not a closed-channel norm conservation test. No outgoing wave is renormalized.', '',
        '9B feasibility is quantified from measured B8 scaling and explicit material channel/matrix counts. Full material repeated scattering and its thermal trace require a different structured representation; the single-source results are not labeled 9B. No macroscopic thermal occupation is silently cut off.', '',
        '| Grid | Slices | Thickness nm | T K | Mode count | Full-disk inelastic weight | Born-limit complex error |', '|---:|---:|---:|---:|---:|---:|---:|']
    for row in results['cases']:
        if row['status'] != 'COMPLETE':
            continue
        text.append(f"| {row['settings']['grid']} | {row['settings']['slices']} | {row['thickness_nm']:.6g} | {row['settings']['temperature_k']} | {row['mode_count']} | {row['observables']['full_0_20_mrad']:.7g} | {row['pure_Born_limit_complex_relative_error']} |")
    text += ['', '```bash', 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.dynamical_si_eels', '```', '',
        'Repeat to resume compatible completed-mode prefixes. Raw signed transition energies and absolute momentum-resolved weights are stored for detector/resolution reprocessing. The finite source approximation and remaining discretization gates are explicit; no overall material PASS is inferred.', '',
        '![spectra](dynamical_vs_Born_spectra.png)']
    (output/'report.md').write_text('\n'.join(text)+'\n')


if __name__ == '__main__':
    main()
