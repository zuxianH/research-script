"""Benchmark 11A/11B: independent MOD trajectories, effective harmonic modes.

Training uses actual finite-temperature forces, not odd-force projections or
intensity data. Validation is a separately seeded entire trajectory. Volume is
fixed at the independently validated zero-pressure 0-K reference.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import constants

from yoshioka.molecular_dynamics import checkpointed_md
from .workflow_io import ROOT, ResourceLimits, environment, json_write, sha256


def fit_effective(cs, reference, folder, count, ridge, frame_indices=None):
    from hiphive import ForceConstantPotential, StructureContainer
    displacement = np.load(folder/'displacements_m.npy', mmap_mode='r')
    forces = np.load(folder/'forces_n.npy', mmap_mode='r')
    if frame_indices is None:
        frame_indices = np.linspace(0, len(displacement)-1, count, dtype=int)
    container = StructureContainer(cs)
    for index in frame_indices:
        atoms = reference.copy()
        atoms.new_array('displacements', np.asarray(displacement[index])*1e10)
        atoms.new_array('forces', np.asarray(forces[index])*1e-10/constants.elementary_charge)
        container.add_structure(atoms)
    matrix, target = container.get_fit_data()
    scale = np.linalg.norm(matrix, axis=0)
    if np.any(scale == 0):
        raise ValueError('unconstrained effective harmonic parameter')
    normalized = matrix/scale
    if ridge:
        coefficients = np.linalg.solve(normalized.T@normalized+ridge*np.eye(len(scale)), normalized.T@target)
    else:
        coefficients = np.linalg.lstsq(normalized, target, rcond=None)[0]
    params = coefficients/scale
    return ForceConstantPotential(cs, params), {'training_frames': list(map(int, frame_indices)), 'ridge': ridge,
        'training_force_relative_RMS_error': float(np.linalg.norm(matrix@params-target)/np.linalg.norm(target)),
        'parameter_count': len(params), 'rank': int(np.linalg.matrix_rank(normalized)),
        'condition_number': float(np.linalg.cond(normalized)), 'force_target': 'actual finite-T MOD forces, including anharmonic even-force components'}


def validate_forces(fcp, reference, folder, count=32):
    displacement = np.load(folder/'displacements_m.npy', mmap_mode='r')
    forces = np.load(folder/'forces_n.npy', mmap_mode='r')
    indices = np.linspace(0, len(displacement)-1, count, dtype=int)
    fc = fcp.get_force_constants(reference).get_fc_array(2)
    prediction = -np.einsum('ijab,sjb->sia', fc, np.asarray(displacement[indices])*1e10)
    actual = np.asarray(forces[indices])*1e-10/constants.elementary_charge
    return {'validation_frames': indices.tolist(), 'force_relative_RMS_error': float(np.linalg.norm(prediction-actual)/np.linalg.norm(actual)),
            'force_absolute_RMS_error_ev_per_angstrom': float(np.sqrt(np.mean((prediction-actual)**2))),
            'validation_scope': 'whole independent seed trajectory, never adjacent-frame train/test split'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='benchmarks/configs/anharmonic_20261102_v1.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = ROOT/config['output']
    output.mkdir(parents=True, exist_ok=True)
    from ase.io import read
    unit = read(ROOT/'benchmark_results/si_lattice_dynamics/equilibrium.extxyz')
    limits = ResourceLimits(config['runtime_limit_seconds'], config['memory_limit_gib'], 100)
    results = {'benchmark': 'Benchmark 11 effective harmonic modes and MD spectra', 'status': 'INCOMPLETE', 'parameters': config,
        'force_model': {'pair_style': 'tersoff/mod', 'potential_sha256': sha256(ROOT/'Si.tersoff.mod')},
        'volume_policy': 'fixed validated 0-K equilibrium volume at all temperatures; thermal expansion not included or conflated with frequency renormalization',
        'dynamics': [], 'fits': [], 'environment': environment(), 'source_sha256': sha256(__file__)}
    for temperature in config['temperatures_k']:
        for repetitions in config['supercells']:
            for seed in config['seeds']:
                settings = {'temperature_k': temperature, 'repetitions': repetitions, 'seed': seed,
                    'timestep_fs': config['timestep_fs'], 'sample_stride': config['sample_stride'],
                    'equilibration_steps': config['equilibration_steps'], 'production_steps': config['production_steps'],
                    'thermostat_time_fs': config['thermostat_time_fs']}
                folder, state = checkpointed_md(unit, ROOT/'Si.tersoff.mod', settings, output/'trajectories', limits)
                row = {'settings': settings, 'folder': str(folder), 'state': state}
                results['dynamics'].append(row)
                json_write(output/'checkpoint.json', results)
                print('MD', temperature, repetitions, seed, state['status'], 'mean T', state.get('mean_temperature_k'), 'seconds', state['cumulative_seconds'], flush=True)
                if state['status'] != 'COMPLETE':
                    json_write(output/'results.json', results)
                    return
    # Timestep control is an independent trajectory at half dt with the same
    # physical duration and sampling cadence; it is not mixed into training.
    for temperature in config['temperatures_k']:
        settings = {'temperature_k': temperature, 'repetitions': config['supercells'][0], 'seed': config['seeds'][1]+177,
            'timestep_fs': config['timestep_fs']/2, 'sample_stride': config['sample_stride']*2,
            'equilibration_steps': config['equilibration_steps']*2, 'production_steps': config['production_steps']*2,
            'thermostat_time_fs': config['thermostat_time_fs']}
        folder, state = checkpointed_md(unit, ROOT/'Si.tersoff.mod', settings, output/'trajectories', limits)
        results['dynamics'].append({'settings': settings, 'folder': str(folder), 'state': state, 'role': 'timestep_control'})
        json_write(output/'checkpoint.json', results)
        print('MD half timestep', temperature, state['status'], 'mean T', state.get('mean_temperature_k'), flush=True)
    from hiphive import ClusterSpace
    from yoshioka.lattice_dynamics import real_space_from_supercell
    import contextlib
    cache = {}
    q = np.array([[0, 0, 0], [0, .5, 0], [.17, .23, .11]])
    for temperature in config['temperatures_k']:
        for repetitions in config['supercells']:
            reference = unit.repeat(tuple(repetitions))
            train = next(d for d in results['dynamics'] if d['settings']['temperature_k'] == temperature and d['settings']['repetitions'] == repetitions and d['settings']['seed'] == config['seeds'][0])
            validation = next(d for d in results['dynamics'] if d['settings']['temperature_k'] == temperature and d['settings']['repetitions'] == repetitions and d['settings']['seed'] == config['seeds'][1])
            setups = [(count, config['selected_cutoff_angstrom'], 0.) for count in config['training_counts']]
            setups += [(config['training_counts'][-1], cutoff, 0.) for cutoff in config['cutoffs_angstrom']]
            setups += [(config['training_counts'][-1], config['selected_cutoff_angstrom'], ridge) for ridge in config['ridge_values']]
            for count, cutoff, ridge in dict.fromkeys(setups):
                limits.check()
                started = time.monotonic()
                with (output/'hiphive_fit.log').open('a') as log, contextlib.redirect_stdout(log):
                    if cutoff not in cache:
                        cache[cutoff] = ClusterSpace(unit, [cutoff])
                    fcp, fitted = fit_effective(cache[cutoff], reference, Path(train['folder']), count, ridge)
                    heldout = validate_forces(fcp, reference, Path(validation['folder']))
                    fc = fcp.get_force_constants(reference).get_fc_array(2)
                    real, metadata = real_space_from_supercell(unit, repetitions, fc)
                omega, eigen = real.modes(q)
                row = {'temperature_k': temperature, 'supercell': repetitions, 'count': count, 'cutoff_angstrom': cutoff,
                    **fitted, **heldout, 'q_fractional': q.tolist(), 'energies_mev': (omega*constants.hbar/(1e-3*constants.elementary_charge)).tolist(),
                    'model_metadata': metadata, 'elapsed_seconds': time.monotonic()-started}
                tag = f'T{temperature}_s'+''.join(map(str, repetitions))+f'_n{count}_r{cutoff:g}_ridge{ridge:g}'
                target = output/(tag+'.npz')
                real.save(target)
                fcp.write(str(output/(tag+'.fcp')))
                row['model_file'] = str(target)
                results['fits'].append(row)
                json_write(output/'checkpoint.json', results)
                print('fit', tag, 'heldout', heldout['force_relative_RMS_error'], 'seconds', row['elapsed_seconds'], flush=True)
    results['resource_costs'] = limits.costs()
    json_write(output/'results.json', results)


if __name__ == '__main__':
    main()
