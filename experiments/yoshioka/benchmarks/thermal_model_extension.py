"""Versioned restart, higher-order-force and intrinsic-linewidth experiments.

Run from project root, e.g. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m benchmarks.thermal_model_extension --stage restart.
Historical results and fixed-harmonic models are read-only inputs.
"""
import argparse
import contextlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy import constants

from benchmarks.workflow_io import ResourceLimits, environment, json_write, sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT/'benchmark_results/14_two_phonon_eels/complete_leading_20261102_v1/thermal_comparison'


def source_record():
    names = ['benchmarks/thermal_model_extension.py', 'yoshioka/thermal_model_validation.py',
             'yoshioka/molecular_dynamics_checked.py', 'yoshioka/mod_forces.py', 'Si.tersoff.mod',
             'benchmark_results/si_lattice_dynamics/equilibrium.extxyz']
    return {name: sha256(ROOT/name) for name in names}


def checked_data():
    path = ROOT/'benchmark_results/repair_20261102_v2/B11_checked/results.json'
    data = json.loads(path.read_text())
    if not data['trajectory_energy_quality_gate']['passed']:
        raise ValueError('corrected reference failed its energy gate')
    return path, data


def run_restart(output):
    from ase.io import read
    from yoshioka.molecular_dynamics_checked import checkpointed_md
    from yoshioka.mod_forces import ModifiedTersoffForces
    folder = output/'restart'
    folder.mkdir(parents=True, exist_ok=True)
    settings = {'repetitions': [3, 4, 4], 'temperature_k': 1000, 'seed': 519723,
                'timestep_fs': .5, 'sample_stride': 10, 'equilibration_steps': 1000,
                'production_steps': 6000, 'thermostat_time_fs': 100}
    criteria = {'max_relative_fresh_force_error': 1e-10,
                'max_restart_relative_trajectory_RMS_difference': 1e-7,
                'max_energy_drift_over_thermal_KE': 1e-3,
                'reason': 'fresh-force and energy gates retained; restart equivalence tolerance allows floating-point changes over a 3ps chaotic trajectory'}
    if (folder/'results.json').exists():
        previous=json.loads((folder/'results.json').read_text())
        if previous['settings']!=settings or previous['criteria']!=criteria:
            raise ValueError('completed restart evidence has a different contract; use a new output directory')
        print('restart: reusing the completed stop/resume experiment and fresh-force evidence',flush=True)
        return previous
    json_write(folder/'contract.json', {'settings': settings, 'criteria': criteria,
                'source': source_record(), 'command': 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.thermal_model_extension --stage restart'})
    limits = ResourceLimits(300, 4)
    unit = read(ROOT/'benchmark_results/si_lattice_dynamics/equilibrium.extxyz')
    continuous, continuous_state = checkpointed_md(unit, ROOT/'Si.tersoff.mod', settings, folder/'continuous', limits)
    class StopAfterCommit(ResourceLimits):
        def check(self, estimated_bytes=0):
            super().check(estimated_bytes)
            for state_path in (folder/'resumed').glob('*/state.json'):
                state = json.loads(state_path.read_text())
                if state['completed_frames'] >= 200 and state['status'] != 'COMPLETE':
                    raise TimeoutError('intentional restart-equivalence test stop after first committed prefix')
    partial, partial_state = checkpointed_md(unit, ROOT/'Si.tersoff.mod', settings, folder/'resumed', StopAfterCommit(300, 4))
    # Preserve the partial state before continuing the same contract.
    json_write(folder/'interrupted_state.json', partial_state)
    resumed, resumed_state = checkpointed_md(unit, ROOT/'Si.tersoff.mod', settings, folder/'resumed', limits)
    differences = {}
    for name in ['displacements_m', 'velocities_m_s', 'forces_n', 'temperature_k', 'total_energy_j']:
        a = np.load(continuous/f'{name}.npy', mmap_mode='r')
        b = np.load(resumed/f'{name}.npy', mmap_mode='r')
        differences[name] = float(np.linalg.norm(a-b)/max(np.linalg.norm(a), 1e-300))
    atoms = unit.repeat(tuple(settings['repetitions']))
    fresh_rows = []
    with ModifiedTersoffForces(atoms, ROOT/'Si.tersoff.mod') as calc:
        for label, path in [('continuous', continuous), ('resumed', resumed)]:
            u = np.load(path/'displacements_m.npy', mmap_mode='r')
            f = np.load(path/'forces_n.npy', mmap_mode='r')
            for frame in [198, 199, 200, 201, 398, 399, 400, 401, 599]:
                fresh = calc.evaluate(atoms.positions+u[frame]*1e10)['forces_ev_per_angstrom']*constants.elementary_charge/1e-10
                fresh_rows.append({'trajectory': label, 'frame': frame,
                    'relative_force_error': float(np.linalg.norm(fresh-f[frame])/np.linalg.norm(f[frame]))})
    passed = (partial_state['status'] == 'INCOMPLETE' and partial_state['completed_frames'] == 200 and
              resumed_state.get('resumed_from_frames') == 200 and continuous_state['status'] == resumed_state['status'] == 'COMPLETE' and
              max(differences.values()) < criteria['max_restart_relative_trajectory_RMS_difference'] and
              max(r['relative_force_error'] for r in fresh_rows) < criteria['max_relative_fresh_force_error'] and
              max(s['max_energy_drift_relative_to_thermal_kinetic_energy'] for s in [continuous_state, resumed_state]) < criteria['max_energy_drift_over_thermal_KE'])
    result = {'status': 'PASS' if passed else 'FAIL', 'criteria': criteria, 'settings': settings,
              'continuous': str(continuous), 'resumed': str(resumed), 'partial_completed_frames': partial_state['completed_frames'],
              'resumed_from_frames': resumed_state.get('resumed_from_frames'), 'relative_RMS_differences': differences,
              'fresh_force_checks': fresh_rows,
              'energy_drifts': [s['max_energy_drift_relative_to_thermal_kinetic_energy'] for s in [continuous_state, resumed_state]],
              'resource_costs': limits.costs(), 'source': source_record()}
    json_write(folder/'results.json', result)
    print('restart', result['status'], differences, flush=True)
    return result


def select_trajectory(data, temperature, seed, repetitions=(4, 6, 6)):
    return next(r for r in data['trajectories'] if r['settings']['temperature_k'] == temperature and
                r['settings']['seed'] == seed and r['settings']['repetitions'] == list(repetitions))


def build_fit(cs, reference, data_folder, frames):
    from hiphive import ForceConstantPotential, StructureContainer
    container = StructureContainer(cs)
    u = np.load(data_folder/'displacements_m.npy', mmap_mode='r')
    forces = np.load(data_folder/'forces_n.npy', mmap_mode='r')
    for frame in frames:
        atoms = reference.copy()
        atoms.new_array('displacements', np.asarray(u[frame])*1e10)
        atoms.new_array('forces', np.asarray(forces[frame])*1e-10/constants.elementary_charge)
        container.add_structure(atoms)
    matrix, target = container.get_fit_data()
    scales = np.linalg.norm(matrix, axis=0)
    if np.any(scales == 0):
        raise ValueError('unconstrained higher-order coefficient')
    normalized = matrix/scales
    coeff, residuals, rank, singular = np.linalg.lstsq(normalized, target, rcond=None)
    params = coeff/scales
    return ForceConstantPotential(cs, params), {'training_frames': frames.tolist(), 'parameter_count': len(params),
        'rank': int(rank), 'condition_number': float(singular[0]/singular[-1]),
        'training_force_relative_RMS_error': float(np.linalg.norm(matrix@params-target)/np.linalg.norm(target))}


def force_validation(fcp, reference, data_folder, frames):
    from hiphive.calculators import ForceConstantCalculator
    calc = ForceConstantCalculator(fcp.get_force_constants(reference))
    u = np.load(data_folder/'displacements_m.npy', mmap_mode='r')
    force = np.load(data_folder/'forces_n.npy', mmap_mode='r')
    predictions, actual = [], []
    for index in frames:
        atoms = reference.copy()
        atoms.positions += np.asarray(u[index])*1e10
        atoms.calc = calc
        predictions.append(atoms.get_forces())
        actual.append(np.asarray(force[index])*1e-10/constants.elementary_charge)
    predictions, actual = np.asarray(predictions), np.asarray(actual)
    residual = predictions-actual
    # Five non-overlapping temporal blocks, retaining their correlated nature.
    block_id = np.minimum((frames/(len(u)/5)).astype(int), 4)
    blocks = []
    for block in np.unique(block_id):
        selected = block_id == block
        blocks.append({'block': int(block), 'frames': frames[selected].tolist(),
            'relative_RMS_error': float(np.linalg.norm(residual[selected])/np.linalg.norm(actual[selected]))})
    amplitudes = np.linalg.norm(u[frames], axis=-1).ravel()
    atom_error = np.linalg.norm(residual, axis=-1).ravel()
    edges = np.quantile(amplitudes, [0, .25, .5, .75, 1])
    amplitude_rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (amplitudes >= lo) & (amplitudes <= hi)
        amplitude_rows.append({'displacement_range_angstrom': [float(lo*1e10), float(hi*1e10)],
                               'force_residual_RMS_ev_per_angstrom': float(np.sqrt(np.mean(atom_error[mask]**2)))})
    return {'validation_frames': frames.tolist(), 'force_relative_RMS_error': float(np.linalg.norm(residual)/np.linalg.norm(actual)),
            'force_absolute_RMS_error_ev_per_angstrom': float(np.sqrt(np.mean(residual**2))),
            'temporal_blocks': blocks, 'residual_by_displacement': amplitude_rows}, predictions, actual


def model_constraints(fcp, unit, reference, output_model):
    from hiphive.calculators import ForceConstantCalculator
    from yoshioka.lattice_dynamics import real_space_from_supercell
    from yoshioka.mod_forces import force_constant_residuals
    fcs = fcp.get_force_constants(reference)
    fc2 = fcs.get_fc_array(2)
    model, metadata = real_space_from_supercell(unit, (4, 6, 6), fc2)
    model.save(output_model)
    q = (np.indices((8, 8, 8)).reshape(3, -1).T+.5)/8-.5
    # Inspect signed dynamical eigenvalues; modes() would reject instability.
    eigenvalues = np.linalg.eigvalsh(model.dynamical(q))
    calc = ForceConstantCalculator(fcs)
    rng = np.random.default_rng(432619)
    delta = rng.normal(scale=.03, size=(len(reference), 3))
    atoms = reference.copy(); atoms.positions += delta; atoms.calc = calc
    forces = atoms.get_forces()
    translated = atoms.copy(); translated.positions += [.011, -.023, .017]; translated.calc = calc
    translated_forces = translated.get_forces()
    # Inversion is a space-group operation of diamond. Include the mapping of
    # basis atoms so that this tests covariance, not force parity at fixed sites.
    cell = np.diag(reference.cell)
    scaled = reference.positions % cell
    target = (-scaled) % cell
    distance = target[:, None]-scaled[None]
    distance -= np.rint(distance/cell)*cell
    mapping = np.argmin(np.linalg.norm(distance, axis=-1), axis=1)
    inversion = None
    if len(set(mapping)) == len(reference) and np.max(np.linalg.norm(distance[np.arange(len(reference)), mapping], axis=1)) < 1e-5:
        inv = reference.copy(); transformed = np.zeros_like(delta); transformed[mapping] = -delta
        inv.positions += transformed; inv.calc = calc
        inversion = float(np.linalg.norm(inv.get_forces()[mapping]+forces)/np.linalg.norm(forces))
    return {'model_metadata': metadata, 'fc2_sum_rules': force_constant_residuals(fc2),
            'translational_force_relative_residual': float(np.linalg.norm(translated_forces-forces)/np.linalg.norm(forces)),
            'inversion_force_relative_residual': inversion,
            'minimum_dynamical_eigenvalue_s_minus_2': float(eigenvalues.min()),
            'negative_modes_on_shifted_8_mesh': int(np.sum(eigenvalues < -1e-8*np.max(abs(eigenvalues)))),
            'local_stability_scope': 'quadratic Hessian at reference only; local positivity does not establish global or finite-temperature stability of a truncated polynomial'}


def run_forces(output):
    from ase.io import read
    from hiphive import ClusterSpace
    folder = output/'force_models'; folder.mkdir(parents=True, exist_ok=True)
    input_path, data = checked_data()
    limits = ResourceLimits(600, 4)
    unit = read(ROOT/'benchmark_results/si_lattice_dynamics/equilibrium.extxyz')
    reference = unit.repeat((4, 6, 6))
    config = {'cutoffs_angstrom': [[5.], [5., 3.5]], 'training_counts': [16, 32, 64],
              'training_seed': 723571, 'final_validation_seed': 837213,
              'training_time_range_ps': [0, 80], 'model_selection_time_range_ps': [80, 100],
              'validation_time_range_ps': [0, 100], 'force_gate_relative_RMS': .15,
              'frequency_gate_mev': 'abs(E_model-E_MD)<max(0.5,2*original_MD_spread); inherited from B11',
              'model_choice': 'compare fixed quadratic and fixed cubic forms, no EELS/TACAW fitting; candidate not promoted without full dynamic validation'}
    json_write(folder/'contract.json', {'config': config, 'input': str(input_path), 'input_sha256': sha256(input_path), 'source': source_record()})
    results = {'status': 'INCOMPLETE', 'config': config, 'fits': [], 'input_sha256': sha256(input_path),
               'limitations': ['only validated fixed 0K volume', 'no ensemble/temperature transfer acceptance',
                               'cubic local Hessian does not establish global thermal stability',
                               'interaction range and supercell are inherited sensitivity controls, not proven converged for new cubic model']}
    with (folder/'hiphive.log').open('w') as log, contextlib.redirect_stdout(log):
        spaces = {str(c): ClusterSpace(unit, c) for c in config['cutoffs_angstrom']}
    for temperature in [300, 1000]:
        train = select_trajectory(data, temperature, 723571)
        validation = select_trajectory(data, temperature, 837213)
        train_folder = Path(train['folder']); validation_folder = Path(validation['folder'])
        n = len(np.load(train_folder/'displacements_m.npy', mmap_mode='r'))
        for cutoff in config['cutoffs_angstrom']:
            for count in config['training_counts']:
                limits.check(); started = time.monotonic()
                name = f'T{temperature}_order{len(cutoff)+1}_n{count}'
                frames = np.linspace(0, int(.8*n)-1, count, dtype=int)
                validation_frames = np.linspace(0, n-1, 80, dtype=int)
                with (folder/'hiphive.log').open('a') as log, contextlib.redirect_stdout(log):
                    fcp, fit = build_fit(spaces[str(cutoff)], reference, train_folder, frames)
                    selected, _, _ = force_validation(fcp, reference, train_folder, np.linspace(int(.8*n), n-1, 16, dtype=int))
                    heldout, prediction, actual = force_validation(fcp, reference, validation_folder, validation_frames)
                    constraints = model_constraints(fcp, unit, reference, folder/f'{name}.npz') if count == 64 else None
                    fcp.write(str(folder/f'{name}.fcp'))
                np.savez_compressed(folder/f'{name}_heldout.npz', frames=validation_frames,
                                    predicted_forces_ev_per_angstrom=prediction, actual_forces_ev_per_angstrom=actual)
                row = {'temperature_k': temperature, 'cutoffs_angstrom': cutoff, 'count': count,
                       **fit, **heldout, 'model_selection_relative_RMS_error': selected['force_relative_RMS_error'],
                       'force_gate_passed': heldout['force_relative_RMS_error'] < .15,
                       'training_trajectory': str(train_folder), 'validation_trajectory': str(validation_folder),
                       'actual_training_temperature_k': train['state']['mean_temperature_k'],
                       'actual_validation_temperature_k': validation['state']['mean_temperature_k'],
                       'constraints': constraints, 'elapsed_seconds': time.monotonic()-started,
                       'combined_status': 'NOT_ACCEPTED_PENDING_DYNAMICAL_VALIDATION',
                       'model_file': str(folder/f'{name}.fcp')}
                results['fits'].append(row); json_write(folder/f'{name}.json', row); json_write(folder/'checkpoint.json', results)
                print('force model', name, heldout['force_relative_RMS_error'], 'seconds', row['elapsed_seconds'], flush=True)
    old = json.loads((ROOT/'benchmark_results/repair_20261102_v2/B11_refit/results.json').read_text())
    results['historical_quadratic_refit'] = [{'temperature_k': r['temperature_k'], 'count': r['count'],
        'force_relative_RMS_error': r['force_relative_RMS_error'], 'frequency_passes': sum(s['original_frequency_gate_pass'] for s in r['frequency_rows']),
        'frequency_total': len(r['frequency_rows'])} for r in old['fits']]
    results['status'] = 'COMPLETED_MODEL_CAPACITY_DIAGNOSTIC'
    results['B11A_status'] = 'FAIL_COMBINED_VALIDATION_RETAINED'
    results['resource_costs'] = limits.costs(); results['source'] = source_record()
    json_write(folder/'results.json', results)
    return results


def run_linewidth(output):
    from ase.io import read
    from yoshioka.lattice_dynamics import RealSpaceForceConstants
    from yoshioka.mode_spectra import project_normal_coordinates
    from yoshioka.thermal_model_validation import fit_mode_profile, linewidth_gate, mode_periodogram
    folder = output/'linewidth'; folder.mkdir(parents=True, exist_ok=True)
    input_path, data = checked_data(); limits = ResourceLimits(600, 4)
    unit = read(ROOT/'benchmark_results/si_lattice_dynamics/equilibrium.extxyz')
    model = RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
    omega, eigen = model.modes([0, 0, 0]); group = [9, 10, 11]
    config = {'mode_subspace': 'Gamma optical triplet', 'group': group, 'temperatures_k': [300, 1000],
              'durations_ps': [20, 50, 100], 'windows': ['hann', 'rectangular'],
              'primary_seeds': [723571, 837213], 'search_windows_mev': [4, 8],
              'line_shapes': ['lorentzian', 'gaussian'],
              'inherited_gates': {'minimum_width_over_window_FWHM': 2, 'maximum_choice_spread_over_width': .5, 'maximum_relative_RMS_residual': .1},
              'additional_gate': 'zero-width profile MSE>1.5*best MSE; diagnostic, not a confidence interval'}
    json_write(folder/'contract.json', {'config': config, 'input_sha256': sha256(input_path), 'source': source_record()})
    result = {'status': 'INCOMPLETE', 'config': config, 'fits': [], 'summaries': [],
              'reference': 'corrected classical MOD NVE trajectories; thermostat only during equilibration',
              'lifetimes_are_classical_model_only': True}
    old_rows = data['frequency_rows']
    for temperature in config['temperatures_k']:
        center = next(r['MD_center_mev'] for r in old_rows if r['temperature_k'] == temperature and r['q_fractional'] == [0.,0.,0.] and r['group'] == group)
        for seed in config['primary_seeds']:
            tr = select_trajectory(data, temperature, seed); atoms = unit.repeat((4,6,6)); path = Path(tr['folder'])
            velocity = np.load(path/'velocities_m_s.npy', mmap_mode='r')
            cadence = tr['state']['sample_interval_s']
            projected = project_normal_coordinates(velocity, atoms.positions*1e-10, atoms.get_masses()*constants.atomic_mass, model, [0,0,0], eigen[:,group])
            for duration in config['durations_ps']:
                count = int(round(duration*1e-12/cadence))
                # Include independent non-overlapping 20ps blocks for the first seed,
                # plus prefixes at all durations for both independent seeds.
                starts = list(range(0, len(projected)-count+1, count)) if duration == 20 and seed == 723571 else [0]
                for start in starts:
                    for window in config['windows']:
                        limits.check()
                        energy, power = mode_periodogram(projected[start:start+count], cadence, window)
                        np.savez_compressed(folder/f'T{temperature}_s{seed}_{duration}ps_start{start}_{window}.npz', energy_mev=energy, power=power)
                        choices = [(4.,'lorentzian')]
                        if duration == 100 and start == 0:
                            choices += [(8.,'lorentzian'), (4.,'gaussian')]
                        for search, shape in choices:
                            fit = fit_mode_profile(energy, power, center, duration, window, search, shape)
                            fit.update(temperature_k=temperature, seed=seed, start_frame=start,
                                       actual_temperature_k=tr['state']['mean_temperature_k'], trajectory=str(path))
                            result['fits'].append(fit); json_write(folder/'checkpoint.json', result)
            print('linewidth', temperature, seed, 'completed fits', len(result['fits']), flush=True)
        rows = [r for r in result['fits'] if r['temperature_k'] == temperature]
        result['summaries'].append({'temperature_k': temperature, **linewidth_gate(rows)})
    result['status'] = 'NOT_RESOLVED' if any(r['status'] == 'NOT_RESOLVED' for r in result['summaries']) else 'RESOLVED_IN_TESTED_CLASSICAL_MODEL'
    result['resource_costs'] = limits.costs(); result['source'] = source_record()
    result['additional_missing_controls'] = ['new independent equal-temperature size, timestep and thermostat preparation controls', 'volume and ensemble variation', 'quantum nuclear linewidths']
    json_write(folder/'results.json', result)
    return result


def run_force_controls(output):
    """Interaction-range and supercell diagnostics for the new cubic candidate."""
    from ase.io import read
    from hiphive import ClusterSpace,ForceConstantPotential
    from hiphive.calculators import ForceConstantCalculator
    import spglib
    folder=output/'force_controls';folder.mkdir(parents=True,exist_ok=True)
    _,data=checked_data();limits=ResourceLimits(600,4)
    unit=read(ROOT/'benchmark_results/si_lattice_dynamics/equilibrium.extxyz');reference=unit.repeat((4,6,6))
    contract={'range_cutoffs_angstrom':[[5.,3.5],[5.,4.5]],'training_seed':723571,'independent_validation_seed':837213,
        'train_frames':64,'training_time_ps':[0,80],'validation_blocks':5,'force_gate_relative_RMS':.15,
        'new_range_is_not_selected_by_final_holdout':'fixed sensitivity experiment, no model promotion',
        'volume_policy':'fixed original0K cell only; strain/volume transfer remains untested'}
    json_write(folder/'contract.json',{'settings':contract,'source':source_record()})
    results={'status':'INCOMPLETE','contract':contract,'range_models':[],'supercell_controls':[],'symmetry_controls':[]}
    with (folder/'hiphive.log').open('w') as log,contextlib.redirect_stdout(log):space=ClusterSpace(unit,[5.,4.5])
    for temperature in [300,1000]:
        limits.check();start=time.monotonic();train=select_trajectory(data,temperature,723571);valid=select_trajectory(data,temperature,837213)
        n=len(np.load(Path(train['folder'])/'displacements_m.npy',mmap_mode='r'));frames=np.linspace(0,int(.8*n)-1,64,dtype=int)
        with (folder/'hiphive.log').open('a') as log,contextlib.redirect_stdout(log):
            candidate,fit=build_fit(space,reference,Path(train['folder']),frames)
            heldout,predicted,actual=force_validation(candidate,reference,Path(valid['folder']),np.linspace(0,n-1,80,dtype=int))
            candidate.write(str(folder/f'T{temperature}_cubic_range4.5.fcp'))
            constraints=model_constraints(candidate,unit,reference,folder/f'T{temperature}_cubic_range4.5.npz')
        results['range_models'].append({'temperature_k':temperature,'cutoffs_angstrom':[5.,4.5],**fit,**heldout,'constraints':constraints,'elapsed_seconds':time.monotonic()-start})
        np.savez_compressed(folder/f'T{temperature}_range4.5_heldout.npz',actual=actual,predicted=predicted)
        # Validate a nontrivial actual space-group operation, including its
        # fractional translation and atomic permutation.
        symmetry=spglib.get_symmetry((reference.cell,reference.get_scaled_positions(),reference.numbers),symprec=1e-5)
        chosen=next(i for i,r in enumerate(symmetry['rotations']) if np.linalg.det(r)<0 and not np.array_equal(r,np.eye(3,dtype=int)))
        rotation=symmetry['rotations'][chosen];translation=symmetry['translations'][chosen]
        fractional=reference.get_scaled_positions();target=(fractional@rotation.T+translation)%1
        differences=target[:,None]-fractional[None];differences-=np.rint(differences)
        distance=np.linalg.norm(differences@np.asarray(reference.cell),axis=-1);mapping=np.argmin(distance,axis=1)
        if len(set(mapping))!=len(reference) or np.max(distance[np.arange(len(reference)),mapping])>1e-5:raise ValueError('space-group mapping failed')
        cart_rotation=np.linalg.inv(np.asarray(reference.cell))@rotation.T@np.asarray(reference.cell)
        for label,potential in [('short_range',ForceConstantPotential.read(str(output/'force_models'/f'T{temperature}_order3_n64.fcp'))),('long_range',candidate)]:
            calc=ForceConstantCalculator(potential.get_force_constants(reference));rng=np.random.default_rng(987321)
            displacement=rng.normal(scale=.04,size=(len(reference),3));atoms=reference.copy();atoms.positions+=displacement;atoms.calc=calc;forces=atoms.get_forces()
            transformed=reference.copy();change=np.empty_like(displacement);change[mapping]=displacement@cart_rotation
            transformed.positions+=change;transformed.calc=calc;transformed_forces=transformed.get_forces()
            residual=float(np.linalg.norm(transformed_forces[mapping]-forces@cart_rotation)/np.linalg.norm(forces))
            results['symmetry_controls'].append({'temperature_k':temperature,'model':label,'rotation':rotation.tolist(),'translation_fractional':translation.tolist(),'covariance_relative_residual':residual})
        if temperature==1000:
            for dt in [1,.5]:
                tr=next(r for r in data['trajectories'] if r['settings']['temperature_k']==1000 and r['settings']['repetitions']==[3,4,4] and r['settings']['timestep_fs']==dt)
                small=unit.repeat((3,4,4));small_n=len(np.load(Path(tr['folder'])/'displacements_m.npy',mmap_mode='r'))
                with (folder/'hiphive.log').open('a') as log,contextlib.redirect_stdout(log):
                    heldout,_,_=force_validation(candidate,small,Path(tr['folder']),np.linspace(0,small_n-1,80,dtype=int))
                results['supercell_controls'].append({'temperature_k':temperature,'repetitions':[3,4,4],'timestep_fs':dt,
                    'actual_temperature_k':tr['state']['mean_temperature_k'],**heldout,
                    'scope':'same fitted4x6x6 model on distinct small-cell trajectories; temperatures differ, not pure size or timestep convergence'})
        json_write(folder/'checkpoint.json',results);print('force range control',temperature,results['range_models'][-1]['force_relative_RMS_error'],flush=True)
    results['status']='COMPLETED_CONSTRAINT_AND_RANGE_DIAGNOSTICS';results['combined_status']='NOT_ACCEPTED_PENDING_FULL_DYNAMIC_VOLUME_ENSEMBLE_VALIDATION'
    results['resource_costs']=limits.costs();json_write(folder/'results.json',results);return results


def run_force_dynamics(output):
    """Bounded dynamical holdout of the cubic candidate, without promotion."""
    from ase.io import read
    from ase import units
    from ase.md.verlet import VelocityVerlet
    from hiphive import ForceConstantPotential
    from hiphive.calculators import ForceConstantCalculator
    from yoshioka.lattice_dynamics import RealSpaceForceConstants
    from yoshioka.mode_spectra import project_normal_coordinates, periodogram_modes, fit_windowed_mode
    from benchmarks.anharmonic_validation import groups_at_q
    folder = output/'cubic_dynamics'; folder.mkdir(parents=True, exist_ok=True)
    _, data = checked_data(); limits = ResourceLimits(600,4)
    unit = read(ROOT/'benchmark_results/si_lattice_dynamics/equilibrium.extxyz')
    harmonic = RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
    duration_ps, timestep_fs, stride = 20., .5, 10
    nframes = int(round(duration_ps*1000/timestep_fs/stride))
    config = {'duration_ps': duration_ps, 'timestep_fs': timestep_fs, 'sample_stride': stride,
        'ensemble': 'NVE; initialize from frame0 of entire independent corrected MOD validation trajectory; no thermostat',
        'force_model': 'fixed cubic cutoffs5,3.5Angstrom trained only seed723571 first80ps,64frames',
        'frequency_gate': 'inherited abs(center-MODcenter)<max(0.5meV,2*original_MDspread)',
        'energy_gate_over_thermal_KE': .001,
        'promotion': 'bounded dynamical diagnostic only;20ps does not resolve full100ps/range/size/volume/ensemble validation'}
    json_write(folder/'contract.json', {'config':config,'source':source_record()})
    result = {'status':'INCOMPLETE','config':config,'trajectories':[],'frequency_rows':[]}
    for temperature in [300,1000]:
        tr=select_trajectory(data,temperature,837213); source=Path(tr['folder'])
        atoms=unit.repeat((4,6,6)); positions0=atoms.positions.copy()
        atoms.positions += np.load(source/'displacements_m.npy',mmap_mode='r')[0]*1e10
        atoms.set_velocities(np.load(source/'velocities_m_s.npy',mmap_mode='r')[0]*1e-5/units.fs)
        model_path=output/'force_models'/f'T{temperature}_order3_n64.fcp'
        fcp=ForceConstantPotential.read(str(model_path)); atoms.calc=ForceConstantCalculator(fcp.get_force_constants(unit.repeat((4,6,6))))
        integrator=VelocityVerlet(atoms,timestep_fs*units.fs,logfile=None)
        target=folder/f'T{temperature}'; target.mkdir(exist_ok=True)
        velocity=np.lib.format.open_memmap(target/'velocities_m_s.npy',mode='w+',shape=(nframes,len(atoms),3),dtype='float64')
        energy=np.empty(nframes); temp=np.empty(nframes); max_displacement=0.; completed=0; started=time.monotonic(); error=None
        try:
            for frame in range(nframes):
                if frame%100==0: limits.check()
                integrator.run(stride)
                velocity[frame]=atoms.get_velocities()*units.fs*1e5
                energy[frame]=atoms.get_total_energy()*constants.elementary_charge
                temp[frame]=atoms.get_temperature()
                delta=atoms.positions-positions0; cell=np.diag(atoms.cell); delta-=np.rint(delta/cell)*cell
                max_displacement=max(max_displacement,float(np.max(np.linalg.norm(delta,axis=1))))
                completed=frame+1
        except (TimeoutError,ValueError,MemoryError) as exc:
            error=str(exc)
        velocity.flush(); np.save(target/'total_energy_j.npy',energy[:completed]); np.save(target/'temperature_k.npy',temp[:completed])
        drift=float(np.max(abs(energy[:completed]-energy[0]))/(1.5*(len(atoms)-1)*constants.Boltzmann*np.mean(temp[:completed]))) if completed else None
        row={'temperature_k':temperature,'completed_frames':completed,'requested_frames':nframes,
            'status':'COMPLETE' if completed==nframes else 'INCOMPLETE','error':error,'energy_drift_over_thermal_KE':drift,
            'maximum_displacement_angstrom':max_displacement,'mean_temperature_k':float(np.mean(temp[:completed])) if completed else None,
            'elapsed_seconds':time.monotonic()-started,'input_model_sha256':sha256(model_path),'initial_MD_state_sha256':sha256(source/'state.json')}
        result['trajectories'].append(row);json_write(target/'state.json',row);json_write(folder/'checkpoint.json',result)
        print('cubic MD',temperature,row,flush=True)
        if completed==nframes:
            for q in [[0.,0.,0.],[0.,.5,0.]]:
                _,vectors,groups=groups_at_q(harmonic,q)
                for group in groups:
                    target_row=next(r for r in data['frequency_rows'] if r['temperature_k']==temperature and r['q_fractional']==q and r['group']==group)
                    normal=project_normal_coordinates(velocity,positions0*1e-10,atoms.get_masses()*constants.atomic_mass,harmonic,q,vectors[:,group])
                    e,power=periodogram_modes(normal,stride*timestep_fs*1e-15)
                    fit=fit_windowed_mode(e,power,target_row['MD_center_mev'],duration_ps)
                    np.savez_compressed(target/f'q{q[1]}_group{group[0]}.npz',energy_mev=e,power=power)
                    diff=fit['center_mev']-target_row['MD_center_mev']
                    result['frequency_rows'].append({'temperature_k':temperature,'q_fractional':q,'group':group,**fit,
                        'MOD_center_mev':target_row['MD_center_mev'],'MOD_spread_mev':target_row['MD_conservative_spread_mev'],
                        'candidate_minus_MOD_mev':diff,'original_frequency_gate_pass':abs(diff)<max(.5,2*target_row['MD_conservative_spread_mev']),
                        'frequency_scope':'20ps dynamical candidate; not bare cubic Hessian; independent seed initial state'})
        json_write(folder/'checkpoint.json',result)
    result['status']=('COMPLETED_BOUNDED_DYNAMICAL_DIAGNOSTIC' if all(r['status']=='COMPLETE' for r in result['trajectories']) else 'FAIL_THERMAL_STABILITY' if any('Displacement' in str(r['error']) for r in result['trajectories']) else 'INCOMPLETE')
    result['combined_acceptance']='NOT_ACCEPTED_PENDING_ALL_ORIGINAL_VALIDATION'
    result['resource_costs']=limits.costs(); result['source']=source_record();json_write(folder/'results.json',result)
    return result


def run_failure_controls(output):
    """Repeat cubic basin escape with halved integration timestep."""
    from ase.io import read
    from ase import units
    from ase.md.verlet import VelocityVerlet
    from hiphive import ForceConstantPotential
    from hiphive.calculators import ForceConstantCalculator
    folder=output/'cubic_dynamics'/'timestep_failure_controls_v3';folder.mkdir(parents=True,exist_ok=True)
    _,data=checked_data();limits=ResourceLimits(180,4);tr=select_trajectory(data,1000,837213);source=Path(tr['folder'])
    unit=read(ROOT/'benchmark_results/si_lattice_dynamics/equilibrium.extxyz');reference=unit.repeat((4,6,6))
    u=np.load(source/'displacements_m.npy',mmap_mode='r')[0]*1e10;v=np.load(source/'velocities_m_s.npy',mmap_mode='r')[0]
    contract={'maximum_duration_fs':400,'timesteps_fs':[.5,.25],'source_trajectory_state_sha256':sha256(source/'state.json'),
        'columns':['time_fs','max_displacement_angstrom','temperature_k','energy_ev','max_force_ev_per_angstrom'],
        'purpose':'same independent MODinitialstate; force model unchanged; no force clipping or displacement reset; guard records model escape'}
    json_write(folder/'contract.json',{'settings':contract,'source':source_record()});rows=[]
    models=[('short_range',output/'force_models'/'T1000_order3_n64.fcp'),('long_range',output/'force_controls'/'T1000_cubic_range4.5.fcp')]
    if (output/'quartic_candidate'/'T1000.fcp').exists():models.append(('quartic',output/'quartic_candidate'/'T1000.fcp'))
    for label,model_path in models:
        for dt in [.5,.25]:
            limits.check();atoms=reference.copy();atoms.positions+=u;atoms.set_velocities(v*1e-5/units.fs)
            fcp=ForceConstantPotential.read(str(model_path));atoms.calc=ForceConstantCalculator(fcp.get_force_constants(reference))
            integrator=VelocityVerlet(atoms,dt*units.fs,logfile=None);records=[];error=None;start=time.monotonic()
            for step in range(int(contract['maximum_duration_fs']/dt)):
                try:
                    forces=atoms.get_forces();energy=atoms.get_total_energy();delta=atoms.positions-reference.positions;delta-=np.rint(delta/np.diag(atoms.cell))*np.diag(atoms.cell)
                    records.append([step*dt,float(np.max(np.linalg.norm(delta,axis=1))),atoms.get_temperature(),energy,float(np.max(np.linalg.norm(forces,axis=1)))])
                    integrator.run(1)
                except ValueError as exc:
                    error=str(exc);break
            np.save(folder/f'{label}_dt{dt}_records.npy',np.asarray(records))
            rows.append({'model':label,'model_sha256':sha256(model_path),'dt_fs':dt,'completed_steps':len(records),
                'last_record_time_fs':records[-1][0],'stop':error,'elapsed_seconds':time.monotonic()-start})
    result={'status':('FAIL_THERMAL_STABILITY' if any('Displacement' in str(row['stop']) for row in rows) else 'COMPLETED_BOUNDED_STABILITY_CONTROL'),'rows':rows,'contract':contract,'resource_costs':limits.costs(),
        'inference':'compare each polynomial model at both timesteps; matched escape times support model instability; held-out force RMS and a positive reference Hessian alone are insufficient'}
    json_write(folder/'results.json',result);return result


def run_quartic(output):
    """Test stabilizing higher-order capacity suggested by cubic basin escape."""
    from ase.io import read
    from ase import units
    from ase.md.verlet import VelocityVerlet
    from hiphive import ClusterSpace
    from hiphive.calculators import ForceConstantCalculator
    from yoshioka.lattice_dynamics import RealSpaceForceConstants
    from yoshioka.mode_spectra import project_normal_coordinates,periodogram_modes,fit_windowed_mode
    from benchmarks.anharmonic_validation import groups_at_q
    folder=output/'quartic_candidate';folder.mkdir(parents=True,exist_ok=True)
    _,data=checked_data();limits=ResourceLimits(600,4);unit=read(ROOT/'benchmark_results/si_lattice_dynamics/equilibrium.extxyz');reference=unit.repeat((4,6,6))
    contract={'cutoffs_angstrom':[5.,4.5,3.5],'training_frames':64,'training_seed':723571,'training_time_ps':[0,80],
        'model_selection_time_ps':[80,100],'validation_seed':837213,
        'purpose':'even-order stabilization candidate after measured cubic failure; not accepted from forceRMS alone',
        'dynamical_target_ps':20,'timestep_fs':.5,'dynamic_temperature_k':1000,
        'adaptive_resource_policy':'measure100 warmed steps; if forecast remaining20ps exceeds remaining600s budget, retain2ps bounded pilot and exact continuation estimate',
        'selection_limitation':'independent seed validation previously inspected for cubic diagnostics; no untouched final acceptance set claimed',
        'force_gate_relative_RMS':.15,'energy_gate_over_thermal_KE':.001,'original_frequency_gate':'abs(diff)<max(.5meV,2*originalspread)'}
    json_write(folder/'contract.json',{'settings':contract,'source':source_record()})
    with (folder/'hiphive.log').open('w') as log,contextlib.redirect_stdout(log):space=ClusterSpace(unit,[5.,4.5,3.5])
    result={'status':'INCOMPLETE','contract':contract,'fits':[],'dynamic':None,'frequency_rows':[]};candidates={}
    for temperature in [300,1000]:
        limits.check();train=select_trajectory(data,temperature,723571);valid=select_trajectory(data,temperature,837213)
        n=len(np.load(Path(train['folder'])/'displacements_m.npy',mmap_mode='r'));frames=np.linspace(0,int(.8*n)-1,64,dtype=int);start=time.monotonic()
        with (folder/'hiphive.log').open('a') as log,contextlib.redirect_stdout(log):
            fcp,fit=build_fit(space,reference,Path(train['folder']),frames)
            selected,_,_=force_validation(fcp,reference,Path(train['folder']),np.linspace(int(.8*n),n-1,16,dtype=int))
            heldout,predicted,actual=force_validation(fcp,reference,Path(valid['folder']),np.linspace(0,n-1,80,dtype=int))
            fcp.write(str(folder/f'T{temperature}.fcp'));constraints=model_constraints(fcp,unit,reference,folder/f'T{temperature}.npz')
        candidates[temperature]=fcp;np.savez_compressed(folder/f'T{temperature}_heldout.npz',actual=actual,predicted=predicted)
        result['fits'].append({'temperature_k':temperature,**fit,**heldout,'model_selection_force_relative_RMS_error':selected['force_relative_RMS_error'],
            'constraints':constraints,'elapsed_seconds':time.monotonic()-start})
        json_write(folder/'checkpoint.json',result);print('quartic force',temperature,heldout['force_relative_RMS_error'],flush=True)
    tr=select_trajectory(data,1000,837213);source=Path(tr['folder']);atoms=reference.copy();positions0=atoms.positions.copy()
    atoms.positions+=np.load(source/'displacements_m.npy',mmap_mode='r')[0]*1e10
    atoms.set_velocities(np.load(source/'velocities_m_s.npy',mmap_mode='r')[0]*1e-5/units.fs)
    atoms.calc=ForceConstantCalculator(candidates[1000].get_force_constants(reference));integrator=VelocityVerlet(atoms,.5*units.fs,logfile=None)
    # Warm compilation then estimate the actual per-step cost. The physical
    # pilot prefix is retained in the final trajectory, never discarded.
    start=time.monotonic();initial_energy=atoms.get_total_energy()*constants.elementary_charge
    velocities=[];energies=[];temperatures=[];maximum_displacements=[];completed_steps=0;error=None
    nsteps=40000;stride=10;estimate=None
    for step in range(nsteps//stride):
        try:
            if step%100==0:limits.check()
            integrator.run(stride);completed_steps+=stride
            velocities.append(atoms.get_velocities()*units.fs*1e5);energies.append(atoms.get_total_energy()*constants.elementary_charge);temperatures.append(atoms.get_temperature())
            displacement=atoms.positions-positions0;displacement-=np.rint(displacement/np.diag(atoms.cell))*np.diag(atoms.cell)
            maximum_displacements.append(float(np.max(np.linalg.norm(displacement,axis=1))))
            if completed_steps==100:clock=time.monotonic()
            if completed_steps==200:
                estimate=(time.monotonic()-clock)/100*(nsteps-completed_steps)
                if estimate>limits.seconds-(time.monotonic()-limits.started)-10:nsteps=4000
        except (TimeoutError,MemoryError,ValueError) as exc:
            error=str(exc);break
        if completed_steps>=nsteps:break
    velocity=np.asarray(velocities);np.save(folder/'T1000_velocities_m_s.npy',velocity);np.save(folder/'T1000_energy_j.npy',np.asarray(energies));np.save(folder/'T1000_temperature_k.npy',np.asarray(temperatures));np.save(folder/'T1000_max_displacement_angstrom.npy',np.asarray(maximum_displacements))
    drift=float(np.max(abs(np.asarray(energies)-initial_energy))/(1.5*(len(atoms)-1)*constants.Boltzmann*np.mean(temperatures))) if energies else None
    duration_ps=completed_steps*.5/1000
    result['dynamic']={'completed_steps':completed_steps,'requested_steps':nsteps,'duration_ps':duration_ps,'error':error,
        'energy_drift_over_thermal_KE':drift,'mean_temperature_k':float(np.mean(temperatures)) if temperatures else None,
        'elapsed_seconds':time.monotonic()-start,'forecast_remaining20ps_seconds_at200steps':estimate,
        'maximum_displacement_angstrom':max(maximum_displacements) if maximum_displacements else None}
    if error is None and duration_ps>=19.999:
        harmonic=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
        for q in [[0.,0.,0.],[0.,.5,0.]]:
            _,vectors,groups=groups_at_q(harmonic,q)
            for group in groups:
                target=next(r for r in data['frequency_rows'] if r['temperature_k']==1000 and r['q_fractional']==q and r['group']==group)
                normal=project_normal_coordinates(velocity,positions0*1e-10,atoms.get_masses()*constants.atomic_mass,harmonic,q,vectors[:,group])
                e,power=periodogram_modes(normal,5e-15);fit=fit_windowed_mode(e,power,target['MD_center_mev'],duration_ps)
                np.savez_compressed(folder/f'T1000_q{q[1]}_g{group[0]}.npz',energy_mev=e,power=power)
                difference=fit['center_mev']-target['MD_center_mev']
                result['frequency_rows'].append({'q_fractional':q,'group':group,**fit,'MOD_center_mev':target['MD_center_mev'],
                    'difference_mev':difference,'original_frequency_gate_pass':abs(difference)<max(.5,2*target['MD_conservative_spread_mev'])})
    result['status']='FAIL_THERMAL_STABILITY' if error and 'Displacement' in error else 'COMPLETED_BOUNDED_QUARTIC_DIAGNOSTIC' if error is None else 'INCOMPLETE'
    result['combined_acceptance']='NOT_ACCEPTED_WITHOUT_FULL_ORIGINAL_FREQUENCY_SIZE_RANGE_VOLUME_ENSEMBLE_GATES'
    result['resource_costs']=limits.costs();result['source']=source_record();json_write(folder/'results.json',result)
    return result


def run_dependency_audit(output):
    base = ROOT/'benchmark_results/repair_20261102_v2'
    quality = json.loads((base/'MD_quality_status.json').read_text())
    affected = []
    # Historical defective trajectory references occur in textual JSON/source,
    # not inferred from mere directory names or numerical similarity.
    bad = json.loads((base/'B11/results.json').read_text())
    bad_paths = [r['folder'] for r in bad['trajectories']]
    for parent in [ROOT/'benchmark_results', ROOT/'benchmarks']:
        for path in parent.rglob('*.json'):
            if 'starting_source' in path.parts or path.stat().st_size > 4*1024**2:
                continue
            content = path.read_text(errors='replace')
            matches = [value for value in bad_paths if value in content]
            if matches:
                affected.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path), 'defective_generation_paths': matches,
                                 'status': 'HISTORICAL_REFERENCE_REQUIRES_REPAIRED_REPLACEMENT_OR_SCOPE_REVIEW'})
    result = {'status': 'AUDITED', 'quality_record': str(base/'MD_quality_status.json'),
              'quality_sha256': sha256(base/'MD_quality_status.json'), 'original_quality': quality,
              'explicit_downstream_references': affected,
              'fixed_harmonic_reference': 'unchanged, generated from independent 0K MOD central force constants',
              'preservation': 'no historical file modified or removed',
              'interpretation': 'a textual reference is a dependency candidate, not automatic proof that every stored quantity used defective frames'}
    json_write(output/'dependency_audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['restart','forces','linewidth','controls','dynamics','failure-controls','quartic','audit','all'], default='all')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    json_write(args.output/f'environment_{args.stage}.json', {'environment': environment(), 'source': source_record(),
              'thread_environment': {name: __import__('os').environ.get(name) for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']},
              'command': ' '.join(__import__('sys').argv)})
    for name, method in [('restart',run_restart),('forces',run_forces),('linewidth',run_linewidth),('controls',run_force_controls),('dynamics',run_force_dynamics),('failure-controls',run_failure_controls),('quartic',run_quartic),('audit',run_dependency_audit)]:
        if args.stage in ['all', name]:
            result = method(args.output)
            print(name, result.get('status'), 'cumulative seconds', time.monotonic()-started, flush=True)


if __name__ == '__main__':
    main()
