"""Independent mode correlations, fit convergence and linewidth audit for B11."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import constants
from scipy.optimize import linear_sum_assignment

from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.mode_spectra import (PLANCK_MEV_PS, fit_windowed_mode,
    hann_fwhm_mev, periodogram_modes, project_normal_coordinates)
from .workflow_io import ROOT, environment, json_write, sha256


def groups_at_q(model, q):
    omega, vectors = model.modes(q)
    energies = omega*constants.hbar/(1e-3*constants.elementary_charge)
    physical = np.ones(len(energies), bool)
    if np.linalg.norm(q) == 0:
        t = np.tile(np.eye(3), (len(model.positions_m), 1))*np.repeat(np.sqrt(model.masses_kg), 3)[:, None]
        t /= np.linalg.norm(t, axis=0)
        overlap = np.sum(abs(vectors.conj().T @ t)**2, axis=-1)
        physical[np.argsort(overlap)[-3:]] = False
    groups = []
    for index in np.flatnonzero(physical):
        if not groups or abs(energies[index]-energies[groups[-1][0]]) > .02:
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))
    return energies, vectors, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='benchmark_results/anharmonic_validation_20261102_v1')
    args = parser.parse_args()
    output = ROOT/args.output
    output.mkdir(parents=True, exist_ok=True)
    source_path = ROOT/'benchmark_results/anharmonic_20261102_v1/results.json'
    source = json.loads(source_path.read_text())
    from ase.io import read
    unit = read(ROOT/'benchmark_results/si_lattice_dynamics/equilibrium.extxyz')
    material = json.loads((ROOT/'benchmark_results/si_lattice_dynamics/manifest.json').read_text())
    reference_path = ROOT/'benchmark_results/si_lattice_dynamics'/material['artifacts']['force_constants']['path']
    reference = RealSpaceForceConstants.load(reference_path)
    selected = {temperature: next(row for row in source['fits'] if row['temperature_k'] == temperature and row['supercell'] == [4, 6, 6] and row['count'] == 64 and row['cutoff_angstrom'] == 5 and row['ridge'] == 0) for temperature in (300, 1000)}
    models = {temperature: RealSpaceForceConstants.load(row['model_file']) for temperature, row in selected.items()}
    results = {'benchmark': 'Benchmark 11A/11B independent validation', 'status': 'INCOMPLETE',
        'parameters': source['parameters'], 'input_results_sha256': sha256(source_path), 'environment': environment(),
        'force_validation': selected, 'frequency_rows': [], 'MD_mode_fits': [], 'convergence': [],
        'linewidth_convention': 'energy FWHM Gamma=2 hbar gamma, amplitude decay time 1/gamma=2 hbar/Gamma, population decay time=hbar/Gamma; no lifetime assigned when unresolved',
        'volume_policy': source['volume_policy'], 'nuclear_treatment': 'classical MOD NVE trajectories; not exact quantum anharmonic dynamics and not silently equated to bead-labelled TACAW dynamics'}
    started = time.monotonic()
    for temperature in (300, 1000):
        model = models[temperature]
        for q_index, q in enumerate((np.array([0., 0., 0.]), np.array([0., .5, 0.]))):
            reference_e, reference_v, groups = groups_at_q(reference, q)
            omega, vectors = model.modes(q)
            model_e = omega*constants.hbar/(1e-3*constants.elementary_charge)
            _, assignment = linear_sum_assignment(-abs(reference_v.conj().T @ vectors)**2)
            for group_index, group in enumerate(groups):
                matched = assignment[group]
                predicted = float(np.mean(model_e[matched]))
                ref_energy = float(np.mean(reference_e[group]))
                if predicted <= 0:
                    raise ValueError('unstable physical effective mode')
                projector_error = float(np.linalg.norm(reference_v[:, group] @ reference_v[:, group].conj().T-vectors[:, matched] @ vectors[:, matched].conj().T))
                fits = []
                for trajectory_index, trajectory in enumerate(source['dynamics']):
                    if trajectory['settings']['temperature_k'] != temperature:
                        continue
                    repetitions = trajectory['settings']['repetitions']
                    atoms = unit.repeat(tuple(repetitions))
                    folder = Path(trajectory['folder'])
                    cadence = trajectory['state']['sample_interval_s']
                    velocity = np.load(folder/'velocities_m_s.npy', mmap_mode='r')
                    projected = project_normal_coordinates(velocity, atoms.positions*1e-10, atoms.get_masses()*constants.atomic_mass, model, q, vectors[:, matched])
                    for block, (lo, hi) in enumerate(((0, len(projected)), (0, len(projected)//2), (len(projected)//2, len(projected)))):
                        energy, power = periodogram_modes(projected[lo:hi], cadence)
                        duration = (hi-lo)*cadence*1e12
                        fit = fit_windowed_mode(energy, power, predicted, duration)
                        fit.update(temperature_k=temperature, q_fractional=q.tolist(), group=group, reference_energy_mev=ref_energy,
                                   effective_energy_mev=predicted, trajectory_index=trajectory_index, repetitions=repetitions,
                                   seed=trajectory['settings']['seed'], timestep_fs=trajectory['settings']['timestep_fs'],
                                   actual_mean_temperature_k=trajectory['state']['mean_temperature_k'], block=block,
                                   Nyquist_energy_mev=constants.h/(2*cadence)/(1e-3*constants.elementary_charge))
                        fits.append(fit)
                        results['MD_mode_fits'].append(fit)
                        if block == 0:
                            np.savez_compressed(output/f'PSD_T{temperature}_q{q_index}_g{group_index}_traj{trajectory_index}.npz', energy_mev=energy, velocity_PSD=power,
                                                projected_velocities=projected, group=np.array(group))
                # Independent seeds at the selected large cell define the main
                # comparison. Half segments and other settings quantify spread,
                # without pretending correlated segments are independent runs.
                main = [r for r in fits if r['repetitions'] == [4, 6, 6] and r['block'] == 0]
                centers = np.array([r['center_mev'] for r in main])
                center = float(np.mean(centers))
                blocks = [r['center_mev'] for r in fits if r['repetitions'] == [4, 6, 6] and r['block'] != 0]
                spread = max(float(np.max(abs(np.array(blocks)-center))), max(r['energy_bin_mev']/2 for r in main))
                widths = np.array([r['fitted_intrinsic_FWHM_mev'] for r in main])
                width = float(np.mean(widths))
                width_spread = max(abs(r['fitted_intrinsic_FWHM_mev']-width) for r in fits if r['repetitions'] == [4, 6, 6])
                window = max(r['Hann_power_window_FWHM_mev'] for r in main)
                linewidth_resolved = width > 2*window and width_spread < .5*width and all(r['fit_relative_RMS_residual'] < .1 for r in main)
                row = {'temperature_k': temperature, 'q_fractional': q.tolist(), 'reference_mode_group': group, 'matched_effective_group': matched.tolist(),
                    'reference_energy_mev': ref_energy, 'effective_energy_mev': predicted, 'MD_center_mev': center,
                    'MD_center_spread_mev': spread, 'effective_minus_MD_mev': predicted-center,
                    'frequency_validation_passed': abs(predicted-center) < max(.5, 2*spread),
                    'temperature_shift_mev': predicted-ref_energy, 'shift_exceeds_MD_resolution_and_spread': abs(predicted-ref_energy) > 2*spread,
                    'mode_projector_difference': projector_error, 'fitted_linewidth_FWHM_mev': width,
                    'linewidth_spread_mev': width_spread, 'window_FWHM_mev': window,
                    'linewidth_status': 'RESOLVED_WITHIN_CLASSICAL_MD_MODEL' if linewidth_resolved else 'NOT_RESOLVED',
                    'population_lifetime_ps': float(constants.hbar/(width*1e-3*constants.elementary_charge)*1e12) if linewidth_resolved else None,
                    'unresolved_resolution_indicator_mev': max(2*window, width+width_spread) if not linewidth_resolved else None}
                results['frequency_rows'].append(row)
                print('mode', temperature, q.tolist(), group, 'EHM', predicted, 'MD', center, 'spread', spread, 'linewidth', row['linewidth_status'], flush=True)
                json_write(output/'checkpoint.json', results)
    results['force_target_passed'] = all(r['force_relative_RMS_error'] < source['parameters']['heldout_force_relative_RMS_tolerance'] for r in selected.values())
    results['frequency_validation_status'] = 'PASS' if all(r['frequency_validation_passed'] for r in results['frequency_rows']) else 'INCOMPLETE'
    results['frequency_shift_status'] = 'INCOMPLETE: fit/sampling convergence and instantaneous-force fidelity remain separate gates'
    results['linewidth_status'] = 'INCOMPLETE: resolution and setting-dependent mode statuses are reported individually'
    results['EELS_status'] = 'NOT_ASSESSED: only validated effective modes may feed a separately labelled EELS prediction'
    results['elapsed_seconds'] = time.monotonic()-started
    json_write(output/'results.json', results)
    write_outputs(results, output)


def write_outputs(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for temperature in (300, 1000):
        rows = [r for r in results['frequency_rows'] if r['temperature_k'] == temperature]
        axes[0].errorbar([r['effective_energy_mev'] for r in rows], [r['MD_center_mev'] for r in rows],
                         yerr=[r['MD_center_spread_mev'] for r in rows], fmt='.', label=f'{temperature} K target')
        axes[1].plot(range(len(rows)), [r['temperature_shift_mev'] for r in rows], '.-', label=f'{temperature} K target')
    axes[0].plot([0, 80], [0, 80], 'k--')
    axes[0].set(xlabel='effective harmonic energy (meV)', ylabel='independent mode-velocity spectrum center (meV)')
    axes[1].set(xlabel='matched mode-group index', ylabel='fixed-volume shift from 0-K model (meV)')
    for ax in axes:
        ax.legend()
    fig.savefig(output/'mode_energy_comparison.png', dpi=160)
    plt.close(fig)
    text = ['# Benchmark 11: fixed-volume effective modes and intrinsic-linewidth audit', '',
        f"Frequency comparison: **{results['frequency_validation_status']}**. Overall frequency-shift milestone remains **INCOMPLETE** pending all convergence/fidelity gates.", '',
        'Classical trajectories use the exact supplied MOD file, separate random seeds for training and held-out validation, 192/576 atoms, 1 fs timestep, 5 fs sampling, 5 ps NVT equilibration then 20 ps NVE production. Independent half-timestep controls have the same physical duration/cadence. All actual mean temperatures, energy drifts and raw SI displacement/force/velocity arrays are retained. The volume is the validated 0-K equilibrium volume; thermal expansion is not included.', '',
        'Effective second-order fits use actual finite-temperature forces, hiPhive symmetry and acoustic constraints. Training-size, cutoff and ridge variations are recorded. Held-out errors do not disappear with more data: the instantaneous-force fidelity target is assessed separately from approximate mode-frequency agreement. No nonlinear forces were removed from the targets.', '',
        'Mode assignment uses eigenvector overlaps and compares degenerate subspaces. Independent mass-weighted normal velocities use the atomic gauge exp(+iq.R)/sqrt(Ncells). Positive-frequency PSD peaks are compared with mass-weighted dynamical-matrix energies. Correlated half segments provide a conservative spread and are not counted as independent trajectories.', '',
        'The 20 ps Hann power-window FWHM is about 0.298 meV, with bin spacing h/T=0.207 meV; 10 ps half segments are twice as broad. The 5 fs sampling Nyquist energy is 413.57 meV. A Lorentzian is forward-convolved with the known Hann window for line fitting. A numerical sub-window fitted width alone is never a resolved intrinsic linewidth. Resolution, replica spread, timestep, size and mode mixing must all be assessed.', '',
        'Energy FWHM Gamma corresponds to amplitude decay time 2 hbar/Gamma and population decay time hbar/Gamma. Unresolved modes have no precise lifetime. No EELS peak width or instrumental Gaussian is used as a phonon lifetime.', '',
        '| Target K | q | Mode group | Effective meV | MD meV | MD spread meV | Width status |', '|---:|---|---|---:|---:|---:|---|']
    for r in results['frequency_rows']:
        text.append(f"| {r['temperature_k']} | {r['q_fractional']} | {r['reference_mode_group']} | {r['effective_energy_mev']:.4f} | {r['MD_center_mev']:.4f} | {r['MD_center_spread_mev']:.4f} | {r['linewidth_status']} |")
    text += ['', '```bash', 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.anharmonic_modes',
             'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.anharmonic_validation', '```', '',
             '![mode energies](mode_energy_comparison.png)', '',
             'Sources: [hiPhive effective harmonic models](https://hiphive.materialsmodeling.org/advanced_topics/effective_harmonic_models.html), [LAMMPS modified Tersoff](https://docs.lammps.org/pair_tersoff_mod.html). Agreement establishes behavior within this empirical model, not independent validation against experiment or DFT.']
    (output/'report.md').write_text('\n'.join(text)+'\n')


if __name__ == '__main__':
    main()
