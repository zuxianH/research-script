"""B11 fit-uncertainty audit and explicitly unaccepted effective-harmonic EELS."""

import contextlib
import json
from pathlib import Path

import numpy as np
from scipy import constants
from scipy.optimize import linear_sum_assignment

from yoshioka.electron_phonon import LobatoSilicon, common_diagonal_covariance, full_space_born, transverse_modes
from yoshioka.free_propagation import TransverseGrid
from yoshioka.lattice_dynamics import RealSpaceForceConstants, finite_periodic_modes
from .anharmonic_modes import fit_effective
from .eels_observables import DETECTORS, absolute_spectrum, detector_masks
from .workflow_io import ROOT, json_write, sha256


def aligned_energy_difference(reference, candidate, q):
    omega, vectors = reference.modes(q)
    other, e_other = candidate.modes(q)
    _, assigned = linear_sum_assignment(-abs(vectors.conj().T @ e_other)**2)
    return (other[assigned]-omega)*constants.hbar/(1e-3*constants.elementary_charge)


def stability(model, mesh=12):
    q = np.array(list(np.ndindex(mesh, mesh, mesh)), dtype=float)/mesh
    matrix = model.dynamical(q)
    hermitian = np.linalg.norm(matrix-matrix.swapaxes(-1, -2).conj(), axis=(-2, -1))/np.linalg.norm(matrix, axis=(-2, -1))
    omega, vectors = np.linalg.eigh(matrix)
    translations = np.tile(np.eye(3), (4, 1))*np.repeat(np.sqrt(model.masses_kg), 3)[:, None]
    translations /= np.linalg.norm(translations, axis=0)
    overlaps = np.sum(abs(vectors[0].conj().T @ translations)**2, axis=-1)
    physical = np.ones(omega.shape, bool)
    physical[0, np.argsort(overlaps)[-3:]] = False
    return {'mesh': mesh, 'max_hermiticity_residual': float(np.max(hermitian)),
        'minimum_physical_eigenvalue_s_inv2': float(np.min(omega[physical])),
        'removed_Gamma_translations': 3, 'stable': bool(np.all(omega[physical] > 0)),
        'acoustic_translation_residual': float(np.linalg.norm(matrix[0] @ translations)/np.linalg.norm(matrix[0]))}


def candidate_spectrum(model, temperature, output, name):
    modes = finite_periodic_modes(model, (2, 2, 2))
    omega = modes['omega_rad_s']
    nbar = 1/np.expm1(constants.hbar*omega/(constants.Boltzmann*temperature))
    covariance, diagnostic = common_diagonal_covariance(modes['displacements_m']*np.sqrt(2*nbar+1)[None, None])
    grid = TransverseGrid(2*model.cell_m[0, 0], 2*model.cell_m[1, 1], 128, 128)
    energy = 60000*constants.elementary_charge
    qxy, _, angles = transverse_modes(grid, energy, 20.)
    position = modes['positions_m'].copy()
    position[:, 2] += 12e-10
    length = 2*model.cell_m[2, 2]+24e-10
    loss = full_space_born(LobatoSilicon.from_abtem_table(), position, modes['displacements_m'], covariance,
        qxy, omega, energy, grid.lx_m*grid.ly_m, length)*np.sqrt(nbar+1)[:, None]
    gain = full_space_born(LobatoSilicon.from_abtem_table(), position, modes['displacements_m'], covariance,
        qxy, -omega, energy, grid.lx_m*grid.ly_m, length)*np.sqrt(nbar)[:, None]
    probabilities = abs(np.concatenate((loss, gain)))**2
    transfers = np.r_[omega, -omega]*constants.hbar
    axis = np.arange(-120, 120.025, .05)
    spectra = {detector: absolute_spectrum(transfers, np.sum(probabilities[:, mask], axis=1), axis)
               for detector, mask in detector_masks(angles).items()}
    path = output/(name+'.npz')
    np.savez_compressed(path, energy_grid_mev=axis, **{f'spectrum_{key}': values for key, values in spectra.items()},
                        probabilities=probabilities, transfers_j=transfers, qxy_m_inv=qxy)
    return {'temperature_k': temperature, 'artifact': str(path), 'mode_count': len(omega), 'covariance': diagnostic,
        'total_weight_within_20_mrad': float(np.sum(probabilities)),
        'scope': 'finite 32-atom phonon bath, first Born, fixed volume, Bose occupations; candidate full EHM model is not accepted when independent frequency gates fail',
        'intrinsic_linewidths_used': False, 'instrumental_Gaussian_FWHM_mev': 2,
        'quantum_status': 'classical-force-fitted effective frequencies with quantum harmonic occupations is a stated approximation, not exact quantum anharmonic dynamics'}


def main():
    output = ROOT/'benchmark_results/anharmonic_followup_20261102_v1'
    output.mkdir(parents=True, exist_ok=True)
    source_path = ROOT/'benchmark_results/anharmonic_20261102_v1/results.json'
    source = json.loads(source_path.read_text())
    validation_path = ROOT/'benchmark_results/anharmonic_validation_20261102_v1/results.json'
    validation = json.loads(validation_path.read_text())
    from ase.io import read
    from hiphive import ClusterSpace
    from yoshioka.lattice_dynamics import real_space_from_supercell
    unit = read(ROOT/'benchmark_results/si_lattice_dynamics/equilibrium.extxyz')
    selected = validation['force_validation']
    result = {'benchmark': 'Benchmark 11 convergence and candidate EELS', 'status': 'INCOMPLETE',
        'source_results_sha256': sha256(source_path), 'independent_validation_sha256': sha256(validation_path),
        'fit_variations': [], 'blocked_fit_uncertainty': [], 'stability': {}, 'EELS_candidates': [],
        'interpretation': 'frequency and force-fidelity failures remain; no candidate is silently promoted to a validated anharmonic Hamiltonian'}
    with (output/'hiphive_blocks.log').open('w') as log, contextlib.redirect_stdout(log):
        cs = ClusterSpace(unit, [5.])
    for temperature in (300, 1000):
        primary_row = selected[str(temperature)]
        primary = RealSpaceForceConstants.load(primary_row['model_file'])
        result['stability'][str(temperature)] = stability(primary)
        for row in source['fits']:
            if row['temperature_k'] != temperature:
                continue
            candidate = RealSpaceForceConstants.load(row['model_file'])
            differences = []
            for q in ([0, 0, 0], [0, .5, 0], [.17, .23, .11]):
                delta = aligned_energy_difference(primary, candidate, q)
                differences.extend(delta[3:] if q == [0, 0, 0] else delta)
            result['fit_variations'].append({'temperature_k': temperature, 'supercell': row['supercell'],
                'training_frames': row['count'], 'cutoff_angstrom': row['cutoff_angstrom'], 'ridge': row['ridge'],
                'max_matched_frequency_change_mev': float(np.max(np.abs(differences))),
                'training_force_error': row['training_force_relative_RMS_error'], 'heldout_force_error': row['force_relative_RMS_error'],
                'size_temperature_confound': 'small/large NVE trajectories have different recorded actual temperatures; differences are combined finite-size/temperature sensitivity, not a pure equal-temperature size error' if row['supercell'] != [4, 6, 6] else None})
        trajectory = next(d for d in source['dynamics'] if d['settings']['temperature_k'] == temperature and d['settings']['repetitions'] == [4, 6, 6] and d['settings']['seed'] == 932731)
        reference = unit.repeat((4, 6, 6))
        block_models = []
        for block in range(4):
            indices = np.linspace(block*1000, (block+1)*1000-1, 16, dtype=int)
            with (output/'hiphive_blocks.log').open('a') as log, contextlib.redirect_stdout(log):
                fcp, fitted = fit_effective(cs, reference, Path(trajectory['folder']), 16, 0., indices)
                fc = fcp.get_force_constants(reference).get_fc_array(2)
                model, _ = real_space_from_supercell(unit, (4, 6, 6), fc)
            block_models.append(model)
            model.save(output/f'block_T{temperature}_{block}.npz')
        for row in validation['frequency_rows']:
            if row['temperature_k'] != temperature:
                continue
            shifts = []
            for model in block_models:
                delta = aligned_energy_difference(primary, model, row['q_fractional'])
                shifts.append(float(np.mean(delta[row['matched_effective_group']])))
            conservative_fit_spread = max(abs(value) for value in shifts)
            result['blocked_fit_uncertainty'].append({'temperature_k': temperature, 'q': row['q_fractional'],
                'mode_group': row['matched_effective_group'], 'block_frequency_deviations_mev': shifts,
                'conservative_fit_spread_mev': conservative_fit_spread,
                'MD_center_spread_mev': row['MD_center_spread_mev'],
                'temperature_shift_mev': row['temperature_shift_mev'],
                'shift_exceeds_combined_spread': abs(row['temperature_shift_mev']) > 2*(conservative_fit_spread+row['MD_center_spread_mev']),
                'statistical_scope': 'four contiguous 5 ps training blocks; range is a sensitivity estimate, not independent-sample standard error'})
        result['EELS_candidates'].append(candidate_spectrum(primary, temperature, output, f'effective_candidate_T{temperature}'))
        zero = RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
        result['EELS_candidates'].append(candidate_spectrum(zero, temperature, output, f'fixed_harmonic_T{temperature}'))
    result['frequency_shift_status'] = 'INCOMPLETE: selected modes fail independent MD frequency targets; fit and MD spreads explicitly quantified'
    result['linewidth_status'] = 'NOT_RESOLVED: no finite lifetime inserted in EELS'
    result['effective_EELS_status'] = 'INCOMPLETE: diagnostic candidates computed; full effective-mode model has not passed frequency fidelity'
    result['failure_investigation'] = 'held-out errors closely track training errors and saturate with training size/cutoff/ridge; the tested harmonic model class has appreciable anharmonic force residual. Static force-fit effective frequencies also differ from selected dynamical spectral centers; no unique microscopic cause is asserted without a self-energy analysis.'
    json_write(output/'results.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, temperature in zip(axes, (300, 1000)):
        for kind, label in (('fixed_harmonic', 'fixed 0-K force constants + thermal occupation'), ('effective_candidate', 'candidate effective harmonic model')):
            with np.load(output/f'{kind}_T{temperature}.npz') as d:
                ax.plot(d['energy_grid_mev'], d['spectrum_full_0_20_mrad'], label=label)
        ax.set(title=f'{temperature} K, candidate comparison (not accepted)', xlim=(-100, 100), xlabel='electron energy loss (meV)', ylabel='absolute first-Born inelastic probability / meV')
        ax.legend(fontsize=7)
    fig.savefig(output/'candidate_effective_EELS.png', dpi=160)
    plt.close(fig)
    text = ['# Benchmark 11: convergence audit and effective-harmonic EELS candidates', '',
        '**Frequency-shift validation: INCOMPLETE. Intrinsic linewidths: NOT_RESOLVED. Effective-harmonic EELS: diagnostic candidates only.**', '',
        'The 36 fit variations and four additional blocked fits per temperature quantify training-size, cutoff, regularization and finite-size/temperature sensitivity. Independent trajectories provide mode spectra. Actual NVE temperatures vary with seed and size; the finite-size comparison is explicitly confounded by those recorded temperatures and is not presented as a pure matched-temperature extrapolation.', '',
        'The larger-cell held-out force errors are 14.8249% at 300 K and 28.8903% at 1000 K, against the predeclared 15% target. Training and validation errors track each other and barely change with more frames or longer cutoffs. This is evidence against overfitting as the dominant limitation. It does not make the harmonic force approximation exact.', '',
        'Independent dynamical mode centers fail the frequency target for one of ten tested groups at 300 K (largest discrepancy 0.744 meV) and four of ten at 1000 K (largest 2.672 meV). The predeclared tolerance is not relaxed. A static effective force fit is not asserted to equal a frequency-dependent anharmonic self-energy. The precise origin of each dynamical discrepancy remains unresolved.', '',
        'Candidate EELS spectra are computed for a declared finite 32-atom phonon bath in first Born, using fixed-volume temperature-dependent force constants and Bose occupations. They are compared with fixed 0-K force constants at the same occupation temperature. Since the full effective-mode models fail selected validation gates, these curves are explicitly unaccepted diagnostics. They are not passed off as validated anharmonic EELS.', '',
        'No intrinsic linewidth or imaginary channel energy is added. The 2 meV Gaussian is an explicit plotting resolution, not a phonon lifetime. All unbroadened signed mode weights and absolute normalization are preserved. Classical force-fitted frequencies plus quantum harmonic occupations are an approximation and are kept distinct from the bead-labelled TACAW nuclear treatment.', '',
        '```bash', 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.anharmonic_followup', '```', '',
        '![candidate spectra](candidate_effective_EELS.png)']
    (output/'report.md').write_text('\n'.join(text)+'\n')


if __name__ == '__main__':
    main()
