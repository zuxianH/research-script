"""Reduced EHM EELS using only independently accepted Gamma mode subspaces."""

import json

import numpy as np
from scipy import constants

from yoshioka.electron_phonon import LobatoSilicon, common_diagonal_covariance, full_space_born, transverse_modes
from yoshioka.free_propagation import TransverseGrid
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from .eels_observables import absolute_spectrum, detector_masks
from .workflow_io import ROOT, json_write


def main():
    output = ROOT/'benchmark_results/validated_mode_eels_20261102_v1'
    output.mkdir(parents=True, exist_ok=True)
    validation = json.loads((ROOT/'benchmark_results/anharmonic_validation_20261102_v1/results.json').read_text())
    results = {'status': 'INCOMPLETE', 'reduced_prediction_status': 'PASS', 'scope':
        'Only independently accepted Gamma subspaces, embedded with 1/sqrt(8) periodic-bath weight in a fixed 32-atom slab. This is a reduced effective-harmonic prediction, not full material EELS. Unvalidated mode groups and all other q sectors are omitted, with no renormalization.',
        'thermal_approximation': 'classical MD force-fitted frequencies with Bose harmonic occupations; not exact quantum anharmonic dynamics',
        'linewidths': 'none: no imaginary energies, no fitted MD damping or self-energy added; only normalized 2 meV display Gaussian',
        'double_counting': 'covariance built only from the selected reduced model, included once; no separate Debye-Waller multiplier; no additional frequency shift', 'cases': []}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for temperature in (300, 1000):
        model = RealSpaceForceConstants.load(validation['force_validation'][str(temperature)]['model_file'])
        accepted = [r for r in validation['frequency_rows'] if r['temperature_k'] == temperature
            and r['q_fractional'] == [0, 0, 0] and r['frequency_validation_passed']]
        indices = sorted({i for r in accepted for i in r['matched_effective_group']})
        omega, eigenvectors = model.modes([0, 0, 0])
        omega, eigenvectors = omega[indices], eigenvectors[:, indices]
        cells = np.array(list(np.ndindex(2, 2, 2)))
        positions = ((cells @ model.cell_m)[:, None]+model.positions_m[None]).reshape(-1, 3)
        profile = eigenvectors.reshape(4, 3, -1)*np.sqrt(constants.hbar/(2*model.masses_kg[:, None, None]*omega[None, None]))/np.sqrt(8)
        if np.linalg.norm(profile.imag) > 1e-12*np.linalg.norm(profile):
            raise ValueError('Gamma reduced mode basis must be real')
        displacement = np.tile(profile.real, (8, 1, 1))
        nbar = 1/np.expm1(constants.hbar*omega/(constants.Boltzmann*temperature))
        covariance, diagnostic = common_diagonal_covariance(displacement*np.sqrt(2*nbar+1)[None, None])
        grid = TransverseGrid(2*model.cell_m[0, 0], 2*model.cell_m[1, 1], 128, 128)
        qxy, _, angles = transverse_modes(grid, 60000*constants.elementary_charge, 20.)
        positions[:, 2] += 12e-10
        length = 2*model.cell_m[2, 2]+24e-10
        loss = full_space_born(LobatoSilicon.from_abtem_table(), positions, displacement, covariance,
            qxy, omega, 60000*constants.elementary_charge, grid.lx_m*grid.ly_m, length)*np.sqrt(nbar+1)[:, None]
        gain = full_space_born(LobatoSilicon.from_abtem_table(), positions, displacement, covariance,
            qxy, -omega, 60000*constants.elementary_charge, grid.lx_m*grid.ly_m, length)*np.sqrt(nbar)[:, None]
        weights = abs(np.concatenate((loss, gain)))**2
        transfers = constants.hbar*np.r_[omega, -omega]
        axis = np.arange(-120, 120.025, .05)
        spectra = {name: absolute_spectrum(transfers, np.sum(weights[:, mask], axis=1), axis)
            for name, mask in detector_masks(angles).items()}
        artifact = output/f'validated_Gamma_T{temperature}.npz'
        np.savez_compressed(artifact, energy_grid_mev=axis, transfers_j=transfers, probabilities=weights,
            qxy_m_inv=qxy, selected_effective_modes=indices, **{'spectrum_'+name:values for name, values in spectra.items()})
        results['cases'].append({'temperature_k': temperature, 'accepted_mode_groups': [r['matched_effective_group'] for r in accepted],
            'selected_modes': indices, 'accepted_frequency_errors_mev': [r['effective_minus_MD_mev'] for r in accepted],
            'retained_reduced_model_inelastic_weight_20mrad': float(np.sum(weights)), 'covariance': diagnostic,
            'artifact': str(artifact), 'full_material_omitted_weight': 'not bounded by this reduced calculation'})
        ax.plot(axis, spectra['full_0_20_mrad'], label=f'{temperature} K, {len(indices)} accepted Gamma modes')
    ax.set(xlim=(-90, 90), xlabel='electron energy loss (meV)', ylabel='absolute reduced-model probability / meV')
    ax.legend()
    fig.savefig(output/'validated_reduced_Gamma_spectra.png', dpi=160)
    plt.close(fig)
    json_write(output/'results.json', results)


if __name__ == '__main__':
    main()
