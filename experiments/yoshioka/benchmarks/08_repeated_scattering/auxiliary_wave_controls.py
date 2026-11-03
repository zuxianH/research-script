"""Controlled auxiliary-wave, channel-phase and classical/Kubo comparisons."""

import json
from pathlib import Path

import numpy as np
from scipy import constants
from scipy.linalg import expm
from scipy.special import ive, roots_hermite

from yoshioka.coupled_channels import SplitChannelPropagator, fourier_populations
from yoshioka.elastic_propagation import electron_speed
from yoshioka.free_propagation import electron_wavenumber
from yoshioka.harmonic_oscillator import normalized_hermite_polynomials
from yoshioka.spectral_data import kubo_factor
from .repeated_scattering import model
from .thermal_reference import exact_weights
from .workflow_io import ROOT, json_write, sha256


def coordinate_auxiliary(config, order, p=12, n=4, thickness_nm=80):
    basis, grid, quanta, segments, initial = model(config, p, n, thickness_nm)
    nodes, weights = roots_hermite(order)
    x1, x2 = np.meshgrid(nodes, nodes, indexing='ij')
    nuclear = np.column_stack((x1.ravel(), x2.ravel()))
    quadrature = (weights[:, None]*weights[None]/np.pi).ravel()
    xx, yy = np.meshgrid(np.arange(n)*grid.dx_m, np.arange(n)*grid.dy_m)
    fx, fy = np.meshgrid(np.fft.fftfreq(n, grid.dx_m), np.fft.fftfreq(n, grid.dy_m))
    electron_energy = config['electron_energy_kev']*1000*constants.elementary_charge
    fourier = np.exp(2j*np.pi*(xx.ravel()[:, None]*fx.ravel()[None]+yy.ravel()[:, None]*fy.ravel()[None]))/n
    dispersion = (2*np.pi)**2*(fx.ravel()**2+fy.ravel()**2)/(2*electron_wavenumber(electron_energy))
    kinetic = (fourier*dispersion[None]) @ fourier.conj().T
    current = np.ones((len(nuclear), n*n), dtype=np.complex128)/np.sqrt(grid.lx_m*grid.ly_m)
    for region, segment in enumerate(segments):
        if region in (0, 2, 4):
            current = current @ expm(-1j*kinetic*segment.length_m).T
            continue
        interaction = 0 if region == 1 else 1
        if interaction == 0:
            phase = 2*np.pi*np.arange(n)[None, :]/n+np.zeros((n, 1))
        else:
            xi, yi = np.meshgrid(np.arange(n), np.arange(n))
            phase = 2*np.pi*(xi+yi)/n+.7
        shift = np.sqrt(2)*(nuclear @ np.asarray(config['displacement_couplings_xi'][interaction]))
        field = config['region_potential_ev'][interaction]*constants.elementary_charge*np.cos(phase.ravel()[None]-shift[:, None])
        matrices = np.broadcast_to(kinetic, (len(nuclear), n*n, n*n)).copy()
        diagonal = np.arange(n*n)
        matrices[:, diagonal, diagonal] += field/(constants.hbar*electron_speed(electron_energy))
        values, vectors = np.linalg.eigh(matrices)
        coefficients = np.einsum('qji,qj->qi', vectors.conj(), current)
        current = np.einsum('qij,qj->qi', vectors, coefficients*np.exp(-1j*values*segment.length_m))
    hermites = normalized_hermite_polynomials(nodes, p)
    first = np.repeat(hermites, order, axis=1)
    second = np.tile(hermites, (1, order))
    nuclear_projection = first[basis[:, 0]]*second[basis[:, 1]]*quadrature[None]
    wave = (nuclear_projection @ current).reshape(len(basis), n, n)
    return wave, (basis, grid, quanta, segments, initial)


def main():
    output = ROOT/'benchmark_results/auxiliary_controls_20261102_v2'
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/'benchmarks/configs/repeated_20261102_v1.json').read_text())
    result = {'benchmark': 'controlled TACAW auxiliary-wave and Kubo comparison', 'physical_cutoff': 24, 'quadrature': [], 'Kubo_controls': [],
        'previous_underconverged_control': 'v1 p=12 had complex boundary-state error 1.5579e-5 despite probability L1 3.6342e-8; its saved result is retained',
        'scope': 'synthetic spatial two-mode vacuum scattering and separate illustrative harmonic oscillator; not supplied material-data validation'}
    previous = None
    for order in (24, 32, 48):
        auxiliary, (basis, grid, quanta, segments, initial) = coordinate_auxiliary(config, order, p=24)
        result['quadrature'].append({'nodes_per_mode': order, 'complex_change': None if previous is None else float(np.linalg.norm(auxiliary-previous)/np.linalg.norm(initial))})
        previous = auxiliary
    energy = config['electron_energy_kev']*1000*constants.elementary_charge
    frozen, frozen_metrics = SplitChannelPropagator(grid, energy, np.zeros(len(basis))).propagate(initial, segments, 64)
    full, full_metrics = SplitChannelPropagator(grid, energy, basis@quanta).propagate(initial, segments, 64)
    auxiliary_probability = fourier_populations(auxiliary, grid)
    frozen_probability = fourier_populations(frozen, grid)
    full_probability = fourier_populations(full, grid)
    result['auxiliary_vs_zero_channel_phase'] = {'complex_relative_error': float(np.linalg.norm(auxiliary-frozen)/np.linalg.norm(initial)),
        'probability_L1': float(np.sum(abs(auxiliary_probability-frozen_probability))), 'propagation': frozen_metrics}
    result['finite_channel_energy_effect'] = {'complex_relative_difference': float(np.linalg.norm(auxiliary-full)/np.linalg.norm(initial)),
        'probability_L1': float(np.sum(abs(auxiliary_probability-full_probability))), 'propagation': full_metrics,
        'interpretation': 'known physical approximation difference; no arbitrary global or channel phase alignment'}
    np.savez_compressed(output/'spatial_auxiliary_comparison.npz', auxiliary_wave=auxiliary, full_channel_wave=full,
        zero_channel_phase_wave=frozen, basis=basis, mode_quanta_j=quanta)
    ell = np.arange(-50, 51)
    for temperature in (300, 1000):
        quantum_mev = 60.
        quantum = quantum_mev*1e-3*constants.elementary_charge
        x = quantum/(constants.Boltzmann*temperature)
        classical_n = 1/x
        quantum_n = 1/np.expm1(x)
        corrected_linear = classical_n*kubo_factor(np.array([-quantum_mev, quantum_mev]), temperature)
        target = np.array([quantum_n, quantum_n+1])
        result['Kubo_controls'].append({'temperature_k': temperature, 'operator': 'linear displacement',
            'maximum_weight_error': float(np.max(abs(corrected_linear-target))),
            'interpretation': 'classical harmonic linear-position correlation equals the quantum Kubo correlation; correction is exact for this control'})
        for eta in (.01, .1, .5):
            classical_argument = 2*eta*classical_n
            # Symmetric Fourier coefficients of exp[-2 eta n_cl(1-cos wt)].
            classical = ive(abs(ell), classical_argument)
            corrected = classical*kubo_factor(ell*quantum_mev, temperature)
            exact = exact_weights(ell, eta, quantum, temperature)
            result['Kubo_controls'].append({'temperature_k': temperature, 'eta': eta, 'operator': 'nonlinear exp(-i Q u)',
                'raw_corrected_weight_sum': float(np.sum(corrected)), 'exact_weight_sum': float(np.sum(exact)),
                'absolute_line_weight_L1_difference': float(np.sum(abs(corrected-exact))),
                'zero_energy_weight_difference': float(corrected[ell == 0][0]-exact[ell == 0][0]),
                'multiphonon_absolute_weight_difference': float(np.sum(abs(corrected[abs(ell) >= 2]-exact[abs(ell) >= 2]))),
                'normalization': 'none; the discrepancy is not removed by unit-area scaling'})
            np.savez_compressed(output/f'nonlinear_Kubo_T{temperature}_eta{eta:g}.npz', ell=ell, classical=classical, corrected=corrected, exact=exact)
    velocity = electron_speed(60000*constants.elementary_charge)
    traversal = 20e-9/velocity
    result['illustrative_material_traversal_scale'] = {'thickness_nm': 20., 'beam_energy_kev': 60.,
        'traversal_time_fs': traversal*1e15, '70mev_channel_phase_rad': 70e-3*constants.elementary_charge/constants.hbar*traversal,
        'thermal_motion_angstrom': {str(t): float(np.sqrt(constants.Boltzmann*t/(28.085*constants.atomic_mass))*traversal*1e10) for t in (300, 1000)},
        'interpretation': 'dimensionless/kinematic scale estimate only; not a uniform detector-error bound or verified TACAW thickness'}
    result['status'] = 'PASS' if result['auxiliary_vs_zero_channel_phase']['complex_relative_error'] < 1e-6 and result['quadrature'][-1]['complex_change'] < 1e-8 else 'INCOMPLETE'
    result['nonlinear_classical_exactness'] = 'NOT_EXACT: measured discrepancies are approximation differences, not failures of the channel solver'
    json_write(output/'results.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, temperature in zip(axes, (300, 1000)):
        with np.load(output/f'nonlinear_Kubo_T{temperature}_eta0.5.npz') as d:
            mask = abs(d['ell']) <= 5
            ax.plot(d['ell'][mask], d['exact'][mask], 'o-', label='exact quantum')
            ax.plot(d['ell'][mask], d['corrected'][mask], 'x--', label='classical nonlinear + Kubo factor')
        ax.set(title=f'{temperature} K, eta=0.5 diagnostic', xlabel='signed oscillator energy transfer / quantum', ylabel='unrenormalized line weight')
        ax.legend(fontsize=8)
    fig.savefig(output/'nonlinear_Kubo_control.png', dpi=160)
    plt.close(fig)
    (output/'report.md').write_text('\n'.join([
        '# Controlled auxiliary-wave and quantum-statistical comparisons', '',
        f"Auxiliary zero-channel-phase consistency: **{result['status']}**. Complex error {result['auxiliary_vs_zero_channel_phase']['complex_relative_error']:.8g}.", '',
        'A separate coordinate-space Gauss-Hermite integral propagates the electron through the two spatial scattering regions at fixed nuclear coordinates, then projects onto final Fock channels. It agrees with the actual coupled-channel solver when only channel-energy phases during traversal are omitted. Nuclear oscillator lengths, matrix elements and final energy assignments are retained. Restoring channel energies measures a physical auxiliary-wave approximation difference; neither global nor per-channel phases are aligned.', '',
        'For a classical harmonic linear-position correlation, the Kubo factor recovers the exact quantum loss/gain weights. For the nonlinear exp(-i Q u) operator the classical trajectory correlation is not the exact quantum Kubo correlation. The supplied factor does not make it exact. The measured unnormalized multiphonon/elastic differences are stored, without forcing the total to one.', '',
        'This supports separate treatment of classical-region one-phonon and nonlinear multiphonon comparisons. It does not establish the accuracy of the supplied material TACAW arrays. Those arrays already have Kubo correction according to the user and are not corrected again.', '',
        'The 20 nm/60 keV traversal estimates are illustrative scales, not measured TACAW geometry or detector-error bounds.', '',
        'Reference: [TACAW theory](https://doi.org/10.1103/PhysRevLett.134.036402).', '',
        '```bash', 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.auxiliary_wave_controls', '```', '',
        '![Kubo controls](nonlinear_Kubo_control.png)'])+'\n')


if __name__ == '__main__':
    main()
