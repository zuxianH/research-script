"""Validated finite-T two-insertion controls and a reduced Si optical pilot."""

import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy import constants
from scipy.linalg import expm

from yoshioka.thermal_dyson_words import (words_through_two, label, gram_matrix,
    thermal_moment, propagate_words, thermal_groups)
from yoshioka.bulk_thermal_source import occupations, mode_group
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.fourier_depth_fields import complex_depth_field, fourier_depth_provider
from yoshioka.free_propagation import electron_wavenumber
from yoshioka.dynamical_source import full_transverse_vectors
from yoshioka.elastic_propagation import electron_speed
from .approximate_eels import prepare_background, PRIMITIVE
from .eels_observables import absolute_spectrum
from .workflow_io import ROOT, ResourceLimits, json_write


def dense_thermal_control(temperature, dimension=64):
    """Noncommuting two-beam electron Hamiltonians, identical finite Fock H."""
    quantum = 60e-3*constants.elementary_charge; length = 1.535553097624577e-9
    hv = constants.hbar*electron_speed(60000*constants.elementary_charge)
    n = np.arange(dimension); lowering = np.diag(np.sqrt(n[1:]), 1); raising = lowering.T
    words = words_through_two(1); lookup = {w: i for i, w in enumerate(words)}
    energies = np.array([sum(label(j, 1)[1] for j in w)*quantum for w in words])
    regions = [(0.4, np.array([[0., 11.], [11., 6.]])*constants.elementary_charge,
                    np.array([[.3, 1.2j], [.7, -.2]])*constants.elementary_charge),
               (0.6, np.array([[3., 4.j], [-4.j, -2.]])*constants.elementary_charge,
                    np.array([[.2j, -.8], [1.3j, .1]])*constants.elementary_charge)]
    coefficients = np.zeros((len(words), 2), complex); coefficients[0, 0] = 1
    reference = np.zeros((3*2*dimension, dimension), complex); reference[:dimension] = np.eye(dimension)
    full_matrices = []
    for fraction, elastic, coupling in regions:
        word_h = (np.kron(np.eye(len(words)), elastic)+np.diag(np.repeat(energies, 2))).astype(complex)
        for i, word in enumerate(words[1:], 1):
            parent = lookup[word[1:]]; matrix = coupling if word[0] == 0 else coupling.conj().T
            word_h[2*i:2*i+2, 2*parent:2*parent+2] += matrix
        coefficients = (expm(-1j*fraction*length/hv*word_h)@coefficients.ravel()).reshape(len(words), 2)
        diagonal = np.kron(elastic, np.eye(dimension))+np.kron(np.eye(2), np.diag(n*quantum))
        interaction = np.kron(coupling, raising)+np.kron(coupling.conj().T, lowering)
        augmented = np.kron(np.eye(3), diagonal)+np.kron(np.diag(np.ones(2), -1), interaction)
        reference = expm(-1j*fraction*length/hv*augmented)@reference
        full_matrices.append((fraction, diagonal, interaction))
    reconstructed = np.zeros((3, 2, dimension, dimension), complex)
    for word, electron in zip(words, coefficients):
        matrix = np.eye(dimension)
        for letter in word: matrix = matrix@(raising if letter == 0 else lowering)
        reconstructed[len(word)] += electron[:, None, None]*matrix[None]
    reconstructed *= np.exp(-1j*n*quantum*length/hv)[None, None, None]
    error = float(np.linalg.norm(reconstructed.reshape(reference.shape)-reference))
    nbar = occupations(np.array([quantum/constants.hbar]), temperature)
    ratio = np.exp(-quantum/(constants.Boltzmann*temperature)); initial = (1-ratio)*ratio**n
    gram = gram_matrix(words, 1, nbar)
    probability_wick = np.einsum('ie,ij,je->e', coefficients, gram, coefficients.conj()).real
    reference_wave = reference.reshape(3, 2, dimension, dimension).sum(axis=0)
    probability_dense = np.sum(abs(reference_wave)**2*initial[None, None], axis=(1, 2))
    order_scaling = []
    for g in (.25, .5, 1.):
        exact = np.zeros((2*dimension, dimension), complex); exact[:dimension] = np.eye(dimension)
        for fraction, diagonal, interaction in full_matrices:
            exact = expm(-1j*fraction*length/hv*(diagonal+g*interaction))@exact
        approximate = np.sum(reconstructed*np.array([1, g, g*g])[:, None, None, None], axis=0).reshape(2*dimension, dimension)
        difference = np.sqrt(np.sum(abs(exact-approximate)**2*initial[None]))
        p_exact = np.sum(abs(exact.reshape(2, dimension, dimension))**2*initial[None, None], axis=(1, 2))
        p_approx = np.sum(abs(approximate.reshape(2, dimension, dimension))**2*initial[None, None], axis=(1, 2))
        order_scaling.append({'g': g, 'thermal_complex_state_error': float(difference),
            'maximum_detector_intensity_error': float(np.max(abs(p_exact-p_approx))),
            'exact_total_weight': float(np.sum(p_exact)), 'truncated_total_weight': float(np.sum(p_approx))})
    return {'temperature_k': temperature, 'fock_dimension_reference_only': dimension,
        'electron_beams': 2, 'bath_word_count': len(words), 'complex_order_amplitude_error': error,
        'Wick_minus_explicit_thermal_trace_maximum': float(np.max(abs(probability_wick-probability_dense))),
        'omitted_reference_initial_weight': float(ratio**dimension), 'order_scaling': order_scaling,
        'passed': bool(error < 1e-10 and np.max(abs(probability_wick-probability_dense)) < 1e-10),
        'scope': 'identical linear Hamiltonians; exact dense order0/1/2 versus oscillator-word coefficients, without arbitrary phase alignment'}


def material_pilot(config, output, limits):
    primitive = RealSpaceForceConstants.load(PRIMITIVE)
    result = {'scope': 'Reduced primitive-Gamma optical triplet only; bulk acoustic/non-Gamma modes are absent. Full thermal covariance is the same mean-field background as9A. Not full9B.',
        'model': 'linear displacement coupling with all elastic diffraction and leading0/1/2 interaction amplitudes; not nonlinear single-event multiphonon scattering', 'cases': []}
    axis = np.arange(-180, 180.025, .05)
    for temperature in (300, 1000):
        for grid_n, steps in ((17, 257), (17, 513), (25, 513), (33, 513), (49, 513), (65, 513), (49, 1025)):
            path = output/f'repeated_optical_T{temperature}_n{grid_n}_z{steps}.npz'
            # Preserve and reuse the two original current-task pilot artifacts.
            legacy = output/f'repeated_optical_T{temperature}_z{steps}.npz'
            if grid_n == 17 and legacy.with_suffix('.json').exists(): path = legacy
            record_path = path.with_suffix('.json')
            if record_path.exists(): result['cases'].append(json.loads(record_path.read_text())); continue
            settings = {'temperature_k': temperature, 'grid_unit': grid_n, 'slices': steps}
            positions, reference, length, covariance, grid, background, incident, _ = prepare_background(config, settings, output, limits)
            # One oriented cell contains two primitive cells. The primitive
            # Gamma optical triplet is one folded sector, with weight1/2.
            omega, d, shift, branches = mode_group(primitive, positions, [grid.lx_m, grid.ly_m], [0, 0, 0],
                weight=.5, reference_positions=reference)
            start = time.monotonic()
            source = complex_depth_field(positions, covariance, grid, length, steps, output/'repeated_source.npy', d, shift)
            words, fourier, shifts, energies = propagate_words(grid, omega, np.tile(shift, (len(omega), 1)), length, steps,
                fourier_depth_provider(source, grid), lambda i: np.asarray(background[i, 0]), limits=limits)
            del source; (output/'repeated_source.npy').unlink()
            nbar = occupations(omega, temperature); spectra = {}; total = {}
            for order in (1, 2):
                lines = thermal_groups(words, fourier, len(omega), nbar, constants.hbar*omega, maximum_order=order)
                weights = []; transfers = []; ground = 0.; all_weight = 0.; gain = 0.; loss = 0.
                for line in lines:
                    Q = full_transverse_vectors(grid)+shifts[line['indices'][0]]
                    mask = np.linalg.norm(Q, axis=1)/electron_wavenumber(60000*constants.elementary_charge)*1000 <= 20+1e-12
                    value = float(np.sum(line['weight'][mask])); all_weight += value
                    if not any(line['net']): ground += value; continue
                    weights.append(value); transfers.append(line['energy_j'])
                    if line['energy_j'] > 1e-30: loss += value
                    elif line['energy_j'] < -1e-30: gain += value
                spectra[f'order{order}'] = absolute_spectrum(np.array(transfers), weights, axis)
                total[f'order{order}'] = {'detector_total_including_elastic': all_weight,
                'unchanged_bath_weight': ground, 'changed_bath_weight': float(np.sum(weights)),
                    'positive_energy_weight': loss, 'negative_energy_weight': gain}
                total[f'order{order}']['all_beam_total_weight'] = float(sum(np.sum(line['weight']) for line in lines))
            np.savez_compressed(path, energy_mev=axis, words=np.array([str(w) for w in words]),
                word_fourier_coefficients=fourier, signed_energies_j=energies,
                **{'spectrum_'+k: v for k, v in spectra.items()})
            row = {'temperature_k': temperature, 'slices': steps, 'grid_unit': grid_n,
                'active_modes': len(omega), 'active_mode_branches': branches.tolist(),
                'mode_energies_mev': (constants.hbar*omega/(1e-3*constants.elementary_charge)).tolist(),
                'thickness_nm': config['selected_repeats']*.38388827440614426,
                'bath_words': len(words), 'results': total, 'seconds': time.monotonic()-start,
                'normalization': 'primitiveGamma weight1/2 matches its sector in a one-oriented-cell folded bath; omitted folded and bulk sectors are not renormalized',
                'signed_change_total_weight': total['order2']['detector_total_including_elastic']-total['order1']['detector_total_including_elastic'],
                'artifact': str(path), 'no_normalization': True}
            json_write(record_path, row); result['cases'].append(row)
    json_write(output/'leading_repeated_material.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='benchmarks/configs/approximate_eels_20261102_v1.json')
    p.add_argument('--stage', choices=('validation', 'material', 'all'), default='all')
    args = p.parse_args(); config = json.loads(Path(args.config).read_text()); output = ROOT/config['output']
    limits = ResourceLimits(600, 2)
    if args.stage in ('validation', 'all'):
        rows = [dense_thermal_control(t) for t in (300, 1000)]
        json_write(output/'thermal_Dyson_validation.json', {'rows': rows, 'resources': limits.costs()})
        print(json.dumps(rows, indent=2), flush=True)
        if not all(r['passed'] for r in rows): raise RuntimeError('independent thermal Dyson validation failed')
    if args.stage in ('material', 'all'):
        print(json.dumps(material_pilot(config, output, limits), indent=2), flush=True)


if __name__ == '__main__': main()
