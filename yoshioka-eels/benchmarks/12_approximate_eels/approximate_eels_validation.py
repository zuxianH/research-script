"""Independent numerical-source and approximation diagnostics for bulk-q EELS."""

import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy import constants

from yoshioka.bulk_thermal_source import mode_group, occupations, importance_nodes
from yoshioka.bloch_source_windowed import source_exit_bloch
from yoshioka.fourier_depth_fields import complex_depth_field, fourier_depth_provider
from yoshioka.dynamical_source import full_transverse_vectors
from yoshioka.free_propagation import electron_wavenumber
from yoshioka.electron_phonon import full_space_born, LobatoSilicon
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from .approximate_eels import prepare_background, selected_covariance, geometry, PRIMITIVE, ORIENTED
from .workflow_io import ROOT, ResourceLimits, json_write


def numerical_source_control(config, output, limits):
    primitive = RealSpaceForceConstants.load(PRIMITIVE)
    rows = []
    q = np.array([.071, -.113, .193])
    for steps in (129, 257, 513):
        settings = {'temperature_k': 300, 'grid_unit': 17, 'slices': steps, 'elastic': False}
        positions, reference, length, covariance, grid, background, incident, _ = prepare_background(config, settings, output, limits)
        omega, d, shift, _ = mode_group(primitive, positions, [grid.lx_m, grid.ly_m], q, reference_positions=reference)
        source = complex_depth_field(positions, covariance, grid, length, steps, output/'validation_source.npy', d, shift)
        provider = fourier_depth_provider(source, grid)
        errors = []; norms = []; max_abs = 0.
        for sign in (1, -1):
            at = provider if sign == 1 else lambda i: provider(i).conj()
            actual = source_exit_bloch(grid, 60000*constants.elementary_charge, sign*omega, length, steps,
                at, lambda i: np.asarray(incident[i]), None, sign*shift).reshape(len(omega), -1)
            Q = full_transverse_vectors(grid)+sign*shift
            mask = np.linalg.norm(Q, axis=1)/electron_wavenumber(60000*constants.elementary_charge)*1000 <= 20+1e-12
            expected = full_space_born(LobatoSilicon.from_abtem_table(), positions, d if sign == 1 else d.conj(), covariance,
                Q[mask], sign*omega, 60000*constants.elementary_charge, grid.lx_m*grid.ly_m, length)
            thermal = np.sqrt(occupations(omega, 300)+(1 if sign == 1 else 0))[:, None]
            difference = (actual[:, mask]-expected)*thermal
            errors.append(float(np.sum(abs(difference)**2))); norms.append(float(np.sum(abs(expected*thermal)**2)))
            max_abs = max(max_abs, float(np.max(abs(difference))))
        del source; (output/'validation_source.npy').unlink()
        rows.append({'slices': steps, 'grid_unit': 17, 'q_fractional': q.tolist(),
            'complex_vector_absolute_error': float(np.sqrt(sum(errors))),
            'complex_vector_relative_error': float(np.sqrt(sum(errors)/sum(norms))),
            'maximum_complex_absolute_error': max_abs, 'phase_alignment': False})
    return rows


def linearization_diagnostic(config, output, temperature, power=8):
    primitive = RealSpaceForceConstants.load(PRIMITIVE)
    oriented = RealSpaceForceConstants.load(ORIENTED)
    positions, reference, length = geometry(oriented, config['selected_repeats'], config['padding_angstrom'])
    covariance = selected_covariance(config, output, temperature)
    from .born_quadrature import bloch_beams
    nodes, inverse_pdf = importance_nodes(power, 419731)
    numerator = denominator = above_point_one = above_one = 0.
    eta_max = 0.
    for q, weight in zip(nodes, inverse_pdf):
        omega, d, shift, _ = mode_group(primitive, positions, np.diag(oriented.cell_m)[:2], q, weight, reference)
        Qxy, _, inside = bloch_beams(primitive, np.diag(oriented.cell_m)[:2], q[None], 60000*constants.elementary_charge)
        Qxy = Qxy[0, inside[0]]
        from yoshioka.elastic_propagation import electron_speed
        k = electron_wavenumber(60000*constants.elementary_charge); v = electron_speed(60000*constants.elementary_charge)
        a = full_space_born(LobatoSilicon.from_abtem_table(), positions, d, covariance, Qxy, omega,
            60000*constants.elementary_charge, np.prod(np.diag(oriented.cell_m)[:2]), length)
        probability = abs(a)**2*(occupations(omega, temperature)+1)[:, None]
        for m, frequency in enumerate(omega):
            Q = np.column_stack((Qxy, -np.sum(Qxy**2, axis=1)/(2*k)-frequency/v))
            eta = np.einsum('ia,ab,ib->i', Q, covariance, Q)
            numerator += float(np.sum(probability[m]*eta)); denominator += float(np.sum(probability[m]))
            above_point_one += float(np.sum(probability[m, eta > .1])); above_one += float(np.sum(probability[m, eta > 1]))
            eta_max = max(eta_max, float(np.max(eta)))
    return {'temperature_k': temperature, 'sampled_q': len(nodes), 'detector': '0–20 mrad',
        'emission_weighted_mean_Q_U_Q': numerator/denominator,
        'emission_weight_fraction_Q_U_Q_above_0_1': above_point_one/denominator,
        'emission_weight_fraction_Q_U_Q_above_1': above_one/denominator,
        'maximum_Q_U_Q_in_represented_detector': eta_max,
        'interpretation': 'A momentum-weighted displacement-expansion diagnostic, not a rigorous omitted-event bound. The one-linear-source model omits direct nonlinear multiphonon vertices.'}


def nonlinear_phase_controls():
    """Independent nonlinear oscillator vertex; not a material event-rate bound."""
    from yoshioka.harmonic_oscillator import fock_scattering_matrix
    rows = []; quantum = 60e-3*constants.elementary_charge
    for temperature in (300, 1000):
        nbar = occupations(np.array([quantum/constants.hbar]), temperature)[0]
        ratio = np.exp(-quantum/(constants.Boltzmann*temperature)); n = np.arange(64)
        initial = (1-ratio)*ratio**n
        for eta_t in (.01, .03, .08):
            xi = np.sqrt(eta_t/(2*nbar+1))
            matrix = fock_scattering_matrix(float(xi), 128)[:64, :64]
            probabilities = abs(matrix)**2*initial[None]
            transfer = n[:, None]-n[None]
            one = float(np.sum(probabilities[abs(transfer) == 1]))
            multi = float(np.sum(probabilities[abs(transfer) >= 2]))
            linear_dw = float(np.exp(-eta_t)*xi*xi*(2*nbar+1))
            rows.append({'temperature_k': temperature, 'oscillator_energy_mev': 60,
                'eta_thermal': eta_t, 'xi': float(xi), 'working_dimension': 128,
                'retained_initial_final_states': 64, 'exact_single_quantum_weight': one,
                'linear_DW_single_quantum_weight': linear_dw,
                'relative_single_quantum_difference': abs(one-linear_dw)/one,
                'direct_nonlinear_multiquantum_over_single': multi/one,
                'omitted_initial_weight': float(ratio**64),
                'scope': 'single instantaneous phase operator in one oscillator; not sequential material scattering or a rigorous full-material omission bound'})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='benchmarks/configs/approximate_eels_20261102_v1.json')
    args = p.parse_args(); config = json.loads(Path(args.config).read_text()); output = ROOT/config['output']
    limits = ResourceLimits(600, 2)
    rows = numerical_source_control(config, output, limits)
    result = {'independent_source': rows,
        'source_control_pass': rows[-1]['complex_vector_relative_error'] < 1e-4,
        'linearization': [linearization_diagnostic(config, output, t) for t in (300, 1000)],
        'nonlinear_phase_controls': nonlinear_phase_controls(),
        'resources': limits.costs()}
    json_write(output/'independent_validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
