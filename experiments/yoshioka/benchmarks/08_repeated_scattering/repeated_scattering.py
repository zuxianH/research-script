"""Benchmarks 3B and 8: solvable propagation and two-mode spatial scattering.

This is an explicitly synthetic, noncommuting periodic electron model, separate
from the material convergence repair. It is not a converged silicon model.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import constants
from scipy.special import gammaln

from yoshioka.coupled_channels import (ChannelSegment, SplitChannelPropagator, channel_basis,
    fourier_populations, linear_displacement_block, oscillator_phase_block, thermal_channel_weights)
from yoshioka.elastic_propagation import electron_speed
from yoshioka.free_propagation import TransverseGrid, electron_wavenumber
from .channel_reference import dense_generators, dense_propagate, distorted_first_order, dyson_orders
from .eels_observables import absolute_spectrum
from .workflow_io import ROOT, ResourceLimits, environment, fingerprint, json_write, sha256


def model(config, p, n, thickness_nm, linear=False, working_dimension=48, quadrature_order=None):
    basis = channel_basis(2, p)
    grid = TransverseGrid(*[v*1e-9 for v in config['transverse_lengths_nm']], n, n)
    quanta = np.asarray(config['mode_quanta_mev'])*1e-3*constants.elementary_charge
    xx, yy = np.meshgrid(2*np.pi*np.arange(n)/n, 2*np.pi*np.arange(n)/n)
    identity = np.eye(len(basis))
    regions = []
    for harmonic, offset, xi, energy_ev in zip(((1, 0), (1, 1)), (0., .7), config['displacement_couplings_xi'], config['region_potential_ev']):
        phase = harmonic[0]*xx+harmonic[1]*yy+offset
        if linear:
            field = np.cos(phase)[..., None, None]*identity + np.sin(phase)[..., None, None]*linear_displacement_block(basis, xi)
        else:
            matrix = oscillator_phase_block(basis, xi, working_dimension, quadrature_order)
            field = .5*(np.exp(1j*phase)[..., None, None]*matrix +
                        np.exp(-1j*phase)[..., None, None]*matrix.conj().T)
        regions.append(field*energy_ev*constants.elementary_charge)
    length = thickness_nm*1e-9
    zero = np.zeros((len(basis), len(basis)), dtype=np.complex128)
    segments = [ChannelSegment(fraction*length, field) for fraction, field in
        zip((.125, .125, .375, .25, .125), (zero, regions[0], zero, regions[1], zero))]
    entrance = np.zeros((len(basis), n, n), dtype=np.complex128)
    entrance[0] = 1/np.sqrt(grid.lx_m*grid.ly_m)
    return basis, grid, quanta, segments, entrance


def observables(wave, grid, basis, quanta, electron_energy_j):
    probability = fourier_populations(wave, grid)
    populations = np.sum(probability, axis=(-2, -1))
    qx, qy = np.meshgrid(2*np.pi*np.fft.fftfreq(grid.nx, grid.dx_m), 2*np.pi*np.fft.fftfreq(grid.ny, grid.dy_m))
    angles = np.hypot(qx, qy)/electron_wavenumber(electron_energy_j)*1000
    masks = {'zero_beam': (qx == 0) & (qy == 0), 'central_0_1_mrad': angles < 1,
             'annulus_1_4_mrad': (angles >= 1) & (angles <= 4),
             'offaxis_1_0': (np.indices(grid.shape)[0] == 0) & (np.indices(grid.shape)[1] == 1)}
    transfers = basis @ quanta
    values = {'norm': float(np.sum(populations)), 'ground_population': float(populations[0]),
              'one_phonon_population': float(np.sum(populations[np.sum(basis, axis=1) == 1])),
              'two_or_more_population': float(np.sum(populations[np.sum(basis, axis=1) >= 2]))}
    for name, mask in masks.items():
        detector = np.sum(probability[:, mask], axis=-1)
        values[f'{name}/all'] = float(np.sum(detector))
        values[f'{name}/inelastic'] = float(np.sum(detector[1:]))
        for lo, hi in ((0, 40), (40, 65), (65, 90), (90, 500)):
            e = transfers/(1e-3*constants.elementary_charge)
            values[f'{name}/{lo}_{hi}_mev'] = float(np.sum(detector[(e >= lo) & (e < hi)]))
    return values, populations, probability, transfers


def compare_values(coarse, fine, absolute=1e-10, significance=1e-6):
    rows = []
    for name, value in fine.items():
        delta = abs(coarse[name]-value)
        relative = delta/abs(value) if value else None
        rows.append({'observable': name, 'coarse': coarse[name], 'fine': value, 'absolute_change': delta,
                     'relative_change': relative, 'passed': delta < absolute if abs(value) < significance else relative < 1e-4})
    return {'rows': rows, 'passed': all(row['passed'] for row in rows),
            'max_significant_relative_change': max((r['relative_change'] for r in rows if abs(r['fine']) >= significance), default=0),
            'max_absolute_change': max(r['absolute_change'] for r in rows)}


def solvable_regression(output):
    grid = TransverseGrid(1e-9, 1e-9, 2, 2)
    basis = channel_basis(1, 24)
    quantum, force = np.array([60., 47.])*1e-3*constants.elementary_charge
    energy = 60000*constants.elementary_charge
    theta = 1.3
    length = theta*constants.hbar*electron_speed(energy)/quantum
    initial = np.zeros((len(basis), 2, 2), dtype=np.complex128)
    initial[0] = 1e9
    solver = SplitChannelPropagator(grid, energy, basis[:, 0]*quantum)
    wave, record = solver.propagate(initial, [ChannelSegment(length, linear_displacement_block(basis, [force]))], 4)
    lam = force/quantum
    alpha = lam*(np.exp(-1j*theta)-1)
    n = basis[:, 0]
    exact = np.exp(1j*lam**2*(theta-np.sin(theta))-.5*abs(alpha)**2)*alpha**n*np.exp(-.5*gammaln(n+1))
    numerical = wave[:, 0, 0]*1e-9
    error = float(np.linalg.norm(numerical-exact))
    result = {'benchmark': 'Benchmark 3B solvable driven-oscillator propagation', 'status': 'PASS' if error < 1e-10 and record['max_relative_norm_drift'] < 1e-10 else 'FAIL',
              'physical_cutoff': 24, 'theta_omega_L_over_v': theta, 'quantum_mev': 60, 'force_energy_mev': 47,
              'thickness_m': length, 'analytical_alpha_real_imag': [float(alpha.real), float(alpha.imag)],
              'analytical_global_phase_rad': float(lam**2*(theta-np.sin(theta))),
              'complex_amplitude_vector_error': error, 'propagation': record,
              'scope': 'actual common-v channel propagation; uniform electron control; no phase alignment; no general spatial solver inference'}
    np.savez_compressed(output/'solvable_3B_amplitudes.npz', numerical=numerical, exact=exact)
    json_write(output/'solvable_3B.json', result)
    return result


def independent_controls(config, output):
    rows = []
    energy = config['electron_energy_kev']*1000*constants.elementary_charge
    matrix_errors = []
    for linear in (False, True):
        basis, grid, quanta, segments, initial = model(config, 3, 4, 80, linear)
        _, _, _, quadrature_segments, _ = model(config, 3, 4, 80, linear, quadrature_order=96)
        matrix_errors.append(max(float(np.linalg.norm(a.potential_j-b.potential_j)) for a, b in zip(segments, quadrature_segments))/constants.elementary_charge)
        reference = dense_generators(grid, energy, basis@quanta, quadrature_segments)
        first0, first1 = distorted_first_order(initial, reference, 48)
        check0, check1 = distorted_first_order(initial, reference, 96)
        dyson = dyson_orders(initial, reference, 3)
        full_reference = dense_propagate(initial, reference, g=1.)
        solver = SplitChannelPropagator(grid, energy, basis@quanta)
        for steps in (8, 16, 32, 64):
            wave, record = solver.propagate(initial, segments, steps, g=1.)
            rows.append({'linear_control': linear, 'steps_per_segment': steps,
                         'complex_relative_error': float(np.linalg.norm(wave-full_reference)/np.linalg.norm(full_reference)),
                         **record})
        weak = []
        for g in (1e-3, 2e-3, .01, .1, 1., 2.):
            full = dense_propagate(initial, reference, g)
            values, populations, probability, transfers = observables(full, grid, basis, quanta, energy)
            single = first0+g*first1
            single_values, _, _, _ = observables(single, grid, basis, quanta, energy)
            weak.append({'g': g, 'full_observables': values, 'single_transition_observables': single_values,
                         'complex_departure_from_first_order': float(np.linalg.norm(full-single)/np.linalg.norm(initial)),
                         'minimum_signed_detector_difference': min(values[k]-single_values[k] for k in values if 'inelastic' in k)})
        noncommuting = np.linalg.norm(reference[1][3] @ reference[3][3]-reference[3][3] @ reference[1][3])/(np.linalg.norm(reference[1][3])*np.linalg.norm(reference[3][3]))
        total = np.sum(basis, axis=1)
        rows.append({'linear_control': linear, 'first_order_quadrature_complex_change': float(np.linalg.norm(first1-check1)/np.linalg.norm(initial)),
                     'Dyson_first_order_agreement': float(np.linalg.norm(first1-dyson[1])/np.linalg.norm(initial)),
                     'single_transition_two_phonon_weight_g1': float(np.sum(abs(first1[total == 2])**2)*grid.dx_m*grid.dy_m),
                     'Dyson_second_order_two_phonon_weight': float(np.sum(abs(dyson[2, total == 2])**2)*grid.dx_m*grid.dy_m),
                     'noncommutativity_relative_norm': float(noncommuting), 'coupling_sweep': weak})
        np.savez_compressed(output/f"small_reference_{'linear' if linear else 'nonlinear'}.npz", dyson_amplitudes=dyson,
                            full=full_reference, first0=first0, first1=first1, basis=basis)
    result = {'rows': rows, 'matrix_element_block_errors_ev': matrix_errors,
              'scope': 'small independent 4x4 electron, total p=3, 2-mode system; dense exponential uses independently quadrature-evaluated nonlinear couplings',
              'passed': all(r['complex_relative_error'] < 1e-6 and r['max_relative_norm_drift'] < 1e-9 for r in rows if r.get('steps_per_segment') == 64) and max(matrix_errors) < 1e-10}
    json_write(output/'independent_controls.json', result)
    return result


def thermal_trace_control(config, output):
    """Explicit incoherent trace vs complete deterministic phase-cycle estimator."""
    energy = config['electron_energy_kev']*1000*constants.elementary_charge
    basis, grid, quanta, segments, initial = model(config, 2, 4, 20)
    generators = dense_generators(grid, energy, basis@quanta, segments)
    # Each column is a separately propagated initial crystal state, never a
    # coherent average of Boltzmann amplitudes. The Fourier phase cycle cancels
    # cross terms exactly and validates an alternative density-trace estimator.
    exits = []
    for m in range(len(basis)):
        entrance = np.zeros_like(initial)
        entrance[m] = initial[0]
        exits.append(dense_propagate(entrance, generators))
    exits = np.array(exits)
    rows = []
    for temperature in (0, 300, 1000):
        weights = thermal_channel_weights(basis, quanta, temperature)
        explicit = np.einsum('m,mcyx->cyx', weights, abs(exits)**2)
        phases = np.exp(2j*np.pi*np.arange(len(basis))[:, None]*np.arange(len(basis))[None]/len(basis))
        probes = np.einsum('rm,m,mcyx->rcyx', phases, np.sqrt(weights), exits)
        cycled = np.mean(abs(probes)**2, axis=0)
        rows.append({'temperature_k': temperature, 'initial_retained_thermal_weight': float(np.sum(weights)),
                     'initial_omitted_thermal_weight': float(1-np.sum(weights)),
                     'relative_trace_error': float(np.linalg.norm(explicit-cycled)/np.linalg.norm(explicit)),
                     'scope': 'same finite represented Hilbert space, unnormalized omitted thermal weight reported; not thermal convergence of full material'})
    json_write(output/'thermal_trace_control.json', rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='benchmarks/configs/repeated_20261102_v1.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = ROOT/config['output']
    output.mkdir(parents=True, exist_ok=True)
    limits = ResourceLimits(config['runtime_limit_seconds'], config['memory_limit_gib'], 100)
    sources = {name: sha256(ROOT/name) for name in ('yoshioka/coupled_channels.py', 'benchmarks/channel_reference.py', 'benchmarks/repeated_scattering.py', 'yoshioka/harmonic_oscillator.py')}
    result = {'benchmark': 'Benchmark 8 synthetic two-mode spatial repeated scattering', 'status': 'INCOMPLETE',
              'parameters': config, 'environment': environment(), 'source_sha256': sources,
              'solvable_3B': solvable_regression(output), 'independent_controls': independent_controls(config, output),
              'thermal_trace_control': thermal_trace_control(config, output), 'cases': [],
              'physical_scope': 'two nondegenerate synthetic modes, periodic transverse electron field, two separated rigid-atom cosine interactions and free propagation between them; artificial g scales off-diagonal interactions only'}
    energy = config['electron_energy_kev']*1000*constants.elementary_charge
    specifications = []
    for p in config['physical_cutoffs']:
        specifications.append((p, config['selected_grid'], config['selected_steps'], config['maximum_thickness_nm'], config['maximum_g'], False))
    for n in config['electron_grids']:
        specifications.append((config['selected_cutoff'], n, config['selected_steps'], config['maximum_thickness_nm'], config['maximum_g'], False))
    for steps in config['longitudinal_steps']:
        specifications.append((config['selected_cutoff'], config['selected_grid'], steps, config['maximum_thickness_nm'], config['maximum_g'], False))
    for length in config['thicknesses_nm']:
        for g in config['couplings_g']:
            for linear in (False, True):
                specifications.append((config['selected_cutoff'], config['selected_grid'], config['selected_steps'], length, g, linear))
    seen = set()
    for p, n, steps, length, g, linear in specifications:
        settings = {'p': p, 'n': n, 'steps': steps, 'thickness_nm': length, 'g': g, 'linear': linear}
        digest = fingerprint({'settings': settings, 'physics': {key: config[key] for key in ('mode_quanta_mev', 'region_potential_ev', 'displacement_couplings_xi', 'transverse_lengths_nm', 'electron_energy_kev')}, 'source': sources})
        if digest in seen:
            continue
        seen.add(digest)
        folder = output/'cache'/digest[:20]
        folder.mkdir(parents=True, exist_ok=True)
        record_path = folder/'result.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record['digest'] != digest or sha256(record['wave_artifact']) != record['wave_sha256']:
                raise ValueError('invalid completed channel checkpoint')
        else:
            count = (p+2)*(p+1)//2
            try:
                limits.check(estimated_bytes=16*n*n*count*count*10+300*1024**2)
            except (TimeoutError, MemoryError) as exc:
                result['resource_stop'] = str(exc)
                break
            started = time.monotonic()
            basis, grid, quanta, segments, initial = model(config, p, n, length, linear)
            wave, metrics = SplitChannelPropagator(grid, energy, basis@quanta).propagate(initial, segments, steps, g=g)
            values, populations, probability, transfers = observables(wave, grid, basis, quanta, energy)
            target = folder/'wave.npz'
            np.savez_compressed(target, wave=wave, populations=populations, momentum_channel_probability=probability, energy_transfers_j=transfers, basis=basis)
            record = {'digest': digest, 'settings': settings, 'channel_count': count, 'observables': values, 'propagation': metrics,
                      'wave_artifact': str(target), 'wave_sha256': sha256(target), 'elapsed_seconds': time.monotonic()-started}
            json_write(record_path, record)
        result['cases'].append(record)
        print('B8', settings, 'depletion', 1-record['observables']['ground_population'], 'seconds', record['elapsed_seconds'], flush=True)
        json_write(output/'checkpoint.json', result)
    def series_comparison(key, settings):
        selected = [r for r in result['cases'] if all(r['settings'][k] == v for k, v in settings.items())]
        selected.sort(key=lambda r: r['settings'][key])
        return [{'coarse_settings': a['settings'], 'fine_settings': b['settings'], **compare_values(a['observables'], b['observables'])} for a, b in zip(selected[:-1], selected[1:])]
    base = {'thickness_nm': config['maximum_thickness_nm'], 'g': config['maximum_g'], 'linear': False}
    result['channel_refinement'] = series_comparison('p', {**base, 'n': config['selected_grid'], 'steps': config['selected_steps']})
    result['electron_refinement'] = series_comparison('n', {**base, 'p': config['selected_cutoff'], 'steps': config['selected_steps']})
    result['longitudinal_refinement'] = series_comparison('steps', {**base, 'p': config['selected_cutoff'], 'n': config['selected_grid']})
    result['solver_status'] = 'PASS' if result['independent_controls']['passed'] and result['solvable_3B']['status'] == 'PASS' else 'FAIL'
    converged = all(result[k] and result[k][-1]['passed'] for k in ('channel_refinement', 'electron_refinement', 'longitudinal_refinement'))
    result['repeated_interaction_status'] = 'PASS' if converged else 'INCOMPLETE'
    result['status'] = 'PASS' if converged and result['solver_status'] == 'PASS' else 'INCOMPLETE'
    result['resource_costs'] = limits.costs()
    json_write(output/'results.json', result)
    write_outputs(result, output)


def write_outputs(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for control in result['independent_controls']['rows']:
        if 'coupling_sweep' not in control:
            continue
        sweep = control['coupling_sweep']
        label = 'linear displacement' if control['linear_control'] else 'nonlinear rigid-atom'
        axes[0].loglog([r['g'] for r in sweep], [1-r['full_observables']['ground_population'] for r in sweep], '.-', label=label)
        axes[1].plot([r['g'] for r in sweep], [r['complex_departure_from_first_order'] for r in sweep], '.-', label=label)
    axes[0].set(xlabel='artificial off-diagonal scale g', ylabel='ground-channel depletion')
    axes[1].set(xlabel='artificial off-diagonal scale g', ylabel='complex departure from single-transition reference')
    for ax in axes:
        ax.legend()
    fig.savefig(output/'repeated_interaction_controls.png', dpi=160)
    plt.close(fig)
    text = ['# Benchmarks 3B and 8: actual channel propagation', '',
            f"Benchmark 3B: **{result['solvable_3B']['status']}**; complex amplitude error {result['solvable_3B']['complex_amplitude_vector_error']:.8g}.", '',
            f"Benchmark 8 solver: **{result['solver_status']}**; repeated-interaction/convergence milestone: **{result['repeated_interaction_status']}**.", '',
            'This is a declared synthetic two-mode spatial model, not full silicon. The electron has periodic transverse dynamics and two separated cosine interactions with different spatial harmonics and displacement directions. Mode energies are nondegenerate. The kinetic, channel-energy and interaction operators do not all commute.', '',
            'The common-v Hamiltonian uses SI energy U, no longitudinal carrier, and initial delta_nm/sqrt(A). The full propagation is fourth-order unitary splitting with exact local Hermitian channel exponentials; an independent dense reference uses explicit Fourier vectors and a dense exponential. Nonlinear oscillator blocks are separately evaluated by coordinate quadrature for the reference. No arbitrary phase alignment or normalization is used.', '',
            'Hd contains electron kinetic energy, channel energy and channel-diagonal interaction. g scales only the off-diagonal K. The independent single-transition amplitude explicitly integrates Ud(L,z) K Ud(z,0) using converged Gauss-Legendre quadrature. It retains diagonal elastic scattering and differs from pure Born. Dyson amplitudes of orders 0–3 are computed by a separate block-triangular exponential, and summed before squaring. Signed full-minus-first-order detector differences are reported; they need not be positive.', '',
            'The linear-displacement control permits only one-quantum changes per interaction, so its first-order two-phonon amplitude is zero. The nonlinear rigid-atom model has direct multiphonon matrix elements; a final two-phonon channel alone therefore does not establish two interactions.', '',
            'Acceptance: complex solver error <1e-6, norm drift <1e-9, relevant detector/population changes <1e-4 relative. Before calculation, observables below probability 1e-6 use absolute change <1e-10. All absolute and relative errors remain recorded. The physical p cutoff is distinct from working dimension 48; matrix-element quadrature is independent. Artificial g values are diagnostics.', '',
            'The thermal control compares an explicit incoherent finite-Hilbert-space trace with a complete phase-cycle trace estimator. It reports the omitted infinite thermal weight without renormalization. This validates trace evaluation; it does not establish macroscopic thermal convergence.', '',
            '```bash', 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.repeated_scattering', '```', '',
            'Each completed configuration has an input/source identity, wave checksum and atomic result record; the same command resumes completed configurations. Resource bounds are checked before each configuration.', '',
            '![controls](repeated_interaction_controls.png)']
    (output/'report.md').write_text('\n'.join(text)+'\n')


if __name__ == '__main__':
    main()
