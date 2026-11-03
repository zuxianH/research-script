"""Finish B8 matrix-element, absolute-spectrum and first-order comparisons."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.constants import elementary_charge
from scipy.integrate import trapezoid

from yoshioka.coupled_channels import oscillator_phase_block
from yoshioka.distorted_channels import distorted_wave_orders
from .channel_reference import dense_generators, distorted_first_order
from .eels_observables import absolute_spectrum
from .repeated_scattering import compare_values, model, observables
from .workflow_io import ROOT, environment, json_write, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='benchmark_results/repeated_20261102_v2')
    args = parser.parse_args()
    output = ROOT/args.output
    output.mkdir(parents=True, exist_ok=True)
    previous_path = ROOT/'benchmark_results/repeated_20261102_v1/results.json'
    previous = json.loads(previous_path.read_text())
    config = previous['parameters']
    energy = config['electron_energy_kev']*1000*elementary_charge
    started = time.monotonic()
    record = {'benchmark': 'Benchmark 8 completed spectrum controls', 'parameters': config,
              'previous_results_sha256': sha256(previous_path), 'environment': environment(),
              'solver_status': previous['solver_status'], 'solvable_3B': previous['solvable_3B'],
              'matrix_element_refinement': [], 'first_order_validation': [], 'spectra': [],
              'scope': 'synthetic two-mode spatial Hamiltonian; all artificial coupling scales labelled diagnostics'}
    basis, grid, quanta, segments, initial = model(config, 12, 16, 80)
    for xi in config['displacement_couplings_xi']:
        blocks = [oscillator_phase_block(basis, xi, working_dimension=d) for d in (24, 48, 64)]
        quadrature = [oscillator_phase_block(basis, xi, working_dimension=64, quadrature_order=n) for n in (32, 64, 128)]
        record['matrix_element_refinement'].append({'xi': xi, 'physical_p': 12,
            'D_24_48_block_error': float(np.linalg.norm(blocks[0]-blocks[1])),
            'D_48_64_block_error': float(np.linalg.norm(blocks[1]-blocks[2])),
            'quad_32_64_block_error': float(np.linalg.norm(quadrature[0]-quadrature[1])),
            'quad_64_128_block_error': float(np.linalg.norm(quadrature[1]-quadrature[2])),
            'independent_quad_vs_Fock_error': float(np.linalg.norm(quadrature[-1]-blocks[-1]))})
    for linear in (False, True):
        small_basis, small_grid, small_quanta, small_segments, small_initial = model(config, 3, 4, 80, linear)
        generators = dense_generators(small_grid, energy, small_basis@small_quanta, small_segments)
        exact0, exact1 = distorted_first_order(small_initial, generators, 96)
        numerical0, numerical1 = distorted_wave_orders(small_grid, energy, small_basis@small_quanta, small_initial, small_segments, 64)
        record['first_order_validation'].append({'linear': linear,
            'zeroth_order_complex_error': float(np.linalg.norm(numerical0-exact0)/np.linalg.norm(small_initial)),
            'first_order_complex_error': float(np.linalg.norm(numerical1-exact1)/np.linalg.norm(small_initial))})
    axis = np.arange(-20, 1020.025, .05)
    for linear in (False, True):
        for length in (20, 80):
            basis, grid, quanta, segments, initial = model(config, 12, 16, length, linear)
            zero, first = distorted_wave_orders(grid, energy, basis@quanta, initial, segments, 64)
            for g in (0, .01, .1, 1, 2):
                case = next(c for c in previous['cases'] if c['settings'] == {'p': 12, 'n': 16, 'steps': 64, 'thickness_nm': length, 'g': g, 'linear': linear})
                if sha256(case['wave_artifact']) != case['wave_sha256']:
                    raise ValueError('completed full-propagation wave checksum mismatch')
                with np.load(case['wave_artifact'], allow_pickle=False) as d:
                    wave = d['wave'].copy()
                single = zero+g*first
                full_values, populations, _, transfers = observables(wave, grid, basis, quanta, energy)
                single_values, single_populations, _, _ = observables(single, grid, basis, quanta, energy)
                spectra = {}
                for elastic in (True, False):
                    chosen = np.ones(len(basis), bool) if elastic else np.sum(basis, axis=1) > 0
                    spectra['full_with_elastic' if elastic else 'full_inelastic'] = absolute_spectrum(transfers[chosen], populations[chosen], axis)
                    spectra['first_with_elastic' if elastic else 'first_inelastic'] = absolute_spectrum(transfers[chosen], single_populations[chosen], axis)
                target = output/f"{'linear' if linear else 'nonlinear'}_L{length}_g{g:g}.npz"
                np.savez_compressed(target, energy_grid_mev=axis, **spectra, full_line_weights=populations,
                    first_line_weights=single_populations, transfers_j=transfers, basis=basis)
                denominator = trapezoid(spectra['full_inelastic'], axis)
                area = trapezoid(spectra['full_with_elastic'], axis)
                record['spectra'].append({'linear': linear, 'thickness_nm': length, 'g': g, 'artifact': str(target),
                    'full_inelastic_weight': float(denominator), 'first_inelastic_weight': float(np.sum(single_populations[1:])),
                    'full_minus_first_spectral_L1_relative_to_full_inelastic': float(trapezoid(abs(spectra['full_inelastic']-spectra['first_inelastic']), axis)/denominator) if denominator > 1e-20 else None,
                    'minimum_signed_inelastic_spectrum_difference_per_mev': float(np.min(spectra['full_inelastic']-spectra['first_inelastic'])),
                    'spectrum_area_error': float(abs(area-np.sum(populations))),
                    'full_observables': full_values, 'first_observables': single_values,
                    'complex_full_minus_first_relative_norm': float(np.linalg.norm(wave-single)/np.linalg.norm(initial))})
    # Physical-cutoff convergence of the actual broadened spectra is independent
    # of their plotting area and independent of the first-order departure.
    record['spectral_channel_refinement'] = []
    previous_spectrum = None
    for p in config['physical_cutoffs']:
        case = next(c for c in previous['cases'] if c['settings'] == {'p': p, 'n': 16, 'steps': 64, 'thickness_nm': 80, 'g': 2, 'linear': False})
        with np.load(case['wave_artifact'], allow_pickle=False) as d:
            spectrum = absolute_spectrum(d['energy_transfers_j'][1:], d['populations'][1:], axis)
        if previous_spectrum is not None:
            error = float(trapezoid(abs(spectrum-previous_spectrum), axis)/trapezoid(spectrum, axis))
            record['spectral_channel_refinement'].append({'coarse_p': previous_p, 'fine_p': p, 'relative_L1': error, 'passed': error < 1e-4})
        previous_p, previous_spectrum = p, spectrum
    record['checks'] = {
        'independent_solver': previous['independent_controls']['passed'],
        'independent_first_order': max(r['first_order_complex_error'] for r in record['first_order_validation']) < 1e-6,
        'matrix_element_working_basis': max(r['D_48_64_block_error'] for r in record['matrix_element_refinement']) < 1e-10,
        'matrix_element_quadrature': max(r['independent_quad_vs_Fock_error'] for r in record['matrix_element_refinement']) < 1e-10,
        'absolute_spectral_area': max(r['spectrum_area_error'] for r in record['spectra']) < 1e-10,
        'spectral_physical_cutoff': record['spectral_channel_refinement'][-1]['passed'],
        'population_refinements': previous['status'] == 'PASS'}
    record['status'] = 'PASS' if all(record['checks'].values()) else 'INCOMPLETE'
    record['elapsed_seconds'] = time.monotonic()-started
    json_write(output/'results.json', record)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, linear in enumerate((False, True)):
        for g in (.1, 1, 2):
            case = next(c for c in record['spectra'] if c['linear'] == linear and c['thickness_nm'] == 80 and c['g'] == g)
            with np.load(case['artifact']) as d:
                axes[row, 0].plot(axis, d['full_inelastic'], label=f'full, g={g}')
                axes[row, 0].plot(axis, d['first_inelastic'], '--', label=f'first, g={g}')
                axes[row, 1].plot(axis, d['full_inelastic']-d['first_inelastic'], label=f'g={g}')
        axes[row, 0].set_title('linear displacement' if linear else 'nonlinear rigid-atom matrix elements')
        for ax in axes[row]:
            ax.set(xlim=(-5, 350), xlabel='electron energy loss (meV)', ylabel='absolute incident-normalized probability / meV')
            ax.legend(fontsize=7)
    fig.savefig(output/'full_and_single_transition_spectra.png', dpi=160)
    plt.close(fig)
    text = ['# Benchmark 8: converged nonlinear/linear spectral comparisons', '', f"Status: **{record['status']}** for the declared synthetic model.", '',
        'The previous solver/convergence report is preserved in repeated_20261102_v1. This supplement completes matrix-element working-dimension and quadrature refinements through D=64 and 128 nodes, and full-versus-single-transition spectra at the same physical p=12, 16×16 electron grid, 64 fourth-order steps per segment, g=0–2, and 20/80 nm thickness.', '',
        'The scalable first-order evaluator integrates each local insertion analytically between diagonal propagators and retains transverse dynamics through fourth-order splitting. It is separately checked against the independent dense Gauss-Legendre Ud K Ud integral. Full and single-transition calculations use identical nonlinear or linear coupling matrices and identical Hd. Orthogonal final channels are added as probabilities. Dyson orders are amplitudes and interfere within a channel.', '',
        'All unbroadened weights are stored. Gaussian FWHM is 2 meV and density units are probability/meV. No truncated spectrum or cropped range is renormalized. Inelastic and total spectra are stored separately, and signed differences can be negative. The finite physical cutoff and plotting-area checks have separate acceptance fields.', '',
        '```bash', 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.repeated_scattering', 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.repeated_scattering_spectra', '```', '',
        '![spectra](full_and_single_transition_spectra.png)']
    (output/'report.md').write_text('\n'.join(text)+'\n')
    json_write(output/'manifest.json', {'status': record['status'], 'results_sha256': sha256(output/'results.json'),
        'source_sha256': sha256(__file__), 'distorted_solver_sha256': sha256(ROOT/'yoshioka/distorted_channels.py')})


if __name__ == '__main__':
    main()
