"""Independent complex source/atomic-sum controls for the actual fitted modes."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy import constants
from .controlled_eels import DEFAULT, JOBS, physical_guard
from .approximate_eels import prepare_background, primitive_path
from .workflow_io import ROOT, ResourceLimits, json_write, sha256
from yoshioka.bulk_thermal_source import mode_group, occupations
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.fourier_depth_fields import complex_depth_field, fourier_depth_provider
from yoshioka.bloch_source_windowed import source_exit_bloch
from yoshioka.dynamical_source import full_transverse_vectors
from yoshioka.free_propagation import electron_wavenumber
from yoshioka.electron_phonon import full_space_born, LobatoSilicon


def run(output, name):
    physical_guard(output)
    path = output/(name+'_independent_source.json')
    if path.exists(): print('Existing independent controls preserved:', name); return
    config = json.loads((output/'configs'/f'{name}.json').read_text()); folder = ROOT/config['output']
    temperature = config['temperatures_k'][0]; force = primitive_path(config, temperature)
    primitive = RealSpaceForceConstants.load(force); rows = []; limits = ResourceLimits(600, 2)
    nodes = ([.071, -.113, .193], [.001, -.002, .003], [.35, -.45, .12])
    for steps in (257, 513, 1025):
        settings = {'temperature_k': temperature, 'grid_unit': 17, 'slices': steps, 'elastic': False}
        positions, reference, length, covariance, grid, _, incident, _ = prepare_background(config, settings, folder, limits)
        for q in nodes:
            limits.check(); start = time.monotonic()
            omega, d, shift, _ = mode_group(primitive, positions, [grid.lx_m, grid.ly_m], q, reference_positions=reference)
            target = folder/'independent_control_source.npy'
            mapped = complex_depth_field(positions, covariance, grid, length, steps, target, d, shift)
            provider = fourier_depth_provider(mapped, grid); differences = []; expected_norms = []; maximum = 0.
            for sign in (1, -1):
                at = provider if sign == 1 else lambda i: provider(i).conj()
                actual = source_exit_bloch(grid, 60000*constants.elementary_charge, sign*omega, length, steps,
                    at, lambda i: np.asarray(incident[i]), None, sign*shift).reshape(len(omega), -1)
                Q = full_transverse_vectors(grid)+sign*shift
                mask = np.linalg.norm(Q, axis=1)/electron_wavenumber(60000*constants.elementary_charge)*1000 <= 20+1e-12
                expected = full_space_born(LobatoSilicon.from_abtem_table(), positions, d if sign == 1 else d.conj(), covariance,
                    Q[mask], sign*omega, 60000*constants.elementary_charge, grid.lx_m*grid.ly_m, length)
                thermal = np.sqrt(occupations(omega, temperature)+(1 if sign == 1 else 0))[:, None]
                difference = (actual[:, mask]-expected)*thermal
                differences.append(float(np.sum(abs(difference)**2))); expected_norms.append(float(np.sum(abs(expected*thermal)**2)))
                maximum = max(maximum, float(np.max(abs(difference))))
            del mapped; target.unlink()
            rows.append({'q_fractional': q, 'slices': steps, 'physical_modes': len(omega),
                'complex_absolute_vector_error': np.sqrt(sum(differences)),
                'complex_relative_vector_error': np.sqrt(sum(differences)/sum(expected_norms)),
                'maximum_complex_absolute_error': maximum, 'seconds': time.monotonic()-start,
                'phase_alignment': False})
    finest = [r for r in rows if r['slices'] == 1025]
    result = {'job': name, 'rows': rows, 'passed': all(r['complex_relative_vector_error'] < 1e-4 for r in finest),
        'criterion': 'Existing independent-source complex-vector relative tolerance 1e-4, no phase alignment; all three q controls.',
        'force_sha256': sha256(force), 'temperature_k': temperature, 'resources': limits.costs()}
    json_write(path, result); print(name, 'independent source', result['passed'], max(r['complex_relative_vector_error'] for r in finest), flush=True)
    if not result['passed']: raise RuntimeError('Independent source validation failed')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', default=DEFAULT); p.add_argument('--job', choices=JOBS, required=True)
    args = p.parse_args(); run(ROOT/args.output, args.job)


if __name__ == '__main__': main()
