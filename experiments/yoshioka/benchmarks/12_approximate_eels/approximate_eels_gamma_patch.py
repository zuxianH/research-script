"""Integrate the thermal acoustic Gamma cell instead of omitting its interior.

The uniform nodal mesh excludes only the three rigid translations at Gamma,
but assigning zero acoustic weight to its entire quadrature cell depresses
the unresolved low-energy continuum. Replace that ONE cell by independently
scrambled importance quadrature of the actual matrix-element-weighted EELS.
No point mass, fitted scale, broadening change or second q weight is added.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy import constants
from scipy.integrate import trapezoid
from yoshioka.bulk_thermal_source import importance_nodes, born_thermal_batch
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from .approximate_eels import geometry, selected_covariance, primitive_path, ORIENTED
from .approximate_eels_analysis import available_born
from .born_quadrature import line_spectra
from .workflow_io import ROOT, ResourceLimits, json_write, sha256


def run(config, output, power=8, replicas=8, selected_mesh=None):
    limits = ResourceLimits(600, 2); records = []
    oriented = RealSpaceForceConstants.load(ORIENTED)
    _, positions, length = geometry(oriented, config['selected_repeats'], config['padding_angstrom'])
    for temperature in config['temperatures_k']:
        rows = available_born(output, temperature)
        if selected_mesh is not None: rows = [row for row in rows if row[0] == selected_mesh]
        if not rows: continue
        mesh, axis, original, state = rows[-1]
        path = output/f'Gamma_cell_T{temperature}_mesh{mesh}.npz'
        quadrature_path = path.with_name(path.stem+'_quadrature.npz')
        if not quadrature_path.exists():
            nodes = []; weights = []
            for replica in range(replicas):
                q, inverse = importance_nodes(power, 941971+replica)
                nodes.append(q/mesh); weights.append(inverse/(len(q)*mesh**3))
            if path.exists():
                with np.load(path) as saved:
                    np.testing.assert_array_equal(saved['q_fractional'], np.array(nodes))
            np.savez_compressed(quadrature_path, q_fractional=np.array(nodes), quadrature_weights=np.array(weights),
                normalized_BZ_cell_volume=1/mesh**3, seeds=np.arange(941971, 941971+replicas))
        if path.exists(): continue
        force = primitive_path(config, temperature); primitive = RealSpaceForceConstants.load(force)
        covariance = selected_covariance(config, output, temperature)
        folder = output/'born'/f'T{temperature}_mesh{mesh}'
        energy0 = np.load(folder/'energies_mev.npy', mmap_mode='r')[0]
        weight0 = np.load(folder/'weights.npy', mmap_mode='r')[0]
        removed = line_spectra(energy0, weight0, axis)
        estimates = []; raw = []
        for replica in range(replicas):
            limits.check()
            q, inverse = importance_nodes(power, 941971+replica)
            q /= mesh
            energies, weights, _ = born_thermal_batch(primitive, positions, covariance,
                np.diag(oriented.cell_m)[:2], length, config['padding_angstrom']*1e-10,
                q, inverse/(len(q)*mesh**3), temperature, 60000*constants.elementary_charge)
            estimates.append(line_spectra(energies, weights, axis)-removed)
            raw.append({'q': q, 'energies': energies, 'weights': weights})
        estimates = np.array(estimates); correction = np.mean(estimates, axis=0)
        sem = np.std(estimates, axis=0, ddof=1)/np.sqrt(replicas)
        np.savez_compressed(path, energy_mev=axis, correction=correction, standard_error=sem,
            independent_corrections=estimates, q_fractional=np.array([r['q'] for r in raw]),
            energies_mev=np.array([r['energies'] for r in raw]), weights=np.array([r['weights'] for r in raw]))
        selected = (abs(axis) < 5); compare = (axis >= 5)&(axis < 90)
        row = {'temperature_k': temperature, 'mesh': mesh, 'cell_fractional_bounds': [-.5/mesh, .5/mesh],
            'cell_volume_fraction': 1/mesh**3, 'replicas': replicas, 'q_per_replica': 2**power,
            'force_sha256': sha256(force), 'full_detector_added_weight_abs_E_lt5': float(trapezoid(correction[selected, 0], axis[selected])),
            'full_detector_relative_5_90_shape_effect_L1': float(trapezoid(abs(correction[compare, 0]), axis[compare])/trapezoid(original[compare, 0], axis[compare])),
            'replacement': 'Remove the existingGamma-node optical contribution once; integrate all6branches inside its normalized BZ cell. No continuumGamma point mass.',
            'artifact': str(path)}
        json_write(path.with_suffix('.json'), row); records.append(row)
        print('Gamma cell', temperature, mesh, row['full_detector_added_weight_abs_E_lt5'], flush=True)
    json_write(output/'Gamma_cell_latest.json', {'records': records, 'resources': limits.costs()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='benchmarks/configs/approximate_eels_20261102_v1.json')
    p.add_argument('--mesh', type=int)
    args = p.parse_args(); config = json.loads(Path(args.config).read_text()); output = ROOT/config['output']
    run(config, output, selected_mesh=args.mesh)


if __name__ == '__main__': main()
