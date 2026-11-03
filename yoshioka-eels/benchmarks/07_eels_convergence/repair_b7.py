"""Extend, without mutating, the accepted B7 angular-cell calculation.

The reference runner already tests the last two adjacent comparisons. Historical
failures are kept as evidence; they are not an ever-failing logical AND.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from yoshioka.lattice_dynamics import RealSpaceForceConstants
from .detector_cubature import checkpointed_cells
from .repair_convergence import compare_cases, summarize_arrays
from .workflow_io import ROOT, ResourceLimits, json_write, sha256


def finest_sequence_passes(comparisons, required=2):
    return len(comparisons) >= required and all(c['physical_passed'] for c in comparisons[-required:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='benchmarks/configs/repair_b7_20261102_v2.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = ROOT/config['output']
    output.mkdir(parents=True, exist_ok=True)
    old_path = ROOT/'benchmark_results/detector_cells_20261102_v1/results.json'
    old = json.loads(old_path.read_text())
    spec = json.loads((ROOT/'benchmark_results/si_eels_convergence/results.json').read_text())['model_specification']
    primitive = RealSpaceForceConstants.load(ROOT/'benchmark_results/si_eels_convergence/primitive_force_constants.npz')
    limits = ResourceLimits(config['runtime_limit_seconds'], config['memory_limit_gib'], 100)
    result = {'status': 'INCOMPLETE', 'parameters': config, 'cases': old['cases'].copy(),
              'comparisons': old['comparisons'].copy(), 'history_sha256': sha256(old_path),
              'reuse': 'read-only original cases and summary arrays; new meshes have separate checkpoints',
              'joint_source_status': 'INCOMPLETE', 'physical_quadrature_status': 'INCOMPLETE'}
    for mesh in config['meshes']:
        started = time.monotonic()
        folder, state, arrays = checkpointed_cells(primitive, spec, mesh, config.get('shift', [0, 0, 0]), output, limits)
        result['latest_checkpoint'] = str(folder/'state.json')
        result['source_cost'] = {k: state.get(k) for k in ('completed_q', 'total_q', 'cumulative_seconds', 'latest_invocation_seconds', 'resumed_from_q', 'stop_reason')}
        if state['status'] != 'COMPLETE':
            result['resource_costs'] = limits.costs()
            json_write(output/'results.json', result)
            print('checkpoint', mesh, result['source_cost'], flush=True)
            return
        case, spectra = summarize_arrays(arrays)
        case.update(settings={'mesh': mesh, 'shift': config.get('shift', [0, 0, 0]), 'route': 'Born with exact cell-detector overlaps'},
                    artifact=str(folder/'summary.npz'), checkpoint=state)
        np.savez_compressed(case['artifact'], energy_grid_mev=np.arange(-10, 100.025, .05),
            **{f'spectrum_{name}': spectra[:, j] for j, name in enumerate(('full_0_20_mrad', 'central_0_5_mrad', 'annulus_5_10_mrad', 'annulus_10_20_mrad'))})
        result['comparisons'].append(compare_cases(result['cases'][-1], case, output))
        result['cases'].append(case)
        print('B7 mesh', mesh, 'comparison', result['comparisons'][-1]['physical_passed'], 'summary seconds', time.monotonic()-started, flush=True)
    result['physical_quadrature_status'] = 'PASS' if finest_sequence_passes(result['comparisons'], config['required_successive_comparisons']) else 'INCOMPLETE'
    result['resource_costs'] = limits.costs()
    json_write(output/'results.json', result)


if __name__ == '__main__':
    main()
