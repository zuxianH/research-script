"""Measure source phases on saved production sectors without changing results."""

import argparse
import cProfile
import json
import pstats
import time
from pathlib import Path

import numpy as np

from .approximate_eels import prepare_background, primitive_path
from .workflow_io import ROOT, ResourceLimits, json_write
from yoshioka.bulk_thermal_source import importance_nodes, streamed_q
from yoshioka.lattice_dynamics import RealSpaceForceConstants


def run(config, output):
    target = output/'phase_timing_pilot.json'
    if target.exists():
        print('Existing timing pilot preserved:', target)
        return
    limits = ResourceLimits(config['runtime_limit_seconds'], config['memory_limit_gib'], 10000)
    rows = []
    phase_names = {'mode_group': 'mode_diagonalization_and_mapping',
                   'complex_depth_field': 'source_field_construction',
                   'source_exit_bloch': 'signed_final_channel_propagation',
                   'full_space_born': 'Born_reference'}
    for temperature in config['temperatures_k']:
        settings = next(case for case in config['source_cases'] if case['temperature_k'] == temperature)
        primitive = RealSpaceForceConstants.load(primitive_path(config, temperature))
        before = time.monotonic()
        positions, unpadded, length, covariance, grid, background, incident, _ = prepare_background(config, settings, output, limits)
        setup_seconds = time.monotonic()-before
        nodes, inverse = importance_nodes(settings['power'], settings['seed'])
        for iq in range(2):
            limits.check()
            profiler = cProfile.Profile(); before = time.monotonic()
            profiler.enable()
            energy, weight, born, _ = streamed_q(primitive, positions, covariance, grid, length,
                settings['slices'], nodes[iq], temperature, incident, background,
                output/'phase_timing_source.npy', weight=inverse[iq], reference_positions=unpadded)
            profiler.disable(); elapsed = time.monotonic()-before
            phases = {label: 0. for label in phase_names.values()}
            for (_, _, name), (_, _, _, cumulative, _) in pstats.Stats(profiler).stats.items():
                if name in phase_names: phases[phase_names[name]] += cumulative
            folder = output/'source'/settings['name']
            reference = np.load(folder/'weights.npy', mmap_mode='r')[iq]
            error = float(np.max(abs(weight-reference)))
            if error > 1e-12 or not np.isfinite(weight).all():
                raise RuntimeError('Timing rerun disagrees with the saved production sector')
            rows.append({'temperature_k': temperature, 'production_case': settings['name'], 'q_index': iq,
                'q_fractional': nodes[iq], 'physical_branches': len(energy)//2,
                'profiled_total_seconds': elapsed, 'inclusive_phase_seconds': phases,
                'other_seconds': elapsed-sum(phases.values()),
                'cached_background_setup_seconds': setup_seconds if iq == 0 else 0.,
                'maximum_absolute_saved_weight_difference': error})
    json_write(target, {'rows': rows, 'resources': limits.costs(),
        'scope': 'Four explicitly profiled sectors using cached incident/background fields; not a retrospective decomposition of full production time.',
        'timing_convention': 'Inclusive cProfile times for disjoint top-level source phases; Python profiling overhead included. Cold background construction and covariance are excluded; their original costs are reported separately.'})
    print('Timing pilot complete; sectors:', len(rows), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='benchmarks/configs/approximate_eels_production_20261102_v1.json')
    args = p.parse_args(); config = json.loads(Path(args.config).read_text())
    run(config, ROOT/config['output'])


if __name__ == '__main__': main()
