"""Bounded controlled four-model EELS calculations using the validated source."""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .approximate_eels import (PRIMITIVE, ORIENTED, geometry, primitive_path,
    selected_covariance, run_covariance, run_born, run_source_case)
from .approximate_eels_gamma_patch import run as gamma_run
from .guarded_run import profile, validate_guard
from .workflow_io import ROOT, ResourceLimits, json_write, sha256
from yoshioka.lattice_dynamics import RealSpaceForceConstants

DEFAULT = 'benchmark_results/controlled_eels_comparison_20261102_v1'
PREVIOUS = ROOT/'benchmark_results/approximate_eels_20261102_v1'
JOBS = ('fitted_r4_T300', 'fitted_r4_T1000', 'fixed_r8_T300', 'fitted_r8_T300')


def physical_guard(output):
    """Freeze actual calculation sources/constants, without adopting stale data."""
    current = profile('benchmarks.controlled_eels')
    extras = ('approximate_eels', 'approximate_eels_gamma_patch', 'approximate_eels_analysis')
    current['source_sha256'].update({f'benchmarks/{name}.py': sha256(ROOT/'benchmarks'/f'{name}.py') for name in extras})
    record = validate_guard(output/'calculation_guard.json', current)
    snapshots = output/'source_snapshot'
    for name, expected in current['source_sha256'].items():
        p = snapshots/name
        if not p.exists(): p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes((ROOT/name).read_bytes())
        if sha256(p) != expected: raise ValueError('Calculation source snapshot changed')
    return record


def validate_covariances(config, output):
    for temperature in config['temperatures_k']:
        actual = sha256(primitive_path(config, temperature))
        for mesh in config['covariance_meshes']:
            path = output/'covariance'/f'T{temperature}_mesh{mesh}.json'
            if not path.exists(): continue
            record = json.loads(path.read_text()); covariance = np.asarray(record['covariance_m2'])
            if record.get('force_constants_sha256') != actual:
                raise ValueError('Covariance force-model identity changed')
            if covariance.shape != (3, 3) or not np.isfinite(covariance).all() or np.min(np.linalg.eigvalsh(covariance)) <= 0:
                raise ValueError('Invalid thermal covariance')


def setup(output):
    if not (output/'protected_before.json').exists():
        raise ValueError('Hash protected inputs before setup')
    if (output/'plan.json').exists():
        physical_guard(output); print('Existing setup verified'); return
    before = json.loads((output/'protected_before.json').read_text())['files']
    provenance = []
    def reuse(source, target):
        if source.is_dir():
            for path in sorted(source.rglob('*')):
                if path.is_file(): reuse(path, target/path.relative_to(source))
            return
        digest = sha256(source); key = str(source.relative_to(ROOT))
        if before[key]['sha256'] != digest: raise ValueError('Historical input changed before reuse')
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        if sha256(target) != digest: raise ValueError('Copy verification failed')
        provenance.append({'source': key, 'target': str(target.relative_to(output)), 'sha256': digest})
    jobs = []
    for name in JOBS:
        model, r, t = name.split('_'); repeats = int(r[1:]); temperature = int(t[1:])
        folder = output/name; folder.mkdir()
        prior = PREVIOUS/('ehm' if model == 'fitted' else '')
        force = PREVIOUS/'ehm'/f'force_constants_T{temperature}.npz' if model == 'fitted' else PRIMITIVE
        for mesh in (16, 24, 32, 64):
            source = prior/'covariance'/f'T{temperature}_mesh{mesh}.json'
            if source.exists(): reuse(source, folder/'covariance'/source.name)
        if repeats == 4:
            for mesh in (24, 32):
                reuse(prior/'born'/f'T{temperature}_mesh{mesh}', folder/'born'/f'T{temperature}_mesh{mesh}')
                for extension in ('.npz', '.json', '_quadrature.npz'):
                    p = prior/f'Gamma_cell_T{temperature}_mesh{mesh}{extension}'
                    if p.exists(): reuse(p, folder/p.name)
        # Preserve approximately the old physical dz at twice the specimen
        # thickness; 545 is an explicit numerical choice, not a fitted thickness.
        steps = 385 if repeats == 4 else 545
        settings = lambda label, grid, slices, power, seed: dict(name=label, temperature_k=temperature,
            grid_unit=grid, slices=slices, power=power, seed=seed)
        cases = [settings(f'prod_T{temperature}_seed{i}', 49, steps, 6, 611173+i) for i in range(8)]
        refinements = [settings('electron', 65, steps, 5, 611173),
            settings('slices', 49, 2*steps-1, 5, 611173),
            settings('joint', 65, 2*steps-1, 4, 611173)]
        config = {'output': str(folder.relative_to(ROOT)), 'runtime_limit_seconds': 600, 'memory_limit_gib': 2,
            'temperatures_k': [temperature], 'selected_repeats': repeats, 'padding_angstrom': 12,
            'covariance_meshes': [16, 24, 32, 64], 'selected_covariance_mesh': 32,
            'born_meshes': [32, 48, 64], 'born_batch': 256,
            'primitive_force_constants_by_temperature': {str(temperature): str(force.relative_to(ROOT))},
            'source_cases': cases, 'refinement_cases': refinements,
            'covariance_case': settings('covariance64', 49, steps, 4, 611173),
            'model': model, 'job': name}
        validate_covariances(config, folder)
        target = output/'configs'/f'{name}.json'; json_write(target, config)
        jobs.append({'name': name, 'config': str(target.relative_to(ROOT))})
    criteria = json.loads((PREVIOUS/'criteria.json').read_text())
    criteria.update({'workflow': 'controlled four-model and physical thickness comparison',
        'geometry': 'r4 and r8 same transverse cell, fixed equilibrium volume; thickness is a physical parameter',
        'mandatory_jobs': list(JOBS), 'not_required': ['1000K thickness extension', 'FC3/FC4', 'new MD', 'full9B', 'B7 independent joint'],
        'acceptance_note': 'Retain every detector and window. Report unresolved or competing features individually. No altered peak windows or broadenings.'})
    json_write(output/'criteria.json', criteria)
    json_write(output/'reuse_provenance.json', {'files': provenance, 'scope': 'Verified independent copies; no writable hard links to historical arrays.'})
    json_write(output/'plan.json', {'jobs': jobs, 'previous_fixed_r4': str(PREVIOUS.relative_to(ROOT)),
        'order': 'fitted_r4_T300, fitted_r4_T1000; then both r8 jobs. Pilot before production.',
        'fixed_volume': True, 'original_B11A_status': 'FAIL preserved', 'linewidth_status': 'NOT_RESOLVED preserved'})
    physical_guard(output)
    print('Prepared jobs:', ', '.join(JOBS), flush=True)


def check_source_contract(config, case, folder):
    path = folder/'source'/case['name']/'state.json'
    if not path.exists(): return 0
    state = json.loads(path.read_text()); contract = state['contract']
    correct = contract['settings'] == case and contract['repeats'] == config['selected_repeats']
    correct &= contract['padding_angstrom'] == config['padding_angstrom']
    correct &= contract['force'] == sha256(primitive_path(config, case['temperature_k']))
    correct &= contract['algorithm'] == sha256(ROOT/'yoshioka/bulk_thermal_source.py')
    correct &= contract['covariance'] == selected_covariance(config, folder, case['temperature_k']).tolist()
    if not correct: raise ValueError('Completed-prefix physical contract changed')
    if state['completed_q'] > state['total_q'] or state['total_q'] != 2**case['power']:
        raise ValueError('Invalid committed source prefix')
    return state['completed_q']


def run(output, name, stage):
    physical_guard(output)
    config = json.loads((output/'configs'/f'{name}.json').read_text()); folder = ROOT/config['output']
    validate_covariances(config, folder)
    limits = ResourceLimits(config['runtime_limit_seconds'], config['memory_limit_gib'], 10000)
    start = time.monotonic(); before = after = None
    if stage == 'covariance': run_covariance(config, folder, limits)
    elif stage == 'born': run_born(config, folder, limits)
    elif stage == 'gamma':
        for mesh in config['born_meshes']: gamma_run(config, folder, selected_mesh=mesh)
    else:
        if stage == 'pilot': cases = config['source_cases'][:1]
        elif stage == 'source': cases = config['source_cases']
        elif stage == 'refine': cases = config['refinement_cases']
        elif stage == 'covtest':
            config = {**config, 'selected_covariance_mesh': 64}; cases = [config['covariance_case']]
        else: raise ValueError(stage)
        before = sum(check_source_contract(config, c, folder) for c in cases)
        try:
            for case in cases:
                limits.check()
                state = run_source_case(config, case, folder, limits, 4 if stage == 'pilot' else None)
                if state['status'] != 'COMPLETE': break
        except (TimeoutError, MemoryError) as exc: print('Bounded stop:', exc, flush=True)
        after = sum(check_source_contract(config, c, folder) for c in cases)
        print(name, stage, 'committed', before, '->', after, '/', sum(2**c['power'] for c in cases), flush=True)
    record = {'job': name, 'stage': stage, 'seconds': time.monotonic()-start, 'before_q': before, 'after_q': after,
        'resources': limits.costs()}
    if stage == 'pilot':
        state = json.loads((folder/'source'/config['source_cases'][0]['name']/'state.json').read_text())
        record['committed_q_seconds'] = state['cumulative_seconds']
        record['seconds_per_q'] = state['cumulative_seconds']/state['completed_q']
        record['estimated_512_q_seconds'] = 512*record['seconds_per_q']
        oriented = RealSpaceForceConstants.load(ORIENTED)
        record['thickness_nm'] = config['selected_repeats']*oriented.cell_m[2, 2]*1e9
        record['dz_angstrom'] = geometry(oriented, config['selected_repeats'])[2]/config['source_cases'][0]['slices']*1e10
        json_write(folder/'pilot.json', record)
    path = folder/'invocations.json'; history = json.loads(path.read_text()) if path.exists() else []
    history.append(record); json_write(path, history)
    print(json.dumps(record), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default=DEFAULT)
    p.add_argument('--stage', required=True, choices=('setup', 'covariance', 'pilot', 'born', 'gamma', 'source', 'refine', 'covtest'))
    p.add_argument('--job', choices=JOBS)
    args = p.parse_args(); output = ROOT/args.output
    if args.stage == 'setup': setup(output)
    else:
        if args.job is None: raise ValueError('Select one bounded job')
        run(output, args.job, args.stage)


if __name__ == '__main__': main()
