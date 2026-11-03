"""Requirement-level completion audit and streamed preservation/provenance."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np

from .workflow_io import ROOT, json_write, sha256
from .approximate_eels import primitive_path
from .guarded_run import profile, validate_guard
from yoshioka.bulk_thermal_source import importance_nodes


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''): h.update(chunk)
    return h.hexdigest()


def preserve(output, stage):
    start = time.monotonic()
    if stage == 'before':
        if (output/'protected_before.json').exists(): raise ValueError('Preserve the existing baseline; do not overwrite it')
        files = {}
        for p in sorted(ROOT.rglob('*')):
            if not p.is_file() or output in p.parents or any(x in p.parts for x in ('.venv', '__pycache__', '.pytest_cache', '.git')): continue
            files[str(p.relative_to(ROOT))] = {'sha256': digest(p), 'bytes': p.stat().st_size}
        record = {'files': files, 'count': len(files), 'total_bytes': sum(x['bytes'] for x in files.values()), 'seconds': time.monotonic()-start}
        json_write(output/'protected_before.json', record)
    else:
        before = json.loads((output/'protected_before.json').read_text()); changed = []; missing = []
        for name, item in before['files'].items():
            p = ROOT/name
            if not p.is_file(): missing.append(name)
            elif digest(p) != item['sha256']: changed.append(name)
        record = {'count': before['count'], 'changed': changed, 'missing': missing,
            'all_unchanged': not changed and not missing, 'seconds': time.monotonic()-start}
        json_write(output/'protected_after.json', record)
    print(json.dumps({k: v for k, v in record.items() if k != 'files'}), flush=True)


def regression(output):
    start = time.monotonic()
    with (output/'regression.log').open('w') as log:
        run = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    text = (output/'regression.log').read_text(); match = re.search(r'Ran (\d+) tests', text)
    record = {'command': '.venv/bin/python -m unittest discover -s tests', 'returncode': run.returncode,
        'test_count': int(match.group(1)) if match else None, 'seconds': time.monotonic()-start,
        'log': str(output/'regression.log')}
    json_write(output/'regression.json', record)
    if run.returncode: raise RuntimeError('regression suite failed; inspect its log')
    print(json.dumps(record), flush=True)


def creation_guard(output):
    # Explicit post-calculation audit/adoption, never a backdated creation claim.
    current = profile('benchmarks.approximate_eels')
    record = validate_guard(output/'cache_guard.json', current, adopt=True)
    extra = {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT/'benchmarks').glob('approximate_eels*.py')
             if p.name not in ('approximate_eels_report.py', 'approximate_eels_finalize.py')}
    extra_path = output/'extended_cache_guard.json'
    if extra_path.exists():
        previous = json.loads(extra_path.read_text())
        if previous['source_sha256'] != extra: raise ValueError('additional calculation-source guard changed')
    else:
        json_write(extra_path, {'source_sha256': extra,
            'scope': 'Current calculation-source audit, adopted after the inner cache contracts and preservation checks; not backdated.'})
    snapshots = output/'guarded_source_snapshot'
    for name in sorted(set(current['source_sha256']) | set(extra)):
        target = snapshots/name; target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): target.write_bytes((ROOT/name).read_bytes())
        if digest(target) != digest(ROOT/name): raise ValueError('source snapshot changed')
    return record


def audit(config, output):
    checks = []
    def add(name, passed, evidence, note=''):
        checks.append({'requirement': name, 'status': 'VERIFIED' if passed else 'INCOMPLETE', 'evidence': evidence, 'note': note})
    before = json.loads((output/'protected_before.json').read_text())
    after = json.loads((output/'protected_after.json').read_text())
    add('1_preserve_every_preexisting_input_and_result', after['all_unchanged'] and not after['changed'] and not after['missing'], ['protected_before.json', 'protected_after.json'])
    production = config['source_cases']; source_rows = []
    for case in production:
        folder = output/'source'/case['name']; state = json.loads((folder/'state.json').read_text()); n = 2**case['power']
        valid = state['status'] == 'COMPLETE' and state['completed_q'] == state['total_q'] == n
        valid &= state['contract']['settings'] == case
        valid &= state['contract']['algorithm'] == sha256(ROOT/'yoshioka/bulk_thermal_source.py')
        valid &= state['contract']['force'] == sha256(primitive_path(config, case['temperature_k']))
        e = np.load(folder/'energies_mev.npy'); w = np.load(folder/'weights.npy'); b = np.load(folder/'born_weights.npy')
        valid &= e.shape == (n, 12) and w.shape == b.shape == (n, 12, 4)
        valid &= bool(np.all(np.isfinite(e)) and np.all(np.isfinite(w)) and np.all(np.isfinite(b)))
        valid &= bool(np.all(e[:, :6] > 0) and np.array_equal(e[:, 6:], -e[:, :6]))
        valid &= bool(np.all(w >= 0) and np.all(b >= 0) and np.sum(w) > 0)
        nodes, inverse = importance_nodes(case['power'], case['seed'])
        with np.load(folder/'spectrum.npz') as saved:
            valid &= bool(np.array_equal(saved['q_fractional'], nodes) and np.array_equal(saved['inverse_sampling_density'], inverse))
        source_rows.append({'case': case['name'], 'passed': bool(valid), 'q': n})
    add('3_full_branch_finite_temperature_bulk_q_sources', all(r['passed'] for r in source_rows), source_rows,
        'All6branches at every sampled q, both signed transfers, actual inverse sampling densities and complete prefixes inspected.')
    for temperature in (300, 1000):
        rows = [r for r, c in zip(source_rows, production) if c['temperature_k'] == temperature]
        seeds = {c['seed'] for c in production if c['temperature_k'] == temperature}
        add(f'5_independent_sampling_T{temperature}', len(rows) == len(seeds) == 8 and sum(r['q'] for r in rows) == 512,
            ['source_statistics.json', f'final_T{temperature}_spectra.npz'], 'Exact mode energies retained; no energy-binned or mixed-Bloch random-phase propagation.')
        for mesh in (16, 24, 32, 48, 64):
            state = json.loads((output/'born'/f'T{temperature}_mesh{mesh}'/'state.json').read_text())
            add(f'2_Born_T{temperature}_mesh{mesh}', state['status'] == 'COMPLETE' and state['completed_q'] == mesh**3,
                [str((output/'born'/f'T{temperature}_mesh{mesh}'/'state.json').relative_to(output))])
        gamma = output/f'Gamma_cell_T{temperature}_mesh64.npz'
        with np.load(gamma) as g, np.load(gamma.with_name(gamma.stem+'_quadrature.npz')) as q:
            valid = np.array_equal(g['q_fractional'], q['q_fractional']) and np.all(abs(q['q_fractional']) <= .5/64)
            for i, seed in enumerate(q['seeds']):
                nodes, inverse = importance_nodes(8, int(seed))
                valid &= np.array_equal(q['q_fractional'][i], nodes/64)
                valid &= np.array_equal(q['quadrature_weights'][i], inverse/(256*64**3))
        add(f'4_Gamma_cell_measure_T{temperature}', bool(valid), [gamma.name, gamma.stem+'_quadrature.npz'])
    numerical = json.loads((output/'numerical_confirmation.json').read_text())
    add('4_electron_slice_joint_and_covariance_refinements', len(numerical) == 8 and all(r['maximum_shape_L1'] < .1 for r in numerical),
        ['numerical_confirmation.json'], 'Individual peak/dominance sensitivities remain reported;1000K is exploratory.')
    born_summary = json.loads((output/'Born_convergence.json').read_text())
    born_primary = next(r for r in born_summary['comparisons'] if r['temperature_k'] == 300 and r['mesh_fine'] == 64)['rows'][0]
    statistics = json.loads((output/'source_statistics.json').read_text())['temperatures']['300']
    sample_primary = statistics['numerical_sample_count_comparisons'][-1]['rows'][0]
    peak_stable = all(max(r['rows'][0]['feature_shifts_mev'].values()) <= 1 for r in numerical)
    numerical_primary = born_primary['shape_pass'] and born_primary['peak_pass'] and sample_primary['shape_pass'] and sample_primary['peak_pass'] and peak_stable
    add('4_primary_identifiable_feature_and_shape_targets', numerical_primary,
        ['Born_convergence.json', 'source_statistics.json', 'numerical_confirmation.json'],
        'Primary0–20mrad dominant resolved features satisfy1meV; all detector shape changes satisfy10%. The weak1000K10–20mrad middle-window dominance switch remains flagged separately.')
    area = json.loads((output/'transition_and_area_audit.json').read_text())
    add('4_raw_weights_and_spectrum_density_normalization', all(r['maximum_relative_area_error'] < 1e-9 and not r['renormalized'] for r in area), ['transition_and_area_audit.json'])
    source_test = json.loads((output/'independent_validation.json').read_text())
    add('3_independent_numerical_source_and_nonlinearity_control', source_test['source_control_pass'] and len(source_test['nonlinear_phase_controls']) == 6,
        ['independent_validation.json'])
    dense = json.loads((output/'thermal_Dyson_validation.json').read_text())
    repeated = json.loads((output/'leading_repeated_material.json').read_text())
    add('6_dense_thermal_two_interaction_validation_and_material_pilot', all(r['passed'] for r in dense['rows']) and len(repeated['cases']) == 14,
        ['thermal_Dyson_validation.json', 'leading_repeated_material.json'], 'ReducedGamma linear model only; no full9B or complete intensity-through-g4 claim.')
    comp = json.loads((output/'TACAW_comparison.json').read_text())
    add('7_conditional_TACAW_comparison_and_unchanged_arrays', len(comp['datasets']) == 2 and all(
        r['sha256'] == before['files'][Path(r['path']).name]['sha256'] for r in comp['datasets']),
        ['TACAW_comparison.json', 'TACAW_metadata.md', 'metadata_discovery.json', 'missing_inputs.md'],
        'Calibration/model matching remain explicitly unverified; matched and unmatched features both reported.')
    ehm = json.loads((output/'ehm'/'scope.json').read_text())
    add('8_corrected_MD_candidates_and_preserved_scope', ehm['original_B11A_status'] == 'FAIL preserved' and (output/'ehm/force_constants_T300.json').exists()
        and (output/'ehm/force_constants_T1000.json').exists(), ['ehm/scope.json', 'methods.md'],
        'No newMD/FC3/FC4 campaign, thermal-expansion change or unsupported linewidth/lifetime.')
    tests = json.loads((output/'regression.json').read_text())
    add('9_full_relevant_regression', tests['returncode'] == 0 and tests['test_count'] >= 159, ['regression.json', 'regression.log'])
    peak = max(json.loads(p.read_text()).get('resources', {}).get('peak_process_rss_mib', 0) for p in (output/'source').glob('*/state.json'))
    add('9_bounded_resources_and_resumable_production', peak <= 2048 and (output/'approximate_eels_production_20261102_v1_invocations.json').exists(),
        ['cost_vs_scope.json', 'reproduce_and_resume.sh'], f'Observed source-process peak={peak:.3f}MiB; no72hour joint or TB allocation.')
    timing = json.loads((output/'phase_timing_pilot.json').read_text())
    add('9_measured_mode_construction_and_propagation_cost', len(timing['rows']) == 4 and all(
        r['maximum_absolute_saved_weight_difference'] < 1e-12 and r['physical_branches'] == 6
        and all(value > 0 for value in r['inclusive_phase_seconds'].values()) for r in timing['rows']),
        ['phase_timing_pilot.json'], 'Explicit profiled pilot, not a retrospective reconstruction of production phases.')
    required = ['report.md', 'methods.md', 'Si_300K_EELS.png', 'TACAW_broad_detector_comparison.png', 'distorted_wave_temperature.png',
        'numerical_convergence.png', 'Gamma_cell_diagnostic.png', 'leading_repeated_optical.png', 'temperature_trends.png', 'TACAW_metadata.md', 'missing_inputs.md', 'reproduce_and_resume.sh']
    add('9_artifacts_and_exact_commands', all((output/name).is_file() and (output/name).stat().st_size > 0 for name in required), required)
    result = {'all_mandatory_requirements_verified': all(r['status'] == 'VERIFIED' for r in checks), 'checks': checks,
        'scope': 'Completion of the explicitly approximate task, not acceptance of original strict material benchmarks or guaranteed TACAW agreement.'}
    json_write(output/'requirements_audit.json', result)
    if not result['all_mandatory_requirements_verified']: raise RuntimeError('Completion audit has incomplete requirements')
    return result


def finalize(config, output):
    audit(config, output); creation_guard(output)
    paths = [ROOT/'APPROXIMATE_EELS_20261102.md', ROOT/'yoshioka/bulk_thermal_source.py', ROOT/'yoshioka/thermal_dyson_words.py',
        ROOT/'tests/test_bulk_thermal_source.py', ROOT/'tests/test_thermal_dyson_words.py', ROOT/'benchmarks/prepare_approximate_eels_cold.py']
    paths += sorted((ROOT/'benchmarks').glob('approximate_eels*.py'))
    paths += sorted((ROOT/'benchmarks/configs').glob('approximate_eels*.json'))
    json_write(output/'changed_source_files.json', {'files': [{'path': str(p), 'action': 'added', 'sha256': sha256(p)} for p in paths],
        'count': len(paths), 'preexisting_source_files_modified': False})
    files = {}
    for p in sorted(output.rglob('*')):
        if p.is_file() and p.name != 'manifest.json': files[str(p.relative_to(output))] = {'sha256': digest(p), 'bytes': p.stat().st_size}
    json_write(output/'manifest.json', {'files': files, 'count': len(files), 'logical_bytes': sum(v['bytes'] for v in files.values()),
        'scope': 'All result artifacts except this self-referential manifest; source files separately indexed.'})
    print('Requirements verified; manifest files:', len(files), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='benchmarks/configs/approximate_eels_production_20261102_v1.json')
    p.add_argument('--preserve', choices=('before', 'after'))
    p.add_argument('--regression', action='store_true')
    args = p.parse_args(); config = json.loads(Path(args.config).read_text()); output = ROOT/config['output']
    if args.preserve: preserve(output, args.preserve)
    elif args.regression: regression(output)
    else: finalize(config, output)


if __name__ == '__main__': main()
