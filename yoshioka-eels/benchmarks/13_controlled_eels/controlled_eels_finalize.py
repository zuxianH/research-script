"""Completion, regression, and streamed integrity audits of controlled EELS."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from .controlled_eels import DEFAULT, PREVIOUS, JOBS, physical_guard, check_source_contract, validate_covariances, setup
from .controlled_eels_run import progress
from .approximate_eels_finalize import preserve, regression
from .workflow_io import ROOT, json_write, sha256


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''): value.update(chunk)
    return value.hexdigest()


def documents(output):
    if (output/'manifest.json').exists(): raise ValueError('Preserve archived documents')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('missing_inputs.md', 'TACAW_metadata.md', 'TACAW_metadata.json'):
        (output/name).write_bytes((PREVIOUS/name).read_bytes())
    relative = str(output.relative_to(ROOT)); quoted = shlex.quote(relative)
    resume = ['#!/usr/bin/env bash', 'set -euo pipefail', f'cd {shlex.quote(str(ROOT))}',
        'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1',
        '# Complete prefixes are audited without propagation or history changes.',
        'for job in '+ ' '.join(JOBS)+'; do',
        f'  .venv/bin/python -m benchmarks.controlled_eels_run --output {quoted} --job "$job" --stages born,source,refine,covtest',
        f'  .venv/bin/python -m benchmarks.controlled_eels_validate --output {quoted} --job "$job"',
        'done',
        '# Explicit report regeneration before archival:',
        f'# .venv/bin/python -m benchmarks.controlled_eels_report --output {quoted}',
        '# Full regression, protected-file checks and archival audit:',
        f'# .venv/bin/python -m benchmarks.controlled_eels_finalize --output {quoted} --stage regression',
        f'# .venv/bin/python -m benchmarks.controlled_eels_finalize --output {quoted} --stage after',
        f'# .venv/bin/python -m benchmarks.controlled_eels_finalize --output {quoted} --stage audit']
    (output/'reproduce_and_resume.sh').write_text('\n'.join(resume)+'\n')
    cold = ['#!/usr/bin/env bash', 'set -euo pipefail', f'cd {shlex.quote(str(ROOT))}',
        'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1',
        'controlled_output=${1:?Provide a NEW output path beneath the project root}',
        '.venv/bin/python -m benchmarks.controlled_eels_finalize --output "$controlled_output" --stage prepare',
        'for job in '+ ' '.join(JOBS)+'; do',
        '  .venv/bin/python -m benchmarks.controlled_eels_run --output "$controlled_output" --job "$job"',
        '  .venv/bin/python -m benchmarks.controlled_eels_validate --output "$controlled_output" --job "$job"', 'done',
        '# Preserve the initial64q evidence, then repeat the justified affected-model refinement.',
        '.venv/bin/python -m benchmarks.controlled_eels_report --output "$controlled_output"',
        '.venv/bin/python -m benchmarks.controlled_eels_sampling --output "$controlled_output" --job fitted_r8_T300 --power 7',
        '.venv/bin/python -m benchmarks.controlled_eels_run --output "$controlled_output" --job fitted_r8_T300 --stages source',
        '.venv/bin/python -m benchmarks.controlled_eels_finalize --output "$controlled_output" --stage regression',
        '.venv/bin/python -m benchmarks.controlled_eels_finalize --output "$controlled_output" --stage after',
        '.venv/bin/python -m benchmarks.controlled_eels_finalize --output "$controlled_output" --stage resume',
        '.venv/bin/python -m benchmarks.controlled_eels_report --output "$controlled_output"',
        '.venv/bin/python -m benchmarks.controlled_eels_finalize --output "$controlled_output" --stage audit']
    (output/'calculate_new.sh').write_text('\n'.join(cold)+'\n')
    (output/'methods.md').write_text('''# Controlled four-model EELS methods

All numerical source/propagation and quadrature kernels are reused unchanged from the preceding approximate workflow. calculation_guard.json snapshots their actual code, CODATA constants, Lobato table and package versions. Mechanical quantities are SI; scattering wavevectors are angular inverse metres, with no extra2pi in the phase. The previous constants_and_physics.json is hash-referenced from environment.json. New orchestration checks force, covariance, geometry and committed prefixes before reuse. reuse_provenance.json records independently copied historical data; no writable hard links are used.

Two phonon models (fixed harmonic; independently temperature-fitted harmonic) are crossed with two electron treatments (first Born; elastic distorted waves with one linear inelastic source). The fixed-volume Si[110] specimens have four/eight oriented repeats and one transverse periodic cell. The primitive bulk measure is normalized d^3q/Omega_BZ, or unit-volume primitive fractional coordinates. All six branches remain; only the three exact-Gamma rigid translations are excluded.

Every signed phonon sector has its actual frequency and outgoing electron Bloch shift. Emission uses the conjugate displacement coefficient and -q_perp modulo an electron reciprocal vector; absorption is its adjoint with opposite shift. Thermal occupations nB+1/nB multiply probabilities. The covariance is integrated from the selected force model and Bose statistics. Its mean elastic potential and displacement gradient include Debye-Waller attenuation once. Changed covariance and specimen geometry have distinct background/incident caches. Depth amplitudes interfere coherently within each final channel; probabilities add across orthogonal bath channels.

The control variate is dense Born plus mean[(distorted(q)-Born(q))/p(q)]. Eight independent scrambled Sobol batches initially contain64q sectors. The affected fitted eight-repeat300K model is additionally refined to128q/batch, using identical nested prefixes and preserving its64q evidence. The proposal is half uniform BZ and half Gamma radial importance sampling, with explicit inverse mixture densities. A square-root weight enters the displacement and1/N enters its final probability average once. Prefix imports retain raw integrands without1/N rescaling. The Born Gamma cell is replaced once by eight independent256-node importance batches, retaining actual matrix elements, detector and specimen phases.

The estimator retains deterministic Born quadrature bias. Its statistical bands do not cover that bias or phonon-model error. Corresponding model/thickness calculations share q nodes and source seeds; when final counts differ, their common nested prefix is explicitly verified. Gamma nodes also match at the same Born mesh but have seeds independent of the sources. Paired uncertainty differences the eight matched batch-integral estimates before taking variances, preserving cross-model covariance even when per-batch q counts differ. Shared baseline/Gamma contributions cancel in distorted-minus-Born uncertainty. Pointwise Student-t intervals and500 batch bootstraps describe finite-eight-batch sensitivity, not guaranteed simultaneous coverage.

Separate checks cover Born meshes, covariance16/24/32/64, source counts16/32/64, transverse49/65, longitudinal385/769 or545/1089, and joint refinement. Matched-subset propagation tests estimate numerical sensitivity; they are not full fine-grid production calculations. The target remains1meV for identifiable dominant features and0.1 conditional shape L1 over5-90meV. Undefined/competing maxima are reported for every physical detector. Thickness is a physical change, not a numerical convergence parameter.

Relative integrated-weight changes use |second-first|/|second|; second is the fine result for numerical comparisons. The shape L1 integrates the difference between curves independently conditioned on the same5-90meV window. Raw model weights are also reported separately. The600s resource check occurs between committed work units; an in-flight q calculation and final bookkeeping can extend invocation wall time slightly beyond600s.

All raw signed transitions are saved. Gaussian display broadening stays2meV FWHM; deposition0.0025meV and output0.05meV. Spectra have units model inelastic weight/meV and are integrated against raw retained weights. Raw spectra are not clipped or renormalized. Only the explicitly conditional5-90meV shape comparison divides by each curve's own window integral. Physical detectors remain0-20,0-5,5-10 and10-20mrad. Raw transition windows0-40,40-65,65-90meV remain distinct from feature-search windows10-40,40-60,60-85meV.

Corrected classical-MD force fits are reused without tuning to EELS; their actual MD temperatures remain in B11_refit. Bose statistics are used for the harmonic response. This combination does not establish a self-consistent quantum anharmonic phonon model. The original effective-harmonic FAIL and unresolved linewidth statuses remain. No thermal expansion, FC3/FC4, repeated inelastic source events, direct nonlinear multiphonon vertices or fitted intrinsic lifetimes are added.

TACAW reductions and metadata are reused with their verified historical identities. The generating7.5fs energy calibration, native resolution, physical angular acceptance and geometry remain unverified. No second Kubo correction, energy shift, fitted scale or claim of absolute/matched-model validation is introduced.
''')
    for filename in ('calculate_new.sh', 'reproduce_and_resume.sh'):
        subprocess.run(['bash', '-n', str(output/filename)], check=True)
    print('Reproduction and methods written:', output)


def verify_resume(output):
    before = {str(p.relative_to(output)): (sha256(p), p.stat().st_mtime_ns) for p in output.glob('*/source/*/state.json')}
    before.update({str(p.relative_to(output)): (sha256(p), p.stat().st_mtime_ns) for p in output.glob('*/born/*/state.json')})
    start = time.monotonic()
    with (output/'resume_verification.log').open('w') as log:
        result = subprocess.run(['bash', str(output/'reproduce_and_resume.sh')], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    unchanged = all((sha256(output/name), (output/name).stat().st_mtime_ns) == value for name, value in before.items())
    record = {'returncode': result.returncode, 'seconds': time.monotonic()-start, 'all_prefixes_unchanged': unchanged,
        'prefixes_checked': len(before), 'scope': 'Exact resume script; physical-state hashes and mtimes must remain unchanged.'}
    json_write(output/'resume_verification.json', record)
    if result.returncode or not unchanged: raise RuntimeError('Resume verification failed')
    print(json.dumps(record), flush=True)


def audit(output):
    physical_guard(output); checks = []
    def add(name, passed, evidence, scope=''):
        checks.append({'requirement': name, 'verified': bool(passed), 'evidence': evidence, 'scope': scope})
    comparison = json.loads((output/'comparison.json').read_text())
    add('four_models_at_both_temperatures_and_300K_thickness', comparison['status'] == 'COMPUTED' and len(comparison['curves']) == 12,
        ['comparison.json', 'four_models_T300.png', 'four_models_T1000.png', 'thickness_T300.png'])
    for job in JOBS:
        config = json.loads((output/'configs'/f'{job}.json').read_text()); folder = ROOT/config['output']
        validate_covariances(config, folder)
        for stage in ('born', 'source', 'refine', 'covtest'):
            state = progress(output, job, stage)
            add(job+'/'+stage, state['complete'], state, 'All declared prefixes required, not merely a successful process exit.')
        for case in config['source_cases']:
            add(job+'/'+case['name']+'/physical_contract', check_source_contract(config, case, folder) == 2**case['power'],
                [str((folder/'source'/case['name']/'state.json').relative_to(output))])
            state = json.loads((folder/'source'/case['name']/'state.json').read_text())
            if 'prefix_import' in state:
                imported = state['prefix_import']; original = Path(imported['source'])
                exact = True
                for key, expected in imported['source_array_sha256'].items():
                    old = np.load(original/(key+'.npy')); new = np.load(folder/'source'/case['name']/(key+'.npy'), mmap_mode='r')
                    exact &= sha256(original/(key+'.npy')) == expected and np.array_equal(old, new[:len(old)])
                add(job+'/'+case['name']+'/nested_prefix_integrity', exact, [str(original)],
                    'Original integrands preserved bitwise; final1/N applies only when forming the refined spectrum.')
        control = json.loads((output/(job+'_independent_source.json')).read_text())
        add(job+'/independent_complex_amplitudes', control['passed'] and len(control['rows']) == 9,
            [job+'_independent_source.json'], 'Actual force model, generic/small-q/weak-sector points, both signs, no phase alignment.')
        pilot = json.loads((folder/'pilot.json').read_text())
        add(job+'/measured_affordable_pilot', pilot['after_q'] >= 4 and pilot['seconds_per_q'] > 0,
            [job+'/pilot.json'])
    numerical = json.loads((output/'finest_numerical_checks.json').read_text())
    add('separate_complete_numerical_diagnostics', len(numerical) == 24, ['finest_numerical_checks.json'],
        'Scientific peak/shape gates are reported separately, including every unresolved or sensitive detector window; this audit does not turn a failed observable into PASS.')
    acceptance = comparison['finest_acceptance']
    add('primary_shapes_and_identifiable_features_reach_targets', acceptance['primary_all_checks_pass']
        and acceptance['all_detector_shape_checks_pass'] and acceptance['all_identifiable_feature_pairs_within_1mev'],
        ['comparison.json', 'finest_numerical_checks.json'], 'Unidentified acoustic windows retain their original failed/undefined peak gates.')
    weak = json.loads((output/'weak_feature_diagnosis.json').read_text())['rows']
    add('weak_feature_and_refinement_evidence_preserved', len(weak) == 6
        and all(r['resolved_replicates']+r['unresolved_replicates'] == 500
                and sum(r['resolved_peak_bins_0_1mev'].values()) == r['resolved_replicates'] for r in weak)
        and (output/'source64_evidence/comparison.json').exists(),
        ['weak_feature_diagnosis.json', 'weak_feature_diagnosis.png', 'source64_evidence/comparison.json'])
    add('raw_area_preserved', all(r['passed'] and not r['renormalized'] for r in comparison['area_audits']), ['comparison.json'])
    add('paired_common_q_and_independent_uncertainties', len(comparison['physical_comparisons']) == 8 and len(comparison['uncertainty']) == 6,
        ['comparison.json', 'paired_differences.png'], 'Eight source scrambles and eight independent Gamma scrambles; shape/peak/weight/gain-loss sensitivity retained.')
    tests = json.loads((output/'regression.json').read_text())
    add('full_regression', tests['returncode'] == 0 and tests['test_count'] >= 169, ['regression.json', 'regression.log'])
    after = json.loads((output/'protected_after.json').read_text())
    add('all_preexisting_files_unchanged', after['all_unchanged'], ['protected_before.json', 'protected_after.json'])
    costs = json.loads((output/'cost_vs_scope.json').read_text())
    add('bounded_recorded_resources', costs['maximum_recorded_process_rss_mib'] <= 2048, ['cost_vs_scope.json'])
    resume = json.loads((output/'resume_verification.json').read_text())
    add('exact_resume_without_repropagation', resume['returncode'] == 0 and resume['all_prefixes_unchanged'], ['reproduce_and_resume.sh', 'resume_verification.json'])
    add('original_physical_model_and_reference_limits_preserved', comparison['original_B11A_status'] == 'FAIL preserved'
        and comparison['intrinsic_linewidths'] == 'NOT_RESOLVED preserved', ['report.md', 'TACAW_metadata.md', 'missing_inputs.md'])
    status = {'task_deliverables_complete': all(row['verified'] for row in checks), 'checks': checks,
        'scientific_numerical_summary': numerical,
        'scope': 'Completion of the requested controlled approximate comparison; not new strict material benchmark or calibrated TACAW acceptance.'}
    json_write(output/'requirements_audit.json', status)
    if not status['task_deliverables_complete']: raise RuntimeError('Missing task deliverable; inspect requirements_audit.json')
    paths = sorted((ROOT/'benchmarks').glob('controlled_eels*.py'))+[ROOT/'tests/test_controlled_eels.py']
    json_write(output/'changed_source_files.json', {'files': [{'path': str(p.relative_to(ROOT)), 'action': 'added', 'sha256': sha256(p)} for p in paths],
        'count': len(paths), 'preexisting_files_modified': False, 'configs': [str(p.relative_to(ROOT)) for p in sorted((output/'configs').glob('*.json'))]})
    snapshots = output/'implementation_snapshot'
    for source in paths:
        p = snapshots/source.relative_to(ROOT); p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and digest(p) != digest(source): raise ValueError('Implementation snapshot changed')
        if not p.exists(): p.write_bytes(source.read_bytes())
    files = {str(p.relative_to(output)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
        for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    json_write(output/'manifest.json', {'files': files, 'count': len(files), 'logical_bytes': sum(r['bytes'] for r in files.values())})
    print('All task deliverables verified; result files:', len(files), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', default=DEFAULT)
    p.add_argument('--stage', choices=('audit', 'regression', 'before', 'after', 'prepare', 'commands', 'resume'), default='audit')
    args = p.parse_args(); output = ROOT/args.output
    if args.stage in ('before', 'after'): preserve(output, args.stage)
    elif args.stage == 'regression': regression(output)
    elif args.stage == 'prepare':
        output.mkdir(parents=True, exist_ok=False); preserve(output, 'before'); setup(output); documents(output)
    elif args.stage == 'commands': documents(output)
    elif args.stage == 'resume': verify_resume(output)
    else: audit(output)


if __name__ == '__main__': main()
