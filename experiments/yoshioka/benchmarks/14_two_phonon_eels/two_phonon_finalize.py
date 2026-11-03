"""Standalone methods, measured-cost report and immutable reproduction audit."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time
import math

from .two_phonon_eels import DEFAULT,PREVIOUS,CONTROLLED
from .workflow_io import ROOT,json_write,sha256
from .approximate_eels_finalize import preserve,regression,digest


JOBS=[('main',8,[]),('importance',8,['--proposal-kind','born']),
      ('electron',4,['--grid','65']),('slices',4,['--steps','769']),
      ('joint',4,['--grid','65','--steps','769']),
      ('covariance',4,['--covariance-mesh','64']),('lowq',5,['--radius','.2'])]


def documents(output):
    result=json.loads((output/'results.json').read_text());final=result['final']
    convergence=json.loads((output/'pair_convergence.json').read_text())
    rows=[];invocations=[]
    for job,_,_ in JOBS:
        states=[json.loads(p.read_text()) for p in sorted((output/'pairs'/job).glob('*/state.json'))]
        rows.append(dict(job=job,completed_per_batch=[s['completed_pairs'] for s in states],
            committed_pair_work_seconds=sum(s['cumulative_seconds'] for s in states),
            recorded_targets=[s.get('target_pairs_latest_request') for s in states]))
    for p in sorted((output/'invocations').glob('*.json')):
        invocations.append({'record':str(p.relative_to(output)),**json.loads(p.read_text())})
    maximum_recorded_peak=max(r['resources']['peak_process_rss_mib'] for r in invocations)
    costs=dict(jobs=rows,recorded_invocations=invocations,
        cumulative_committed_pair_work_seconds=sum(r['committed_pair_work_seconds'] for r in rows),
        maximum_recorded_calculation_process_peak_rss_mib=maximum_recorded_peak,
        maximum_recorded_invocation_wall_seconds=max(r['resources']['elapsed_seconds'] for r in invocations),
        scope='Sum of committed per-pair wall work across concurrent processes, not elapsed task time or CPU time. Setup, source construction and propagation are recorded separately. Interrupted uniform extensions lack final OS peak records; their complete pair work is retained in states.',
        resource_limits='600s bounded invocations, checks during propagation,2GiB process RSS target. Timeout overshoot is one checked work interval plus cleanup.',
        pilot_budget_note='The~90min serial-pilot estimate underestimated the sum of concurrent wall work. Actual measured costs are reported; no scope-matched TACAW runtime or speedup is claimed.')
    json_write(output/'cost_vs_scope.json',costs)
    for name in ('TACAW_metadata.json','TACAW_metadata.md','missing_inputs.md'):
        source=CONTROLLED/name;target=output/name
        if target.exists() and digest(target)!=digest(source):raise ValueError('copied reference metadata changed')
        if not target.exists():shutil.copy2(source,target)
    diagnostic=PREVIOUS/'independent_validation.json'
    if not (output/'nonlinear_omission_diagnostics.json').exists():
        json_write(output/'nonlinear_omission_diagnostics.json',{'source':str(diagnostic),'sha256':sha256(diagnostic),
            'record':json.loads(diagnostic.read_text()),'interpretation':'Existing nonlinear oscillator controls, not a material omission bound. Intermediate dynamical transfers can exceed the Born final-transfer diagnostic.'})
    if not (output/'additional_cost_forecast.json').exists():
        timing=next(r['committed_pair_work_seconds'] for r in rows if r['job']==result['selected_job'])/final['total_pairs']
        primary=final['detectors'][0]['raw']
        ratios={'integrated_loss_10percent':primary['all_loss']['relative_halfwidth95']/.1,
                'two_creation_10percent':primary['creation_creation']['relative_halfwidth95']/.1,
                'shape_L1_10percent':result['last_pair_comparison'][0]['windows']['5_160']['shape_L1']/.1}
        estimates={}
        for name,ratio in ratios.items():
            n=8*2**math.ceil(math.log2(max(final['total_pairs'],final['total_pairs']*ratio**2)/8))
            estimates[name]={'rough_required_total_pairs':n,'pairs_per_batch':n//8,
                'additional_pairs':n-final['total_pairs'],'additional_summed_worker_seconds_at_observed_rate':(n-final['total_pairs'])*timing,
                'rough_elapsed_seconds_at_same_four_worker_throughput':(n-final['total_pairs'])*timing/4}
        json_write(output/'additional_cost_forecast.json',{'observed_seconds_per_pair':timing,'forecasts':estimates,
            'assumptions':'Preliminary N^-1/2 extrapolation, no peak-stability guarantee; rare-region/QMC and hardware effects can change it.'})
    methods=r'''# Leading two-interaction silicon EELS

This is the fixed-harmonic,300K,60keV,Si[110] four-repeat specimen,1.535553098nm,16 explicit atoms,12 angstrom vacuum at each end. The transverse cell is unchanged. Primary49x49/385 steps and independent65x65/385,49x49/769,65x65/769 checks refine numerical sampling, not the specimen. Full-branch covariance32^3 is checked against64^3. All six primitive branches and both Bose signs are retained. Mechanical quantities are SI. Q and phonon q are angular inverse metres; no extra2pi enters a displacement phase. One angstrom is1e-10metres; inverse angstrom is1e10 inverse metres; meV is1e-3 times the elementary charge in joules; amu is the recorded atomic-mass constant in kg. Angular omega=2pi*f, with ordinary frequency f in Hz. Constants and the actual masses/cells/Lobato parameters are copied with hashes in reuse_provenance.json.

The original thermal_dyson_words solver propagates21 complex waves for two selected modes: order0, four signed single words and16 ordered double words. The elastic incident wave is propagated again inside this triangular action; its independently cached depth wave checks the one-word coefficients. The immutable mean elastic field is reused. No full mode-pair Hamiltonian or macroscopic finite-temperature occupation basis is stored.

For mode lambda=(q,nu) and signed letter s, H_s includes the actual signed hbar*omega and electron kinetic energy in its outgoing Bloch sector. A word has summed energy and summed Bloch shifts. Sources are streamed. Local one-insertion sinc coefficients and8x8 ordered two-insertion quadrature are reused unchanged. Independent dense Fock propagation checks this local/order construction; longitudinal variation and kinetic splitting are checked separately.

For distinct final modes, both depth orderings reach the same bath state. Their amplitudes A(lambda,mu)+A(mu,lambda) interfere before squaring. Orthogonal final bath states add as probabilities. Each signed pair has factor f_lambda(s)f_mu(t), with f(+)=nB+1,f(-)=nB. The continuous response integrates this squared amplitude over normalized primitive dq*dq', sums branches and all four sign pairs, and multiplies by1/2 to count each unordered final state once. Mixed signs exchange a phonon between modes and can have nonzero energy while preserving total phonon number.

Sources have unit BZ weight. Each pair's exact intensity is multiplied once by1/[2 p(q,nu)p(q',mu)]. Sample averaging supplies1/N once. Each marginal one-mode/return integral uses inverse joint density and the average of the two marginal estimates. No detector-integrated one-phonon weight is squared, no Poisson rule is assumed, and no random-phase amplitude estimate is squared.

Finite discrete identical-mode terms are explicitly validated against the thermal Gram trace and a direct thermal occupation sum. Two identical creations have thermal factor2(nB+1)^2. Mixed identical-mode terms contribute the connected unchanged-bath variance nB(nB+1)|A(ann,create)+A(create,ann)|^2. In the normalized3D bulk continuum limit, diagonal double-integral terms scale as sum(w_q^2) and carry no separate point mass. The coherent single-integral return survives:

    R = integral sum_nu [(nB+1) A(ann,create) + nB A(create,ann)].

Its order2 elastic correction is2 Re(psi0* R). Its order4 square is estimated without noisy-amplitude bias using independent batch means:

    (|sum_b R_b|^2 - sum_b |R_b|^2)/(B(B-1)).

Finite-sample values can be signed. No clipping is used. Together with the connected two-mode response and the reused one-source spectrum, these terms form the squared amplitude retained through orders0+1+2 in the continuum harmonic bath. They do not provide every exact inclusive intensity term throughg^4: interference with amplitude orders3 and4 is missing.

The initial normalized proposal is half uniform BZ and half radial Gamma sampling, with six uniformly sampled branch labels per q. Radial density is1/(4pi R r^2),R=.35. Full support and both low-q variables remain. Diagnosed rare-pair variance motivated a separate proposal with fresh seeds:20% joint uniform,40% radial/uniform-branch,40% continuous cells from the existing64^3 single-Born joint q/branch weights. Cell masses are normalized only to construct a PDF. The exact propagated integrand is not replaced by Born products. Dynamical matrices are recomputed at every actual continuous node. The actual mixture density corrects both q and branch sampling. The proposal cannot remove a branch or a low-q cell. A different radial radius tests low-q sensitivity separately.

The old uniform run is preserved through256pairs/batch. Its attempted512 request is explicitly stopped after a complete256 prefix to reallocate work; original request/status records and interrupt logs remain. The new importance production has eight independent fresh scrambles, nested through256pairs/batch. Both schemes remain separately reported. Count ceilings and unchanged tolerances are recorded before their respective extensions.

Nominal Student-t95 intervals use eight independent batch integrals, not individual Sobol points. They are not simultaneous confidence bands and exclude discretization and harmonic-model errors. The fresh importance scrambles are independent of the pilot used to choose their PDF. Five hundred batch bootstraps expose competing feature maxima; spectral features are not identified phonon modes. Peak windows/prominence and2meV FWHM are fixed. Conditional shapes use5-90 and newly declared5-160meV; raw signed line weights remain primary.

The energy axis[-180,180]meV covers actual sum/difference transitions and Gaussian tails. A normalized2meV FWHM Gaussian is applied once to each final transfer, with conservative.0025meV deposition and.05meV display sampling. Density units are model weight/meV. No cropped/truncated spectrum is renormalized. Genuine absorption is separated from Gaussian tails below zero. The full order012 spectrum additionally includes the unchanged-bath zero-energy line and its signed corrections.

This linear two-event model omits direct nonlinear multiphonon vertices, which can reach and interfere in the same two-phonon final states. An expansion in interaction count is distinct from an expansion in displacement. Reused nonlinear oscillator/Q.U.Q diagnostics are not a full-material bound. Original full9B stays INCOMPLETE,11A stays FAIL and intrinsic linewidths remain unresolved. Display FWHM is not a lifetime.

TACAW overlays retain the disclosed5-90meV shape conditioning and uncertain7.5fs generating-cadence assumption. They do not establish calibrated axes, native resolution, matched acceptance/geometry/force provenance, absolute intensities, a two-phonon mechanism or a speedup. No extra Kubo correction, fitted shift or fitted raw scale is applied.
'''
    (output/'methods.md').write_text(methods)
    command_prefix=f'.venv/bin/python -m benchmarks.two_phonon_eels --output {shlex.quote(str(output))} run'
    resume=['#!/usr/bin/env bash','set -euo pipefail',f'cd {shlex.quote(str(ROOT))}',
            'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1',
            '# Complete-prefix audit is immutable; incomplete invocations retain committed work.']
    for job,power,extra in JOBS:
        resume.append(command_prefix+' '+shlex.join(['--job',job,'--power',str(power),*extra]))
    (output/'reproduce_and_resume.sh').write_text('\n'.join(resume)+'\n')
    (output/'reproduce_and_resume.sh').chmod(0o755)
    cold=['#!/usr/bin/env bash','set -euo pipefail',f'cd {shlex.quote(str(ROOT))}',
        'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1',
        'output=${1:?Supply a new result directory}',
        'if test -e "$output"; then echo "Refusing to overwrite an existing directory" >&2; exit 1; fi',
        '.venv/bin/python - "$output" <<\'PY\'',
        'import sys', 'from pathlib import Path',
        'from benchmarks.approximate_eels_finalize import preserve',
        'p=Path(sys.argv[1]).resolve();p.mkdir(parents=True);preserve(p,"before")','PY',
        '.venv/bin/python -m benchmarks.two_phonon_eels --output "$output" setup',
        '.venv/bin/python -m benchmarks.two_phonon_eels --output "$output" importance-setup']
    for job,power,extra in JOBS:
        cold+=['for bounded_pass in $(seq 1 20); do',
            '  .venv/bin/python -m benchmarks.two_phonon_eels --output "$output" run '+shlex.join(['--job',job,'--power',str(power),*extra]),
            '  if .venv/bin/python - "$output" '+shlex.join([job,str(2**power)])+" <<'PY'",
            'import json,sys','from pathlib import Path',
            'p=Path(sys.argv[1])/"pairs"/sys.argv[2];n=int(sys.argv[3])',
            'rows=[json.loads(f.read_text())["completed_pairs"] for f in p.glob("*/state.json")]',
            'sys.exit(0 if len(rows)==8 and min(rows)>=n else 1)','PY','  then break; fi','done']
    cold+=['.venv/bin/python -m benchmarks.two_phonon_validation --output "$output"',
        '.venv/bin/python -m benchmarks.two_phonon_validation --stage material --output "$output"',
        '.venv/bin/python -m benchmarks.two_phonon_report --output "$output"',
        '.venv/bin/python -m benchmarks.two_phonon_finalize --output "$output" regression',
        '.venv/bin/python -m benchmarks.two_phonon_finalize --output "$output" documents',
        '.venv/bin/python -m benchmarks.two_phonon_finalize --output "$output" verify-resume',
        '.venv/bin/python -m benchmarks.two_phonon_finalize --output "$output" preserve-after',
        '.venv/bin/python -m benchmarks.two_phonon_finalize --output "$output" archive']
    (output/'calculate_new.sh').write_text('\n'.join(cold)+'\n');(output/'calculate_new.sh').chmod(0o755)
    first=final['detectors'][0];r=first['raw'];ratio=first['pair_loss_over_one_phonon_loss']
    text=['# Leading two-phonon silicon EELS: measured result','',
        f'Overall status: **{result["status"]}**. This is a linear two-insertion approximation, not full nonlinear Benchmark9B.',
        '', '| Component | Status |','|---|---|',
        f'| Solver/dense/thermal controls | {result["solver_implementation"]} |',
        f'| Primary electron/source numerical controls | {result["numerical_controls"]} |',
        f'| All-detector numerical peak controls | {result["numerical_all_detector_controls"]} |',
        f'| Integrated-loss sampling target | {result["statistical_acceptance"]} |',
        f'| Pair spectrum/feature convergence | {result["pair_integration"]} |',
        '| Full material9B / nonlinear vertices / higher orders | INCOMPLETE / omitted / omitted |',
        '| TACAW calibrated comparison | BLOCKED_BY_METADATA; diagnostic overlays available |','',
        f'The primary specimen is fixed harmonic Si[110],300K,60keV,1.535553nm,49x49 pixels and385 steps including vacuum. Every primitive branch is sampled. The selected production has {final["total_pairs"]} mode pairs in8 independent batches; source amplitudes retain actual frequencies and intermediate Bloch sectors. The initial uniform/radial result is separately preserved.',
        '', '| Process,0–20mrad | Raw model weight | Nominal95% half-width |', '|---|---:|---:|']
    for name in ('creation_creation','annihilation_annihilation','mixed_all_signed','all_loss','all_gain','loss_90_125','loss_125_160'):
        text.append(f'| {name} | {r[name]["mean"]:.8g} | {r[name]["halfwidth95"]:.5g} |')
    text+=['',f'The total positive-energy two-interaction weight is {100*ratio["ratio"]:.4g}% of the raw one-phonon loss weight, with nominal95% half-width {100*ratio["halfwidth95"]:.3g} percentage points. This includes mixed creation–annihilation losses. Pure two-creation weight is {100*r["creation_creation"]["mean"]/ratio["denominator_raw_loss"]:.4g}% of that same denominator. These are model-weight ratios, not absolute detector cross sections.',
        '',f'Final integrated-loss relative95% half-width: {100*r["all_loss"]["relative_halfwidth95"]:.4g}%. Pure two-creation relative half-width: {100*r["creation_creation"]["relative_halfwidth95"]:.4g}%. A resolved integral does not establish a resolved2meV spectrum or every weak-window maximum.',
        '', '| Integration scheme | Pairs | Loss weight | Relative95% half-width |','|---|---:|---:|---:|']
    for name,items in convergence.items():
        for row in items:
            value=row['primary_loss'];text.append(f'| {name} | {row["total_pairs"]} | {value["mean"]:.8g} | {100*value["relative_halfwidth95"]:.3g}% |')
    text+=['','The unchanged target is conditional shape L1<0.1 and identifiable dominant-peak shifts<=1meV. The final count comparison is:','',
        '| Detector | Shape L1,5–90 | Shape L1,5–160 | Largest identifiable maximum change(meV) |','|---|---:|---:|---:|']
    for row in result['last_pair_comparison']:
        shifts=[v for v in row['identifiable_feature_shifts_mev'].values() if v is not None]
        text.append(f'| {row["detector"]} | {row["windows"]["5_90"]["shape_L1"]:.5g} | {row["windows"]["5_160"]["shape_L1"]:.5g} | {max(shifts,default=0):.4g} |')
    text+=['','All detected/competing maxima, bootstrap spreads, regional contributions and raw window weights are retained in JSON; conditional TACAW matching also lists unmatched features. Do not interpret an exchanged strongest maximum as a measured phonon-mode shift. No noisy contribution is removed or renormalized.',
        '', '| Numerical sensitivity | Matching pairs | Primary loss change | Primary shape L1,5–160 |','|---|---:|---:|---:|']
    for row in result['numerical']:
        if row['status']=='INCOMPLETE':text.append(f'| {row["kind"]} | incomplete | — | — |');continue
        change=row['observables']['all_loss']['relative_difference_fine_denominator'][0]
        text.append(f'| {row["kind"]} | {row["matched_pairs"]} | {100*change:.5g}% | {row["detector_comparisons"][0]["windows"]["5_160"]["shape_L1"]:.5g} |')
    text+=['','Weak-detector numerical peak gates remain separate from the primary controls:','',
           '| Refinement | Detector | Feature | Dominant-maximum change(meV) |','|---|---|---|---:|']
    for row in result['numerical']:
        for peak_row in row.get('competing_peak_diagnosis',[]):
            text.append(f'| {row["kind"]} | {peak_row["detector"]} | {peak_row["feature"]} | {peak_row["dominant_shift_mev"]:.4g} |')
    text+=['','All detector shape controls pass, but these peak switches fail the unchanged1meV gate. Peak heights and near-tie diagnostics are retained in numerical_convergence.json. They are not evidence for a changed phonon frequency.']
    text+=['','These are fixed-pair full-support sensitivity controls, not complete fine-grid production or a rigorous BZ-wide bias bound. Low-q proposal changes are reported with paired-batch uncertainty in low_q_diagnostic.json. The acoustic Gamma cell is continuously sampled rather than removed.',
        '',f'Largest two-interaction broadened-area relative error: {final["maximum_broadened_area_relative_error"]:.4g}. The full order012 zero-loss line and separate signed elastic corrections are saved. The all-beam order2 balance residual is {final["maximum_all_beam_order2_balance"]:.4g}; no exact probability conservation is imposed on a truncated perturbation series.',
        '',f'Committed pair work summed across concurrent workers: {costs["cumulative_committed_pair_work_seconds"]:.3f}s. This is not end-to-end elapsed time. Maximum recorded calculation-process peak RSS: {maximum_recorded_peak:.3f}MiB. Latest bounded invocations, setup and per-pair field/propagation times are in cost_vs_scope.json. Intentional uniform-prefix stops and timeout/resume evidence are preserved.',
        '',f'Conditional TACAW5–90meV shape L1: one-source {result["conditional_TACAW"]["one_phonon_shape_L1"]:.5g}; limited-order sum {result["conditional_TACAW"]["limited_order_shape_L1"]:.5g}. These comparisons inherit unverified energy calibration, acceptance, native resolution, geometry and force provenance. A high-energy tail is not proof of a matched two-phonon mechanism.',
        '', 'The initial and improved proposals agree only within their measured integration uncertainty; the old noisy run is not treated as accepted reference truth. If the pair-shape or feature gates remain incomplete, retain that status. Improving the one-phonon-plus-small-correction curve does not establish convergence of the correction itself.',
        '', 'Direct nonlinear multiphonon vertices can reach the same final states and interfere. Their omission is not bounded by the present result. Existing nonlinear oscillator controls are indexed in nonlinear_omission_diagnostics.json. Benchmarks1–6,3B,8 and historical7/9A/9B/10/11 statuses remain unchanged.',
        '', 'Reproduce or resume with reproduce_and_resume.sh. calculate_new.sh requires a fresh directory and runs bounded passes. Regression: .venv/bin/python -m unittest discover -s tests. See methods.md, raw_array_index.json, changed_source_files.json, protected_before/after.json and manifest.json.',
        '', '![Leading two-interaction spectrum](two_phonon_EELS.png)',
        '', '![Sampling convergence](statistical_convergence.png)',
        '', '![Conditional TACAW comparison](conditional_TACAW_overlay.png)']
    forecasts=json.loads((output/'additional_cost_forecast.json').read_text())['forecasts']
    text+=['','Additional-cost forecasts use a preliminary inverse-square-root sample law, not a convergence guarantee:',
           '', '| Target | Estimated total pairs | Extra summed worker time | Approximate extra time with4 workers |',
           '|---|---:|---:|---:|']
    for name,entry in forecasts.items():
        text.append(f'| {name} | {entry["rough_required_total_pairs"]} | {entry["additional_summed_worker_seconds_at_observed_rate"]/3600:.3g}h | {entry["rough_elapsed_seconds_at_same_four_worker_throughput"]/60:.3g}min |')
    text+=['','A bounded next integration pilot could stratify the joint proposal by pair energy while retaining exact matrix-element/detector weights and full support. It must pass known-integral and thermal pair-counting tests and demonstrate variance reduction before a larger extension. A phonon DOS or product of detector-integrated one-phonon weights is not a replacement.',
           '', 'The broad extra continuum and high-energy model tail are computed, but their individual maxima remain underconverged. The integral is positive relative to nominal sampling uncertainty, yet misses the requested10% precision. This is useful limited-order evidence, not an accepted two-phonon spectrum.']
    if (output/'regression.json').exists():
        tests=json.loads((output/'regression.json').read_text())
        text+=['',f'Regression: {tests["test_count"]} tests, return code {tests["returncode"]}, {tests["seconds"]:.3f}s.']
    if (output/'protected_after.json').exists():
        preserved=json.loads((output/'protected_after.json').read_text())
        text+=['',f'Protected-file check: {preserved["count"]} files, all unchanged={preserved["all_unchanged"]}; both TACAW arrays remain unchanged.']
    (output/'report.md').write_text('\n'.join(text)+'\n')
    index=[]
    historical=[CONTROLLED/'fixed_r4_T300_spectra.npz',PREVIOUS/'born/T300_mesh64/energies_mev.npy',
        PREVIOUS/'born/T300_mesh64/weights.npy',PREVIOUS/'Gamma_cell_T300_mesh64.npz']
    for b in range(8):
        historical.extend(PREVIOUS/'source'/f'prod_T300_seed{b}'/name for name in ('energies_mev.npy','weights.npy','born_weights.npy'))
    for p in historical:index.append({'path':str(p),'sha256':sha256(p),'status':'protected historical raw one-phonon input; read-only'})
    for p in sorted((output/'analysis').glob('*.npz')):
        index.append({'path':str(p),'sha256':sha256(p),'status':'new raw signed pair/single arrays, batch amplitudes and broadened densities'})
    json_write(output/'raw_array_index.json',index)
    subprocess.run(['bash','-n',str(output/'reproduce_and_resume.sh')],check=True)
    subprocess.run(['bash','-n',str(output/'calculate_new.sh')],check=True)


def archive(output):
    before=json.loads((output/'protected_before.json').read_text())['files']
    paths=[p for folder in ('yoshioka','benchmarks','tests') for p in (ROOT/folder).glob('*.py') if str(p.relative_to(ROOT)) not in before]
    source=[]
    for path in sorted(paths):
        name=path.relative_to(ROOT);target=output/'implementation_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        source.append({'path':str(name),'action':'added','sha256':sha256(path)})
    json_write(output/'changed_source_files.json',{'files':source,'preexisting_source_files_modified':False})
    after=json.loads((output/'protected_after.json').read_text())
    tests=json.loads((output/'regression.json').read_text())
    result=json.loads((output/'results.json').read_text())
    validations=json.loads((output/'independent_validation.json').read_text())
    material=json.loads((output/'material_source_validation.json').read_text())
    resume=json.loads((output/'resume_verification.json').read_text())
    checks={'protected_files':after['all_unchanged'],'regression':tests['returncode']==0 and tests['test_count']>=169,
        'independent_dense':all(r['passed'] for r in validations.values()),'material_sources':material['passed'],
        'resume':resume['returncode']==0 and resume['all_hashes_and_mtimes_unchanged'],
        'pair_spectral_area':result['final']['maximum_broadened_area_relative_error']<1e-10,
        'full_order012_area':max(result['final']['full_order012_area_relative_error'])<1e-10,
        'all_branches':min(result['final']['branch_counts'])>0}
    report=(output/'report.md').read_text()
    checks['report_has_current_tests_and_weak_detector_status']='All-detector numerical peak controls' in report and f'{tests["test_count"]} tests' in report
    for job,power,_ in JOBS:
        values=[json.loads(p.read_text())['completed_pairs'] for p in (output/'pairs'/job).glob('*/state.json')]
        checks[f'complete_prefix_{job}']=len(values)==8 and min(values)>=2**power
    audit=dict(task_deliverables_complete=all(checks.values()),checks=checks,
        protected_files_unchanged=after['all_unchanged'],protected_count=after['count'],tests=tests,
        scientific_status=result['status'],scientific_completion_distinct_from_task_completion=True,
        all_six_branches_sampled=result['final']['branch_counts'],
        original_full9B_status='INCOMPLETE preserved',nonlinear_omissions_disclosed=True)
    json_write(output/'requirements_audit.json',audit)
    files={str(p.relative_to(output)):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    json_write(output/'manifest.json',{'files':files,'count':len(files),'total_bytes':sum(r['bytes'] for r in files.values())})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('stage',choices=('documents','regression','preserve-after','archive','verify-resume'))
    a=p.parse_args()
    if a.stage=='documents':documents(a.output)
    elif a.stage=='regression':regression(a.output)
    elif a.stage=='preserve-after':preserve(a.output,'after')
    elif a.stage=='archive':archive(a.output)
    else:
        paths=list((a.output/'pairs').rglob('state.json'))+list((a.output/'pairs').rglob('pair*.npz'))+list((a.output/'background').rglob('*'))
        paths=[p for p in paths if p.is_file()]
        before={str(p):[digest(p),p.stat().st_mtime_ns] for p in paths}
        start=time.monotonic()
        run=subprocess.run(['bash',str(a.output/'reproduce_and_resume.sh')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        after={str(p):[digest(p),p.stat().st_mtime_ns] for p in paths}
        json_write(a.output/'resume_verification.json',{'returncode':run.returncode,'seconds':time.monotonic()-start,
            'count':len(paths),'all_hashes_and_mtimes_unchanged':before==after,'output':run.stdout})
        if run.returncode or before!=after:raise RuntimeError('resume equivalence failed')


if __name__=='__main__':main()
