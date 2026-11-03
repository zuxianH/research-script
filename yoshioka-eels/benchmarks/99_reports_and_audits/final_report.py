"""Assemble actual statuses, immutable-input audit, costs and reproduction commands."""

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import time

from .workflow_io import ROOT, environment, json_write, sha256
from .guarded_run import profile, validate_guard


RUNS = ['elastic_repair_20261102_v1', 'elastic_repair_20261102_v2', 'repair_20261102_v1',
        'detector_cells_20261102_v1', 'repeated_20261102_v1', 'repeated_20261102_v2',
        'dynamical_si_20261102_v1', 'dynamical_si_bath4_20261102_v1', 'dynamical_si_padding_20261102_v1', 'phonon_bath_audit_20261102_v1', 'material_evidence_20261102_v1', 'tacaw_20261102_v1',
        'tacaw_20261102_v2', 'tacaw_material_20261102_v1', 'auxiliary_controls_20261102_v1',
        'auxiliary_controls_20261102_v2', 'anharmonic_20261102_v1', 'anharmonic_validation_20261102_v1',
        'anharmonic_followup_20261102_v1', 'validated_mode_eels_20261102_v1', 'evidence_20261102_v1']
GUARDED = [('elastic_streaming', 'elastic_repair_20261102_v2'), ('detector_cubature', 'detector_cells_20261102_v1'),
           ('repeated_scattering', 'repeated_20261102_v1'), ('dynamical_si_eels', 'dynamical_si_resume_20261102_v1'),
           ('anharmonic_modes', 'anharmonic_20261102_v1'),
           ('dynamical_si_eels','dynamical_si_bath4_20261102_v1'),
           ('dynamical_si_eels','dynamical_si_padding_20261102_v1')]


def streamed_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(name):
    return json.loads((ROOT/'benchmark_results'/name/'results.json').read_text())


def main():
    output = ROOT/'benchmark_results/final_evaluation_20261102_v1'
    output.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((ROOT/'benchmark_results/repair_20261102_v1/baseline.json').read_text())
    preservation = {'checked_files': len(baseline['files']), 'changed': [], 'missing': [], 'hashes': {}}
    for name, record in baseline['files'].items():
        path = ROOT/name
        if not path.exists():
            preservation['missing'].append(name)
            continue
        digest = streamed_hash(path)
        preservation['hashes'][name] = digest
        if digest != record['sha256']:
            preservation['changed'].append(name)
    preservation['passed'] = not preservation['changed'] and not preservation['missing']
    json_write(output/'preservation.json', preservation)
    if not preservation['passed']:
        raise ValueError('Protected baseline changed; investigate before publishing statuses')
    # No historical guard is invented. Adoption happens here after the current
    # source/constants audit, while original cache contracts remain untouched.
    guards = {}
    commands = []
    for module, config_name in GUARDED:
        config_path = ROOT/f'benchmarks/configs/{config_name}.json'
        config = json.loads(config_path.read_text())
        target = ROOT/config['output']
        current = profile('benchmarks.'+module)
        guard = validate_guard(target/'cache_guard.json', current, adopt=True)
        guards[config_name] = guard
        snapshot = target/'guarded_source_snapshot'
        for name, digest in current['source_sha256'].items():
            destination = snapshot/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.write_bytes((ROOT/name).read_bytes())
            if sha256(destination) != digest:
                raise ValueError('calculation source snapshot changed')
        commands.append(f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.guarded_run benchmarks.{module} --config benchmarks/configs/{config_name}.json')
        cold = {**config, 'output': config['output']+'_cold_reproduction'}
        json_write(output/f'{config_name}_cold.json', cold)
    json_write(output/'constants_and_cache_profiles.json', guards)
    results = {name: read(name) for name in RUNS if (ROOT/'benchmark_results'/name/'results.json').exists()}
    b7 = results['detector_cells_20261102_v1']
    b8 = results['repeated_20261102_v1']
    b8s = results['repeated_20261102_v2']
    b9 = results['dynamical_si_20261102_v1']
    b11 = results['anharmonic_validation_20261102_v1']
    complete = [r for r in b9['cases'] if r['status'] == 'COMPLETE']
    if len(complete) != 11 or len(results['dynamical_si_padding_20261102_v1']['cases']) != 2 or any(
            c['status'] != 'COMPLETE' for c in results['dynamical_si_padding_20261102_v1']['cases']):
        raise ValueError('Final evaluation requires the declared 11 material cases and two padding/slice cases to finish')
    final_pair = b7['comparisons'][-1]
    tested = {'solver_complex_error': min(r['complex_relative_error'] for r in b8['independent_controls']['rows']
        if 'complex_relative_error' in r)} if any('complex_relative_error' in r for r in b8['independent_controls']['rows']) else {}
    rows = [
        {'milestone':'8 solver agreement','status':b8s['status'],'scope':'synthetic two-mode spatial common-v Hermitian solver',
         'evidence':'Independent dense complex-amplitude error 3.9535e-9; 3B exact propagation error 9.2792e-15; no phase alignment.',
         'finest':'physical p=12 (91 channels), 16/24 electron grid, 64 fourth-order steps/segment, working D=48/64, quadrature 128',
         'limits':'synthetic model, not full silicon'},
        {'milestone':'8 repeated interactions','status':b8s['status'],'scope':'linear and nonlinear two-region controls, g=0,.01,.1,1,2; L=20/80 nm',
         'evidence':'Weak inelastic scaling approaches g^2; linear Dyson second-order two-phonon amplitude is nonzero; full minus first-order detector differences measured.',
         'finest':'p=12, same Hd in full and distorted first-order references','limits':'nonlinear direct multiphonon transitions are separated from repeated interactions'},
        {'milestone':'9A','status':'INCOMPLETE','scope':'actual finite-slab Si elastic incoming/outgoing propagation with one linear source',
         'evidence':f'{len(complete)} completed base cases plus 2 slice/padding cases; Born-control complex error 2.9131e-9; a grid window still changes 2.90% and bath refinement changes the worst window 111%.',
         'finest':'129^2/1025 slices at 2 repeats; 65^2/2049 at 52 repeats; periodic phonon baths 2x2x2 and 2x2x4 as completed',
         'limits':'phonon bath, material grid/slices/padding and joint convergence not established; 9A is undepleted'},
        {'milestone':'9B','status':'INCOMPLETE','scope':'full repeated material channel propagation',
         'evidence':'93-mode p=2 gives 4465 channels and 1.35 TB dense local matrices at 65^2; actual B8 timing/count scaling recorded.',
         'finest':'representation feasibility audit, no full material propagation','limits':'structured/sparse representation and controlled material thermal trace required'},
        {'milestone':'10 metadata completeness','status':'INCOMPLETE','scope':'immutable supplied TACAW arrays and associated analysis',
         'evidence':'Hashes match original HiPhive_TRPMD arrays; both 600x267x266 float64, all finite and nonnegative. 60 keV/208 slices/unit-integral/Kubo supplied by user.',
         'finest':'metadata manifest distinguishes verified, derived, user-supplied and missing fields','limits':'generation absent here; exact force hash, axes, geometry, window and elastic handling remain unverified'},
        {'milestone':'10 physical comparability','status':'BLOCKED_BY_METADATA','scope':'TACAW versus harmonic/anharmonic Yoshioka',
         'evidence':'Auxiliary-wave small-system control error 1.9006e-8; finite channel-phase effect measured; linear Kubo recovery and nonlinear multiphonon differences evaluated.',
         'finest':'independent quadrature, p=24; no double Kubo correction','limits':'matched material force/geometry/detector/native resolution are not established'},
        {'milestone':'10 spectral comparison','status':'BLOCKED_BY_METADATA','scope':'quantitative comparison; authorized qualitative processing completed',
         'evidence':'Conservative energy rebin, disclosed 5–90 meV shape conditioning, added 2 meV display convolution, absolute Yoshioka weights, conditional residuals and TACAW index maps/cuts saved.',
         'finest':'300/1000 K supplied arrays and all completed material/candidate spectra','limits':'unknown momentum calibration/native window/elastic subtraction prevent absolute, linewidth and physical residual-map acceptance'},
        {'milestone':'10 runtime and memory comparison','status':'NOT_ASSESSED','scope':'matched scientific accuracy and physical scope',
         'evidence':'New calculation wall/RSS records and hardware/software profile saved; TACAW generation cost unavailable.',
         'finest':'cost-versus-scope table','limits':'no speedup claimed from averaged arrays or unmatched methods'},
        {'milestone':'11A frequency shifts','status':'FAIL','scope':'tested full effective-harmonic models at fixed volume',
         'evidence':'Held-out force errors 0.14825/0.28890 at 300/1000 K (target .15); 9/10 and 6/10 mode groups pass independent frequency targets; largest discrepancy 2.6715 meV.',
         'finest':'10 MD trajectories, up to 576 atoms and 20 ps; 36 fit variants, independent seeds, blocked fits, mesh-12 stability; reduced accepted-Gamma EELS separately computed',
         'limits':'tested full models fail frequency fidelity; fixed volume/classical MD plus Bose EHM is an approximation, not quantum anharmonic validation'},
        {'milestone':'11B linewidths','status':'INCOMPLETE','scope':'mode-projected independent MD, window-convolved spectral fits',
         'evidence':'All tested intrinsic linewidths NOT_RESOLVED under predeclared resolution/spread gates. 20 ps Hann power-window FWHM .29789 meV, Nyquist 413.567 meV.',
         'finest':'20/10 ps segments, 5 fs cadence, 1/.5 fs timestep, two supercells, seeds and late-coherence diagnostics',
         'limits':'finite-size/temperature confounding, sampling and late coherence prevent reliable lifetimes; no linewidth added to EELS'}]
    status = {'benchmark_1':'PASS','benchmark_2_new':results['elastic_repair_20261102_v2']['status'],
        'benchmark_2_historical':read('elastic_si110')['status'],'benchmark_3':'PASS','benchmark_3B':b8['solvable_3B']['status'],
        'benchmark_4':'PASS','benchmark_5':'PASS','benchmark_6':'PASS','benchmark_7':'INCOMPLETE', 'milestones':rows,
        'no_overall_material_PASS':True, 'protected_originals_unchanged':preservation['passed']}
    json_write(output/'statuses.json', status)
    hardware = {'platform': platform.platform(), 'processor': platform.processor(), 'logical_CPU_count': os.cpu_count(),
        'CPU_model': next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),None),
        'software':environment(), 'precision':'float64/complex128', 'intended_processes':1,
        'intended_threads':{'OPENBLAS_NUM_THREADS':1,'OMP_NUM_THREADS':1},
        'exception':'A default-thread B9 variant was explicitly stopped; its cache and reason record are preserved. It changed degenerate eigenvector hashes, not the declared physics.'}
    costs = []
    for name, result in results.items():
        cost = result.get('resource_costs', {'elapsed_seconds': result.get('elapsed_seconds', result.get('processing_seconds', result.get('inspection_seconds')))})
        costs.append({'run':name,'status':result.get('status'), 'latest_reported_invocation_cost':cost,
            'cache':'cold pilot and resumed prefix costs remain in per-run states; latest wall is not cumulative source time',
            'scope_matched_to_TACAW':False})
        directory = ROOT/'benchmark_results'/name
        records = []
        for state_path in directory.glob('cache/*/state.json'):
            state = json.loads(state_path.read_text())
            records.append({'path':str(state_path.relative_to(ROOT)), 'status':state.get('status'),
                'cumulative_seconds':state.get('cumulative_seconds'), 'latest_invocation_seconds':state.get('latest_invocation_seconds'),
                'completed_q':state.get('completed_q'), 'completed_modes':state.get('completed_modes')})
        if records:
            costs[-1]['all_preserved_cache_generations'] = records
        manifest = {'results_sha256':sha256(directory/'results.json'), 'hardware':hardware,
            'guard':str(directory/'cache_guard.json') if (directory/'cache_guard.json').exists() else None,
            'constants_profile':str(output/'constants_and_cache_profiles.json'),
            'figures':[p.name for p in sorted(directory.glob('*.png'))], 'cost':cost,
            'preservation_audit':str(output/'preservation.json')}
        json_write(directory/'evidence_manifest_20261102.json',manifest)
    costs.append({'run':'TACAW generation','status':'NOT_ASSESSED','wall_seconds':None,'peak_RSS':None,
        'reason':'Generation scripts/logs absent here, confirmed by user. Current data processing time is not simulation cost.', 'speedup':None})
    reuse_path = ROOT/'benchmark_results/dynamical_si_20261102_v1/bath4_cache_reuse.json'
    json_write(output/'cost_vs_scope.json',{'hardware':hardware,'calculations':costs,
        'cache_reuse_do_not_double_count':json.loads(reuse_path.read_text())})
    changed = []
    for directory in ('yoshioka','benchmarks','tests'):
        for p in sorted((ROOT/directory).rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix not in ('.py','.json','.md'):
                continue
            name=str(p.relative_to(ROOT))
            if name not in baseline['files']:
                changed.append({'path':str(p),'sha256':sha256(p),'change':'added'})
    json_write(output/'changed_source_files.json',changed)
    test_log = (ROOT/'benchmark_results/regression_20261102.log').read_text()
    match = re.search(r'Ran (\d+) tests in ([0-9.]+)s',test_log)
    tests = {'count':int(match.group(1)) if match else None,'seconds':float(match.group(2)) if match else None,
        'passed':test_log.rstrip().endswith('OK'),'log':str(ROOT/'benchmark_results/regression_20261102.log')}
    json_write(output/'tests.json', tests)
    requirements = {
        'original_inputs_results_preserved': preservation['passed'],
        'B2_repaired_and_original_FAIL_retained': results['elastic_repair_20261102_v2']['status']=='PASS' and read('elastic_si110')['status']=='FAIL',
        'B7_old_failed_observables_and_measures_audited': (ROOT/'benchmark_results/repair_20261102_v1/historical_audit.json').exists(),
        'B7_independent_source_validation':json.loads((ROOT/'benchmark_results/repair_20261102_v1/source_validation.json').read_text())['passed'],
        'B7_dense_and_shifted_quadrature_evaluated':len(b7['cases'])==3,
        'B7_unmet_convergence_reported_without_relaxation':b7['status']=='INCOMPLETE' and b7['accepted_settings'] is None,
        '3B_actual_propagation':b8['solvable_3B']['status']=='PASS',
        'B8_independent_solver_and_spectrum_checks':b8s['status']=='PASS',
        'B9A_material_calculations_complete':len(complete)==11,
        'B9B_measured_representation_infeasibility_instead_of_source_substitution':bool(b9['9B_feasibility']['dense_representation_estimates']),
        'B10_read_only_hash_and_metadata_audit':len(results['tacaw_20261102_v1']['datasets'])==2,
        'B10_user_authorized_qualitative_processing':results['tacaw_material_20261102_v1']['qualitative_processing_status']=='PASS',
        'B10_independent_auxiliary_Kubo_control':results['auxiliary_controls_20261102_v2']['status']=='PASS',
        'B10_quantitative_claims_and_costs_blocked_honestly':results['tacaw_material_20261102_v1']['quantitative_comparison_status']=='BLOCKED_BY_METADATA',
        'B11_real_displacement_force_data_and_heldout_fits':len(results['anharmonic_20261102_v1']['fits'])==36,
        'B11_MD_mode_frequency_and_linewidth_route':len(b11['MD_mode_fits'])==300,
        'B11_failed_full_model_and_unresolved_linewidths_retained':b11['status']=='INCOMPLETE',
        'B11_reduced_prediction_only_accepted_frequency_subspaces':results['validated_mode_eels_20261102_v1']['reduced_prediction_status']=='PASS',
        'regression_tests_passed':tests['passed'],
        'no_full_material_scientific_convergence_claim':True,
        'FRFPMS_skipped':True}
    json_write(output/'requirements_audit.json',requirements)
    commands += ['OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.repeated_scattering_spectra',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.auxiliary_wave_controls',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.anharmonic_validation',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.anharmonic_followup',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.validated_mode_eels',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.material_evidence',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.phonon_bath_audit',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.tacaw_comparison',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.tacaw_qualitative',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.tacaw_material_comparison',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.evidence_diagnostics',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.final_report']
    (output/'reproduce_and_resume.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+str(ROOT)+'\n'+'\n'.join(commands)+'\n')
    text = ['# Benchmark evaluation, 2026-11-02', '',
        'The new Benchmark 2 run passes. Benchmark 7 remains incomplete. The synthetic repeated-channel solver passes; full material repeated scattering and quantitative TACAW agreement are not established. The tested full EHM models fail selected validation targets, and intrinsic linewidths remain unresolved. Software tests alone do not confer a material PASS.', '',
        f"Protected files: **{preservation['checked_files']}/{preservation['checked_files']} unchanged**, covering the earlier 127-file protected set. All {tests['count']} tests pass in {tests['seconds']} s. Historical FAIL/INCOMPLETE reports and original inputs are preserved. This directory is not a git repository. No AGENTS.md was present in the project or its ancestors.", '',
        '| Milestone | Status | Scope / measured evidence | Finest completed settings | Limitations |', '|---|---|---|---|---|']
    for row in rows:
        text.append('| '+' | '.join(row[key] for key in ('milestone','status','evidence','finest','limits'))+' |')
    text += ['', '## Benchmark 2 and 7 repair evidence', '',
        'Benchmark 2 retains the original supplied static structure and finite Lobato slice projection, periodic bulk boundary and antialias filtering. The strict new comparison uses the actual same abTEM/ASE constants in both paths. Its older different-constants result remains a separate sensitivity study. The accepted grid is 768×768 with 256 slices per original repeat, tested at 13/26/52 repeats. Raw complex implementation error at 52 repeats is 1.874e-12, with no global phase or intensity fit. Final grid, slice and joint observable changes are below 1%; the worst final joint change is 0.00378817. The original failure required both transverse and slice refinement, independently of the B7 phonon problem. The complete failed beam/detector rows include absolute intensities in the versioned repair results.', '',
        'For Benchmark 7 the old 24→32 central L1 change was 0.071769233, window change 0.21940531 and peak change 0.35 meV. Dense nodal 64→96 improves L1 but still changes an outer clipped pixel by 9.36%. A constant-density geometric test changes that pixel area from 3.77233 to 4.15449 mrad² versus exact 4.00888 mrad²: detector-edge sampling is a demonstrated cause. Exact cell/detector intersections replace binary masks, with explicit halo source evaluation and no fitted rescaling or second angular-area weight.', '',
        f"After this correction, 48→64 gives L1 0.00140797, worst window change 0.0169397 and peak shift 0.05 meV. 64→96 gives L1 {final_pair['spectral_changes']['maximum_relative_L1']:.8g}, worst window change {final_pair['physical_detector_changes']['maximum_acceptance_error']:.8g}, and peak shift {final_pair['maximum_peak_shift_mev']} meV. Only the final pair passes. The central 40–65 meV window remains limited by energy-boundary/branch sampling, localized by the saved q-shell and branch attribution. Two successive passes and the conditional final joint numerical-source refinement are missing; there are no accepted complete B7 settings. Mesh 128 is a further computation, not a claimed result.", '',
        'The exact-forward transverse contribution is a weighted BZ-fiber sample, not a separately added bulk point mass. Its weight decreases as N^-2 while its associated angular-cell area does the same. The density estimate approaches 2.69775e-17 probability/mrad². This sample stays inside all physical detector integrals; the old raw-point metric remains visible as a diagnostic. At finite loss Qz=-[Qperp²/(2k)+omega/v] is nonzero at Qperp=0. Gamma optical modes are retained; only three rigid translations are removed. Genuine discrete points in finite models receive no bin-area factor. See B7_diagnostics.json for the measure derivation, shifted-mesh comparison, full L1 numerators/denominators, absolute errors and competing/boundary maxima.', '',
        'The accelerator is separately checked against actual inverse-FFT, finite-difference and Simpson source propagation in 13 representative sectors through grid 320, 8192 longitudinal samples and 14 Å padding. Global relative complex error is 2.54854e-7. This establishes implementation agreement for the declared first-Born physics, not arbitrary coupled-channel Poisson weights.', '',
        '## Common equations and approximations', '',
        'The solver uses i hbar v d_z phi=[-(hbar v/2k) laplacian_perp + epsilon_n-epsilon_m + U] phi, with SI energies U and h=-eV for electrostatic potential V. No extra longitudinal carrier is added. Entrance plane waves are 1/sqrt(A); dx dy sum_channel,pixel |phi|² is monitored for closed Hermitian propagation. Fourier coefficients are FFT(phi) sqrt(A)/Npixel and their squared moduli are discrete populations. Orthogonal crystal channels are added as probabilities even when degenerate. Different Dyson orders in one channel are amplitudes and interfere.', '',
        'B8 splits H=Hd+gK, where Hd retains kinetic, channel energies and channel-diagonal elastic terms. Its independent first order is -i/(hbar v) integral Ud(L,z) K Ud(z,0) dz. Nonlinear matrix elements permit direct multiphonon transitions; the linear control and explicit Dyson orders identify repeated interactions. No arbitrary phase alignment, output renormalization or extra Debye–Waller factor is applied.', '',
        '9A is undepleted: unit-norm incoming propagation plus source-channel intensity is not a closed-channel norm. At finite T the linear-source occupation trace is exactly nB+1 and nB, with covariance included once. It is not a full nonlinear thermal Fock calculation. The small-system trace control compares explicit incoherent initial states with the alternative trace and reports omitted initial weight. The B9 thermal diagnostic quantifies why macroscopic absolute occupation p=2 would be inadequate. Full 9B remains uncomputed with this dense representation.', '',
        'Detailed balance interchanges initial/final thermal states and conjugates the scattering operator: S(-Q,-E)=exp(-beta E) S(Q,E) when F(Q)†=F(-Q). Distorted-wave conjugation reverses incoming/outgoing paths, so a same-pixel gain/loss identity is not imposed without reciprocal geometry. The Bose occupation factor identity is tested separately.', '',
        'The SI constants, actual Lobato table, package versions and source hashes are recorded in constants_and_cache_profiles.json. meV→J uses 1e-3 e, Å→m uses 1e-10, inverse Å→inverse m uses 1e10, u→kg uses the recorded atomic mass constant, and omega=2pi f. The Q appearing in displacement phases is angular, with no extra 2pi. The strict B2 native-constant exception is recorded explicitly in its source audit.', '',
        '## TACAW and anharmonic conclusions', '',
        'The supplied TACAW arrays are immutable comparison artifacts, not exact references. The user confirms 60 keV, 208 slices, a unit integral over px,py,E and already-applied Kubo correction; the potential match is tentative. No second Kubo correction is applied. Associated analysis uses a conditional 7.5 fs sampling interval; this is not independently verified generation metadata. Missing axes, geometry, force hash, elastic treatment and native spectral window prevent quantitative matching. Qualitative shape spectra, temperature index maps/cuts and residuals are saved with all assumptions. No arbitrary energy shift or fitted scale is used, and no statistical bars or TACAW generation timing are inferred from averaged arrays.', '',
        'The controlled auxiliary-wave comparison uses independent coordinate quadrature and the actual channel solver with channel-energy phases switched off. Their complex error is 1.90056e-8 at p=24. Restoring channel phases changes the controlled state by relative norm 0.0111556, without phase alignment. Kubo recovers the linear harmonic classical response but does not make the nonlinear phase-operator multiphonon spectrum exactly quantum. The illustrative traversal estimate is a kinematic scale, not a verified TACAW thickness or uniform detector-error bound.', '',
        'The EHM workflow uses actual MOD displacement-force data, separate seeds for training and validation, periodic displacements, symmetry/translation constraints and fixed zero-temperature equilibrium volume. Ten NVE production trajectories follow NVT equilibration. Training counts, cutoffs, ridge values, supercells, timestep and segment lengths are varied. Independent projected MD spectra are matched by eigenvector/subspace overlap. Saturating held-out and training force errors plus selected frequency discrepancies show a limitation of the tested harmonic force-fit model class. Actual NVE temperatures differ between ensembles, so size/timestep changes are disclosed as combined sensitivity rather than pure equal-temperature convergence.', '',
        'The linewidth route fits a Lorentzian forward convolved with the actual Hann power window. It checks duration, blocks, seeds, cadence, timestep, size, late coherence and mode subspaces. FWHM is an energy width; an amplitude decay exp(-t/tau_A) would have FWHM=2 hbar/tau_A and population lifetime tau_A/2, but no lifetime is assigned here. Zero padding is not resolution. No EELS peak width is called an intrinsic mode width. Full EHM EELS curves remain unaccepted candidates; a separate reduced prediction uses only independently accepted Gamma subspaces with no renormalization or intrinsic linewidth.', '',
        '## Costs, files and reproduction', '',
        f"Hardware: {hardware['CPU_model']}, {hardware['logical_CPU_count']} logical CPUs, one intended process with BLAS/OMP threads fixed to one. Software versions and actual per-run costs are in cost_vs_scope.json. B2 streaming reduced peak RSS from a failed 6180 MiB all-stack attempt to 2608.8 MiB. The reported final B2 invocation is 528.3 s; cumulative cache time remains separate. B7 nodal mesh-96 source time is approximately 1558.5 s and all completed/restarted generations have their own costs. No scope-unmatched TACAW speedup is claimed.", '',
        'All new source/config/test files are indexed with full paths and hashes in changed_source_files.json. New reusable modules cover integration measures, detector cells, coupled channels, distorted first order, dynamical material sources, spectral data, MD and mode spectra. Original source files and reports in the protected baseline are unchanged. Raw signed lines, momentum weights, propagated waves, MD forces/displacements/velocities and minimally processed spectra remain available in versioned directories.', '',
        'Primary reproduction and resume commands follow. Repeat a guarded bounded runner until its declared cases finish; a controlled stop preserves the committed prefix. Guards reject changed source/constants/tables/thread environments; runtime limits alone do not invalidate physical caches. Guards were adopted after an explicit current-provenance audit, not backdated. The old nodal accelerator source snapshot is preserved in repair_20261102_v1/source_snapshot; the improved cell runner is the current primary B7 route. For a cold run, use the corresponding *_cold.json configuration saved here; it points to a new output directory.', '',
        '```bash', 'cd '+str(ROOT), *commands, '```', '',
        'The original SIGTERM cause was not established. A new single-thread 900 s invocation also exited 143 without a Python exception; its signal producer is unknown and its committed mode prefix was valid. Shorter 600 s invocations completed all remaining cases. The separate thread-variant stop was intentional and has its own record. None is labelled physical or numerical instability. FRFPMS was skipped.', '',
        'Primary method references: [TACAW](https://doi.org/10.1103/PhysRevLett.134.036402), [hiPhive EHM](https://hiphive.materialsmodeling.org/advanced_topics/effective_harmonic_models.html), [LAMMPS MOD](https://docs.lammps.org/pair_tersoff_mod.html). Agreement within an empirical model does not independently validate Si against experiment or DFT.']
    (output/'report.md').write_text('\n'.join(text)+'\n')
    material = results['material_evidence_20261102_v1']
    extra = ['', '## Measured material refinements and figures', '',
        'These are numerical comparisons at fixed specimen and temperature. They do not replace the still-missing bulk phonon and full-channel checks. The inherited 1% criterion and weak probability scale 1e-18 are unchanged.', '',
        '| Changed parameter | Coarse → fine | Worst observable | Coarse / fine absolute weights | Absolute change | Acceptance error | Maximum spectrum L1 |',
        '|---|---|---|---|---:|---:|---:|']
    for comparison in material['refinements']:
        parameter = comparison['changed_parameter']
        name, worst = max(comparison['observables'].items(),key=lambda item:item[1]['acceptance_error'])
        settings = f"{comparison['coarse'][parameter]} → {comparison['fine'][parameter]}"
        extra.append(f"| {parameter} | {settings} | {name} | {worst['coarse']:.8g} / {worst['fine']:.8g} | {worst['absolute_change']:.5g} | {worst['acceptance_error']:.6g} | {max(comparison['spectrum_L1_relative'].values()):.6g} |")
    extra += ['', 'The raw source data were unaffected by a corrected postprocessing filename collision between the padding cases. The diagnostic pre-fix processed file and audit remain preserved; final artifact identities include padding and are checked for uniqueness.', '',
        'The transverse extent is the periodic compatibility cell. Replicating the same field into a larger electron cell preserves its original Fourier subspace: the atom sum and incident area grow together, and newly forbidden beams cancel. This normalization is tested. Adding independent transverse phonon modes is a separate, unresolved bath refinement.', '',
        '| 52-repeat bath-2 temperature K | Absolute 20 mrad inelastic weight |', '|---:|---:|']
    for c in complete:
        if c['settings']['thickness_repeats']==52 and c['settings']['phonon_z']==2:
            extra.append(f"| {c['settings']['temperature_k']} | {c['observables']['full_0_20_mrad']:.10g} |")
    extra += ['', 'The fixed-thickness longitudinal bath refinement is a large failure of convergence: the 20 mrad weight changes from 0.003900317793 to 0.001977089642. A phonon-momentum projector decomposition reproduces each total to below 2e-18 absolute error. The qz=0 contributions are 0.003893626792 and 0.001958746786; multiplying by the respective bath-z counts gives 0.00778725 and 0.00783499, consistent with a central quadrature sample being overweighted on the coarse grid. The bath spacings are 26 and 13 times 2pi/L. The independent atom slab remains 19.96219027 nm in both cases. This is not evidence for a physical bulk delta mass, and the phonon covariance also changes with the bath. The projector is a basis-invariant subspace measurement; its orthogonal components reproduce the original incoherent total.', '',
        'The same completed bath-4 cache is copied into the consolidated run after checking its exact contract. Its 1307.94 s source time appears in both cache records but is only one computation; cost_vs_scope.json explicitly records this reuse. The final consolidation performs no extra source propagation.']
    extra += ['', 'Figures:', '',
        f"- [B7 spectra and absolute differences]({ROOT}/benchmark_results/evidence_20261102_v1/B7_cells_and_absolute_differences.png)",
        f"- [B8 populations and detector departures]({ROOT}/benchmark_results/evidence_20261102_v1/B8_populations_and_detector_departures.png)",
        f"- [Material detector spectra]({ROOT}/benchmark_results/material_evidence_20261102_v1/detector_thickness_temperature.png)",
        f"- [Conditional TACAW comparison]({ROOT}/benchmark_results/tacaw_material_20261102_v1/conditional_material_spectra_and_residuals.png)",
        f"- [Mode correlations and spectra]({ROOT}/benchmark_results/evidence_20261102_v1/B11_mode_correlations_and_spectra.png)"]
    with (output/'report.md').open('a') as stream:
        stream.write('\n'.join(extra)+'\n')
    diagnostic_folder = ROOT/'benchmark_results/evidence_20261102_v1'
    json_write(diagnostic_folder/'evidence_manifest_20261102.json', {
        'diagnostics_sha256':sha256(diagnostic_folder/'diagnostics.json'),
        'source_sha256':sha256(ROOT/'benchmarks/evidence_diagnostics.py'),
        'constants_profile':str(output/'constants_and_cache_profiles.json'),
        'figures':[str(p) for p in sorted(diagnostic_folder.glob('*.png'))]})
    json_write(output/'manifest.json',{'report_sha256':sha256(output/'report.md'),'source_sha256':sha256(__file__),
        'status':'COMPLETED_EVALUATION_WITH_EXPLICIT_SCIENTIFIC_LIMITATIONS','tests':tests,'preservation_passed':True,
        'run_result_hashes':{name:sha256(ROOT/'benchmark_results'/name/'results.json') for name in results}})


if __name__ == '__main__':
    main()
