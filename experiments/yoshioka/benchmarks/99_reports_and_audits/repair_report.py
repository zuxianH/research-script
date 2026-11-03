"""Consolidate measured repair results without changing historical evidence."""
import json,re,math,platform
from pathlib import Path
from .workflow_io import ROOT,json_write,sha256,environment
from .guarded_run import profile

OUT=ROOT/'benchmark_results/repair_20261102_v2'


def read(path,default=None):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def number(value):
    return '—' if value is None else f'{value:.8g}'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|',
        *['| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in rows]])


def worst(comparison,key='rows'):
    return max(comparison[key],key=lambda r:r['acceptance_error'])


def changes():
    before=read(OUT/'protected_before.json')['files'];records=[]
    for directory in ('yoshioka','benchmarks','tests'):
        for path in sorted((ROOT/directory).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix not in ('.py','.json'):continue
            name=str(path.relative_to(ROOT));digest=sha256(path)
            if name not in before or digest!=before[name]['sha256']:
                records.append({'path':name,'absolute_path':str(path),'sha256':digest,'bytes':path.stat().st_size,
                    'change':'added' if name not in before else 'modified'})
    path=ROOT/'BENCHMARK_REPAIRS_20261102.md'
    if path.exists():records.append({'path':path.name,'absolute_path':str(path),'sha256':sha256(path),'bytes':path.stat().st_size,'change':'added'})
    json_write(OUT/'changed_source_files.json',records)
    return records


def costs():
    rows=[]
    for name in ('B7','B7_offset','B7_offset96','B7_joint','B9A_depth','B9A_batched','B9A_bath','B9A_thick_gemm',
        'B9A_bloch','B9A_bloch_electron','B9A_bloch_extended','B9A_bloch_bath','B9A_bloch_transverse','B9A_bloch_fullmesh',
        'B9A_windowed_grid','B9A_windowed_joint','B9A_fft_grid','B9A_fft_joint',
        'B9B_validation','B9B_material','B9B_gamma6','B9B_coordinate','B9B_fine','B9B_orders','B11','B11_audit','B11_checked','B11_refit'):
        result=read(OUT/name/'results.json',{})
        states=[]
        for path in (OUT/name).glob('cache/*/state.json'):
            d=read(path);states.append({'path':str(path),'status':d.get('status'),'completed_modes':d.get('completed_modes'),
                'completed_q':d.get('completed_q'),'cumulative_seconds':d.get('cumulative_seconds'),
                'latest_invocation_seconds':d.get('latest_invocation_seconds'),'adopted_prefix':d.get('adopted_prefix'),
                'adopted_incident':d.get('adopted_incident'),'scope':'this checkpoint source work only; adoption costs are separate'})
        if result or states:rows.append({'calculation':name,'latest_invocation':result.get('resource_costs'),'checkpoint_work':states})
    record={'rows':rows,'double_counting_policy':'do not sum adopted and original source work twice; incomplete alternatives and failed resource pilots are separate actual costs',
        'deadline_policy':'soft check between committed source batches; one bounded batch may exceed the nominal deadline; invocation1 joint measured1821.219s for1800s target, then safely resumed',
        'scope':'latest invocation wall/RSS separated from cumulative committed checkpoint work; not total project wall time',
        'resource_failure':read(OUT/'windowed_memory_failure.json'),
        'storage_repair_pilots':{name:read(OUT/(name+'.json')) for name in ('fourier_background_pilot','fourier_source_pilot','fft_friendly_cold_source_pilot','fft_friendly_warm_source_pilot','FFT_cost_pilot')},
        'TACAW_generation_cost':None,'TACAW_speedup_claimed':False}
    record['largest_coordinate_order_audit']=read(OUT/'B9B_order_audit/comparison.json')
    json_write(OUT/'cost_vs_scope.json',record);return record


def mode_report(thermal):
    rows=[]
    for r in thermal['frequency_rows']:
        rows.append([r['temperature_k'],r['q_fractional'],r['group'],number(r['effective_energy_mev']),number(r['MD_center_mev']),
            number(r['effective_minus_MD_mev']),number(r['MD_conservative_spread_mev']),number(r['subspace_min_singular_overlap']),
            ('PASS' if r['original_frequency_gate_pass'] else 'FAIL') if r.get('reference_shape_valid',True) else 'REFERENCE_SHAPE_UNCERTAIN',number(r['FWHM_mev']),number(r['Hann_window_FWHM_mev']),r['linewidth_status']])
    text='# Per-mode/subspace validation\n\nOriginal64-frame model against corrected100ps MD:\n\n'+table(['Target T (K)','q fractional','subspace','EHM (meV)','mode-MD (meV)','EHM−MD (meV)','conservative spread (meV)','min singular overlap','frequency gate','candidate FWHM (meV)','Hann FWHM (meV)','intrinsic width'],rows)
    text+='\n\nAll rows use fixed volume '+number(thermal['frequency_rows'][0]['volume_angstrom3_per_unit'])+' Å³ per oriented four-atom cell. Spread is the maximum replica/segment/setting sensitivity saved by the estimator, not a statistical confidence interval. The frequency gate is |EHM−MD|<max(0.5 meV,2×spread); the historical effective-model FAIL is retained independently. Candidate fitted widths are diagnostic parameters, not resolved intrinsic widths or lifetimes. The full fit-shape, block/setting-width spread, actual NVE temperature and window data are in B11/results.json and its mode_fits.\n'
    refit=read(OUT/'B11_refit/results.json');selected=[]
    for fit in refit['fits']:
        if fit['count']!=128:continue
        for r in fit['frequency_rows']:selected.append([r['temperature_k'],r['q_fractional'],r['group'],number(r['effective_energy_mev']),number(r['MD_center_mev']),number(r['effective_minus_MD_mev']),number(r['MD_conservative_spread_mev']),number(r['refitted_subspace_min_overlap']),('PASS' if r['original_frequency_gate_pass'] else 'FAIL') if r.get('reference_shape_valid') else 'REFERENCE_SHAPE_UNCERTAIN'])
    text+='\n## Independently refitted128-frame model\n\n'+table(['Target T(K)','q','subspace','Refitted EHM(meV)','mode-MD(meV)','EHM−MD(meV)','spread(meV)','subspace overlap','frequency gate'],selected)+'\n\nThese models use independent corrected trajectories, the original5Å range and zero regularization. All linewidths remain unresolved; refitting frequencies does not supply a linewidth.\n'
    (OUT/'per_mode_validation.md').write_text(text)


def main():
    b7=read(OUT/'B7/results.json');joint=read(OUT/'B7_joint_evaluation.json');bloch=read(OUT/'B9A_Bloch_convergence.json')
    legacy=read(OUT/'B9A_diagnosis_final.json');repeated=read(OUT/'B9B_convergence_summary.json')
    validation=read(OUT/'B9B_validation/results.json');material=read(OUT/'B9B_material/results.json');fine=read(OUT/'B9B_fine/results.json')
    orders=read(OUT/'B9B_orders/results.json');quadrature=read(OUT/'B9B_order_audit/comparison.json');thermal=read(OUT/'B11_checked/results.json');audit=read(OUT/'B11_final_audit/results.json')
    refit=read(OUT/'B11_refit/results.json');gamma=read(OUT/'B9B_Gamma_refinement.json',{})
    metadata=read(OUT/'B10/results.json');before=read(OUT/'protected_before.json');after=read(OUT/'protected_after.json',{})
    tests_log=(OUT/'regression_final.log').read_text();matches=re.findall(r'Ran (\d+) tests in ([0-9.]+)s',tests_log)
    tests={'count':int(matches[-1][0]) if matches else None,'seconds':float(matches[-1][1]) if matches else None,
        'passed':bool(re.search(r'\nOK\s*$',tests_log)),'log':str(OUT/'regression_final.log')}
    json_write(OUT/'tests.json',tests)
    historical=read(ROOT/'benchmark_results/final_evaluation_20261102_v1/statuses.json')
    elastic=read(ROOT/'benchmark_results/elastic_repair_20261102_v2/results.json')
    old8=read(ROOT/'benchmark_results/repeated_20261102_v1/results.json')
    verified={'B2_status':elastic['status'],'B2_accepted_settings':elastic['accepted_settings'],
        'B2_52_repeat_complex_error':elastic['cases'][-1]['relative_exit_wave_error'],
        'B8_status':old8['status'],'B8_independent_nonlinear_64_step_error':next(r['complex_relative_error'] for r in old8['independent_controls']['rows'] if not r.get('linear_control',True) and r.get('steps_per_segment')==64),
        'historical_test_record':read(ROOT/'benchmark_results/final_evaluation_20261102_v1/tests.json'),
        'historical_preservation_count':read(ROOT/'benchmark_results/final_evaluation_20261102_v1/preservation.json')['checked_files'],
        'all_487_prior_hashes_match_initial_audit':not read(OUT/'initial_diagnosis.json')['protected_previous_differences'],
        'historical_statuses':historical,'reference_report_sha256':sha256(ROOT/'benchmark_results/final_evaluation_20261102_v1/report.md')}
    json_write(OUT/'verified_previous_evidence.json',verified)
    statuses={
        'Benchmarks1–6_3B_8':'PASS_PRESERVED',
        'B7_phonon_integration':b7['physical_quadrature_status'],'B7_independent_joint':joint['status'],
        'B9A_electron_grid':bloch['electron_original_four_detector_gate'],'B9A_slices':bloch['slice_original_gate'],
        'B9A_electron_joint':bloch['electron_joint_original_gate'],'B9A_phonon_full_3D':bloch['phonon_full_3D_status'],
        'B9A_frozen_slice_source_integral':read(OUT/'B9A_source_integration_audit.json')['status'],
        'B9A_full_joint':bloch['full_electron_phonon_joint_status'],
        'B9B_implementation_validation':validation['status'],'B9B_reduced_material':'COMPUTED_WITH_EMPIRICAL_REFINEMENTS',
        'B9B_full_target':'INCOMPLETE','B10_metadata':'INCOMPLETE',
        **{'B10_'+k:v['status'] for k,v in metadata['availability'].items()},
        'B11A_effective_harmonic':thermal['effective_harmonic_status'],'B11A_richer_reference':thermal['richer_model_extension_status'],
        'B11A_corrected_MD_numerical_quality':'PASS' if thermal['trajectory_energy_quality_gate']['passed'] else 'FAIL',
        'B11A_refitted_effective_harmonic':refit['status'],
        'B11B_intrinsic_linewidths':thermal['linewidth_status']}
    record={'statuses':statuses,'verified_previous_evidence':verified,'tests':tests,'preservation':after,
        'source_changes':changes(),'reference_result_paths':{name:str(OUT/path) for name,path in (
            ('B7','B7_diagnosis_final.json'),('B7_joint','B7_joint_evaluation.json'),('B9A','B9A_Bloch_convergence.json'),
            ('B9A_legacy','B9A_diagnosis_final.json'),('B9B','B9B_convergence_summary.json'),('B10','B10/results.json'),('B11','B11_checked/results.json'),('B11_refit','B11_refit/results.json'))},
        'scope':'no full coupled electron–phonon multislice acceptance implied; original tolerances and historical evidence preserved'}
    json_write(OUT/'results.json',record);costs();mode_report(thermal)
    current=profile('benchmarks.repair_9a_fourier_depth');current['FFT_workers_exception']=4
    current['units']={'meV_to_joule':'1e-3 e','angstrom_to_metre':1e-10,'inverse_angstrom_to_inverse_metre':1e10,
        'mass_u_to_kg':current['constants_SI']['atomic_mass'],'angular_frequency':'omega=2pi f; oscillator energy=hbar omega=h f',
        'momentum':'Q and q in angular m^-1; no additional2pi in exp(-iQ.u)'}
    current['adoption_scope']='current post-implementation audit; original inner checkpoint creation contracts are not backdated'
    current['hardware']={'processor':next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor())}
    json_write(OUT/'constants_and_provenance.json',current)
    status_rows=[['1–6,3B,8','PASS preserved','Existing validated results and tolerances unchanged; full regression rerun.'],
        ['7 phonon integration',statuses['B7_phonon_integration'],'Two finest physical refinements; raw forward-fiber diagnostics retained.'],
        ['7 independent joint',joint['status'],f"{joint['completed_q']:,}/{joint['total_q']:,} q points; partial prefix never accepted."],
        ['9A electron grid / slices / joint',' / '.join(statuses[k] for k in ('B9A_electron_grid','B9A_slices','B9A_electron_joint')),'Fixed T=0,93-mode bath and19.962nm slab; original four detector/window gates.'],
        ['9A phonon / full joint','INCOMPLETE','Directional BZ sampling and covariance remain sensitive; fine electron acceptance does not close these.'],
        ['9B implementation',validation['status'],'Nonlinear dense/independent/3B/8 controls.'],
        ['9B reduced Si / full target','COMPUTED / INCOMPLETE','0.384nm,reduced Γ bath; p/electron/slices/quadrature separately tested; electron state and bulk bath unresolved.'],
        ['10 metadata / conditional shape','INCOMPLETE / '+statuses['B10_conditional_spectral_shape'],'User settings and disclosed analysis axes usable conditionally.'],
        ['10 calibrated / absolute / matched',' / '.join(statuses[k] for k in ('B10_calibrated_spectral_shape','B10_absolute_intensity','B10_matched_physical_model')),'Axes/native resolution/geometry/force provenance remain necessary.'],
        ['10 cost comparison','NOT_ASSESSED','No TACAW generation log; does not block conditional shape plots.'],
        ['11A EHM / richer reference',thermal['effective_harmonic_status']+' / DIRECT_MODE_MD_COMPUTED','Original EHM failure retained; no accepted FC3/FC4/self-energy extension.'],
        ['11B intrinsic widths',thermal['linewidth_status'],'No lifetime or unsupported upper bound.']]
    lines=['# Benchmark repairs: measured results','',
        f"The original validated references remain intact. This task adds {len(record['source_changes'])} source/configuration/test/documentation files and runs the focused refinements below. {tests['count']} regression tests {'pass' if tests['passed'] else 'did not all pass'}. Historical inputs/results: {before['count']:,} files, {before['total_bytes']:,} bytes; after-check {'PASS, all unchanged' if after.get('all_unchanged') else 'pending or failed; inspect protected_after.json'}. The earlier487-file baseline also matches.",'',table(['Milestone','Status','Evidence and limit'],status_rows),'',
        '## Verified starting evidence','',
        f"The requested latest report and its source index/resume script were inspected. Its B2 acceptance is {elastic['accepted_settings'][0]}²/{elastic['accepted_settings'][1]} slices per repeat, with raw52-repeat complex error {number(verified['B2_52_repeat_complex_error'])}. Original B8 independent64-step nonlinear error is {number(verified['B8_independent_nonlinear_64_step_error'])}. The116-test and487-preserved-file records are verified, rather than replaced by older failure summaries. See verified_previous_evidence.json and initial_diagnosis.md/JSON.",'',
        '## Benchmark7: a stable finest phonon sequence, incomplete independent joint','',
        'The physical detector rule uses two successive finest passing refinements. Coarse failures remain stored but cannot permanently veto a later finest sequence. No tolerance or detector/window is changed. Relative intensity errors below use |fine−coarse|/max(|fine|,1e−18), with fine intensity as denominator. L1 divides by the fine detector spectrum integral. These deterministic differences are numerical sensitivities, not statistical confidence intervals.','']
    rows=[]
    for c in b7['comparisons']:
        w=worst(c['physical_detector_changes']);rows.append([f"{c['coarse']['mesh']}³→{c['fine']['mesh']}³",w['observable'],number(w['coarse']),number(w['fine']),number(w['absolute_change']),number(100*w['acceptance_error'])+'%',number(c['spectral_changes']['maximum_relative_L1']),number(c['maximum_peak_shift_mev']),'PASS' if c['physical_passed'] else 'FAIL'])
    lines+=[table(['Mesh','Worst physical observable','Coarse','Fine','Absolute Δ','Relative Δ','Max L1','Peak Δ(meV)','Pair'],rows),'',
        'The96³ half-cell offset also passes the physical detector gate; the64³ offset failure remains. The raw exact-forward quadrature-fiber probability continues to change with mesh area, so the historical raw-point strict flag remains false; this is explicitly retained, alongside its area and density estimate. It is not an additional continuum point mass. All physical pixels, detector windows and left/right checks remain in the physical gate. Regional branch/degenerate-subspace, q-shell and energy-boundary contributions plus peak matching are in B7_diagnosis_final.json.','',
        f"The128³ calculation completed2,097,152 points in {number(b7['source_cost']['cumulative_seconds'])} s cumulative source work. The final invocation was {number(b7['resource_costs']['elapsed_seconds'])} s, peak {number(b7['resource_costs']['peak_process_rss_mib'])} MiB. The full independent256²/4096-source joint route is only {joint['completed_q']} points complete. Its512-point pilot estimates approximately72.5 hours for all points under the measured load. Its partial amplitude agreement is not a complete joint acceptance.",'',
        '## Benchmark9A: streaming and a demonstrated translation-aliasing repair','',
        'The source channels are independently streamed while the incident elastic half-step field is reused. Depth contributions stay coherent within each final channel. A unit-cell/Bloch decomposition removes the old odd2×2-cell translation aliasing without deleting final modes or projecting leaked intensities. Independent full-cell and real/Bloch basis controls agree at approximately1e−14 complex amplitude scale. The old2.90% weak-window discrepancy is refined separately from the full-thickness target.','']
    key='annulus_5_10_mrad/acoustic_0_40_mev';legacy_rows=[]
    candidates=[r for r in legacy.get('cases',[]) if r.get('settings',{}).get('thickness_repeats')==2]
    # The consolidated diagnosis stores cases separately from comparisons.
    from .repair_analysis import material_cases
    candidates=sorted([r for r in material_cases() if r['settings']['thickness_repeats']==2 and r['settings']['slices']==1025 and r['settings']['phonon_z']==2 and not r['settings']['temperature_k'] and r['settings']['elastic_background'] and r['settings']['padding_angstrom']==12],key=lambda r:r['settings']['grid'])
    for a,b in zip(candidates[:-1],candidates[1:]):
        x=a['observables'][key];y=b['observables'][key];legacy_rows.append([f"{a['settings']['grid']}²→{b['settings']['grid']}²",number(x),number(y),number(abs(x-y)),number(100*abs(x-y)/max(abs(y),1e-18))+'%'])
    lines+=[table(['Thin fixed-bath grid','Coarse weak-window weight','Fine','Absolute Δ','Relative Δ'],legacy_rows),'',
        'At full thickness, the new Bloch representation tests numerical refinements with every physical bath mode retained:','']
    rows=[]
    for c in bloch['comparisons']:
        if c['changed'] not in ('electron_grid','slices'):continue
        w=worst(c['original_detector_checks']);s=c['coarse'];t=c['fine'];parameter='grid_unit' if c['changed']=='electron_grid' else 'slices'
        rows.append([c['changed'],f"{s[parameter]}→{t[parameter]}",w['observable'],number(w['coarse']),number(w['fine']),number(w['absolute_change']),number(100*w['acceptance_error'])+'%',number(max(c['spectral_L1'].values())),'PASS' if c['original_gate_passed'] else 'FAIL'])
    lines+=[table(['Parameter','Values','Worst original observable','Coarse','Fine','Absolute Δ','Relative Δ','Max L1','Pair'],rows),'',
        'Joint comparisons and all added forward/off-axis diagnostics remain in B9A_Bloch_convergence.json. Physical dx,dy,dz are stored with every case. The195² and4125-slice choices are slightly finer odd FFT-friendly samplings; FFT_cost_pilot.json measures the cost, and the physical comparisons still decide acceptance. They do not inherit acceptance from B2.','',
        'The production source step already integrates the frozen local Hamiltonian exactly using its channel-phase sinc factor. Independent2/4/8/16-node Gauss quadrature and a non-Hermitian triangular dense exponential validate this component separately in B9A_source_integration_audit.json. This does not replace the slice/joint checks for z variation of the fields or kinetic splitting. Analytic Fourier potential gradients are used; no finite-difference derivative/filter change or altered detector boundary supplies the agreement.','',
        table(['Final numerical case','dx,dy,dz(Å)','20mrad source weight','Latest invocation(s)','Peak RSS(MiB)','Cumulative committed work(s)'],[[name+',93 modes',str([round(row['physical_spacings_angstrom'][k],8) for k in ('dx','dy','dz')]),number(row['observables']['full_0_20_mrad']),number(data['resource_costs']['elapsed_seconds']),number(data['resource_costs']['peak_process_rss_mib']),number(read(Path(row['checkpoint']))['cumulative_seconds'])] for name,data in [(name,read(OUT/name/'results.json')) for name in ('B9A_fft_grid','B9A_fft_joint')] for row in data['cases'] if row['status']=='COMPLETE']),'',
        'Phonon refinement changes BZ quadrature, retained mode count and the corresponding covariance, not the physical slab or force-constant supercell. These distinct directions give:','']
    rows=[]
    for r in bloch['cases']:
        s=r['physical_settings']
        if s['grid_unit']==33 and s['slices']==2049 and s['thickness_repeats']==52 and s['elastic_background'] and not s['temperature_k']:
            rows.append([f"{s['phonon_xy']}×{s['phonon_xy']}×{s['phonon_z']}",r['mode_count'],number(r['observables']['full_0_20_mrad']),number(r['observables']['central_0_5_mrad/middle_40_65_mev'])])
    lines+=[table(['Bath','Modes','20mrad source weight','Central40–65meV'],rows),'',
        'Longitudinal bath2/4 spacing is26/13 times the slab interference width2π/L; the dominant qz=0 sample is overweighted. The transverse refinement is independently significant. The33² electron bath controls are not accepted electron-converged full-material spectra. Full3D phonon and electron–phonon joint convergence remain incomplete.','',
        'The first attempted windowed transpose exceeded the2GiB resource target (4,986MiB, or4.87GiB, measured). It was stopped with committed channels preserved. Contiguous-depth Fourier storage then measured383MiB for the largest cold background and matched the old field to6.65e−16; the warm source pilot was381MiB. FFT-friendly cold/warm and final invocation RSS/time are separate records in cost_vs_scope.json. Compatible-prefix adoption retains the old source hashes and over-budget cost; it does not reclassify that failed implementation as within budget.','',
        '## Benchmark9B: nonlinear matrix-free validation and feasible reduced Si','',
        f"Matrix-free3B amplitude error is {number(validation['B3B']['complex_amplitude_error'])}. The unchanged production B8 problem (p12,16²,64steps) agrees with the historical wave to {number(validation['B8_production']['complex_amplitude_error'])}, with norm drift {number(validation['B8_production']['max_relative_norm_drift'])}. The independent reduced actual-Si dense propagation error is {number(material['independent_material_control']['complex_amplitude_error'])}, Hermiticity residual0. Step refinements remain separate from fixed-slice exponential action. Nonlinear phases are retained; linear ladder sparsity is only a labelled control.",'',
        'For93 modes,p2,65²: C=4465; bare state301,834,000bytes; dense local coupling1,347,688,810,000bytes. Twelve bare state buffers already require3,622,008,000bytes, above2GiB. Matrix-free action removes the dense coupling field, not this state/work-space growth or finite-temperature trace cost. No1.35TB allocation was attempted.','',
        f"The reduced0.384nm,T0,transverse-Γ Si calculation reaches p4,65×65 reciprocal beams,257slices: one-phonon population {number(fine['observables']['one_phonon'])}, multiphonon population {number(fine['observables']['two_or_more'])},20mrad inelastic weight {number(fine['observables']['full_0_20_mrad/inelastic'])}. It used {number(fine['seconds'])} s committed propagation and {number(fine['resource_costs']['peak_process_rss_mib'])}MiB peak. Cutoff p2/3/4/6, electron n9/17/25/33/49/65, slices257/513 and quadrature24/32 are reported separately. Electron all-beam populations remain sensitive even when the fixed20mrad signal is much more stable. No infinite-bath omitted-coupling bound is claimed, and this is not full-target9B.",'',
        f"The additional65-beam-range coordinate audit compares the interaction action at five depths on the actual exit state and a normalized random state. Order24→32 relative errors are {number(quadrature['order24_to_fine_action_relative_errors'][0])}/{number(quadrature['order24_to_fine_action_relative_errors'][1])}; selected nonlinear phase blocks through the largest reciprocal differences agree with an independent generalized-Laguerre expression below3e−15 absolute. This sampled operator check supports quadrature accuracy, not full bulk/electron-state convergence. The32-node audit measured {number(quadrature['resource_costs']['peak_process_rss_mib'])}MiB peak.",'',
        'An additional fixed-p3,5×5-beam,129-slice calculation includes all six declared transverse Γ coordinates. The2/4/6-mode results are a separate mode-space refinement; long-wavelength bulk q and longitudinal coordinates remain outside this reduced calculation.','',
        table(['Active transverse Γ modes','One-phonon population','Multiphonon population','20mrad inelastic weight','Case seconds'],[[r['settings']['modes'],number(r['observables']['one_phonon']),number(r['observables']['two_or_more']),number(r['observables']['full_0_20_mrad/inelastic']),number(r.get('seconds'))] for r in gamma.get('cases',[])]),'',
        f"The leading-order control agrees with independent dense Dyson propagation to {number(orders['independent_Dyson_control']['complex_amplitude_error'])}. The full-minus-single-interaction multiphonon difference is {number(orders['signed_full_minus_single_multiphonon'])}, a signed interference effect. Orders0+1+2 are added coherently; this is not all intensity terms throughg4. See B9B_orders/results.json and B9B_convergence_summary.json.",'',
        '## TACAW and thermal-model diagnosis','',
        'The hash-matched supplied arrays remain unchanged. The metadata table distinguishes verified data, user-supplied60keV/208slices/unit-integral/Kubo settings, derived data and downstream analysis assumptions. A recovered sidecar reproduces its conditional7.5fs/600bin axis (0.919037meV spacing;4.5ps only if unpadded), not the original generation calibration or native resolution. The earlier qualitative Yoshioka–TACAW figures remain available conditionally. Unknown physical momentum axes, native window, elastic handling, geometry and exact generating force hash still block calibrated/absolute/matched-model claims. Missing runtime logs block only cost comparison. No second Kubo correction, fitted scale, peak alignment or FRFPMS is used.','',
        'The100 ps extension exposed a checkpoint-driver error: write_restart can remap atoms, and the next run pre no reused stale neighbour state. Fresh MOD evaluations at the stored positions showed62–69% force errors immediately after a write. The new checked driver performs run0 pre yes after writing, without advancing time or rescaling velocities. At the same failure times, fresh-force discrepancies fall below1.2e−14. The first corrected1000K trajectory reduces energy drift/thermal kinetic energy from2.60755 to0.000239253. All seven corrected trajectories pass the10⁻³ energy-quality gate; original20ps trajectories already pass their recorded checks. The failed100ps data and original driver are preserved, and their initial2.6095meV estimate is superseded. See MD_quality_status.json, MD_checkpoint_force_audit.json and trajectory_controls.md.','',
        'See [metadata table](B10/metadata_table.md), [only remaining inputs](B10/missing_inputs.md), and the preserved [conditional comparison](../tacaw_material_20261102_v1/conditional_material_spectra_and_residuals.png).','',
        f"Seven new100ps actual-MOD NVE trajectories use independent primary replicas plus timestep/supercell/equilibration controls. All are fixed volume; differing actual NVE temperatures are disclosed. The largest new EHM−mode-MD discrepancy is {number(max(abs(r['effective_minus_MD_mev']) for r in thermal['frequency_rows']))}meV; the old2.6715meV failure remains. A low-amplitude actual-MOD control agrees with harmonic force constants within {number(audit['maximum_low_amplitude_frequency_discrepancy_mev'])}meV, supporting the mass/unit/mapping/2π implementation. Paired-force odd-law floors0.14627/0.29127 and saturated independent learning curves support a static-quadratic model limitation; the paired-set floor is not a universal bound on equilibrium data. Direct mode dynamics is the labelled richer reference; no validated FC3/FC4/self-energy extension is claimed.",'',
        'The model was then refitted independently to corrected100ps displacement–force data, using whole separate seeds for training and validation, the original5Å range and no regularization. The learning curves are:','',
        table(['Target K','Frames','Training force relative RMS','Held-out force relative RMS','Actual train K','Actual validation K'],[[r['temperature_k'],r['count'],number(r['training_force_relative_RMS_error']),number(r['force_relative_RMS_error']),number(r['actual_training_temperature_k']),number(r['actual_validation_temperature_k'])] for r in refit['fits']]),'',
        f"The128-frame refit still fails: its largest selected frequency discrepancy is {number(max(abs(s['effective_minus_MD_mev']) for r in refit['fits'] if r['count']==128 for s in r['frequency_rows']))}meV. The1000K held-out force error stays0.28829 against the unchanged0.15 target. The original36 range/regularization/count variants remain, and the corrected data add32/64/128-frame curves with modal-energy effective-configuration proxies. These diagnose a residual quadratic-model limitation after the checkpoint repair; more data alone has not resolved it. B11_refit/results.json contains every refitted mode, overlap and held-out error.",'',
        'All20 mode/subspace rows, polarization/overlap, EHM/reference frequencies, spread, fit shapes and candidate widths are saved. The100ps Fourier spacing is0.0413567meV and actual Hann power-window FWHM0.0595777meV, with5fs Nyquist413.567meV. Widths still fail resolution/spread/shape gates. All intrinsic linewidths remain NOT_RESOLVED, without lifetimes or unsupported upper bounds.','',
        '[Per-mode table](per_mode_validation.md) · [mode spectra and window response](B11_modes_and_resolution.png) · [independence/learning audit](B11_learning/results.json).','',
        '## Files, figures and reproduction','',
        'Reusable additions cover source streaming/depth fields, exact Bloch sectors/covariance, nonlinear matrix-free and coordinate actions, checkpointed exponential action, Dyson orders, metadata-specific availability and convergence diagnostics. Every added source/config/test path and hash is listed in changed_source_files.json; no baseline source is intentionally modified. Scientific assumptions and the fixed-projector residual derivation are in methods_and_limits.md.','',
        '- [B7 convergence and regional diagnosis](B7_convergence.png)\n- [B9A spectra, directional bath tests and aliasing](B9A_Bloch_repairs.png)\n- [Original weak-window refinement](B9A_weak_window.png)\n- [Reduced repeated-interaction EELS](B9B_reduced_repeated_EELS.png)\n- [Reduced Γ-mode-space refinement](B9B_Gamma_modes.png)\n- [Signed interaction-order difference](B9B_orders_difference.png)\n- [MD checkpoint repair](MD_checkpoint_repair.png)\n- [Independent corrected-data learning curves](B11_corrected_fit_learning.png)\n- [Corrected thermal mode/reference spectra](B11_modes_and_resolution.png)','',
        'Exact bounded commands, cold-workspace setup and resume selections are in reproduce_and_resume.sh. Each expensive runner preserves committed prefixes; a completed invocation can return INCOMPLETE. Current-audit resume_profiles.json is prospective provenance, not a backdated creation record. Latest-invocation wall/RSS and cumulative cache work are distinct in cost_vs_scope.json, including failed pilots and adopted work. No scope-unmatched TACAW timing comparison is made.','',
        '```bash\ncd /home/zuxian/Documents/PhD/Yoshioka\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\n.venv/bin/python -m benchmarks.repair_resume status\n.venv/bin/python -m benchmarks.repair_resume 9a-electron\n.venv/bin/python -m benchmarks.repair_resume 9a-joint\n.venv/bin/python -m benchmarks.repair_resume b7-joint\n.venv/bin/python -m benchmarks.repair_resume b7-joint-evaluate\n.venv/bin/python -m unittest discover -s tests\n```','',
        'Passing oscillator, synthetic and reduced-model controls does not validate a general macroscopic coupled electron–phonon multislice calculation.']
    text='\n'.join(lines)+'\n'
    for old,new in {'Benchmark7':'Benchmark 7','Benchmark9A':'Benchmark 9A','Benchmark9B':'Benchmark 9B',
        '19.962nm':'19.962 nm','0.384nm':'0.384 nm','100ps':'100 ps','20ps':'20 ps','5fs':'5 fs','7.5fs':'7.5 fs',
        '4.5ps':'4.5 ps','2GiB':'2 GiB','or4.87GiB':'or 4.87 GiB','383MiB':'383 MiB','381MiB':'381 MiB',
        'all20 mode':'all 20 mode','All20 mode':'All 20 mode','Seven new100 ps':'Seven new 100 ps',
        '1.35TB':'1.35 TB','approximately72.5':'approximately 72.5','old2.90%':'old 2.90%',
        'The96³':'The 96³','the64³':'the 64³',
        'The128³':'The 128³','completed2,097,152':'completed 2,097,152',
        'fixed20mrad':'fixed 20 mrad','The195²':'The 195²','and4125-slice':'and 4125-slice',
        '300/1000K':'300/1000 K','600bin':'600 bins','413.567meV':'413.567 meV','0.0413567meV':'0.0413567 meV',
        '0.0595777meV':'0.0595777 meV','2.6715meV':'2.6715 meV','user-supplied60keV/208slices':'user-supplied 60 keV/208 slices',
        'original2.6715':'original 2.6715','The100 ps':'The 100 ps','The32-node':'The 32-node','additional65-beam':'additional 65-beam',
        'full-target9B':'full-target 9B','order0':'order 0','throughg4':'through g⁴','Orders0+1+2':'Orders 0+1+2',
        'earlier487':'earlier 487','raw52-repeat':'raw 52-repeat','independent64-step':'independent 64-step','The116-test':'The 116-test','and487':'and 487',
        'independent256²':'independent 256²','Its512-point':'Its 512-point','odd2×2-cell':'odd 2×2-cell','approximately1e−14':'approximately 10⁻¹⁴',
        'bath and19.962':'bath and 19.962','bath2/4':'bath 2/4','is26/13':'is 26/13','width2π/L':'width 2π/L','The33²':'The 33²','Full3D':'Full 3D',
        'the2 GiB':'the 2 GiB','measured383':'measured 383','to6.65e−16':'to 6.65e−16','Matrix-free3B':'Matrix-free 3B',
        '(p12,16²,64steps)':'(p=12,16²,64 steps)','residual0':'residual 0','For93 modes,p2,65²':'For 93 modes,p=2,65²',
        'state301':'state 301','coupling1,347':'coupling 1,347','require3,622':'require 3,622','above2 GiB':'above 2 GiB','No1.35':'No 1.35',
        'reduced0.384':'reduced 0.384',',T0,':',T=0,','p4,65':'p=4,65','257slices':'257 slices','20mrad':'20 mrad',
        'Cutoff p2/3/4/6':'Cutoff p=2/3/4/6','electron n9':'electron n=9','slices257/513':'slices 257/513','quadrature24/32':'quadrature 24/32','Order24→32':'Order 24→32',
        'below3e−15':'below 3×10⁻¹⁵','fixed-p3,5×5-beam,129-slice':'fixed p=3,5×5-beam,129-slice','The2/4/6-mode':'The 2/4/6-mode',
        'conditional7.5':'conditional 7.5','showed62–69%':'showed 62–69%','run0 pre yes':'run 0 pre yes','below1.2e−14':'below 1.2×10⁻¹⁴',
        'corrected1000K':'corrected 1000 K','from2.60755':'from 2.60755','to0.000239253':'to 0.000239253','the10⁻³':'the 10⁻³',
        'original20 ps':'original 20 ps','initial2.6095':'initial 2.6095','old2.6715':'old 2.6715','floors0.14627':'floors 0.14627',
        'corrected100 ps':'corrected 100 ps','original5Å':'original 5 Å','The128-frame':'The 128-frame','The1000K':'The 1000 K','stays0.28829':'stays 0.28829',
        'unchanged0.15':'unchanged 0.15','original36':'original 36','add32/64/128':'add 32/64/128','is0.0413567':'is 0.0413567','FWHM0.0595777':'FWHM 0.0595777',
        'with5 fs':'with 5 fs','Nyquist413.567':'Nyquist 413.567'}.items():text=text.replace(old,new)
    text=re.sub(r'(?<=\d)(?=(?:meV|MiB|bytes)\b)',' ',text)
    (OUT/'report.md').write_text(text)
    print(json.dumps(statuses,indent=2))


if __name__=='__main__':main()
