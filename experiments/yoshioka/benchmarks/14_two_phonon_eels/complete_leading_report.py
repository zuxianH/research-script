"""Rebuild the evidence index and figures from executed complete-pair arrays."""
import json
from pathlib import Path
import re
import time
import numpy as np
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .workflow_io import ROOT,json_write,sha256
from .complete_leading_eels import DEFAULT,finite_json
from .approximate_eels import DETECTORS


def main():
    output=DEFAULT;bulk=output/'bulk_static_v2';rows=[]
    for scheme in ('born','adapted','stratified'):
        d=json.loads((bulk/f'{scheme}_analysis.json').read_text());latest=d['prefixes'][-1]
        accept=json.loads((bulk/f'{scheme}_acceptance.json').read_text())
        counts=[];seconds=[];rss=[]
        for p in sorted((bulk/'pairs'/scheme).glob('batch*/state.json')):
            state=json.loads(p.read_text());counts.append(state['completed_pairs']);seconds.append(state['cumulative_seconds'])
        for p in (bulk/'invocations').glob(f'*_{scheme}_batch*.json'):
            rss.append(json.loads(p.read_text())['peak_process_rss_mib'])
        window_ratios=[v for key,val in latest['raw'].items() if key.startswith(('loss_','gain_')) for v in val['relative_halfwidth95'] if v is not None]
        shape=[q['windows']['5_160']['shape_L1'] for q in d['comparisons']]
        n=latest['total_pairs'];perpair=latest['mean_pair_seconds'];primary=latest['raw']['all_loss']['relative_halfwidth95'][0]
        forecast={}
        for name,value in (('primary_loss',primary),('worst_detector_loss',max(latest['raw']['all_loss']['relative_halfwidth95'])),('all_windows',max(window_ratios)),('worst_shape',max(shape))):
            needed=n*(value/.1)**2
            unit=36*8 if scheme=='stratified' else 8
            balanced=int(unit*2**max(0,int(np.ceil(np.log2(max(needed/unit,1))))))
            forecast[name]={'conditional_total_pairs':balanced,'extra_summed_worker_hours':max(0,balanced-n)*perpair/3600,
                'assumption':'Illustrative N^-1/2 law; unverified for rare features and randomized QMC. Excludes changed-grid cost, deterministic bias and feature identification. Not convergence evidence or an execution authorization.'}
        rows.append({'scheme':scheme,'pairs':n,'counts':counts,'primary_raw_loss':latest['raw']['all_loss']['mean'][0],
            'primary_halfwidth95':latest['raw']['all_loss']['halfwidth95'][0],
            'primary_relative_halfwidth95':primary,'all_detector_loss_relative_halfwidth95':latest['raw']['all_loss']['relative_halfwidth95'],
            'shape_L1_5_160':shape,'worst_window_relative_halfwidth95':max(window_ratios),
            'summed_worker_wall_seconds':sum(seconds),'mean_timed_pair_seconds':perpair,'max_worker_peak_rss_mib':max(rss),
            'all_gates':accept['all_gates'],'conditional_budgets':forecast,
            'raw_analysis_sha256':sha256(bulk/f'{scheme}_analysis.json')})
    profile=json.loads((bulk/'profile.json').read_text())
    json_write(output/'cost_vs_accuracy.json',finite_json({'same_physical_target':'Static Taylor,4repeats=1.535553nm,300K,60keV,49²/385steps; all four signed sectors, four detectors,2meV FWHM.',
        'schemes':rows,'exact_wave_pruning_profile':profile,'timing_scope':'Summed per-worker wall times are not elapsed job time or CPU time. Concurrent calculations share the documented workstation. No scheme reached matched target accuracy; variance/cost summaries are diagnostic.'}))
    # Absolute component spectra and honest independent-replicate uncertainty.
    final=next(r for r in rows if r['scheme']=='stratified');n=final['pairs']//8
    with np.load(bulk/f'stratified_n{n}_analysis.npz') as z:
        energy=z['energy_mev'];select=(energy>=0)&(energy<=160)
        comps={k:z[f'spectra_{k}'].mean(axis=0) for k in ('direct_weights','sequential_weights','interference_weights','weights')}
        batch=z['batch_spectra'];mean=batch.mean(axis=0);half=student_t.ppf(.975,7)*batch.std(axis=0,ddof=1)/np.sqrt(8)
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,constrained_layout=True)
    labels={'direct_weights':'Direct quadratic','sequential_weights':'Sequential linear','interference_weights':'Signed interference','weights':'Coherent total'}
    colors={'direct_weights':'#d58917','sequential_weights':'#477bbb','interference_weights':'#ae4e76','weights':'#152c3b'}
    for j,ax in enumerate(axes.flat):
        for key in labels:ax.plot(energy[select],comps[key][select,j],label=labels[key],color=colors[key],lw=1.4)
        ax.fill_between(energy[select],(mean-half)[select,j],(mean+half)[select,j],color='#152c3b',alpha=.13,label='Total 95% interval')
        ax.axhline(0,color='0.5',lw=.5);ax.set_title(['0–20 mrad','0–5 mrad','5–10 mrad','10–20 mrad'][j]);ax.set_xlabel('Energy loss (meV)');ax.set_ylabel('Raw model probability / meV')
    axes[0,0].legend(fontsize=8);fig.suptitle(f'Complete leading changed-occupation pair spectrum • {final["pairs"]} pairs • UNCONVERGED\nStatic Taylor reference, 300 K, 1.536 nm, 60 keV; fixed 2 meV resolution',fontsize=12)
    fig.savefig(output/'complete_pair_components.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    x=np.arange(4);width=.24
    for i,r in enumerate(rows):
        axes[0].bar(x+(i-1)*width,r['all_detector_loss_relative_halfwidth95'],width,label=f'{r["scheme"]} ({r["pairs"]})')
        axes[1].bar(x+(i-1)*width,r['shape_L1_5_160'],width)
    for ax in axes:ax.set_xticks(x,['0–20','0–5','5–10','10–20']);ax.set_xlabel('Detector (mrad)');ax.axhline(.1,color='black',ls='--',lw=1,label='Unchanged0.1 target')
    axes[0].set_ylabel('Integrated-loss relative95% half-width');axes[1].set_ylabel('Nested-count spectral shape L1 (5–160meV)');axes[0].legend(fontsize=8)
    fig.suptitle('Distinct statistical gates; no candidate has converged')
    fig.savefig(output/'sampling_gates.png',dpi=160);plt.close(fig)
    # Dependency record: affected models remain scoped and historical failures retained.
    dependency={'fixed_harmonic_reference':'Unchanged0K MOD force constants; independent of defective MD trajectories.',
        'new_bulk':'Static undisplaced elastic reference differs from historical thermal-DW sequential-only model. Its new direct vertex and interference invalidate transferring that old spectrum acceptance to this new model.',
        'old_sequential_results':'Preserved. Still valid as limited-order controls at their original reference; incomplete as complete displacement-order-two amplitudes.',
        'elastic_reference_families':'Bare Taylor and thermal normal ordering have different epsilon0 references; exactepsilon1 identity tested in reduced model, not proof their truncated terms coincide.',
        'MD':'See thermal_comparison/dependency_audit.json. Original defective-generation dependencies are retained as historical/superseded candidates. New fits use corrected independent trajectories.',
        'matrix_free_changes':'Optional real transverse finite bath; old Gamma defaults rerun by regression and independent dense comparison. Longitudinal active modes are rejected in matrix-free APIs and tested separately by dense reference.',
        'new_pair_missing_creation_byte_hash':'Recorded explicitly; currentSi parameters match56preserved creation contracts and prior protected resume byte hash. Future guarded resumes verify that evidence, liveconfig, fullproposalhash and everyraw file.',
        'numerical_control_cache':'Creation contracts and raw hashes added; existing completed controls anchored by preservedresults_fine.json, explicitly retrospective. Changed physical kernels require new versions.',
        'external_TACAW':'User confirmed exactgenerating records unavailable locally; absolute comparison remains blocked. No fitted scale/energy shift or repeatedKubo correction.'}
    json_write(output/'dependency_invalidation.json',dependency)
    log=(output/'final_regression.log').read_text();m=re.search(r'Ran (\d+) tests in ([\d.]+)s',log)
    tests={'tests':int(m.group(1)) if m else None,'seconds':float(m.group(2)) if m else None,'passed':bool(m and '\nOK\n' in log),'log_sha256':sha256(output/'final_regression.log')}
    status={'complete_leading_amplitude':'IMPLEMENTED_AND_VERIFIED_REDUCED_ACTUAL_SILICON','complete_bulk_two_phonon':'NOT_CONVERGED_STATISTICAL_AND_ELECTRON_GATES',
        'B7':'PHYSICAL_SEQUENCE_HISTORICAL_PASS;INDEPENDENT_JOINT_PARTIAL768_OF2097152',
        'B9A':'CONDITIONAL_THIN_LONGITUDINAL_PASS;3D_THIN_AND_ORIGINAL_THICK_TARGET_INCOMPLETE',
        'B9B':'REDUCED_NON_GAMMA_FULL_NONLINEAR_MATRIX_FREE_VERIFIED;ORIGINAL_TARGET_INCOMPLETE',
        'B10':'FRESH_REDUCED_FROZEN_COMPARISON_VERIFIED;EXTERNAL_BLOCKED_USER_CONFIRMED_MISSING_METADATA',
        'B11A':'FAIL_COMBINED_VALIDATION_CUBIC_AND_QUARTIC_THERMAL_INSTABILITY','B11B':'NOT_RESOLVED_NO_LIFETIMES',
        'regression':tests,'no_overall_material_PASS':True,'no_external_publication':True,
        'generated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    json_write(output/'statuses.json',status)
    table='\n'.join(f'| {r["scheme"]} | {r["pairs"]} | {r["primary_raw_loss"]:.6g} ± {r["primary_halfwidth95"]:.3g} | {100*r["primary_relative_halfwidth95"]:.1f}% | {r["shape_L1_5_160"][0]:.3f} | {r["summed_worker_wall_seconds"]:.1f}s | {r["max_worker_peak_rss_mib"]:.1f}MiB |' for r in rows)
    text=f'''# Complete leading quantum vibrational EELS: executed extension

The complete direct-quadratic plus sequential-linear changed-occupation amplitude is implemented and verified against an independent actual-silicon nonlinear reference. **The full bulk spectrum and original~20nm material targets remain incomplete.** This is measured code-and-calculation progress, not an overall material PASS.

## What changed and what ran

New quadratic finite-atom fields retain longitudinal displacement, Fourier/Bloch phases and the repository sign convention. The direct amplitude and both sequential orderings are coherently summed for the same target state; direct intensity, sequential intensity, signed interference and total remain separate. All four finite-temperature sign sectors are evaluated analytically without truncating initial populations. Static Taylor and thermal normal-ordered elastic reference families are explicit. Identical-mode normalization and the epsilon-scaled complex P2 amplitude are checked independently in reduced actual silicon; the phonon Hamiltonian stays fixed. See [methods](methods.md) and [material evidence](material_reference/report.md).

The final regression currently records **{tests['tests']} tests, {'PASS' if tests['passed'] else 'not verified'}, {tests['seconds']}s**; the starting suite had184 passing tests in24.186s. Raw logs retain real failures and repairs. [Regression log](final_regression.log), [baseline](baseline_tests.log), [full statuses](statuses.json).

The calculation set includes2240 continuous bulk pair evaluations (64 pilot + 512 Born + 512 adapted + 1152 stratified), paired electron/slice controls, independent dense/Fourier/thermal tests, repaired-MD stop/resume, richer force models and their dynamical failure controls, mode-linewidth audits, and fresh matched reduced material correlations. Individual completed prefixes and raw batch spectra are retained; no submitted or partial job is called complete.

## Complete bulk pair integration

Same model for every row: static Taylor elastic reference, four silicon repeats (1.535553 nm), 300 K, 60 keV, 49² electron grid, 385 depth steps, four detectors and 2 meV FWHM. Raw positive-energy changed-pair weight is reported per declared model incident normalization; it is not a calibrated experimental cross section. Each method uses eight independent scrambles. The adaptive method uses a separate pilot and independent production seeds. Branch stratification preserves a balanced 2^p Sobol sequence in every one of 36 strata.

| Sampler | Pairs | Primary raw loss ± 95% half-width | Relative half-width | Shape L1,5–160 | Summed worker wall time | Peak worker RSS |
|---|---:|---:|---:|---:|---:|---:|
{table}

All candidates fail the unchanged integrated/energy-window/shape/feature gates. The scalar loss interval does not certify the spectrum. Slice refinement passes on matched nodes, while electron sensitivity still fails through 125 pixels (97→125: primary loss changes 1.68%, and the 5–10 mrad window changes 23.77%). These are paired-node sensitivities, not a global BZ error bound. The finite-model bulk result remains numerically unaccepted. [Every window/gain gate and paired covariance-aware ratios](bulk_static_v2/stratified_acceptance.json), [all candidate costs and conditional budgets](cost_vs_accuracy.json), [numerical controls through 125 pixels](bulk_numerical_controls/results_larger.json).

The strong signed interference shows why adding separate positive intensities would be wrong. Orthogonal target states with equal energy are not coherently merged. Returning-to-initial-state paths are excluded from this changed-occupation pair observable and are not appended as positive zero-loss intensity. This leading pair result omits A1*A3 and A0*A4 terms of a complete fourth-order total spectrum. There is no global probability renormalization.

![Absolute component spectra](complete_pair_components.png)

![Independent acceptance gates](sampling_gates.png)

## Verification and scope by workstream

- **A:** actual-Si P2U(epsilon)|0>/epsilon² converges to direct+sequential complex amplitudes with the expected epsilon² residual. Direct-only/sequential-only, identical/distinct modes, quadrature/Fock working dimensions and both elastic-reference conventions are tested. Weak central intensity cancellation required a more accurate exponential action, not a relaxed tolerance.
- **B7/B9A:** the original independent joint numerical reference has768 of 2,097,152 points completed in a new audited copy. Roughly40.5 summed worker hours remain at the measured contracted rate. The thin fixed-qxy 16/32/64 longitudinal Born-control sequence passes original 1% / 0.2 meV criteria. Full transverse-bath refinement fails. The original 19.962 nm depth-source diagnostics, shifted/guided quadratures and independent stratified replicas remain underresolved in weak windows. [Integration report](integration/REPORT.md).
- **B9B:** both matrix-free actions now accept explicit real transverse finite-supercell standing-wave baths beyond Γ; default Γ behavior remains tested. Full nonlinear finite-bath propagation, diffraction and repeated interactions are retained. Longitudinal active modes are explicitly rejected by those actions and remain available in the independent dense reference. Occupation, initial thermal mass, work quadrature and electron/z refinement are separately recorded. [Nonlinear/material report](material_reference/report.md).
- **B11A:** real interrupted/restarted MOD trajectories match continuous arrays bitwise and fresh forces agree. Cubic and quartic candidates substantially improve held-out forces, yet both fail 1000 K dynamic stability, reproduced at half timestep. Quartic force errors 0.02142 / 0.05805 do not establish acceptance. Fixed harmonic evidence is unaffected.
- **B11B:**56 additional window/duration/seed/fit/profile experiments do not resolve intrinsic widths. No lifetime is reported. Width/lifetime conventions and finite-window effects remain explicit. [Thermal-model report](thermal_comparison/report.md).
- **B10/F:** exact frozen quantum correlation and channel sums agree for the same reduced material operator; thermal projection and linearized harmonic FRFPMS controls are verified. Classical harmonic trajectories and full finite-channel propagation are separately identified. TRPMD and nonlinear corrected FRFPMS are not fabricated. The user confirms that original generating metadata are unavailable locally; old TACAW arrays remain blocked for calibrated comparison despite successful fresh reduced controls.

## Measured optimization

Exact disconnected-wave removal reduces 25 wave fields to 17 for the declared pair observable, giving{profile['speedup']:.2f}×on a two-node matched-amplitude timing pilot; it is not a complete-production accuracy speedup. Vectorization alone did not improve that pilot. The independent B7 contraction preserves the inverse-FFT/finite-difference/Simpson reference, while resident ephemeral fields improve the bounded thick-source run at increased measured memory. The reduced nonGamma FFT matrix-free action agrees with dense/ProductPhase amplitudes and is 30.48× faster than ProductPhase on the matched p3, n5, 65-step case (complex-vector difference 4.73e-17). Quadrature 16 versus 24 gives 3.09× on its checked finite point. These are finite-Hamiltonian solver measurements. No target-accuracy speedup is claimed for the unconverged bulk samplers. [Component wall/RSS profiles](bulk_static_v2/component_profiles), [source storage optimization](integration/resident_source_optimization.json).

## Remaining resources and inputs

Completing B7 needs its recorded~40.5 worker hours and checkpointed production allocation. Full3D/thick integration, stable thermal force models, intrinsic linewidth identification, finite-temperature bath convergence and electron accuracy remain separate blockers. Bulk inverse-square-root forecasts are conditional estimates only; weak-window and feature behavior do not yet justify a promised sample count. The new spectrum's numerical/physical scope cannot inherit historical sequential-only acceptance. The original TACAW force, axes, timing/window, geometry and discarded absolute normalization are genuinely unavailable locally.

Exact bounded commands and guarded resumes are in [resume.sh](resume.sh), [integration resume](integration/REPORT.md#reproduce-and-resume) and the material/thermal reports. Implemented-but-not-run and unsupported paths are distinguished in [implementation scope](implementation_scope.json). Low-level analysis gate booleans are preliminary monitors; the complete acceptance JSON is authoritative. The next actual action, completed counts and active jobs are in [checkpoint](checkpoint.json). No push, publication, scheduler submission, reset or deletion of historical evidence occurred. Source revision is represented by preserved source bytes and SHA256 because this directory has no Git history or configured research launcher. [Starting manifest](starting_manifest.json), [dependency/invalidation record](dependency_invalidation.json), [source/output manifest](final_manifest.json).
'''
    (output/'report.md').write_text(text)
    print('Wrote report/status/cost/PNG artifacts',flush=True)

if __name__=='__main__':main()
