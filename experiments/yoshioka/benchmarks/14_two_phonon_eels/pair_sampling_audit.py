"""Read-only complete-pair sampling audit and paired control-variate statistics.

Never propagates an electron, changes a historical checkpoint, or treats a
proposal-density table as a known complete-pair integral. Independent scramble
integrals, rather than individual weighted evaluations, define uncertainty.
Run directly from the project root; the public helpers are importable with
``importlib.import_module('benchmarks.14_two_phonon_eels.pair_sampling_audit')``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.stats import t as student_t

from benchmarks.complete_acceptance import evaluate, paired_ratio
from benchmarks.complete_leading_eels import finite_json, nodes
from benchmarks.two_phonon_analysis import AXIS, WINDOWS, PROCESSES, spectral_comparison
from benchmarks.approximate_eels import DETECTORS
from benchmarks.born_quadrature import line_spectra
from yoshioka.phonon_pair_importance import BornPairProposal
from yoshioka.stratified_pairs import BranchStratifiedPairs

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'benchmark_results/14_two_phonon_eels/complete_leading_20261102_v1'
DEFAULT = ROOT/'benchmark_results/14_two_phonon_eels/pair_convergence_20261102_v1/sampling_audit'
COMPONENTS = ('direct_weights', 'sequential_weights', 'interference_weights', 'weights')
WINDOW_NAMES = tuple(WINDOWS)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(finite_json(value),indent=2,sort_keys=True)+'\n')
    temp.replace(path)


def raw_vector(energies, weights):
    """All fixed unbroadened acceptance windows; signs remain state-resolved."""
    e=np.asarray(energies); w=np.asarray(weights)
    if e.shape[-1]!=4 or w.shape!=e.shape+(len(DETECTORS),):
        raise ValueError('Expected four changed-state sign sectors and four detectors')
    values=[]
    for key,(lo,hi) in WINDOWS.items():
        choose=(e>=lo)&(e<hi)
        if key=='all_loss': choose=e>0
        if key=='all_gain': choose=e<0
        values.append(np.sum(w*choose[...,None],axis=-2))
    return np.stack(values,axis=-2)


def estimate(batches):
    a=np.asarray(batches,dtype=float); count=len(a)
    if count<2: raise ValueError('At least two independent scramble integrals are required')
    mean=a.mean(axis=0); variance=a.var(axis=0,ddof=1)/count
    half=student_t.ppf(.975,count-1)*np.sqrt(variance)
    return {'mean':mean,'standard_error':np.sqrt(variance),'halfwidth95':half,
            'relative_halfwidth95':np.divide(half,abs(mean),out=np.full_like(mean,np.inf),where=abs(mean)>1e-18)}


def paired_control_variate(fine, coarse, coarse_baseline, relationship):
    """I_f = estimated I_c + mean(F_f-F_c), with all covariance explicit.

    Inputs are complete-vector *scramble integrals*. The coefficient is fixed
    at one. ``independent`` requires separate baseline scrambles; ``paired``
    requires baseline row b and correction row b belong to the same independent
    replicate. An unknown relationship is rejected. There is no zero-variance
    known baseline assumption and no fit on evaluation data.
    """
    f=np.asarray(fine,dtype=float);c=np.asarray(coarse,dtype=float)
    b=np.asarray(coarse_baseline,dtype=float)
    if f.shape!=c.shape or f.shape[1:]!=b.shape[1:] or min(len(f),len(b))<2:
        raise ValueError('Incompatible complete-vector independent scramble arrays')
    if not all(np.isfinite(x).all() for x in (f,c,b)):
        raise ValueError('Nonfinite paired control-variate input')
    n=len(f);m=len(b);d=f-c; mean=b.mean(axis=0)+d.mean(axis=0)
    fine_variance=f.var(axis=0,ddof=1)/n
    correction_variance=d.var(axis=0,ddof=1)/n
    baseline_variance=b.var(axis=0,ddof=1)/m
    fc_cov=np.sum((f-f.mean(axis=0))*(c-c.mean(axis=0)),axis=0)/(n-1)
    if relationship=='independent':
        cov=np.zeros_like(mean);variance=baseline_variance+correction_variance
        denominator=baseline_variance**2/(m-1)+correction_variance**2/(n-1)
        df=np.divide(variance**2,denominator,out=np.full_like(mean,float(min(n,m)-1)),where=denominator>0)
    elif relationship=='paired':
        if n!=m: raise ValueError('Paired baseline and correction need the same independent replicate count')
        cov=np.sum((b-b.mean(axis=0))*(d-d.mean(axis=0)),axis=0)/(n-1)/n
        # Direct variance of the sum avoids cancellation of near-equal terms.
        variance=(b+d).var(axis=0,ddof=1)/n
        df=np.full_like(mean,n-1,dtype=float)
    else:
        raise ValueError('Explicit independent or paired baseline relationship required')
    half=student_t.ppf(.975,df)*np.sqrt(variance)
    return dict(mean=mean,halfwidth95=half,standard_error=np.sqrt(variance),degrees_of_freedom=df,
        fine_mean=f.mean(axis=0),coarse_paired_mean=c.mean(axis=0),coarse_baseline_mean=b.mean(axis=0),
        correction_mean=d.mean(axis=0),fine_mean_variance=fine_variance,
        correction_mean_variance=correction_variance,coarse_baseline_mean_variance=baseline_variance,
        baseline_correction_mean_covariance=cov,fine_coarse_integral_covariance=fc_cov,
        correction_variance_over_fine=np.divide(correction_variance,fine_variance,out=np.full_like(mean,np.nan),where=fine_variance>0),
        total_variance_over_fine=np.divide(variance,fine_variance,out=np.full_like(mean,np.nan),where=fine_variance>0),
        baseline_relationship=relationship,coarse_integral_treated_as_exact=False,coefficient=1.,
        method='Independent scramble means; all paired fine/coarse covariance retained. Independent baseline uses Welch-Satterthwaite t intervals; paired baseline uses covariance of the summed replicate.')


def control_variate_acceptance(fine_raw,coarse_raw,baseline_raw,fine_spectra,coarse_spectra,
        baseline_spectra,relationship, *, coarse_prefix_spectra=None,
        numerical_status='NOT_ESTABLISHED',complete_coherent_pair=False):
    """Evaluate every authoritative gate for an estimated complete-pair CV.

    Spectral correction is pointwise and signed. Shape/feature gates require an
    explicitly supplied previous balanced-prefix CV spectrum. No such prefix
    means those gates fail. The caller must verify mode-pair and Hamiltonian
    matching before invoking this helper; the explicit scope flag rejects a
    sequential-only or proposal-table baseline.
    """
    if not complete_coherent_pair:
        raise ValueError('Both integrands and the baseline must contain the complete coherent pair observable')
    f=np.asarray(fine_raw);c=np.asarray(coarse_raw);b=np.asarray(baseline_raw)
    if f.shape[1:]!=(len(WINDOWS),len(DETECTORS)):
        raise ValueError('Every declared energy window and detector is required')
    result=paired_control_variate(f,c,b,relationship)
    fs=np.asarray(fine_spectra);cs=np.asarray(coarse_spectra);bs=np.asarray(baseline_spectra)
    if fs.shape!=cs.shape or fs.shape[1:]!=(len(AXIS),len(DETECTORS)) or bs.shape[1:]!=fs.shape[1:] or len(fs)!=len(f) or len(bs)!=len(b):
        raise ValueError('The full fixed-axis spectra and raw integrals must use the same independent scrambles')
    mean_spectrum=bs.mean(axis=0)+(fs-cs).mean(axis=0)
    comparisons=None if coarse_prefix_spectra is None else spectral_comparison(np.asarray(coarse_prefix_spectra),mean_spectrum)
    raw={name:{key:result[key][i] for key in ('mean','halfwidth95')} for i,name in enumerate(WINDOW_NAMES)}
    gates=evaluate(raw,comparisons,numerical_status)
    return dict(statistics=result,raw=raw,mean_spectrum=mean_spectrum,comparisons=comparisons,
        all_gates=gates,negative_spectral_bins=int(np.sum(mean_spectrum<0)),
        no_clipping_or_rescaling=True,coarse_integral_treated_as_exact=False,
        limitations='Nominal marginal 95% intervals, not simultaneous coverage. Sampling does not establish discretization error. Shape and feature comparisons are the unchanged authoritative balanced-prefix criteria.')


def _read_scheme(folder,scheme):
    masses=np.load(folder/('adapted_mass.npy' if scheme=='adapted' else 'proposal_mass.npy'))
    proposal=BornPairProposal(masses,64); strat=BranchStratifiedPairs(proposal)
    states=[json.loads(p.read_text()) for p in sorted((folder/'pairs'/scheme).glob('batch*/state.json'))]
    if len(states)!=8: raise ValueError('Historical comparison requires exactly eight independent batches')
    counts=[s['completed_pairs'] for s in states]
    if len(set(counts))!=1: raise ValueError('Unbalanced completed batch counts')
    count=counts[0];factor=36 if scheme=='stratified' else 1
    power=int(np.log2(count/factor))
    if factor*2**power!=count: raise ValueError('Historical prefix lacks balanced Sobol power')
    all_e=[];all_w={k:[] for k in COMPONENTS};all_q=[];all_b=[];all_iw=[];elapsed=[];files={};audit=[]
    for batch,state in enumerate(states):
        q,branches,iw=nodes(folder,scheme,power,state['contract']['seed'])
        energies=[];weights={k:[] for k in COMPONENTS};times=[]
        for i in range(count):
            p=folder/'pairs'/scheme/f'batch{batch:02d}'/f'pair{i:06d}.npz'
            digest=sha256(p)
            if digest!=state['files'][p.name]: raise ValueError(f'Committed raw file hash changed: {p}')
            files[str(p)]=digest
            with np.load(p) as z:
                for key,expected in [('q_fractional',q[i]),('branches',branches[i]),('inverse_marginals',iw[i])]:
                    if not np.array_equal(z[key],expected): raise ValueError(f'Node/branch/pdf mismatch: {p} {key}')
                e=z['energies_mev'];energies.append(e)
                for key in COMPONENTS:weights[key].append(z[key])
                reconstructed=z['direct_weights']+z['sequential_weights']+z['interference_weights']
                scale=max(float(np.max(z['direct_weights']+z['sequential_weights'])),1e-300)
                if np.max(abs(reconstructed-z['weights']))>2e-12*scale:raise ValueError('Lost signed coherent component identity')
                if np.any(z['weights']<0):raise ValueError('Negative exact coherent intensity')
                times.append(float(z['elapsed_seconds']))
        for side in range(2):
            density=proposal.density(q[:,side],branches[:,side])
            expected=(6*strat.branch_probability[branches[:,side]]/density if scheme=='stratified' else 1/density)
            if not np.allclose(iw[:,side],expected,rtol=4e-15,atol=0): raise ValueError('Independent density reconstruction failed')
        hist=np.bincount(branches[:,0]*6+branches[:,1],minlength=36)
        if scheme=='stratified' and not np.all(hist==2**power):raise ValueError('Unequal branch-stratum point counts')
        nested=nodes(folder,scheme,max(power-1,0),state['contract']['seed'])
        prefix=min(len(nested[0]),len(q))
        if not all(np.array_equal(a[:prefix],b[:prefix]) for a,b in zip((q,branches,iw),nested)):
            raise ValueError('Sobol nested prefix was not preserved')
        audit.append(dict(batch=batch,seed=state['contract']['seed'],sobol_power_per_net=power,
            independent_nets=36 if scheme=='stratified' else 1,count=count,stratum_counts=hist,
            node_branch_inverse_density_exactly_reproduced=True,committed_raw_hashes_verified=True,
            density_reconstruction_passed=True,nested_prefix_passed=True,
            cumulative_worker_seconds=state['cumulative_seconds'],measured_pair_seconds=sum(times)))
        all_e.append(energies)
        for key in COMPONENTS:all_w[key].append(weights[key])
        files[str(folder/'pairs'/scheme/f'batch{batch:02d}'/'state.json')]=sha256(folder/'pairs'/scheme/f'batch{batch:02d}'/'state.json')
        all_q.append(q);all_b.append(branches);all_iw.append(iw);elapsed.append(times)
    seeds=[s['contract']['seed'] for s in states]
    expanded=[s+104729*k for s in seeds for k in (range(36) if scheme=='stratified' else [0])]
    if len(expanded)!=len(set(expanded)):raise ValueError('Scramble seed reuse within a scheme')
    return dict(e=np.asarray(all_e),w={k:np.asarray(v) for k,v in all_w.items()},q=np.asarray(all_q),
        branches=np.asarray(all_b),inverse=np.asarray(all_iw),elapsed=np.asarray(elapsed),audit=audit,
        files=files,seeds=seeds,states=states,source_folder=folder)


def _influence(values):
    a=np.asarray(values);flat=a.reshape((-1,)+a.shape[2:]);total=flat.sum(axis=0)
    ordered=np.sort(flat,axis=0)[::-1]
    shares={str(k):np.divide(ordered[:k].sum(axis=0),total,out=np.zeros_like(total),where=total!=0) for k in (1,5,10,20)}
    squares=(flat*flat).sum(axis=0)
    return dict(top_evaluation_fraction=shares,
        weight_concentration_count=np.divide(total*total,squares,out=np.zeros_like(total),where=squares>0),
        interpretation='Concentration diagnostic only; individual Sobol nodes are dependent and this count is not an effective sample size for confidence intervals.')


def _top_records(data,score,count=20):
    order=np.argsort(np.asarray(score).ravel())[::-1][:count];n=data['q'].shape[1];rows=[]
    for at in order:
        batch,i=divmod(int(at),n)
        rows.append(dict(batch=batch,index=i,score=float(np.asarray(score)[batch,i]),
            q_fractional=data['q'][batch,i],branches=data['branches'][batch,i],inverse_marginals=data['inverse'][batch,i],
            energies_mev=data['e'][batch,i],weights=data['w']['weights'][batch,i],
            direct_weights=data['w']['direct_weights'][batch,i],sequential_weights=data['w']['sequential_weights'][batch,i],
            interference_weights=data['w']['interference_weights'][batch,i],
            file=f'pairs/stratified/batch{batch:02d}/pair{i:06d}.npz',
            source_npz=str((data['source_folder']/f'pairs/stratified/batch{batch:02d}/pair{i:06d}.npz').resolve()),
            source_sha256=sha256(data['source_folder']/f'pairs/stratified/batch{batch:02d}/pair{i:06d}.npz')))
    return rows


def _scheme_summary(folder,scheme,data,output):
    count=data['e'].shape[1];raw=raw_vector(data['e'],data['w']['weights']);batch=raw.mean(axis=1)
    stat=estimate(batch);old=json.loads((folder/f'{scheme}_analysis.json').read_text());latest=old['prefixes'][-1]
    for i,key in enumerate(WINDOW_NAMES):
        if not np.allclose(stat['mean'][i],latest['raw'][key]['mean'],rtol=1e-13,atol=1e-25):raise ValueError('Historical raw mean disagrees with independent reduction')
        if not np.allclose(stat['halfwidth95'][i],latest['raw'][key]['halfwidth95'],rtol=1e-12,atol=1e-25):raise ValueError('Historical raw uncertainty disagrees')
    gates=evaluate({key:{k:stat[k][i] for k in ('mean','halfwidth95')} for i,key in enumerate(WINDOW_NAMES)},old['comparisons'])
    oldg=json.loads((folder/f'{scheme}_acceptance.json').read_text())['all_gates']
    for k in ['integrated_loss_all_detectors','integrated_gain_all_detectors','all_energy_windows_all_detectors','shape','features_identified_and_stable','statistical_gates_passed','declared_target_accepted']:
        if gates[k]!=oldg[k]:raise ValueError('Authoritative acceptance gate mismatch')
    sign=estimate(data['w']['weights'].mean(axis=1))
    components={key:estimate(raw_vector(data['e'],data['w'][key]).mean(axis=1)) for key in COMPONENTS}
    per_sign_variance=data['w']['weights'].mean(axis=1).var(axis=0,ddof=1)/8
    result=dict(pairs=count*8,pairs_per_scramble=count,independent_scrambles=8,audit=data['audit'],
        proposal_mass_sha256=sha256(folder/('adapted_mass.npy' if scheme=='adapted' else 'proposal_mass.npy')),
        full_support_lower_joint_marginal_density=.2/6,raw=stat,components=components,sign_statistics=sign,
        per_sign_mean_variance=per_sign_variance,all_gates=gates,concentration=_influence(raw),
        historical_analysis_raw_mean_and_t_intervals_reproduced=True,
        costs=dict(summed_checkpoint_worker_seconds=sum(s['cumulative_seconds'] for s in data['states']),
            summed_pair_seconds=float(data['elapsed'].sum()),mean_pair_seconds=float(data['elapsed'].mean()),
            preparation_seconds_included=False,preparation_limitation='Historical state measures committed pair work. Cache preparation and pilot costs are reported separately; elapsed time is not inferred by dividing by worker count.'),
        mean_variance_times_worker_seconds=stat['standard_error']**2*sum(s['cumulative_seconds'] for s in data['states']),
        nonzero_evaluation_counts=np.sum(raw>0,axis=(0,1)),
        nonzero_scramble_counts=np.sum(batch>0,axis=0),
        confidence_scope='Student-t nominal marginal95 across independent randomized nets; no simultaneous coverage or deterministic error included.')
    np.savez_compressed(output/f'{scheme}_raw.npz',energy=data['e'],weights=data['w']['weights'],
        q_fractional=data['q'],branches=data['branches'],inverse_marginals=data['inverse'],
        window_evaluations=raw,window_batches=batch,**{k:v for k,v in data['w'].items() if k!='weights'})
    if scheme=='stratified':
        strata=raw.reshape(8,count//36,36,len(WINDOWS),4).mean(axis=1)/36
        var=strata.var(axis=0,ddof=1)/8
        covariance=np.einsum('bswd,btwd->stwd',strata-strata.mean(axis=0),strata-strata.mean(axis=0))/56
        totalvar=batch.var(axis=0,ddof=1)/8
        result['stratification']=dict(stratum_mean=strata.mean(axis=0),stratum_variance_contribution=var,
            sum_diagonal_stratum_variance=var.sum(axis=0),observed_total_mean_variance=totalvar,
            empirical_cross_stratum_covariance_remainder=totalvar-var.sum(axis=0),
            explanation='Per-stratum means include1/36 branch-pair average. Marginal inverse weights already carry6*pi_b/p_joint. Cross-stratum covariance is retained in the total batch variance; its noisy eight-replicate decomposition is descriptive, not an allocation oracle.')
        np.savez_compressed(output/'stratum_covariance.npz',covariance_of_means=covariance,stratum_batch_integrals=strata)
        result['per_stratum_sign_mean']=data['w']['weights'].reshape(8,count//36,36,4,4).mean(axis=(0,1))/36
        result['per_stratum_sign_variance_of_mean']=data['w']['weights'].reshape(8,count//36,36,4,4).mean(axis=1).var(axis=0,ddof=1)/(36**2*8)
    return result


def _known_integrals(folder):
    result={}
    for scheme in ('born','adapted','stratified'):
        values=[];power=8 if scheme=='stratified' else 13;seeds=list(range(50271101,50271109))
        for seed in seeds:
            q,b,iw=nodes(folder,scheme,power,seed)
            values.append(np.mean((1+q[:,0,0]*q[:,1,2])*(b[:,0]+1)*(b[:,1]+1)*iw.prod(axis=1)))
        s=estimate(np.asarray(values)); half=float(s['halfwidth95']);error=float(s['mean'])-441.
        result[scheme]=dict(target=441.,scramble_values=values,mean=float(s['mean']),halfwidth95=half,
            error=error,passed=abs(error)<max(5*half,.2),seeds=seeds,sobol_power=power,
            scope='Polynomial/branch known integral tests the proposal and inverse density only; it is not a known Born or complete-pair physical integral.')
        if not result[scheme]['passed']:raise ValueError('Known-integral density check failed')
    return result


def _make_panel(data):
    raw=raw_vector(data['e'],data['w']['weights']);n=data['e'].shape[1]
    loss=raw[:,:,WINDOW_NAMES.index('all_loss')]
    d=data['w']['direct_weights'];s=data['w']['sequential_weights'];t=data['w']['weights']
    # Weight by observed component size when ranking cancellation; avoid arbitrary
    # enormous ratios for numerically empty targets.
    component=(d+s).sum(axis=(2,3));complete=t.sum(axis=(2,3))
    fraction=np.divide(complete,component,out=np.ones_like(complete),where=component>0)
    cancel_score=component*np.maximum(-np.log10(np.maximum(fraction,1e-300)),0)
    centered=raw-raw.mean(axis=1)[:,None]
    scale=np.maximum(raw.mean(axis=(0,1)),np.max(raw.mean(axis=(0,1)))*1e-8)
    variance_score=np.sum((centered/scale)**2,axis=(2,3))
    categories={'primary_loss':_top_records(data,loss[:,:,0]),
        'failed_angular_window_loss':_top_records(data,loss[:,:,2]),
        'component_weighted_cancellation':_top_records(data,cancel_score),
        'all_window_standardized_squared_influence':_top_records(data,variance_score)}
    rng=np.random.default_rng(50261102);representative=[]
    for batch in range(8):
        for index in rng.choice(n,size=2,replace=False):
            index=int(index)
            representative.append(dict(batch=batch,index=index,q_fractional=data['q'][batch,index],
                branches=data['branches'][batch,index],inverse_marginals=data['inverse'][batch,index],
                file=f'pairs/stratified/batch{batch:02d}/pair{index:06d}.npz',
                source_npz=str((data['source_folder']/f'pairs/stratified/batch{batch:02d}/pair{index:06d}.npz').resolve()),
                source_sha256=sha256(data['source_folder']/f'pairs/stratified/batch{batch:02d}/pair{index:06d}.npz'),
                selection='Fixed RNG50261102 selected two indices per existing independent scramble without consulting amplitudes/intensities/energies. This is an index-selected representative subset, not a fresh external sample.'))
    return dict(source='Complete static49x49/385-step stratified production',rankings=categories,
        independent_of_observed_values_representative=representative,
        scope='Extreme selections diagnose numerical mechanisms; neither these nor this small representative subset certify a global BZ discretization bound. Use new balanced independent nets for a final numerical-error estimate.',
        total_over_component_fraction=fraction,minimum_component_relative_fraction=float(fraction.min()))


def _freeze_panel8(candidates,output):
    rows=[];used=set()
    for category,ranked in sorted(candidates['rankings'].items()):
        row=next(r for r in ranked if (r['batch'],r['index']) not in used)
        used.add((row['batch'],row['index']))
        rows.append(dict(row,categories=[category],selection='Observed-value mechanism extreme; not an integration certificate.'))
    representative=candidates['independent_of_observed_values_representative']
    for batch in range(4):
        row=next(r for r in representative if r['batch']==batch and (r['batch'],r['index']) not in used)
        used.add((row['batch'],row['index']));rows.append(dict(row,categories=['index_selected_representative']))
    for i,row in enumerate(rows):row['panel_id']=f'diagnostic{i:02d}'
    separate=[]
    for batch in range(8):
        row=next(r for r in representative if r['batch']==batch and (r['batch'],r['index']) not in used)
        used.add((row['batch'],row['index']))
        separate.append(dict(row,panel_id=f'representative{batch:02d}',categories=['separate_index_selected_representative']))
    write_json(output/'panel8.json',dict(initial_panel8=rows,separate_representative8=separate,selection_seed=50261102,
        limitation='Representative indices were selected without observed values from existing nets; these are not fresh external BZ scrambles and do not certify global electron error.'))


def _historical_cv(historical,coarse_data,output):
    """New postprocessing of old eight joint electron/depth refinement pairs."""
    folder=historical/'bulk_numerical_controls';energies=[];coarse=[];fine=[];hashes={};timing=[]
    for batch in range(8):
        arrays=[]
        for case in ('base','fast125'):
            path=folder/case/f'control_pair{batch}.npz';hashes[str(path)]=sha256(path)
            with np.load(path) as z:
                arrays.append({k:z[k].copy() for k in ['q_fractional','branches','inverse_marginals','energies_mev','weights','elapsed_seconds']})
        c,f=arrays
        for k in ('q_fractional','branches','inverse_marginals','energies_mev'):
            if not np.array_equal(c[k],f[k]):raise ValueError('Historical refinement control is not mode-pair matched')
        for k,key in [('q_fractional','q'),('branches','branches'),('inverse_marginals','inverse')]:
            if not np.array_equal(c[k],coarse_data[key][batch,0]):raise ValueError('Historical refinement is not the shared coarse-net prefix')
        if not np.allclose(c['weights'],coarse_data['w']['weights'][batch,0],rtol=3e-13,atol=1e-30):
            raise ValueError('Historical coarse control and512-node baseline evaluate different observables')
        energies.append(c['energies_mev']);coarse.append(c['weights']);fine.append(f['weights'])
        timing.append(dict(batch=batch,coarse=float(c['elapsed_seconds']),fine=float(f['elapsed_seconds'])))
    energies=np.asarray(energies);coarse=np.asarray(coarse);fine=np.asarray(fine)
    coarse_raw=raw_vector(energies,coarse);fine_raw=raw_vector(energies,fine)
    baseline_raw=raw_vector(coarse_data['e'],coarse_data['w']['weights']).mean(axis=1)
    coarse_spectra=np.asarray([line_spectra(e,w,AXIS) for e,w in zip(energies,coarse)])
    fine_spectra=np.asarray([line_spectra(e,w,AXIS) for e,w in zip(energies,fine)])
    path=historical/'bulk_static_v2/born_n64_analysis.npz';hashes[str(path)]=sha256(path)
    with np.load(path) as z:baseline_spectra=z['batch_spectra'].copy()
    cv=control_variate_acceptance(fine_raw,coarse_raw,baseline_raw,fine_spectra,coarse_spectra,
        baseline_spectra,'paired',complete_coherent_pair=True)
    direct_fine=estimate(fine_raw);loss=WINDOW_NAMES.index('all_loss')
    coarse_seconds=sum(s['cumulative_seconds'] for s in coarse_data['states'])
    fine_seconds=sum(t['fine'] for t in timing);coarse_pair_seconds=sum(t['coarse'] for t in timing)
    cv_seconds=coarse_seconds+fine_seconds
    stats=cv['statistics'];fine_variance=direct_fine['standard_error']**2
    cv_var=np.asarray(stats['standard_error'])**2
    ratios=np.divide(cv_var*cv_seconds,fine_variance*fine_seconds,
        out=np.full_like(fine_variance,np.nan),where=fine_variance>0)
    worst=np.unravel_index(np.nanargmax(np.divide(stats['halfwidth95'],abs(stats['mean']),out=np.full_like(stats['mean'],np.inf),where=abs(stats['mean'])>1e-18)),stats['mean'].shape)
    report=dict(status='DIAGNOSTIC_ONLY_NO_ELECTRON_CONVERGENCE',physical_reference='Same4-repeat300K60keV static Taylor changed-occupation complete coherent pair observable, fixed physical detectors and2meVFWHM.',
        fine_setting=dict(electron_pixels=125,depth_steps=769),coarse_setting=dict(electron_pixels=49,depth_steps=385),
        scope='Joint electron-and-depth refinement CV, not electron-only. Original125 setting itself has failed97→125 convergence. Eight correction nodes alone do not provide a robust full-vector variance ranking.',
        baseline_pairs=512,correction_pairs=8,independent_scrambles=8,baseline_relationship='paired',
        baseline_uncertainty_included=True,baseline_correction_covariance_included=True,
        provenance='Matched correction node is first node of each corresponding64-point coarse Sobol scramble. Base control intensities match the committed coarse baseline node. No additional scientific propagation.',
        cv=cv,direct_eight_fine_estimate=direct_fine,cost_adjusted_variance_ratio_cv_over_eight_fine=ratios,
        windows_improved_at_measured_committed_work=int(np.sum(ratios<1)),windows_compared=ratios.size,
        costs=dict(coarse512_committed_seconds=coarse_seconds,fine8_measured_pair_seconds=fine_seconds,
            duplicate_coarse8_control_measured_pair_seconds=coarse_pair_seconds,
            total_unique_retained_work_seconds=cv_seconds,
            including_duplicate_historical_coarse_control_seconds=cv_seconds+coarse_pair_seconds,
            preparation_pilot_and_postprocessing_excluded_from_worker_comparison=True,
            coarse_reuse='The correction coarse evaluations are already present in the512 baseline. Reusing them avoids a second physical evaluation; original duplicate control cost is recorded separately.'),
        primary_loss=dict(cv_mean=stats['mean'][loss,0],cv_halfwidth95=stats['halfwidth95'][loss,0],
            direct_fine_mean=direct_fine['mean'][loss,0],direct_fine_halfwidth95=direct_fine['halfwidth95'][loss,0],
            correction_variance_over_fine=stats['correction_variance_over_fine'][loss,0],
            total_variance_over_fine=stats['total_variance_over_fine'][loss,0],
            measured_cost_adjusted_variance_ratio=ratios[loss,0]),
        worst_cv_relative_interval=dict(window=WINDOW_NAMES[worst[0]],detector=DETECTORS[worst[1]],mean=stats['mean'][worst],halfwidth95=stats['halfwidth95'][worst]),
        decision='Do not select this historical CV for production: the electron setting is unconverged, no balanced correction half-prefix establishes shape/features, and observed gains vary across weak windows. Measure matched complete-pair residuals on new independent balanced scrambles after numerical diagnosis.',
        input_hashes=hashes,source_sha256=sha256(__file__))
    write_json(output/'historical_joint_cv.json',report)
    np.savez_compressed(output/'historical_joint_cv.npz',energies_mev=energies,fine_weights=fine,coarse_weights=coarse,
        fine_raw=fine_raw,coarse_raw=coarse_raw,baseline_raw=baseline_raw,
        fine_batch_spectra=fine_spectra,coarse_batch_spectra=coarse_spectra,baseline_batch_spectra=baseline_spectra,
        cv_mean_spectrum=cv['mean_spectrum'])
    return report



def _load_electron_case(case):
    state=json.loads((case/'state.json').read_text());contract=state['contract']
    panelpath=Path(contract['panel'])
    if sha256(panelpath)!=contract['panel_sha256']:raise ValueError('Frozen diagnostic panel changed')
    panel=json.loads(panelpath.read_text());panel=panel.get('pairs',panel.get('panel',panel))
    rows=[];hashes={str(case/'state.json'):sha256(case/'state.json'),str(panelpath):sha256(panelpath)}
    for name,digest in sorted(state['files'].items()):
        path=case/name
        if sha256(path)!=digest:raise ValueError('Committed electron-panel output changed')
        hashes[str(path)]=digest
        with np.load(path) as z:row={k:z[k].copy() for k in z}
        index=int(path.stem[4:]);p=panel[index]
        if str(row['source_sha256'])!=p['source_sha256']:raise ValueError('Electron record and frozen-source hash differ')
        if sha256(Path(str(row['source_path'])))!=str(row['source_sha256']):raise ValueError('Source material pair changed')
        row['panel_index']=index;row['categories']=p.get('categories',[]);rows.append(row)
    settings=json.loads((case/'numerical_settings.json').read_text());hashes[str(case/'numerical_settings.json')]=sha256(case/'numerical_settings.json')
    return dict(contract=contract,rows=rows,hashes=hashes,settings=settings,panel_size=len(panel),case=case)


def _masked_l2(values,masks):
    return np.sqrt(np.einsum('dp,p->d',np.asarray(masks,float),abs(np.asarray(values).ravel())**2))


def _amplitude_panel_pair(coarse,fine,coarse_setting,fine_setting):
    from scipy import constants
    from yoshioka.free_propagation import TransverseGrid
    from yoshioka.pair_galerkin import frequency_indices
    from yoshioka.two_phonon import masks_for_shift,SIGNS
    from yoshioka.bulk_thermal_source import occupations
    for key in ('q_fractional','branches','inverse_marginals','energies_mev','final_shifts','omega_rad_s'):
        if not np.array_equal(coarse[key],fine[key]):raise ValueError(f'Physical mode-pair mismatch: {key}')
    cell=np.asarray(coarse_setting['cell_lengths_angstrom'])*1e-10
    if not np.array_equal(coarse_setting['cell_lengths_angstrom'],fine_setting['cell_lengths_angstrom']):raise ValueError('Physical transverse cell changed')
    n=coarse['direct_amplitudes'].shape[-1];m=fine['direct_amplitudes'].shape[-1];common=min(n,m)
    grids=[TransverseGrid(*cell,size,size) for size in (n,m,common)]
    ia=frequency_indices(common,n);ib=frequency_indices(common,m)
    nbar=occupations(fine['omega_rad_s'],300);inverse=.5*np.prod(fine['inverse_marginals'])
    components={};reconstructed_coarse=np.zeros((4,4));reconstructed_fine=np.zeros((4,4))
    common_detector_identity=True;normalization_identity=True;cancel_bounds=[]
    for name in ('direct','sequential','total'):
        ka='total_amplitudes' if name=='total' else name+'_amplitudes'
        aa=coarse[ka] if ka in coarse else coarse['direct_amplitudes']+coarse['sequential_amplitudes']
        bb=fine[ka] if ka in fine else fine['direct_amplitudes']+fine['sequential_amplitudes']
        norms={k:[] for k in ['common_absolute_l2_difference','common_relative_l2_difference','common_coarse_l2','common_fine_l2',
            'added_coarse_l2','added_fine_l2','common_thermal_weighted_square_difference',
            'added_fine_thermal_weight','added_coarse_thermal_weight']}
        for sign,(sa,sb) in enumerate(SIGNS):
            shift=fine['final_shifts'][sign]
            ma=masks_for_shift(grids[0],shift);mb=masks_for_shift(grids[1],shift);mc=masks_for_shift(grids[2],shift)
            mac=ma.reshape(4,n,n)[:,ia[:,None],ia].reshape(4,-1)
            mbc=mb.reshape(4,m,m)[:,ib[:,None],ib].reshape(4,-1)
            common_detector_identity &= np.array_equal(mac,mc) and np.array_equal(mbc,mc)
            ac=aa[sign][ia[:,None],ia];bc=bb[sign][ib[:,None],ib]
            an=_masked_l2(ac,mc);bn=_masked_l2(bc,mc);dn=_masked_l2(bc-ac,mc)
            # Explicit complementary masks avoid subtracting near-equal norms.
            keepa=np.zeros((n,n),bool);keepa[ia[:,None],ia]=True
            keepb=np.zeros((m,m),bool);keepb[ib[:,None],ib]=True
            added_a=_masked_l2(aa[sign],ma&~keepa.ravel()[None])
            added_b=_masked_l2(bb[sign],mb&~keepb.ravel()[None])
            factor=inverse*(nbar[0]+(sa==1))*(nbar[1]+(sb==1))
            norms['common_absolute_l2_difference'].append(dn)
            norms['common_relative_l2_difference'].append(np.divide(dn,bn,out=np.full(4,np.nan),where=bn>0))
            norms['common_coarse_l2'].append(an);norms['common_fine_l2'].append(bn)
            norms['added_coarse_l2'].append(added_a);norms['added_fine_l2'].append(added_b)
            norms['common_thermal_weighted_square_difference'].append(factor*dn*dn)
            norms['added_fine_thermal_weight'].append(factor*added_b**2)
            norms['added_coarse_thermal_weight'].append(factor*added_a**2)
            if name=='total':
                reconstructed_coarse[sign]=factor*(an*an+added_a**2)
                reconstructed_fine[sign]=factor*(bn*bn+added_b**2)
        components[name]={k:np.asarray(v) for k,v in norms.items()}
    if not common_detector_identity:raise ValueError('Common physical reciprocal detector intersection changed')
    for recorded,reconstructed in [(coarse['weights'],reconstructed_coarse),(fine['weights'],reconstructed_fine)]:
        if not np.allclose(recorded,reconstructed,rtol=2e-12,atol=1e-28):
            raise ValueError('Saved amplitude normalization does not reconstruct raw detector intensity')
    d=fine['direct_weights'];s=fine['sequential_weights'];t=fine['weights']
    condition=np.sqrt(np.divide(d+s,t,out=np.full_like(t,np.nan),where=t>0))
    bound=np.divide(components['direct']['common_absolute_l2_difference']+components['sequential']['common_absolute_l2_difference'],
        components['total']['common_fine_l2'],out=np.full((4,4),np.nan),where=components['total']['common_fine_l2']>0)
    return dict(panel_index=coarse['panel_index'],categories=coarse['categories'],
        source_path=str(coarse['source_path']),source_sha256=str(coarse['source_sha256']),
        components=components,cancellation_condition_sqrt_component_over_total=condition,
        component_difference_norm_bound_over_total=bound,
        all_common_detector_masks_identical=True,raw_intensities_reconstructed_from_complex_components=True,
        arbitrary_phase_or_scale_fit=False,common_integer_reciprocal_halfwidth=(common-1)//2,
        raw_component_differences={key:fine[key]-coarse[key] for key in COMPONENTS},
        final_shifts_m_inv=fine['final_shifts'],energies_mev=fine['energies_mev'])


def analyze_electron_panels(base,output,case_names):
    """All-vector selected-panel diagnostic, never a bulk acceptance claim."""
    output.mkdir(parents=True,exist_ok=True);started=time.monotonic();cases={};pending={};hashes={}
    for name in case_names:
        folder=base/name
        if not (folder/'state.json').exists():pending[name]='No committed case';continue
        d=_load_electron_case(folder);hashes.update(d['hashes'])
        if len(d['rows'])!=d['panel_size']:
            pending[name]=dict(committed_pairs=len(d['rows']),panel_pairs=d['panel_size']);continue
        cases[name]=d
    comparisons=[];names=list(cases)
    pairings=list(zip(names[:-1],names[1:]))
    if len(names)>2:pairings.append((names[0],names[-1]))
    for a,b in pairings:
        x=cases[a];y=cases[b]
        for key in ('panel_sha256','fixed_physics','born','kernels','lobato_file_sha256'):
            if x['contract'][key]!=y['contract'][key]:raise ValueError(f'Case contract differs in physics: {key}')
        rows=[_amplitude_panel_pair(c,f,x['settings'],y['settings']) for c,f in zip(x['rows'],y['rows'])]
        energy=np.asarray([r['energies_mev'] for r in y['rows']]);cx=np.asarray([r['weights'] for r in x['rows']]);fy=np.asarray([r['weights'] for r in y['rows']])
        cr=raw_vector(energy,cx);fr=raw_vector(energy,fy);delta=fr-cr;n=len(rows)
        cs=line_spectra(energy,cx/n,AXIS);fs=line_spectra(energy,fy/n,AXIS);shape=spectral_comparison(cs,fs)
        descriptive=estimate(fr);difference=estimate(delta)
        rel=np.divide(abs(difference['mean']),abs(descriptive['mean']),out=np.full_like(descriptive['mean'],np.inf),where=abs(descriptive['mean'])>1e-18)
        near=abs(descriptive['mean'])<=1e-18
        numerical_windows=(rel<.01)| (near&(abs(difference['mean'])<=1e-18))
        raw={key:{k:descriptive[k][i] for k in ['mean','halfwidth95']} for i,key in enumerate(WINDOW_NAMES)}
        authoritative=evaluate(raw,shape,'NOT_ESTABLISHED_DIAGNOSTIC_PANEL')
        row=dict(coarse=a,fine=b,pairs=n,rows=rows,window_names=WINDOW_NAMES,detectors=DETECTORS,processes=PROCESSES,
            descriptive_fine_statistics=descriptive,descriptive_paired_differences=difference,
            relative_mean_window_change=rel,panel_only_onepercent_window_sensitivity_pass=numerical_windows,
            panel_only_all_windows_sensitivity_pass=bool(numerical_windows.all()),
            spectral_comparison=shape,all_observable_gate_evaluator=authoritative,
            scope='Four observed-value extremes plus four index-selected representatives. Whole-panel means and t-scaled scatter are descriptive; nodes share some source scramble batches and are selected nonrandomly. These are not BZ integrals, BZ confidence intervals, production prefixes, or a global numerical-convergence certificate.',
            numerical_scope='Same physical reciprocal spacing, changed retained cutoff and intermediate support; each tested Galerkin product is projected at every H action. Products meet2N−1 transfer support. No detector interpolation or fitted phases/scales.',
            whole_panel_statistics_are_bulk_acceptance=False,
            costs=dict(coarse_measured_pair_seconds=sum(float(r['measured_wall_seconds']) for r in x['rows']),
                fine_measured_pair_seconds=sum(float(r['measured_wall_seconds']) for r in y['rows']),
                coarse_preparation_seconds=x['settings']['product_preparation_seconds'],fine_preparation_seconds=y['settings']['product_preparation_seconds'],
                coarse_peak_rss_mib=max(float(r['peak_process_rss_mib']) for r in x['rows']),
                fine_peak_rss_mib=max(float(r['peak_process_rss_mib']) for r in y['rows'])))
        comparisons.append(row)
        np.savez_compressed(output/f'{a}__{b}.npz',energy_mev=AXIS,coarse_mean_spectrum=cs,fine_mean_spectrum=fs,
            fine_window_vectors=fr,coarse_window_vectors=cr,paired_window_differences=delta,
            coarse_direct=np.asarray([r['direct_amplitudes'] for r in x['rows']]),
            fine_direct=np.asarray([r['direct_amplitudes'] for r in y['rows']]))
    result=dict(status='DIAGNOSTIC_ONLY_NOT_ACCEPTED',numerical_status='NOT_ESTABLISHED_DIAGNOSTIC_PANEL',
        cases={n:dict(settings=d['settings'],retained=d['contract']['retained'],product=d['contract']['product'],steps=d['contract']['steps'],
            substeps=d['contract']['local_substeps'],pairs=len(d['rows']),all_committed_hashes_verified=True) for n,d in cases.items()},
        pending_cases=pending,comparisons=comparisons,input_hashes=hashes,source_sha256=sha256(__file__),
        postprocessing_seconds=time.monotonic()-started)
    write_json(output/'analysis.json',result)
    lines=['# Frozen electron-panel complete-vector diagnostics','',
        'These comparisons use four selected extremes plus four index-selected representative pairs. They diagnose numerical effects; their whole-panel means and t-scaled paired scatter do not estimate a bulk BZ integral or certify global error. The unchanged authoritative evaluator always receives NOT_ESTABLISHED_DIAGNOSTIC_PANEL.','',
        'All 56 raw window/detector combinations, gain and loss, 2meV fixed-axis spectrum shapes, and every feature criterion are retained. Complex direct, sequential and total amplitudes are compared by each of four sign sectors and four physical detectors on common integer reciprocal labels. Added-beam contributions are recorded separately. Saved raw intensities are independently reconstructed from those complex amplitudes and the analytic Bose/importance factors. No phase or scale fit is used.','',
        '| Coarse | Fine | Primary loss change | Maximum declared-window relative change | Primary shape L1(5–160) | Common-mask mapping |','|---|---|---:|---:|---:|---|']
    for c in comparisons:
        at=WINDOW_NAMES.index('all_loss');shape=c['spectral_comparison'][0]['windows']['5_160']['shape_L1']
        lines.append(f"| {c['coarse']} | {c['fine']} | {100*c['relative_mean_window_change'][at,0]:.6g}% | {100*np.max(c['relative_mean_window_change']):.6g}% | {shape:.6g} | exact |")
    lines+=['','Empty or nearly zero selected-panel windows retain the declared1e−18 absolute rule; they do not establish that unsampled bulk weight is zero. Descriptive paired differences retain per-node fine/coarse covariance. Deliberate extreme selection and shared historical scrambles prevent interpreting these t-scaled scatter values as independent production uncertainty.','',
        f'Pending cases at this snapshot: {json.dumps(pending)}.','',
        '[All complex and complete-vector evidence](analysis.json)','']
    (output/'report.md').write_text('\n'.join(lines));write_json(output/'manifest.json',{str(p.relative_to(output)):sha256(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'})
    print(json.dumps(dict(status=result['status'],complete_cases=list(cases),pending=pending,comparisons=len(comparisons))),flush=True)
    return result



def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--historical',type=Path,default=OLD);p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--electron-panel-analysis',action='store_true');p.add_argument('--case-names',nargs='+',default=['diag_n17_m35_z769','diag_n25_m49_z769','diag_n33_m65_z769','diag_n49_m99_z769']);a=p.parse_args()
    if a.electron_panel_analysis:
        analyze_electron_panels(a.output.parent/'electron_cases',a.output/'electron_panel_analysis',a.case_names);return
    a.output.mkdir(parents=True,exist_ok=True);started=time.monotonic();folder=a.historical/'bulk_static_v2'
    data={};summaries={};files={}
    for scheme in ('born','adapted','stratified'):
        data[scheme]=_read_scheme(folder,scheme);files.update(data[scheme]['files'])
        summaries[scheme]=_scheme_summary(folder,scheme,data[scheme],a.output)
    for name in ['proposal_mass.npy','adapted_mass.npy','config.json','criteria.json','inputs.json','adaptation.json']:
        files[str(folder/name)]=sha256(folder/name)
    for scheme in data:
        for suffix in ['_analysis.json','_acceptance.json']:
            files[str(folder/(scheme+suffix))]=sha256(folder/(scheme+suffix))
    raw_pair_count=sum(d['e'].shape[0]*d['e'].shape[1] for d in data.values())
    adapted=json.loads((folder/'adaptation.json').read_text());pilot=[];pilot_hashes={}
    for name,digest in adapted['pilot_files'].items():
        path=ROOT/name
        if sha256(path)!=digest:raise ValueError('Frozen adaptation pilot changed')
        pilot_hashes[str(path)]=digest
        with np.load(path) as z:pilot.append((z['q_fractional'].copy(),z['branches'].copy()))
    pilot_states=[json.loads(p.read_text()) for p in sorted((a.historical/'bulk_static/pairs/born').glob('batch*/state.json'))]
    reused=0
    for batch in range(8):
        for index in range(8):
            with np.load(a.historical/f'bulk_static/pairs/born/batch{batch:02d}/pair{index:06d}.npz') as z:
                reused+=int(np.array_equal(z['q_fractional'],data['born']['q'][batch,index]) and np.array_equal(z['branches'],data['born']['branches'][batch,index]))
    if set(data['adapted']['seeds'])&{s['contract']['seed'] for s in pilot_states}:raise ValueError('Adapted assessment seed reused for design')
    selected=_make_panel(data['stratified']);write_json(a.output/'diagnostic_panel_candidates.json',selected)
    _freeze_panel8(selected,a.output)
    historical_cv=_historical_cv(a.historical,data['born'],a.output);files.update(historical_cv['input_hashes'])
    compare=[]
    for i,name in enumerate(WINDOW_NAMES):
        for j,det in enumerate(DETECTORS):
            row=dict(window=name,detector=det)
            for scheme in summaries:
                r=summaries[scheme];row[scheme]=dict(mean=r['raw']['mean'][i,j],halfwidth95=r['raw']['halfwidth95'][i,j],
                    relative_halfwidth95=r['raw']['relative_halfwidth95'][i,j],
                    variance_times_worker_seconds=r['mean_variance_times_worker_seconds'][i,j])
            compare.append(row)
    sampler=dict(counts_verified=True,raw_hashes_verified=True,node_pdf_and_sobol_balance_verified=True,
        strata=36,stratified_points_per_stratum_per_scramble=4,
        seeds={s:d['seeds'] for s,d in data.items()},adapted_pilot_seeds=[s['contract']['seed'] for s in pilot_states],
        pilot_pairs=64,pilot_worker_seconds=sum(s['cumulative_seconds'] for s in pilot_states),
        adapted_pilot_score_windows_mev=adapted['windows_mev'],
        adaptation_objective_coverage_limitation='Pilot score begins at5meV; authoritative raw0–40meV windows include0–5meV. All detector objectives are scored, but near-zero transitions, spectral shapes and feature gates are not directly optimized. Full proposal support prevents estimator bias; full-vector assessment is still required.',
        pilot_born_production_reused_node_prefix_count=reused,
        adapted_production_independent_of_pilot=True,
        qualification='Born production reuses all64 complete-amplitude design-pilot q/branch nodes. Its integral remains an initial fixed-proposal estimate, but it is not a held-out comparator for the learned proposal. Stratified batch seeds share the base number with Born, while their36 branch-conditioned nets differ; cross-scheme uncertainty must not assume independence without a design argument.',
        proposal='Mixture .2 uniform+.4 radial+.4 frozen table; uniform term preserves allBZ/branch support. Stratum conditional inverse marginal=6*pi_b/p_joint, equal stratum averaging1/36; pair integrand contains its existing1/2 unordered-pair factor.',
        scope='No propagation or new bulk production. All evaluated physical arrays are historical49x49/385step data and fail current numerical convergence.')
    output=dict(historical_root=str(a.historical),window_names=WINDOW_NAMES,detectors=DETECTORS,processes=PROCESSES,
        sampler_audit=sampler,schemes=summaries,comparison_all_windows=compare,known_integral_validation=_known_integrals(folder),
        status='NOT_ACCEPTED_ELECTRON_AND_SAMPLING_LIMITATIONS',
        next_strategy='Do not extend49-pixel production. First measure matched complete-pair fine-minus-coarse and matched Born residuals on independent balanced diagnostic scrambles at an electron setting supported by refinement. Include a separately estimated complete-coherent coarse/Born baseline and its variance. Select by worst declared-window cost-adjusted uncertainty, plus unchanged shape/features, rather than primary loss alone.',
        cv_scope='Coarse-electron and Born control variates must both be complete coherent pair integrands with the same bath, thermal signs, reference, normalization and detectors. A Born proposal table is not that integral. A compatible Born pair integral is currently not a known baseline; it must be calculated with uncertainty.',
        historical_joint_numerical_cv=historical_cv,
        helper='paired_control_variate and control_variate_acceptance in the isolated audit module retain baseline error and covariance, and reject missing complete-observable scope or incomplete vector.',
        source_sha256=sha256(__file__),postprocessing_seconds=time.monotonic()-started,
        peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(a.output/'sampling_audit.json',output);write_json(a.output/'input_manifest.json',dict(raw_files=files,pilot_files=pilot_hashes,source_sha256=sha256(__file__)))
    write_json(a.output/'status.json',dict(status=output['status'],sampling_integrity='VERIFIED',
        electron_convergence='NOT_ESTABLISHED',bulk_acceptance='NOT_ACCEPTED',new_scientific_propagations=0,
        complete_vector_cv='EXECUTED_JOINT_NUMERICAL_DIAGNOSTIC_NOT_SELECTED_FOR_PRODUCTION',
        electron_panel_analysis=str(a.output/'electron_panel_analysis/analysis.json'),active_jobs=[]))
    write_json(a.output/'checkpoint.json',dict(status=output['status'],completed_pairs_inspected=raw_pair_count,historical_results_modified=False,new_propagations=0,active_jobs=[],
        next_discriminating_action=output['next_strategy'],resume_command=f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python benchmarks/14_two_phonon_eels/pair_sampling_audit.py --output {a.output}'))
    lines=['# Complete-pair sampling audit','',
        'This new read-only audit independently reconstructed all 2,176 completed production q/branch nodes, inverse densities, stored component sums and committed raw-file hashes. It also checked the 64-pair independent design-pilot files. No scientific propagation was started. Historical spectra remain electron-unconverged.','',
        '## Sampling construction','',
        'Born and adapted each contain eight independent scrambled 64-point Sobol nets. Stratified production contains eight independent batches, each with 36 separate four-point (2²) branch-pair nets. All nested prefixes and branch counts match the generating implementation exactly. The conditional marginal inverse is 6π_b/p(q,b); averaging the 36 strata supplies 1/36. Uniform proposal support is retained.','',
        'The adapted production seeds are independent of the 64 complete-amplitude pilot. Born production reuses all 64 pilot nodes as its initial prefixes; therefore that Born comparator is not held out from proposal design. This does not invalidate its fixed-proposal estimate. Different schemes are not silently treated as paired independent estimates.','',
        'The learned pilot score covers ten windows beginning at±5meV. Authoritative raw0–40meV loss/gain windows include0–5meV; their near-zero part was not directly optimized. Shapes and feature identification were not pilot objectives either. Importance estimation remains unbiased because proposal support is retained.','',
        '## Full acceptance vector','',
        '| Scheme | Pairs | Primary loss relative 95% halfwidth | Worst window/detector halfwidth | Largest shape L1 | Worker seconds |','|---|---:|---:|---:|---:|---:|']
    for scheme,r in summaries.items():
        rel=np.asarray(r['raw']['relative_halfwidth95']);at=np.unravel_index(np.nanargmax(rel),rel.shape)
        comparisons=json.loads((folder/f'{scheme}_analysis.json').read_text())['comparisons'];largest=max(w['shape_L1'] for c in comparisons for w in c['windows'].values())
        lines.append(f"| {scheme} | {r['pairs']} | {100*rel[WINDOW_NAMES.index('all_loss'),0]:.2f}% | {100*rel[at]:.2f}% ({WINDOW_NAMES[at[0]]}, {DETECTORS[at[1]]}) | {largest:.5f} | {r['costs']['summed_checkpoint_worker_seconds']:.2f} |")
    lines+=['',
        'Every scheme fails integrated loss, gain, some required energy windows, shape and feature stability. The authoritative calculation remains NOT_ACCEPTED. Nominal marginal Student-t intervals use the eight independent scramble integrals; individual Sobol evaluations never supply confidence degrees of freedom. Per-stratum covariance is retained in total-batch uncertainty.','',
        '## Diagnostics and next measurement','',
        'The machine-readable audit contains all 56 window/detector statistics, every sign sector, direct/sequential/signed-interference/coherent components, per-stratum variance decomposition and empirical cross-stratum covariance remainder. Top-weight fractions and concentration counts identify influential nodes; these descriptive counts are not independent-sample effective sizes. The frozen panel file includes 20 ranked nodes per diagnostic objective and 16 index-selected representative nodes chosen without consulting intensities.','',
        output['next_strategy'],'',
        'An additional new postprocessing calculation used the existing eight matched49-pixel/385-step to125-pixel/769-step records, with the512-pair49/385 coarse estimate. It is a joint electron-and-depth control variate, not an electron-only test. Shared scramble covariance is retained because each correction node is the first node of the corresponding64-point coarse net. Every raw acceptance window and both gain/loss sides are assessed. The single-node-per-batch correction has no half-count balanced prefix; unchanged shape/feature gates therefore remain unestablished. This diagnostic does not select a converged electron setting or justify production. [Measured joint-CV evidence](historical_joint_cv.json).','',
        'The supplied control-variate helper uses I_f=estimated I_c+mean(F_f−F_c), explicitly propagates coarse-baseline variance and paired covariance, and processes the entire fixed spectral axis. An unknown baseline relationship, missing full vector, or sequential-only scope is rejected. A previous balanced-prefix spectrum is mandatory for shape/features; otherwise those gates fail. Signed correction spectra are never clipped or rescaled.','',
        'Cost-adjusted variance comparisons in the audit use measured committed worker work only. The separate pilot cost is recorded. Shared preparation cost and reliable end-to-end elapsed time are not reconstructed from incomplete historical timing. No convergence forecast is used as acceptance evidence.','',
        f"Postprocessing wall time: {output['postprocessing_seconds']:.3f} s; process peak RSS: {output['peak_process_rss_mib']:.2f} MiB.",'',
        '[New Galerkin all-vector panel analysis](electron_panel_analysis/report.md) · [Full audit](sampling_audit.json) · [Panel candidates](diagnostic_panel_candidates.json) · [Input manifest](input_manifest.json) · [Checkpoint](checkpoint.json)','']
    (a.output/'report.md').write_text('\n'.join(lines))
    write_json(a.output/'output_manifest.json',{str(q.relative_to(a.output)):sha256(q) for q in a.output.rglob('*') if q.is_file() and q!=a.output/'output_manifest.json'})
    print(json.dumps({'status':output['status'],'raw_pairs':raw_pair_count,'pilot_pairs':64,'postprocessing_seconds':output['postprocessing_seconds'],'output':str(a.output)}),flush=True)

if __name__=='__main__':main()
