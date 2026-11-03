"""Versioned, bounded complete direct+sequential silicon mode-pair calculations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import time
import resource
import numpy as np
from scipy import constants
from scipy.stats import t as student_t
from .workflow_io import ROOT, ResourceLimits, json_write, sha256, fingerprint, environment
from .approximate_eels import PRIMITIVE, ORIENTED, geometry, DETECTORS
from .two_phonon_eels import ready_background
from .two_phonon_analysis import AXIS, WINDOWS, FEATURES, spectral_comparison, detected_features
from .born_quadrature import line_spectra
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.two_phonon import pair_nodes
from yoshioka.phonon_pair_importance import BornPairProposal
from yoshioka.stratified_pairs import BranchStratifiedPairs
from yoshioka.leading_two_phonon import propagate_bulk_leading_pair

DEFAULT=ROOT/'benchmark_results/14_two_phonon_eels/complete_leading_20261102_v1'
OLD=ROOT/'benchmark_results/two_phonon_eels_20261102_v1'
KERNELS=('yoshioka/leading_two_phonon.py','yoshioka/quadratic_fields.py','yoshioka/stratified_pairs.py',
         'yoshioka/two_phonon.py','yoshioka/thermal_dyson_words.py','yoshioka/phonon_pair_importance.py',
         'yoshioka/bulk_thermal_source.py','yoshioka/fourier_depth_fields.py',
         'yoshioka/electron_phonon.py','yoshioka/dynamical_source.py','yoshioka/lattice_dynamics.py',
         'benchmarks/complete_leading_eels.py')


def finite_json(value):
    if isinstance(value, np.ndarray):return finite_json(value.tolist())
    if isinstance(value, np.generic):return finite_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):return None
    if isinstance(value, dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):return [finite_json(v) for v in value]
    return value


def setup(output, reference_kind='static', temperature=300, repeats=4):
    output.mkdir(parents=True,exist_ok=True)
    config=dict(selected_repeats=repeats,padding_angstrom=12,selected_covariance_mesh=32,
        temperature_k=temperature,energy_ev=60000,primary_grid=49,primary_steps=385,
        runtime_limit_seconds=600,memory_limit_gib=2,batches=8,seed_start=1929001,
        reference_kind=reference_kind,gaussian_fwhm_mev=2,
        observable='Occupation changes (+,+),(-,-),(+,-),(-,+) of distinct continuum modes. Positive/negative energy windows are separately declared. Returns excluded.',
        thermal_average='Analytic Bose moments of probabilities; no thermal Fock cutoff.',
        units='SI internally; energies in meV; all reciprocal vectors angular rad/m; normalized primitive fractional BZ volume1; incident probability1.',
        elastic_reference=('Undisplaced atom Taylor reference; full active harmonic bath H fixed.' if reference_kind=='static' else
        'Full thermal-DW reference; Wick-ordered quadratic residual removes its thermal expectation; pair observable is Wick degree2.'))
    path=output/'config.json'
    if path.exists() and json.loads(path.read_text())!=config:raise ValueError('immutable configuration mismatch; use new output')
    if not path.exists():json_write(path,config)
    # The physical harmonic input is unchanged. Zero covariance is intentional
    # only in a bare Taylor reference, and is not a changed phonon Hamiltonian.
    source=ROOT/f'benchmark_results/approximate_eels_20261102_v1/covariance/T{temperature:g}_mesh32.json'
    physical=json.loads(source.read_text())
    covariance=np.zeros((3,3)) if reference_kind=='static' else np.array(physical['covariance_m2'])
    target=output/'covariance'/f'T{temperature:g}_mesh32.json'
    record=dict(covariance_m2=covariance.tolist(),physical_covariance_source=str(source),
                physical_covariance_sha256=sha256(source),reference_kind=reference_kind,
                force_constants_sha256=sha256(PRIMITIVE))
    if target.exists() and json.loads(target.read_text())!=record:raise ValueError('reference covariance changed')
    if not target.exists():json_write(target,record)
    # ready_background's historical interface hardcodes T300; temperature does
    # not enter V0 for the static reference. Retain an explicit compatible alias.
    if temperature!=300:
        alias=output/'covariance/T300_mesh32.json'
        if not alias.exists():json_write(alias,record)
    criteria=dict(integrated_loss_relative_95_halfwidth_max=.1,shape_L1_max=.1,
        identifiable_feature_shift_mev_max=1.,resolution_fwhm_mev=2.,shape_windows_mev=[[5,90],[5,160]],
        raw_windows={k:[float(x) if np.isfinite(x) else None for x in v] for k,v in WINDOWS.items()},detectors=DETECTORS,weak_absolute_floor=1e-18,
        declared_before_production=True,
        acceptance='Independent scramble uncertainty, nested count stability, all detectors/windows/features, and separately controlled electron/source discretization. No global truncated-spectrum renormalization.')
    if not (output/'criteria.json').exists():json_write(output/'criteria.json',criteria)
    p=output/'proposal_mass.npy'
    if not p.exists():shutil.copy2(OLD/'importance/mass.npy',p)
    json_write(output/'environment.json',environment())
    json_write(output/'inputs.json',dict(primitive_sha256=sha256(PRIMITIVE),oriented_sha256=sha256(ORIENTED),
        force_potential_sha256=sha256(ROOT/'Si.tersoff.mod'),proposal_mass_sha256=sha256(p),
        proposal_source=str(OLD/'importance/mass.npy'),proposal_use='Frozen historical one-phonon Born pilot table ONLY as full-support density. No Born surrogate is silently substituted for F.'))


def nodes(output, scheme, power, seed):
    if scheme=='radial':return pair_nodes(power,seed)
    p=BornPairProposal(np.load(output/('adapted_mass.npy' if scheme=='adapted' else 'proposal_mass.npy')),64)
    if scheme in ('born','adapted'):return p.nodes(power,seed)
    if scheme=='stratified':return BranchStratifiedPairs(p).nodes(power,seed)
    raise ValueError('unknown scheme')


def physical_setup(output, grid=None, steps=None):
    config=json.loads((output/'config.json').read_text())
    settings=dict(temperature_k=300,grid_unit=grid or config['primary_grid'],slices=steps or config['primary_steps'])
    limits=ResourceLimits(config['runtime_limit_seconds'],config['memory_limit_gib'])
    data=ready_background(config,settings,output,limits)
    return config,settings,data


def run_batch(task):
    output,scheme,power,batch,grid,steps,max_new,vectorized=task
    output=Path(output);config,settings,data=physical_setup(output,grid,steps)
    positions,reference,length,covariance,transverse,background,incident,_=data
    model=RealSpaceForceConstants.load(PRIMITIVE)
    limits=ResourceLimits(config['runtime_limit_seconds'],config['memory_limit_gib'])
    folder=output/'pairs'/scheme/f'batch{batch:02d}';folder.mkdir(parents=True,exist_ok=True)
    seed=config['seed_start']+batch+(1000000 if scheme=='adapted' else 0)
    q,branches,inverse=nodes(output,scheme,power,seed)
    contract=dict(config=config,settings=settings,scheme=scheme,seed=seed,vectorized=vectorized,
        sampling_mass_sha256=sha256(output/('adapted_mass.npy' if scheme=='adapted' else 'proposal_mass.npy')) if scheme!='radial' else None,
        kernels={n:sha256(ROOT/n) for n in KERNELS},inputs=json.loads((output/'inputs.json').read_text()))
    state_path=folder/'state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(contract=contract,completed_pairs=0,files={},cumulative_seconds=0.)
    if state['contract']!=contract:raise ValueError('incompatible physical/source/sampling cache; preserve it and use a fresh output')
    for i in range(min(state['completed_pairs'],len(q))):
        path=folder/f'pair{i:06d}.npz'
        if sha256(path)!=state['files'][path.name]:raise ValueError('changed committed pair')
        with np.load(path) as d:
            if not (np.array_equal(q[i],d['q_fractional']) and np.array_equal(branches[i],d['branches']) and np.array_equal(inverse[i],d['inverse_marginals'])):
                raise ValueError('changed nested node or importance weight')
    started=time.monotonic();new=0;status='COMPLETE';reason=None
    try:
        for i in range(state['completed_pairs'],len(q)):
            if max_new is not None and new>=max_new:raise TimeoutError('bounded new-pair count reached')
            limits.check();before=time.monotonic()
            result=propagate_bulk_leading_pair(model,positions,reference,covariance,transverse,length,settings['slices'],
                q[i],branches[i],inverse[i],background,folder/'scratch',config['temperature_k'],limits,
                save_amplitudes=(i==0),workers=1,vectorized=vectorized,reference_kind=config['reference_kind'])
            result.update(q_fractional=q[i],branches=branches[i],inverse_marginals=inverse[i])
            path=folder/f'pair{i:06d}.npz';temp=path.with_suffix('.tmp.npz');io=time.monotonic()
            np.savez_compressed(temp,**result);temp.replace(path)
            elapsed=time.monotonic()-before;state['files'][path.name]=sha256(path)
            state.update(completed_pairs=i+1,cumulative_seconds=state['cumulative_seconds']+elapsed,
                         io_seconds_last=time.monotonic()-io,target_pairs_latest_request=len(q))
            json_write(state_path,state);new+=1
    except TimeoutError as error:status='PARTIAL';reason=str(error)
    row=dict(batch=batch,new_pairs=new,completed_pairs=state['completed_pairs'],requested_pairs=len(q),status=status,
        reason=reason,seconds=time.monotonic()-started,peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        contract_fingerprint=fingerprint(contract))
    invocation=output/'invocations'/f'{time.time_ns()}_{os.getpid()}_{scheme}_batch{batch}.json';json_write(invocation,row)
    print(scheme,row,flush=True);return row


def run(output,scheme,power,batches=8,workers=2,grid=None,steps=None,max_new=None,vectorized=False):
    # Prepare shared immutable fields sequentially; processes never race writes.
    config,settings,data=physical_setup(output,grid,steps)
    for mapped in data[5:7]:mapped._mmap.close()
    tasks=[(str(output),scheme,power,b,settings['grid_unit'],settings['slices'],max_new,vectorized) for b in range(batches)]
    if workers==1:return [run_batch(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:return list(pool.map(run_batch,tasks))


def profile(output):
    config,settings,data=physical_setup(output)
    positions,reference,length,covariance,grid,background,incident,_=data
    model=RealSpaceForceConstants.load(PRIMITIVE);q,b,w=nodes(output,'born',1,config['seed_start'])
    rows=[];saved={}
    for vectorized in (False,True):
        for i in range(2):
            start=time.monotonic()
            result=propagate_bulk_leading_pair(model,positions,reference,covariance,grid,length,settings['slices'],q[i],b[i],w[i],
                background,output/'profile_scratch',config['temperature_k'],save_amplitudes=True,
                vectorized=False,retain_all_words=not vectorized,reference_kind=config['reference_kind'])
            saved[vectorized,i]=result
            rows.append(dict(vectorized=vectorized,pair=i,seconds=time.monotonic()-start,
                components={k:float(v) for k,v in result.items() if k.endswith('_seconds')},
                peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    agreement=[]
    for i in range(2):
        a=saved[False,i];b=saved[True,i]
        agreement.append(dict(pair=i,amplitude_relative_error=float(np.linalg.norm(a['direct_amplitudes']+a['sequential_amplitudes']-b['direct_amplitudes']-b['sequential_amplitudes'])/np.linalg.norm(b['direct_amplitudes']+b['sequential_amplitudes'])),
            max_intensity_absolute_difference=float(np.max(abs(a['weights']-b['weights'])))))
        np.savez_compressed(output/f'profile_pair{i}.npz',**b)
    json_write(output/'profile.json',dict(rows=rows,matched_accuracy=agreement,
        speedup=float(sum(r['seconds'] for r in rows if not r['vectorized'])/sum(r['seconds'] for r in rows if r['vectorized'])),
        baseline='25 waves:21 sequential words plus4 direct, scalar action',optimized='17 waves:13 necessary sequential words plus4 direct; same complete pair amplitude/grid/steps. Vectorization pilot did not improve measured cost.',
        limits='Two-node timing pilot, not an end-to-end speedup or target convergence.'))
    print(json.dumps({'profile':rows,'agreement':agreement}),flush=True)


def analyze(output,scheme,batches=8):
    root=output/'pairs'/scheme;counts=[]
    for b in range(batches):counts.append(json.loads((root/f'batch{b:02d}/state.json').read_text())['completed_pairs'])
    n=min(counts);unit=36 if scheme=='stratified' else 1
    balanced=unit*2**int(np.floor(np.log2(n/unit))) if n>=unit else 0
    if not balanced:raise ValueError('no balanced independent-batch prefix')
    prefixes=[]
    for count in sorted(set([balanced//2,balanced])):
        if count<unit:continue
        all_spectra=[];all_raw={};components={k:[] for k in ('direct_weights','sequential_weights','interference_weights','weights')}
        timing=[]
        for b in range(batches):
            energy=[];values={k:[] for k in components}
            for i in range(count):
                with np.load(root/f'batch{b:02d}/pair{i:06d}.npz') as d:
                    energy.append(d['energies_mev'])
                    for key in values:values[key].append(d[key])
                    timing.append(float(d['elapsed_seconds']))
            e=np.array(energy)
            for key,v in values.items():
                a=np.array(v);components[key].append(np.stack([line_spectra(e[:,s],a[:,s]/count,AXIS) for s in range(4)],axis=1).sum(axis=1))
            a=np.array(values['weights'])
            for key,(lo,hi) in WINDOWS.items():
                select=(e>=lo)&(e<hi)
                if key=='all_loss':select=e>0
                if key=='all_gain':select=e<0
                all_raw.setdefault(key,[]).append(np.sum(a*select[...,None],axis=(0,1))/count)
            for key,v in values.items():
                for sign,select in (('loss',e>0),('gain',e<0),('creation_creation',np.broadcast_to(np.arange(4)==0,e.shape))):
                    all_raw.setdefault(f'{key}_{sign}',[]).append(np.sum(np.array(v)*select[...,None],axis=(0,1))/count)
        spectra=np.array(components['weights']);mean=spectra.mean(axis=0);sem=spectra.std(axis=0,ddof=1)/np.sqrt(batches)
        raw={}
        for key,v in all_raw.items():
            v=np.array(v);m=v.mean(axis=0);h=student_t.ppf(.975,batches-1)*v.std(axis=0,ddof=1)/np.sqrt(batches)
            raw[key]=dict(mean=m,halfwidth95=h,relative_halfwidth95=np.divide(h,abs(m),out=np.full_like(h,np.inf),where=abs(m)>1e-30))
        np.savez_compressed(output/f'{scheme}_n{count}_analysis.npz',energy_mev=AXIS,batch_spectra=spectra,mean=mean,standard_error=sem,
            **{f'batch_{k}':np.array(v) for k,v in all_raw.items()},**{f'spectra_{k}':np.array(v) for k,v in components.items()})
        prefixes.append(dict(pairs_per_batch=count,total_pairs=count*batches,raw=raw,mean_spectrum=mean,
            mean_pair_seconds=float(np.mean(timing)),summed_pair_seconds=float(np.sum(timing))))
    comparisons=spectral_comparison(prefixes[0]['mean_spectrum'],prefixes[-1]['mean_spectrum']) if len(prefixes)>1 else None
    gates=dict(integrated_loss_all_detectors=bool(np.all(np.array(prefixes[-1]['raw']['all_loss']['relative_halfwidth95'])<.1)),
        shape=bool(comparisons and all(r['shape_pass'] for r in comparisons)),
        identifiable_features=bool(comparisons and all(r['all_peak_gate_pass'] for r in comparisons)),
        numerical_discretization='NOT_ESTABLISHED_BY_SAMPLING_RUN',
        full_fourth_order_total_spectrum='NOT_COMPUTED: A1*A3 and A0*A4 omitted')
    for p in prefixes:del p['mean_spectrum']
    result=dict(scheme=scheme,counts=counts,balanced_used_pairs_per_batch=balanced,batches=batches,prefixes=prefixes,
        comparisons=comparisons,gates=gates,status='NUMERICALLY_UNCONVERGED',
        uncertainty='Student t95 interval across independent scrambles; excludes deterministic/electron/interaction truncation errors; pairs within a scramble not treated independent.')
    json_write(output/f'{scheme}_analysis.json',finite_json(result));print(scheme,gates,flush=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT/'bulk_static_v2')
    sub=p.add_subparsers(dest='stage',required=True)
    s=sub.add_parser('setup');s.add_argument('--reference-kind',choices=('static','thermal_normal_ordered'),default='static');s.add_argument('--temperature',type=int,default=300);s.add_argument('--repeats',type=int,default=4)
    sub.add_parser('profile')
    r=sub.add_parser('run');r.add_argument('--scheme',choices=('radial','born','stratified','adapted'),default='born');r.add_argument('--power',type=int,default=3);r.add_argument('--batches',type=int,default=8);r.add_argument('--workers',type=int,default=2);r.add_argument('--max-new',type=int);r.add_argument('--grid',type=int);r.add_argument('--steps',type=int)
    a=sub.add_parser('analyze');a.add_argument('--scheme',choices=('radial','born','stratified','adapted'),default='born');a.add_argument('--batches',type=int,default=8)
    args=vars(p.parse_args());stage=args.pop('stage');output=args.pop('output')
    if stage=='setup':setup(output,**args)
    elif stage=='profile':profile(output)
    elif stage=='run':run(output,**args)
    else:analyze(output,**args)

if __name__=='__main__':main()
