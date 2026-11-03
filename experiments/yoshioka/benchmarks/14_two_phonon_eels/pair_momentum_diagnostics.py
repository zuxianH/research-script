"""Guarded coherent Born intermediate-momentum diagnostics; no production net."""
import argparse,json,time,resource
from pathlib import Path
import numpy as np
from .workflow_io import ROOT,ResourceLimits,json_write,sha256,environment
from .approximate_eels import PRIMITIVE,ORIENTED,DETECTORS,geometry
from .two_phonon_analysis import WINDOWS,AXIS,spectral_comparison
from .born_quadrature import line_spectra
from .complete_leading_eels import finite_json
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.free_propagation import TransverseGrid
from yoshioka.bulk_thermal_source import mode_group,occupations
from yoshioka.electron_phonon import LobatoSilicon
from yoshioka.pair_intermediate_momentum import (detector_output_labels,build_fourier_sources,
    propagate_intermediate_resolved,cumulative_amplitudes,projected_trace)

BASE=ROOT/'benchmark_results/14_two_phonon_eels/pair_convergence_continuation_20261102_v1/momentum_audit'
KERNELS=('yoshioka/pair_intermediate_momentum.py','yoshioka/electron_phonon.py','yoshioka/bulk_thermal_source.py',
    'yoshioka/lattice_dynamics.py','yoshioka/dynamical_source.py','yoshioka/free_propagation.py',
    'yoshioka/elastic_propagation.py','yoshioka/two_phonon.py','benchmarks/approximate_eels.py',
    'benchmarks/two_phonon_analysis.py','benchmarks/born_quadrature.py','benchmarks/workflow_io.py')


def raw_windows(trace):
    e=trace['energies_mev'];w=trace['weights'];values=[]
    for key,(lo,hi) in WINDOWS.items():
        pick=(e>=lo)&(e<hi)
        if key=='all_loss':pick=e>0
        if key=='all_gain':pick=e<0
        values.append(np.sum(w*pick[:,None],axis=-2))
    return np.stack(values,axis=-2)


def reduction(r,grid,iw):
    h=r['intermediate_integer_labels'];n=grid.nx
    sizes=[v for v in (9,17,25,33,49,65,81,97,129,193,n) if v<=n and np.max(abs(r['output_integer_labels']))<=v//2]
    sizes=sorted(set(sizes))
    rect=np.array([np.broadcast_to(np.all(abs(h)<=v//2,axis=1),(4,2,len(h))) for v in sizes])
    maximum=float(np.max(r['intermediate_Q_over_k0']))
    edges=np.unique(np.r_[0.,.02,.05,.1,.2,.3,.4,.5,.6,.75,1.,np.nextafter(maximum,np.inf)])
    radial=np.array([r['intermediate_Q_over_k0']<edge for edge in edges])
    cumulative={};arrays={};nb=occupations(r['omega_rad_s'],300)
    for name,masks,labels in [('rect',rect,np.array(sizes)),('radial',radial,edges)]:
        c=cumulative_amplitudes(r,masks)
        trace=projected_trace(r['direct_amplitudes'][None],c['sequential'],r['omega_rad_s'],nb,grid,
                             r['final_shifts'],r['output_integer_labels'],iw)
        windows=raw_windows(trace)
        spectra=np.array([line_spectra(trace['energies_mev'],w,AXIS) for w in trace['weights']])
        arrays.update({f'{name}_'+k:v for k,v in c.items()})
        arrays.update({f'{name}_'+k:trace[k] for k in ('weights','direct_weights','sequential_weights','interference_weights')})
        arrays.update({f'{name}_labels':labels,f'{name}_masks':masks,f'{name}_window_weights':windows,f'{name}_spectra':spectra})
        delta=windows-windows[-1]
        rel=np.divide(abs(delta),abs(windows[-1]),out=np.full_like(delta,np.nan),where=abs(windows[-1])>1e-18)
        cumulative[name]=dict(labels=labels,node_counts=masks.sum(axis=-1),window_order=list(WINDOWS),
            windows=windows,absolute_difference_from_largest=delta,relative_difference_from_largest=rel,
            spectral_sensitivity=[spectral_comparison(s,spectra[-1]) for s in spectra],
            scope='Frozen selected-node sensitivity. No BZ confidence interval, tail-to-infinity bound, or acceptance certificate.')
    increments=np.diff(arrays['radial_ordered'],axis=0)
    arrays['radial_ordered_shell_increments']=increments
    arrays['radial_shell_sequential_increments']=increments.sum(axis=2)
    arrays['radial_shell_counts']=np.diff(radial.astype(int),axis=0).sum(axis=-1)
    # Exact interference identity for successive cumulative coherent states.
    before=arrays['radial_total'][:-1];inc=arrays['radial_total'][1:]-before
    arrays['radial_increment_self_probability']=abs(inc)**2
    arrays['radial_increment_interference_probability']=2*np.real(before*inc.conj())
    arrays['radial_increment_total_probability']=abs(arrays['radial_total'][1:])**2-abs(before)**2
    # Bound roundoff by the probabilities being subtracted, rather than by a
    # potentially cancelling probability increment. This is an algebraic check.
    residual=abs(arrays['radial_increment_self_probability']+arrays['radial_increment_interference_probability']-
        arrays['radial_increment_total_probability'])
    scale=abs(arrays['radial_total'][1:])**2+abs(before)**2+abs(inc)**2+abs(arrays['radial_increment_interference_probability'])
    roundoff_bound=128*np.finfo(float).eps*scale+np.finfo(float).tiny
    if np.any(residual>roundoff_bound):raise AssertionError('coherent probability increment identity exceeded floating-point bound')
    arrays['radial_increment_identity_residual']=residual
    arrays['radial_increment_identity_roundoff_bound']=roundoff_bound
    finaltrace=projected_trace(r['direct_amplitudes'],r['sequential_amplitudes'],r['omega_rad_s'],nb,grid,r['final_shifts'],r['output_integer_labels'],iw)
    mask=finaltrace['detector_masks']
    condition=[]
    for j in range(4):
        pick=mask[:,j]
        d=r['direct_amplitudes']*pick;s=r['sequential_amplitudes']*pick
        condition.append((np.linalg.norm(d)+np.linalg.norm(s))/max(np.linalg.norm(d+s),1e-300))
    cumulative['cancellation']=dict(direct_sequential_condition_by_detector=condition,
        shell_norm_sum_over_coherent_sequential=float(np.sum(np.sqrt(np.sum(abs(increments.sum(axis=2))**2,axis=(-1,-2))))/max(np.linalg.norm(r['sequential_amplitudes']),1e-300)),
        definition='Norms of complex amplitudes, no independent-shell probabilities.')
    arrays.update(detector_masks=mask,energies_mev=finaltrace['energies_mev'],window_names=np.array(list(WINDOWS)),energy_axis_mev=AXIS)
    return cumulative,arrays


def compare_prior(path,r,arrays,n):
    prior_state=json.loads((path.parent/'state.json').read_text())
    if prior_state['files'].get(path.name)!=sha256(path):raise ValueError('changed committed prior Born file')
    c=prior_state['contract']
    if not c['born'] or c['steps']!=r['steps'] or c['local_substeps']!=r['local_substeps'] or c['retained']!=n:
        raise ValueError('unmatched prior Born configuration')
    index=list(arrays['rect_labels']).index(n)
    labels=r['output_integer_labels'];iy,ix=labels[:,0]%n,labels[:,1]%n
    with np.load(path) as z:
        for key in ('q_fractional','branches','inverse_marginals','omega_rad_s','final_shifts'):
            if not np.array_equal(z[key],r[key]):raise ValueError('unmatched prior '+key)
        new={'direct_amplitudes':r['direct_amplitudes'],'sequential_amplitudes':arrays['rect_sequential'][index],
             'total_amplitudes':arrays['rect_total'][index]}
        result={}
        for key,value in new.items():
            old=z[key][:,iy,ix] if key in z else (z['direct_amplitudes']+z['sequential_amplitudes'])[:,iy,ix]
            result[key]=dict(absolute_l2=float(np.linalg.norm(value-old)),relative_l2=float(np.linalg.norm(value-old)/max(np.linalg.norm(old),1e-300)))
        result['all_detector_weight_max_absolute']=float(np.max(abs(arrays['rect_weights'][index]-z['weights'])))
    result.update(prior=str(path),sha256=sha256(path),retained_grid=n,
       compared_subspace='exact union of all four physical detector supports and sign sectors; no phase/scale fit')
    if max(result[k]['relative_l2'] for k in new)>1e-8:raise ValueError('independent prior Born extraction verification failed: '+str(result))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE);parser.add_argument('--case',required=True)
    parser.add_argument('--panel',type=Path,required=True);parser.add_argument('--index',type=int,required=True)
    parser.add_argument('--retained',type=int,default=65);parser.add_argument('--steps',type=int,default=769)
    parser.add_argument('--substeps',type=int,default=1);parser.add_argument('--seconds',type=float,default=600)
    parser.add_argument('--verify-prior',type=Path,action='append',default=[])
    a=parser.parse_args();folder=a.output/a.case;folder.mkdir(parents=True,exist_ok=True)
    panel=json.loads(a.panel.read_text());row=panel['pairs'][a.index];source=Path(row['source_path'])
    if sha256(source)!=row['source_sha256']:raise ValueError('changed frozen source')
    potential=LobatoSilicon.from_abtem_table()
    inputs=[PRIMITIVE,ORIENTED,ROOT/'Si.tersoff.mod',ROOT/'benchmark_results/14_two_phonon_eels/complete_leading_20261102_v1/bulk_static_v2/criteria.json']
    contract=finite_json(dict(version=2,retained=a.retained,steps=a.steps,local_substeps=a.substeps,
        source=str(source),source_sha256=sha256(source),panel=str(a.panel.resolve()),panel_sha256=sha256(a.panel),index=a.index,
        kernels={p:sha256(ROOT/p) for p in KERNELS},entrypoint_sha256=sha256(__file__),
        inputs={str(p.relative_to(ROOT)):sha256(p) for p in inputs},lobato_sha256=sha256(potential.source_path),
        prior_files={str(p):sha256(p) for p in a.verify_prior},environment=environment(),
        physics='fourrepeat1.535553nm,300K,60keV,staticC0,originalV0=0control,unitincident,fourdetectors,2meVFWHM',
        operator='same paraxial kinetic Strang and exact RK4 polynomial; each ordered intermediate beam explicit',
        field_representation='original finite-kz Fourier midpoint grid, all displacement components; contact Fourierrepresented not exactevent',
        runtime_seconds=a.seconds,memory_gib=4,scientific_workers=1,fft_workers=1))
    statepath=folder/'state.json'
    if statepath.exists():
        state=json.loads(statepath.read_text())
        if state['contract']!=contract:raise ValueError('changed momentum case contract; use new identifier')
        for name,value in state.get('files',{}).items():
            if sha256(folder/name)!=value:raise ValueError('changed committed momentum output')
        if state.get('status')=='COMPLETE':print(a.case,'COMPLETE immutable read-only reuse');return
    else:
        state=dict(contract=contract,status='PREPARED',files={});json_write(statepath,state)
        snap=folder/'source';snap.mkdir()
        for p in KERNELS:(snap/p.replace('/','__')).write_bytes((ROOT/p).read_bytes())
        (snap/'driver.py').write_bytes(Path(__file__).read_bytes())
    limits=ResourceLimits(a.seconds,4);started=time.monotonic();linear=quad=None
    try:
        with np.load(source) as z:q=z['q_fractional'];branches=z['branches'];iw=z['inverse_marginals']
        oriented=RealSpaceForceConstants.load(ORIENTED);model=RealSpaceForceConstants.load(PRIMITIVE)
        positions,reference,length=geometry(oriented,4,12)
        grid=TransverseGrid(oriented.cell_m[0,0],oriented.cell_m[1,1],a.retained,a.retained)
        omega=[];d=[];shifts=[]
        for side in range(2):
            om,disp,shift,labels=mode_group(model,positions,[grid.lx_m,grid.ly_m],q[side],1.,reference)
            at=int(np.flatnonzero(labels==branches[side])[0]);omega.append(om[at]);d.append(disp[:,:,at]);shifts.append(shift)
        omega=np.array(omega);shifts=np.array(shifts);d=np.stack(d,axis=-1)
        out=detector_output_labels(grid,shifts)
        np.savez_compressed(folder/'parameters.npz',q_fractional=q,branches=branches,inverse_marginals=iw,
           positions_m=positions,reference_m=reference,displacements_m=d,omega_rad_s=omega,shifts_m_inv=shifts,
           cell_lengths_m=[grid.lx_m,grid.ly_m],length_m=length,output_integer_labels=out)
        before=time.monotonic();linear,quad,T=build_fourier_sources(positions,d,shifts,grid,length,a.steps,out,folder/'scratch',limits=limits)
        field_seconds=time.monotonic()-before
        before=time.monotonic()
        r=propagate_intermediate_resolved(grid,omega,shifts,length,a.steps,lambda i:np.asarray(linear[i]),
            lambda i:np.asarray(quad[i,0]),T,out,a.substeps,lambda i:np.asarray(quad[i,1]),limits)
        propagation_seconds=time.monotonic()-before
        r.update(q_fractional=q,branches=branches,inverse_marginals=iw)
        cumulative,arrays=reduction(r,grid,iw)
        verification=[compare_prior(p,r,arrays,json.loads((p.parent/'state.json').read_text())['contract']['retained']) for p in a.verify_prior]
        np.savez_compressed(folder/'resolved.npz',**r)
        np.savez_compressed(folder/'cumulative.npz',**arrays)
        result=dict(status='DIAGNOSTIC_ONLY_NO_PRODUCTION_ACCEPTANCE',source=row,retained=a.retained,steps=a.steps,
            transfer_inventory=T,output_beams=len(out),intermediate_beams=a.retained**2,
            incident_k0_m_inv=r['incident_k0_m_inv'],incident_wavelength_m=r['incident_wavelength_m'],
            actual_Q_over_k0_range=[float(np.min(r['intermediate_Q_over_k0'])),float(np.max(r['intermediate_Q_over_k0']))],
            prior_complex_verification=verification,cumulative=cumulative,
            costs=dict(field_seconds=field_seconds,propagation_seconds=propagation_seconds,total_seconds=time.monotonic()-started,
                peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024),
            interpretation='Complex amplitudes, direct/contact and cross-shell interference retained. No infinite-tail bound, paraxial-failure claim, or selected-node BZ convergence certificate.')
        json_write(folder/'analysis.json',finite_json(result))
        state.update(status='COMPLETE',files={name:sha256(folder/name) for name in ('parameters.npz','resolved.npz','cumulative.npz','analysis.json')})
        json_write(statepath,state);print(a.case,result['costs'],'verified',verification,flush=True)
    except (TimeoutError,MemoryError) as error:
        state.update(status='PARTIAL_RESOURCE_LIMIT',reason=str(error));json_write(statepath,state);raise
    finally:
        for mapped in (linear,quad):
            if mapped is not None:mapped._mmap.close()
        for name in ('linear.npy','quadratic.npy'):(folder/'scratch'/name).unlink(missing_ok=True)


if __name__=='__main__':main()
