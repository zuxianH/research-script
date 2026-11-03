"""Measured two-phonon convergence, figures, conditional reference and report."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import t as student_t

from .two_phonon_eels import DEFAULT,PREVIOUS,CONTROLLED,ORIENTED
from .two_phonon_analysis import (AXIS,PROCESSES,FEATURES,load_job,raw_batches,spectra,
    estimate,summarize,spectral_comparison,detected_features,baseline)
from .approximate_eels import DETECTORS
from .approximate_eels_analysis import features
from .approximate_eels_compare import match_features
from .workflow_io import json_write,sha256
from yoshioka.lattice_dynamics import RealSpaceForceConstants


def counts(output,job):
    paths=sorted((output/'pairs'/job).glob('*/state.json'))
    return [json.loads(p.read_text())['completed_pairs'] for p in paths]


def collect(output):
    convergence={};means={};final={};data={};batch_spectra={}
    for job in ('main','importance'):
        numbers=counts(output,job)
        if len(numbers)!=8:continue
        rows=[];previous=None
        for n in (8,16,32,64,128,256):
            if min(numbers)<n:break
            result,mean,s,d=summarize(output,job,n)
            row={'n':n,'total_pairs':8*n,'primary_loss':result['detectors'][0]['raw']['all_loss'],
                 'primary_creation':result['detectors'][0]['raw']['creation_creation']}
            if previous is not None:row['comparison_from_previous']=spectral_comparison(previous,mean)
            rows.append(row);previous=mean
            final[job]=result;means[job]=mean;data[job]=d;batch_spectra[job]=s
        convergence[job]=rows
    json_write(output/'pair_convergence.json',convergence)
    return convergence,final,means,data,batch_spectra


def numerical(output,production_mean):
    rows=[]
    for job in ('electron','slices','joint','covariance'):
        number=counts(output,job)
        if len(number)!=8 or min(number)<16:
            rows.append({'kind':job,'status':'INCOMPLETE','completed_counts':number});continue
        fine=load_job(output,job,16);coarse=load_job(output,'main',16)
        np.testing.assert_array_equal(coarse['nodes'],fine['nodes'])
        np.testing.assert_array_equal(coarse['branches'],fine['branches'])
        a=spectra(coarse).sum(axis=2);b=spectra(fine).sum(axis=2)
        correction=b-a
        refined=production_mean+np.mean(correction,axis=0)
        compared=spectral_comparison(production_mean,refined)
        competing=[]
        for j,detector in enumerate(compared):
            for key,shift in detector['identifiable_feature_shifts_mev'].items():
                if shift is None or shift<=1:continue
                before=detector['coarse_features'][key]['all_peaks_mev']
                after=detector['fine_features'][key]['all_peaks_mev']
                coarse_peaks=sorted([(e,float(np.interp(e,AXIS,production_mean[:,j]))) for e in before],key=lambda x:x[1],reverse=True)
                fine_peaks=sorted([(e,float(np.interp(e,AXIS,refined[:,j]))) for e in after],key=lambda x:x[1],reverse=True)
                competing.append(dict(detector=detector['detector'],feature=key,dominant_shift_mev=shift,
                    coarse_peaks_and_heights=coarse_peaks[:5],fine_peaks_and_heights=fine_peaks[:5],
                    coarse_leading_relative_height_gap=(coarse_peaks[0][1]-coarse_peaks[1][1])/coarse_peaks[0][1] if len(coarse_peaks)>1 and coarse_peaks[0][1]>0 else None,
                    fine_leading_relative_height_gap=(fine_peaks[0][1]-fine_peaks[1][1])/fine_peaks[0][1] if len(fine_peaks)>1 and fine_peaks[0][1]>0 else None,
                    status='PEAK_GATE_NOT_MET; competing maxima require integration/statistical evidence, not a phonon-frequency interpretation'))
        raw_a=raw_batches(coarse);raw_b=raw_batches(fine)
        raw_rows={}
        for key in raw_a:
            av=np.mean(raw_a[key],axis=0);bv=np.mean(raw_b[key],axis=0)
            delta,sem,half=estimate(raw_b[key]-raw_a[key])
            raw_rows[key]=dict(coarse=av,fine=bv,difference=delta,halfwidth95=half,
                absolute_difference=abs(delta),relative_difference_fine_denominator=abs(delta)/np.maximum(abs(bv),1e-18))
        rows.append(dict(kind=job,status='SAMPLED_CONTROLS_COMPUTED',matched_pairs=128,
            settings=fine['contract']['settings'],observables=raw_rows,
            competing_peak_diagnosis=competing,
            detector_comparisons=compared,paired_subset_comparison=spectral_comparison(np.mean(a,axis=0),np.mean(b,axis=0)),
            scope='Primary production plus independent fixed-node fine-minus-coarse integral estimate. Numerical sensitivity, not complete fine-grid production or a rigorous bias bound.',
            primary_targets_met=compared[0]['shape_pass'] and compared[0]['identifiable_peak_pass']))
    json_write(output/'numerical_convergence.json',rows)
    return rows


def low_q(output,production_data):
    number=counts(output,'lowq')
    if len(number)!=8 or min(number)<32:
        result={'status':'INCOMPLETE','completed_counts':number}
    else:
        a=load_job(output,'main',32);b=load_job(output,'lowq',32)
        ra=raw_batches(a);rb=raw_batches(b)
        delta,sem,half=estimate(rb['all_loss']-ra['all_loss'])
        result={'status':'COMPUTED','coarse_radius':.35,'alternative_radius':.2,
            'same_physics_and_full_BZ':True,'paired_batch_loss_difference':delta,'difference_halfwidth95':half,
            'zero_difference_inside_95_interval':abs(delta)<=half,
            'spectral_comparisons':spectral_comparison(np.mean(spectra(a).sum(axis=2),axis=0),np.mean(spectra(b).sum(axis=2),axis=0)),
            'scope':'Different PDFs and partly common random nodes; this tests integration/variance, not a changed low-q physical cutoff.'}
    q=production_data['nodes'];energy=production_data['energy'];weight=production_data['weight']
    small=np.all(abs(q)<1/(2*64),axis=-1)
    select=np.any(small,axis=-1)
    all_loss=np.sum(weight*(energy>0)[...,None],axis=2)
    result['Gamma_cell_definition']='primitive fractional cube[-1/128,1/128]^3; source modes recomputed continuously, no deleted acoustic cell'
    result['sampled_pairs_touching_Gamma_cell']=int(select.sum())
    result['loss_weight_from_pairs_touching_Gamma_cell']=np.mean(np.mean(all_loss*select[...,None],axis=1),axis=0)
    result['fraction_of_loss_from_touching_Gamma_cell']=np.sum(all_loss*select[...,None],axis=(0,1))/np.maximum(np.sum(all_loss,axis=(0,1)),1e-100)
    json_write(output/'low_q_diagnostic.json',result)
    return result


def regional(output,data):
    raw=np.sum(data['weight']*(data['energy']>0)[...,None],axis=2)
    branches=data['branches'];table=np.zeros((6,6,4))
    for a in range(6):
        for b in range(6):table[a,b]=np.sum(raw*((branches[:,:,0]==a)&(branches[:,:,1]==b))[...,None],axis=(0,1))/(data['batches']*data['n'])
    r=np.linalg.norm(data['nodes'],axis=-1);bins=np.array([0,.05,.15,.35,np.inf])
    regions=np.zeros((4,4,4))
    for a in range(4):
        for b in range(4):
            mask=(r[:,:,0]>=bins[a])&(r[:,:,0]<bins[a+1])&(r[:,:,1]>=bins[b])&(r[:,:,1]<bins[b+1])
            regions[a,b]=np.sum(raw*mask[...,None],axis=(0,1))/(data['batches']*data['n'])
    json_write(output/'branch_and_q_regions.json',dict(branch_pair_loss_weights=table,
        branch_labels='frequency-sorted primitive branch0..5 at each q; no arbitrary eigenvector interpolation',
        radial_fractional_q_boundaries=[0,.05,.15,.35,'BZ edge'],radial_pair_loss_weights=regions,
        normalization='All sampled contributions retained, ordered-pair1/2 already applied. Table sum equals total raw loss.',
        phonon_q_not_electron_Q=True))


def feature_uncertainty(output,batch_spectra):
    rng=np.random.default_rng(90178);batch=batch_spectra.sum(axis=2)
    full=np.mean(batch,axis=0);items=[]
    indices=rng.integers(0,8,size=(500,8))
    for j,name in enumerate(DETECTORS):
        samples={k:[] for k in FEATURES};missing={k:0 for k in FEATURES}
        for row in indices:
            found=detected_features(AXIS,np.mean(batch[row,:,j],axis=0))
            for key,value in found.items():
                if value['peak_mev'] is None:missing[key]+=1
                else:samples[key].append(value['peak_mev'])
        central=detected_features(AXIS,full[:,j])
        for key in FEATURES:
            items.append(dict(detector=name,feature=key,nominal=central[key],unresolved_resamples=missing[key],
                bootstrap500_peak_percentile95_mev=np.percentile(samples[key],[2.5,97.5]) if samples[key] else None,
                scope='Finite-eight-batch bootstrap of strongest local maxima; competing maxima are retained, not tracked as identical modes.'))
    json_write(output/'feature_uncertainty.json',items)
    return items


def conditional_tacaw(output,mean):
    from .approximate_eels_analysis import shape
    path=PREVIOUS/'TACAW_T300_raw_reductions.npz'
    with np.load(path) as d:
        axis=d['broadened_axis_mev'];step=axis[1]-axis[0]
        tacaw_density=d['all_recorded_momenta_broadened_weights']/step
        tacaw=np.interp(AXIS,axis,tacaw_density)
    first=baseline()['spectrum'][:,0]
    # Preserve the historical conditioning GRID as well as its physical window.
    # Floating arange endpoints otherwise change an edge sample when extending
    # the display from[-100,100] to[-180,180]. No reference recalibration occurs.
    with np.load(CONTROLLED/'fixed_r4_T300_spectra.npz') as d:
        historical_axis=d['energy_mev'].copy();historical_first=d['DW'][:,0].copy()
    historical_tacaw=np.interp(historical_axis,axis,tacaw_density)
    historical_pair=np.interp(historical_axis,AXIS,mean[:,0])
    ht,ta=shape(historical_axis,historical_tacaw)
    ha,aa=shape(historical_axis,historical_first)
    hb,ba=shape(historical_axis,historical_first+historical_pair)
    ty=tacaw/ta;a=first/aa;b=(first+mean[:,0])/ba
    select=(historical_axis>=5)&(historical_axis<90)
    result=dict(status='CONDITIONAL_ONLY',calibrated_status='BLOCKED_BY_METADATA',
        source=str(path),sha256=sha256(path),cadence_fs_assumed=7.5,
        raw_array_files_unchanged=True,Kubo_already_applied=True,second_Kubo_applied=False,
        conditioning='Each curve divided by its own5-90meV integral for shape display only; same convention extends into the displayed high-energy tail.',
        historical_conditioning_grid_preserved=True,
        one_phonon_shape_L1=float(trapezoid(abs(ha[select]-ht[select]),historical_axis[select])),
        limited_order_shape_L1=float(trapezoid(abs(hb[select]-ht[select]),historical_axis[select])),
        one_phonon_features=features(AXIS,first),limited_order_features=features(AXIS,first+mean[:,0]),
        feature_matching=match_features(features(AXIS,tacaw),features(AXIS,first+mean[:,0])),
        limitations='Unverified generating cadence/sign/native resolution/detector/geometry/force provenance. High-energy tails do not validate a two-phonon mechanism. No absolute scale or speedup comparison.')
    json_write(output/'conditional_TACAW.json',result)
    np.savez_compressed(output/'conditional_TACAW.npz',energy_mev=AXIS,TACAW_conditioned=ty,
        one_phonon_conditioned=a,limited_order_conditioned=b,
        raw_one_phonon=first,raw_limited_order=first+mean[:,0])
    return result


def figures(output,final,means,data,batch_spectra,convergence,numerical_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    job='importance' if 'importance' in means else 'main'
    mean=means[job];s=batch_spectra[job];process=s.mean(axis=0)
    sampling_label='Sampling INCOMPLETE' if json.loads((output/'results.json').read_text())['status']=='INCOMPLETE' else 'Approximate numerical targets met'
    _,_,half=estimate(s.sum(axis=2));first=baseline()['spectrum']
    fig,ax=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    ax[0].plot(AXIS,mean[:,0],color='black',label='Two linear interactions: changed bath')
    ax[0].fill_between(AXIS,mean[:,0]-half[:,0],mean[:,0]+half[:,0],color='grey',alpha=.2,label='Pointwise nominal 95% interval')
    for k,label,color in ((0,'Two created','tab:blue'),(1,'Two annihilated','tab:purple')):ax[0].plot(AXIS,process[:,k,0],label=label,lw=1.,color=color)
    ax[0].plot(AXIS,process[:,2:,0].sum(axis=1),label='Creation + annihilation',lw=1.,color='tab:green')
    ax[0].set(xlim=(-160,160),xlabel='Electron energy loss (meV)',ylabel='Model weight / meV',title='Leading two-interaction processes, 300 K')
    ax[1].plot(AXIS,first[:,0],label='One linear interaction')
    ax[1].plot(AXIS,first[:,0]+mean[:,0],label='Limited-order changed-bath sum')
    ax[1].plot(AXIS,mean[:,0],label='Two-interaction contribution')
    ax[1].set(xlim=(5,160),yscale='log',ylim=(1e-13,2e-5),xlabel='Electron energy loss (meV)',ylabel='Model weight / meV',title='Raw model scale; no intensity normalization')
    for a in ax:a.legend(fontsize=8);a.grid(alpha=.2)
    fig.suptitle('Si [110], 60 keV, 1.536 nm, all six branches and bulk q pairs; 2 meV display FWHM\n'+sampling_label+'; linear two-insertion model omits nonlinear vertices and higher orders',fontsize=10)
    fig.savefig(output/'two_phonon_EELS.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7.3),layout='constrained')
    for j,a in enumerate(axes.ravel()):
        a.plot(AXIS,mean[:,j],label='Two interactions');a.fill_between(AXIS,mean[:,j]-half[:,j],mean[:,j]+half[:,j],alpha=.2)
        a.set(xlim=(-160,160),title=DETECTORS[j].replace('_',' '),xlabel='Energy loss (meV)',ylabel='Model weight / meV');a.grid(alpha=.2)
    fig.suptitle(sampling_label+': two-interaction spectra and nominal 95% batch intervals',fontsize=11)
    fig.savefig(output/'detectors_and_processes.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for name,rows in convergence.items():
        n=np.array([r['total_pairs'] for r in rows]);y=np.array([r['primary_loss']['mean'] for r in rows]);h=np.array([r['primary_loss']['halfwidth95'] for r in rows])
        label='Uniform + radial' if name=='main' else 'Born-informed mixture'
        axes[0].errorbar(n,y,yerr=h,marker='o',capsize=3,label=label)
        axes[1].plot(n,h/abs(y),marker='o',label=label)
    axes[0].set(xscale='log',xlabel='Total sampled pairs',ylabel='Raw positive-energy model weight',title='Eight independent batch estimates')
    axes[1].axhline(.1,color='black',ls='--',label='10% target')
    axes[1].set(xscale='log',yscale='log',xlabel='Total sampled pairs',ylabel='Nominal 95% relative half-width',title='Sampling uncertainty; no numerical bias included')
    for a in axes:a.legend(fontsize=8);a.grid(alpha=.2)
    fig.savefig(output/'statistical_convergence.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    for row in numerical_rows:
        if row['status']=='INCOMPLETE':continue
        ax.bar(row['kind'],row['detector_comparisons'][0]['windows']['5_160']['shape_L1'])
    ax.axhline(.1,color='black',ls='--',label='10% shape target')
    ax.set(ylabel='Primary conditional shape L1, 5–160 meV',title='Fixed-pair numerical sensitivities, including joint refinement')
    ax.legend();fig.savefig(output/'numerical_convergence.png',dpi=160);plt.close(fig)
    with np.load(output/'conditional_TACAW.npz') as t:
        fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
        for a,limits in zip(axes,((5,90),(90,160))):
            display=(AXIS>=limits[0])&(AXIS<=limits[1])
            for key,label in [('TACAW_conditioned','TACAW: conditional calibration'),('one_phonon_conditioned','Yoshioka one interaction'),('limited_order_conditioned','Yoshioka limited-order sum')]:a.plot(AXIS[display],t[key][display],label=label)
            a.set(xlim=limits,xlabel='Energy loss (meV)',ylabel='Shape conditioned on 5–90 meV / meV');a.grid(alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle('Conditional comparison: unmatched acceptance, geometry and native resolution',fontsize=11)
        fig.savefig(output/'conditional_TACAW_overlay.png',dpi=160);plt.close(fig)
    # Signed corrections belong to unchanged bath states, at zero loss.
    result=final[job];elastic=result['elastic_return']
    fig,ax=plt.subplots(figsize=(8.5,4.4),layout='constrained')
    x=np.arange(4);width=.35
    ax.bar(x-width/2,elastic['order2_interference'],width,label='Signed order2 elastic interference')
    ax.bar(x+width/2,elastic['order4_return_square_unbiased'],width,label='Order4 return-square subset (unbiased estimate)')
    ax.set_xticks(x,[n.replace('_',' ') for n in DETECTORS],rotation=12)
    ax.set(ylabel='Raw integrated model weight',title='Unchanged-bath corrections at zero loss; not a positive scattering probability')
    ax.legend(fontsize=8);ax.axhline(0,color='black',lw=.6)
    fig.savefig(output/'elastic_sector_corrections.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    args=p.parse_args();output=args.output
    conv,final,means,data,batches=collect(output)
    job='importance' if 'importance' in final else 'main'
    numeric=numerical(output,means[job]);low=low_q(output,data[job]);regional(output,data[job])
    uncertainty=feature_uncertainty(output,batches[job]);tacaw=conditional_tacaw(output,means[job])
    last=conv[job][-1];comparison=last.get('comparison_from_previous',[])
    primary=comparison[0] if comparison else {}
    stat=final[job]['primary_statistical_pass']
    numeric_pass=all(row.get('primary_targets_met',False) for row in numeric)
    numeric_all_pass=all(row.get('status')!='INCOMPLETE' and all(v['shape_pass'] and v['identifiable_peak_pass'] for v in row['detector_comparisons']) for row in numeric)
    integration_pass=bool(primary.get('shape_pass',False) and primary.get('identifiable_peak_pass',False))
    result=dict(status='APPROXIMATE_TARGETS_MET' if stat and numeric_pass and integration_pass else 'INCOMPLETE',
        selected_job=job,selected_pairs=final[job]['total_pairs'],independent_batches=8,
        solver_implementation='PASS',statistical_acceptance='PASS' if stat else 'INCOMPLETE',
        numerical_controls='PRIMARY_SAMPLED_CONTROLS_PASS' if numeric_pass else 'INCOMPLETE',
        numerical_all_detector_controls='SAMPLED_CONTROLS_PASS' if numeric_all_pass else 'INCOMPLETE',
        pair_integration='PASS_APPROXIMATE_TARGETS' if integration_pass else 'INCOMPLETE',
        full_material_9B='INCOMPLETE',nonlinear_vertices='NOT_INCLUDED',higher_orders='NOT_INCLUDED',
        final=final[job],uniform_reference=final.get('main'),last_pair_comparison=comparison,
        numerical=numeric,low_q=low,conditional_TACAW=tacaw,
        limitations=['Linear two-insertion model; not complete two-phonon scattering including direct nonlinear vertices.',
            'Not complete inclusive intensity throughg4; no norm restoration.',
            'Fixed-pair numerical sensitivity is not a full fine-grid production bound.',
            'Nominal Student-t intervals have only8 independent scrambles and do not cover model/discretization error.',
            'Spectral maxima are features, not automatically phonon modes; unresolved/competing peaks retained.',
            'TACAW provenance remains unmatched; no validated absolute comparison or speedup.'])
    json_write(output/'results.json',result)
    figures(output,final,means,data,batches,conv,numeric)
    print(json.dumps({k:result[k] for k in ('status','selected_pairs','solver_implementation','statistical_acceptance','numerical_controls','pair_integration')},indent=2))


if __name__=='__main__':main()
