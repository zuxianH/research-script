"""Raw process-resolved reductions and independent-batch uncertainty."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
from scipy.stats import t as student_t

from yoshioka.free_propagation import TransverseGrid
from yoshioka.two_phonon import masks_for_shift
from .approximate_eels import DETECTORS
from .born_quadrature import line_spectra
from .two_phonon_eels import DEFAULT, CONTROLLED, PREVIOUS, ORIENTED
from .workflow_io import json_write


AXIS=np.arange(-180,180.025,.05)
PROCESSES=('creation_creation','annihilation_annihilation','creation_annihilation','annihilation_creation')
WINDOWS={'loss_0_40':(0,40),'loss_40_65':(40,65),'loss_65_90':(65,90),
         'loss_90_125':(90,125),'loss_125_160':(125,160),
         'gain_0_40':(-40,0),'gain_40_65':(-65,-40),'gain_65_90':(-90,-65),
         'gain_90_125':(-125,-90),'gain_125_160':(-160,-125),
         'all_loss':(0,np.inf),'all_gain':(-np.inf,0),'loss_5_90':(5,90),'loss_5_160':(5,160)}
FEATURES={'acoustic':(10,40),'middle':(40,60),'optical':(60,85),
          'sum_90_125':(90,125),'sum_125_160':(125,160)}


def estimate(batches):
    a=np.asarray(batches); n=len(a)
    mean=np.mean(a,axis=0)
    sem=np.std(a,axis=0,ddof=1)/np.sqrt(n) if n>1 else np.full_like(mean,np.nan)
    return mean,sem,student_t.ppf(.975,n-1)*sem


def load_job(output,job='main',n=None,batches=8):
    folders=[output/'pairs'/job/f'batch{b:02d}' for b in range(batches)]
    counts=[json.loads((p/'state.json').read_text())['completed_pairs'] for p in folders]
    n=min(counts) if n is None else n
    if n<1 or min(counts)<n: raise ValueError('incomplete independent batch prefix')
    energy=[]; weight=[]; cross=[]; single_e=[]; single_w=[]; returned=[]; delta=[]; balances=[]
    field_time=[];prop_time=[]; branches=[];nodes=[];inverse=[];incident=None
    for folder in folders:
        e=[];w=[];x=[];se=[];sw=[];r=[];d=[];bal=[];br=[];q=[];iw=[]
        for i in range(n):
            with np.load(folder/f'pair{i:06d}.npz') as data:
                e.append(data['energies_mev']); w.append(data['weights']);x.append(data['exchange_interference'])
                se.append(data['single_energies_mev']);sw.append(data['single_weights'])
                r.append(data['return_amplitude']);d.append(data['elastic_interference'])
                bal.append(float(data['all_beam_order2_balance']));br.append(data['branches']);q.append(data['q_fractional'])
                iw.append(data['inverse_marginals'])
                field_time.append(float(data['field_seconds']));prop_time.append(float(data['propagation_seconds']))
                if incident is None:incident=data['incident_coefficients'].copy()
        energy.append(e);weight.append(w);cross.append(x);single_e.append(se);single_w.append(sw)
        returned.append(np.mean(r,axis=0));delta.append(np.mean(d,axis=0));balances.extend(bal)
        branches.append(br);nodes.append(q);inverse.append(iw)
    return dict(n=n,batches=batches,energy=np.array(energy),weight=np.array(weight),cross=np.array(cross),
        single_energy=np.array(single_e),single_weight=np.array(single_w),returned=np.array(returned),
        elastic_delta=np.array(delta),incident=incident,branches=np.array(branches),nodes=np.array(nodes),inverse=np.array(inverse),
        max_order2_balance=float(np.max(abs(np.array(balances)))),field_times=np.array(field_time),prop_times=np.array(prop_time),
        contract=json.loads((folders[0]/'state.json').read_text())['contract'])


def spectra(data):
    result=[]
    for energy,weight in zip(data['energy'],data['weight']):
        result.append(np.stack([line_spectra(energy[:,k],weight[:,k]/data['n'],AXIS) for k in range(4)],axis=1))
    return np.array(result)


def raw_batches(data):
    e=data['energy'];w=data['weight'];result={}
    for name,(lo,hi) in WINDOWS.items():
        select=(e>=lo)&(e<hi)
        if name=='all_loss':select=e>0
        result[name]=np.sum(w*select[...,None],axis=(1,2))/data['n']
    result['creation_creation']=np.mean(w[:,:,0],axis=1)
    result['annihilation_annihilation']=np.mean(w[:,:,1],axis=1)
    result['mixed_all_signed']=np.sum(np.mean(w[:,:,2:],axis=1),axis=1)
    result['all_changed_bath']=np.sum(w,axis=(1,2))/data['n']
    result['exchange_interference']=np.sum(data['cross'],axis=(1,2))/data['n']
    result['elastic_order2_interference']=data['elastic_delta']
    return result


def detected_features(axis,y):
    result={}
    for name,(lo,hi) in FEATURES.items():
        select=(axis>=lo)&(axis<hi);x=axis[select];z=y[select]
        peaks,_=find_peaks(z,prominence=max(float(np.max(z))*.05,0),distance=40)
        if len(peaks):
            j=peaks[np.argmax(z[peaks])]
            result[name]=dict(status='resolved_local_maximum',peak_mev=float(x[j]),all_peaks_mev=x[peaks].tolist())
        else:
            result[name]=dict(status='NO_RESOLVED_LOCAL_MAXIMUM',peak_mev=None,all_peaks_mev=[])
    return result


def spectral_comparison(coarse,fine):
    rows=[]
    for j,name in enumerate(DETECTORS):
        a=coarse[:,j];b=fine[:,j]
        fa=detected_features(AXIS,a);fb=detected_features(AXIS,b)
        shifts={key:abs(fa[key]['peak_mev']-fb[key]['peak_mev']) if fa[key]['peak_mev'] is not None and fb[key]['peak_mev'] is not None else None for key in fa}
        windows={}
        for window in ((5,90),(5,160)):
            select=(AXIS>=window[0])&(AXIS<window[1])
            aa=trapezoid(a[select],AXIS[select]);bb=trapezoid(b[select],AXIS[select])
            shape=float(trapezoid(abs(a[select]/aa-b[select]/bb),AXIS[select])) if aa>0 and bb>0 else None
            windows[f'{window[0]}_{window[1]}']=dict(shape_L1=shape,coarse_weight=float(aa),fine_weight=float(bb),
                relative_weight_change=float(abs(bb-aa)/abs(bb)) if bb else None)
        rows.append(dict(detector=name,windows=windows,coarse_features=fa,fine_features=fb,
            identifiable_feature_shifts_mev=shifts,unresolved_features=[k for k,v in shifts.items() if v is None],
            shape_pass=all(x['shape_L1'] is not None and x['shape_L1']<.1 for x in windows.values()),
            identifiable_peak_pass=all(v<=1 for v in shifts.values() if v is not None),
            all_peak_gate_pass=all(v is not None and v<=1 for v in shifts.values())))
    return rows


def baseline():
    with np.load(CONTROLLED/'fixed_r4_T300_spectra.npz') as d:
        first=np.stack([np.interp(AXIS,d['energy_mev'],d['DW'][:,j],left=0,right=0) for j in range(4)],axis=1)
        first_sem=np.stack([np.interp(AXIS,d['energy_mev'],d['DW_standard_error'][:,j],left=0,right=0) for j in range(4)],axis=1)
    comparison=json.loads((CONTROLLED/'comparison.json').read_text())
    row=next(c for c in comparison['curves'] if c['id']=='fixed_r4_T300_DW')
    raw=[d['raw_unbroadened_window_weights'] for d in row['detectors']]
    loss=np.array([sum(r[k] for k in ('loss_0_40','loss_40_65','loss_65_90')) for r in raw])
    gain=np.array([sum(r[k] for k in ('gain_0_40','gain_40_65','gain_65_90')) for r in raw])
    # Independent source and local-Gamma uncertainty, using raw retained lines.
    delta=[]
    for i in range(8):
        folder=PREVIOUS/'source'/f'prod_T300_seed{i}'
        e=np.load(folder/'energies_mev.npy');w=np.load(folder/'weights.npy');b=np.load(folder/'born_weights.npy')
        delta.append(np.sum((w-b)*(e>0)[...,None],axis=(0,1))/len(e))
    with np.load(PREVIOUS/'Gamma_cell_T300_mesh64.npz') as d:
        ge=d['energies_mev'];gw=d['weights']
        gamma=np.sum(gw*(ge>0)[...,None],axis=(1,2))
    sem=np.sqrt(np.var(delta,axis=0,ddof=1)/8+np.var(gamma,axis=0,ddof=1)/len(gamma))
    return dict(spectrum=first,spectrum_sem=first_sem,raw_loss=loss,raw_gain=gain,raw_loss_sem=sem)


def unbiased_return_square(batch_amplitudes):
    """Independent-batch U statistic for |integral R|², without sample bias."""
    r=np.asarray(batch_amplitudes);n=len(r)
    if n<2: raise ValueError('independent batches required for unbiased quadratic estimate')
    return (abs(np.sum(r,axis=0))**2-np.sum(abs(r)**2,axis=0))/(n*(n-1))


def summarize(output,job='main',n=None,batches=8,write=True):
    data=load_job(output,job,n,batches);s=spectra(data)
    mean,sem,half=estimate(s.sum(axis=2))
    raw=raw_batches(data);summary=[];ref=baseline()
    for j,name in enumerate(DETECTORS):
        values={}
        for key,array in raw.items():
            m,se,h=estimate(array[:,j]);m=float(m);se=float(se);h=float(h)
            values[key]=dict(mean=m,standard_error=se,halfwidth95=h,
                relative_halfwidth95=h/abs(m) if abs(m)>1e-18 else None)
        denominator=ref['raw_loss'][j]
        ratio=values['all_loss']['mean']/denominator
        ratio_sem=np.sqrt((values['all_loss']['standard_error']/denominator)**2+
            (ratio*ref['raw_loss_sem'][j]/denominator)**2)
        summary.append(dict(detector=name,raw=values,
            pair_loss_over_one_phonon_loss=dict(ratio=float(ratio),standard_error=float(ratio_sem),
                halfwidth95=float(student_t.ppf(.975,min(batches,8)-1)*ratio_sem),
                denominator_raw_loss=float(denominator),denominator_sem=float(ref['raw_loss_sem'][j]),
                normalization='same incident norm and physical detector; all positive raw transitions. Denominator has Born quadrature bias outside statistical interval.'),
            features=detected_features(AXIS,mean[:,j])))
    incident=data['incident'];grid_n=data['contract']['settings']['grid_unit']
    from yoshioka.lattice_dynamics import RealSpaceForceConstants
    oriented=RealSpaceForceConstants.load(ORIENTED)
    grid=TransverseGrid(oriented.cell_m[0,0],oriented.cell_m[1,1],grid_n,grid_n)
    masks=masks_for_shift(grid,(0,0))
    return_square=unbiased_return_square(data['returned'])
    # Jackknife uncertainty for the unbiased cross-batch quadratic statistic.
    jack=np.array([masks@unbiased_return_square(np.delete(data['returned'],i,axis=0)) for i in range(batches)]) if batches>2 else None
    return_sem=np.sqrt((batches-1)/batches*np.sum((jack-np.mean(jack,axis=0))**2,axis=0)) if jack is not None else None
    total_raw=np.mean(raw['all_changed_bath'],axis=0)
    integrated=trapezoid(mean,AXIS,axis=0)
    area_error=np.abs(integrated-total_raw)/np.maximum(abs(total_raw),1e-100)
    result=dict(job=job,pairs_per_batch=data['n'],independent_batches=batches,total_pairs=data['n']*batches,
        detectors=summary,maximum_broadened_area_relative_error=float(np.max(area_error)),
        raw_signed_weight=total_raw,broadened_signed_weight=integrated,
        maximum_all_beam_order2_balance=data['max_order2_balance'],
        mean_source_field_seconds=float(np.mean(data['field_times'])),mean_propagation_seconds=float(np.mean(data['prop_times'])),
        primary_statistical_pass=bool(summary[0]['raw']['all_loss']['halfwidth95']<.1*abs(summary[0]['raw']['all_loss']['mean'])),
        primary_two_creation_statistical_pass=bool(summary[0]['raw']['creation_creation']['halfwidth95']<.1*abs(summary[0]['raw']['creation_creation']['mean'])),
        elastic_return=dict(order2_interference=np.mean(data['elastic_delta'],axis=0),
            order4_return_square_unbiased=masks@return_square,order4_return_square_jackknife_sem=return_sem,
            negative_pixel_count=int(np.sum(return_square<0)),
            scope='continuum bulk-bath return trace; independent-batch quadratic estimator can be signed at finite sampling. Not complete inclusive intensity throughg4.'),
        branch_counts=np.bincount(data['branches'].ravel(),minlength=6),
        sampled_energy_range_mev=[float(data['energy'].min()),float(data['energy'].max())],
        negatives_in_raw_pair_weights=int(np.sum(data['weight']<0)),
        uncertainty='Student-t95 across independent scrambles, not simultaneous coverage; excludes numerical/model bias.')
    if write:
        folder=output/'analysis';folder.mkdir(exist_ok=True)
        stem=f'{job}_n{data["n"]}_b{batches}'
        order0=masks@abs(incident)**2
        order2=np.mean(data['elastic_delta'],axis=0)
        order4=masks@return_square
        elastic_line=line_spectra([0.],(order0+order2+order4)[None],AXIS)
        full_order012=ref['spectrum']+mean+elastic_line
        full_raw=order0+order2+order4+ref['raw_loss']+ref['raw_gain']+total_raw
        result['full_order012_area_relative_error']=np.abs(trapezoid(full_order012,AXIS,axis=0)-full_raw)/np.maximum(abs(full_raw),1e-100)
        result['elastic_return']['order0_weight']=order0
        result['full_order012_scope']='Square of amplitude orders0+1+2 in the continuum harmonic bath, with independent-batch return-square estimator and reused one-source control variate; not every exact inclusive intensity term throughg4.'
        json_write(folder/f'{stem}.json',result)
        np.savez_compressed(folder/f'{stem}.npz',energy_mev=AXIS,pair_batch_spectra=s,pair_mean=mean,
            pair_standard_error=sem,pair_halfwidth95=half,process_mean=np.mean(s,axis=0),
            one_phonon=ref['spectrum'],one_phonon_sem=ref['spectrum_sem'],
            limited_order_changed_bath=ref['spectrum']+mean,
            full_order012=full_order012,elastic_order0_weight=order0,
            elastic_order2_weight=order2,elastic_order4_weight=order4,
            raw_pair_energies_mev=data['energy'],raw_pair_weights=data['weight'],
            raw_exchange_interference=data['cross'],raw_single_energies_mev=data['single_energy'],raw_single_weights=data['single_weight'],
            return_batch_amplitudes=data['returned'],elastic_order2_batch_interference=data['elastic_delta'],
            elastic_order4_return_square=return_square,incident_coefficients=incident,
            q_fractional=data['nodes'],branches=data['branches'],inverse_marginals=data['inverse'])
    return result,mean,s,data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--job',default='main')
    p.add_argument('--n',type=int);p.add_argument('--batches',type=int,default=8)
    a=p.parse_args();r,*_=summarize(a.output,a.job,a.n,a.batches)
    print(json.dumps({k:r[k] for k in ('job','total_pairs','mean_propagation_seconds','primary_statistical_pass','maximum_broadened_area_relative_error')},indent=2))
    print(json.dumps(r['detectors'][0],indent=2))


if __name__=='__main__':main()
