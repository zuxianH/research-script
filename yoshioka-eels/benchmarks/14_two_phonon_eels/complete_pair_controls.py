"""Paired electron/slice sensitivity of the new complete material amplitude."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import t as student_t
from .workflow_io import json_write,sha256
from .complete_leading_eels import DEFAULT,setup,physical_setup,nodes,KERNELS,finite_json
from .approximate_eels import PRIMITIVE,DETECTORS
from .complete_leading_guard import audit_inputs
from .two_phonon_analysis import AXIS,spectral_comparison
from .born_quadrature import line_spectra
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.leading_two_phonon import propagate_bulk_leading_pair


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT/'bulk_numerical_controls');p.add_argument('--fine',action='store_true');p.add_argument('--larger',action='store_true');a=p.parse_args();output=a.output;output.mkdir(parents=True,exist_ok=True)
    criteria={'intensity_relative_max':.01,'shape_L1_max':.1,'feature_shift_mev_max':1.,'declared_before_calculation':True,
        'scope':'Eight matched full-support node pairs. Sensitivity control; not a rigorous global BZ discretization bound.'}
    json_write(output/'criteria.json',criteria);model=RealSpaceForceConstants.load(PRIMITIVE)
    cases=[('base',49,385),('electron',65,385),('slices',49,769),('joint',65,769)]
    if a.fine or a.larger:cases += [('fine81',81,769),('fine97',97,769)]
    if a.larger:cases += [('fast125',125,769)]
    results={};cost=[]
    for name,n,steps in cases:
        folder=output/name;setup(folder);config,settings,data=physical_setup(folder,n,steps)
        positions,reference,length,cov,grid,bg,inc,_=data
        physical={'kernels':{n:sha256(Path(__file__).resolve().parents[2]/n) for n in KERNELS},
            'inputs':audit_inputs(folder),'config':config,'settings':settings}
        sidecar=folder/'control_contract.json'
        if sidecar.exists():
            committed=json.loads(sidecar.read_text())
            if committed['physical']!=physical:raise ValueError('Changed numerical-control physical/source contract; use a new version')
        else:
            old=list(folder.glob('control_pair*.npz'))
            origin='Creation-time contract before new calculations'
            if old:
                anchor=output/'results_fine.json';original=json.loads(anchor.read_text())
                if original['kernels']!=physical['kernels']:raise ValueError('Old control kernel provenance mismatch')
                for raw in old:
                    if original['raw_file_hashes'][str(raw.relative_to(output))]!=sha256(raw):raise ValueError('Old control array changed')
                origin='Retrospective current audit of preserved completed record '+sha256(anchor)+'; not a backdated creation record'
            committed={'physical':physical,'raw_files':{p.name:sha256(p) for p in old},'origin':origin}
            json_write(sidecar,committed)
        weights=[];energy=[];direct=[];seq=[];interf=[];pair_times=[];reused=0
        started=time.monotonic()
        for b in range(8):
            q,branch,iw=nodes(folder,'born',0,config['seed_start']+b)
            path=folder/f'control_pair{b}.npz'
            if path.exists():
                if committed['raw_files'].get(path.name)!=sha256(path):raise ValueError('Changed numerical-control array')
                reused+=1
            if not path.exists():
                r=propagate_bulk_leading_pair(model,positions,reference,cov,grid,length,steps,q[0],branch[0],iw[0],bg,
                    folder/'scratch',300,save_amplitudes=(b==0),reference_kind='static')
                r.update(q_fractional=q[0],branches=branch[0],inverse_marginals=iw[0]);np.savez_compressed(path,**r)
                committed['raw_files'][path.name]=sha256(path);json_write(sidecar,committed)
            with np.load(path) as d:
                energy.append(d['energies_mev']);weights.append(d['weights']);direct.append(d['direct_weights']);seq.append(d['sequential_weights']);interf.append(d['interference_weights']);pair_times.append(float(d['elapsed_seconds']))
        e=np.array(energy);w=np.array(weights);loss=np.sum(w*(e>0)[...,None],axis=1)
        spectrum=sum(line_spectra(ee,ww/8,AXIS) for ee,ww in zip(e,w))
        results[name]={'energy':e,'weight':w,'loss_batches':loss,'mean_spectrum':spectrum,'settings':settings}
        np.savez_compressed(folder/'controls.npz',energy=e,weight=w,direct=direct,sequential=seq,interference=interf,loss_batches=loss,energy_mev=AXIS,mean_spectrum=spectrum)
        cost.append({'case':name,'seconds':time.monotonic()-started,'grid':n,'steps':steps,'reused_pairs':reused,'sum_measured_pair_seconds':sum(pair_times)})
        bg._mmap.close();inc._mmap.close()
    comparison=[];base=results['base']
    for name,_,_ in cases[1:]:
        r=results[name];delta=r['loss_batches']-base['loss_batches'];mean=delta.mean(axis=0);half=student_t.ppf(.975,7)*delta.std(axis=0,ddof=1)/np.sqrt(8)
        relative=abs(mean)/abs(r['loss_batches'].mean(axis=0));spec=spectral_comparison(base['mean_spectrum'],r['mean_spectrum'])
        comparison.append({'case':name,'paired_loss_change':mean,'paired_difference_halfwidth95':half,'relative_mean_loss_change':relative,
            'spectral_sensitivity':spec,'intensity_gate':bool(np.all(relative<.01)),
            'shape_gate':all(s['shape_pass'] for s in spec),'feature_gate':all(s['all_peak_gate_pass'] for s in spec)})
    adjacent=[]
    if a.fine or a.larger:
        pairs=[('joint','fine81'),('fine81','fine97')]
        if a.larger:pairs.append(('fine97','fast125'))
        for old,new in pairs:
            coarse,fine=results[old],results[new]
            delta=fine['loss_batches']-coarse['loss_batches']
            rel=abs(delta.mean(axis=0))/abs(fine['loss_batches'].mean(axis=0))
            spec=spectral_comparison(coarse['mean_spectrum'],fine['mean_spectrum'])
            adjacent.append({'from':old,'to':new,'relative_loss_change':rel,
                'paired_loss_change':delta.mean(axis=0),'paired_change_halfwidth95':student_t.ppf(.975,7)*delta.std(axis=0,ddof=1)/np.sqrt(8),
                'spectral_sensitivity':spec,'intensity_gate':bool(np.all(rel<.01)),
                'shape_gate':all(z['shape_pass'] for z in spec),'feature_gate':all(z['all_peak_gate_pass'] for z in spec)})
    record={'adjacent_fine_grids':adjacent,'raw_file_hashes':{str(p.relative_to(output)):sha256(p) for p in output.glob('*/control_pair*.npz')},'criteria':criteria,'comparisons':comparison,'cost':cost,'kernels':{n:sha256(Path(__file__).resolve().parents[2]/n) for n in KERNELS},
        'source_sha256':sha256(__file__),'status':'SAMPLED_SENSITIVITY_ONLY','full_target':'INCOMPLETE'}
    json_write(output/('results_larger.json' if a.larger else 'results_fine_reaudited.json' if a.fine else 'results_reaudited.json'),finite_json(record));print([(r['case'],r['relative_mean_loss_change'],r['intensity_gate'],r['shape_gate'],r['feature_gate']) for r in comparison],flush=True)

if __name__=='__main__':main()
