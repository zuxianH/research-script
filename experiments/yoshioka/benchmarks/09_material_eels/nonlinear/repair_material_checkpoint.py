"""Fine reduced-material electron representation with slice-level resume."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy import constants

from yoshioka.coordinate_channel_action import CoordinateMaterialAction
from yoshioka.checkpointed_channel_action import checkpointed_action
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.free_propagation import electron_wavenumber
from .eels_observables import detector_masks,absolute_spectrum,PEAK_GROUPS
from .workflow_io import ROOT,ResourceLimits,json_write,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='benchmarks/configs/repair_material_checkpoint_20261102_v2.json')
    args=parser.parse_args();config=json.loads(Path(args.config).read_text());output=ROOT/config['output'];output.mkdir(parents=True,exist_ok=True)
    limits=ResourceLimits(config['runtime_limit_seconds'],config['memory_limit_gib'],100);s=config['settings']
    fc=ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz';model=RealSpaceForceConstants.load(fc)
    h=CoordinateMaterialAction(model,s['p'],s['n'],s['slices'],modes=s['modes'],quadrature_order=s['order'],maximum_bytes=1800*1024**2)
    contract={'settings':s,'force_constants_sha256':sha256(fc),'sources':{f:sha256(ROOT/f) for f in ('benchmarks/repair_material_checkpoint.py','yoshioka/checkpointed_channel_action.py','yoshioka/coordinate_channel_action.py','yoshioka/reciprocal_material.py')}}
    wave,state=checkpointed_action(h,output/'checkpoint',contract,limits,16)
    result={'status':state['status'],'settings':s,'checkpoint':str(output/'checkpoint/state.json'),'state':state,'full_target_status':'INCOMPLETE','resource_costs':limits.costs()}
    if state['status']=='COMPLETE':
        prob=abs(wave)**2;occupation=np.sum(h.basis,axis=1);values={'ground_population':float(np.sum(prob[0])),
            'one_phonon':float(np.sum(prob[occupation==1])),'two_or_more':float(np.sum(prob[occupation>=2])),'norm':float(np.sum(prob))}
        theta=np.linalg.norm(h.qxy,axis=-1)/electron_wavenumber(h.energy_j)*1000;spectra={};axis=np.arange(-10,600.025,.05)
        for name,mask in detector_masks(theta.ravel()).items():
            weights=np.sum(prob.reshape(len(h.basis),-1)[:,mask],axis=1);values[name+'/inelastic']=float(np.sum(weights[1:]));values[name+'/all']=float(np.sum(weights))
            E=h.channel_j/(1e-3*constants.elementary_charge)
            for key,(lo,hi) in PEAK_GROUPS.items():values[name+'/'+key]=float(np.sum(weights[(E>=lo)&(E<hi)]))
            spectra[name]=absolute_spectrum(h.channel_j[1:],weights[1:],axis)
        artifact=output/'wave_and_spectrum.npz';np.savez_compressed(artifact,wave=wave,basis=h.basis,qxy_m_inv=h.qxy,transfers_j=h.channel_j,probability=prob,energy_grid_mev=axis,**{'spectrum_'+k:v for k,v in spectra.items()})
        result.update(artifact=str(artifact),observables=values,seconds=state['cumulative_seconds'],state_bytes=wave.nbytes,
            propagation={'max_norm_drift':state['max_norm_drift'],'overlap_gram_error':h.gram_error,'quadrature_order':s['order'],'work_memory_estimate_bytes':h.estimated_work_bytes})
    json_write(output/'results.json',result);print(result,flush=True)


if __name__=='__main__':main()
