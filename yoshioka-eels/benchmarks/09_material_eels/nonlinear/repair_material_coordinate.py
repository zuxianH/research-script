"""Converge the same reduced Si Hamiltonian using a checked coordinate action."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy import constants

from yoshioka.coordinate_channel_action import CoordinateMaterialAction
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.free_propagation import electron_wavenumber
from .eels_observables import detector_masks,absolute_spectrum,PEAK_GROUPS
from .repeated_scattering import compare_values
from .workflow_io import ROOT,ResourceLimits,fingerprint,json_write,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='benchmarks/configs/repair_material_coordinate_20261102_v2.json')
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    output=ROOT/config['output'];output.mkdir(parents=True,exist_ok=True)
    limits=ResourceLimits(config['runtime_limit_seconds'],config['memory_limit_gib'],100)
    model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
    result={'status':'INCOMPLETE','full_target_status':'INCOMPLETE','cases':[],'refinements':[],
        'physical_scope':'same finite-z 0.383888 nm Si slab and transverse Gamma bath as B9B_material; T=0, vacuum-projected inactive Gamma modes; no bulk q convergence claimed',
        'parameters':config,'source_sha256':{f:sha256(ROOT/f) for f in ('benchmarks/repair_material_coordinate.py','yoshioka/coordinate_channel_action.py','yoshioka/reciprocal_material.py')},
        'coordinate_action_validation':'benchmark_results/repair_20261102_v2/coordinate_pilot.json',
        'original_matrix_free_material_control':'benchmark_results/repair_20261102_v2/B9B_material/results.json'}
    for settings in config['cases']:
        digest=fingerprint({'settings':settings,'sources':result['source_sha256']});path=output/(digest[:20]+'.json')
        if path.exists():row=json.loads(path.read_text())
        else:
            limits.check();started=time.monotonic()
            h=CoordinateMaterialAction(model,settings['p'],settings['n'],settings['slices'],modes=settings['modes'],quadrature_order=settings['order'],maximum_bytes=1800*1024**2)
            wave,record=h.propagate(limits)
            prob=abs(wave)**2;theta=np.linalg.norm(h.qxy,axis=-1)/electron_wavenumber(h.energy_j)*1000
            masks=detector_masks(theta.ravel());occupation=np.sum(h.basis,axis=1)
            values={'ground_population':float(np.sum(prob[0])),'one_phonon':float(np.sum(prob[occupation==1])),
                    'two_or_more':float(np.sum(prob[occupation>=2])),'norm':float(np.sum(prob))}
            spectra={};axis=np.arange(-10,600.025,.05)
            for name,mask in masks.items():
                weights=np.sum(prob.reshape(len(h.basis),-1)[:,mask],axis=1)
                values[name+'/inelastic']=float(np.sum(weights[1:]));values[name+'/all']=float(np.sum(weights))
                for key,(lo,hi) in PEAK_GROUPS.items():
                    e=h.channel_j/(1e-3*constants.elementary_charge);values[name+'/'+key]=float(np.sum(weights[(e>=lo)&(e<hi)]))
                spectra[name]=absolute_spectrum(h.channel_j[1:],weights[1:],axis)
            artifact=output/(digest[:20]+'.npz')
            np.savez_compressed(artifact,wave=wave,basis=h.basis,qxy_m_inv=h.qxy,transfers_j=h.channel_j,probability=prob,
                energy_grid_mev=axis,**{'spectrum_'+k:v for k,v in spectra.items()})
            row={'settings':settings,'artifact':str(artifact),'observables':values,'seconds':time.monotonic()-started,'propagation':record,
                'state_bytes':wave.nbytes,'mode_labels':h.mode_labels,'mode_energies_mev':(h.omega*constants.hbar/(1e-3*constants.elementary_charge)).tolist()}
            json_write(path,row)
        result['cases'].append(row);json_write(output/'checkpoint.json',result)
        print('coordinate material',settings,'one',row['observables']['one_phonon'],'multi',row['observables']['two_or_more'],'20mrad',row['observables']['full_0_20_mrad/inelastic'],'seconds',row['seconds'],flush=True)
    for i,a in enumerate(result['cases']):
        for b in result['cases'][i+1:]:
            changes=[k for k in a['settings'] if a['settings'][k]!=b['settings'][k]]
            if len(changes)==1:
                comparison=compare_values(a['observables'],b['observables'])
                result['refinements'].append({'changed':changes[0],'coarse':a['settings'],'fine':b['settings'],**comparison})
    result['status']='REDUCED_MATERIAL_CALCULATION_COMPLETED'
    result['resource_costs']=limits.costs();json_write(output/'results.json',result)


if __name__=='__main__':main()
