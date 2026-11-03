"""Depth-major revision of the independent-channel Si refinement.

No C^2 coupling arrays and no final-channel history in z. The single incident
history and Fourier potential/one source batch are disk checkpoints. Memory is
O(batch*Nxy), apart from bounded Fourier construction buffers and mapped pages.
"""
import argparse
import json
import mmap
from pathlib import Path
import time

import numpy as np
from scipy import constants

from yoshioka.dynamical_source import plane_field, propagate_incident, full_transverse_vectors
from yoshioka.source_streaming import streamed_source_exit, mapped_slice_provider
from yoshioka.free_propagation import TransverseGrid, electron_wavenumber
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from .dynamical_si_eels import source_specimen
from yoshioka.depth_fields_batched import convert_depth_field,depth_field,depth_provider,incident_depth
from .eels_observables import DETECTORS, PEAK_GROUPS, absolute_spectrum, detector_masks
from .workflow_io import ROOT, ResourceLimits, fingerprint, json_write, sha256


def calculate_streamed_case(model, settings, output, limits, batch=1):
    n, steps, repeats, bath, temperature = [settings[k] for k in ('grid','slices','thickness_repeats','phonon_z','temperature_k')]
    positions, d, omega, nbar, covariance, covdiag = source_specimen(model, repeats, bath, temperature)
    padding = settings['padding_angstrom']*1e-10
    positions[:, 2] += padding
    length = repeats*model.cell_m[2, 2]+2*padding
    grid = TransverseGrid(2*model.cell_m[0, 0], 2*model.cell_m[1, 1], n, n)
    energy = 60000*constants.elementary_charge
    contract = {'settings': settings, 'physics': fingerprint([positions.tolist(),d.tolist(),omega.tolist(),covariance.tolist()]),
                'sources': {f:sha256(ROOT/f) for f in ('benchmarks/repair_9a_batched.py','yoshioka/source_streaming.py','yoshioka/dynamical_source.py','yoshioka/depth_fields.py','yoshioka/depth_fields_batched.py')},
                'force_constants': sha256(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')}
    digest = fingerprint(contract)
    folder = output/'cache'/digest[:20]
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder/'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'contract':contract,'digest':digest,
                'completed_modes':0,'mode_count':len(omega),'cumulative_seconds':0.,'status':'INCOMPLETE'}
    if state['digest'] != digest:
        raise ValueError('stream checkpoint contract mismatch')
    if state['status'] == 'COMPLETE':
        return json.loads((folder/'case.json').read_text())
    started = time.monotonic()
    resumed = state['completed_modes']
    limits.check(estimated_bytes=400*1024**2+batch*n*n*16*20)
    arrays = {}
    for name in ('loss','gain'):
        path = folder/(name+'.npy')
        arrays[name] = np.load(path,mmap_mode='r+') if path.exists() else np.lib.format.open_memmap(path,mode='w+',dtype=np.complex128,shape=(len(omega),n,n))
    background_path = folder/'background_depth.npy'
    # Adopt only an identical physical prefix, with explicit old-code evidence.
    initial_output=ROOT/'benchmark_results/repair_20261102_v2/B9A_depth'
    if not state.get('adoption_checked'):
        for prior in initial_output.glob('cache/*/state.json'):
            previous=json.loads(prior.read_text())
            if previous['contract']['settings']!=settings or previous['contract']['physics']!=contract['physics']:continue
            count=previous['completed_modes']
            if count:
                for name in arrays:
                    old=np.load(prior.parent/(name+'.npy'),mmap_mode='r')
                    arrays[name][:count]=old[:count];arrays[name].flush()
                state['completed_modes']=count;resumed=count
                state['adopted_prefix']={'checkpoint':str(prior),'sha256':sha256(prior),'modes':count,'source_contract':previous['contract'],'cumulative_old_seconds':previous['cumulative_seconds']}
            state['adopted_incident']=str(prior.parent/'incident.npy')
            state['incident_diagnostic']=previous['incident_diagnostic']
            state['adopted_depth_background']=str(prior.parent/'background_depth.npy')
            state['adoption_validation']='test_depth_construction_batch_equivalence; identical physical fingerprint; only chunk_size64 to512 changes'
            break
        state['adoption_checked']=True;json_write(state_path,state)
    if state.get('adopted_depth_background'):
        background=np.load(state['adopted_depth_background'],mmap_mode='r')
    elif state.get('background_complete'):
        background = np.load(background_path,mmap_mode='r')
    else:
        if state.get('adopted_fourier_background'):
            previous=np.load(state['adopted_fourier_background'],mmap_mode='r')
            background,_=convert_depth_field(previous,grid,background_path);del previous
        else:
            background,_=depth_field(positions,covariance,grid,length,steps,background_path)
        state['background_complete'] = True
        json_write(state_path,state)
    if state.get('adopted_incident'):
        incident=np.load(state['adopted_incident'],mmap_mode='r')
    elif state.get('incident_complete'):
        incident = np.load(folder/'incident.npy',mmap_mode='r')
    else:
        exit_wave,incident,diagnostic = incident_depth(grid,energy,background,length,folder/'incident.npy',settings['elastic_background'])
        np.save(folder/'incident_exit.npy',exit_wave)
        state.update(incident_complete=True,incident_diagnostic=diagnostic)
        json_write(state_path,state)
    bg_at = depth_provider(background)
    def incident_at(index):
        value = np.asarray(incident[index]).copy()
        if index % 16 == 15:
            incident._mmap.madvise(mmap.MADV_DONTNEED)
        return value
    committed = time.monotonic()
    try:
        for lo in range(resumed,len(omega),batch):
            limits.check()
            hi = min(lo+batch,len(omega))
            path = folder/'working_source_depth.npy'
            source,_ = depth_field(positions,covariance,grid,length,steps,path,d[:,:,lo:hi])
            source_at = depth_provider(source)
            for sign,name,weight in ((1,'loss',nbar[lo:hi]+1),(-1,'gain',nbar[lo:hi])):
                if np.any(weight):
                    amplitudes = streamed_source_exit(grid,energy,sign*omega[lo:hi],length,steps,
                        source_at,incident_at,(lambda i:bg_at(i)[0]) if settings['elastic_background'] else None)
                    arrays[name][lo:hi] = amplitudes*np.sqrt(weight)[:,None,None]
                else:
                    arrays[name][lo:hi] = 0
                arrays[name].flush()
                arrays[name]._mmap.madvise(mmap.MADV_DONTNEED)
            del source_at,source
            path.unlink()
            Path(str(path)+'.reality.json').unlink(missing_ok=True)
            state['completed_modes'] = hi
            state['cumulative_seconds'] += time.monotonic()-committed
            json_write(state_path,state)
            committed = time.monotonic()
            if hi % 12 == 0:
                print('9A progress',settings,'modes',hi,'/',len(omega),flush=True)
    except (TimeoutError,MemoryError) as exc:
        state['stop_reason'] = str(exc)
    state['latest_invocation_seconds'] = time.monotonic()-started
    state['resumed_from_modes'] = resumed
    state['status'] = 'COMPLETE' if state['completed_modes']==len(omega) else 'INCOMPLETE'
    state['covariance_diagnostic'] = covdiag
    json_write(state_path,state)
    if state['status'] != 'COMPLETE':
        return {'status':'INCOMPLETE','settings':settings,'checkpoint':str(state_path),'completed_modes':state['completed_modes']}
    qxy = full_transverse_vectors(grid)
    theta = qxy/electron_wavenumber(energy)*1000
    masks = detector_masks(np.linalg.norm(theta,axis=1))
    masks['offaxis_pixel_5_4'] = (theta[:,0]>=5)&(theta[:,0]<10)&(theta[:,1]>=0)&(theta[:,1]<5)&(np.linalg.norm(theta,axis=1)<=20)
    masks['exact_forward_mode'] = np.linalg.norm(qxy,axis=1)==0
    transfer = np.r_[omega,-omega]*constants.hbar
    e_mev = transfer/(1e-3*constants.elementary_charge)
    detector_weights = {name:np.zeros(2*len(omega)) for name in masks}
    for sign,name in enumerate(('loss','gain')):
        for lo in range(0,len(omega),batch):
            hi=min(lo+batch,len(omega))
            prob=abs(arrays[name][lo:hi].reshape(hi-lo,-1))**2
            for key,mask in masks.items():
                detector_weights[key][sign*len(omega)+lo:sign*len(omega)+hi]=np.sum(prob[:,mask],axis=1)
            arrays[name]._mmap.madvise(mmap.MADV_DONTNEED)
    axis=np.arange(-120,120.025,.05)
    spectra={name:absolute_spectrum(transfer,w,axis) for name,w in detector_weights.items()}
    values={name:float(np.sum(w)) for name,w in detector_weights.items()}
    for name,w in detector_weights.items():
        for key,(lo,hi) in PEAK_GROUPS.items():
            values[name+'/'+key]=float(np.sum(w[(e_mev>=lo)&(e_mev<hi)]))
        values[name+'/gain']=float(np.sum(w[transfer<0]))
    artifact=folder/'summary.npz'
    np.savez_compressed(artifact,energy_grid_mev=axis,transfers_j=transfer,omega_rad_s=omega,qxy_m_inv=qxy,
        **{'spectrum_'+k:v for k,v in spectra.items()},**{'line_weights_'+k:v for k,v in detector_weights.items()})
    row={'status':'COMPLETE','settings':settings,'mode_count':len(omega),'observables':values,'artifact':str(artifact),
         'checkpoint':str(state_path),'thickness_nm':float(repeats*model.cell_m[2,2]*1e9),
         'max_incident_norm_drift':state['incident_diagnostic']['max_incident_norm_drift'],
         'streaming':'one outgoing batch, one shared incoming checkpoint; coherent depth sum then final-channel intensity trace',
         'physical_spacings_angstrom':{'dx':grid.dx_m*1e10,'dy':grid.dy_m*1e10,'dz':length/steps*1e10},
         'live_outgoing_state_bytes':batch*n*n*16,'source_disk_working_bytes':batch*n*n*steps*16}
    json_write(folder/'case.json',row)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='benchmarks/configs/repair_9a_batched_20261102_v2.json')
    args=parser.parse_args(); config=json.loads(Path(args.config).read_text())
    output=ROOT/config['output']; output.mkdir(parents=True,exist_ok=True)
    model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
    limits=ResourceLimits(config['runtime_limit_seconds'],config['memory_limit_gib'],100)
    results={'status':'INCOMPLETE','parameters':config,'cases':[],'full_material_convergence':'INCOMPLETE'}
    for s in config['cases']:
        try:
            row=calculate_streamed_case(model,s,output,limits,config['batch_size'])
        except (TimeoutError,MemoryError) as exc:
            results['resource_stop']=str(exc);break
        results['cases'].append(row)
        json_write(output/'results.json',results)
        print('9A case',s,row['status'],row.get('observables',{}).get('annulus_5_10_mrad/acoustic_0_40_mev'),flush=True)
        if row['status']!='COMPLETE':break
    results['resource_costs']=limits.costs();json_write(output/'results.json',results)


if __name__=='__main__': main()
