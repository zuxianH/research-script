"""Symmetry-preserving B9A on the same bath and periodic physical specimen."""
import argparse,json,time,mmap
from pathlib import Path
import numpy as np
from scipy import constants

from yoshioka.bloch_source import bath_groups,complex_depth_field,source_exit_bloch
from yoshioka.depth_fields import depth_provider,incident_depth
from yoshioka.source_fields_gemm import depth_field
from yoshioka.dynamical_source import full_transverse_vectors
from yoshioka.electron_phonon import common_diagonal_covariance,LobatoSilicon,full_space_born
from yoshioka.free_propagation import TransverseGrid,electron_wavenumber
from yoshioka.lattice_dynamics import RealSpaceForceConstants,finite_periodic_modes
from .eels_observables import detector_masks,PEAK_GROUPS,absolute_spectrum
from .workflow_io import ROOT,ResourceLimits,fingerprint,json_write,sha256


def calculate(model,s,output,limits,block=3):
    if s['temperature_k']!=0:raise ValueError('this new Bloch runner currently validates T=0 only')
    n=s['grid_unit'];steps=s['slices'];repeats=s['thickness_repeats'];bathz=s['phonon_z']
    finite=finite_periodic_modes(model,(2,2,bathz));cov,diag=common_diagonal_covariance(finite['displacements_m'])
    positions,groups=bath_groups(model,(2,2,bathz),repeats)
    omega=np.concatenate([g['omega_rad_s'] for g in groups]);count=len(omega)
    if count!=len(finite['omega_rad_s']):raise ValueError('Bloch bath branch count changed')
    pad=s['padding_angstrom']*1e-10;positions[:,2]+=pad;length=repeats*model.cell_m[2,2]+2*pad
    grid=TransverseGrid(model.cell_m[0,0],model.cell_m[1,1],n,n);energy=60000*constants.elementary_charge
    allowed=int((600*1024**2)//(n*n*steps*16))
    if allowed<1:raise MemoryError('one source field exceeds bounded construction workspace')
    block=min(block,allowed)
    contract={'settings':s,'covariance':cov.tolist(),'mode_energies':omega.tolist(),'positions':positions.tolist(),
        'force_constants':sha256(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz'),
        'sources':{f:sha256(ROOT/f) for f in ('benchmarks/repair_9a_bloch.py','yoshioka/bloch_source.py','yoshioka/source_fields_gemm.py','yoshioka/depth_fields.py')}}
    digest=fingerprint(contract);folder=output/'cache'/digest[:20];folder.mkdir(parents=True,exist_ok=True)
    path=folder/'state.json';state=json.loads(path.read_text()) if path.exists() else {'digest':digest,'contract':contract,'completed_modes':0,'mode_count':count,'cumulative_seconds':0.,'status':'INCOMPLETE','squared_Born_error':0.,'Born_squared_norm':0.}
    if state['digest']!=digest:raise ValueError('Bloch source checkpoint contract mismatch')
    if state['status']=='COMPLETE' and (folder/'case.json').exists():return json.loads((folder/'case.json').read_text())
    arrays={}
    for name in ('loss','born'):
        file=folder/(name+'.npy');arrays[name]=np.load(file,mmap_mode='r+') if file.exists() else np.lib.format.open_memmap(file,mode='w+',dtype=complex,shape=(count,n,n))
    started=time.monotonic();last=started;resumed=state['completed_modes']
    bgfile=folder/'background.npy'
    if state.get('background_complete'):background=np.load(bgfile,mmap_mode='r')
    else:
        background,_=depth_field(positions,cov,grid,length,steps,bgfile);state['background_complete']=True;json_write(path,state)
    if state.get('incident_complete'):incident=np.load(folder/'incident.npy',mmap_mode='r')
    else:
        exit_wave,incident,d=incident_depth(grid,energy,background,length,folder/'incident.npy',s['elastic_background'])
        np.save(folder/'incident_exit.npy',exit_wave);state.update(incident_complete=True,incident_diagnostic=d);json_write(path,state)
    bg=depth_provider(background)
    def inc(i):
        value=np.asarray(incident[i]).copy()
        if i%32==31:incident._mmap.madvise(mmap.MADV_DONTNEED)
        return value
    base_q=full_transverse_vectors(grid);k=electron_wavenumber(energy);offset=0
    try:
        for group in groups:
            w=group['omega_rad_s'];shift=group['bloch_shift_m_inv'];d=group['displacements_m']
            Q=base_q+shift;selected=np.linalg.norm(Q,axis=1)/k*1000<=20+1e-12
            lo=max(0,state['completed_modes']-offset)
            while lo<len(w):
                limits.check(estimated_bytes=2*block*n*n*steps*16+350*1024**2)
                hi=min(lo+block,len(w));file=folder/'source.npy'
                source=complex_depth_field(positions,cov,grid,length,steps,file,d[:,:,lo:hi],shift)
                amplitudes=source_exit_bloch(grid,energy,w[lo:hi],length,steps,depth_provider(source),inc,
                    (lambda i:bg(i)[0]) if s['elastic_background'] else None,shift)
                reference=full_space_born(LobatoSilicon.from_abtem_table(),positions,d[:,:,lo:hi],cov,Q[selected],w[lo:hi],energy,grid.lx_m*grid.ly_m,length)
                saved=np.zeros((hi-lo,n*n),complex);saved[:,selected]=reference
                arrays['loss'][offset+lo:offset+hi]=amplitudes;arrays['born'][offset+lo:offset+hi]=saved.reshape(hi-lo,n,n)
                if not s['elastic_background']:
                    state['squared_Born_error']+=float(np.sum(abs(amplitudes.reshape(hi-lo,-1)[:,selected]-reference)**2))
                    state['Born_squared_norm']+=float(np.sum(abs(reference)**2))
                for a in arrays.values():a.flush();a._mmap.madvise(mmap.MADV_DONTNEED)
                state['completed_modes']=offset+hi;state['cumulative_seconds']+=time.monotonic()-last;json_write(path,state);last=time.monotonic()
                del source;file.unlink();lo=hi
                if state['completed_modes']%12==0:print('Bloch progress',s,state['completed_modes'],'/',count,flush=True)
            offset+=len(w)
    except (TimeoutError,MemoryError) as exc:state['stop_reason']=str(exc)
    state['status']='COMPLETE' if state['completed_modes']==count else 'INCOMPLETE';state['latest_invocation_seconds']=time.monotonic()-started;state['resumed_from_modes']=resumed
    json_write(path,state)
    if state['status']!='COMPLETE':return {'status':'INCOMPLETE','settings':s,'checkpoint':str(path),'completed_modes':state['completed_modes']}
    weights={key:np.zeros(count) for key in (*detector_masks(np.zeros(1)),'offaxis_pixel_5_4','exact_forward_mode')}
    reference_weights={key:np.zeros(count) for key in weights};offset=0
    momentum=np.zeros((4,8,8));edges=np.linspace(-20,20,9);energies=omega*constants.hbar/(1e-3*constants.elementary_charge)
    for group in groups:
        hi=offset+len(group['omega_rad_s']);Q=base_q+group['bloch_shift_m_inv'];theta=Q/k*1000;radius=np.linalg.norm(theta,axis=1)
        masks=detector_masks(radius);masks['offaxis_pixel_5_4']=(theta[:,0]>=5)&(theta[:,0]<10)&(theta[:,1]>=0)&(theta[:,1]<5)&(radius<=20)
        masks['exact_forward_mode']=np.linalg.norm(Q,axis=1)<1e-4
        probability=abs(arrays['loss'][offset:hi].reshape(hi-offset,-1))**2;ref=abs(arrays['born'][offset:hi].reshape(hi-offset,-1))**2
        for key,mask in masks.items():weights[key][offset:hi]=np.sum(probability[:,mask],axis=1);reference_weights[key][offset:hi]=np.sum(ref[:,mask],axis=1)
        for j,(loE,hiE) in enumerate(PEAK_GROUPS.values()):
            chosen=(energies[offset:hi]>=loE)&(energies[offset:hi]<hiE);inside=radius<=20+1e-12
            momentum[j]+=np.histogram2d(theta[inside,0],theta[inside,1],bins=(edges,edges),weights=np.sum(probability[chosen][:,inside],axis=0))[0]
        offset=hi
    axis=np.arange(-120,120.025,.05);values={};spectra={}
    for name,w in weights.items():
        values[name]=float(np.sum(w));spectra[name]=absolute_spectrum(omega*constants.hbar,w,axis)
        for key,(lo,hi) in PEAK_GROUPS.items():values[name+'/'+key]=float(np.sum(w[(energies>=lo)&(energies<hi)]))
        values[name+'/gain']=0.
    artifact=folder/'summary.npz';np.savez_compressed(artifact,energy_grid_mev=axis,transfers_j=omega*constants.hbar,
        **{'spectrum_'+name:value for name,value in spectra.items()},**{'line_weights_'+name:value for name,value in weights.items()},
        momentum_window_weights=momentum,mode_q_fractional=np.concatenate([np.broadcast_to(g['q_fractional'],(len(g['omega_rad_s']),3)) for g in groups]))
    row={'status':'COMPLETE','settings':s,'mode_count':count,'observables':values,'artifact':str(artifact),'checkpoint':str(path),
        'max_incident_norm_drift':state['incident_diagnostic']['max_incident_norm_drift'],'thickness_nm':float(repeats*model.cell_m[2,2]*1e9),
        'physical_spacings_angstrom':{'dx':grid.dx_m*1e10,'dy':grid.dy_m*1e10,'dz':length/steps*1e10},
        'pure_Born_complex_error':float(np.sqrt(state['squared_Born_error']/state['Born_squared_norm'])) if state['Born_squared_norm'] else None,
        'Born_full_detector_weight':float(np.sum(reference_weights['full_0_20_mrad'])),
        'symmetry':'exact unit-cell electron Bloch sectors; no post-propagation removal of leaked weight; all93bath modes retained',
        'geometry_equivalence':'unit cell represents the same transversely periodic2x2 slab, with final Bloch shifts; phonon normalization remains1/sqrt(4*bathz)'}
    json_write(folder/'case.json',row);return row


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='benchmarks/configs/repair_9a_bloch_20261102_v2.json')
    args=parser.parse_args();config=json.loads(Path(args.config).read_text());output=ROOT/config['output'];output.mkdir(parents=True,exist_ok=True)
    model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz');limits=ResourceLimits(config['runtime_limit_seconds'],config['memory_limit_gib'],100)
    result={'status':'INCOMPLETE','parameters':config,'cases':[],'full_target_status':'INCOMPLETE'}
    for s in config['cases']:
        try:row=calculate(model,s,output,limits,config['mode_block_size'])
        except (TimeoutError,MemoryError) as exc:result['resource_stop']=str(exc);break
        result['cases'].append(row);json_write(output/'results.json',result)
        print('Bloch case',s,row['status'],row.get('observables',{}).get('full_0_20_mrad'),row.get('pure_Born_complex_error'),flush=True)
        if row['status']!='COMPLETE':break
    result['resource_costs']=limits.costs();json_write(output/'results.json',result)


if __name__=='__main__':main()
