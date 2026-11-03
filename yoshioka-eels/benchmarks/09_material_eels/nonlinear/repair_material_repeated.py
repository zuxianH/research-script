"""Finite-z nonlinear repeated Si scattering in a declared reduced Gamma bath."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy import constants
from scipy.linalg import expm

from yoshioka.reciprocal_material import ReciprocalMaterialHamiltonian
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.harmonic_oscillator import quadrature_scattering_block
from yoshioka.free_propagation import electron_wavenumber
from .eels_observables import detector_masks,absolute_spectrum,PEAK_GROUPS
from .repeated_scattering import compare_values
from .workflow_io import ROOT,ResourceLimits,fingerprint,json_write,sha256


def dense_quadrature_slice(h,index):
    """Independent C-block quadrature and explicit reciprocal-beam shifts."""
    count=int(np.prod(h.shape));channels=len(h.basis);n=h.n
    diagonal=(h.channel_j[:,None,None]+h.kinetic_j[None]+h.scalar[index])
    matrix=np.diag(diagonal.ravel().astype(complex))
    for profile,phase,source,target in h.terms:
        block=np.ones((channels,channels),complex)
        for mode,xi in enumerate(phase.xi):
            factor=quadrature_scattering_block(float(xi),int(np.max(h.basis)),96)
            occupation=h.basis[:,mode];block*=factor[occupation[:,None],occupation[None]]
        ys,xs=np.mgrid[:n,:n];ys=ys[source];xs=xs[source]
        yt,xt=np.mgrid[:n,:n];yt=yt[target];xt=xt[target]
        for sy,sx,ty,tx in zip(ys.ravel(),xs.ravel(),yt.ravel(),xt.ravel()):
            i=np.arange(channels)*n*n+ty*n+tx;j=np.arange(channels)*n*n+sy*n+sx
            matrix[np.ix_(i,j)]+=profile[index]*block
            matrix[np.ix_(j,i)]+=profile[index].conj()*block.conj().T
    return matrix


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='benchmarks/configs/repair_material_20261102_v2.json')
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    output=ROOT/config['output'];output.mkdir(parents=True,exist_ok=True)
    limits=ResourceLimits(config['runtime_limit_seconds'],config['memory_limit_gib'],100)
    model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
    result={'status':'INCOMPLETE','full_target_status':'INCOMPLETE','cases':[],'refinements':[],
        'scope':'one oriented Si unit cell, finite z, transversely periodic unaliased electron Galerkin basis; T=0, complete declared transverse Gamma families only; all other Gamma coordinates vacuum-projected; bulk q bath omitted',
        'nonlinear':'active exp(-i Qperp.u) retained exactly; inactive vacuum DW once; active phases include their own amplitude DW; no extra full covariance factor',
        'physical_parameters':{'electron_energy_kev':60,'thickness_nm':float(model.cell_m[2,2]*1e9),'padding_angstrom':12},
        'parameters':config,'source_sha256':{f:sha256(ROOT/f) for f in ('benchmarks/repair_material_repeated.py','yoshioka/reciprocal_material.py','yoshioka/matrix_free_channels.py')}}
    # Independent nontrivial material control, identical physical/discrete H.
    h=ReciprocalMaterialHamiltonian(model,2,3,65)
    wave=np.zeros(h.shape,complex);wave[0,1,1]=1;reference=wave.ravel().copy();maxherm=0.
    started=time.monotonic()
    for i in range(h.steps):
        dense=dense_quadrature_slice(h,i);maxherm=max(maxherm,float(np.linalg.norm(dense-dense.conj().T)/np.linalg.norm(dense)))
        reference=expm(-1j*h.dz*dense/(constants.hbar*h.velocity))@reference
        wave=h.exponential_action(wave,i)
    err=float(np.linalg.norm(wave.ravel()-reference))
    result['independent_material_control']={'complex_amplitude_error':err,'hermiticity_residual':maxherm,'seconds':time.monotonic()-started,
        'p':2,'electron_beams':9,'slices':65,'passed':err<1e-10 and maxherm<1e-12,'reference':'independent Gauss-Hermite channel blocks, explicit reciprocal shift matrix, dense expm at every slice'}
    print('material independent',result['independent_material_control'],flush=True)
    json_write(output/'checkpoint.json',result)
    for settings in config['cases']:
        digest=fingerprint({'settings':settings,'source':result['source_sha256']});path=output/(digest[:20]+'.json')
        if path.exists():row=json.loads(path.read_text())
        else:
            limits.check();started=time.monotonic()
            h=ReciprocalMaterialHamiltonian(model,settings['p'],settings['n'],settings['slices'],modes=settings['modes'])
            wave,record=h.propagate(limits=limits)
            probability=abs(wave)**2;theta=np.linalg.norm(h.qxy,axis=-1)/electron_wavenumber(h.energy_j)*1000
            masks=detector_masks(theta.ravel());transfer=h.channel_j;occupation=np.sum(h.basis,axis=1)
            values={'ground_population':float(np.sum(probability[0])),'one_phonon':float(np.sum(probability[occupation==1])),
                    'two_or_more':float(np.sum(probability[occupation>=2])),'norm':float(np.sum(probability))}
            axis=np.arange(-10,250.025,.05);spectra={}
            for name,mask in masks.items():
                weights=np.sum(probability.reshape(len(h.basis),-1)[:,mask],axis=1)
                values[name+'/inelastic']=float(np.sum(weights[1:]));values[name+'/all']=float(np.sum(weights))
                for key,(lo,hi) in PEAK_GROUPS.items():
                    e=transfer/(1e-3*constants.elementary_charge);values[name+'/'+key]=float(np.sum(weights[(e>=lo)&(e<hi)]))
                spectra[name]=absolute_spectrum(transfer[1:],weights[1:],axis)
            artifact=output/(digest[:20]+'.npz')
            np.savez_compressed(artifact,wave=wave,basis=h.basis,qxy_m_inv=h.qxy,transfers_j=transfer,probability=probability,
                energy_grid_mev=axis,**{'spectrum_'+k:v for k,v in spectra.items()})
            row={'settings':settings,'artifact':str(artifact),'observables':values,'seconds':time.monotonic()-started,
                'propagation':record,'state_bytes':wave.nbytes,'mode_labels':h.mode_labels,
                'mode_energies_mev':(h.omega*constants.hbar/(1e-3*constants.elementary_charge)).tolist()}
            json_write(path,row)
        result['cases'].append(row);json_write(output/'checkpoint.json',result)
        print('material',settings,row['observables']['one_phonon'],row['observables']['two_or_more'],row['seconds'],flush=True)
    for i,a in enumerate(result['cases']):
        for b in result['cases'][i+1:]:
            changes=[k for k in a['settings'] if a['settings'][k]!=b['settings'][k]]
            if len(changes)==1:
                comparison=compare_values(a['observables'],b['observables'])
                result['refinements'].append({'changed':changes[0],'coarse':a['settings'],'fine':b['settings'],**comparison})
    result['status']='REDUCED_MATERIAL_CALCULATION_COMPLETED'
    result['representation_status']='SEE_SEPARATE_CUTOFF_ELECTRON_SLICE_COMPARISONS'
    result['resource_costs']=limits.costs();json_write(output/'results.json',result)


if __name__=='__main__':main()
