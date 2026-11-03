"""Identical-physics dense/structured B3B and B8 validation and cost pilot."""
import json
from pathlib import Path
import time

import numpy as np
from scipy import constants
from scipy.special import gammaln

from yoshioka.coupled_channels import channel_basis,SplitChannelPropagator
from yoshioka.matrix_free_channels import (ProductPhase,PhaseTerm,LadderOperator,LocalChannelOperator,
    OperatorSegment,propagate_matrix_free,representation_bytes)
from yoshioka.free_propagation import TransverseGrid
from yoshioka.elastic_propagation import electron_speed
from .repeated_scattering import model
from .channel_reference import dense_generators,dense_propagate
from .workflow_io import ROOT,json_write,ResourceLimits


def structured_b8(config,p,n,length,linear=False,g=1.):
    basis=channel_basis(2,p);grid=TransverseGrid(*[v*1e-9 for v in config['transverse_lengths_nm']],n,n)
    quanta=np.asarray(config['mode_quanta_mev'])*1e-3*constants.elementary_charge
    xx,yy=np.meshgrid(2*np.pi*np.arange(n)/n,2*np.pi*np.arange(n)/n)
    regions=[]
    for harmonic,offset,xi,ev in zip(((1,0),(1,1)),(0.,.7),config['displacement_couplings_xi'],config['region_potential_ev']):
        angle=harmonic[0]*xx+harmonic[1]*yy+offset
        if linear:
            phase=LadderOperator(basis,xi)
            local=LocalChannelOperator(grid,basis@quanta,[PhaseTerm(.5*ev*constants.elementary_charge*np.sin(angle),phase)],
                scalar_j=ev*constants.elementary_charge*np.cos(angle),g=g)
        else:
            phase=ProductPhase(basis,xi,48)
            local=LocalChannelOperator(grid,basis@quanta,[PhaseTerm(.5*ev*constants.elementary_charge*np.exp(1j*angle),phase)],g=g)
        regions.append(local)
    segments=[OperatorSegment(frac*length*1e-9,field) for frac,field in zip((.125,.125,.375,.25,.125),(None,regions[0],None,regions[1],None))]
    initial=np.zeros((len(basis),n,n),dtype=np.complex128);initial[0]=1/np.sqrt(grid.lx_m*grid.ly_m)
    return basis,grid,quanta,segments,initial


def main():
    output=ROOT/'benchmark_results/repair_20261102_v2/B9B_validation';output.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'benchmarks/configs/repeated_20261102_v1.json').read_text())
    energy=60000*constants.elementary_charge;limits=ResourceLimits(1800,2,100)
    results={'status':'INCOMPLETE','B8':[],'memory':[representation_bytes(m,p,65,65) for m in (93,189) for p in (1,2,3)],
             'full_material_status':'INCOMPLETE','nonlinear_method':'projected product oscillator phases, exact tensor contractions or bounded tiles; not linearized'}
    grid=TransverseGrid(1e-9,1e-9,2,2);basis=channel_basis(1,24)
    quantum,force=np.array([60.,47.])*1e-3*constants.elementary_charge
    theta=1.3;length=theta*constants.hbar*electron_speed(energy)/quantum
    initial=np.zeros((len(basis),2,2),complex);initial[0]=1e9
    local=LocalChannelOperator(grid,basis[:,0]*quantum,[PhaseTerm(.5,LadderOperator(basis,[force]))])
    wave,record=propagate_matrix_free(initial,grid,energy,basis[:,0]*quantum,[OperatorSegment(length,local)],4)
    lam=force/quantum;alpha=lam*(np.exp(-1j*theta)-1);n=basis[:,0]
    exact=np.exp(1j*lam**2*(theta-np.sin(theta))-.5*abs(alpha)**2)*alpha**n*np.exp(-.5*gammaln(n+1))
    error=float(np.linalg.norm(wave[:,0,0]*1e-9-exact))
    results['B3B']={'complex_amplitude_error':error,'passed':error<1e-10,**record}
    print('matrix-free 3B',error,flush=True)
    for linear in (False,True):
        p,n,L=3,4,80
        basis,grid,quanta,old,initial=model(config,p,n,L,linear)
        _,_,_,quadrature,_=model(config,p,n,L,linear,quadrature_order=96)
        reference=dense_propagate(initial,dense_generators(grid,energy,basis@quanta,quadrature))
        _,_,_,segments,_=structured_b8(config,p,n,L,linear)
        for steps in (8,16,32,64):
            started=time.monotonic()
            wave,record=propagate_matrix_free(initial,grid,energy,basis@quanta,segments,steps)
            dense_split,_=SplitChannelPropagator(grid,energy,basis@quanta).propagate(initial,old,steps)
            err=float(np.linalg.norm(wave-reference)/np.linalg.norm(reference))
            agreement=float(np.linalg.norm(wave-dense_split)/np.linalg.norm(dense_split))
            row={'linear':linear,'p':p,'grid':n,'steps':steps,'complex_error_vs_independent_dense':err,
                 'complex_error_vs_identical_dense_split':agreement,'seconds':time.monotonic()-started,**record}
            results['B8'].append(row);print('matrix-free B8',linear,steps,err,agreement,flush=True)
            json_write(output/'checkpoint.json',results)
    # Same strong production B8 system as the protected reference, not just a
    # small toy. Its cached wave is read only; no passing result is regenerated.
    source=json.loads((ROOT/'benchmark_results/repeated_20261102_v1/results.json').read_text())
    old=next(c for c in source['cases'] if c['settings']=={'p':12,'n':16,'steps':64,'thickness_nm':80,'g':2,'linear':False})
    basis,grid,quanta,segments,initial=structured_b8(config,12,16,80,False,2)
    started=time.monotonic();wave,record=propagate_matrix_free(initial,grid,energy,basis@quanta,segments,64)
    with np.load(old['wave_artifact']) as data: reference=data['wave']
    err=float(np.linalg.norm(wave-reference)/np.linalg.norm(reference))
    results['B8_production']={'settings':old['settings'],'complex_amplitude_error':err,'seconds':time.monotonic()-started,**record,'passed':err<1e-10}
    np.savez_compressed(output/'production_wave.npz',wave=wave)
    results['status']='PASS' if results['B3B']['passed'] and results['B8_production']['passed'] and all(r['complex_error_vs_independent_dense']<1e-6 and r['complex_error_vs_identical_dense_split']<1e-10 and r['max_relative_norm_drift']<1e-9 for r in results['B8'] if r['steps']==64) else 'FAIL'
    results['resource_costs']=limits.costs();json_write(output/'results.json',results)
    print('matrix-free validation',results['status'],results['B8_production'],flush=True)


if __name__=='__main__':main()
