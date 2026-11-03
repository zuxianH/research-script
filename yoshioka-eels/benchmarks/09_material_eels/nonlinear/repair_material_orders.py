"""Coherent leading repeated amplitudes, with an independent dense control."""
import json,time
from pathlib import Path
import numpy as np
from scipy import constants
from scipy.linalg import expm

from yoshioka.coordinate_channel_action import CoordinateMaterialAction
from yoshioka.reciprocal_material import ReciprocalMaterialHamiltonian
from yoshioka.dyson_channel_action import DysonChannelAction
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from .repair_material_repeated import dense_quadrature_slice
from .eels_observables import absolute_spectrum
from .workflow_io import ROOT,ResourceLimits,json_write


def main():
    output=ROOT/'benchmark_results/repair_20261102_v2/B9B_orders';output.mkdir(parents=True,exist_ok=True)
    limits=ResourceLimits(600,2,100);model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
    a=CoordinateMaterialAction(model,2,3,65);source=DysonChannelAction(a,2)
    reference_h=ReciprocalMaterialHamiltonian(model,2,3,65)
    state=np.zeros(source.shape,complex);state[0,0,1,1]=1
    reference=state.reshape(3,-1).copy();dim=reference.shape[1]
    channel=np.repeat(np.arange(len(a.basis)),a.n*a.n)
    for i in range(a.steps):
        full=dense_quadrature_slice(reference_h,i)
        diagonal=full*(channel[:,None]==channel[None,:]);off=full-diagonal
        augmented=np.kron(np.eye(3),diagonal)+np.kron(np.diag(np.ones(2),-1),off)
        reference=(expm(-1j*a.dz*augmented/(constants.hbar*a.velocity))@reference.ravel()).reshape(3,dim)
        state=source.step(state,i)
    err=float(np.linalg.norm(state.reshape(3,-1)-reference))
    result={'status':'INCOMPLETE','independent_Dyson_control':{'complex_amplitude_error':err,'passed':err<1e-10,
        'reference':'explicit quadrature Hamiltonian blocks, channel-diagonal projection, dense triangular three-order exponential, identical65slices'},
        'interpretation':'Dyson order counts off-diagonal interactions. Direct nonlinear multiphonon transitions already appear in order1. Coherent order amplitudes are added before squaring; signed corrections need not be positive. Squaring orders0+1+2 does not provide every exact intensity contribution throughg4.',
        'full_target_status':'INCOMPLETE'}
    print('Dyson independent',err,flush=True)
    h=CoordinateMaterialAction(model,4,17,257);orders=DysonChannelAction(h,2).propagate(limits)
    cache=ROOT/'benchmark_results/repair_20261102_v2/B9B_coordinate'
    record=next(json.loads(p.read_text()) for p in cache.glob('*.json') if p.name not in ('checkpoint.json','results.json','invocation_1_checkpoint.json','restart_decision.json') and json.loads(p.read_text()).get('settings')=={'p':4,'n':17,'slices':257,'modes':2,'order':24})
    full=np.load(record['artifact'])['wave'];one=orders[0]+orders[1];two=np.sum(orders,axis=0)
    occupation=np.sum(h.basis,axis=1);transfer=h.channel_j
    axis=np.arange(-10,400.025,.05);spectra={};populations={}
    for name,wave in (('full',full),('one_interaction',one),('orders_through_two',two)):
        population=np.sum(abs(wave)**2,axis=(1,2));populations[name]=population
        spectra[name]=absolute_spectrum(transfer[1:],population[1:],axis)
    np.savez_compressed(output/'orders.npz',orders=orders,full=full,first_only=one,through_second=two,basis=h.basis,
        energy_grid_mev=axis,transfers_j=transfer,**{'spectrum_'+k:v for k,v in spectra.items()})
    result['settings']={'p':4,'n':17,'slices':257,'modes':2,'quadrature_order':24,'scope':'same reduced one-unit-cell transverse-Gamma Si Hamiltonian'}
    result['populations']={name:{'one_phonon':float(np.sum(p[occupation==1])),'two_or_more':float(np.sum(p[occupation>=2])),
        'represented_norm':float(np.sum(p))} for name,p in populations.items()}
    result['complex_full_minus_one_norm']=float(np.linalg.norm(full-one));result['complex_full_minus_through_two_norm']=float(np.linalg.norm(full-two))
    result['signed_full_minus_single_multiphonon']=result['populations']['full']['two_or_more']-result['populations']['one_interaction']['two_or_more']
    result['elastic_zero_order_norm_drift']=abs(float(np.sum(abs(orders[0])**2))-1)
    result['status']='PASS_REDUCED_ORDER_IMPLEMENTATION' if err<1e-10 else 'FAIL'
    result['resource_costs']=limits.costs();json_write(output/'results.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
