"""Bounded finite-Si physical-coupling comparison, independent of bulk sampling.

The primary bath is one explicitly selected real transverse optical Gamma
coordinate from the already verified two-coordinate family.  Its unselected
partner is returned to the inactive *vacuum* covariance.  This deliberate
anisotropic finite bath is neither the previous two-mode bath nor bulk Si.
All populations are absolute; different target states never interfere.

Run this file from the project root.  Every case has a source/input/config
identity, resumable complex vectors and 600 s / 4 GiB guards by default.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy import constants
from scipy.sparse.linalg import LinearOperator, expm_multiply

from benchmarks.workflow_io import ResourceLimits, environment, fingerprint, json_write, sha256
from yoshioka.coordinate_channel_action import CoordinateMaterialAction
from yoshioka.leading_reference import (MaterialBath, DenseDisplacedMaterial,
    gamma_material_bath, analytic_phase_blocks, displacement_taylor_blocks,
    thermal_initial_weights, normal_ordered_bath)
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.free_propagation import electron_wavenumber

DEFAULT = ROOT/'benchmark_results/14_two_phonon_eels/pair_convergence_20261102_v1/finite_validity'
MODEL = ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz'
SOURCES = ('yoshioka/leading_reference.py','yoshioka/coordinate_channel_action.py',
    'yoshioka/reciprocal_material.py','yoshioka/matrix_free_channels.py',
    'yoshioka/electron_phonon.py','yoshioka/lattice_dynamics.py',
    'yoshioka/coupled_channels.py','yoshioka/harmonic_oscillator.py',
    'benchmarks/workflow_io.py','benchmarks/14_two_phonon_eels/pair_finite_validity.py')


def single_gamma_bath(model, z_cells=1):
    """One optical transverse coordinate; repeated qz=0, fixed per-point bath.

    The one-cell full vacuum covariance is repeated (a deliberately frozen
    inactive environment, not a recalculated finite-supercell bath).  The
    active coordinate is normalized over every atom of the repeated specimen.
    """
    base=gamma_material_bath(model)
    full_cov=base.inactive_covariance_m2+np.einsum('iam,ibm->iab',base.displacement_m,base.displacement_m)
    positions=np.concatenate([base.positions_m+z*model.cell_m[2] for z in range(z_cells)])
    displacement=np.tile(base.displacement_m[:,:,:1],(z_cells,1,1))/np.sqrt(z_cells)
    inactive=np.tile(full_cov,(z_cells,1,1))-np.einsum('iam,ibm->iab',displacement,displacement)
    cell=base.cell_m.copy();cell[2]*=z_cells
    labels=[{**base.labels[0], 'selected_coordinate':0,'q_fractional':[0.,0.,0.],
             'standing_wave_family':'self-conjugate Gamma: one real coordinate',
             'inactive_environment':'repeated one-cell full vacuum covariance minus active normalized coordinate',
             'z_cells':z_cells}]
    return MaterialBath(cell,positions,displacement,base.omega_rad_s[:1],inactive,labels)


class TaylorFFT:
    """Unaliased exact Taylor Hamiltonian action, same finite electron PHP.

    This independently contracts atom Fourier derivatives, rather than
    obtaining coefficients from finite differences of the nonlinear action.
    Enlarged-Fock products supply P X_m X_n P, including boundary terms.
    """
    def __init__(self, h):
        self.h=h;self.shape=h.shape;self.n=h.n;self.pad=h.pad
        difference=np.rint(np.fft.fftfreq(h.pad)*h.pad).astype(int)
        cell=h.cell if hasattr(h,'cell') else None
        # The physical transfer vectors follow directly from the retained qxy.
        spacing=np.array([h.qxy[0,1,0]-h.qxy[0,0,0], h.qxy[1,0,1]-h.qxy[0,0,1]])
        qx,qy=np.meshgrid(spacing[0]*difference,spacing[1]*difference)
        self.Q=np.stack((qx,qy),axis=-1).reshape(-1,2)
        self.xi=np.einsum('qa,iam->iqm',self.Q,h.d[:,:2,:])
        self.blocks=[]
        for atom_xi in self.xi:
            self.blocks.append(displacement_taylor_blocks(h.basis,atom_xi)[1:])
        self.cached=None;self.fields=None
        self.kinetic=(h.channel_j[:,None,None]+h.kinetic_j[None])

    def set_slice(self,index):
        if self.cached==index:return
        c=len(self.h.basis);fourier=np.zeros((3,c,c,self.pad,self.pad),complex)
        identity=np.eye(c)
        for atom,profile in enumerate(self.h.profiles):
            value=profile[:,index]
            fourier[0]+=np.einsum('q,ab->abq',value,identity).reshape(c,c,self.pad,self.pad)
            for order,block in enumerate(self.blocks[atom],1):
                fourier[order]+=np.einsum('q,qab->abq',value,block).reshape(c,c,self.pad,self.pad)
        potential=np.fft.ifft2(fourier,axes=(-2,-1))*(self.pad*self.pad)
        residual=np.linalg.norm(potential-potential.conj().swapaxes(1,2))/max(np.linalg.norm(potential),np.finfo(float).tiny)
        if residual>2e-12:raise ValueError('Taylor real-position fields lost Hermiticity')
        self.fields=potential;self.cached=index

    def apply_order(self,wave,index,order):
        self.set_slice(index);wave=np.asarray(wave).reshape(self.shape)
        embedded=np.zeros((len(self.h.basis),self.pad,self.pad),complex)
        ix=self.h.beam_indices;embedded[:,ix[:,None],ix[None,:]]=wave
        position=np.fft.ifft2(embedded,axes=(-2,-1),norm='ortho')
        output=np.einsum('abij,bij->aij',self.fields[order],position,optimize=True)
        reciprocal=np.fft.fft2(output,axes=(-2,-1),norm='ortho')
        result=reciprocal[:,ix[:,None],ix[None,:]]
        return result+self.kinetic*wave if order==0 else result

    def augmented(self,wave,index,adjoint=False):
        x=np.asarray(wave).reshape((4,*self.shape))
        out=np.array([self.apply_order(v,index,0) for v in x])
        if adjoint:
            out[0]+=self.apply_order(x[1],index,1)+self.apply_order(x[2],index,2)
            out[1]+=self.apply_order(x[3],index,1)
        else:
            out[1]+=self.apply_order(x[0],index,1)
            out[2]+=self.apply_order(x[0],index,2)
            out[3]+=self.apply_order(x[1],index,1)
        return out

    def step(self,wave,index):
        self.set_slice(index);count=wave.size;scale=-1j*self.h.dz/(constants.hbar*self.h.velocity)
        def action(x,adjoint=False):
            return ((scale.conjugate() if adjoint else scale)*self.augmented(x,index,adjoint)).ravel()
        def batch(x,adjoint=False):return np.column_stack([action(x[:,i],adjoint) for i in range(x.shape[1])])
        op=LinearOperator((count,count),dtype=complex,matvec=action,rmatvec=lambda x:action(x,True),
            matmat=batch,rmatmat=lambda x:batch(x,True))
        trace=4*(np.sum(self.h.channel_j)*self.n**2+len(self.h.basis)*np.sum(self.h.kinetic_j)+wave[0].size*self.h.scalar[index])
        return expm_multiply(op,wave.ravel(),traceA=scale*trace).reshape(wave.shape)


def masks(h):
    angle=np.linalg.norm(h.qxy,axis=-1)/electron_wavenumber(h.energy_j)*1000
    return {'all_retained_beams':np.ones_like(angle,bool),'full_0_20_mrad':angle<=20+1e-12,
      'central_0_5_mrad':angle<5,'annulus_5_10_mrad':(angle>=5)&(angle<10),
      'annulus_10_20_mrad':(angle>=10)&(angle<=20+1e-12)}


def observables(h,full,orders,initial):
    changed=np.sum(abs(h.basis-np.asarray(initial)[None]),axis=1)==2
    rows={}
    for label,mask in masks(h).items():
        f=full[changed][:,mask];d=orders[2,changed][:,mask];s=orders[3,changed][:,mask]
        total=float(np.sum(abs(d+s)**2));actual=float(np.sum(abs(f)**2))
        rows[label]={'full':actual,'leading':total,'direct':float(np.sum(abs(d)**2)),
          'sequential':float(np.sum(abs(s)**2)),'signed_interference':float(2*np.real(np.vdot(d,s))),
          'full_over_leading_minus_one':actual/total-1 if total>1e-30 else None,
          'complex_difference_norm':float(np.linalg.norm(f-d-s)),
          'complex_relative_difference':float(np.linalg.norm(f-d-s)/np.linalg.norm(d+s)) if total>1e-30 else None,
          'component_cancellation_ratio':float((np.sum(abs(d)**2)+np.sum(abs(s)**2))/total) if total>1e-30 else None}
    return rows


def case(settings,output=DEFAULT):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    identity={'settings':settings,'sources':{s:sha256(ROOT/s) for s in SOURCES},'model_sha256':sha256(MODEL)}
    key=fingerprint(identity)[:20];path=output/(key+'.json')
    if path.exists():
        result=json.loads(path.read_text())
        if result['identity']!=identity:raise RuntimeError('identity mismatch')
        if result['status']=='COMPLETE':
            if sha256(ROOT/result['artifact'])!=result['artifact_sha256']:raise RuntimeError('amplitude hash mismatch')
            return result
    limits=ResourceLimits(settings.get('runtime_limit_seconds',600),4,100000)
    model=RealSpaceForceConstants.load(MODEL);bath=single_gamma_bath(model,settings.get('z_cells',1))
    h=CoordinateMaterialAction(model,settings['p'],settings['n'],settings['steps'],bath=bath,
      quadrature_order=settings['quadrature_order'],maximum_bytes=2*1024**3)
    t=TaylorFFT(h);limits.check()
    initial=np.asarray(settings.get('initial',[0]),dtype=int)
    where=next(i for i,state in enumerate(h.basis) if np.array_equal(state,initial))
    wave=np.zeros(h.shape,complex);wave[where,h.n//2,h.n//2]=1
    orders=np.zeros((4,*h.shape),complex);orders[0]=wave
    checkpoint=output/(key+'_checkpoint.npz');info=output/(key+'_checkpoint.json');first=0
    if checkpoint.exists():
        saved=json.loads(info.read_text())
        if saved['identity']!=identity or saved['checkpoint_sha256']!=sha256(checkpoint):raise RuntimeError('checkpoint provenance mismatch')
        raw=np.load(checkpoint);wave=raw['full'];orders=raw['orders'];first=int(raw['next_slice'])
    started=time.monotonic();stage_seconds={'full':0.,'leading':0.}
    def save(index,status):
        temporary=checkpoint.with_suffix('.tmp.npz');np.savez_compressed(temporary,full=wave,orders=orders,next_slice=index)
        temporary.replace(checkpoint)
        json_write(info,{'identity':identity,'status':status,'next_slice':index,'checkpoint_sha256':sha256(checkpoint),'cost':limits.costs()})
    try:
        for i in range(first,h.steps):
            if i%4==0:limits.check()
            mark=time.monotonic();wave=h.exponential_action(wave,i);stage_seconds['full']+=time.monotonic()-mark
            mark=time.monotonic();orders=t.step(orders,i);stage_seconds['leading']+=time.monotonic()-mark
            if (i+1)%16==0 or i+1==h.steps:save(i+1,'COMPLETE' if i+1==h.steps else 'RUNNING')
    except (TimeoutError,MemoryError) as error:
        save(i,'PARTIAL_RESOURCE_GUARD')
        result={'status':'PARTIAL_RESOURCE_GUARD','identity':identity,'next_slice':i,'error':str(error),'cost':limits.costs()}
        json_write(path,result);print(json.dumps(result),flush=True);return result
    artifact=output/(key+'.npz');np.savez_compressed(artifact,full=wave,orders=orders,basis=h.basis,
      qxy_m_inv=h.qxy,channel_j=h.channel_j,initial=initial,bath_displacements_m=bath.displacement_m,
      bath_inactive_covariance_m2=bath.inactive_covariance_m2,bath_positions_m=bath.positions_m,bath_cell_m=bath.cell_m)
    result={'status':'COMPLETE','identity':identity,'artifact':str(artifact.relative_to(ROOT)),
      'artifact_sha256':sha256(artifact),'cost':limits.costs(),'stage_seconds_this_execution':stage_seconds,
      'propagation_seconds_this_execution':time.monotonic()-started,'observables':observables(h,wave,orders,initial),
      'bath_labels':bath.labels,'mode_energies_mev':bath.omega_rad_s*constants.hbar/constants.elementary_charge*1000,
      'temperature_k':0.,'initial_thermal_tail':0.,'norm':float(np.sum(abs(wave)**2)),
      'electron_spacing_m_inv':[float(h.qxy[0,1,0]-h.qxy[0,0,0]),float(h.qxy[1,0,1]-h.qxy[0,0,1])],
      'electron_retained_component_cutoff_m_inv':np.max(abs(h.qxy),axis=(0,1)),
      'product_grid':h.pad,'product_grid_convention':'zero embedded retained beams, 2n-1 transfer grid, no cyclic beam wraparound',
      'depth_step_m':h.dz,'slab_thickness_nm':bath.cell_m[2,2]*1e9,'environment':environment(),
      'reference':'raw inactive vacuum covariance static Taylor; full nonlinear exp(-i K.u); epsilon=1, fixed phonon H',
      'observable':'sum final states with L1 occupation difference two relative to declared initial; same-state elastic-return excluded',
      'broadening':'2 meV FWHM Gaussian can be applied to raw channel energies; these one-mode vacuum lines all have identical 2hbar omega transfer'}
    json_write(path,result);print(json.dumps({'case':key,'settings':settings,'cost':result['cost'],'observables':result['observables']}),flush=True)
    return result


def validation(output=DEFAULT):
    """Independent actual-material operator and thermal-transition checks."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model=RealSpaceForceConstants.load(MODEL);bath=single_gamma_bath(model)
    h=CoordinateMaterialAction(model,4,3,17,bath=bath,quadrature_order=18)
    t=TaylorFFT(h);dense=DenseDisplacedMaterial(bath,4,3,17)
    rng=np.random.default_rng(61683);test=rng.normal(size=h.shape)+1j*rng.normal(size=h.shape);test/=np.linalg.norm(test)
    derivatives=dense.derivative_hamiltonians();errors=[]
    for order in range(3):
        for i in (0,8,16):
            errors.append({'order':order,'slice':i,'relative_action_difference':float(np.linalg.norm(t.apply_order(test,i,order).ravel()-derivatives[order,i]@test.ravel())/np.linalg.norm(derivatives[order,i]@test.ravel()))})
    vacuum=np.zeros(h.shape,complex);vacuum[0,1,1]=1;orders=np.zeros((4,*h.shape),complex);orders[0]=vacuum
    full=vacuum.copy()
    for i in range(h.steps):full=h.exponential_action(full,i);orders=t.step(orders,i)
    dfull=dense.propagate();dorders=dense.leading()
    propagation={'full_complex_absolute_error':float(np.linalg.norm(full.ravel()-dfull)),
      'all_four_leading_complex_absolute_error':float(np.linalg.norm(orders.reshape(4,-1)-dorders)),
      'leading_P2_complex_absolute_error':float(np.linalg.norm((orders[2]+orders[3]).ravel()-(dorders[2]+dorders[3])))}
    adjoint_test=rng.normal(size=(4,*h.shape))+1j*rng.normal(size=(4,*h.shape))
    other=rng.normal(size=adjoint_test.shape)+1j*rng.normal(size=adjoint_test.shape)
    lhs=np.vdot(adjoint_test,t.augmented(other,8));rhs=np.vdot(t.augmented(adjoint_test,8,True),other)
    adjoint_residual=float(abs(lhs-rhs)/max(abs(lhs),abs(rhs)))
    # The actual Si propagation has non-vacuum initial states.  This specifically
    # catches substituting a vacuum P2 projector for a finite-T transition class.
    initial=np.zeros(dense.dimension,complex);where=np.flatnonzero(dense.basis[:,0]==1)[0];initial[where*9+4]=1
    thermal_full=dense.propagate(initial=initial);thermal_orders=dense.leading(initial=initial)
    thermal_obs=observables(h,thermal_full.reshape(h.shape),thermal_orders.reshape((4,*h.shape)),[1])
    thermal_selected=(abs(dense.basis[:,0]-1)==2);vacuum_selected=(dense.basis[:,0]==2)
    thermal_wrong=float(np.sum(abs(thermal_full.reshape(h.shape)[vacuum_selected])**2))
    norm=float(np.linalg.norm(thermal_full.reshape(h.shape)[thermal_selected]))
    normal,variance=normal_ordered_bath(bath,300)
    normal_dense=DenseDisplacedMaterial(normal,4,3,17,normal_order_variances=variance)
    normal_difference=float(np.linalg.norm(normal_dense.propagate()-dfull))
    data={'status':'PASS' if max(v['relative_action_difference'] for v in errors)<2e-11 and max(propagation.values())<2e-11 and adjoint_residual<2e-12 and normal_difference<2e-11 else 'FAIL',
      'derivative_actions':errors,'propagation':propagation,'augmented_adjoint_residual':adjoint_residual,
      'thermal_initial_n1_actual_material':{'observables':thermal_obs,'selected_final_occupations':dense.basis[thermal_selected],
         'wrong_vacuum_projector_probability':thermal_wrong,'correct_changed_two_probability':norm**2,
         'temperature_note':'non-vacuum material-operator regression, not a converged Gibbs trace'},
      'normal_ordered_same_H_epsilon1_complex_difference':normal_difference,
      'reference_family_note':'Counterterm reconstructs same exact H at epsilon1; raw Taylor and full-DW perturbation families/leading amplitudes differ. No counterterm omitted.'}
    json_write(output/'operator_validation.json',data);print((output/'operator_validation.json').read_text(),flush=True);return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--validate',action='store_true');args=parser.parse_args()
    if args.validate:validation(args.output)
    elif args.config:case(json.loads(args.config.read_text()),args.output)
    else:parser.error('--config or --validate required')


if __name__=='__main__':main()
