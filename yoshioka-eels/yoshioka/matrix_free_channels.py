"""Hermitian channel operators and exponential actions without pixelwise C².

Products of infinite oscillator phase-operator blocks are applied either by
tensor contraction (small mode count), or by bounded channel tiles (many modes).
The latter removes storage, not the quadratic arithmetic cost. Neither route
linearizes exp(-i Q.u). Physical occupation cutoff and working Fock D differ.
"""
from dataclasses import dataclass
from math import prod

import numpy as np
from scipy.constants import hbar
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, expm_multiply

from .harmonic_oscillator import fock_scattering_matrix
from .elastic_propagation import electron_speed
from .free_propagation import electron_wavenumber, grid_norm_squared


class ProductPhase:
    """Action of P exp[-i sum xi X] P; no exponentiation after projecting X."""
    def __init__(self, basis, xi, working_dimension=48, maximum_tensor_entries=200000, tile=64):
        self.basis=np.asarray(basis,dtype=int)
        self.xi=np.asarray(xi,dtype=float)
        if self.basis.ndim!=2 or self.xi.shape!=(self.basis.shape[1],) or working_dimension<=np.max(self.basis):
            raise ValueError('incompatible basis, xi or working dimension')
        self.extents=tuple(int(np.max(self.basis[:,i]))+1 for i in range(self.basis.shape[1]))
        self.factors=[fock_scattering_matrix(float(x),working_dimension)[:p,:p].copy() for x,p in zip(self.xi,self.extents)]
        self.tensor_entries=prod(self.extents)
        self.tensor_route=self.tensor_entries<=maximum_tensor_entries
        self.indices=tuple(self.basis.T)
        self.tile=tile
        self.diagonal=np.prod(np.array([f[n,n] for f,n in zip(self.factors,self.basis.T)]),axis=0)

    def apply(self, wave, adjoint=False):
        wave=np.asarray(wave,dtype=np.complex128)
        if wave.shape[0]!=len(self.basis): raise ValueError('channel dimension mismatch')
        factors=[a.conj().T for a in self.factors] if adjoint else self.factors
        tensor_bytes=self.tensor_entries*int(np.prod(wave.shape[1:]))*16*3
        if self.tensor_route and tensor_bytes<=256*1024**2:
            full=np.zeros((*self.extents,*wave.shape[1:]),dtype=np.complex128)
            full[self.indices]=wave
            for axis,factor in enumerate(factors):
                full=np.moveaxis(np.tensordot(factor,full,axes=(1,axis)),0,axis)
            return full[self.indices]
        output=np.zeros_like(wave)
        flat=wave.reshape(len(wave),-1)
        target=output.reshape(len(wave),-1)
        for lo in range(0,len(wave),self.tile):
            hi=min(lo+self.tile,len(wave))
            for left in range(0,len(wave),self.tile):
                right=min(left+self.tile,len(wave))
                block=np.ones((hi-lo,right-left),dtype=np.complex128)
                for mode,factor in enumerate(factors):
                    block*=factor[self.basis[lo:hi,mode,None],self.basis[None,left:right,mode]]
                target[lo:hi]+=block@flat[left:right]
        return output


class LadderOperator:
    """Sparse exact retained linear-displacement control, O(M C) edges."""
    def __init__(self,basis,couplings_j):
        basis=np.asarray(basis);lookup={tuple(v):i for i,v in enumerate(basis)}
        rows=[];columns=[];data=[]
        for j,state in enumerate(basis):
            for m,c in enumerate(couplings_j):
                final=state.copy();final[m]+=1;i=lookup.get(tuple(final))
                if i is not None and c:
                    value=c*np.sqrt(final[m]);rows.extend((i,j));columns.extend((j,i));data.extend((value,np.conj(value)))
        self.matrix=csr_matrix((data,(rows,columns)),shape=(len(basis),len(basis)),dtype=np.complex128)
        self.diagonal=np.zeros(len(basis),dtype=np.complex128)
    def apply(self,wave,adjoint=False):
        matrix=self.matrix.conj().T if adjoint else self.matrix
        return (matrix@wave.reshape(len(wave),-1)).reshape(wave.shape)


@dataclass
class PhaseTerm:
    coefficient: np.ndarray
    phase: object
    # Term = coefficient F + conjugate(coefficient) F†. The user supplies 1/2
    # for cosine potentials explicitly; it is never inserted a second time.


class LocalChannelOperator:
    def __init__(self,grid,energies_j,terms=(),scalar_j=0.,g=1.,initial_energy_j=0.):
        self.grid=grid;self.energy=np.asarray(energies_j)-initial_energy_j
        self.terms=list(terms);self.scalar=np.broadcast_to(np.asarray(scalar_j,dtype=float),grid.shape)
        self.g=float(g)
        self.shape=(len(self.energy),*grid.shape)
        self.diagonal=np.broadcast_to(self.energy[:,None,None],self.shape).copy()+self.scalar
        for term in self.terms:
            self.diagonal+=2*np.real(np.asarray(term.coefficient)[None]*term.phase.diagonal[:,None,None])

    def apply(self,wave):
        output=(self.energy[:,None,None]+self.scalar)*wave
        for term in self.terms:
            c=np.asarray(term.coefficient)[None]
            output=output+self.g*(c*term.phase.apply(wave)+c.conj()*term.phase.apply(wave,True))
            if self.g!=1:
                diagonal=2*np.real(c*term.phase.diagonal[:,None,None])
                output+=(1-self.g)*diagonal*wave
        return output

    def linear_operator(self,scale=1.):
        count=int(np.prod(self.shape))
        def action(values,factor):
            return (factor*self.apply(np.asarray(values).reshape(self.shape))).ravel()
        def matmat(values,factor):
            return np.column_stack([action(values[:,j],factor) for j in range(values.shape[1])])
        return LinearOperator((count,count),dtype=np.complex128,
            matvec=lambda v:action(v,scale),rmatvec=lambda v:action(v,np.conj(scale)),
            matmat=lambda v:matmat(v,scale),rmatmat=lambda v:matmat(v,np.conj(scale)))

    def exponential_action(self,wave,distance_m,velocity):
        scale=-1j*distance_m/(hbar*velocity)
        result=expm_multiply(self.linear_operator(scale),np.asarray(wave).ravel(),
                             traceA=scale*np.sum(self.diagonal))
        return result.reshape(self.shape)


@dataclass
class OperatorSegment:
    length_m: float
    local: object  # None denotes an exactly free segment


def propagate_matrix_free(wave,grid,energy_j,energies_j,segments,steps=32,order=4):
    """Same second/fourth-order splitting as the preserved dense reference.

    Exact exponential *actions* replace stored local eigensystems. Slice-size
    and z-dependence errors remain separate; boundaries must be explicit.
    """
    if order not in (2,4) or steps<1: raise ValueError('invalid splitting order or steps')
    current=np.asarray(wave,dtype=np.complex128).copy()
    k=electron_wavenumber(energy_j);velocity=electron_speed(energy_j)
    qx=2*np.pi*np.fft.fftfreq(grid.nx,grid.dx_m);qy=2*np.pi*np.fft.fftfreq(grid.ny,grid.dy_m)
    kinetic=(qy[:,None]**2+qx[None]**2)/(2*k)
    composition=(1.,) if order==2 else (1/(2-2**(1/3)),-2**(1/3)/(2-2**(1/3)),1/(2-2**(1/3)))
    history=[float(np.sum(grid_norm_squared(current,grid)))]
    for segment in segments:
        if segment.local is None:
            current=np.fft.ifft2(np.fft.fft2(current,axes=(-2,-1))*np.exp(-1j*segment.length_m*kinetic),axes=(-2,-1))
            current*=np.exp(-1j*segment.length_m*np.asarray(energies_j)/(hbar*velocity))[:,None,None]
        else:
            for _ in range(steps):
                for c in composition:
                    step=c*segment.length_m/steps
                    half=np.exp(-.5j*step*kinetic)
                    current=np.fft.ifft2(np.fft.fft2(current,axes=(-2,-1))*half,axes=(-2,-1))
                    current=segment.local.exponential_action(current,step,velocity)
                    current=np.fft.ifft2(np.fft.fft2(current,axes=(-2,-1))*half,axes=(-2,-1))
        history.append(float(np.sum(grid_norm_squared(current,grid))))
    return current,{'norm_history':history,'max_relative_norm_drift':float(np.max(abs(np.asarray(history)/history[0]-1))),
                    'steps_per_segment':steps,'order':order,'local_dense_pixel_matrices':False}


def representation_bytes(modes,cutoff,nx,ny,work_vectors=12,tile=64):
    from math import comb
    channels=comb(modes+cutoff,cutoff);state=16*channels*nx*ny
    return {'modes':modes,'cutoff':cutoff,'channels':channels,'bare_state_bytes':state,
            'dense_local_coupling_bytes':16*channels**2*nx*ny,
            'work_vectors':work_vectors,'iterative_working_estimate_bytes':state*work_vectors+16*tile**2,
            'excluded':'stored slices and thermal initial-state batches; finite-T trace assessed separately'}
