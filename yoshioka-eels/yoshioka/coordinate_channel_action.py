"""Nonlinear coordinate-quadrature action with an unaliased electron FFT.

This is a factorization of P H P, not a linear-displacement approximation.
Gauss-Hermite overlaps map retained Fock states to oscillator coordinates;
the atom potential is scalar there. No C² local channel matrix is stored.
The default Gamma bath or an explicit real transverse finite non-Gamma bath
uses the same action; longitudinal active displacements are rejected.
The quadrature order is an independent convergence parameter.
"""
from math import prod
import numpy as np
from scipy import constants
from scipy.special import roots_hermite
from scipy.sparse.linalg import LinearOperator,expm_multiply

from .harmonic_oscillator import normalized_hermite_polynomials
from .coupled_channels import channel_basis
from .reciprocal_material import resolve_material_bath
from .electron_phonon import LobatoSilicon
from .elastic_propagation import electron_speed
from .free_propagation import electron_wavenumber


class CoordinateMaterialAction:
    def __init__(self,model,p,n,steps,modes=2,quadrature_order=24,padding_m=12e-10,maximum_bytes=1500*1024**2,bath=None,displacement_scale=1.):
        if n%2!=1 or steps%2!=1:raise ValueError('odd electron grid and z sampling required')
        if not np.isfinite(displacement_scale):raise ValueError('finite displacement scale required')
        self.displacement_scale=float(displacement_scale)
        self.d,self.omega,self.inactive_covariance,self.mode_labels,cell,positions,self.bath_scope=resolve_material_bath(model,modes,bath)
        modes=len(self.omega)
        self.basis=channel_basis(modes,p);self.n=n;self.steps=steps;self.pad=2*n-1
        nodes_count=quadrature_order**modes
        # Profiles were previously estimated without the actual atom count.
        # The list plus stacked array coexist; Fourier/Lobato work is separate.
        profile_bytes=16*len(positions)*self.pad**2*steps
        construction_bytes=16*self.pad**2*steps*12
        estimate=(2*profile_bytes+construction_bytes+16*nodes_count*self.pad**2*5
                  +16*nodes_count*len(self.basis)*2
                  +32*len(positions)*nodes_count*self.pad)
        if estimate>maximum_bytes:raise MemoryError('coordinate tensor work-space preflight; use bounded channel tiles instead')
        self.estimated_work_bytes=estimate
        self.energy_j=60000*constants.elementary_charge;self.velocity=electron_speed(self.energy_j)
        self.length=cell[2,2]+2*padding_m;self.dz=self.length/steps
        self.channel_j=self.basis@(constants.hbar*self.omega)
        r=n//2;beam=np.arange(-r,r+1);self.beam_indices=beam%self.pad
        qx,qy=np.meshgrid(2*np.pi*beam/cell[0,0],2*np.pi*beam/cell[1,1])
        self.qxy=np.stack((qx,qy),axis=-1)
        self.kinetic_j=constants.hbar*self.velocity*(qx*qx+qy*qy)/(2*electron_wavenumber(self.energy_j))
        self.shape=(len(self.basis),n,n)
        nodes,weights=roots_hermite(quadrature_order)
        indices=np.array(list(np.ndindex(*(quadrature_order,)*modes)))
        coordinates=nodes[indices];sqrt_weight=np.sqrt(np.prod(weights[indices]/np.sqrt(np.pi),axis=1))
        hermites=normalized_hermite_polynomials(nodes,p)
        self.overlap=np.broadcast_to(sqrt_weight[:,None],(nodes_count,len(self.basis))).copy()
        for m in range(modes):self.overlap*=hermites[self.basis[None,:,m],indices[:,None,m]]
        self.gram_error=float(np.linalg.norm(self.overlap.T@self.overlap-np.eye(len(self.basis))))
        self.quadrature_order=quadrature_order
        potential=LobatoSilicon.from_abtem_table();area=cell[0,0]*cell[1,1]
        kz=2*np.pi*np.fft.fftfreq(steps,self.dz)
        difference=np.rint(np.fft.fftfreq(self.pad)*self.pad).astype(int)
        qx,qy=np.meshgrid(2*np.pi*difference/cell[0,0],2*np.pi*difference/cell[1,1])
        Q=np.column_stack((qx.ravel(),qy.ravel()))
        profiles=[];phase_x=[];phase_y=[]
        for atom,pos in enumerate(positions):
            cov=self.inactive_covariance[atom]
            exponent=np.einsum('qa,ab,qb->q',Q,cov[:2,:2],Q)[:,None]+2*(Q@cov[:2,2])[:,None]*kz+cov[2,2]*kz*kz
            kernel=potential.fourier_energy_j_m3(np.sum(Q*Q,axis=1)[:,None]+kz*kz)*np.exp(-.5*exponent)
            fourier=kernel*np.exp(-1j*(pos[2]+padding_m-self.dz/2)*kz)/(area*self.dz)
            profile=np.fft.ifft(fourier,axis=-1)*np.exp(-1j*Q@pos[:2])[:,None]
            profiles.append(profile)
            shifts=self.displacement_scale*np.sqrt(2)*coordinates@self.d[atom,:2,:].T
            phase_x.append(np.exp(-1j*shifts[:,0,None]*(2*np.pi*difference/cell[0,0])))
            phase_y.append(np.exp(-1j*shifts[:,1,None]*(2*np.pi*difference/cell[1,1])))
        self.profiles=np.array(profiles);self.phase_x=np.array(phase_x);self.phase_y=np.array(phase_y)
        self.scalar=np.sum(self.profiles[:,0].real,axis=0)
        self.cached_index=None;self.potential=None

    def set_slice(self,index):
        if self.cached_index==index:return
        fourier=np.zeros((len(self.overlap),self.pad,self.pad),complex)
        for atom in range(len(self.profiles)):
            fourier+=self.profiles[atom,:,index].reshape(self.pad,self.pad)[None]*self.phase_y[atom,:,:,None]*self.phase_x[atom,:,None,:]
        potential=np.fft.ifft2(fourier,axes=(-2,-1))*(self.pad*self.pad)
        residual=np.linalg.norm(potential.imag)/max(np.linalg.norm(potential.real),np.finfo(float).tiny)
        if residual>2e-12:raise ValueError('coordinate potential lost Hermiticity')
        self.potential=potential.real.reshape(len(self.overlap),-1).copy()
        self.cached_index=index

    def apply(self,wave,index):
        self.set_slice(index);wave=np.asarray(wave,dtype=complex)
        output=(self.channel_j[:,None,None]+self.kinetic_j[None])*wave
        embedded=np.zeros((len(self.basis),self.pad,self.pad),complex)
        ix=self.beam_indices;embedded[:,ix[:,None],ix[None,:]]=wave
        position=np.fft.ifft2(embedded,axes=(-2,-1),norm='ortho').reshape(len(self.basis),-1)
        coordinates=self.overlap@position
        projected=self.overlap.T@(self.potential*coordinates)
        reciprocal=np.fft.fft2(projected.reshape(embedded.shape),axes=(-2,-1),norm='ortho')
        return output+reciprocal[:,ix[:,None],ix[None,:]]

    def exponential_action(self,wave,index):
        self.set_slice(index)
        scale=-1j*self.dz/(constants.hbar*self.velocity);count=wave.size
        def action(x,s):return (s*self.apply(np.asarray(x).reshape(self.shape),index)).ravel()
        def batch(x,s):return np.column_stack([action(x[:,i],s) for i in range(x.shape[1])])
        operator=LinearOperator((count,count),dtype=complex,matvec=lambda x:action(x,scale),rmatvec=lambda x:action(x,scale.conjugate()),
            matmat=lambda x:batch(x,scale),rmatmat=lambda x:batch(x,scale.conjugate()))
        trace=np.sum(self.channel_j)*self.n**2+len(self.basis)*np.sum(self.kinetic_j)+count*self.scalar[index]
        return expm_multiply(operator,wave.ravel(),traceA=scale*trace).reshape(self.shape)

    def propagate(self,limits=None):
        wave=np.zeros(self.shape,complex);wave[0,self.n//2,self.n//2]=1;drift=0.
        for i in range(self.steps):
            if limits is not None and i%8==0:limits.check()
            wave=self.exponential_action(wave,i)
            drift=max(drift,abs(float(np.sum(abs(wave)**2))-1))
        return wave,{'max_norm_drift':drift,'bath_scope':self.bath_scope,'overlap_gram_error':self.gram_error,'quadrature_order':self.quadrature_order,
                     'work_memory_estimate_bytes':self.estimated_work_bytes,'slice_rule':'full H exponential at midpoint, independently refine z'}
