"""Reduced real transverse Si baths in an unaliased reciprocal electron basis.

Finite-z Lobato fields use the same independent Fourier construction as B9A.
The default activates selected in-plane Gamma modes. An optional explicit
finite supercell bath activates real transverse standing-wave modes, including
non-Gamma modes. Omitted coordinates contribute a declared inactive covariance
once. Neither finite bath constitutes full macroscopic silicon.
"""
import numpy as np
from scipy import constants
from scipy.sparse.linalg import LinearOperator,expm_multiply

from .matrix_free_channels import ProductPhase
from .coupled_channels import channel_basis
from .electron_phonon import LobatoSilicon
from .elastic_propagation import electron_speed
from .free_propagation import electron_wavenumber


def transverse_gamma_bath(model,count):
    omega,v=model.modes(np.zeros(3))
    groups=([9,10,11],[5,6],[3,4],[7,8])
    vectors=[];frequencies=[];labels=[]
    for group in groups:
        block=v[:,group].reshape(len(model.positions_m),3,len(group))
        z=block[:,2,:]
        values,rotation=np.linalg.eigh(z.conj().T@z)
        rotated=(v[:,group]@rotation).reshape(len(model.positions_m),3,len(group))
        for i in np.flatnonzero(values<1e-10):
            vector=rotated[:,:,i]
            if np.linalg.norm(vector.imag)>1e-10:raise ValueError('Gamma gauge is not real')
            vectors.append(vector.real);frequencies.append(float(np.mean(omega[group])))
            labels.append({'original_degenerate_group':list(group),'longitudinal_weight':float(values[i])})
    if count not in (2,4,6):raise ValueError('retain complete declared transverse families: 2,4,6 modes')
    vectors=np.moveaxis(np.array(vectors[:count]),0,-1)
    frequencies=np.array(frequencies[:count])
    displacement=vectors*np.sqrt(constants.hbar/(2*model.masses_kg[:,None,None]*frequencies[None,None]))
    full=v[:,3:].reshape(len(model.positions_m),3,-1)*np.sqrt(constants.hbar/(2*model.masses_kg[:,None,None]*omega[None,None,3:]))
    inactive=np.einsum('iam,ibm->iab',full,full.conj()).real-np.einsum('iam,ibm->iab',displacement,displacement)
    if np.min(np.linalg.eigvalsh(inactive)) < -1e-12*np.max(abs(inactive)):
        raise ValueError('inactive covariance is not positive semidefinite')
    if np.linalg.norm(displacement[:,2])>1e-10*np.linalg.norm(displacement):raise ValueError('active modes must be transverse')
    return displacement,frequencies,inactive,labels[:count]


def resolve_material_bath(model,modes=2,bath=None):
    """Original Gamma default or an explicit real transverse finite bath."""
    if bath is None:
        displacement,omega,covariance,labels=transverse_gamma_bath(model,modes)
        cell=np.asarray(model.cell_m);positions=np.asarray(model.positions_m)
        scope='original transverse Gamma bath'
    else:
        # A duck-typed declared finite bath avoids a circular dependency on
        # the independent dense validation module. Non-Gamma standing-wave
        # modes require the matching transverse supercell in bath.cell_m.
        raw_displacement=np.asarray(bath.displacement_m)
        if np.linalg.norm(raw_displacement.imag)>1e-12*max(np.linalg.norm(raw_displacement.real),np.finfo(float).tiny):
            raise ValueError('optional material bath requires real standing-wave displacements')
        displacement=np.asarray(raw_displacement.real,dtype=float)
        omega=np.asarray(bath.omega_rad_s,dtype=float)
        covariance=np.asarray(bath.inactive_covariance_m2,dtype=float)
        labels=list(bath.labels)
        cell=np.asarray(bath.cell_m,dtype=float);positions=np.asarray(bath.positions_m,dtype=float)
        if displacement.shape!=(len(positions),3,len(omega)) or covariance.shape!=(len(positions),3,3):
            raise ValueError('optional material bath has inconsistent atom/mode/covariance dimensions')
        if np.any(omega<=0) or not all(np.all(np.isfinite(a)) for a in (displacement,omega,covariance,cell,positions)):
            raise ValueError('optional material bath needs positive frequencies and finite displacements')
        if np.linalg.norm(displacement[:,2])>1e-10*np.linalg.norm(displacement):
            raise ValueError('matrix-free optional material bath supports transverse displacements only; longitudinal phases need a different action')
        if np.linalg.norm(cell-np.diag(np.diag(cell)))>1e-12*np.linalg.norm(cell):
            raise ValueError('matrix-free material action requires an orthorhombic supercell')
        if np.min(np.linalg.eigvalsh(covariance)) < -1e-12*np.max(abs(covariance)):
            raise ValueError('optional inactive covariance is not positive semidefinite')
        modes=len(omega)
        scope='explicit finite real transverse bath; no bulk integration inferred'
    return displacement,omega,covariance,labels,cell,positions,scope


class ReciprocalMaterialHamiltonian:
    """Action of the finite electron Galerkin Hamiltonian; no cyclic aliasing.

    Q_perp = k_out-k_in is an electron transfer. The default bath is Gamma;
    an explicit finite real standing-wave bath may contain non-Gamma modes.
    Active displacements must be transverse in this factorization.
    Terms shift reciprocal beams with zero outside the retained electron basis.
    Nonlinear products F(Q_perp.u) are never replaced by displacement derivatives.
    """
    def __init__(self,model,p,n,steps,padding_m=12e-10,modes=2,working_dimension=48,bath=None):
        if n%2!=1 or steps%2!=1:raise ValueError('odd electron grids and slice counts required')
        self.n=n;self.steps=steps
        self.d,self.omega,self.inactive_covariance,self.mode_labels,cell,positions,self.bath_scope=resolve_material_bath(model,modes,bath)
        modes=len(self.omega)
        self.cell_m=cell.copy();self.positions_m=positions.copy()
        self.basis=channel_basis(modes,p)
        self.energy_j=60000*constants.elementary_charge;self.velocity=electron_speed(self.energy_j)
        self.length=cell[2,2]+2*padding_m;self.dz=self.length/steps
        self.channel_j=self.basis@(constants.hbar*self.omega)
        r=n//2;beam=np.arange(-r,r+1)
        qx,qy=np.meshgrid(2*np.pi*beam/cell[0,0],2*np.pi*beam/cell[1,1])
        self.qxy=np.stack((qx,qy),axis=-1)
        self.kinetic_j=constants.hbar*self.velocity*(qx*qx+qy*qy)/(2*electron_wavenumber(self.energy_j))
        self.shape=(len(self.basis),n,n);self.terms=[];self.scalar=np.zeros(steps)
        kz=2*np.pi*np.fft.fftfreq(steps,self.dz);area=cell[0,0]*cell[1,1]
        potential=LobatoSilicon.from_abtem_table()
        positions=positions.copy();positions[:,2]+=padding_m
        for dy in range(n):
            for dx in range(-(n-1),n):
                if dy==0 and dx<0:continue
                Q=np.array([2*np.pi*dx/cell[0,0],2*np.pi*dy/cell[1,1]])
                squared=np.sum(Q*Q)+kz*kz
                for atom,pos in enumerate(positions):
                    cov=self.inactive_covariance[atom]
                    exponent=Q@cov[:2,:2]@Q+2*(Q@cov[:2,2])*kz+cov[2,2]*kz*kz
                    fourier=potential.fourier_energy_j_m3(squared)*np.exp(-.5*exponent)*np.exp(-1j*(pos[2]-self.dz/2)*kz)/(area*self.dz)
                    profile=np.fft.ifft(fourier)*np.exp(-1j*Q@pos[:2])
                    if dx==0 and dy==0:
                        if np.linalg.norm(profile.imag)>1e-12*np.linalg.norm(profile.real):raise ValueError('nonreal scalar field')
                        self.scalar+=profile.real
                    else:
                        phase=ProductPhase(self.basis,Q@self.d[atom,:2,:],working_dimension)
                        source=(slice(0,n-dy),slice(max(0,-dx),min(n,n-dx)))
                        target=(slice(dy,n),slice(max(0,dx),min(n,n+dx)))
                        self.terms.append((profile,phase,source,target))
        self.slice=0

    def apply(self,wave,index=None):
        index=self.slice if index is None else index
        wave=np.asarray(wave,dtype=np.complex128)
        output=(self.channel_j[:,None,None]+self.kinetic_j[None]+self.scalar[index])*wave
        for profile,phase,source,target in self.terms:
            output[(slice(None),*target)]+=profile[index]*phase.apply(wave[(slice(None),*source)])
            output[(slice(None),*source)]+=profile[index].conj()*phase.apply(wave[(slice(None),*target)],True)
        return output

    def exponential_action(self,wave,index,fraction=1.):
        scale=-1j*fraction*self.dz/(constants.hbar*self.velocity)
        count=wave.size
        def action(x,factor):return (factor*self.apply(np.asarray(x).reshape(self.shape),index)).ravel()
        def batch(x,factor):return np.column_stack([action(x[:,i],factor) for i in range(x.shape[1])])
        operator=LinearOperator((count,count),dtype=np.complex128,matvec=lambda x:action(x,scale),rmatvec=lambda x:action(x,scale.conjugate()),
            matmat=lambda x:batch(x,scale),rmatmat=lambda x:batch(x,scale.conjugate()))
        trace=np.sum(self.channel_j)*self.n*self.n+len(self.basis)*np.sum(self.kinetic_j)+count*self.scalar[index]
        return expm_multiply(operator,wave.ravel(),traceA=scale*trace).reshape(self.shape)

    def propagate(self,initial=None,limits=None):
        wave=np.zeros(self.shape,complex) if initial is None else np.asarray(initial).copy()
        if initial is None:wave[0,self.n//2,self.n//2]=1
        maximum=0.
        for i in range(self.steps):
            if limits is not None and i%8==0:limits.check()
            wave=self.exponential_action(wave,i)
            maximum=max(maximum,abs(float(np.sum(abs(wave)**2))-1))
        return wave,{'max_norm_drift':maximum,'bath_scope':self.bath_scope,'slice_rule':'midpoint piecewise-constant full-H exponential action; z refinement remains necessary'}


def projected_coupling_residual(operator,target_wave,retained):
    """(1-P)H Phi_P for a declared fixed Hermitian target representation.

    Duhamel's identity and unitary target evolution imply error <= integral
    ||residual|| dz/(hbar v), for matching initial states and exact PHP
    evolution. A time quadrature estimate is not itself a rigorous integral
    bound. An enlarged *finite* target is not the infinite oscillator space.
    Numerical propagation errors and changes to P require separate accounting.
    """
    wave=np.asarray(target_wave)
    if np.any(wave[~np.asarray(retained,dtype=bool)]):raise ValueError('state must lie in fixed retained projector')
    residual=operator(wave).copy();residual[retained]=0
    return residual
