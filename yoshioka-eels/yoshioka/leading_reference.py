"""Independent finite-Si displacement-order and full nonlinear reference.

This deliberately dense, bounded reference complements the matrix-free bulk
workflow. SI units; row cell vectors; angular electron momenta; channel-major
reciprocal-beam arrays. The phonon Hamiltonian NEVER scales with epsilon.

The atom Fourier phase is exp[-i K.(R+epsilon*u)]. Taylor coefficients are
therefore -i K.u and -1/2 (K.u)^2. The quadratic product is formed in an enlarged
occupation basis before projection, never by squaring a truncated coordinate.
Inactive modes enter one Gaussian covariance only. Optional normal ordering
adds +1/2 <(K.u)^2>_T to V2 and its exact exponential counterpart, explicitly
changing the epsilon-scaling convention; it does not change phonon energies.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import constants
from scipy.linalg import eigh
from scipy.special import eval_genlaguerre, gammaln
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import expm_multiply

from .coupled_channels import channel_basis, linear_displacement_block
from .electron_phonon import LobatoSilicon
from .elastic_propagation import electron_speed
from .free_propagation import electron_wavenumber
from .harmonic_oscillator import quadrature_scattering_block, fock_scattering_matrix
from .lattice_dynamics import finite_periodic_modes
from .reciprocal_material import transverse_gamma_bath


def analytic_phase_blocks(basis, xi):
    """Infinite-oscillator exp(-i xi X), exact retained blocks, batched xi.

    Closed Laguerre elements provide an independent route from ProductPhase's
    working-Fock exponentials and the Gauss-Hermite cross-check.
    """
    basis=np.asarray(basis,dtype=int);xi=np.atleast_2d(xi)
    if xi.shape[1] != basis.shape[1]: raise ValueError('mode count mismatch')
    result=np.ones((len(xi),len(basis),len(basis)),complex)
    for mode in range(basis.shape[1]):
        a=basis[:,mode,None];b=basis[None,:,mode]
        lo=np.minimum(a,b);delta=abs(a-b)
        x=xi[:,mode,None,None]
        result*=np.exp(-x*x/2+0.5*(gammaln(lo+1)-gammaln(lo+delta+1)))*(-1j*x)**delta*eval_genlaguerre(lo,delta,x*x)
    return result


@lru_cache(maxsize=4096)
def _numerical_single_phase(x,maximum,work,method):
    return (quadrature_scattering_block(float(x),maximum,work) if method=='quadrature'
            else fock_scattering_matrix(float(x),work)[:maximum+1,:maximum+1])


def numerical_phase_blocks(basis,xi,work,method='quadrature'):
    basis=np.asarray(basis);xi=np.atleast_2d(xi)
    maximum=int(np.max(basis))
    if work<=maximum: raise ValueError('working size must exceed largest retained occupation')
    result=np.ones((len(xi),len(basis),len(basis)),complex)
    for mode in range(basis.shape[1]):
        n=basis[:,mode]
        values,inverse=np.unique(xi[:,mode],return_inverse=True)
        matrices=np.array([_numerical_single_phase(float(x),maximum,int(work),method)[n[:,None],n[None,:]] for x in values])
        result*=matrices[inverse]
    return result


def quadrature_phase_blocks(basis,xi,order):
    return numerical_phase_blocks(basis,xi,order,'quadrature')


@lru_cache(maxsize=16)
def _coordinate_products(states):
    basis=np.array(states,dtype=int);modes=basis.shape[1]
    enlarged=channel_basis(modes,int(np.max(np.sum(basis,axis=1)))+2)
    lookup={tuple(v):i for i,v in enumerate(enlarged)}
    retained=np.array([lookup[tuple(v)] for v in basis])
    coordinates=[linear_displacement_block(enlarged,np.eye(modes)[m]) for m in range(modes)]
    linear=np.array([x[np.ix_(retained,retained)] for x in coordinates])
    quadratic=np.array([[(a[retained]@b[:,retained]) for b in coordinates] for a in coordinates])
    return linear,quadratic


def displacement_taylor_blocks(basis,xi,normal_order_variances=None):
    """P1P,-i PXP,-1/2 PX²P, with an independently enlarged product space."""
    basis=np.asarray(basis);xi=np.atleast_2d(xi)
    linear,quadratic=_coordinate_products(tuple(map(tuple,basis)))
    identity=np.eye(len(basis))
    first=-1j*np.einsum('km,mab->kab',xi,linear)
    second=-.5*np.einsum('km,kn,mnab->kab',xi,xi,quadratic,optimize=True)
    if normal_order_variances is not None:
        second+=.5*((xi*xi)@normal_order_variances)[:,None,None]*identity[None]
    return np.broadcast_to(identity,(len(xi),len(basis),len(basis))).copy(),first,second


@dataclass
class MaterialBath:
    cell_m: np.ndarray
    positions_m: np.ndarray
    displacement_m: np.ndarray
    omega_rad_s: np.ndarray
    inactive_covariance_m2: np.ndarray
    labels: list


def gamma_material_bath(model,count=2):
    d,omega,cov,labels=transverse_gamma_bath(model,count)
    return MaterialBath(model.cell_m.copy(),model.positions_m.copy(),d,omega,cov,labels)


def finite_material_bath(model,repetitions=(2,1,1),active_indices=(0,1)):
    """Real supercell normal coordinates, including non-Gamma/longitudinal modes.

    All omitted supercell modes are explicitly vacuum projected; selecting a
    few active modes is a model restriction, not convergence of bulk silicon.
    """
    finite=finite_periodic_modes(model,repetitions)
    indices=np.asarray(active_indices,dtype=int)
    if len(np.unique(indices)) != len(indices): raise ValueError('duplicate active index')
    full=finite['displacements_m'];active=full[:,:,indices]
    covariance=np.einsum('iam,ibm->iab',full,full)-np.einsum('iam,ibm->iab',active,active)
    labels=[]
    cell_count=int(np.prod(repetitions));nb=len(model.positions_m)
    for index in indices:
        vector=finite['eigenvectors'][:,:,index].reshape(cell_count,nb,3)
        mean=np.mean(vector,axis=0,keepdims=True)
        labels.append({'finite_mode_index':int(index),'non_gamma_weight':float(np.linalg.norm(vector-mean)**2),
                       'longitudinal_weight':float(np.sum(vector[:,:,2]**2)),
                       'energy_mev':float(finite['omega_rad_s'][index]*constants.hbar/(1e-3*constants.elementary_charge))})
    return MaterialBath(np.asarray(repetitions)[:,None]*model.cell_m,finite['positions_m'],active,
                        finite['omega_rad_s'][indices],covariance,labels)


def repeated_material_bath(model,cells_z=2,active_indices=(12,13)):
    """Repeated finite Si slab; same transverse q family, in-phase qz=0 modes.

    Active coordinates are normalized over all specimen atoms. The omitted
    finite-periodic bath and its vacuum covariance are recomputed at each
    geometry. Thus this is a declared finite-crystal thickness study, not a
    fixed bulk bath convergence test or a surface-phonon model.
    """
    if not isinstance(cells_z,int) or cells_z<1: raise ValueError('positive integer cells_z required')
    base=finite_material_bath(model,(2,1,1),active_indices)
    finite=finite_periodic_modes(model,(2,1,cells_z));nb=len(model.positions_m)
    active=np.concatenate([base.displacement_m[x*nb:(x+1)*nb]/np.sqrt(cells_z)
                           for x,y,z in np.ndindex(2,1,cells_z)],axis=0)
    omega=base.omega_rad_s
    eigen=active*np.sqrt(2*finite['masses_kg'][:,None,None]*omega[None,None]/constants.hbar)
    eigen=eigen.reshape(-1,len(omega));vectors=finite['eigenvectors'].reshape(len(eigen),-1)
    target=eigen*omega[None]**2
    residual=np.linalg.norm(vectors@(finite['omega_rad_s'][:,None]**2*(vectors.T@eigen))-target)/np.linalg.norm(target)
    if residual>1e-10: raise ValueError('repeated qz=0 active family is not an eigenbasis')
    full=finite['displacements_m']
    covariance=np.einsum('iam,ibm->iab',full,full)-np.einsum('iam,ibm->iab',active,active)
    labels=[{**label,'z_cells':cells_z,'qz_fractional':0.,'mechanical_eigen_residual':float(residual)} for label in base.labels]
    return MaterialBath(np.array([2,1,cells_z])[:,None]*model.cell_m,finite['positions_m'],active,omega,covariance,labels)


def normal_ordered_bath(bath,temperature_k=0.):
    """Use full Gaussian DW H0 with active thermal normal-order counterterms.

    At epsilon=1 this represents the same exact potential as the supplied
    inactive-only bath; epsilon != 1 is a distinct scaling convention.
    """
    omega=bath.omega_rad_s
    variance=np.ones_like(omega) if temperature_k==0 else 1+2/np.expm1(constants.hbar*omega/(constants.Boltzmann*temperature_k))
    active=np.einsum('iam,ibm,m->iab',bath.displacement_m,bath.displacement_m,variance)
    return MaterialBath(bath.cell_m,bath.positions_m,bath.displacement_m,omega,bath.inactive_covariance_m2+active,bath.labels),variance


class DenseDisplacedMaterial:
    """Bounded independent full-3D displaced atomic potential in finite beams.

    The z Fourier reconstruction and midpoint piecewise-constant propagator
    match the existing reciprocal material reference. Unlike that transverse
    Gamma reference, longitudinal displacements and commensurate non-Gamma
    standing waves are retained. Every beam transfer is Galerkin projected;
    there is no electron FFT wraparound and no frozen propagation step.
    """
    def __init__(self,bath,p=2,n=3,steps=65,padding_m=12e-10,
                 normal_order_variances=None,phonon_evolution=True,maximum_bytes=1024**3):
        if n%2!=1 or steps%2!=1: raise ValueError('odd n and z steps required')
        self.bath=bath;self.basis=channel_basis(len(bath.omega_rad_s),p)
        self.n=n;self.steps=steps;self.shape=(len(self.basis),n,n)
        self.dimension=int(np.prod(self.shape));self.p=p
        self.estimated_hamiltonian_bytes=3*steps*self.dimension**2*16
        if self.estimated_hamiltonian_bytes>maximum_bytes: raise MemoryError('dense reference preflight')
        self.energy_j=60000*constants.elementary_charge;self.velocity=electron_speed(self.energy_j)
        self.hv=constants.hbar*self.velocity
        self.length=bath.cell_m[2,2]+2*padding_m;self.dz=self.length/steps
        self.channel_j=self.basis@(constants.hbar*bath.omega_rad_s)
        self.normal_order_variances=None if normal_order_variances is None else np.asarray(normal_order_variances,dtype=float)
        if self.normal_order_variances is not None and self.normal_order_variances.shape!=(self.basis.shape[1],):
            raise ValueError('one variance per active oscillator required')
        beam=np.arange(-(n//2),n//2+1)
        qx,qy=np.meshgrid(2*np.pi*beam/bath.cell_m[0,0],2*np.pi*beam/bath.cell_m[1,1])
        self.qxy=np.stack((qx,qy),axis=-1)
        kinetic=self.hv*(qx*qx+qy*qy)/(2*electron_wavenumber(self.energy_j))
        energies=self.channel_j if phonon_evolution else np.zeros_like(self.channel_j)
        self.free_diagonal=(energies[:,None,None]+kinetic[None]).ravel()
        kz=2*np.pi*np.fft.fftfreq(steps,self.dz)
        potential=LobatoSilicon.from_abtem_table();area=bath.cell_m[0,0]*bath.cell_m[1,1]
        self.terms=[]
        for dy in range(-(n-1),n):
            for dx in range(-(n-1),n):
                qxy=np.array([2*np.pi*dx/bath.cell_m[0,0],2*np.pi*dy/bath.cell_m[1,1]])
                wavevector=np.column_stack((np.broadcast_to(qxy,(steps,2)),kz))
                ys,xs=np.mgrid[:n,:n];valid=(ys+dy>=0)&(ys+dy<n)&(xs+dx>=0)&(xs+dx<n)
                source=(ys[valid]*n+xs[valid]);target=(ys[valid]+dy)*n+xs[valid]+dx
                for atom,pos in enumerate(bath.positions_m):
                    covariance=bath.inactive_covariance_m2[atom]
                    exponent=np.einsum('ka,ab,kb->k',wavevector,covariance,wavevector)
                    coefficient=potential.fourier_energy_j_m3(np.sum(wavevector**2,axis=1))*np.exp(-.5*exponent)
                    coefficient=coefficient*np.exp(-1j*(kz*(pos[2]+padding_m-self.dz/2)+qxy@pos[:2]))/(area*self.dz)
                    xi=wavevector@bath.displacement_m[atom]
                    self.terms.append((coefficient,xi,source,target))
        self._derivatives=None

    def _assemble(self,kind,epsilon=1.,method='analytic',quadrature_order=24,working_dimension=48):
        count=3 if kind=='derivatives' else 1
        output=np.zeros((count,self.steps,self.dimension,self.dimension),complex)
        index=np.arange(self.dimension)
        output[0,:,index,index]=self.free_diagonal[:,None]
        c=len(self.basis);offset=np.arange(c)*self.n**2
        for coefficient,xi,source,target in self.terms:
            if kind=='derivatives':
                blocks=displacement_taylor_blocks(self.basis,xi,self.normal_order_variances)
            else:
                phase=(analytic_phase_blocks(self.basis,epsilon*xi) if method=='analytic' else
                       numerical_phase_blocks(self.basis,epsilon*xi,quadrature_order if method=='quadrature' else working_dimension,method))
                if self.normal_order_variances is not None:
                    phase*=np.exp(.5*epsilon**2*((xi*xi)@self.normal_order_variances))[:,None,None]
                blocks=(phase,)
            for order,block in enumerate(blocks):
                profiles=np.fft.ifft(coefficient[:,None,None]*block,axis=0)
                for src,dst in zip(source,target):
                    output[order,:,offset[:,None]+dst,offset[None,:]+src]+=profiles.transpose(1,2,0)
        # Check one slice at a time: whole-array Hermitian differences would
        # triple peak memory in exactly the reference sizes this guards.
        norm2=0.;residual2=0.
        for order in output:
            for matrix in order:
                difference=matrix-matrix.conj().T
                norm2+=float(np.vdot(matrix,matrix).real)
                residual2+=float(np.vdot(difference,difference).real)
        residual=np.sqrt(residual2/max(norm2,np.finfo(float).tiny))
        if residual>5e-12: raise ValueError(f'non-Hermitian material Fourier field {residual}')
        for order in output:
            for matrix in order: matrix[:]=(matrix+matrix.conj().T)/2
        return output

    def derivative_hamiltonians(self):
        if self._derivatives is None: self._derivatives=self._assemble('derivatives')
        return self._derivatives

    def hamiltonians(self,epsilon=1.,method='analytic',quadrature_order=24,working_dimension=48):
        if method not in ('analytic','quadrature','fock'): raise ValueError('unknown matrix element method')
        return self._assemble('full',epsilon,method,quadrature_order,working_dimension)[0]

    def initial_state(self,occupation=None):
        state=np.zeros(self.dimension,complex)
        occupation=tuple(np.zeros(self.basis.shape[1],int) if occupation is None else occupation)
        channel=next(i for i,value in enumerate(self.basis) if tuple(value)==occupation)
        state[channel*self.n**2+self.n**2//2]=1
        return state

    def evolve(self,hamiltonians,initial=None,return_operator=False,solver='action'):
        state=(np.eye(self.dimension,dtype=complex) if return_operator else
               self.initial_state() if initial is None else np.asarray(initial,dtype=complex).copy())
        if solver not in ('action','eigh'): raise ValueError('unknown exponential solver')
        for matrix in hamiltonians:
            if solver=='action':
                scale=-1j*self.dz/self.hv
                state=expm_multiply(scale*matrix,state,traceA=scale*np.trace(matrix))
            else:
                energy,vectors=eigh(matrix,check_finite=False)
                phase=np.exp(-1j*self.dz*energy/self.hv)
                state=vectors@(phase[:,None]*(vectors.conj().T@state)) if state.ndim==2 else vectors@(phase*(vectors.conj().T@state))

        return state

    def propagate(self,epsilon=1.,initial=None,method='analytic',quadrature_order=24,return_operator=False,working_dimension=48,solver='action'):
        return self.evolve(self.hamiltonians(epsilon,method,quadrature_order,working_dimension),initial,return_operator,solver)

    def propagate_taylor(self,epsilon,linear=True,quadratic=True,initial=None):
        h0,h1,h2=self.derivative_hamiltonians()
        return self.evolve(h0+epsilon*linear*h1+epsilon**2*quadratic*h2,initial)

    def leading(self,initial=None):
        """Return (A0,A1,A[V2],A[V1 V1]); displacement order, not vertex order."""
        initial=self.initial_state() if initial is None else np.asarray(initial,dtype=complex)
        state=np.zeros((4,*initial.shape),complex);state[0]=initial
        h0,h1,h2=self.derivative_hamiltonians()
        for i in range(self.steps):
            a,b,c=(csr_matrix(x[i]) for x in (h0,h1,h2))
            triangular=bmat([[a,None,None,None],[b,a,None,None],[c,None,a,None],[None,b,None,a]],format='csr')
            scale=-1j*self.dz/self.hv
            values=state.ravel() if initial.ndim==1 else state.reshape(4*self.dimension,-1)
            state=expm_multiply(scale*triangular,values,traceA=scale*4*np.trace(h0[i])).reshape(state.shape)
        return state

    def project_total(self,state,total):
        return np.asarray(state).reshape(*np.asarray(state).shape[:-1],*self.shape)[...,np.sum(self.basis,axis=1)==total,:,:]


def thermal_initial_weights(omega,temperature_k,basis):
    """Unnormalized infinite Gibbs probabilities of represented initial states.

    Returns missing thermal mass explicitly. No conditional renormalization;
    for a unitary bounded detector the missing initial contribution is <=tail.
    Final occupation truncation is a separate error requiring refinement.
    """
    if temperature_k<0: raise ValueError('temperature must be nonnegative')
    if temperature_k==0:
        weights=np.all(np.asarray(basis)==0,axis=1).astype(float)
    else:
        x=constants.hbar*np.asarray(omega)/(constants.Boltzmann*temperature_k)
        weights=np.exp(-np.asarray(basis)@x)*np.prod(-np.expm1(-x))
    return weights,max(0.,1-float(np.sum(weights)))
