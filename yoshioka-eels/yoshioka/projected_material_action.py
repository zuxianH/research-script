"""Bounded nonlinear PHP action with independent field and propagation grids.

Atom profiles are constructed in transfer tiles and stored as read-only arrays.
Only one z slice is resident during propagation. Local C by C channel fields
are formed from converged oscillator phase blocks; every H action embeds then
projects the full N by N reciprocal rectangle using M=2N-1 (no cyclic alias).
"""
import json
import mmap
from pathlib import Path
import numpy as np
from scipy import constants
from scipy.linalg import eigh_tridiagonal
from scipy.special import roots_hermite
from .coupled_channels import channel_basis
from .harmonic_oscillator import normalized_hermite_polynomials
from .reciprocal_material import resolve_material_bath, ReciprocalMaterialHamiltonian
from .electron_phonon import LobatoSilicon
from .elastic_propagation import electron_speed
from .free_propagation import electron_wavenumber
from .provenance import fingerprint, json_write, sha256


def field_preflight(atoms, n, fields, steps, channels, working_dimension=32, tile=128, storage='mapped'):
    m=2*n-1; q=m*m
    profiles=16*atoms*q*steps
    # Includes phase tensors, construction copies/Lobato 5-term arrays, local
    # matrices, FFT buffers, 24 state vectors, and a 256 MiB library allowance.
    phase=16*atoms*q*channels**2
    construction=16*tile*max(fields,steps)*(12+5)
    buffers=16*q*(channels**2*8+24*channels)+16*q*atoms*2
    oscillator=16*tile*working_dimension*working_dimension*3
    # A mapped template is populated completely before its pages are dropped.
    # During full-storage conversion it can also coexist with its copied array.
    mapped_construction_profile=16*q*steps
    resident=2*phase+construction+buffers+oscillator+mapped_construction_profile+256*1024**2
    if storage=='full': resident+=profiles
    elif storage!='mapped': raise ValueError('storage must be mapped or full')
    return dict(atoms=atoms,n=n,m=m,field_points=fields,propagation_intervals=steps,channels=channels,
                full_complex_profile_bytes=profiles,phase_bytes=phase,construction_bytes=construction,
                mapped_construction_profile_bytes=mapped_construction_profile,
                estimated_peak_rss_bytes=resident,storage=storage,transfer_tile=tile,
                note='Conservative estimate, not measured RSS; mapped pages dropped after construction and each read')


def phase_blocks(basis, xi, dimension=32, quadrature_order=None):
    """Vectorized low-state blocks of the large oscillator exponential.

    Diagonalize the same D-dimensional X=a+a† used by fock_scattering_matrix;
    exponentiate there, then project. Optional Hermite quadrature is independent.
    """
    basis=np.asarray(basis);xi=np.asarray(xi);p=int(basis.max())
    if dimension<=p: raise ValueError('working oscillator dimension must exceed p')
    if quadrature_order is None:
        nodes,vectors=eigh_tridiagonal(np.zeros(dimension),np.sqrt(np.arange(1,dimension)))
        left=vectors[:p+1]
    else:
        nodes,weights=roots_hermite(quadrature_order)
        left=normalized_hermite_polynomials(nodes,p)*np.sqrt(weights/np.sqrt(np.pi))
        nodes=nodes*np.sqrt(2)
    output=np.ones((len(xi),len(basis),len(basis)),complex)
    for mode in range(basis.shape[1]):
        blocks=np.einsum('ik,qk,jk->qij',left,np.exp(-1j*xi[:,mode,None]*nodes),left,optimize=True)
        # Real oscillator wavefunctions make each phase block complex symmetric.
        # Enforce that algebraic construction identity before forming a field.
        blocks=(blocks+blocks.swapaxes(-1,-2))/2
        states=basis[:,mode]
        output*=blocks[:,states[:,None],states[None,:]]
    return output


class ProjectedMaterialAction:
    exponential_action=ReciprocalMaterialHamiltonian.exponential_action

    def __init__(self,model,p,n,steps,field_points,folder,bath=None,working_dimension=32,
                 quadrature_order=None,padding_m=1.2e-9,maximum_bytes=4*1024**3,
                 limits=None,storage='mapped',initial_index=0,coalesce_profiles=False):
        if min(n,steps,field_points)<1 or any(v%2!=1 for v in (n,steps,field_points)):
            raise ValueError('positive odd electron and depth grids required')
        self.d,self.omega,self.inactive_covariance,self.mode_labels,cell,positions,self.bath_scope=resolve_material_bath(model,2,bath)
        self.basis=channel_basis(len(self.omega),p)
        self.basis=self.basis[sorted(range(len(self.basis)),key=lambda i:(sum(self.basis[i]),*tuple(-self.basis[i])))]
        self.n=n;self.steps=steps;self.field_points=field_points;self.pad=2*n-1
        groups={}
        for atom,pos in enumerate(positions):
            key=(self.d[atom].tobytes(),self.inactive_covariance[atom].tobytes(),pos[:2].tobytes()) if coalesce_profiles else atom
            groups.setdefault(key,[]).append(atom)
        groups=list(groups.values())
        self.preflight=field_preflight(len(groups),n,field_points,steps,len(self.basis),working_dimension,storage=storage)
        self.preflight.update(physical_atoms=len(positions),profile_templates=len(groups),coalesce_profiles=coalesce_profiles)
        if self.preflight['estimated_peak_rss_bytes']>maximum_bytes:
            raise MemoryError('bounded field/action construction preflight exceeds RSS budget')
        if limits is not None: limits.check(self.preflight['estimated_peak_rss_bytes'])
        self.cell_m=cell.copy();self.positions_m=positions.copy()
        self.energy_j=60000*constants.e;self.velocity=electron_speed(self.energy_j)
        self.length=cell[2,2]+2*padding_m;self.dz=self.length/steps
        channel=self.basis@(constants.hbar*self.omega)
        self.channel_j=channel-channel[initial_index]
        beam=np.arange(-(n//2),n//2+1);self.beam_indices=beam%self.pad
        qx,qy=np.meshgrid(2*np.pi*beam/cell[0,0],2*np.pi*beam/cell[1,1])
        self.qxy=np.stack((qx,qy),axis=-1)
        self.kinetic_j=constants.hbar*self.velocity*(qx*qx+qy*qy)/(2*electron_wavenumber(self.energy_j))
        self.shape=(len(self.basis),n,n)
        difference=np.rint(np.fft.fftfreq(self.pad)*self.pad).astype(int)
        qx,qy=np.meshgrid(2*np.pi*difference/cell[0,0],2*np.pi*difference/cell[1,1])
        self.Q=np.column_stack((qx.ravel(),qy.ravel()))
        potential=LobatoSilicon.from_abtem_table()
        kz=2*np.pi*np.fft.fftfreq(field_points,self.length/field_points)
        frequencies=np.rint(np.fft.fftfreq(field_points)*field_points).astype(int)
        contract=dict(version=2,n=n,steps=steps,field_points=field_points,padding_m=padding_m,
                      cell=cell.tolist(),positions=positions.tolist(),covariance=self.inactive_covariance.tolist(),
                      lobato_sha256=sha256(potential.source_path))
        if coalesce_profiles: contract.update(version=3,profile_groups=groups)
        self.field_digest=fingerprint(contract)
        folder=Path(folder)/self.field_digest;folder.mkdir(parents=True,exist_ok=True)
        manifest=folder/'manifest.json'
        saved=json.loads(manifest.read_text()) if manifest.exists() else None
        if manifest.exists() and json.loads(manifest.read_text())['contract']!=contract:
            raise ValueError('field cache contract mismatch')
        self.profiles=[];self.phase=[];self.scalar=np.zeros(steps)
        for template,group in enumerate(groups):
            atom=group[0];pos=positions[atom]
            if limits is not None: limits.check()
            file=folder/f'atom_{template:03d}.npy'
            if not manifest.exists():
                temporary=file.with_suffix('.tmp.npy')
                profile=np.lib.format.open_memmap(temporary,mode='w+',dtype=complex,shape=(steps,len(self.Q)))
                cov=self.inactive_covariance[atom]
                negative=(-np.arange(self.pad))%self.pad
                opposite=(negative[:,None]*self.pad+negative[None,:]).ravel()
                canonical=np.flatnonzero(np.arange(len(self.Q))<=opposite)
                # A real atomic potential obeys profile(-Q,z)=conj(profile(Q,z)).
                # Evaluate one member per pair to avoid catastrophic relative
                # roundoff between separately cancelled vacuum-tail transforms.
                for lo in range(0,len(canonical),128):
                    if limits is not None: limits.check()
                    indices=canonical[lo:lo+128]
                    Q=self.Q[indices]
                    exponent=np.einsum('qa,ab,qb->q',Q,cov[:2,:2],Q)[:,None]+2*(Q@cov[:2,2])[:,None]*kz+cov[2,2]*kz*kz
                    fourier=potential.fourier_energy_j_m3(np.sum(Q*Q,axis=1)[:,None]+kz*kz)
                    fourier*=np.exp(-.5*exponent)
                    zphase=np.exp(-1j*(positions[group,2,None]+padding_m-self.dz/2)*kz).sum(axis=0)
                    fourier=fourier.astype(complex)*zphase/(cell[0,0]*cell[1,1]*self.length)
                    # Exact evaluation of the declared finite Fourier field at
                    # propagation midpoints. Fold frequencies if steps < fields;
                    # no interpolation or change of the field band limit.
                    folded=np.zeros((len(Q),steps),complex)
                    for f in range(field_points): folded[:,frequencies[f]%steps]+=fourier[:,f]
                    values=np.fft.ifft(folded,axis=-1)*steps
                    values*=np.exp(-1j*Q@pos[:2])[:,None]
                    if indices[0]==0:
                        zero=values[0]
                        if np.linalg.norm(zero.imag)>1e-12*np.linalg.norm(zero.real):
                            raise ValueError('zero-transfer atomic field is not real')
                        values[0]=zero.real
                    profile[:,indices]=values.T
                    profile[:,opposite[indices]]=values.conj().T
                profile.flush();profile._mmap.madvise(mmap.MADV_DONTNEED);del profile
                temporary.replace(file)
            if saved is not None and sha256(file)!=saved['files'][file.name]:
                raise ValueError('cached field checksum mismatch')
            profile=np.load(file,mmap_mode='r')
            if profile.shape!=(steps,len(self.Q)) or profile.dtype!=np.complex128: raise ValueError('invalid cached field shape/dtype')
            self.scalar+=profile[:,0].real
            profile._mmap.madvise(mmap.MADV_DONTNEED)
            if storage=='full':
                full=np.array(profile);profile._mmap.madvise(mmap.MADV_DONTNEED);profile=full
            self.profiles.append(profile)
            # C² atom phase is small at p2. It is part of preflight, not hidden.
            self.phase.append(phase_blocks(self.basis,self.Q@self.d[atom,:2,:],working_dimension,quadrature_order))
        self.phase=np.asarray(self.phase)
        if not manifest.exists():
            json_write(manifest,{'contract':contract,'files':{f.name:sha256(f) for f in folder.glob('atom_*.npy')}})
        self.cached_index=None;self.potential=None

    def set_slice(self,index):
        if index==self.cached_index: return
        fourier=np.zeros((len(self.Q),len(self.basis),len(self.basis)),complex)
        for atom,profile in enumerate(self.profiles):
            values=np.array(profile[index])
            if isinstance(profile,np.memmap): profile._mmap.madvise(mmap.MADV_DONTNEED)
            fourier+=values[:,None,None]*self.phase[atom]
        local=np.fft.ifft2(fourier.reshape(self.pad,self.pad,len(self.basis),len(self.basis)),axes=(0,1))*self.pad**2
        error=np.linalg.norm(local-local.swapaxes(-1,-2).conj())/np.linalg.norm(local)
        if error>1e-10: raise ValueError('projected local field lost Hermiticity')
        self.potential=local.reshape(-1,len(self.basis),len(self.basis))
        self.cached_index=index

    def apply(self,wave,index=None):
        index=self.cached_index if index is None else index
        self.set_slice(index)
        wave=np.asarray(wave,dtype=complex)
        output=(self.channel_j[:,None,None]+self.kinetic_j[None])*wave
        embedded=np.zeros((len(self.basis),self.pad,self.pad),complex)
        ix=self.beam_indices;embedded[:,ix[:,None],ix[None,:]]=wave
        position=np.fft.ifft2(embedded,axes=(-2,-1),norm='ortho').reshape(len(self.basis),-1)
        projected=np.einsum('qij,jq->iq',self.potential,position,optimize=True)
        reciprocal=np.fft.fft2(projected.reshape(embedded.shape),axes=(-2,-1),norm='ortho')
        return output+reciprocal[:,ix[:,None],ix[None,:]]
