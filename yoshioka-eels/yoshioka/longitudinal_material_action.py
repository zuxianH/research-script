"""All-direction nonlinear projected atom potential, streamed in Fourier tiles.

For each atom and full K=(Qx,Qy,kz), evaluate the exact infinite-oscillator block
P exp(-i sum K.d_mu X_mu) P before the longitudinal Fourier sum. No displacement
expansion is made. This bounded implementation stores the summed C² field;
its disk cost is explicitly checked. It also supports real non-Gamma modes,
provided the full electron cell is retained (no electron-only sector claim).
"""
import json,mmap,shutil,hashlib
from pathlib import Path
from math import lgamma
import numpy as np
from scipy import constants as c
from scipy.special import eval_genlaguerre
from .reciprocal_material import resolve_material_bath,ReciprocalMaterialHamiltonian
from .coupled_channels import channel_basis
from .electron_phonon import LobatoSilicon
from .free_propagation import electron_wavenumber
from .elastic_propagation import electron_speed
from .provenance import fingerprint,json_write,sha256


def exact_phase_blocks(basis,xi):
    basis=np.asarray(basis,int);xi=np.asarray(xi,float)
    out=np.ones((len(xi),len(basis),len(basis)),complex)
    for mode in range(basis.shape[1]):
        x=xi[:,mode];small=basis[:,mode,None];large=basis[None,:,mode]
        low=np.minimum(small,large);high=np.maximum(small,large);delta=high-low
        block=np.empty_like(out)
        for i in range(len(basis)):
            for j in range(len(basis)):
                a=int(low[i,j]);b=int(high[i,j]);k=int(delta[i,j])
                block[:,i,j]=np.exp(-.5*x*x+.5*(lgamma(a+1)-lgamma(b+1)))*(-1j*x)**k*eval_genlaguerre(a,k,x*x)
        out*=block
    return out


class LongitudinalMaterialAction:
    exponential_action=ReciprocalMaterialHamiltonian.exponential_action
    def __init__(self,bath,p,n,steps,field_points,folder,limits=None,padding_m=1.2e-9,
                 maximum_bytes=20*1024**3,maximum_disk_bytes=100*1024**3,minimum_free_bytes=100*1024**3,tile=8,initial_index=0):
        if min(n,steps,field_points)<1 or any(x%2!=1 for x in (n,steps,field_points)):raise ValueError('positive odd grids required')
        self.d,self.omega,self.inactive_covariance,self.mode_labels,cell,positions,self.bath_scope=resolve_material_bath(None,bath=bath,allow_longitudinal=True)
        self.basis=channel_basis(len(self.omega),p);self.basis=self.basis[sorted(range(len(self.basis)),key=lambda i:(sum(self.basis[i]),*tuple(-self.basis[i])))]
        channels=len(self.basis);pad=2*n-1;count=pad*pad
        disk=16*count*steps*channels**2
        # Includes tile phase/field/Lobato copies, one mapped tile plus slice,
        # 24 state vectors, local matrices and FFT/action construction copies.
        estimate=16*(tile*max(field_points,steps)*(channels**2*8+32)+count*(channels**2*8+channels*24))+512*1024**2
        self.preflight=dict(field_cache_bytes=disk,estimated_peak_rss_bytes=estimate,channels=channels,n=n,steps=steps,fields=field_points,tile=tile,oscillator='exact infinite Fock block via associated Laguerre polynomials')
        if estimate>maximum_bytes:raise MemoryError(str(self.preflight))
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        if disk>maximum_disk_bytes or shutil.disk_usage(folder).free<disk+minimum_free_bytes:raise OSError('summed C-squared field exceeds disk/headroom budget: '+str(self.preflight))
        if limits:limits.check(estimate)
        self.n=n;self.steps=steps;self.pad=pad;self.cell_m=cell;self.positions_m=positions
        self.energy_j=60000*c.e;self.velocity=electron_speed(self.energy_j);self.length=cell[2,2]+2*padding_m;self.dz=self.length/steps
        self.channel_j=(self.basis-self.basis[initial_index])@(c.hbar*self.omega)
        beam=np.arange(-(n//2),n//2+1);self.beam_indices=beam%pad
        qx,qy=np.meshgrid(2*np.pi*beam/cell[0,0],2*np.pi*beam/cell[1,1]);self.qxy=np.stack((qx,qy),axis=-1)
        self.kinetic_j=c.hbar*self.velocity*(qx*qx+qy*qy)/(2*electron_wavenumber(self.energy_j));self.shape=(channels,n,n)
        difference=np.rint(np.fft.fftfreq(pad)*pad).astype(int)
        qx,qy=np.meshgrid(2*np.pi*difference/cell[0,0],2*np.pi*difference/cell[1,1]);self.Q=np.column_stack((qx.ravel(),qy.ravel()))
        kz=2*np.pi*np.fft.fftfreq(field_points,self.length/field_points);freq=np.rint(np.fft.fftfreq(field_points)*field_points).astype(int)
        pot=LobatoSilicon.from_abtem_table()
        contract=dict(version=1,n=n,steps=steps,fields=field_points,padding_m=padding_m,basis=self.basis,cell=cell,positions=positions,d=self.d,covariance=self.inactive_covariance,lobato=sha256(pot.source_path),operator_sha256=sha256(__file__))
        self.field_digest=fingerprint(contract);target=folder/self.field_digest;target.mkdir(exist_ok=True)
        file=target/'total.npy';manifest=target/'manifest.json'
        if not manifest.exists():
            tmp=target/'total.tmp.npy'
            progress_file=target/'construction.json'
            progress=json.loads(progress_file.read_text()) if progress_file.exists() else dict(digest=self.field_digest,tiles=[])
            if progress['digest']!=self.field_digest:raise ValueError('construction checkpoint contract mismatch')
            profile=np.lib.format.open_memmap(tmp,mode='r+' if tmp.exists() else 'w+',dtype=complex,shape=(count,steps,channels,channels))
            negative=(-np.arange(pad))%pad;opposite=(negative[:,None]*pad+negative[None,:]).ravel();canonical=np.flatnonzero(np.arange(count)<=opposite)
            # Sum identical displacement/covariance/xy templates before expensive
            # oscillator evaluation; longitudinal positions enter their phases.
            groups={}
            for a,pos in enumerate(positions):groups.setdefault((self.d[a].tobytes(),self.inactive_covariance[a].tobytes(),pos[:2].tobytes()),[]).append(a)
            for entry in progress['tiles']:
                ids=np.array(entry['indices']);data=np.array(profile[ids])
                if hashlib.sha256(data.tobytes()).hexdigest()!=entry['sha256']:raise ValueError('construction checkpoint checksum mismatch')
                profile._mmap.madvise(mmap.MADV_DONTNEED)
            completed=len(progress['tiles'])*tile
            for lo in range(completed,len(canonical),tile):
                if limits:limits.check()
                idx=canonical[lo:lo+tile];Q=self.Q[idx];K=np.empty((len(idx),field_points,3));K[:,:,:2]=Q[:,None,:];K[:,:,2]=kz
                summed=np.zeros((len(idx),field_points,channels,channels),complex)
                kernel=pot.fourier_energy_j_m3(np.sum(K*K,axis=-1))
                for group in groups.values():
                    a=group[0];cov=self.inactive_covariance[a]
                    value=kernel*np.exp(-.5*np.einsum('qfa,ab,qfb->qf',K,cov,K))
                    value=value.astype(complex)*np.exp(-1j*Q@positions[a,:2])[:,None]*np.exp(-1j*(positions[group,2,None]+padding_m-self.dz/2)*kz).sum(axis=0)[None,:]/(cell[0,0]*cell[1,1]*self.length)
                    block=exact_phase_blocks(self.basis,(K@self.d[a]).reshape(-1,len(self.omega))).reshape(len(idx),field_points,channels,channels)
                    summed+=value[:,:,None,None]*block
                folded=np.zeros((len(idx),steps,channels,channels),complex)
                for f in range(field_points):folded[:,freq[f]%steps]+=summed[:,f]
                values=np.fft.ifft(folded,axis=1)*steps
                if idx[0]==0:values[0]=(values[0]+values[0].swapaxes(-1,-2).conj())/2
                profile[idx]=values;profile[opposite[idx]]=values.swapaxes(-1,-2).conj()
                profile.flush()
                both=np.unique(np.r_[idx,opposite[idx]])
                progress['tiles'].append(dict(indices=both.tolist(),sha256=hashlib.sha256(np.array(profile[both]).tobytes()).hexdigest()))
                json_write(progress_file,progress)
                profile._mmap.madvise(mmap.MADV_DONTNEED)
            del profile;tmp.replace(file)
            json_write(manifest,dict(digest=self.field_digest,sha256=sha256(file),preflight=self.preflight))
        saved=json.loads(manifest.read_text())
        if saved['digest']!=self.field_digest or saved['sha256']!=sha256(file):raise ValueError('all-direction field identity mismatch')
        self.profile=np.load(file,mmap_mode='r');self.scalar=np.trace(np.array(self.profile[0]),axis1=-2,axis2=-1).real/channels
        self.profile._mmap.madvise(mmap.MADV_DONTNEED);self.cached_index=None;self.potential=None

    def set_slice(self,index):
        if self.cached_index==index:return
        fourier=np.array(self.profile[:,index]);self.profile._mmap.madvise(mmap.MADV_DONTNEED)
        local=np.fft.ifft2(fourier.reshape(self.pad,self.pad,len(self.basis),len(self.basis)),axes=(0,1))*self.pad**2
        error=np.linalg.norm(local-local.swapaxes(-1,-2).conj())/max(np.linalg.norm(local),np.finfo(float).tiny)
        if error>1e-10:raise ValueError('non-Hermitian all-direction field')
        self.potential=local.reshape(-1,len(self.basis),len(self.basis));self.cached_index=index

    def apply(self,wave,index=None):
        from .projected_material_action import ProjectedMaterialAction
        return ProjectedMaterialAction.apply(self,wave,index)
