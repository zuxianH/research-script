"""Exact joint lateral momentum sector for nonlinear traveling-mode Fock states.

Electron sector r_n = -sum_mu n_mu h_mu (mod repetitions) for an incident
vacuum/plane wave. The interaction is evaluated at physical transfers
Q_ij=2*pi*(ell+(r_i-r_j)/R)/a, with full specimen-normalized complex modes.
Complete q partners make the inactive covariance translation invariant.
This is an isometry into the declared full electron support, not an electron-
only zero-sector reduction. No displacement Taylor expansion is used.
"""
import hashlib,json,mmap,shutil
from pathlib import Path
from math import lgamma,comb
import numpy as np
from scipy import constants as c
from scipy.special import eval_genlaguerre
from scipy.sparse.linalg import LinearOperator,expm_multiply
from .longitudinal_material_action import LongitudinalMaterialAction
from .coupled_channels import channel_basis
from .electron_phonon import LobatoSilicon
from .free_propagation import electron_wavenumber
from .elastic_propagation import electron_speed
from .provenance import fingerprint,json_write,sha256


def complex_phase_entries(basis,xi):
    """xi[...,i,j,mu]=K_ij.d_mu (annihilation coefficient).

    <n| exp[-i(xi a + xi* a†)] |m>, projected from the infinite oscillator.
    """
    count=len(basis);out=np.ones(xi.shape[:-1],complex)
    for mode in range(basis.shape[1]):
        ni=basis[:,mode,None];nj=basis[None,:,mode];low=np.minimum(ni,nj);high=np.maximum(ni,nj);delta=high-low
        x=xi[...,mode];square=abs(x)**2
        normalization=np.exp(np.array([[.5*(lgamma(int(a)+1)-lgamma(int(b)+1)) for a,b in zip(aa,bb)] for aa,bb in zip(low,high)]))
        coefficient=np.where(ni>=nj,x.conj(),x)
        out*=np.exp(-.5*square)*normalization*(-1j*coefficient)**delta*eval_genlaguerre(low,delta,square)
    return out


def momentum_preflight(modes,p,n,steps,fields,tile=2):
    """Estimate all C-squared and tile work before enumerating any Fock basis."""
    channels=comb(int(modes)+int(p),int(p));count=(2*n-1)**2
    disk=16*count*steps*channels**2
    estimate=16*(tile*max(fields,steps)*channels**2*(12+6*modes)+count*(channels**2*8+channels*24))+512*1024**2
    return dict(channels=channels,modes=modes,n=n,field_cache_bytes=disk,estimated_peak_rss_bytes=estimate,joint_momentum=True,tile=tile)


class MomentumMaterialAction:
    set_slice=LongitudinalMaterialAction.set_slice
    def __init__(self,bath,q_indices,p,n,steps,fields,folder,limits=None,padding_m=1.2e-9,repetitions=(5,5),maximum_bytes=20*1024**3,maximum_disk_bytes=100*1024**3,minimum_free_bytes=100*1024**3,tile=2,initial_index=0):
        if any(x%2!=1 or x<1 for x in (n,steps,fields)):raise ValueError('positive odd grids')
        early=momentum_preflight(len(bath.omega_rad_s),p,n,steps,fields,tile)
        if early['estimated_peak_rss_bytes']>maximum_bytes:raise MemoryError('joint-sector early preflight: '+str(early))
        if early['field_cache_bytes']>maximum_disk_bytes:raise OSError('joint-sector early disk preflight: '+str(early))
        self.d=np.asarray(bath.displacement_m,complex);self.omega=np.asarray(bath.omega_rad_s);self.basis=channel_basis(len(self.omega),p)
        self.basis=self.basis[sorted(range(len(self.basis)),key=lambda i:(sum(self.basis[i]),*tuple(-self.basis[i])))]
        q_indices=np.asarray(q_indices,int);reps=np.array(repetitions,int)
        offsets=((self.basis[initial_index]-self.basis)@q_indices[:,:2]+reps//2)%reps-reps//2
        self.offsets=offsets;channels=len(self.basis);pad=2*n-1;count=pad*pad
        self.fullN=reps*(n-1)+1;self.repetitions=reps;self.n=n;self.steps=steps;self.field_points=fields;self.pad=pad
        fullcell=np.asarray(bath.cell_m);positions=np.asarray(bath.positions_m);cov=np.asarray(bath.inactive_covariance_m2)
        if np.min(np.linalg.eigvalsh(cov)) < -1e-10*np.max(abs(cov)):raise ValueError('inactive covariance not positive')
        cell=fullcell.copy();cell[0]/=reps[0];cell[1]/=reps[1];self.cell_m=cell
        origin=np.flatnonzero((positions[:,0]<cell[0,0]-1e-15)&(positions[:,1]<cell[1,1]-1e-15))
        if len(origin)*int(np.prod(reps))!=len(positions):raise ValueError('incomplete specimen')
        for a in origin:
            for x,y in np.ndindex(*reps):
                expected=positions[a]+x*cell[0]+y*cell[1];j=np.argmin(np.linalg.norm(positions-expected,axis=1))
                if np.linalg.norm(positions[j]-expected)>1e-17:raise ValueError('bad translation map')
                phase=np.exp(2j*np.pi*(q_indices[:,0]*x/reps[0]+q_indices[:,1]*y/reps[1]))
                np.testing.assert_allclose(self.d[j],self.d[a]*phase,rtol=1e-9,atol=1e-24)
                np.testing.assert_allclose(cov[j],cov[a],rtol=1e-9,atol=1e-32)
        self.energy_j=60000*c.e;self.velocity=electron_speed(self.energy_j);self.length=cell[2,2]+2*padding_m;self.dz=self.length/steps
        self.channel_j=(self.basis-self.basis[initial_index])@(c.hbar*self.omega)
        self.shape=(channels,n,n);beam=np.arange(-(n//2),n//2+1);self.beam_indices=beam%pad
        mx=reps[0]*beam[None,:]+offsets[:,0,None];my=reps[1]*beam[None,:]+offsets[:,1,None]
        self.valid=(abs(my[:,:,None])<=self.fullN[1]//2)&(abs(mx[:,None,:])<=self.fullN[0]//2)
        qx=2*np.pi*(beam[None,:]+offsets[:,0,None]/reps[0])/cell[0,0];qy=2*np.pi*(beam[None,:]+offsets[:,1,None]/reps[1])/cell[1,1]
        self.qxy=np.stack(np.broadcast_arrays(qx[:,None,:],qy[:,:,None]),axis=-1)
        self.kinetic_channel_j=c.hbar*self.velocity*np.sum(self.qxy*self.qxy,axis=-1)/(2*electron_wavenumber(self.energy_j))
        disk=16*count*steps*channels**2;estimate=16*(tile*max(fields,steps)*channels**2*(12+6*len(self.omega))+count*(channels**2*8+channels*24))+512*1024**2
        self.preflight=dict(channels=channels,modes=len(self.omega),n=n,fullN=self.fullN,field_cache_bytes=disk,estimated_peak_rss_bytes=estimate,joint_momentum=True,tile=tile)
        if estimate>maximum_bytes:raise MemoryError(str(self.preflight))
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        contract=dict(version=1,d=self.d,cov=cov,positions=positions,cell=fullcell,q_indices=q_indices,basis=self.basis,offsets=offsets,n=n,steps=steps,fields=fields,padding=padding_m,code=sha256(__file__),phase_code=sha256(Path(__file__).with_name('longitudinal_material_action.py')))
        self.field_digest=fingerprint(contract);target=folder/self.field_digest;target.mkdir(exist_ok=True);file=target/'total.npy';manifest=target/'manifest.json'
        current=sum(p.stat().st_size for p in folder.rglob('*.npy'))
        if disk>maximum_disk_bytes or (not file.exists() and not (target/'total.tmp.npy').exists() and current+disk>maximum_disk_bytes) or shutil.disk_usage(folder).free<(0 if file.exists() else disk)+minimum_free_bytes:raise OSError('joint-sector summed cache/disk headroom limit: '+str(self.preflight))
        if limits:limits.check(estimate)
        diff=np.rint(np.fft.fftfreq(pad)*pad).astype(int);dx,dy=np.meshgrid(diff,diff);base=np.column_stack((dx.ravel(),dy.ravel()))
        shift=offsets[:,None,:]-offsets[None,:,:]
        kz=2*np.pi*np.fft.fftfreq(fields,self.length/fields);freq=np.rint(np.fft.fftfreq(fields)*fields).astype(int);pot=LobatoSilicon.from_abtem_table()
        if not manifest.exists():
            tmp=target/'total.tmp.npy';pf=target/'construction.json';progress=json.loads(pf.read_text()) if pf.exists() else dict(digest=self.field_digest,tiles=[])
            if progress['digest']!=self.field_digest:raise ValueError('changed construction contract')
            profile=np.lib.format.open_memmap(tmp,mode='r+' if tmp.exists() else 'w+',dtype=complex,shape=(count,steps,channels,channels))
            negative=(-np.arange(pad))%pad;opposite=(negative[:,None]*pad+negative[None,:]).ravel();canonical=np.flatnonzero(np.arange(count)<=opposite)
            for entry in progress['tiles']:
                if hashlib.sha256(np.array(profile[entry['indices']]).tobytes()).hexdigest()!=entry['sha256']:raise ValueError('construction checkpoint checksum')
                profile._mmap.madvise(mmap.MADV_DONTNEED)
            for lo in range(len(progress['tiles'])*tile,len(canonical),tile):
                if limits:limits.check()
                ids=canonical[lo:lo+tile]
                Q=2*np.pi*(base[ids,None,None,:]+shift[None]/reps)/(np.diag(cell)[:2])
                K=np.empty((len(ids),fields,channels,channels,3));K[...,:2]=Q[:,None];K[...,2]=kz[None,:,None,None]
                kernel=pot.fourier_energy_j_m3(np.sum(K*K,axis=-1));summed=np.zeros(K.shape[:-1],complex)
                for a in origin:
                    shifted=positions[a]+[0,0,padding_m-self.dz/2]
                    value=kernel*np.exp(-.5*np.einsum('...a,ab,...b->...',K,cov[a],K)-1j*(K@shifted))/(cell[0,0]*cell[1,1]*self.length)
                    block=complex_phase_entries(self.basis,K@self.d[a])
                    summed+=value*block
                folded=np.zeros((len(ids),steps,channels,channels),complex)
                for f in range(fields):folded[:,freq[f]%steps]+=summed[:,f]
                values=np.fft.ifft(folded,axis=1)*steps
                if ids[0]==0:values[0]=(values[0]+values[0].swapaxes(-1,-2).conj())/2
                profile[ids]=values;profile[opposite[ids]]=values.swapaxes(-1,-2).conj();profile.flush()
                both=np.unique(np.r_[ids,opposite[ids]]);progress['tiles'].append(dict(indices=both.tolist(),sha256=hashlib.sha256(np.array(profile[both]).tobytes()).hexdigest()));json_write(pf,progress);profile._mmap.madvise(mmap.MADV_DONTNEED)
            del profile;tmp.replace(file);json_write(manifest,dict(digest=self.field_digest,sha256=sha256(file),preflight=self.preflight))
        saved=json.loads(manifest.read_text())
        if saved['digest']!=self.field_digest or saved['sha256']!=sha256(file):raise ValueError('field checksum')
        self.profile=np.load(file,mmap_mode='r');self.scalar=np.trace(np.array(self.profile[0]),axis1=-2,axis2=-1).real/channels;self.profile._mmap.madvise(mmap.MADV_DONTNEED)
        self.cached_index=None;self.potential=None

    def expand(self,wave):
        full=np.zeros((len(self.basis),self.fullN[1],self.fullN[0]),complex);beam=np.arange(-(self.n//2),self.n//2+1)
        for j,r in enumerate(self.offsets):
            mx,my=np.meshgrid(self.repetitions[0]*beam+r[0],self.repetitions[1]*beam+r[1]);valid=self.valid[j]
            full[j,my[valid]+self.fullN[1]//2,mx[valid]+self.fullN[0]//2]=wave[j][valid]
        return full

    def apply(self,wave,index):
        self.set_slice(index);wave=np.asarray(wave)*self.valid
        embedded=np.zeros((len(self.basis),self.pad,self.pad),complex);ix=self.beam_indices;embedded[:,ix[:,None],ix[None,:]]=wave
        position=np.fft.ifft2(embedded,axes=(-2,-1),norm='ortho').reshape(len(self.basis),-1)
        projected=np.einsum('qij,jq->iq',self.potential,position,optimize=True)
        reciprocal=np.fft.fft2(projected.reshape(embedded.shape),axes=(-2,-1),norm='ortho')[:,ix[:,None],ix[None,:]]
        return ((self.channel_j[:,None,None]+self.kinetic_channel_j)*wave+reciprocal)*self.valid

    def exponential_action(self,wave,index,fraction=1.):
        scale=-1j*fraction*self.dz/(c.hbar*self.velocity);count=wave.size
        def action(x,factor):return (factor*self.apply(np.asarray(x).reshape(self.shape),index)).ravel()
        def batch(x,factor):return np.column_stack([action(x[:,i],factor) for i in range(x.shape[1])])
        operator=LinearOperator((count,count),dtype=complex,matvec=lambda x:action(x,scale),rmatvec=lambda x:action(x,scale.conjugate()),matmat=lambda x:batch(x,scale),rmatmat=lambda x:batch(x,scale.conjugate()))
        # traceA is an arbitrary finite trace shift; zero avoids counting the
        # unoccupied masked rectangle, and does not change expm_multiply.
        return expm_multiply(operator,wave.ravel(),traceA=0.).reshape(self.shape)
