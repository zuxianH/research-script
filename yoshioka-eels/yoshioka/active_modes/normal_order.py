"""Exact normal-ordered p<=2 phase construction; no displacement truncation.

The full Gaussian exp(-sum |K.d|²/2) is retained. Remaining associated-Laguerre
polynomials are evaluated only on nonzero occupation entries. This is an exact
identity for the projected infinite-oscillator operator, not a linear or
quadratic approximation to the electron potential.
"""
import hashlib,json,mmap,shutil
from pathlib import Path
import numpy as np
from scipy import constants as c
from ..momentum_material_action import MomentumMaterialAction,momentum_preflight
from ..coupled_channels import channel_basis
from ..electron_phonon import LobatoSilicon
from ..free_propagation import electron_wavenumber
from ..elastic_propagation import electron_speed
from ..provenance import fingerprint,json_write,sha256


def normal_phase_entries(basis,xi,gaussian=True):
    basis=np.asarray(basis,int)
    if basis.max()>2:raise ValueError('exact specialized block supports occupation <=2')
    out=np.exp(-.5*np.sum(abs(xi)**2,axis=-1)).astype(complex) if gaussian else np.ones(xi.shape[:-1],complex)
    for mode in range(basis.shape[1]):
        ni=basis[:,mode,None];nj=basis[None,:,mode];low=np.minimum(ni,nj);high=np.maximum(ni,nj);delta=high-low
        for step in (1,2):
            rows,cols=np.where(delta==step)
            if len(rows):
                x=xi[...,rows,cols,mode]
                x=np.where(ni[rows,0]>=nj[0,cols],x.conj(),x)
                out[...,rows,cols]*=(-1j*x if step==1 else -x*x)
        rows,cols=np.where((high==2)&(low<2))
        out[...,rows,cols]/=np.sqrt(2)
        rows,cols=np.where(low==1)
        if len(rows):out[...,rows,cols]*=(delta[rows,cols]+1)-abs(xi[...,rows,cols,mode])**2
        rows,cols=np.where(low==2)
        if len(rows):
            square=abs(xi[...,rows,cols,mode])**2
            out[...,rows,cols]*=1-2*square+.5*square*square
    return out


class NormalOrderedMomentumAction(MomentumMaterialAction):
    def __init__(self,bath,q_indices,p,n,steps,fields,folder,limits=None,padding_m=1.2e-9,repetitions=(5,5),maximum_bytes=20*1024**3,maximum_disk_bytes=100*1024**3,minimum_free_bytes=100*1024**3,tile=2,initial_index=0):
        if p>2:raise ValueError('specialized exact action supports total occupation p<=2')
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
        contract=dict(version=1,d=self.d,cov=cov,positions=positions,cell=fullcell,q_indices=q_indices,basis=self.basis,offsets=offsets,n=n,steps=steps,fields=fields,padding=padding_m,code=sha256(__file__),base_code=sha256(Path(__file__).resolve().parents[1]/'momentum_material_action.py'),phase_code=sha256(Path(__file__).resolve().parents[1]/'longitudinal_material_action.py'))
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
                groups={}
                for atom in origin:groups.setdefault((self.d[atom].tobytes(),cov[atom].tobytes(),positions[atom,:2].tobytes()),[]).append(atom)
                for group in groups.values():
                    a=group[0]
                    zphase=np.exp(-1j*(positions[group,2,None]+padding_m-self.dz/2)*kz).sum(axis=0)
                    effective_cov=cov[a]+(self.d[a]@self.d[a].conj().T).real
                    value=kernel*np.exp(-.5*np.einsum('...a,ab,...b->...',K,effective_cov,K)-1j*(K[...,:2]@positions[a,:2]))*zphase[None,:,None,None]/(cell[0,0]*cell[1,1]*self.length)
                    block=normal_phase_entries(self.basis,K@self.d[a],gaussian=False)
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

