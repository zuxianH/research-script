"""Single Gaussian-contraction source with elastic scattering on both sides.

Each traveling phonon mode drives its matching lateral electron momentum
sector. All coefficients use the full finite-specimen mode normalization.
Full-cell beam masks are applied after each Galerkin action. Kinetic splitting
and midpoint-z errors require independent refinement; no nonlinear two-phonon
prediction is made here.
"""
import numpy as np
from scipy import constants as c
from .electron_phonon import LobatoSilicon
from .free_propagation import electron_wavenumber
from .elastic_propagation import electron_speed


def source_profiles(bath,mode_indices,labels,n,steps,fields,padding=1.2e-9,limits=None):
    cell=bath['cell_m'];pos=bath['positions_m'];d=bath['traveling_displacements_m'];cov=bath['local_covariance_m2']
    primitive=cell.copy();primitive[0]/=5;primitive[1]/=5
    chosen=np.flatnonzero((pos[:,0]<primitive[0,0]-1e-15)&(pos[:,1]<primitive[1,1]-1e-15))
    length=cell[2,2]+2*padding;dz=length/steps;pad=2*n-1
    diff=np.rint(np.fft.fftfreq(pad)*pad).astype(int);kz=2*np.pi*np.fft.fftfreq(fields,length/fields);freq=np.rint(np.fft.fftfreq(fields)*fields).astype(int)
    h=np.array(labels[mode_indices[0]]['q_index']);r=(-h[:2]+2)%5-2
    for m in mode_indices:
        if np.any(((-np.array(labels[m]['q_index'][:2])+2)%5-2)!=r):raise ValueError('one lateral sector per source batch')
    qx,qy=np.meshgrid(2*np.pi*(diff+r[0]/5)/primitive[0,0],2*np.pi*(diff+r[1]/5)/primitive[1,1]);Q=np.column_stack((qx.ravel(),qy.ravel()))
    profiles=np.empty((len(mode_indices),steps,pad*pad),complex);pot=LobatoSilicon.from_abtem_table()
    for lo in range(0,len(Q),32):
        if limits:limits.check(profiles.nbytes+512*1024**2)
        q=Q[lo:lo+32];K=np.empty((len(q),fields,3));K[:,:,:2]=q[:,None,:];K[:,:,2]=kz
        kernel=pot.fourier_energy_j_m3(np.sum(K*K,axis=-1));summed=np.zeros((len(mode_indices),len(q),fields),complex)
        for a in chosen:
            factor=kernel*np.exp(-.5*np.einsum('qfa,ab,qfb->qf',K,cov[a],K))
            factor=factor*np.exp(-1j*(K@np.array([pos[a,0],pos[a,1],pos[a,2]+padding-dz/2])))
            vertex=-1j*np.einsum('qfa,am->mqf',K,d[a][:,mode_indices].conj())
            summed+=vertex*factor[None]/(primitive[0,0]*primitive[1,1]*length)
        folded=np.zeros((len(mode_indices),len(q),steps),complex)
        for f in range(fields):folded[:,:,freq[f]%steps]+=summed[:,:,f]
        profiles[:,:,lo:lo+len(q)]=(np.fft.ifft(folded,axis=-1)*steps).transpose(0,2,1)
    return profiles,r


def distorted_sector(background,profiles,r,omega,elastic=True,order=6,limits=None):
    """Triangular one-insertion dynamics, not a unitary depleted coupled model.

    The local exponential is evaluated by a controlled Taylor polynomial. Its
    triangular structure keeps amplitudes coherent and includes one source
    insertion with any number of diagonal elastic interactions.
    """
    h=background;n=h.n;pad=h.pad;steps=h.steps;ix=h.beam_indices;hv=c.hbar*h.velocity
    beam=np.arange(-(n//2),n//2+1);fullN=5*(n-1)+1;bound=fullN//2
    rx,ry=r;mx,my=np.meshgrid(5*beam+rx,5*beam+ry)
    valid=(abs(mx)<=bound)&(abs(my)<=bound)
    qx=2*np.pi*(beam+rx/5)/h.cell_m[0,0];qy=2*np.pi*(beam+ry/5)/h.cell_m[1,1]
    qx,qy=np.meshgrid(qx,qy);outq=np.stack((qx,qy),axis=-1)
    kinetic=np.sum(outq*outq,axis=-1)/(2*electron_wavenumber(h.energy_j))
    incoming=h.kinetic_j/hv
    diagonal=np.concatenate((incoming[None],kinetic[None]+omega[:,None,None]/h.velocity),axis=0)
    half=np.exp(-.5j*h.dz*diagonal)
    wave=np.zeros((1+len(omega),n,n),complex);wave[0,n//2,n//2]=1
    def local_action(x,field,source):
        embedded=np.zeros((len(x),pad,pad),complex);embedded[:,ix[:,None],ix[None,:]]=x
        position=np.fft.ifft2(embedded,axes=(-2,-1),norm='ortho')
        output=field[None]*position
        output[1:]+=source*position[0][None]
        reciprocal=np.fft.fft2(output,axes=(-2,-1),norm='ortho')[:,ix[:,None],ix[None,:]]
        reciprocal[1:]*=valid
        return (-1j*h.dz/hv)*reciprocal
    drift=0.;last_term=0.
    for index in range(steps):
        if limits and index%16==0:limits.check()
        wave*=half
        if elastic:
            h.set_slice(index);field=h.potential[:,0,0].reshape(pad,pad)
        else:field=np.zeros((pad,pad),complex)
        source=np.fft.ifft2(profiles[:,index].reshape(-1,pad,pad),axes=(-2,-1))*pad**2
        term=wave.copy();updated=wave.copy()
        for k in range(1,order+1):
            term=local_action(term,field,source)/k;updated+=term
        last_term=max(last_term,float(np.linalg.norm(term)))
        wave=updated*half;wave[1:]*=valid
        drift=max(drift,abs(float(np.sum(abs(wave[0])**2))-1))
    return wave[1:],outq,dict(incident_norm_drift=drift,maximum_last_taylor_term=last_term,steps=steps,fields=h.field_points,local_order=order,elastic=elastic,fullN=fullN,valid_beams=int(valid.sum()))
