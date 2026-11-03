"""Coherent intermediate-beam resolution for the V0=0 leading pair control.

No elastic potential is supported. Intermediate free propagation is diagonal,
so each ordered sequential amplitude has an exact retained-beam label h.
The same kinetic Strang split and local RK4 polynomial as pair_galerkin are
used; the local target-energy terms are retained through degree four.

Final beams can be restricted to an explicitly declared detector union: at
this displacement order and V0=0 they cannot scatter back from omitted final
beams. Intermediate h support is always explicit. All coefficients use unit
incident probability; thermal/proposal factors are applied only to final
probabilities. No shell intensity is summed as an independent probability.
"""
from pathlib import Path
import mmap
import numpy as np
from scipy import constants,fft
from .free_propagation import electron_wavenumber
from .elastic_propagation import electron_speed
from .dynamical_source import full_transverse_vectors
from .electron_phonon import LobatoSilicon,mode_plane_coefficients
from .two_phonon import SIGNS,masks_for_shift
from .bulk_thermal_source import occupations


def integer_beams(n):
    if int(n)!=n or n<3 or n%2!=1:raise ValueError('odd electron support >=3 required')
    f=np.rint(np.fft.fftfreq(n)*n).astype(int)
    yy,xx=np.meshgrid(f,f,indexing='ij')
    return np.column_stack((yy.ravel(),xx.ravel()))


def detector_output_labels(grid,shifts):
    """Exact union of the physical historical four detector masks."""
    labels=integer_beams(grid.nx)
    keep=np.any([masks_for_shift(grid,a*shifts[0]+b*shifts[1])[0] for a,b in SIGNS],axis=0)
    return labels[keep]


def local_polynomials(scale,first_energy,final_energy):
    """RK4 matrix-polynomial coefficients for a two-insertion triangle."""
    a=np.asarray(first_energy);b=np.asarray(final_energy)
    p=lambda e:1+scale*e+(scale*e)**2/2+(scale*e)**3/6+(scale*e)**4/24
    single=lambda e:scale*(1+scale*e/2+(scale*e)**2/6+(scale*e)**3/24)
    transfer=scale*(1+scale*(a+b)/2+scale**2*(a*a+a*b+b*b)/6+
        scale**3*(a**3+a*a*b+a*b*b+b**3)/24)
    double=scale**2*(.5+scale*(a+b)/6+scale**2*(a*a+a*b+b*b)/24)
    return p(a),p(b),single(a),single(b),transfer,double


def propagate_intermediate_resolved(grid,omega,shifts,length,steps,linear_at,quadratic_at,
        transfer_size,output_labels=None,local_substeps=1,contact_at=None,limits=None):
    """Return both sequential orderings resolved by actual intermediate h.

    linear_at(i): original emission Fourier coefficients (2,T,T), in J.
    quadratic_at(i): full four signed quadratic coefficients (4,Gout), in J.
    contact_at, if supplied, is the original finite-kz Fourier contact part;
    it is separately propagated but the FULL direct amplitude remains primary.
    This function adds no analytic contact events or changed field convention.
    """
    n=grid.nx
    if grid.shape!=(n,n):raise ValueError('square retained electron support required')
    h=integer_beams(n)
    if int(transfer_size)!=transfer_size or transfer_size%2!=1:raise ValueError('odd transfer grid required')
    T=int(transfer_size)
    omega=np.asarray(omega,dtype=float);shifts=np.asarray(shifts,dtype=float)
    if omega.shape!=(2,) or shifts.shape!=(2,2) or np.any(omega<=0):raise ValueError('two positive frequencies and raw transverse shifts required')
    if int(local_substeps)!=local_substeps or local_substeps<1:raise ValueError('positive integer RK4 substeps required')
    out=detector_output_labels(grid,shifts) if output_labels is None else np.asarray(output_labels,dtype=int)
    if out.ndim!=2 or out.shape[1]!=2 or not len(out) or np.any(abs(out)>n//2):raise ValueError('final integer labels must lie in retained support')
    if len(np.unique(out,axis=0))!=len(out):raise ValueError('duplicate final labels')
    delta=out[:,None,:]-h[None,:,:]
    if np.max(abs(delta))>T//2:raise ValueError('transfer grid omits required physical g-h coefficients')
    signed_energy=constants.hbar*np.r_[omega,-omega]
    signed_shift=np.r_[shifts,-shifts]
    first=np.array([[0 if a==1 else 2,1 if b==1 else 3] for a,b in SIGNS])
    second=first[:,::-1].copy()
    final_energy=np.array([a*constants.hbar*omega[0]+b*constants.hbar*omega[1] for a,b in SIGNS])
    final_shift=np.array([a*shifts[0]+b*shifts[1] for a,b in SIGNS])
    base=full_transverse_vectors(grid)
    oi=(out[:,0]%n)*n+out[:,1]%n
    intermediate_Q=base[None]+signed_shift[:,None]
    final_Q=base[oi][None]+final_shift[:,None]
    k0=electron_wavenumber(60000*constants.elementary_charge)
    hv=constants.hbar*electron_speed(60000*constants.elementary_charge)
    dz=length/steps;scale=-1j*dz/(hv*local_substeps)
    kinetic_x=np.exp(-.5j*dz*np.sum(intermediate_Q**2,axis=-1)/(2*k0))
    kinetic_y=np.exp(-.5j*dz*np.sum(final_Q**2,axis=-1)/(2*k0))
    px,pf,bx,bf,transfer,double=local_polynomials(scale,signed_energy[first],final_energy[:,None])
    phase_x=local_polynomials(scale,signed_energy,signed_energy)[0]
    single_x=local_polynomials(scale,signed_energy,signed_energy)[2]
    phase_f=pf[:,0];single_f=bf[:,0]
    x=np.zeros((4,len(h)),complex)
    y=np.zeros((4,2,len(out),len(h)),complex)
    d=np.zeros((2,4,len(out)),complex)
    iy,ix=h[:,0]%T,h[:,1]%T
    dy,dx=delta[...,0]%T,delta[...,1]%T
    opposite=(-np.arange(T))%T
    for i in range(steps):
        if limits is not None and i%8==0:limits.check(estimated_bytes=y.nbytes*6+x.nbytes*8)
        emission=np.asarray(linear_at(i))
        if emission.shape!=(2,T,T):raise ValueError('emission Fourier provider has wrong shape')
        signed=np.concatenate((emission,emission[:,opposite[:,None],opposite].conj()))
        at_h=signed[:,iy,ix]
        at_delta=signed[second[:,:,None,None],dy[None,None],dx[None,None]]
        q=np.asarray(quadratic_at(i))
        if q.shape!=(4,len(out)):raise ValueError('quadratic provider has wrong final-label shape')
        c=np.zeros_like(q) if contact_at is None else np.asarray(contact_at(i))
        if c.shape!=q.shape:raise ValueError('contact provider has wrong shape')
        direct_source=np.stack((q,c))
        x*=kinetic_x;y*=kinetic_y[:,None,:,None];d*=kinetic_y[None]
        for j in range(int(local_substeps)):
            y*=phase_f[:,None,None,None]
            y+=at_delta*(transfer[:,:,None,None]*x[first][:,:,None,:]+
                         double[:,:,None,None]*at_h[first][:,:,None,:])
            d=phase_f[None,:,None]*d+single_f[None,:,None]*direct_source
            x=phase_x[:,None]*x+single_x[:,None]*at_h
        x*=kinetic_x;y*=kinetic_y[:,None,:,None];d*=kinetic_y[None]
    sequential=y.sum(axis=(1,3))
    return dict(ordered_by_intermediate=y,sequential_amplitudes=sequential,
        direct_amplitudes=d[0],direct_contact_amplitudes=d[1],direct_regular_amplitudes=d[0]-d[1],
        total_amplitudes=d[0]+sequential,first_amplitudes=x,intermediate_integer_labels=h,
        output_integer_labels=out,first_letter_indices=first,second_letter_indices=second,
        intermediate_Q_m_inv=intermediate_Q[first],intermediate_momentum_kg_m_s=constants.hbar*intermediate_Q[first],
        intermediate_Q_over_k0=np.linalg.norm(intermediate_Q[first],axis=-1)/k0,
        final_shifts=final_shift,final_Q_m_inv=final_Q,omega_rad_s=omega,
        final_energy_j=final_energy,retained_grid=n,transfer_grid=T,steps=steps,local_substeps=int(local_substeps),
        incident_k0_m_inv=k0,incident_wavelength_m=2*np.pi/k0,
        ordering_labels=np.array(['first_a_then_b','first_b_then_a']),
        sign_labels=np.array(SIGNS),
        scope='V0=0 exact finite support; detector output union, all intermediate beams. Same paraxial kinetic, Strang split and RK4 local polynomial.')


def cumulative_amplitudes(result,masks):
    """Masks shape(cutoff,sign,ordering,h); sum amplitudes before squaring."""
    masks=np.asarray(masks,dtype=bool)
    if masks.ndim!=4 or masks.shape[1:]!=(4,2,len(result['intermediate_integer_labels'])):
        raise ValueError('one explicit mask per cutoff/sign/ordering/intermediate beam required')
    ordered=np.einsum('csok,sogk->csog',masks,result['ordered_by_intermediate'],optimize=True)
    sequential=ordered.sum(axis=2)
    return dict(ordered=ordered,sequential=sequential,total=result['direct_amplitudes'][None]+sequential)


def projected_trace(direct,sequential,omega,nbar,grid,shifts,output_labels,inverse_marginals):
    """Historical thermal/ordered-pair factors on an explicit output subset."""
    n=grid.nx;labels=np.asarray(output_labels);index=(labels[:,0]%n)*n+labels[:,1]%n
    mask=np.array([masks_for_shift(grid,s)[:,index] for s in shifts])
    factor=.5*float(np.prod(inverse_marginals))*np.array([(nbar[0]+(a==1))*(nbar[1]+(b==1)) for a,b in SIGNS])
    energy=np.array([constants.hbar*(a*omega[0]+b*omega[1])/(1e-3*constants.elementary_charge) for a,b in SIGNS])
    components={'direct_weights':abs(direct)**2,'sequential_weights':abs(sequential)**2,
        'interference_weights':2*np.real(direct*sequential.conj()),'weights':abs(direct+sequential)**2}
    return dict({key:np.einsum('sdg,...sg,s->...sd',mask,value,factor) for key,value in components.items()},
        energies_mev=energy,detector_masks=mask,thermal_importance_factor=factor)


def build_fourier_sources(positions,displacements,shifts,grid,length,steps,output_labels,folder,chunk=128,limits=None):
    """Original finite-kz midpoint fields evaluated at only required transfers.

    The formulas are the pinned 3D-Fourier constructors. Every FFT explicitly
    uses one worker. Truncating this *transfer inventory* omits no required
    coefficient; no product is performed on this grid, so no circular aliasing
    is introduced. Contact is the same finite-kz Fourier distribution, not an
    added analytic event or a changed longitudinal representation.
    """
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    positions=np.asarray(positions);d=np.asarray(displacements);shifts=np.asarray(shifts)
    out=np.asarray(output_labels);K=grid.nx//2;radius=K+int(np.max(abs(out)));T=2*radius+1
    freq=np.rint(np.fft.fftfreq(T)*T).astype(int)
    yy,xx=np.meshgrid(freq,freq,indexing='ij')
    base=np.column_stack((xx.ravel()*2*np.pi/grid.lx_m,yy.ravel()*2*np.pi/grid.ly_m))
    dz=length/steps;kz=2*np.pi*np.fft.fftfreq(steps,dz);area=grid.lx_m*grid.ly_m
    planes,inverse=np.unique(positions[:,2],return_inverse=True)
    axial=np.exp(-1j*(planes[:,None]-dz/2)*kz)
    potential=LobatoSilicon.from_abtem_table()
    linear=np.lib.format.open_memmap(folder/'linear.npy',mode='w+',dtype=complex,shape=(steps,2,T,T))
    flat=linear.reshape(steps,2,T*T)
    for side in range(2):
        for lo in range(0,len(base),chunk):
            if limits is not None:limits.check()
            hi=min(lo+chunk,len(base));Q=base[lo:hi]+shifts[side]
            _,coeff=mode_plane_coefficients(positions,d[:,:,side:side+1],Q)
            xy=coeff[0,:,0]*Q[None,:,0]+coeff[0,:,1]*Q[None,:,1]
            field=-1j*(xy.T@axial+(coeff[0,:,2].T@axial)*kz)
            field*=potential.fourier_energy_j_m3(np.sum(Q*Q,axis=1)[:,None]+kz*kz)/(area*dz)
            flat[:,side,lo:hi]=fft.ifft(field,axis=-1,workers=1).T
            linear._mmap.madvise(mmap.MADV_DONTNEED)
    quadratic=np.lib.format.open_memmap(folder/'quadratic.npy',mode='w+',dtype=complex,shape=(steps,2,4,len(out)))
    qout=np.column_stack((out[:,1]*2*np.pi/grid.lx_m,out[:,0]*2*np.pi/grid.ly_m))
    ca=-(2*np.pi*constants.hbar**2/constants.m_e)*potential.a_angstrom*1e-10
    beta2=4*np.pi**2/(potential.b_angstrom2*1e-20);contact=-np.sum(ca*beta2)
    for sign,(sa,sb) in enumerate(SIGNS):
        if limits is not None:limits.check()
        a=d[:,:,0].conj() if sa==1 else d[:,:,0]
        b=d[:,:,1].conj() if sb==1 else d[:,:,1]
        Q=qout+sa*shifts[0]+sb*shifts[1]
        c0=np.zeros((len(planes),len(Q)),complex);c1=np.zeros_like(c0);c2=np.zeros_like(c0)
        for p in range(len(planes)):
            pick=inverse==p;phase=np.exp(-1j*positions[pick,:2]@Q.T)
            qa=a[pick,:2]@Q.T;qb=b[pick,:2]@Q.T;az=a[pick,2,None];bz=b[pick,2,None]
            c0[p]=np.sum(phase*qa*qb,axis=0);c1[p]=np.sum(phase*(qa*bz+az*qb),axis=0);c2[p]=np.sum(phase*az*bz,axis=0)
        kernel=potential.fourier_energy_j_m3(np.sum(Q*Q,axis=1)[:,None]+kz*kz)/(area*dz)
        full=-(c0.T@axial+(c1.T@axial)*kz+(c2.T@axial)*kz*kz)*kernel
        delta=(c2.T@axial)*contact/(area*dz)
        quadratic[:,0,sign]=fft.ifft(full,axis=-1,workers=1).T
        quadratic[:,1,sign]=fft.ifft(delta,axis=-1,workers=1).T
    linear.flush();quadratic.flush()
    return linear,quadratic,T
