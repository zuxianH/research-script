"""Fixed-support, dealiased Galerkin action for the complete changed pair.

Wave coefficients retain a square of integer reciprocal beams. A separate
product grid represents all transfers through twice that half-width. Every H
action is projected, including between two linear vertices. Thus this is
exp(P H P), rather than P exp(H) P or a circular collocation product. The local
exponential uses explicit RK4 with independently refinable substeps; kinetic
Strang splitting and midpoint depth fields remain separate approximations.
No physical mode, detector, thermal weight, or Taylor vertex is changed.
"""
from pathlib import Path
import time,mmap
import numpy as np
from scipy import constants, fft
from .free_propagation import TransverseGrid, electron_wavenumber
from .elastic_propagation import electron_speed
from .dynamical_source import full_transverse_vectors
from .thermal_dyson_words import words_through_two
from .two_phonon import SIGNS
from .bulk_thermal_source import mode_group, occupations
from .fourier_depth_fields import complex_depth_field
from .quadratic_fields import quadratic_depth_field
from .leading_two_phonon import leading_pair_trace


def frequency_indices(retained, product):
    if retained % 2 != 1 or product % 2 != 1 or product < retained:
        raise ValueError('odd product grid >= odd retained support required')
    return np.rint(np.fft.fftfreq(retained)*retained).astype(int) % product


def to_product(coefficients, product):
    n = coefficients.shape[-1]
    if coefficients.shape[-2] != n:
        raise ValueError('square retained support required')
    index = frequency_indices(n, product)
    padded = np.zeros((*coefficients.shape[:-2], product, product), complex)
    padded[..., index[:, None], index] = coefficients
    return fft.ifft2(padded, axes=(-2, -1), workers=1)*product**2


def from_product(values, retained):
    product = values.shape[-1]
    index = frequency_indices(retained, product)
    full = fft.fft2(values, axes=(-2, -1), workers=1)/product**2
    return full[..., index[:, None], index]


def pair_structure(omega, shifts):
    words = [w for w in words_through_two(2) if len(w)<2 or w[0]%2 != w[1]%2]
    lookup = {w:i for i,w in enumerate(words)}
    signed_energy = constants.hbar*np.r_[omega, -np.asarray(omega)]
    signed_shift = np.r_[shifts, -np.asarray(shifts)]
    energies = np.array([sum(signed_energy[k] for k in w) for w in words])
    word_shifts = np.array([np.sum(signed_shift[list(w)],axis=0) if w else np.zeros(2) for w in words])
    final = np.array([lookup[(0 if a==1 else 2,1 if b==1 else 3)] for a,b in SIGNS])
    return words,lookup,energies,word_shifts,final


def propagate_galerkin_pair(grid, product_grid, omega, shifts, length, steps,
                            source_at, quadratic_at, background_at,
                            local_substeps=1, limits=None, require_dealiased=True):
    n,m = grid.nx,product_grid.nx
    if grid.shape != (n,n) or product_grid.shape != (m,m):
        raise ValueError('square grids required')
    frequency_indices(n,m)
    if (grid.lx_m,grid.ly_m)!=(product_grid.lx_m,product_grid.ly_m):
        raise ValueError('product and retained grids must have the same physical cell')
    if require_dealiased and m < 2*n-1:
        raise ValueError('exact projected transfer products require M>=2N-1')
    if local_substeps < 1 or int(local_substeps)!=local_substeps:
        raise ValueError('positive integer RK4 substeps required')
    words,lookup,energies,word_shifts,final = pair_structure(omega,shifts)
    w=len(words)
    parents=np.array([lookup[a[1:]] for a in words[1:]])
    letters=np.array([a[0] for a in words[1:]])
    all_energy=np.r_[energies,energies[final]]
    all_shifts=np.r_[word_shifts,word_shifts[final]]
    dz=length/steps;hv=constants.hbar*electron_speed(60000*constants.elementary_charge)
    Q=full_transverse_vectors(grid)[None]+all_shifts[:,None]
    kinetic=np.exp(-.5j*dz*np.sum(Q*Q,axis=-1).reshape(-1,n,n)/
                   (2*electron_wavenumber(60000*constants.elementary_charge)))
    c=np.zeros((w+4,n,n),complex);c[0,0,0]=1.
    scale=-1j*dz/(hv*local_substeps);maximum_phase=0.
    for i in range(steps):
        if limits is not None and i%8==0:limits.check()
        emission=source_at(i)
        sources=np.concatenate((emission,emission.conj()))
        # Fetch once: source providers may perform an FFT or own a mapped view.
        quadratic=np.asarray(quadratic_at(i));background=np.asarray(background_at(i))
        maximum_phase=max(maximum_phase,float(np.max(abs(background))*abs(scale)))
        def action(a):
            real=to_product(a,m)
            value=(background[None]+all_energy[:,None,None])*real
            value[1:w]+=sources[letters]*real[parents]
            value[w:]+=quadratic*real[0]
            return scale*from_product(value,n)
        c*=kinetic
        for j in range(local_substeps):
            k1=action(c);k2=action(c+.5*k1);k3=action(c+.5*k2);k4=action(c+k3)
            c+=(k1+2*k2+2*k3+k4)/6
        c*=kinetic
    sequential=np.array([c[lookup[(0 if a==1 else 2,1 if b==1 else 3)]]+
                         c[lookup[(1 if b==1 else 3,0 if a==1 else 2)]] for a,b in SIGNS])
    return dict(direct_amplitudes=c[w:],sequential_amplitudes=sequential,
                total_amplitudes=c[w:]+sequential,word_coefficients=c[:w],
                final_shifts=word_shifts[final],word_shifts=word_shifts,
                word_energies_j=energies,maximum_local_diagonal_phase=maximum_phase,
                retained_grid=n,product_grid=m,local_substeps=local_substeps,
                antialiasing='project each H action; full transfer support; exact products' if m>=2*n-1 else 'DIAGNOSTIC circular products')


def propagate_bulk_galerkin_pair(model,positions,reference,covariance,grid,product_grid,
                                 length,steps,q_pair,branches,inverse_marginals,
                                 background,scratch,temperature_k=300,limits=None,
                                 local_substeps=1,born=False,save_amplitudes=True,
                                 require_dealiased=True,field_halfwidth=None):
    if np.any(np.asarray(covariance)!=0):
        raise ValueError('this path is explicitly the undisplaced static Taylor family')
    scratch=Path(scratch);scratch.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();maps=[];omega=[];shifts=[];displacements=[]
    m=product_grid.nx;n=grid.nx;timings={}
    try:
        before=time.monotonic()
        for side in range(2):
            om,d,shift,labels=mode_group(model,positions,[grid.lx_m,grid.ly_m],q_pair[side],1.,reference)
            at=int(np.flatnonzero(labels==branches[side])[0])
            omega.append(om[at]);shifts.append(shift);displacements.append(d[:,:,at])
        timings['phonon_seconds']=time.monotonic()-before;before=time.monotonic()
        for side in range(2):
            maps.append(complex_depth_field(positions,covariance,product_grid,length,steps,
                        scratch/f'linear{side}.npy',displacements[side][:,:,None],shifts[side]))
        first=[];second=[];final_shifts=[]
        for a,b in SIGNS:
            first.append(displacements[0].conj() if a==1 else displacements[0])
            second.append(displacements[1].conj() if b==1 else displacements[1])
            final_shifts.append(a*shifts[0]+b*shifts[1])
        maps.append(quadratic_depth_field(positions,covariance,product_grid,length,steps,scratch/'quadratic.npy',
                    np.stack(first,axis=-1),np.stack(second,axis=-1),np.array(final_shifts)))
        timings['field_seconds']=time.monotonic()-before
        freq=np.rint(np.fft.fftfreq(m)*m).astype(int)
        mask=None if field_halfwidth is None else (abs(freq[:,None])<=field_halfwidth)&(abs(freq)<=field_halfwidth)
        def field(i,indices):
            value=np.concatenate([np.asarray(maps[k][i]) for k in indices],axis=0).reshape(-1,m,m)
            if mask is not None:value=value*mask
            result=fft.ifft2(value,axes=(-2,-1),workers=1)*m*m
            if i%16==15:
                for k in indices:maps[k]._mmap.madvise(mmap.MADV_DONTNEED)
            return result
        def bg(i):
            if born:return np.zeros((m,m))
            value=np.asarray(background[i,0])
            if i%16==15 and hasattr(background,'_mmap'):
                background._mmap.madvise(mmap.MADV_DONTNEED)
            if mask is None:return value
            return fft.ifft2(fft.fft2(value,workers=1)*mask,workers=1).real
        before=time.monotonic()
        r=propagate_galerkin_pair(grid,product_grid,np.array(omega),np.array(shifts),length,steps,
              lambda i:field(i,(0,1)),lambda i:field(i,(2,)),bg,local_substeps,limits,require_dealiased)
        timings['propagation_seconds']=time.monotonic()-before
        trace=leading_pair_trace(r['direct_amplitudes'],r['sequential_amplitudes'],np.array(omega),
                  occupations(np.array(omega),temperature_k),grid,r['final_shifts'],inverse_marginals)
        trace.update(omega_rad_s=np.array(omega),emission_shifts_m_inv=np.array(shifts),
                  elapsed_seconds=time.monotonic()-started,mapped_field_page_eviction_steps=16,**timings)
        if save_amplitudes:trace.update(r)
        return trace
    finally:
        for mapped in maps:mapped._mmap.close()
        for name in ('linear0.npy','linear1.npy','quadratic.npy'):(scratch/name).unlink(missing_ok=True)
