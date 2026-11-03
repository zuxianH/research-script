"""Contiguous-depth Fourier fields: bounded RSS without a full transpose.

Only one Fourier-field file is stored for a source batch. Its transverse FFT
is evaluated when the propagator requests that depth. The field is never
converted through strided columns of a q-major mapping.
"""
from pathlib import Path
import json,mmap
import numpy as np
from scipy import fft
from .dynamical_source import full_transverse_vectors
from .electron_phonon import LobatoSilicon,mode_plane_coefficients


def fourier_depth_field(positions,covariance,grid,length,steps,target,displacements=None,shift=(0.,0.),chunk=256):
    qxy=full_transverse_vectors(grid)+np.asarray(shift);dz=length/steps;kz=2*np.pi*np.fft.fftfreq(steps,dz)
    planes,inverse=np.unique(positions[:,2],return_inverse=True);axial=np.exp(-1j*(planes[:,None]-dz/2)*kz)
    modes=1 if displacements is None else displacements.shape[-1]
    mapped=np.lib.format.open_memmap(target,mode='w+',dtype=complex,shape=(steps,modes,grid.ny*grid.nx))
    potential=LobatoSilicon.from_abtem_table();area=grid.lx_m*grid.ly_m
    for lo in range(0,len(qxy),chunk):
        hi=min(lo+chunk,len(qxy));Q=qxy[lo:hi]
        exponent=np.einsum('qa,ab,qb->q',Q,covariance[:2,:2],Q)[:,None]+2*(Q@covariance[:2,2])[:,None]*kz+covariance[2,2]*kz*kz
        kernel=potential.fourier_energy_j_m3(np.sum(Q*Q,axis=1)[:,None]+kz*kz)*np.exp(-.5*exponent)/(area*dz)
        if displacements is None:
            coefficients=np.zeros((len(planes),len(Q)),complex)
            for p in range(len(planes)):coefficients[p]=np.sum(np.exp(-1j*positions[inverse==p,:2]@Q.T),axis=0)
            fourier=(coefficients.T@axial)[None]*kernel
        else:
            _,coefficients=mode_plane_coefficients(positions,displacements,Q)
            xy=coefficients[:,:,0]*Q[None,None,:,0]+coefficients[:,:,1]*Q[None,None,:,1]
            fourier=-1j*(xy.transpose(0,2,1)@axial+(coefficients[:,:,2].transpose(0,2,1)@axial)*kz[None,None])*kernel[None]
        values=fft.ifft(fourier,axis=-1,workers=4)
        mapped[:,:,lo:hi]=values.transpose(2,0,1)
        # MAP_SHARED writes remain in the file page cache. This is ephemeral
        # working data, committed only after the complete field is flushed.
        mapped._mmap.madvise(mmap.MADV_DONTNEED)
    mapped.flush();mapped._mmap.madvise(mmap.MADV_DONTNEED)
    return mapped


def complex_depth_field(positions,covariance,grid,length,steps,target,displacements,shift):
    return fourier_depth_field(positions,covariance,grid,length,steps,target,displacements,shift)


def fourier_depth_provider(mapped,grid):
    def provider(i):
        value=fft.ifft2(np.asarray(mapped[i]).reshape(-1,*grid.shape),axes=(-2,-1),workers=4)*(grid.nx*grid.ny)
        if i%32==31:mapped._mmap.madvise(mmap.MADV_DONTNEED)
        return value
    return provider


def depth_field(positions,covariance,grid,length,steps,target,displacements=None):
    if displacements is not None:raise ValueError('real background constructor only; use complex_depth_field for crystal couplings')
    if grid.nx%2==0 or grid.ny%2==0:raise ValueError('odd real background grid required')
    target=Path(target);temporary=target.with_suffix('.fourier.npy')
    mapped=fourier_depth_field(positions,covariance,grid,length,steps,temporary)
    result=np.lib.format.open_memmap(target,mode='w+',dtype=float,shape=(steps,1,*grid.shape))
    iy,ix=np.indices(grid.shape);opposite=(((-iy)%grid.ny)*grid.nx+(-ix)%grid.nx).ravel()
    selected=np.flatnonzero(np.arange(len(opposite))<=opposite);error=scale=0.;maximum=0.
    for i in range(steps):
        raw=np.asarray(mapped[i]);a=raw[:,selected];b=raw[:,opposite[selected]]
        error+=float(np.sum(abs(a-b.conj())**2));scale+=float(np.sum(abs(a)**2+abs(b)**2))
        projected=raw.copy();pair=.5*(a+b.conj());projected[:,selected]=pair;projected[:,opposite[selected]]=pair.conj()
        value=fft.ifft2(projected.reshape(1,*grid.shape),axes=(-2,-1),workers=4)*(grid.nx*grid.ny)
        imaginary=float(np.linalg.norm(value.imag)/max(np.linalg.norm(value),np.finfo(float).tiny))
        if imaginary>1e-10:raise ValueError('real potential slice failed reality check')
        maximum=max(maximum,imaginary);result[i]=value.real
        if i%32==31:
            result.flush();result._mmap.madvise(mmap.MADV_DONTNEED);mapped._mmap.madvise(mmap.MADV_DONTNEED)
    residual=float(np.sqrt(error/max(scale,np.finfo(float).tiny)))
    if residual>1e-12:raise ValueError('unprojected full background failed Hermiticity')
    result.flush();result._mmap.madvise(mmap.MADV_DONTNEED);del mapped;temporary.unlink()
    diagnostic={'global_relative_anti_Hermitian_error_before_projection':residual,'bound':1e-12,
        'maximum_imaginary_residual':maximum,'layout':'depth-major Fourier construction; streamed pair check before use','FFT_workers':4}
    Path(str(target)+'.reality.json').write_text(json.dumps(diagnostic)+'\n')
    return result,diagnostic
