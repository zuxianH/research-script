"""Same atomic Fourier source, with BLAS contraction of longitudinal planes.

This changes summation implementation and work batching only. The preserved
B9A source builder remains the independent small-problem reference.
"""
from pathlib import Path
import json,mmap
import numpy as np
from .dynamical_source import full_transverse_vectors
from .electron_phonon import LobatoSilicon,mode_plane_coefficients
from .depth_fields import convert_depth_field,depth_provider,incident_depth


def plane_field_gemm(positions,covariance,grid,length,steps,target,displacements=None,chunk=512):
    if grid.nx%2==0 or grid.ny%2==0 or steps%2==0:raise ValueError('odd real-field grid required')
    if displacements is not None and np.iscomplexobj(displacements) and np.linalg.norm(displacements.imag)>1e-13*np.linalg.norm(displacements):
        raise ValueError('real finite normal modes required')
    qxy=full_transverse_vectors(grid);dz=length/steps;kz=2*np.pi*np.fft.fftfreq(steps,dz)
    planes,inverse=np.unique(positions[:,2],return_inverse=True)
    axial=np.exp(-1j*(planes[:,None]-dz/2)*kz[None]);modes=1 if displacements is None else displacements.shape[-1]
    values=np.lib.format.open_memmap(target,mode='w+',dtype=complex,shape=(modes,len(qxy),steps))
    potential=LobatoSilicon.from_abtem_table();area=grid.lx_m*grid.ly_m
    for lo in range(0,len(qxy),chunk):
        hi=min(lo+chunk,len(qxy));Q=qxy[lo:hi]
        exponent=np.einsum('qa,ab,qb->q',Q,covariance[:2,:2],Q)[:,None]+2*(Q@covariance[:2,2])[:,None]*kz+covariance[2,2]*kz*kz
        kernel=potential.fourier_energy_j_m3(np.sum(Q*Q,axis=1)[:,None]+kz*kz)*np.exp(-.5*exponent)/(area*dz)
        if displacements is None:
            coefficients=np.zeros((len(planes),len(Q)),complex)
            for i in range(len(planes)):coefficients[i]=np.sum(np.exp(-1j*positions[inverse==i,:2]@Q.T),axis=0)
            fourier=(coefficients.T@axial)[None]*kernel
        else:
            _,coefficients=mode_plane_coefficients(positions,displacements,Q)
            xy=coefficients[:,:,0]*Q[None,None,:,0]+coefficients[:,:,1]*Q[None,None,:,1]
            # m,q,p @ p,k; BLAS replaces the original explicit p reduction.
            transverse=xy.transpose(0,2,1)@axial
            longitudinal=coefficients[:,:,2].transpose(0,2,1)@axial
            fourier=-1j*(transverse+longitudinal*kz[None,None])*kernel[None]
        values[:,lo:hi]=np.fft.ifft(fourier,axis=-1);values.flush();values._mmap.madvise(mmap.MADV_DONTNEED)
    iy,ix=np.indices(grid.shape);opposite=(((-iy)%grid.ny)*grid.nx+(-ix)%grid.nx).ravel()
    indices=np.flatnonzero(np.arange(len(opposite))<=opposite);error=scale=0.
    for lo in range(0,len(indices),chunk):
        first=indices[lo:lo+chunk];second=opposite[first];a=np.asarray(values[:,first]);b=np.asarray(values[:,second])
        error+=float(np.sum(abs(a-b.conj())**2));scale+=float(np.sum(abs(a)**2+abs(b)**2))
    residual=np.sqrt(error/max(scale,np.finfo(float).tiny))
    if residual>1e-12:raise ValueError('unprojected source failed Hermiticity')
    for lo in range(0,len(indices),chunk):
        first=indices[lo:lo+chunk];second=opposite[first]
        projected=.5*(np.asarray(values[:,first])+np.asarray(values[:,second]).conj())
        values[:,first]=projected;values[:,second]=projected.conj();values.flush();values._mmap.madvise(mmap.MADV_DONTNEED)
    Path(str(target)+'.reality.json').write_text(json.dumps({'global_relative_anti_Hermitian_error_before_projection':float(residual),'bound':1e-12,'operation':'same conjugate-pair roundoff projection'})+'\n')
    return values


def depth_field(positions,covariance,grid,length,steps,target,displacements=None):
    target=Path(target);temporary=target.with_suffix('.fourier.npy')
    fourier=plane_field_gemm(positions,covariance,grid,length,steps,temporary,displacements)
    result,diagnostic=convert_depth_field(fourier,grid,target);del fourier
    temporary.unlink();Path(str(temporary)+'.reality.json').unlink(missing_ok=True)
    return result,diagnostic
