"""Depth-contiguous, real, checked field checkpoints for source batching.

This changes storage order only. Fourier construction is the independent B9A
atomic-potential implementation, including its bounded Hermitian-pair check.
"""
from pathlib import Path
import mmap
import numpy as np
from scipy.constants import hbar

from .dynamical_source import real_slice,full_transverse_vectors,plane_field
from .elastic_propagation import electron_speed
from .free_propagation import electron_wavenumber


def convert_depth_field(fourier,grid,target):
    steps=fourier.shape[-1];modes=fourier.shape[0]
    result=np.lib.format.open_memmap(target,mode='w+',dtype=np.float64,shape=(steps,modes,*grid.shape))
    maximum=0.
    # Keep the one Fourier batch resident during conversion: throwing away its
    # pages at every few strided columns causes repeated disk reads.
    for i in range(steps):
        value,residual=real_slice(fourier,i,grid)
        maximum=max(maximum,residual);result[i]=value
        if i%128==127:
            result.flush();result._mmap.madvise(mmap.MADV_DONTNEED)
    result.flush();result._mmap.madvise(mmap.MADV_DONTNEED)
    fourier._mmap.madvise(mmap.MADV_DONTNEED)
    return result,{'maximum_imaginary_residual':maximum,'operation':'inverse transverse FFT and storage transpose; no interpolation/filtering'}


def depth_field(positions,covariance,grid,length,steps,target,displacements=None):
    target=Path(target);temporary=target.with_suffix('.fourier.npy')
    fourier=plane_field(positions,covariance,grid,length,steps,temporary,displacements)
    result,diagnostic=convert_depth_field(fourier,grid,target)
    del fourier
    temporary.unlink();Path(str(temporary)+'.reality.json').unlink(missing_ok=True)
    return result,diagnostic


def depth_provider(mapped):
    def provider(i):
        value=np.asarray(mapped[i]).copy()
        if i%32==31:mapped._mmap.madvise(mmap.MADV_DONTNEED)
        return value
    return provider


def incident_depth(grid,energy,background,length,target,elastic=True):
    steps=len(background);dz=length/steps;q=full_transverse_vectors(grid)
    kinetic=np.exp(-.5j*dz*np.sum(q*q,axis=1).reshape(grid.shape)/(2*electron_wavenumber(energy)))
    history=np.lib.format.open_memmap(target,mode='w+',dtype=np.complex128,shape=(steps,*grid.shape))
    wave=np.ones(grid.shape,dtype=np.complex128)/np.sqrt(grid.lx_m*grid.ly_m)
    at=depth_provider(background);drift=0.
    for i in range(steps):
        wave=np.fft.ifft2(np.fft.fft2(wave)*kinetic);history[i]=wave
        if elastic:wave*=np.exp(-1j*at(i)[0]*dz/(hbar*electron_speed(energy)))
        wave=np.fft.ifft2(np.fft.fft2(wave)*kinetic)
        drift=max(drift,abs(float(np.sum(abs(wave)**2)*grid.dx_m*grid.dy_m)-1))
        if i%32==31:history.flush();history._mmap.madvise(mmap.MADV_DONTNEED)
    history.flush();history._mmap.madvise(mmap.MADV_DONTNEED)
    return wave,history,{'max_incident_norm_drift':drift}
