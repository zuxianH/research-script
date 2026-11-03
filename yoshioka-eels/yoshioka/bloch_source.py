"""Exact unit-cell/Bloch sectors of the same finite periodic phonon bath.

The physical transverse specimen is periodic. A unit-cell electron wave in
each Bloch sector is an equivalent representation of the repeated cell, not a
different specimen or a projection of leaked numerical intensity.
"""
from pathlib import Path
import mmap
import numpy as np
from scipy import constants

from .dynamical_source import full_transverse_vectors
from .electron_phonon import LobatoSilicon,mode_plane_coefficients
from .elastic_propagation import electron_speed
from .free_propagation import electron_wavenumber


def bath_groups(model,repetitions,thickness_repeats):
    repetitions=np.asarray(repetitions,dtype=int);count=int(np.prod(repetitions))
    cells=np.column_stack((np.zeros(thickness_repeats),np.zeros(thickness_repeats),np.arange(thickness_repeats)))
    positions=((cells@model.cell_m)[:,None]+model.positions_m[None]).reshape(-1,3)
    basis=np.tile(np.arange(len(model.positions_m)),thickness_repeats);groups=[]
    for indices in np.array(list(np.ndindex(*repetitions))):
        q=indices/repetitions;omega,vectors=model.modes(q);physical=np.ones(len(omega),bool)
        if not np.any(indices):
            t=np.tile(np.eye(3),(len(model.positions_m),1))*np.repeat(np.sqrt(model.masses_kg),3)[:,None];t/=np.linalg.norm(t,axis=0)
            physical[np.argsort(np.sum(abs(vectors.conj().T@t)**2,axis=1))[-3:]]=False
        if np.any(omega[physical]<=0):raise ValueError('unstable physical Bloch mode')
        omega=omega[physical];vectors=vectors[:,physical].reshape(len(model.positions_m),3,-1)[basis]
        d=vectors*np.sqrt(constants.hbar/(2*model.masses_kg[basis,None,None]*omega[None,None]*count))
        d*=np.exp(1j*positions@(q@model.reciprocal_m_inv))[:,None,None]
        shift=-(q@model.reciprocal_m_inv)[:2]
        groups.append({'q_fractional':q,'omega_rad_s':omega,'displacements_m':d,'bloch_shift_m_inv':shift,'branches':np.flatnonzero(physical)})
    return positions,groups


def complex_depth_field(positions,covariance,grid,length,steps,target,displacements,shift,chunk=256):
    """Periodic part of G_q(r,z), with physical Q_perp=G-q_perp.

    G_lambda0 need not be real. Its adjoint is G_0lambda. No real-field
    projection is applied to a complex crystal-channel matrix element.
    """
    qxy=full_transverse_vectors(grid)+np.asarray(shift);dz=length/steps;kz=2*np.pi*np.fft.fftfreq(steps,dz)
    planes=np.unique(positions[:,2]);axial=np.exp(-1j*(planes[:,None]-dz/2)*kz)
    target=Path(target);temporary=target.with_suffix('.fourier.npy');modes=displacements.shape[-1]
    fourier_field=np.lib.format.open_memmap(temporary,mode='w+',dtype=complex,shape=(modes,len(qxy),steps))
    potential=LobatoSilicon.from_abtem_table();area=grid.lx_m*grid.ly_m
    for lo in range(0,len(qxy),chunk):
        hi=min(lo+chunk,len(qxy));Q=qxy[lo:hi]
        exponent=np.einsum('qa,ab,qb->q',Q,covariance[:2,:2],Q)[:,None]+2*(Q@covariance[:2,2])[:,None]*kz+covariance[2,2]*kz*kz
        kernel=potential.fourier_energy_j_m3(np.sum(Q*Q,axis=1)[:,None]+kz*kz)*np.exp(-.5*exponent)/(area*dz)
        _,coefficients=mode_plane_coefficients(positions,displacements,Q)
        xy=coefficients[:,:,0]*Q[None,None,:,0]+coefficients[:,:,1]*Q[None,None,:,1]
        transverse=xy.transpose(0,2,1)@axial;longitudinal=coefficients[:,:,2].transpose(0,2,1)@axial
        fourier_field[:,lo:hi]=np.fft.ifft(-1j*(transverse+longitudinal*kz[None,None])*kernel[None],axis=-1)
        fourier_field.flush();fourier_field._mmap.madvise(mmap.MADV_DONTNEED)
    depth=np.lib.format.open_memmap(target,mode='w+',dtype=complex,shape=(steps,modes,*grid.shape))
    for i in range(steps):
        depth[i]=np.fft.ifft2(np.asarray(fourier_field[:,:,i]).reshape(modes,*grid.shape),axes=(-2,-1))*(grid.nx*grid.ny)
        if i%128==127:depth.flush();depth._mmap.madvise(mmap.MADV_DONTNEED)
    depth.flush();depth._mmap.madvise(mmap.MADV_DONTNEED);del fourier_field;temporary.unlink()
    return depth


def source_exit_bloch(grid,energy,omega,length,steps,source_at,incident_at,background_at,shift):
    omega=np.asarray(omega);dz=length/steps;velocity=electron_speed(energy)
    Q=full_transverse_vectors(grid)+np.asarray(shift)
    kinetic=np.exp(-.5j*dz*np.sum(Q*Q,axis=1).reshape(grid.shape)/(2*electron_wavenumber(energy)))
    channel=np.exp(-1j*omega*dz/velocity)[:,None,None]
    insertion=(-1j*dz/(constants.hbar*velocity)*np.exp(-.5j*omega*dz/velocity)*np.sinc(omega*dz/(2*np.pi*velocity)))[:,None,None]
    wave=np.zeros((len(omega),*grid.shape),complex)
    for i in range(steps):
        wave=np.fft.ifft2(np.fft.fft2(wave,axes=(-2,-1))*kinetic,axes=(-2,-1))
        wave=channel*wave+insertion*source_at(i)*incident_at(i)[None]
        if background_at is not None:wave*=np.exp(-1j*background_at(i)*dz/(constants.hbar*velocity))
        wave=np.fft.ifft2(np.fft.fft2(wave,axes=(-2,-1))*kinetic,axes=(-2,-1))
    return np.fft.fft2(wave,axes=(-2,-1))*np.sqrt(grid.lx_m*grid.ly_m)/(grid.nx*grid.ny)
