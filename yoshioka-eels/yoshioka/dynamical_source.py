"""Elastic incoming/outgoing propagation with one linear phonon source.

This is an undepleted distorted-wave approximation, not a closed full-channel
solver. Both electron waves see the same real DW-averaged finite-atom potential.
The source is the Cartesian derivative of that potential contracted with a
declared real periodic normal mode. All quantities are SI.
"""

from pathlib import Path
import mmap
import json

import numpy as np
from scipy.constants import hbar

from .electron_phonon import LobatoSilicon, mode_plane_coefficients
from .elastic_propagation import electron_speed
from .free_propagation import electron_wavenumber


def full_transverse_vectors(grid):
    qx, qy = np.meshgrid(2*np.pi*np.fft.fftfreq(grid.nx, grid.dx_m), 2*np.pi*np.fft.fftfreq(grid.ny, grid.dy_m))
    return np.column_stack((qx.ravel(), qy.ravel()))


def plane_field(positions_m, covariance_m2, grid, length_m, steps, target,
                displacements_m=None, chunk_size=64):
    """Finite-atom elastic field or mode sources, computed on slice midpoints.

    Odd transverse/longitudinal grids avoid unpaired Nyquist coefficients. The
    Fourier potential is periodized only over the explicit vacuum-padded length;
    longitudinal padding is a separate refinement. Source blocks are mapped to
    disk. No Born final-amplitude formula is used in their construction.
    """
    if grid.nx % 2 == 0 or grid.ny % 2 == 0 or steps % 2 == 0:
        raise ValueError('this real-field source route requires odd grid and slice counts')
    potential = LobatoSilicon.from_abtem_table()
    if displacements_m is not None and np.iscomplexobj(displacements_m) and np.linalg.norm(displacements_m.imag) > 1e-13*np.linalg.norm(displacements_m):
        raise ValueError('this source route requires declared real normal modes; complex Bloch modes require a separate boundary implementation')
    qxy = full_transverse_vectors(grid)
    dz = length_m/steps
    kz = 2*np.pi*np.fft.fftfreq(steps, dz)
    planes, inverse = np.unique(positions_m[:, 2], return_inverse=True)
    # Evaluate at (j+1/2) dz, with the displacement convention used by B6.
    axial = np.exp(-1j*(planes[:, None]-dz/2)*kz[None])
    modes = 1 if displacements_m is None else displacements_m.shape[-1]
    values = np.lib.format.open_memmap(target, mode='w+', dtype=np.complex128, shape=(modes, len(qxy), steps))
    area = grid.lx_m*grid.ly_m
    for start in range(0, len(qxy), chunk_size):
        stop = min(start+chunk_size, len(qxy))
        transverse = qxy[start:stop]
        squared = np.sum(transverse**2, axis=1)[:, None]+kz[None]**2
        exponent = (np.einsum('qa,ab,qb->q', transverse, covariance_m2[:2, :2], transverse)[:, None]
                    + 2*np.einsum('qa,a->q', transverse, covariance_m2[:2, 2])[:, None]*kz[None]
                    + covariance_m2[2, 2]*kz[None]**2)
        kernel = potential.fourier_energy_j_m3(squared)*np.exp(-.5*exponent)/(area*dz)
        if displacements_m is None:
            coefficients = np.zeros((len(planes), len(transverse)), dtype=np.complex128)
            for plane in range(len(planes)):
                coefficients[plane] = np.sum(np.exp(-1j*positions_m[inverse == plane, :2] @ transverse.T), axis=0)
            fourier = (coefficients.T @ axial)[None]*kernel
        else:
            _, coefficients = mode_plane_coefficients(positions_m, displacements_m, transverse)
            xy = coefficients[:, :, 0]*transverse[None, None, :, 0]+coefficients[:, :, 1]*transverse[None, None, :, 1]
            fourier = -1j*(np.einsum('mpq,pk->mqk', xy, axial)+
                           np.einsum('mpq,pk->mqk', coefficients[:, :, 2], axial)*kz[None, None])*kernel[None]
        values[:, start:stop] = np.fft.ifft(fourier, axis=-1)
        values.flush()
        values._mmap.madvise(mmap.MADV_DONTNEED)
    # Each declared real normal coordinate has a Hermitian Fourier field. FFT
    # roundoff can dominate *relative* reality checks in an almost-zero vacuum
    # slice. First bound the global anti-Hermitian error, then project only that
    # roundoff onto exact conjugate pairs; no physical complex source is dropped.
    iy, ix = np.indices(grid.shape)
    opposite = (((-iy) % grid.ny)*grid.nx+(-ix) % grid.nx).ravel()
    indices = np.flatnonzero(np.arange(len(opposite)) <= opposite)
    error_squared, scale_squared = 0., 0.
    for start in range(0, len(indices), chunk_size):
        first = indices[start:start+chunk_size]
        second = opposite[first]
        a, b = np.asarray(values[:, first]), np.asarray(values[:, second])
        error_squared += float(np.sum(abs(a-b.conj())**2))
        scale_squared += float(np.sum(abs(a)**2+abs(b)**2))
    global_error = np.sqrt(error_squared/max(scale_squared, np.finfo(float).tiny))
    if global_error > 1e-12:
        raise ValueError('real-mode Fourier field fails global Hermiticity before projection')
    for start in range(0, len(indices), chunk_size):
        first = indices[start:start+chunk_size]
        second = opposite[first]
        projected = .5*(np.asarray(values[:, first])+np.asarray(values[:, second]).conj())
        values[:, first] = projected
        values[:, second] = projected.conj()
        values.flush()
        values._mmap.madvise(mmap.MADV_DONTNEED)
    Path(str(target)+'.reality.json').write_text(json.dumps({'global_relative_anti_Hermitian_error_before_projection': float(global_error),
        'bound': 1e-12, 'operation': 'conjugate-pair projection of bounded floating-point roundoff for declared real modes'})+'\n')
    return values


def real_slice(mapped_fourier, index, grid, allow_complex=False):
    values = np.fft.ifft2(np.asarray(mapped_fourier[:, :, index]).reshape(-1, *grid.shape), axes=(-2, -1))*(grid.nx*grid.ny)
    residual = float(np.linalg.norm(values.imag)/max(np.linalg.norm(values), np.finfo(float).tiny))
    if not allow_complex and residual > 1e-10:
        raise ValueError(f'expected Hermitian real normal-mode field, imaginary residual {residual:g}')
    return values if allow_complex else values.real, residual


def propagate_incident(grid, energy_j, background_fourier, length_m, history_path, elastic=True):
    steps = background_fourier.shape[-1]
    dz = length_m/steps
    qxy = full_transverse_vectors(grid)
    kinetic = np.exp(-.5j*dz*np.sum(qxy**2, axis=-1).reshape(grid.shape)/(2*electron_wavenumber(energy_j)))
    history = np.lib.format.open_memmap(history_path, mode='w+', dtype=np.complex128, shape=(steps, *grid.shape))
    wave = np.ones(grid.shape, dtype=np.complex128)/np.sqrt(grid.lx_m*grid.ly_m)
    norm_initial = float(np.sum(abs(wave)**2)*grid.dx_m*grid.dy_m)
    maximum_drift = 0.
    maximum_imaginary = 0.
    for index in range(steps):
        wave = np.fft.ifft2(np.fft.fft2(wave)*kinetic)
        history[index] = wave
        field, imaginary = real_slice(background_fourier, index, grid)
        maximum_imaginary = max(maximum_imaginary, imaginary)
        if elastic:
            wave *= np.exp(-1j*field[0]*dz/(hbar*electron_speed(energy_j)))
        wave = np.fft.ifft2(np.fft.fft2(wave)*kinetic)
        maximum_drift = max(maximum_drift, abs(float(np.sum(abs(wave)**2)*grid.dx_m*grid.dy_m)/norm_initial-1))
        if (index+1) % 32 == 0:
            history.flush()
            history._mmap.madvise(mmap.MADV_DONTNEED)
            background_fourier._mmap.madvise(mmap.MADV_DONTNEED)
    history.flush()
    return wave, history, {'max_incident_norm_drift': maximum_drift, 'maximum_relative_imaginary_background': maximum_imaginary}


def propagate_source_block(grid, energy_j, omega_rad_s, background, sources, incident_before_local,
                           length_m, elastic=True):
    """One insertion with elastic diagonal propagation on both sides."""
    omega = np.asarray(omega_rad_s)
    steps = sources.shape[-1]
    dz = length_m/steps
    qxy = full_transverse_vectors(grid)
    kinetic = np.exp(-.5j*dz*np.sum(qxy**2, axis=-1).reshape(grid.shape)/(2*electron_wavenumber(energy_j)))
    velocity = electron_speed(energy_j)
    wave = np.zeros((len(omega), *grid.shape), dtype=np.complex128)
    channel_phase = np.exp(-1j*omega*dz/velocity)
    insertion = -1j*dz/(hbar*velocity)*np.exp(-.5j*omega*dz/velocity)*np.sinc(omega*dz/(2*np.pi*velocity))
    maximum_imaginary = 0.
    for index in range(steps):
        wave = np.fft.ifft2(np.fft.fft2(wave, axes=(-2, -1))*kinetic, axes=(-2, -1))
        coupling, imaginary = real_slice(sources, index, grid)
        maximum_imaginary = max(maximum_imaginary, imaginary)
        if elastic:
            field, _ = real_slice(background, index, grid)
            phase = np.exp(-1j*field[0]*dz/(hbar*velocity))
        else:
            phase = 1.
        wave = phase*(channel_phase[:, None, None]*wave+
            insertion[:, None, None]*coupling*incident_before_local[index][None])
        wave = np.fft.ifft2(np.fft.fft2(wave, axes=(-2, -1))*kinetic, axes=(-2, -1))
        if (index+1) % 32 == 0:
            sources._mmap.madvise(mmap.MADV_DONTNEED)
            background._mmap.madvise(mmap.MADV_DONTNEED)
            incident_before_local._mmap.madvise(mmap.MADV_DONTNEED)
    amplitudes = np.fft.fft2(wave, axes=(-2, -1))*np.sqrt(grid.lx_m*grid.ly_m)/(grid.nx*grid.ny)
    return amplitudes, {'maximum_relative_imaginary_source': maximum_imaginary}
