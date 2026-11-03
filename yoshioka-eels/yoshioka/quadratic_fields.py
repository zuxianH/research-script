"""Quadratic displaced-atom fields, with explicit Cartesian/Fourier signs.

h(r-R-u) -> h_tilde(Q) exp(-i Q.R) [1 - i Q.u - (Q.u)^2/2].
Displacements have shape (atom, Cartesian, field); an emission uses d*, as
in mode_plane_coefficients. The caller supplies the signed displacements
already conjugated. ``multipliers`` is one for a distinct-mode coherent
cross vertex and one half for an identical-mode ordered monomial.
All Q, including the longitudinal FFT coordinate, are angular wavevectors.
"""
from pathlib import Path
import mmap
import numpy as np
from scipy import fft
from .dynamical_source import full_transverse_vectors
from .electron_phonon import LobatoSilicon


def quadratic_depth_field(positions, covariance, grid, length, steps, target,
                          first, second, shifts, multipliers=None, chunk=128, workers=1):
    positions = np.asarray(positions)
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape or first.ndim != 3 or first.shape[:2] != positions.shape:
        raise ValueError('quadratic displacements require (atoms,3,fields) arrays')
    count = first.shape[-1]
    shifts = np.asarray(shifts)
    if shifts.shape != (count, 2):
        raise ValueError('one final electron Bloch shift per quadratic field')
    multipliers = np.ones(count) if multipliers is None else np.broadcast_to(multipliers, (count,))
    dz = length/steps
    kz = 2*np.pi*np.fft.fftfreq(steps, dz)
    planes, inverse = np.unique(positions[:, 2], return_inverse=True)
    axial = np.exp(-1j*(planes[:, None]-dz/2)*kz)
    potential = LobatoSilicon.from_abtem_table()
    base = full_transverse_vectors(grid)
    mapped = np.lib.format.open_memmap(Path(target), mode='w+', dtype=complex,
                                      shape=(steps, count, grid.nx*grid.ny))
    area = grid.lx_m*grid.ly_m
    for field in range(count):
        a, b = first[:, :, field], second[:, :, field]
        for lo in range(0, len(base), chunk):
            hi = min(lo+chunk, len(base)); Q = base[lo:hi]+shifts[field]
            exponent = np.einsum('qa,ab,qb->q', Q, covariance[:2, :2], Q)[:, None]
            exponent = exponent+2*(Q@covariance[:2, 2])[:, None]*kz+covariance[2, 2]*kz*kz
            kernel = potential.fourier_energy_j_m3(np.sum(Q*Q, axis=1)[:, None]+kz*kz)
            kernel *= np.exp(-.5*exponent)/(area*dz)
            c0 = np.zeros((len(planes), len(Q)), complex)
            c1 = np.zeros_like(c0); c2 = np.zeros_like(c0)
            for plane in range(len(planes)):
                selected = inverse == plane
                phase = np.exp(-1j*positions[selected, :2]@Q.T)
                qa = a[selected, :2]@Q.T; qb = b[selected, :2]@Q.T
                az, bz = a[selected, 2, None], b[selected, 2, None]
                c0[plane] = np.sum(phase*qa*qb, axis=0)
                c1[plane] = np.sum(phase*(qa*bz+az*qb), axis=0)
                c2[plane] = np.sum(phase*az*bz, axis=0)
            fourier = -multipliers[field]*(c0.T@axial+(c1.T@axial)*kz+(c2.T@axial)*kz*kz)*kernel
            mapped[:, field, lo:hi] = fft.ifft(fourier, axis=-1, workers=workers).T
            mapped._mmap.madvise(mmap.MADV_DONTNEED)
    mapped.flush(); mapped._mmap.madvise(mmap.MADV_DONTNEED)
    return mapped
