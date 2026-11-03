"""Independent final-channel source propagation, with bounded live state.

Only (batch,ny,nx) outgoing waves are live. ``source_at`` and ``incident_at``
are slice providers; they can read checkpoints or recompute slices. A provider
must retain the *amplitude* at a given depth, not an intensity. Distinct final
channels are traced only after coherent depth integration has finished.
"""
import numpy as np
from scipy.constants import hbar

from .dynamical_source import full_transverse_vectors
from .elastic_propagation import electron_speed
from .free_propagation import electron_wavenumber


def streamed_source_exit(grid, energy_j, signed_omega_rad_s, length_m, steps,
                         source_at, incident_at, background_at=None):
    """One exact local insertion in the same Strang convention as B9A.

    Diagonal H_lambda = T_perp + U_DW + hbar*omega_lambda; loss omega>0,
    gain omega<0. Incident H_0 = T_perp+U_DW, with common real velocity.
    No final channels couple in this approximation. No depletion is imposed.
    """
    omega = np.asarray(signed_omega_rad_s, dtype=float)
    if omega.ndim != 1 or not np.all(np.isfinite(omega)) or steps < 1 or length_m <= 0:
        raise ValueError('finite signed frequencies, positive length and steps required')
    dz = length_m/steps
    velocity = electron_speed(energy_j)
    q = full_transverse_vectors(grid)
    kinetic = np.exp(-.5j*dz*np.sum(q*q, axis=1).reshape(grid.shape)/(2*electron_wavenumber(energy_j)))
    channel = np.exp(-1j*omega*dz/velocity)[:, None, None]
    insertion = (-1j*dz/(hbar*velocity)*np.exp(-.5j*omega*dz/velocity)*
                 np.sinc(omega*dz/(2*np.pi*velocity)))[:, None, None]
    wave = np.zeros((len(omega), *grid.shape), dtype=np.complex128)
    for index in range(steps):
        wave = np.fft.ifft2(np.fft.fft2(wave, axes=(-2, -1))*kinetic, axes=(-2, -1))
        source = np.asarray(source_at(index))
        incident = np.asarray(incident_at(index))
        if source.shape != wave.shape or incident.shape != grid.shape:
            raise ValueError('slice provider returned an incompatible shape')
        wave = channel*wave+insertion*source*incident[None]
        if background_at is not None:
            wave *= np.exp(-1j*np.asarray(background_at(index))*dz/(hbar*velocity))
        wave = np.fft.ifft2(np.fft.fft2(wave, axes=(-2, -1))*kinetic, axes=(-2, -1))
    return np.fft.fft2(wave, axes=(-2, -1))*np.sqrt(grid.lx_m*grid.ly_m)/(grid.nx*grid.ny)


def mapped_slice_provider(mapped, grid):
    """Read the Fourier-field mapping and discard old resident pages promptly."""
    from .dynamical_source import real_slice
    import mmap
    def provider(index):
        value, _ = real_slice(mapped, index, grid)
        if index % 16 == 15:
            mapped._mmap.madvise(mmap.MADV_DONTNEED)
        return value
    return provider
