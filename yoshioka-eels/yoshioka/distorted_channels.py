"""One channel-changing event with diagonal spatial dynamics retained."""

import numpy as np
from scipy.constants import hbar

from .elastic_propagation import electron_speed
from .free_propagation import electron_wavenumber


def distorted_wave_orders(grid, kinetic_energy_j, channel_energies_j, entrance,
                          segments, steps_per_segment=64, initial_energy_j=0.):
    """Return amplitudes of orders g^0 and g^1, without squaring or normalizing.

    K has only off-diagonal channel elements. Each local substep integrates one
    insertion of K analytically between the two diagonal local propagators.
    Fourth-order kinetic/local splitting retains the same Hd as the full solver.
    Independent dense Gauss-Legendre integrals validate this scalable evaluator.
    """
    energies = np.asarray(channel_energies_j)
    current = np.stack((np.asarray(entrance, dtype=np.complex128), np.zeros_like(entrance, dtype=np.complex128)))
    if current.shape[1:] != (len(energies), *grid.shape) or not isinstance(steps_per_segment, int) or steps_per_segment < 1:
        raise ValueError('entrance shape or step count invalid')
    qx = 2*np.pi*np.fft.fftfreq(grid.nx, grid.dx_m)
    qy = 2*np.pi*np.fft.fftfreq(grid.ny, grid.dy_m)
    kinetic = (qy[:, None]**2+qx[None]**2)/(2*electron_wavenumber(kinetic_energy_j))
    hv = hbar*electron_speed(kinetic_energy_j)
    w = 1/(2-2**(1/3))
    for segment in segments:
        if not np.any(segment.potential_j):
            phase = np.exp(-1j*segment.length_m*(energies-initial_energy_j)/hv)
            current = np.fft.ifft2(np.fft.fft2(current, axes=(-2, -1))*np.exp(-1j*segment.length_m*kinetic), axes=(-2, -1))*phase[None, :, None, None]
            continue
        field = np.broadcast_to(segment.potential_j, (*grid.shape, len(energies), len(energies))).copy()/hv
        indices = np.arange(len(energies))
        diagonal = field[..., indices, indices].real+(energies-initial_energy_j)/hv
        field[..., indices, indices] = 0
        operators = []
        for coefficient in (w, 1-2*w, w):
            dz = coefficient*segment.length_m/steps_per_segment
            phase = np.exp(-1j*dz*diagonal)
            average = .5*(diagonal[..., :, None]+diagonal[..., None, :])
            difference = diagonal[..., None, :]-diagonal[..., :, None]
            insertion = -1j*dz*field*np.exp(-1j*dz*average)*np.sinc(dz*difference/(2*np.pi))
            operators.append((np.exp(-.5j*dz*kinetic), phase, insertion))
        for _ in range(steps_per_segment):
            for free, phase, insertion in operators:
                current = np.fft.ifft2(np.fft.fft2(current, axes=(-2, -1))*free, axes=(-2, -1))
                zero = current[0].transpose(1, 2, 0)
                first = current[1].transpose(1, 2, 0)
                current[1] = (phase*first+np.einsum('...ij,...j->...i', insertion, zero)).transpose(2, 0, 1)
                current[0] = (phase*zero).transpose(2, 0, 1)
                current = np.fft.ifft2(np.fft.fft2(current, axes=(-2, -1))*free, axes=(-2, -1))
    return current[0], current[1]
