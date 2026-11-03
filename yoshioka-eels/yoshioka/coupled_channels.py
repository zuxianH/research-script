"""Hermitian common-velocity paraxial electron/phonon channel propagation.

i hbar v d_z phi = [-(hbar v/2k) laplacian + epsilon_n-epsilon_m+U] phi.
Coordinates and physical Hamiltonians are SI. No carrier, wave normalization,
phase fitting, absorptive channel energy, or extra Debye-Waller factor is added.
Electron arrays are (channel,y,x), with periodic transverse boundaries.
"""

from dataclasses import dataclass
from math import comb

import numpy as np
from scipy.constants import hbar

from .elastic_propagation import electron_speed
from .free_propagation import electron_wavenumber, grid_norm_squared
from .harmonic_oscillator import fock_scattering_matrix, quadrature_scattering_block


def channel_basis(mode_count, cutoff, maximum_channels=100000):
    """Total occupation cutoff. Estimate binomial count before enumeration."""
    if not isinstance(mode_count, int) or mode_count < 1 or not isinstance(cutoff, int) or cutoff < 0:
        raise ValueError('positive mode count and nonnegative integer cutoff required')
    count = comb(mode_count+cutoff, cutoff)
    if count > maximum_channels:
        raise MemoryError(f'requested {count} channels exceeds configured {maximum_channels}')
    # Enumerate compositions, not (cutoff+1)^mode_count candidates.
    def compositions(total, modes):
        if modes == 1:
            yield (total,)
        else:
            for value in range(total+1):
                for remaining in compositions(total-value, modes-1):
                    yield (value,)+remaining
    return np.array([entry for total in range(cutoff+1) for entry in compositions(total, mode_count)], dtype=int)


def oscillator_phase_block(basis, xi, working_dimension=32, quadrature_order=None):
    """Retained multimode <n|exp[-i sum xi_mu X_mu]|m>.

    Working dimension is separate from physical total occupation cutoff. The
    infinite-operator block is extracted from larger single-mode exponentials;
    a finite retained block is not assumed unitary. Quadrature is independent.
    """
    basis = np.asarray(basis)
    xi = np.asarray(xi, dtype=float)
    if xi.shape != (basis.shape[1],) or working_dimension <= np.max(basis):
        raise ValueError('mode count or working dimension inconsistent with basis')
    output = np.ones((len(basis), len(basis)), dtype=np.complex128)
    for mode, coupling in enumerate(xi):
        if quadrature_order is None:
            matrix = fock_scattering_matrix(float(coupling), working_dimension)
        else:
            matrix = quadrature_scattering_block(float(coupling), int(np.max(basis)), quadrature_order)
        occupation = basis[:, mode]
        output *= matrix[occupation[:, None], occupation[None]]
    return output


def linear_displacement_block(basis, coupling):
    """sum_mu coupling_mu(a_mu+a_mu†); every event changes one quantum."""
    basis = np.asarray(basis)
    result = np.zeros((len(basis), len(basis)), dtype=np.complex128)
    lookup = {tuple(n): i for i, n in enumerate(basis)}
    for j, initial in enumerate(basis):
        for mode, scale in enumerate(coupling):
            final = initial.copy()
            final[mode] += 1
            i = lookup.get(tuple(final))
            if i is not None:
                result[i, j] += scale*np.sqrt(final[mode])
                result[j, i] += scale*np.sqrt(final[mode])
    return result


@dataclass
class ChannelSegment:
    length_m: float
    potential_j: np.ndarray


class SplitChannelPropagator:
    """Second/fourth-order unitary splitting, exact local channel exponentials.

    Segment boundaries are exact; no step straddles a discontinuous potential.
    The fourth-order composition uses negative internal substeps for the same
    Hermitian Hamiltonian. It is not an absorbing or reversed physical segment.
    """

    def __init__(self, grid, kinetic_energy_j, channel_energies_j, order=4):
        self.grid = grid
        self.energy = np.asarray(channel_energies_j, dtype=float)
        self.k = electron_wavenumber(kinetic_energy_j)
        self.v = electron_speed(kinetic_energy_j)
        if self.energy.ndim != 1 or not np.all(np.isfinite(self.energy)) or order not in (2, 4):
            raise ValueError('finite channel energy vector and splitting order 2 or 4 required')
        self.order = order
        qx = 2*np.pi*np.fft.fftfreq(grid.nx, grid.dx_m)
        qy = 2*np.pi*np.fft.fftfreq(grid.ny, grid.dy_m)
        self.kinetic_inverse_m = (qy[:, None]**2+qx[None]**2)/(2*self.k)

    def propagate(self, wave, segments, steps_per_segment=32, initial_energy_j=0., g=1., offdiagonal_only_scale=True):
        current = np.asarray(wave, dtype=np.complex128).copy()
        if current.shape != (len(self.energy), *self.grid.shape) or not np.all(np.isfinite(current)):
            raise ValueError('wave must have shape (channel,ny,nx) with finite values')
        if not isinstance(steps_per_segment, int) or steps_per_segment < 1 or not np.isfinite(g):
            raise ValueError('positive integer step count and finite g required')
        history = [float(np.sum(grid_norm_squared(current, self.grid)))]
        hermiticity = []
        composition = (1.,) if self.order == 2 else (1/(2-2**(1/3)), -2**(1/3)/(2-2**(1/3)), 1/(2-2**(1/3)))
        for segment in segments:
            if not np.isfinite(segment.length_m) or segment.length_m <= 0:
                raise ValueError('positive finite segment length required')
            if not np.any(segment.potential_j):
                # Free transverse and channel-energy operators commute. Empty
                # gaps require no dense channel matrices or repeated substeps.
                phase = np.exp(-1j*segment.length_m*(self.energy-initial_energy_j)/(hbar*self.v))
                current = np.fft.ifft2(np.fft.fft2(current, axes=(-2, -1))*
                    np.exp(-1j*segment.length_m*self.kinetic_inverse_m), axes=(-2, -1))*phase[:, None, None]
                hermiticity.append(0.)
                history.append(float(np.sum(grid_norm_squared(current, self.grid))))
                continue
            local = np.broadcast_to(np.asarray(segment.potential_j, dtype=np.complex128), (*self.grid.shape, len(self.energy), len(self.energy))).copy()
            scale = max(float(np.linalg.norm(local)), np.finfo(float).tiny)
            residual = float(np.linalg.norm(local-local.swapaxes(-1, -2).conj())/scale)
            if residual > 1e-12:
                raise ValueError('non-Hermitian channel interaction')
            hermiticity.append(residual)
            diagonal = np.arange(len(self.energy))
            original_diagonal = local[..., diagonal, diagonal].copy()
            local *= g
            if offdiagonal_only_scale:
                local[..., diagonal, diagonal] = original_diagonal
            local[..., diagonal, diagonal] += self.energy-initial_energy_j
            eigenvalues, eigenvectors = np.linalg.eigh(local/(hbar*self.v))
            dt = segment.length_m/steps_per_segment
            operators = []
            for coefficient in composition:
                step = dt*coefficient
                operators.append((np.exp(-.5j*step*self.kinetic_inverse_m), np.exp(-1j*step*eigenvalues)))
            for _ in range(steps_per_segment):
                for kinetic, phase in operators:
                    current = np.fft.ifft2(np.fft.fft2(current, axes=(-2, -1))*kinetic, axes=(-2, -1))
                    values = current.transpose(1, 2, 0)
                    values = np.einsum('...ji,...j->...i', eigenvectors.conj(), values)
                    values *= phase
                    current = np.einsum('...ij,...j->...i', eigenvectors, values).transpose(2, 0, 1)
                    current = np.fft.ifft2(np.fft.fft2(current, axes=(-2, -1))*kinetic, axes=(-2, -1))
            history.append(float(np.sum(grid_norm_squared(current, self.grid))))
        return current, {'norm_history': history, 'max_relative_norm_drift': float(np.max(abs(np.asarray(history)/history[0]-1))),
                         'hermiticity_residuals': hermiticity, 'steps_per_segment': steps_per_segment, 'order': self.order}


def fourier_populations(wave, grid):
    """Absolute probability per electron Fourier mode and orthogonal channel."""
    return grid.lx_m*grid.ly_m/(grid.nx*grid.ny)**2*abs(np.fft.fft2(wave, axes=(-2, -1)))**2


def thermal_channel_weights(basis, mode_quanta_j, temperature_k):
    """Infinite product-oscillator Boltzmann weights retained WITHOUT renormalizing."""
    from scipy.constants import Boltzmann
    basis, quanta = np.asarray(basis), np.asarray(mode_quanta_j)
    if temperature_k < 0 or not np.isfinite(temperature_k) or np.any(quanta <= 0):
        raise ValueError('nonnegative temperature and positive mode quanta required')
    if temperature_k == 0:
        return np.asarray(np.sum(basis, axis=1) == 0, dtype=float)
    ratios = np.exp(-quanta/(Boltzmann*temperature_k))
    return np.exp(basis @ np.log(ratios))*np.prod(1-ratios)
