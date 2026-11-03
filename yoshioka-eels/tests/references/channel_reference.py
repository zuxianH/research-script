"""Independent dense reference and distorted-wave first-order integral for coupled channels."""

import numpy as np
from scipy.constants import hbar
from scipy.linalg import block_diag, expm
from scipy.special import roots_legendre

from yoshioka.elastic_propagation import electron_speed
from yoshioka.free_propagation import electron_wavenumber


def dense_generators(grid, kinetic_energy_j, energies_j, segments, initial_energy_j=0.):
    """Position-first/channel-last dense matrices using explicit Fourier vectors.

    No production splitter, FFT propagator, matrix exponential cache or local
    eigensystem is used. Limit this deliberately to small independent controls.
    """
    count = grid.nx*grid.ny
    channels = len(energies_j)
    if count*channels > 1200:
        raise MemoryError('independent dense reference capped at dimension 1200')
    x, y = np.meshgrid(np.arange(grid.nx)*grid.dx_m, np.arange(grid.ny)*grid.dy_m)
    fx, fy = np.meshgrid(np.fft.fftfreq(grid.nx, grid.dx_m), np.fft.fftfreq(grid.ny, grid.dy_m))
    plane = np.exp(2j*np.pi*(x.ravel()[:, None]*fx.ravel()[None]+y.ravel()[:, None]*fy.ravel()[None]))/np.sqrt(count)
    dispersion = (2*np.pi)**2*(fx.ravel()**2+fy.ravel()**2)/(2*electron_wavenumber(kinetic_energy_j))
    transverse = (plane*dispersion[None]) @ plane.conj().T
    kinetic = np.kron(transverse, np.eye(channels))
    result = []
    for segment in segments:
        field = np.broadcast_to(segment.potential_j, (*grid.shape, channels, channels)).reshape(count, channels, channels)
        interaction = block_diag(*field)/(hbar*electron_speed(kinetic_energy_j))
        channel = np.tile((np.asarray(energies_j)-initial_energy_j)/(hbar*electron_speed(kinetic_energy_j)), count)
        full = kinetic+interaction+np.diag(channel)
        diagonal_field = np.diag(np.diag(interaction))
        diagonal = kinetic+diagonal_field+np.diag(channel)
        result.append((segment.length_m, diagonal, interaction-diagonal_field, full))
    return result


def dense_propagate(wave, generators, g=1.):
    current = wave.transpose(1, 2, 0).ravel().copy()
    for length, diagonal, offdiagonal, _ in generators:
        current = expm(-1j*length*(diagonal+g*offdiagonal)) @ current
    return current.reshape(*wave.shape[1:], wave.shape[0]).transpose(2, 0, 1)


def distorted_first_order(wave, generators, quadrature_order=48):
    """Ud plus explicit Gauss-Legendre integral of one off-diagonal interaction."""
    nodes, weights = roots_legendre(quadrature_order)
    eigensystems = [np.linalg.eigh(diagonal) for _, diagonal, _, _ in generators]
    def evolution(i, distance, vector):
        values, vectors = eigensystems[i]
        return vectors @ (np.exp(-1j*values*distance)*(vectors.conj().T @ vector))
    entrance = wave.transpose(1, 2, 0).ravel().copy()
    zero, first = entrance.copy(), np.zeros_like(entrance)
    for i, (length, _, offdiagonal, _) in enumerate(generators):
        contribution = np.zeros_like(entrance)
        for node, weight in zip(nodes, weights):
            z = (node+1)*length/2
            incoming = evolution(i, z, zero)
            contribution += weight*evolution(i, length-z, -1j*offdiagonal @ incoming)*length/2
        first = evolution(i, length, first)+contribution
        zero = evolution(i, length, zero)
    shape = (*wave.shape[1:], wave.shape[0])
    return zero.reshape(shape).transpose(2, 0, 1), first.reshape(shape).transpose(2, 0, 1)


def dyson_orders(wave, generators, maximum_order=3):
    """Independent block-triangular dense exponential for Dyson amplitudes."""
    dimension = wave.size
    if dimension*(maximum_order+1) > 1200:
        raise MemoryError('Dyson dense control capped at dimension 1200')
    state = np.zeros((maximum_order+1, dimension), dtype=np.complex128)
    state[0] = wave.transpose(1, 2, 0).ravel()
    for length, diagonal, offdiagonal, _ in generators:
        augmented = np.kron(np.eye(maximum_order+1), diagonal)
        augmented += np.kron(np.diag(np.ones(maximum_order), -1), offdiagonal)
        state = (expm(-1j*length*augmented) @ state.ravel()).reshape(maximum_order+1, dimension)
    return state.reshape(maximum_order+1, *wave.shape[1:], wave.shape[0]).transpose(0, 3, 1, 2)
