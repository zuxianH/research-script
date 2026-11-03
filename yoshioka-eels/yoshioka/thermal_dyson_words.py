"""Two insertions of a linear harmonic source, without a thermal Fock basis.

An electron coefficient psi_w multiplies the ordered bath operator b_w.
The leftmost letter acts last. Its channel energy and Bloch shift are the sums
of the signed exchanged quanta/momenta. Thermal traces use bosonic Gaussian
Wick contractions, including all pairings. This is a limited-order linear
model, not a nonlinear phase-operator or full-intensity-through-g^4 solver.
"""

from itertools import product
import numpy as np
from scipy import constants, fft
from numpy.polynomial.legendre import leggauss

from .free_propagation import electron_wavenumber
from .elastic_propagation import electron_speed
from .dynamical_source import full_transverse_vectors


def words_through_two(modes):
    return [()] + [(i,) for i in range(2*modes)] + list(product(range(2*modes), repeat=2))


def label(letter, modes):
    return letter % modes, 1 if letter < modes else -1


def thermal_moment(operators, nbar):
    """Ordered Gaussian boson moment: creation=+1, annihilation=-1."""
    if not operators: return 1.
    if len(operators) % 2: return 0.
    a, sign = operators[0]; result = 0.
    for j in range(1, len(operators)):
        b, other = operators[j]
        if a == b and sign == -other:
            pair = nbar[a] + (1 if sign == -1 else 0)
            result += pair*thermal_moment(operators[1:j]+operators[j+1:], nbar)
    return float(result)


def gram_matrix(words, modes, nbar):
    result = np.zeros((len(words), len(words)))
    for i, word in enumerate(words):
        for j, other in enumerate(words):
            adjoint = tuple((label(k, modes)[0], -label(k, modes)[1]) for k in reversed(other))
            ops = adjoint+tuple(label(k, modes) for k in word)
            result[i, j] = thermal_moment(ops, nbar)
    return result


def net_occupation(word, modes):
    net = np.zeros(modes, int)
    for letter in word:
        mode, sign = label(letter, modes); net[mode] += sign
    return tuple(net)


def local_coefficients(words, energies_j, dz, hv):
    """Exact one insertion and converged 8x8 ordered quadrature for two.

    The quadrature evaluates the continuous divided difference even when
    energies coincide; no subtraction of almost equal exponentials occurs.
    """
    lookup = {word: i for i, word in enumerate(words)}
    phase = np.exp(-1j*energies_j*dz/hv)
    single = np.zeros(len(words), complex); double = np.zeros(len(words), complex)
    x, w = leggauss(8); t2 = (x[:, None]+1)/2; t1 = t2*(x[None, :]+1)/2
    weights = w[:, None]*w[None, :]*t2/4
    for j, word in enumerate(words[1:], 1):
        parent = lookup[word[1:]]; e1 = energies_j[parent]; e2 = energies_j[j]
        single[j] = -1j*dz/hv*np.exp(-.5j*(e1+e2)*dz/hv)*np.sinc((e2-e1)*dz/(2*np.pi*hv))
        if len(word) == 2:
            integral = np.sum(weights*np.exp(-1j*dz/hv*(e2*(1-t2)+e1*(t2-t1))))
            double[j] = (-1j*dz/hv)**2*integral
    return phase, single, double


def local_step(waves, sources, words, energies_j, background_j, dz, hv, coefficients=None):
    phase, single, double = local_coefficients(words, energies_j, dz, hv) if coefficients is None else coefficients
    lookup = {word: i for i, word in enumerate(words)}
    result = phase[:, None, None]*waves
    for j, word in enumerate(words[1:], 1):
        parent = lookup[word[1:]]
        result[j] += single[j]*sources[word[0]]*waves[parent]
        if len(word) == 2:
            result[j] += double[j]*sources[word[0]]*sources[word[1]]*waves[0]
    result *= np.exp(-1j*np.asarray(background_j)*dz/hv)
    return result


def propagate_words(grid, omega, emission_shifts, length, steps, source_at, background_at,
                    energy_j=60000*constants.elementary_charge, limits=None):
    """Stream sources; elastic diffraction is retained in every signed word."""
    modes = len(omega); words = words_through_two(modes)
    signed_energy = constants.hbar*np.r_[omega, -np.asarray(omega)]
    signed_shift = np.r_[emission_shifts, -np.asarray(emission_shifts)]
    energies = np.array([sum(signed_energy[i] for i in word) for word in words])
    shifts = np.array([np.sum(signed_shift[list(word)], axis=0) if word else np.zeros(2) for word in words])
    dz = length/steps; hv = constants.hbar*electron_speed(energy_j)
    Q = full_transverse_vectors(grid)[None]+shifts[:, None]
    kinetic = np.exp(-.5j*dz*np.sum(Q*Q, axis=-1).reshape(len(words), *grid.shape)/(2*electron_wavenumber(energy_j)))
    coefficients = local_coefficients(words, energies, dz, hv)
    waves = np.zeros((len(words), *grid.shape), complex); waves[0] = 1/np.sqrt(grid.lx_m*grid.ly_m)
    for i in range(steps):
        if limits is not None and i % 16 == 0: limits.check()
        waves = fft.ifft2(fft.fft2(waves, axes=(-2, -1), workers=4)*kinetic, axes=(-2, -1), workers=4)
        emission = source_at(i); sources = np.concatenate((emission, emission.conj()))
        waves = local_step(waves, sources, words, energies, background_at(i), dz, hv, coefficients)
        waves = fft.ifft2(fft.fft2(waves, axes=(-2, -1), workers=4)*kinetic, axes=(-2, -1), workers=4)
    coefficients = fft.fft2(waves, axes=(-2, -1), workers=4)*np.sqrt(grid.lx_m*grid.ly_m)/(grid.nx*grid.ny)
    return words, coefficients, shifts, energies


def thermal_groups(words, fourier, modes, nbar, quanta_j, maximum_order=2):
    """Coherent words within each final net occupation, traced over initial T."""
    groups = {}
    for i, word in enumerate(words):
        if len(word) <= maximum_order: groups.setdefault(net_occupation(word, modes), []).append(i)
    lines = []
    for net, indices in groups.items():
        subset = [words[i] for i in indices]; matrix = gram_matrix(subset, modes, nbar)
        a = fourier[indices].reshape(len(indices), -1)
        weight = np.einsum('im,ij,jm->m', a, matrix, a.conj()).real
        if np.min(weight) < -1e-12*max(float(np.max(abs(weight))), 1e-100):
            raise ValueError('thermal Gram trace lost positivity beyond roundoff')
        lines.append({'net': net, 'energy_j': float(np.dot(net, quanta_j)), 'weight': weight,
                      'indices': indices, 'words': subset})
    return lines
