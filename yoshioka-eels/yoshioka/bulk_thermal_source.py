"""Bulk harmonic phonons in streamed finite-slab electron Bloch sectors.

All mechanical quantities are SI. BZ coordinates have the normalized measure
d^3q_fractional on [-1/2,1/2)^3. A supplied quadrature weight enters displacement
amplitudes as its square root, exactly once. Thermal occupations multiply the
resulting emission/absorption probabilities; they are not a finite Fock cutoff.
"""

from pathlib import Path
import numpy as np
from scipy import constants
from scipy.stats import qmc

from .electron_phonon import LobatoSilicon, full_space_born
from .free_propagation import electron_wavenumber
from .elastic_propagation import electron_speed
from .dynamical_source import full_transverse_vectors
from .bloch_source_windowed import source_exit_bloch
from .fourier_depth_fields import complex_depth_field, fourier_depth_provider


def occupations(omega, temperature_k):
    omega = np.asarray(omega)
    if temperature_k < 0 or np.any(omega <= 0):
        raise ValueError('temperature must be nonnegative and physical frequencies positive')
    if temperature_k == 0:
        return np.zeros_like(omega)
    x = constants.hbar * omega / (constants.Boltzmann * temperature_k)
    return np.exp(-x) / -np.expm1(-x)


def physical_modes(model, q_fractional):
    q = np.atleast_2d(q_fractional)
    omega, vectors = model.modes(q)
    physical = np.ones(omega.shape, bool)
    translations = np.tile(np.eye(3), (len(model.positions_m), 1))
    translations *= np.repeat(np.sqrt(model.masses_kg), 3)[:, None]
    translations /= np.linalg.norm(translations, axis=0)
    for i in np.flatnonzero(np.linalg.norm(q - np.rint(q), axis=1) < 1e-13):
        overlap = np.sum(abs(vectors[i].conj().T @ translations)**2, axis=1)
        physical[i, np.argsort(overlap)[-3:]] = False
    if np.any(omega[physical] <= 0):
        raise ValueError('unstable physical phonon; no frequency clipping')
    return omega, vectors, physical


def primitive_indices(primitive, positions):
    fractional = (np.asarray(positions)[:, None] - primitive.positions_m[None]) @ np.linalg.inv(primitive.cell_m)
    distance = np.linalg.norm(fractional - np.rint(fractional), axis=-1)
    indices = np.argmin(distance, axis=1)
    if np.max(np.min(distance, axis=1)) > 1e-7:
        raise ValueError('specimen is not commensurate with primitive force constants')
    return indices


def covariance_integral(model, temperature_k, mesh=24, chunk=512):
    """Independent full-branch BZ quadrature of the per-atom thermal covariance."""
    covariance = np.zeros((len(model.positions_m), 3, 3), complex)
    count = mesh**3
    for start in range(0, count, chunk):
        indices = np.array(np.unravel_index(np.arange(start, min(count, start+chunk)), (mesh,)*3)).T
        q = indices / mesh
        q[q > .5] -= 1
        omega, vectors, physical = physical_modes(model, q)
        safe = np.where(physical, omega, 1.)
        factor = constants.hbar * (2*occupations(safe, temperature_k)+1) / (2*safe*count)
        factor *= physical
        vectors = vectors.reshape(len(q), len(model.positions_m), 3, -1)
        covariance += np.einsum('qiam,qibm,qm->iab', vectors, vectors.conj(), factor) / model.masses_kg[:, None, None]
    common = np.mean(covariance.real, axis=0)
    scale = np.linalg.norm(common)
    diag = {'mesh': mesh, 'temperature_k': temperature_k,
            'site_relative_residual': float(np.max(np.linalg.norm(covariance.real-common, axis=(-2, -1)))/scale),
            'imaginary_relative_residual': float(np.linalg.norm(covariance.imag)/scale),
            'off_diagonal_relative_residual': float(np.linalg.norm(common-np.diag(np.diag(common)))/scale),
            'mean_square_displacement_m2': float(np.trace(common)),
            'excluded': 'three rigid translations at Gamma only'}
    if max(diag['site_relative_residual'], diag['imaginary_relative_residual'], diag['off_diagonal_relative_residual']) > 1e-8:
        raise ValueError(f'bulk silicon covariance symmetry check failed: {diag}')
    # Project roundoff only after the independent symmetry checks.
    return np.diag(np.diag(common)), diag


def importance_nodes(power, seed, radius=.35, uniform_fraction=.5):
    """Scrambled Sobol mixture: full BZ plus an acoustic-Gamma radial proposal.

    The radial component is uniform in radius, not volume: its density is
    1/(4*pi*R*r^2). The balance-mixture inverse density accompanies EVERY node.
    This addresses the integrable finite-T acoustic 1/r^2 intensity without
    deleting small-q contributions. Independent seeds supply independent batches.
    """
    if not 0 < radius < .5 or not 0 < uniform_fraction <= 1:
        raise ValueError('invalid mixture parameters')
    u = qmc.Sobol(4, scramble=True, seed=int(seed)).random_base2(int(power))
    q = u[:, 1:] - .5
    choose = u[:, 0] >= uniform_fraction
    r = radius * u[choose, 1]
    cosine = 2*u[choose, 2]-1
    phi = 2*np.pi*u[choose, 3]
    sine = np.sqrt(1-cosine*cosine)
    q[choose] = r[:, None] * np.column_stack((sine*np.cos(phi), sine*np.sin(phi), cosine))
    norm = np.linalg.norm(q, axis=1)
    density = np.full(len(q), uniform_fraction)
    inside = norm < radius
    density[inside] += (1-uniform_fraction)/(4*np.pi*radius*norm[inside]**2)
    return q, 1/density


def mode_group(primitive, positions, lengths_m, q_fractional, weight=1., reference_positions=None):
    omega, vectors, physical = physical_modes(primitive, q_fractional)
    omega = omega[0, physical[0]]
    vectors = vectors[0][:, physical[0]]
    reference_positions = positions if reference_positions is None else reference_positions
    basis = primitive_indices(primitive, reference_positions)
    qcart = np.asarray(q_fractional) @ primitive.reciprocal_m_inv
    d = vectors.reshape(len(primitive.positions_m), 3, -1)[basis]
    d *= np.sqrt(constants.hbar*weight/(2*primitive.masses_kg[basis, None, None]*omega[None, None]))
    d *= np.exp(1j*np.asarray(reference_positions)@qcart)[:, None, None]
    spacing = 2*np.pi/np.asarray(lengths_m)
    shift = (-qcart[:2]+spacing/2) % spacing-spacing/2
    return omega, d, shift, np.flatnonzero(physical[0])


def born_thermal_batch(primitive, positions, covariance, lengths, length, padding,
                       q_fractional, q_weights, temperature_k, energy_j, max_angle=20):
    """B7 atomic-sum physics, extended to signed transfers and thermal factors.

    This evaluator is independent of the numerical Fourier-gradient/source
    propagation. Return raw, unrenormalized weights for four physical detectors.
    """
    q = np.atleast_2d(q_fractional)
    omega, vectors, physical = physical_modes(primitive, q)
    safe = np.where(physical, omega, 1.)
    basis = primitive_indices(primitive, positions)
    k = electron_wavenumber(energy_j)
    velocity = electron_speed(energy_j)
    spacing = 2*np.pi/np.asarray(lengths)
    radius = k*max_angle/1000
    bounds = np.ceil(radius/spacing+.5).astype(int)
    hx, hy = np.meshgrid(np.arange(-bounds[0], bounds[0]+1), np.arange(-bounds[1], bounds[1]+1))
    regular = np.column_stack((hx.ravel(), hy.ravel()))*spacing
    eigen = vectors.reshape(len(q), len(primitive.positions_m), 3, -1)[:, basis]
    mass = np.sqrt(constants.hbar*np.asarray(q_weights)[:, None, None]/(2*safe[:, :, None]*primitive.masses_kg[basis][None, None]))
    nbar = occupations(safe, temperature_k)
    outputs = []
    for sign in (1, -1):
        qcart = sign*q@primitive.reciprocal_m_inv
        shift = (-qcart[:, :2]+spacing/2) % spacing-spacing/2
        Qxy = regular[None]+shift[:, None]
        angle = np.linalg.norm(Qxy, axis=-1)/k*1000
        Qz = -(np.sum(Qxy**2, axis=-1)[:, None]/(2*k)+sign*safe[:, :, None]/velocity)
        eig = eigen.conj() if sign == 1 else eigen
        projection = np.einsum('qbc,qicm->qmbi', Qxy, eig[:, :, :2])
        projection += Qz[:, :, :, None]*eig[:, :, 2].transpose(0, 2, 1)[:, :, None]
        phase_xy = np.einsum('qbc,ic->qbi', Qxy+qcart[:, None, :2], positions[:, :2])
        phase_z = (Qz+qcart[:, None, None, 2])[:, :, :, None]*positions[None, None, None, :, 2]
        atom_sum = np.sum(projection*mass[:, :, None]*np.exp(-1j*(phase_xy[:, None]+phase_z)), axis=-1)
        squared = np.sum(Qxy**2, axis=-1)[:, None]+Qz*Qz
        exponent = np.einsum('qba,ac,qbc->qb', Qxy, covariance[:2, :2], Qxy)[:, None]
        exponent = exponent+2*(Qxy@covariance[:2, 2])[:, None]*Qz+covariance[2, 2]*Qz*Qz
        potential = LobatoSilicon.from_abtem_table().fourier_energy_j_m3(squared)
        amplitude = -potential*np.exp(-.5*exponent)*atom_sum*np.exp(1j*Qz*(length-padding))/(constants.hbar*velocity*np.prod(lengths))
        probability = abs(amplitude)**2*physical[:, :, None]
        probability *= (nbar+1 if sign == 1 else nbar)[:, :, None]
        masks = [(angle <= 20+1e-12), (angle < 5), ((angle >= 5)&(angle < 10)), ((angle >= 10)&(angle <= 20+1e-12))]
        weights = np.stack([np.sum(probability*m[:, None], axis=-1) for m in masks], axis=-1)
        outputs.append(weights)
    energies = safe*constants.hbar/(1e-3*constants.elementary_charge)
    return np.concatenate((energies, -energies), axis=1), np.concatenate(outputs, axis=1), np.concatenate((physical, physical), axis=1)


def streamed_q(primitive, positions, covariance, grid, length, steps, q_fractional,
               temperature_k, incident, background, scratch, weight=1., elastic=True,
               reference_positions=None):
    """One q sector, every branch; signed thermal source and independent Born."""
    omega, d, shift, branches = mode_group(primitive, positions, [grid.lx_m, grid.ly_m], q_fractional, weight, reference_positions)
    source = complex_depth_field(positions, covariance, grid, length, steps, scratch, d, shift)
    provider = fourier_depth_provider(source, grid)
    energy = 60000*constants.elementary_charge
    bg = (lambda i: np.asarray(background[i, 0])) if elastic else None
    inc = lambda i: np.asarray(incident[i])
    modes = len(omega)
    nbar = occupations(omega, temperature_k)
    weights = np.zeros((2*modes, 4))
    born = np.zeros_like(weights)
    base = full_transverse_vectors(grid)
    k = electron_wavenumber(energy)
    for half, sign in enumerate((1, -1)):
        if sign == -1 and temperature_k == 0:
            continue
        at = provider if sign == 1 else lambda i: provider(i).conj()
        actual_shift = sign*shift
        amplitude = source_exit_bloch(grid, energy, sign*omega, length, steps, at, inc, bg, actual_shift)
        Q = base+actual_shift
        angle = np.linalg.norm(Q, axis=1)/k*1000
        masks = [angle <= 20+1e-12, angle < 5, (angle >= 5)&(angle < 10), (angle >= 10)&(angle <= 20+1e-12)]
        selected = masks[0]
        reference = full_space_born(LobatoSilicon.from_abtem_table(), positions, d if sign == 1 else d.conj(),
                                    covariance, Q[selected], sign*omega, energy, grid.lx_m*grid.ly_m, length)
        factor = nbar+1 if sign == 1 else nbar
        a = abs(amplitude.reshape(modes, -1))**2*factor[:, None]
        b = abs(reference)**2*factor[:, None]
        for j, mask in enumerate(masks):
            weights[half*modes:(half+1)*modes, j] = np.sum(a[:, mask], axis=1)
            born[half*modes:(half+1)*modes, j] = np.sum(b[:, mask[selected]], axis=1)
    del source
    Path(scratch).unlink()
    e = constants.hbar*omega/(1e-3*constants.elementary_charge)
    return np.r_[e, -e], weights, born, np.tile(branches, 2)
