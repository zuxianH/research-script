"""Leading two-insertion bulk phonon integrands, without a thermal Fock basis.

The continuous BZ pair integral uses two distinct modes almost surely. Both
time orderings are summed BEFORE squaring. One half of the ordered-pair
integral counts each final state once, including the two mixed-sign labels.
Identical-mode discrete terms are handled by the original thermal-word Gram
trace; they have zero measure in this continuum pair integral. In contrast,
same-mode return paths contribute a single BZ integral and are retained.
"""
from pathlib import Path
import time

import numpy as np
from scipy import constants
from scipy.stats import qmc

from .bulk_thermal_source import mode_group, occupations
from .dynamical_source import full_transverse_vectors
from .fourier_depth_fields import complex_depth_field, fourier_depth_provider
from .free_propagation import electron_wavenumber
from .thermal_dyson_words import propagate_words


SIGNS = ((1, 1), (-1, -1), (1, -1), (-1, 1))


def mixture_transform(u, radius=.35, uniform_fraction=.5):
    """Map four uniform coordinates to the existing normalized BZ proposal."""
    if not 0 < radius < .5 or not 0 < uniform_fraction <= 1:
        raise ValueError('invalid BZ importance proposal')
    u = np.asarray(u)
    q = u[:, 1:].copy()-.5
    pick = u[:, 0] >= uniform_fraction
    r = radius*u[pick, 1]
    cosine = 2*u[pick, 2]-1
    phi = 2*np.pi*u[pick, 3]
    sine = np.sqrt(1-cosine*cosine)
    q[pick] = r[:, None]*np.column_stack((sine*np.cos(phi), sine*np.sin(phi), cosine))
    norm = np.linalg.norm(q, axis=1)
    density = np.full(len(q), uniform_fraction)
    inside = norm < radius
    density[inside] += (1-uniform_fraction)/(4*np.pi*radius*norm[inside]**2)
    return q, 1/density


def pair_nodes(power, seed, radius=.35, uniform_fraction=.5):
    """Nested 10D scrambled Sobol: two q proposals and two uniform branches.

    The returned marginal inverse weights include the branch counting measure
    (six), but neither sample-count averaging nor ordered-pair factor one half.
    """
    u = qmc.Sobol(10, scramble=True, seed=int(seed)).random_base2(int(power))
    a, wa = mixture_transform(u[:, :4], radius, uniform_fraction)
    b, wb = mixture_transform(u[:, 5:9], radius, uniform_fraction)
    branches = np.column_stack((np.floor(6*u[:, 4]), np.floor(6*u[:, 9]))).astype(int)
    return np.stack((a, b), axis=1), branches, 6*np.column_stack((wa, wb))


def masks_for_shift(grid, shift, energy_j=60000*constants.elementary_charge):
    q = full_transverse_vectors(grid)+np.asarray(shift)
    angle = np.linalg.norm(q, axis=1)/electron_wavenumber(energy_j)*1000
    return np.array([angle <= 20+1e-12, angle < 5,
                     (angle >= 5)&(angle < 10), (angle >= 10)&(angle <= 20+1e-12)])


def pair_trace(words, coefficients, shifts, omega, nbar, grid, inverse_marginals):
    """Return exact per-node intensities; no stochastic amplitude squaring.

    Samples are distinct modes. A separate single-integral mean return
    amplitude is retained for the elastic-sector interference. Sample averages
    and independent-batch uncertainty are applied by the benchmark runner.
    """
    if len(omega) != 2 or len(inverse_marginals) != 2:
        raise ValueError('pair_trace requires two distinct modes')
    lookup = {w: i for i, w in enumerate(words)}
    c = np.asarray(coefficients).reshape(len(words), -1)
    inverse_marginals = np.asarray(inverse_marginals)
    pair_inverse = float(np.prod(inverse_marginals))
    energy = []; weights = []; interference = []; all_beam = []
    for sa, sb in SIGNS:
        a = 0 if sa == 1 else 2
        b = 1 if sb == 1 else 3
        i = lookup[(a, b)]; j = lookup[(b, a)]
        amplitude = c[i]+c[j]
        factor = (nbar[0]+(sa == 1))*(nbar[1]+(sb == 1))
        probability = .5*pair_inverse*factor*abs(amplitude)**2
        cross = .5*pair_inverse*factor*2*np.real(c[i]*c[j].conj())
        masks = masks_for_shift(grid, shifts[i])
        weights.append(masks@probability)
        interference.append(masks@cross)
        all_beam.append(np.sum(probability))
        energy.append(constants.hbar*(sa*omega[0]+sb*omega[1])/(1e-3*constants.elementary_charge))
    single_e = []; single_w = []; single_all = []
    returns = []
    for mode in range(2):
        # Marginals are each unbiased one-mode integral estimates; average them.
        iw = inverse_marginals[mode]/2
        for sign in (1, -1):
            letter = mode if sign == 1 else mode+2
            index = lookup[(letter,)]
            p = iw*(nbar[mode]+(sign == 1))*abs(c[index])**2
            single_w.append(masks_for_shift(grid, shifts[index])@p)
            single_all.append(np.sum(p))
            single_e.append(sign*constants.hbar*omega[mode]/(1e-3*constants.elementary_charge))
        r = (nbar[mode]+1)*c[lookup[(mode+2, mode)]]+nbar[mode]*c[lookup[(mode, mode+2)]]
        returns.append(iw*r)
    returned = np.sum(returns, axis=0)
    zero_masks = masks_for_shift(grid, (0, 0))
    elastic_cross = 2*np.real(c[0].conj()*returned)
    balance = float(np.sum(elastic_cross)+np.sum(single_all))
    return dict(energies_mev=np.array(energy), weights=np.array(weights),
                exchange_interference=np.array(interference), all_beam_weights=np.array(all_beam),
                single_energies_mev=np.array(single_e), single_weights=np.array(single_w),
                single_all_beam=np.array(single_all), return_amplitude=returned,
                elastic_interference=zero_masks@elastic_cross,
                all_beam_order2_balance=balance, incident_coefficients=c[0].copy())


def propagate_bulk_pair(model, positions, reference, covariance, grid, length, steps,
                        q_pair, branches, inverse_marginals, background, scratch,
                        temperature_k=300, limits=None, coupling_scale=1., save_amplitudes=False):
    """Two selected bulk modes, every sign and both depth orderings.

    Each source has unit BZ weight. Importance and branch weights enter the
    final probability (or single-integral return amplitude), exactly once.
    Raw mode shifts are added along a word without incompatible sector folding.
    """
    start = time.monotonic()
    scratch = Path(scratch); scratch.mkdir(parents=True, exist_ok=True)
    providers = []; maps = []; omega = []; shifts = []
    try:
        for side in range(2):
            om, displacement, shift, labels = mode_group(model, positions,
                [grid.lx_m, grid.ly_m], q_pair[side], 1., reference)
            selected = np.flatnonzero(labels == branches[side])
            if len(selected) != 1:
                raise ValueError('sampled branch is absent; exact Gamma translations cannot be sampled')
            at = int(selected[0]); omega.append(om[at]); shifts.append(shift)
            mapped = complex_depth_field(positions, covariance, grid, length, steps,
                scratch/f'source{side}.npy', coupling_scale*displacement[:, :, at:at+1], shift)
            maps.append(mapped); providers.append(fourier_depth_provider(mapped, grid))
        field_seconds = time.monotonic()-start
        source_at = lambda i: np.concatenate([p(i) for p in providers], axis=0)
        words, coefficients, word_shifts, energies = propagate_words(grid, np.array(omega),
            np.array(shifts), length, steps, source_at, lambda i: np.asarray(background[i, 0]), limits=limits)
        propagation_seconds = time.monotonic()-start-field_seconds
        result = pair_trace(words, coefficients, word_shifts, omega,
                            occupations(np.array(omega), temperature_k), grid, inverse_marginals)
        result.update(omega_rad_s=np.array(omega), emission_shifts_m_inv=np.array(shifts),
                      field_seconds=field_seconds, propagation_seconds=propagation_seconds)
        if save_amplitudes:
            result.update(word_coefficients=coefficients, word_shifts=word_shifts,
                          word_energies_j=energies)
        return result
    finally:
        providers.clear()
        for mapped in maps:
            mapped._mmap.close()
        for side in range(2):
            (scratch/f'source{side}.npy').unlink(missing_ok=True)
