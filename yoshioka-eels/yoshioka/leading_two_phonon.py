"""Complete leading displacement-order-two changed-state mode-pair amplitude.

The two sequential time orderings and the direct quadratic vertex are summed
for each SAME final bath state, before thermal probabilities are averaged.
The continuum integral covers distinct modes almost surely; exact identical
mode/return traces belong to the finite-bath reference, not a doubled pair
probability. No positive zero-energy 'two-phonon' term is invented here.

``static`` uses V0 at undisplaced atoms: genuine Taylor displacement order.
``thermal_normal_ordered`` uses the existing thermal-DW V0 and Wick-ordered
quadratic residual. Its scalar subtraction has zero changed-state matrix
element; the pair amplitude is a Wick-degree-two approximation, not a raw
Taylor expansion with an unnoticed thermal expectation counted twice.
"""
from pathlib import Path
import time
import numpy as np
from scipy import constants, fft

from .bulk_thermal_source import mode_group, occupations
from .dynamical_source import full_transverse_vectors
from .elastic_propagation import electron_speed
from .free_propagation import electron_wavenumber
from .fourier_depth_fields import complex_depth_field
from .quadratic_fields import quadratic_depth_field
from .thermal_dyson_words import words_through_two, local_coefficients
from .two_phonon import SIGNS, masks_for_shift


def propagate_leading_pair(grid, omega, shifts, length, steps, source_at, quadratic_at,
                           background_at, energy_j=60000*constants.elementary_charge,
                           limits=None, workers=1, vectorized=True, retain_all_words=False):
    """Stream 21 sequential word waves plus four direct final-state waves."""
    words = words_through_two(2)
    if not retain_all_words:
        # Same-mode two-letter words cannot feed a distinct-mode final state
        # in this triangular order-two hierarchy. Removing these disconnected
        # waves is exact for every reported pair observable, at every T.
        words = [w for w in words if len(w)<2 or w[0]%2 != w[1]%2]
    lookup = {w: i for i, w in enumerate(words)}
    signed_energy = constants.hbar*np.r_[omega, -np.asarray(omega)]
    signed_shift = np.r_[shifts, -np.asarray(shifts)]
    energies = np.array([sum(signed_energy[k] for k in w) for w in words])
    word_shifts = np.array([np.sum(signed_shift[list(w)], axis=0) if w else np.zeros(2) for w in words])
    final_indices = np.array([lookup[(0 if sa == 1 else 2, 1 if sb == 1 else 3)] for sa, sb in SIGNS])
    dz = length/steps; hv = constants.hbar*electron_speed(energy_j)
    phase, single, double = local_coefficients(words, energies, dz, hv)
    direct_e = energies[final_indices]
    direct_single = -1j*dz/hv*np.exp(-.5j*direct_e*dz/hv)*np.sinc(direct_e*dz/(2*np.pi*hv))
    all_shifts = np.r_[word_shifts, word_shifts[final_indices]]
    Q = full_transverse_vectors(grid)[None]+all_shifts[:, None]
    kinetic = np.exp(-.5j*dz*np.sum(Q*Q, axis=-1).reshape(-1, *grid.shape)/(2*electron_wavenumber(energy_j)))
    waves = np.zeros((len(words)+4, *grid.shape), complex)
    waves[0] = 1/np.sqrt(grid.lx_m*grid.ly_m)
    parents = np.array([lookup[w[1:]] for w in words[1:]])
    letters = np.array([w[0] for w in words[1:]])
    second = np.array([w[1] if len(w) == 2 else -1 for w in words[1:]])
    # Explicit vectorized form and scalar control implement identical algebra.
    for i in range(steps):
        if limits is not None and i % 16 == 0: limits.check()
        waves = fft.ifft2(fft.fft2(waves, axes=(-2, -1), workers=workers)*kinetic,
                          axes=(-2, -1), workers=workers)
        emission = source_at(i); quadratic = quadratic_at(i)
        if vectorized:
            sources = np.concatenate((emission, emission.conj()))
            result = np.empty_like(waves)
            result[:len(words)] = phase[:, None, None]*waves[:len(words)]
            result[1:len(words)] += single[1:, None, None]*sources[letters]*waves[parents]
            two = np.flatnonzero(second >= 0)
            result[1+two] += (double[1+two, None, None]*sources[letters[two]]
                             *sources[second[two]]*waves[0])
            result[len(words):] = (phase[final_indices, None, None]*waves[len(words):]
                                  +direct_single[:, None, None]*quadratic*waves[0])
        else:
            from .thermal_dyson_words import local_step
            result = np.empty_like(waves)
            result[:len(words)] = local_step(waves[:len(words)], np.r_[emission, emission.conj()],
                words, energies, 0., dz, hv, (phase, single, double))
            for k, f in enumerate(final_indices):
                result[len(words)+k] = phase[f]*waves[len(words)+k]+direct_single[k]*quadratic[k]*waves[0]
        result *= np.exp(-1j*np.asarray(background_at(i))*dz/hv)
        waves = fft.ifft2(fft.fft2(result, axes=(-2, -1), workers=workers)*kinetic,
                          axes=(-2, -1), workers=workers)
    coefficients = fft.fft2(waves, axes=(-2, -1), workers=workers)*np.sqrt(grid.lx_m*grid.ly_m)/(grid.nx*grid.ny)
    sequential = np.array([coefficients[lookup[(0 if sa == 1 else 2, 1 if sb == 1 else 3)]]
                           +coefficients[lookup[(1 if sb == 1 else 3, 0 if sa == 1 else 2)]] for sa, sb in SIGNS])
    return dict(words=words, word_coefficients=coefficients[:len(words)],
                word_shifts=word_shifts, word_energies_j=energies,
                sequential_amplitudes=sequential, direct_amplitudes=coefficients[len(words):],
                final_shifts=word_shifts[final_indices])


def leading_pair_trace(direct, sequential, omega, nbar, grid, shifts, inverse_marginals,
                       energy_j=60000*constants.elementary_charge):
    """Thermal average of probabilities; exchange, then direct, are coherent."""
    weight = .5*float(np.prod(inverse_marginals))
    quantities = {k: [] for k in ('direct_weights', 'sequential_weights', 'interference_weights', 'weights')}
    energies = []
    for k, (sa, sb) in enumerate(SIGNS):
        factor = weight*(nbar[0]+(sa == 1))*(nbar[1]+(sb == 1))
        masks = masks_for_shift(grid, shifts[k], energy_j)
        d, s = direct[k].ravel(), sequential[k].ravel()
        quantities['direct_weights'].append(masks@(factor*abs(d)**2))
        quantities['sequential_weights'].append(masks@(factor*abs(s)**2))
        quantities['interference_weights'].append(masks@(factor*2*np.real(d*s.conj())))
        quantities['weights'].append(masks@(factor*abs(d+s)**2))
        energies.append(constants.hbar*(sa*omega[0]+sb*omega[1])/(1e-3*constants.elementary_charge))
    result = {k: np.array(v) for k, v in quantities.items()}
    result['energies_mev'] = np.array(energies)
    return result


def propagate_bulk_leading_pair(model, positions, reference, covariance, grid, length, steps,
                                q_pair, branches, inverse_marginals, background, scratch,
                                temperature_k=300, limits=None, coupling_scale=1.,
                                save_amplitudes=False, workers=1, vectorized=False, retain_all_words=False,
                                reference_kind='static'):
    if reference_kind not in ('static', 'thermal_normal_ordered'):
        raise ValueError('explicit static or thermal_normal_ordered elastic reference required')
    if reference_kind == 'static' and np.any(np.asarray(covariance) != 0):
        raise ValueError('static Taylor reference requires zero smoothing covariance')
    scratch = Path(scratch); scratch.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); maps = []; omega = []; shifts = []; displacements = []
    times = {}; mode_start = time.monotonic()
    try:
        for side in range(2):
            om, d, shift, labels = mode_group(model, positions, [grid.lx_m, grid.ly_m], q_pair[side], 1., reference)
            selected = np.flatnonzero(labels == branches[side])
            if len(selected) != 1: raise ValueError('selected physical branch is absent')
            at = int(selected[0]); omega.append(om[at]); shifts.append(shift)
            displacements.append(coupling_scale*d[:, :, at])
        times['phonon_seconds'] = time.monotonic()-mode_start
        field_start = time.monotonic()
        for side in range(2):
            maps.append(complex_depth_field(positions, covariance, grid, length, steps,
                scratch/f'linear{side}.npy', displacements[side][:, :, None], shifts[side]))
        times['linear_field_seconds'] = time.monotonic()-field_start
        first = []; second = []; final_shifts = []
        for sa, sb in SIGNS:
            first.append(displacements[0].conj() if sa == 1 else displacements[0])
            second.append(displacements[1].conj() if sb == 1 else displacements[1])
            final_shifts.append(sa*shifts[0]+sb*shifts[1])
        quadratic_start = time.monotonic()
        maps.append(quadratic_depth_field(positions, covariance, grid, length, steps, scratch/'quadratic.npy',
            np.stack(first, axis=-1), np.stack(second, axis=-1), np.array(final_shifts), workers=workers))
        times['quadratic_field_seconds'] = time.monotonic()-quadratic_start
        def provider(i, indices):
            value = np.concatenate([np.asarray(maps[k][i]) for k in indices], axis=0).reshape(-1, *grid.shape)
            return fft.ifft2(value, axes=(-2, -1), workers=workers)*(grid.nx*grid.ny)
        prop_start = time.monotonic()
        amplitudes = propagate_leading_pair(grid, np.array(omega), np.array(shifts), length, steps,
            lambda i: provider(i, (0, 1)), lambda i: provider(i, (2,)), lambda i: np.asarray(background[i, 0]),
            limits=limits, workers=workers, vectorized=vectorized, retain_all_words=retain_all_words)
        times['propagation_seconds'] = time.monotonic()-prop_start
        accumulation_start = time.monotonic()
        result = leading_pair_trace(amplitudes['direct_amplitudes'], amplitudes['sequential_amplitudes'], omega,
            occupations(np.array(omega), temperature_k), grid, amplitudes['final_shifts'], inverse_marginals)
        times['detector_seconds'] = time.monotonic()-accumulation_start
        result.update(omega_rad_s=np.array(omega), emission_shifts_m_inv=np.array(shifts),
                      reference_kind=reference_kind, **times, elapsed_seconds=time.monotonic()-started)
        if save_amplitudes:
            result.update({k: v for k, v in amplitudes.items() if k != 'words'})
        return result
    finally:
        for mapped in maps: mapped._mmap.close()
        for name in ('linear0.npy', 'linear1.npy', 'quadratic.npy'):
            (scratch/name).unlink(missing_ok=True)
