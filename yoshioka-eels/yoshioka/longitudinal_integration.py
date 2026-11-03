"""Finite-slab longitudinal quadrature and independent numerical source action.

Wavevectors are angular wavevectors in rad/m unless a function explicitly says
``cycles``. Fractional reciprocal coordinates use exp(2*pi*i*q.fractional_R).
Quadrature weights integrate the normalized reciprocal-cell measure; no angular
detector-area factor is implied by these weights.
"""

from pathlib import Path

import numpy as np
from scipy import constants, fft

from .electron_phonon import (LobatoSilicon, derivative_symbol,
                             fourth_order_derivative, mode_plane_coefficients)
from .dynamical_source import full_transverse_vectors
from .elastic_propagation import electron_speed
from .free_propagation import electron_wavenumber


def finite_window_angular(kappa_rad_m, length_m):
    """Integral_0^L exp(i*kappa*z) dz, including its complex phase."""
    if not np.isfinite(length_m) or length_m <= 0:
        raise ValueError('length must be positive and finite')
    kappa = np.asarray(kappa_rad_m, dtype=float)
    if not np.all(np.isfinite(kappa)):
        raise ValueError('angular wavevectors must be finite')
    argument = kappa*length_m/2
    return length_m*np.exp(1j*argument)*np.sinc(argument/np.pi)


def finite_window_cycles(frequency_cycles_m, length_m):
    """Same window with reciprocal coordinates in cycles/m, not rad/m."""
    return finite_window_angular(2*np.pi*np.asarray(frequency_cycles_m), length_m)


def longitudinal_nodes(count, shift=0., seed=None):
    """Uniform periodic mesh, or one unbiased random point in every z stratum.

    Deterministic shift is in fractions of a *mesh interval*: 0 has a Gamma
    point for even count; .5 is midpoint shifted. A seed selects independent
    stratified random quadrature with exactly the same full interval support.
    The weights integrate one normalized reciprocal-cell coordinate.
    """
    if int(count) != count or count <= 0 or not np.isfinite(shift):
        raise ValueError('count must be a positive integer and shift finite')
    count = int(count)
    if seed is None:
        offsets = np.full(count, float(shift))
    else:
        if shift != 0:
            raise ValueError('random strata use an independent seed, not a mesh shift')
        offsets = np.random.default_rng(int(seed)).random(count)
    values = (np.arange(count)+offsets)/count-.5
    values = (values+.5) % 1-.5
    return values, np.full(count, 1/count)


def anisotropic_nodes(shape, shifts=(0., 0., 0.)):
    """Full tensor reciprocal-cell mesh with independent axis refinements."""
    if len(shape) != 3 or len(shifts) != 3:
        raise ValueError('three mesh counts and shifts are required')
    axes = [longitudinal_nodes(n, s)[0] for n, s in zip(shape, shifts)]
    values = np.array(np.meshgrid(*axes, indexing='ij')).reshape(3, -1).T
    return values, np.full(len(values), 1/len(values))


def numerical_source_contraction(potential, positions_m, displacements_m,
                                 covariance_m2, qxy_m_inv, propagator, length_m,
                                 longitudinal_samples, kernel_cache=None):
    """Exactly contract the original inverse-FFT/FD/Simpson source at its exit.

    This performs the same discrete source propagation as
    ``FirstOrderSourcePropagator.propagate_scan``. Linearity lets us contract
    each numerical plane kernel *before* multiplying its mode coefficients.
    The atomic potential is still inverse transformed, differentiated with the
    fourth-order Cartesian stencil, and integrated with the same Simpson and
    free-propagation multipliers. No analytic Born amplitude is used here.
    """
    samples = int(longitudinal_samples)
    cache = {} if kernel_cache is None else kernel_cache
    if samples < 8 or samples % 2:
        raise ValueError('longitudinal_samples must be even and >=8')
    planes, coefficients = mode_plane_coefficients(positions_m, displacements_m, qxy_m_inv)
    key = (qxy_m_inv.tobytes(), planes.tobytes(), covariance_m2.tobytes(),
           potential.a_angstrom.tobytes(), potential.b_angstrom2.tobytes(),
           length_m, samples, propagator.grid.dx_m, propagator.grid.dy_m,
           propagator.grid.lx_m*propagator.grid.ly_m)
    kernels = cache.get(key)
    if kernels is None:
        # Exactly the original inverse-FFT kernel and fourth-order stencil,
        # constructed directly to avoid a dummy full longitudinal source.
        dz = length_m/samples
        kz = 2*np.pi*np.fft.fftfreq(samples, dz)
        squared = np.sum(qxy_m_inv**2, axis=1)[:, None]+kz[None]**2
        dw = np.exp(-.5*(covariance_m2[0, 0]*qxy_m_inv[:, 0, None]**2+
            covariance_m2[1, 1]*qxy_m_inv[:, 1, None]**2+covariance_m2[2, 2]*kz[None]**2))
        fourier = potential.fourier_energy_j_m3(squared)*dw
        kernels = []
        for position in planes:
            field = np.fft.ifft(fourier*np.exp(-1j*kz[None]*position), axis=-1)/(
                propagator.grid.lx_m*propagator.grid.ly_m*dz)
            kernels.append((field, fourth_order_derivative(field, dz)))
        if len(cache) >= 16:
            cache.pop(next(iter(cache)))
        cache[key] = kernels
    count = samples//2
    step = length_m/count
    full, half = propagator.multiplier(step), propagator.multiplier(step/2)
    # A source at even sample zero appears at both slab endpoints because the
    # original inverse-FFT source and Simpson stepper use periodic samples.
    powers = np.arange(count, 0, -1)
    phase = np.exp(np.log(full)[..., None]*powers)
    factors = np.empty((*full.shape, samples), complex)
    factors[..., ::2] = 2*phase
    factors[..., 0] = phase[..., 0]+1
    factors[..., 1::2] = 4*half[..., None]*phase/full[..., None]
    factors *= -1j*step/(6*constants.hbar*propagator.velocity)
    dx = derivative_symbol(qxy_m_inv[:, 0], propagator.grid.dx_m)
    dy = derivative_symbol(qxy_m_inv[:, 1], propagator.grid.dy_m)
    amplitude = np.zeros(full.shape, complex)
    for plane, (field, derivative_z) in enumerate(kernels):
        transverse = coefficients[:, plane, 0]*dx+coefficients[:, plane, 1]*dy
        integrated = np.einsum('mqs,qs->mq', factors, field)
        integrated_z = np.einsum('mqs,qs->mq', factors, derivative_z)
        amplitude -= transverse*integrated+coefficients[:, plane, 2]*integrated_z
    return amplitude


def _depth_fourier(positions, covariance, grid, length, steps, target,
                   displacements=None, shift=(0., 0.), workers=1, chunk=128):
    """Depth-major exact Fourier-gradient field, with an explicit FFT budget."""
    qxy = full_transverse_vectors(grid)+np.asarray(shift)
    dz = length/steps
    kz = 2*np.pi*np.fft.fftfreq(steps, dz)
    planes, inverse = np.unique(positions[:, 2], return_inverse=True)
    axial = np.exp(-1j*(planes[:, None]-dz/2)*kz)
    modes = 1 if displacements is None else displacements.shape[-1]
    mapped = np.lib.format.open_memmap(target, mode='w+', dtype=complex,
                                     shape=(steps, modes, grid.ny*grid.nx))
    potential = LobatoSilicon.from_abtem_table()
    area = grid.lx_m*grid.ly_m
    for lo in range(0, len(qxy), chunk):
        hi = min(lo+chunk, len(qxy)); Q = qxy[lo:hi]
        exponent = np.einsum('qa,ab,qb->q', Q, covariance[:2, :2], Q)[:, None]
        exponent = exponent+2*(Q@covariance[:2, 2])[:, None]*kz+covariance[2, 2]*kz*kz
        kernel = potential.fourier_energy_j_m3(np.sum(Q*Q, axis=1)[:, None]+kz*kz)
        kernel *= np.exp(-.5*exponent)/(area*dz)
        if displacements is None:
            coefficients = np.array([np.sum(np.exp(-1j*positions[inverse == p, :2]@Q.T), axis=0)
                                     for p in range(len(planes))])
            fourier = (coefficients.T@axial)[None]*kernel
        else:
            _, coefficients = mode_plane_coefficients(positions, displacements, Q)
            xy = coefficients[:, :, 0]*Q[None, None, :, 0]+coefficients[:, :, 1]*Q[None, None, :, 1]
            fourier = -1j*(xy.transpose(0, 2, 1)@axial+
                (coefficients[:, :, 2].transpose(0, 2, 1)@axial)*kz[None, None])*kernel[None]
        values = fft.ifft(fourier, axis=-1, workers=workers)
        mapped[:, :, lo:hi] = values.transpose(2, 0, 1)
        # Source files are ephemeral; discard mappings after each contiguous
        # write to keep RSS bounded while retaining the file-backed contents.
        mapped.flush()
        if hasattr(mapped._mmap, 'madvise'):
            import mmap
            mapped._mmap.madvise(mmap.MADV_DONTNEED)
    return mapped


def prepare_distorted_background(positions, covariance, grid, length, steps,
                                 folder, energy_j, workers=1):
    """Numerically construct the fixed elastic background and depth wave.

    Callers own the identity/checkpoint contract. These files may only be
    reused when atom positions, covariance, electron grid and depth agree.
    """
    from .depth_fields import incident_depth
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    background_path = folder/'background.npy'; incident_path = folder/'incident.npy'
    if background_path.exists() and incident_path.exists() and (folder/'background_complete').exists():
        return np.load(background_path, mmap_mode='r'), np.load(incident_path, mmap_mode='r'), None
    mapped = _depth_fourier(positions, covariance, grid, length, steps,
                           folder/'background_fourier.npy', workers=workers)
    result = np.lib.format.open_memmap(background_path, mode='w+', dtype=float,
                                     shape=(steps, 1, *grid.shape))
    imaginary_squared = real_squared = 0.
    for i in range(steps):
        value = fft.ifft2(np.asarray(mapped[i]).reshape(1, *grid.shape), workers=workers)*(grid.nx*grid.ny)
        imaginary_squared += float(np.sum(value.imag**2)); real_squared += float(np.sum(value.real**2))
        result[i] = value.real
    residual = np.sqrt(imaginary_squared/max(real_squared, np.finfo(float).tiny))
    if residual > 1e-12:
        raise ValueError('unprojected elastic background failed reality check')
    result.flush(); del mapped
    (folder/'background_fourier.npy').unlink()
    # incident_depth uses numpy FFTs and the validated real-space split step.
    exit_wave, incident, diagnostic = incident_depth(grid, energy_j, result,
        length, incident_path, True)
    np.save(folder/'incident_exit.npy', exit_wave)
    (folder/'background_complete').write_text('complete\n')
    diagnostic['background_relative_imaginary_residual'] = float(residual)
    return result, incident, diagnostic


def distorted_source_exit(grid, energy_j, omega_rad_s, length, steps,
                          fourier_source, incident, background, shift, workers=1):
    """Depth-dependent Bloch source, identical split-step rule, bounded FFTs."""
    omega = np.asarray(omega_rad_s); dz = length/steps
    velocity = electron_speed(energy_j)
    Q = full_transverse_vectors(grid)+np.asarray(shift)
    kinetic = np.exp(-.5j*dz*np.sum(Q*Q, axis=1).reshape(grid.shape)/(2*electron_wavenumber(energy_j)))
    channel = np.exp(-1j*omega*dz/velocity)[:, None, None]
    insertion = (-1j*dz/(constants.hbar*velocity)*np.exp(-.5j*omega*dz/velocity)*
                 np.sinc(omega*dz/(2*np.pi*velocity)))[:, None, None]
    wave = np.zeros((len(omega), *grid.shape), complex)
    for i in range(steps):
        wave = fft.ifft2(fft.fft2(wave, axes=(-2, -1), workers=workers)*kinetic,
                         axes=(-2, -1), workers=workers)
        source = fft.ifft2(np.asarray(fourier_source[i]).reshape(-1, *grid.shape),
                           axes=(-2, -1), workers=workers)*(grid.nx*grid.ny)
        wave = channel*wave+insertion*source*np.asarray(incident[i])[None]
        if background is not None:
            wave *= np.exp(-1j*np.asarray(background[i, 0])*dz/(constants.hbar*velocity))
        wave = fft.ifft2(fft.fft2(wave, axes=(-2, -1), workers=workers)*kinetic,
                         axes=(-2, -1), workers=workers)
    return fft.fft2(wave, axes=(-2, -1), workers=workers)*np.sqrt(grid.lx_m*grid.ly_m)/(grid.nx*grid.ny)


def fixed_q_distorted_weights(model, positions, unpadded, covariance, grid,
                              length, steps, q_fractional, incident, background,
                              scratch, energy_j, weight=1., workers=1):
    """T=0 all-branch distorted and independent Born weights at one q point."""
    from .bulk_thermal_source import mode_group
    from .electron_phonon import full_space_born
    omega, d, shift, branches = mode_group(model, positions,
        (grid.lx_m, grid.ly_m), q_fractional, weight, unpadded)
    source = _depth_fourier(positions, covariance, grid, length, steps, scratch,
                           d, shift, workers)
    amplitude = distorted_source_exit(grid, energy_j, omega, length, steps,
        source, incident, background, shift, workers)
    Q = full_transverse_vectors(grid)+shift
    angles = np.linalg.norm(Q, axis=1)/electron_wavenumber(energy_j)*1000
    masks = [angles <= 20+1e-12, angles < 5,
             (angles >= 5)&(angles < 10), (angles >= 10)&(angles <= 20+1e-12)]
    selected = masks[0]
    reference = full_space_born(LobatoSilicon.from_abtem_table(), positions,
        d, covariance, Q[selected], omega, energy_j, grid.lx_m*grid.ly_m, length)
    a = abs(amplitude.reshape(len(omega), -1))**2; b = abs(reference)**2
    weights = np.stack([np.sum(a[:, mask], axis=1) for mask in masks], axis=-1)
    born = np.stack([np.sum(b[:, mask[selected]], axis=1) for mask in masks], axis=-1)
    del source
    Path(scratch).unlink()
    return constants.hbar*omega/(1e-3*constants.elementary_charge), weights, born, branches
