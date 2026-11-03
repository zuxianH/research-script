"""Separately identified free-dispersion sensitivity of the V0=0 control.

This is not a replacement for the primary paraxial Hamiltonian. It changes
only the diagonal kinetic rate in the verified intermediate-resolved control.
The displaced-atom vertices, target-energy phases, incident normalization,
Strang splitting and RK4 local polynomial remain fixed. The square-root rate
is a scalar free-dispersion sensitivity, not a complete nonparaxial theory:
no longitudinal-velocity vertex correction or new scattering model is inferred.
"""
import numpy as np
from scipy import constants
from .elastic_propagation import electron_speed
from .pair_intermediate_momentum import local_polynomials


PARAXIAL_ID = 'V0_ZERO_PARAXIAL_RESOLVED_VERIFICATION_V1'
SENSITIVITY_ID = 'V0_ZERO_SCALAR_FREE_DISPERSION_SENSITIVITY_60KEV_V1'


def kinetic_rates(Q, k0, identifier):
    """Positive longitudinal envelope rates in angular inverse metres."""
    Q = np.asarray(Q)
    squared = np.sum(Q * Q, axis=-1)
    if identifier == PARAXIAL_ID:
        return squared / (2 * k0)
    if identifier != SENSITIVITY_ID:
        raise ValueError('explicit registered dispersion identifier required')
    if np.any(squared >= k0 * k0):
        raise ValueError('this diagnostic rejects Q>=k0; no evanescent continuation is invented')
    # Rationalized difference avoids loss of significance near Q=0.
    return squared / (k0 + np.sqrt(k0 * k0 - squared))


def propagate_dispersion_control(reference, length, linear_at, quadratic_at,
                                identifier, limits=None):
    """Use a resolved inventory with an explicitly selected diagonal rate.

    reference supplies labels, phases, shape, support and numerical settings
    from the independently verified primary resolved control. It contributes
    no propagated amplitudes to the new evolution.
    """
    first = np.asarray(reference['first_letter_indices'])
    second = np.asarray(reference['second_letter_indices'])
    out = np.asarray(reference['output_integer_labels'])
    h = np.asarray(reference['intermediate_integer_labels'])
    T = int(reference['transfer_grid'])
    steps = int(reference['steps'])
    substeps = int(reference['local_substeps'])
    omega = np.asarray(reference['omega_rad_s'])
    energy = constants.hbar * np.r_[omega, -omega]
    final_energy = np.asarray(reference['final_energy_j'])
    k0 = float(reference['incident_k0_m_inv'])
    ordered_Q = np.asarray(reference['intermediate_Q_m_inv'])
    intermediate_Q = np.empty((4, len(h), 2))
    for letter in range(4):
        positions = np.argwhere(first == letter)
        intermediate_Q[letter] = ordered_Q[tuple(positions[0])]
        for position in positions[1:]:
            if not np.array_equal(intermediate_Q[letter], ordered_Q[tuple(position)]):
                raise ValueError('inconsistent intermediate momentum inventory')
    dz = length / steps
    hv = constants.hbar * electron_speed(60000 * constants.elementary_charge)
    scale = -1j * dz / (hv * substeps)
    rx = kinetic_rates(intermediate_Q, k0, identifier)
    ry = kinetic_rates(np.asarray(reference['final_Q_m_inv']), k0, identifier)
    if identifier == PARAXIAL_ID:
        # Preserve the reference multiplication order as well as its formula.
        kx = np.exp(-.5j * dz * np.sum(intermediate_Q**2, axis=-1) / (2 * k0))
        ky = np.exp(-.5j * dz * np.sum(np.asarray(reference['final_Q_m_inv'])**2, axis=-1) / (2 * k0))
    else:
        kx = np.exp(-.5j * dz * rx)
        ky = np.exp(-.5j * dz * ry)
    _, pf, _, bf, transfer, double = local_polynomials(scale, energy[first], final_energy[:, None])
    phase_f, single_f = pf[:, 0], bf[:, 0]
    phase_x, _, single_x, *_ = local_polynomials(scale, energy, energy)
    x = np.zeros((4, len(h)), complex)
    y = np.zeros((4, 2, len(out), len(h)), complex)
    direct = np.zeros((4, len(out)), complex)
    delta = out[:, None] - h[None]
    if np.max(abs(delta)) > T // 2:
        raise ValueError('required transfer missing')
    dy, dx = delta[..., 0] % T, delta[..., 1] % T
    iy, ix = h[:, 0] % T, h[:, 1] % T
    opposite = (-np.arange(T)) % T
    for i in range(steps):
        if limits is not None and i % 8 == 0:
            limits.check(estimated_bytes=y.nbytes * 6 + x.nbytes * 8)
        emission = np.asarray(linear_at(i))
        if emission.shape != (2, T, T):
            raise ValueError('linear field shape differs from frozen inventory')
        signed = np.r_[emission, emission[:, opposite[:, None], opposite].conj()]
        at_h = signed[:, iy, ix]
        at_delta = signed[second[:, :, None, None], dy[None, None], dx[None, None]]
        q = np.asarray(quadratic_at(i))
        if q.shape != direct.shape:
            raise ValueError('quadratic field shape differs from frozen inventory')
        x *= kx
        y *= ky[:, None, :, None]
        direct *= ky
        for _ in range(substeps):
            y *= phase_f[:, None, None, None]
            y += at_delta * (transfer[:, :, None, None] * x[first][:, :, None, :]
                             + double[:, :, None, None] * at_h[first][:, :, None, :])
            direct = phase_f[:, None] * direct + single_f[:, None] * q
            x = phase_x[:, None] * x + single_x[:, None] * at_h
        x *= kx
        y *= ky[:, None, :, None]
        direct *= ky
    sequential = y.sum(axis=(1, 3))
    return dict(identifier=identifier, direct_amplitudes=direct,
                sequential_amplitudes=sequential, total_amplitudes=direct + sequential,
                ordered_by_intermediate=y, first_amplitudes=x,
                intermediate_kinetic_rate_m_inv=rx, final_kinetic_rate_m_inv=ry,
                scope='Fixed-support V0=0 free-dispersion sensitivity only; original benchmark and vertices unchanged.')
