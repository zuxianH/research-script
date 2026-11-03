"""Mode-projected classical spectra with explicit finite-window resolution."""

import numpy as np
from scipy import constants
from scipy.optimize import least_squares
from scipy.signal import fftconvolve


PLANCK_MEV_PS = constants.h/(1e-3*constants.elementary_charge)*1e12


def hann_energy_kernel(energy_mev, duration_ps):
    """Unit-area Hann power-window density, per meV, for a continuous window.

    The kernel is the exact finite-duration window transform, not instrumental
    broadening. Zero padding may interpolate it but cannot reduce its width.
    """
    z = np.asarray(energy_mev)*duration_ps/PLANCK_MEV_PS
    transform = .5*np.sinc(z)+.25*(np.sinc(z-1)+np.sinc(z+1))
    return (duration_ps/PLANCK_MEV_PS)*(8/3)*transform**2


def hann_fwhm_mev(duration_ps):
    from scipy.optimize import brentq
    half = hann_energy_kernel(0., duration_ps)/2
    return 2*brentq(lambda energy: hann_energy_kernel(energy, duration_ps)-half,
                   .1*PLANCK_MEV_PS/duration_ps, 2*PLANCK_MEV_PS/duration_ps, xtol=1e-12)


def periodogram_modes(coordinates, sample_interval_s):
    """Two-sided group-summed PSD; axis h*f in meV, with no quantum correction."""
    coordinates = np.asarray(coordinates)
    if coordinates.ndim == 1:
        coordinates = coordinates[:, None]
    count = len(coordinates)
    window = np.hanning(count+1)[:-1]  # periodic Hann, exact segment duration N dt
    transformed = np.fft.fft((coordinates-np.mean(coordinates, axis=0))*window[:, None], axis=0)
    power = np.sum(abs(transformed)**2, axis=1)*sample_interval_s/np.sum(window**2)
    energy = np.fft.fftfreq(count, sample_interval_s)*constants.h/(1e-3*constants.elementary_charge)
    order = np.argsort(energy)
    return energy[order], power[order]


def fit_windowed_mode(energy, power, reference_mev, duration_ps, search_mev=8.):
    """Lorentzian convolved with known Hann window plus a constant background.

    Width is energy FWHM. A fitted sub-resolution width is not a lifetime.
    The caller must assess resolution, independent trajectories, blocks, modes,
    timestep and size before labeling any linewidth resolved.
    """
    energy, power = np.asarray(energy), np.asarray(power)
    select = (energy > 0) & (abs(energy-reference_mev) < search_mev)
    x, y = energy[select], power[select]
    if len(x) < 8 or np.max(y) <= 0:
        raise ValueError('insufficient positive mode spectrum for a line fit')
    scale = np.max(y)
    y = y/scale
    step = np.min(np.diff(x))/16
    fine = np.arange(x[0]-search_mev, x[-1]+search_mev+step/2, step)
    offsets = (np.arange(len(fine))-len(fine)//2)*step
    kernel = hann_energy_kernel(offsets, duration_ps)
    peak = float(x[np.argmax(y)])
    def prediction(parameters):
        center, width, amplitude, background = parameters
        gamma = width/2
        lorentzian = gamma/(np.pi*((fine-center)**2+gamma**2))
        broadened = fftconvolve(lorentzian, kernel, mode='same')*step
        return amplitude*np.interp(x, fine, broadened)+background
    fit = least_squares(lambda parameters: prediction(parameters)-y,
        x0=[peak, max(.3, hann_fwhm_mev(duration_ps)), 1., 0.],
        bounds=([reference_mev-search_mev, .001, 0., 0.], [reference_mev+search_mev, 10., 100., .5]),
        xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=1000)
    return {'center_mev': float(fit.x[0]), 'fitted_intrinsic_FWHM_mev': float(fit.x[1]),
            'raw_peak_bin_mev': peak, 'fit_relative_RMS_residual': float(np.sqrt(np.mean((prediction(fit.x)-y)**2))),
            'fit_success': bool(fit.success), 'Hann_power_window_FWHM_mev': hann_fwhm_mev(duration_ps),
            'energy_bin_mev': float(np.min(np.diff(energy))), 'segment_duration_ps': duration_ps,
            'fit_background_relative_peak': float(fit.x[3])}


def project_normal_coordinates(values, positions_m, masses_kg, model, q_fractional, eigenvectors):
    """Atomic-gauge normal coordinates, sum_i sqrt(M_i) e_i* . u_i.

    eigenvectors are unit-cell mass-weighted columns. Periodic atoms are
    explicitly mapped to the model basis and normalized by sqrt(Ncells).
    """
    from .lattice_dynamics import primitive_atom_mapping
    basis, _ = primitive_atom_mapping(model, positions_m)
    cells = len(positions_m)/len(model.positions_m)
    qcart = np.asarray(q_fractional) @ model.reciprocal_m_inv
    modes = np.asarray(eigenvectors).reshape(len(model.positions_m), 3, -1)[basis]
    modes = modes*np.exp(1j*(positions_m@qcart))[:, None, None]/np.sqrt(cells)
    return np.einsum('tia,iam,i->tm', values, modes.conj(), np.sqrt(masses_kg))
