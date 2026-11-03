"""Finite-window mode-line diagnostics with an explicit non-identifiability gate.

Energy axes use h*f in meV, not hbar times cycles/second. Returned widths are
energy FWHM; they are never converted to lifetimes by this fitting routine.
"""
import numpy as np
from scipy import constants
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

PLANCK_MEV_PS = constants.h / (1e-3 * constants.elementary_charge) * 1e12


def window_power_kernel(energy_mev, duration_ps, window):
    z = np.asarray(energy_mev) * duration_ps / PLANCK_MEV_PS
    if window == 'hann':
        transform = .5*np.sinc(z) + .25*(np.sinc(z-1)+np.sinc(z+1))
        return duration_ps/PLANCK_MEV_PS * (8/3)*transform**2
    if window == 'rectangular':
        return duration_ps/PLANCK_MEV_PS * np.sinc(z)**2
    raise ValueError('window must be hann or rectangular')


def window_fwhm_mev(duration_ps, window):
    from scipy.optimize import brentq
    half = window_power_kernel(0., duration_ps, window)/2
    return 2*brentq(lambda e: window_power_kernel(e, duration_ps, window)-half,
                   .01*PLANCK_MEV_PS/duration_ps, 2*PLANCK_MEV_PS/duration_ps)


def mode_periodogram(coordinates, interval_s, window='hann'):
    values = np.asarray(coordinates)
    if values.ndim == 1:
        values = values[:, None]
    n = len(values)
    if interval_s <= 0 or n < 4:
        raise ValueError('positive cadence and at least four coordinates required')
    weights = np.hanning(n+1)[:-1] if window == 'hann' else np.ones(n)
    if window not in ('hann', 'rectangular'):
        raise ValueError('unknown window')
    transformed = np.fft.fft((values-values.mean(axis=0))*weights[:, None], axis=0)
    power = np.sum(abs(transformed)**2, axis=1)*interval_s/np.sum(weights**2)
    energy = np.fft.fftfreq(n, interval_s)*constants.h/(1e-3*constants.elementary_charge)
    order = np.argsort(energy)
    return energy[order], power[order]


def fit_mode_profile(energy, power, reference_mev, duration_ps, window='hann',
                     search_mev=4., line_shape='lorentzian'):
    """Fit a known window-convolved line; expose a width profile, not errors.

    A fit whose zero-width profile costs essentially the same as its optimum
    is not identifiable, regardless of optimizer convergence. Profile costs
    are diagnostics; periodogram bins are correlated and these are not chi²
    confidence intervals.
    """
    energy, power = np.asarray(energy), np.asarray(power)
    select = (energy > 0) & (abs(energy-reference_mev) < search_mev)
    x, raw = energy[select], power[select]
    if len(x) < 8 or raw.max() <= 0:
        raise ValueError('insufficient positive spectrum')
    y = raw/raw.max()
    # Resolve the narrowest tested intrinsic line; grid spacing remains a
    # numerical parameter independent of the physical trajectory duration.
    spacing = min(np.min(np.diff(x))/16, .002)
    fine = np.arange(x[0]-search_mev, x[-1]+search_mev+spacing/2, spacing)
    offsets = (np.arange(len(fine))-len(fine)//2)*spacing
    kernel = window_power_kernel(offsets, duration_ps, window)
    def prediction(parameters, fixed_width=None):
        center, width, amplitude, background = parameters if fixed_width is None else [parameters[0], fixed_width, parameters[1], parameters[2]]
        if width == 0:
            convolved = window_power_kernel(x-center, duration_ps, window)
        else:
            if line_shape == 'lorentzian':
                gamma = width/2
                line = gamma/(np.pi*((fine-center)**2+gamma**2))
            elif line_shape == 'gaussian':
                sigma = width/(2*np.sqrt(2*np.log(2)))
                line = np.exp(-.5*((fine-center)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
            else:
                raise ValueError('unknown line shape')
            convolved = np.interp(x, fine, fftconvolve(line, kernel, mode='same')*spacing)
        return amplitude*convolved+background
    peak = float(x[np.argmax(y)])
    init_width = max(.1, window_fwhm_mev(duration_ps, window))
    fit = least_squares(lambda p: prediction(p)-y, [peak, init_width, init_width, 0.],
        bounds=([reference_mev-search_mev, .01, 0., 0.], [reference_mev+search_mev, 10., 100., .5]),
        max_nfev=200, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    widths = np.unique(np.r_[0., .025, .05, .1, .25, .5, 1., 2., fit.x[1]])
    profiles = []
    for width in widths:
        fixed = least_squares(lambda p: prediction(p, width)-y,
            fit.x[[0, 2, 3]], bounds=([reference_mev-search_mev, 0., 0.], [reference_mev+search_mev, 100., .5]),
            max_nfev=100, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        profiles.append({'FWHM_mev': float(width), 'mean_squared_residual': float(np.mean((prediction(fixed.x, width)-y)**2))})
    return {'center_mev': float(fit.x[0]), 'candidate_FWHM_mev': float(fit.x[1]),
            'fit_success': bool(fit.success), 'fit_relative_RMS_residual': float(np.sqrt(np.mean((prediction(fit.x)-y)**2))),
            'window': window, 'window_FWHM_mev': window_fwhm_mev(duration_ps, window),
            'duration_ps': duration_ps, 'search_mev': search_mev, 'line_shape': line_shape,
            'profile': profiles, 'fine_grid_mev': spacing, 'lifetime_population_ps': None,
            'profile_interpretation': 'sensitivity diagnostic, not independent-bin likelihood or confidence interval'}


def linewidth_gate(rows):
    """Original B11 width, spread and shape gates, across declared choices."""
    if not rows:
        raise ValueError('no linewidth diagnostics')
    widths = np.array([r['candidate_FWHM_mev'] for r in rows])
    width = float(np.median(widths))
    spread = float(np.max(abs(widths-width)))
    shape = all(r['fit_success'] and r['fit_relative_RMS_residual'] < .1 for r in rows)
    resolution = width > 2*max(r['window_FWHM_mev'] for r in rows)
    stable = spread < .5*width
    # This supplements, and cannot relax, the established gates.
    zero_excluded = all(r['profile'][0]['mean_squared_residual'] > 1.5*min(p['mean_squared_residual'] for p in r['profile']) for r in rows)
    passed = bool(shape and resolution and stable and zero_excluded)
    return {'status': 'RESOLVED_IN_TESTED_CLASSICAL_MODEL' if passed else 'NOT_RESOLVED',
            'candidate_median_FWHM_mev': width, 'maximum_choice_spread_mev': spread,
            'shape_pass': bool(shape), 'resolution_pass': bool(resolution), 'spread_pass': bool(stable),
            'profile_zero_width_excluded_diagnostic': bool(zero_excluded),
            'lifetime_population_ps': constants.hbar/(width*1e-3*constants.elementary_charge)*1e12 if passed else None,
            'convention': 'energy FWHM Gamma=2*hbar/tau_amplitude; tau_population=hbar/Gamma; no lifetime when unresolved'}
