"""Closed-form T=0 references and stable tails; independent of both numerics."""

import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson


def vacuum_amplitudes(xi: float, p: int) -> np.ndarray:
    n = np.arange(p + 1)
    if xi == 0:
        result = np.zeros(p + 1, dtype=np.complex128)
        result[0] = 1
        return result
    log_abs = -xi**2 / 2 + n * np.log(abs(xi)) - gammaln(n + 1) / 2
    return np.asarray(np.exp(log_abs) * (-1j * np.sign(xi))**n, dtype=np.complex128)


def poisson_weights(eta: float, p: int) -> np.ndarray:
    if eta == 0:
        weights = np.zeros(p + 1, dtype=np.float64)
        weights[0] = 1
        return weights
    n = np.arange(p + 1, dtype=np.float64)
    return np.exp(-eta + n * np.log(eta) - gammaln(n + 1))


def omitted_probability(eta: float, p: int) -> float:
    """Stable P(N>p), not 1 minus a rounded cumulative sum."""
    return float(poisson.sf(p, eta))


def omitted_inelastic_fraction(eta: float, p: int) -> float | None:
    """Only defined for p>=1; return zero by convention if no inelastic weight."""
    if p < 1:
        return None
    if eta == 0:
        return 0.0
    return omitted_probability(eta, p) / float(-np.expm1(-eta))


def adaptive_cutoff(eta: float, tail_tolerance: float) -> int:
    if not np.isfinite(eta) or eta < 0:
        raise ValueError("eta must be nonnegative and finite")
    if not np.isfinite(tail_tolerance) or not 0 < tail_tolerance < 1:
        raise ValueError("tail_tolerance must be in (0,1)")
    # Integer bracketing avoids fragile inverse-CDF rounding near probability 1.
    low, high = -1, max(1, int(np.ceil(eta)))
    while omitted_probability(eta, high) >= tail_tolerance:
        low, high = high, 2 * high + 1
    while high - low > 1:
        midpoint = (low + high) // 2
        if omitted_probability(eta, midpoint) < tail_tolerance:
            high = midpoint
        else:
            low = midpoint
    return high


def exact_correlation(eta: float, omega_rad_s: float, times_s: np.ndarray) -> np.ndarray:
    return np.exp(eta * np.expm1(-1j * omega_rad_s * times_s))
