"""Independent Skellam/Bessel reference for the harmonic thermal response.

No oscillator matrix elements or transition-sum code is used here. A positive
Poisson convolution evaluates signed tails without subtracting a CDF from one.
"""

import numpy as np
from scipy.constants import Boltzmann
from scipy.special import gammaln, ive, logsumexp
from scipy.stats import poisson


def reference_parameters(eta: float, quantum_j: float, temperature_k: float) -> dict:
    if not np.isfinite(eta) or eta < 0 or not np.isfinite(quantum_j) or quantum_j <= 0 or not np.isfinite(temperature_k) or temperature_k < 0:
        raise ValueError("eta, energy, and temperature must be physical finite values")
    x = None if temperature_k == 0 else quantum_j / (Boltzmann * temperature_k)
    n_bar = 0.0 if x is None else float(np.exp(-x) / -np.expm1(-x))
    return {"x": x, "n_bar": n_bar, "lambda_plus": eta * (n_bar + 1), "lambda_minus": eta * n_bar,
            "bose_occupation_underflow": bool(temperature_k > 0 and n_bar == 0)}


def _log_bessel_series(order: int, log_product: float) -> float:
    """Log of sum_j (lambda+ lambda-)^j order!/[j! (j+order)!].

    Used when the scaled Bessel value underflows: its leading power is removed
    analytically before evaluation, avoiding a 0 * infinity indeterminacy.
    """
    terms = [0.0]
    log_term = 0.0
    for j in range(1, 10001):
        log_term += log_product - np.log(j) - np.log(j + order)
        terms.append(log_term)
        if log_term < max(terms) - 40 and log_term < terms[-2]:
            return float(logsumexp(terms))
    raise ValueError("Bessel reference series did not converge")


def exact_log_weights(ell: np.ndarray, eta: float, quantum_j: float, temperature_k: float) -> np.ndarray:
    ell = np.asarray(ell)
    if np.any(ell != np.floor(ell)):
        raise ValueError("ell must contain integers")
    parameters = reference_parameters(eta, quantum_j, temperature_k)
    plus, minus = parameters["lambda_plus"], parameters["lambda_minus"]
    result = np.full(ell.shape, -np.inf)
    if eta == 0:
        result[ell == 0] = 0
        return result
    if minus == 0:
        valid = ell >= 0
        result[valid] = -plus + ell[valid] * np.log(plus) - gammaln(ell[valid] + 1)
        return result
    order = np.abs(ell).astype(int)
    z = 2 * np.sqrt(plus) * np.sqrt(minus)
    scaled = ive(order, z)
    valid = scaled > 0
    # For small z, removing the leading power also avoids cancellation of very
    # large opposite log terms, even before the Bessel function underflows.
    use_series = (~valid) | (z < 1e-5)
    result[~use_series] = -(plus + minus) + z + np.log(scaled[~use_series]) + ell[~use_series] * (np.log(plus) - np.log(minus)) / 2
    for index in np.ndindex(ell.shape):
        if use_series[index]:
            k = order[index]
            dominant = plus if ell[index] >= 0 else minus
            result[index] = -(plus + minus) + k * np.log(dominant) - gammaln(k + 1) + _log_bessel_series(int(k), np.log(plus) + np.log(minus))
    return result


def exact_weights(ell: np.ndarray, eta: float, quantum_j: float, temperature_k: float) -> np.ndarray:
    return np.exp(exact_log_weights(ell, eta, quantum_j, temperature_k))


def _poisson_cutoff(mean: float, tolerance: float) -> int:
    high = max(0, int(np.ceil(mean)))
    while poisson.sf(high, mean) >= tolerance:
        high = 2 * high + 1
    low = -1
    while high - low > 1:
        mid = (high + low) // 2
        if poisson.sf(mid, mean) < tolerance:
            high = mid
        else:
            low = mid
    return high


def one_sided_tail(limit: int, positive_mean: float, negative_mean: float, outer_tolerance: float = 1e-18) -> tuple[float, float]:
    """P(X-Y>limit), using sum_y Pois(y;mu-) SF_Pois(limit+y;mu+).

    Return the positive sum and an upper bound on its omitted outer series.
    The bound is tightened by monotonicity of the inner survival function.
    """
    if positive_mean == 0:
        return 0.0, 0.0
    if negative_mean == 0:
        return float(poisson.sf(limit, positive_mean)), 0.0
    cutoff = _poisson_cutoff(negative_mean, outer_tolerance)
    y = np.arange(cutoff + 1)
    value = float(poisson.pmf(y, negative_mean) @ poisson.sf(limit + y, positive_mean))
    bound = float(poisson.sf(cutoff, negative_mean) * poisson.sf(limit + cutoff + 1, positive_mean))
    return value, bound


def exact_window_tail(loss_max: int, gain_max: int, eta: float, quantum_j: float, temperature_k: float) -> dict:
    parameters = reference_parameters(eta, quantum_j, temperature_k)
    plus, minus = parameters["lambda_plus"], parameters["lambda_minus"]
    loss, loss_bound = one_sided_tail(loss_max, plus, minus)
    gain, gain_bound = one_sided_tail(gain_max, minus, plus)
    return {"loss_tail": loss, "gain_tail": gain, "omitted_weight": loss + gain,
            "tail_evaluation_error_bound": loss_bound + gain_bound,
            "omitted_weight_upper_bound": loss + gain + loss_bound + gain_bound}


def energy_window(eta: float, quantum_j: float, temperature_k: float, tolerance: float = 1e-12) -> tuple[int, int, dict]:
    if not 0 < tolerance < 1:
        raise ValueError("energy-window tolerance must be in (0,1)")
    parameters = reference_parameters(eta, quantum_j, temperature_k)
    plus, minus = parameters["lambda_plus"], parameters["lambda_minus"]
    if eta == 0:
        return 0, 0, exact_window_tail(0, 0, eta, quantum_j, temperature_k)
    def find(positive, negative, minimum):
        # A Poisson upper-tail bound provides a bracket; then minimize by search.
        high = max(minimum, _poisson_cutoff(positive, tolerance / 2))
        low = minimum - 1
        while high - low > 1:
            mid = (high + low) // 2
            value, bound = one_sided_tail(mid, positive, negative)
            if value + bound < tolerance / 2:
                high = mid
            else:
                low = mid
        return high
    loss = find(plus, minus, 1)
    gain = 0 if minus == 0 else find(minus, plus, 1)
    return loss, gain, exact_window_tail(loss, gain, eta, quantum_j, temperature_k)


def thermal_correlation(times_s: np.ndarray, omega_rad_s: float, eta: float, quantum_j: float, temperature_k: float) -> np.ndarray:
    parameters = reference_parameters(eta, quantum_j, temperature_k)
    phase = omega_rad_s * np.asarray(times_s)
    return np.exp(parameters["lambda_plus"] * np.expm1(-1j * phase) + parameters["lambda_minus"] * np.expm1(1j * phase))


def elastic_and_leading_terms(eta: float, quantum_j: float, temperature_k: float) -> dict:
    parameters = reference_parameters(eta, quantum_j, temperature_k)
    n_bar = parameters["n_bar"]
    coherent = float(np.exp(-eta * (n_bar + 0.5)))
    zero_energy = float(exact_weights(np.array([0]), eta, quantum_j, temperature_k)[0])
    product = parameters["lambda_plus"] * parameters["lambda_minus"]
    if product < 1e-3:
        # I0(2 sqrt(product))-1, without subtracting two nearly equal numbers.
        correction, term = product, product
        for order in range(2, 12):
            term *= product / order**2
            correction += term
        difference = coherent**2 * correction
    else:
        difference = zero_energy - coherent**2
    return {"thermal_Debye_Waller_exponent": eta * (n_bar + 0.5),
            "coherent_amplitude": coherent, "coherent_weight": coherent**2,
            "zero_energy_weight": zero_energy,
            "zero_energy_minus_coherent_weight_stable": difference,
            "leading_loss_one_phonon": coherent**2 * parameters["lambda_plus"],
            "leading_gain_one_phonon": coherent**2 * parameters["lambda_minus"]}
