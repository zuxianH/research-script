"""Unrenormalized finite-temperature oscillator transition sums, in SI units.

These routines average probabilities of the Benchmark 3 phase-operator matrix
elements. They do not integrate electron/channel propagation equations.
"""

from dataclasses import dataclass

import numpy as np
from scipy.constants import Boltzmann, hbar
from scipy.integrate import trapezoid

from .harmonic_oscillator import _finite, _integer


@dataclass(frozen=True)
class ThermalOscillatorState:
    quantum_energy_j: float
    temperature_k: float

    def __post_init__(self):
        _finite("quantum_energy_j", self.quantum_energy_j, positive=True)
        _finite("temperature_k", self.temperature_k)
        if self.temperature_k < 0:
            raise ValueError("temperature_k must be nonnegative")

    @property
    def x(self) -> float:
        return np.inf if self.temperature_k == 0 else self.quantum_energy_j / (Boltzmann * self.temperature_k)

    @property
    def r(self) -> float:
        return float(np.exp(-self.x))

    @property
    def mean_occupation(self) -> float:
        # Avoid exp(x) overflow at low T and 1-exp(-x) cancellation at high T.
        return float(self.r / -np.expm1(-self.x))

    def probabilities(self, m_max: int) -> np.ndarray:
        _integer("m_max", m_max)
        if self.temperature_k == 0:
            probabilities = np.zeros(m_max + 1)
            probabilities[0] = 1
            return probabilities
        return -np.expm1(-self.x) * np.exp(-self.x * np.arange(m_max + 1))

    def omitted_probability(self, m_max: int) -> float:
        _integer("m_max", m_max)
        return float(np.exp(-self.x * (m_max + 1)))

    def initial_cutoff(self, tolerance: float = 1e-13) -> int:
        if not np.isfinite(tolerance) or not 0 < tolerance < 1:
            raise ValueError("initial tolerance must be in (0,1)")
        if self.temperature_k == 0:
            return 0
        # Retain m=1 even at low positive T to quantify representable tiny gain.
        cutoff = max(1, int(np.floor(-np.log(tolerance) / self.x)))
        while self.omitted_probability(cutoff) >= tolerance:
            cutoff += 1
        return cutoff


@dataclass(frozen=True)
class ThermalTransitionSum:
    ell: np.ndarray
    weights: np.ndarray
    coherent_amplitude: complex
    retained_initial_weight: float
    final_omitted_weight_residual: float
    final_column_norm_residuals: np.ndarray


def transition_sum(matrix: np.ndarray, initial_probabilities: np.ndarray, n_max: int) -> ThermalTransitionSum:
    """Group p_m |F_nm|^2 at signed ell=n-m; never average amplitudes.

    matrix must contain every retained final row and initial diagonal element.
    The signed final-tail residual is recorded without clipping roundoff.
    """
    matrix = np.asarray(matrix, dtype=np.complex128)
    probabilities = np.asarray(initial_probabilities, dtype=np.float64)
    _integer("n_max", n_max)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("matrix must be a finite two-dimensional array")
    if probabilities.ndim != 1 or not probabilities.size or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("initial probabilities must be a finite nonnegative vector")
    m_size = probabilities.size
    if matrix.shape[0] <= max(n_max, m_size - 1) or matrix.shape[1] < m_size:
        raise ValueError("matrix does not contain retained final states and initial diagonals")
    squared = np.abs(matrix[:n_max+1, :m_size])**2
    changes = np.arange(n_max + 1)[:, None] - np.arange(m_size)[None, :]
    ell = np.arange(-(m_size - 1), n_max + 1)
    weights = np.bincount((changes + m_size - 1).ravel(), weights=(squared * probabilities).ravel(), minlength=ell.size)
    residuals = 1 - np.sum(squared, axis=0)
    return ThermalTransitionSum(
        ell, weights, complex(probabilities @ np.diag(matrix[:m_size, :m_size])),
        float(np.sum(probabilities)), float(probabilities @ residuals), residuals,
    )


def final_state_cutoff(matrix: np.ndarray, probabilities: np.ndarray, tolerance: float = 1e-12) -> int:
    """Select n_max using positive high-basis tails; converge the basis separately.

    Unlike subtracting a nearly unit norm, this selection remains useful near
    machine precision. transition_sum separately reports the requested signed
    norm-deficit estimate, including its floating-point residual.
    """
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("final tolerance must be in (0,1)")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.ndim != 2 or probabilities.ndim != 1 or matrix.shape[1] < probabilities.size:
        raise ValueError("matrix must contain all initial-state columns")
    final_weights = np.abs(matrix[:, :probabilities.size])**2 @ probabilities
    tails = np.r_[np.cumsum(final_weights[:0:-1])[::-1], 0.0]
    return int(np.flatnonzero(tails < tolerance)[0])


def restrict_lines(ell: np.ndarray, weights: np.ndarray, loss_max: int, gain_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract a signed window, filling unsupported transitions with zero."""
    _integer("loss_max", loss_max)
    _integer("gain_max", gain_max)
    reported = np.arange(-gain_max, loss_max + 1)
    indices = reported - int(ell[0])
    retained = np.zeros(reported.size, dtype=np.float64)
    supported = (indices >= 0) & (indices < len(weights))
    retained[supported] = weights[indices[supported]]
    return reported, retained


def line_correlation(ell: np.ndarray, weights: np.ndarray, quantum_energy_j: float, times_s: np.ndarray) -> np.ndarray:
    """Sum W_ell exp(-i ell omega t), without renormalizing finite cutoffs."""
    return np.exp(-1j * np.asarray(times_s)[..., None] * (quantum_energy_j / hbar) * ell) @ weights


def line_statistics(ell: np.ndarray, weights: np.ndarray, eta: float, n_bar: float, quantum_energy_j: float) -> dict:
    """Raw signed-change moments; variance is accepted in the normalized limit."""
    weight = float(np.sum(weights))
    mean = float(ell @ weights)
    second = float(ell**2 @ weights)
    variance = second - mean**2
    exact_variance = eta * (2 * n_bar + 1)
    mean_scale = abs(eta) if eta != 0 else 1.0
    variance_scale = abs(exact_variance) if exact_variance != 0 else 1.0
    return {
        "retained_weight": weight, "total_weight_error": abs(weight - 1),
        "raw_mean_ell": mean, "raw_second_moment_ell": second,
        "variance_from_raw_moments": variance,
        "mean_error_scaled": abs(mean - eta) / mean_scale,
        "variance_error_scaled": abs(variance - exact_variance) / variance_scale,
        "raw_mean_energy_loss_j": mean * quantum_energy_j,
        "exact_mean_energy_loss_j": eta * quantum_energy_j,
        "mean_energy_error_scaled": abs(mean - eta) / mean_scale,
    }


def broaden_signed_spectrum(ell: np.ndarray, weights: np.ndarray, quantum_energy_j: float, fwhm_j: float) -> tuple[dict, dict]:
    """Normalized Gaussian kernels on a signed SI energy grid, density in J^-1.

    Eight sigmas beyond each endpoint and spacing <=FWHM/16. No cropped window
    or truncated line distribution is normalized. Negative-energy discrete
    transitions and negative-energy Gaussian tails have distinct meanings.
    """
    _finite("quantum_energy_j", quantum_energy_j, positive=True)
    _finite("fwhm_j", fwhm_j, positive=True)
    ell, weights = np.asarray(ell), np.asarray(weights, dtype=np.float64)
    if ell.ndim != 1 or not ell.size or weights.shape != ell.shape or not np.all(np.isfinite(ell)) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("ell and nonnegative weights must be finite equal-length vectors")
    sigma = fwhm_j / (2 * np.sqrt(2 * np.log(2)))
    energies = ell * quantum_energy_j
    start, stop = float(np.min(energies) - 8 * sigma), float(np.max(energies) + 8 * sigma)
    points = max(257, int(np.ceil((stop - start) / (fwhm_j / 16))) + 1)
    if points > 2_000_000:
        raise ValueError("spectrum grid exceeds two million points; increase FWHM")
    grid = np.linspace(start, stop, points)
    parts = {name: np.zeros_like(grid) for name in ("elastic", "loss", "gain")}
    for change, center, weight in zip(ell, energies, weights):
        name = "elastic" if change == 0 else "loss" if change > 0 else "gain"
        parts[name] += weight * np.exp(-0.5 * ((grid - center) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)
    total = sum(parts.values())
    area = float(trapezoid(total, grid))
    expected = float(np.sum(weights))
    # Every line center is >=8 sigma from either end, giving this global bound.
    from scipy.special import erfc
    bound = float(expected * erfc(8 / np.sqrt(2)))
    arrays = {"energy_grid_j": grid, "total_density_per_j": total, **{f"{name}_density_per_j": values for name, values in parts.items()}}
    return arrays, {
        "fwhm_j": fwhm_j, "sigma_j": float(sigma), "samples": points,
        "energy_min_j": start, "energy_max_j": stop, "density_units": "J^-1",
        "integrated_weight": area, "retained_line_weight": expected,
        "absolute_area_error": abs(area - expected), "finite_gaussian_window_omission_upper_bound": bound,
        "area_error_including_window_bound": abs(area - expected) + bound,
        "component_area_errors": {name: abs(float(trapezoid(values, grid)) - float(np.sum(weights[ell == 0 if name == "elastic" else ell > 0 if name == "loss" else ell < 0]))) for name, values in parts.items()},
        "normalization_applied": False, "deliberate_plot_crop": False,
    }
