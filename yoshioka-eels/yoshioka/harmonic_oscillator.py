"""Independent numerical matrix elements of exp[-i Q u] for one oscillator.

Physical inputs use kg, J, s, and angular wavevectors in m^-1. The dimensionless
coupling is xi=Q*u_zp, with no additional 2*pi. All matrices are complex128.
Neither numerical route uses the closed-form vacuum/Poisson coefficients.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.constants import atomic_mass, elementary_charge, h, hbar
from scipy.linalg import expm
from scipy.special import roots_hermite


def _integer(name: str, value: int, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite(name: str, value: float, positive: bool = False) -> None:
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")


@dataclass(frozen=True)
class HarmonicOscillator:
    mass_kg: float
    quantum_energy_j: float

    def __post_init__(self):
        _finite("mass_kg", self.mass_kg, positive=True)
        _finite("quantum_energy_j", self.quantum_energy_j, positive=True)

    @classmethod
    def from_amu_mev(cls, mass_amu: float, quantum_energy_mev: float):
        """1 u = scipy.constants.atomic_mass kg; 1 meV = 1e-3 e joules."""
        return cls(mass_amu * atomic_mass, quantum_energy_mev * 1e-3 * elementary_charge)

    @property
    def angular_frequency_rad_s(self) -> float:
        return self.quantum_energy_j / hbar

    @property
    def frequency_hz(self) -> float:
        return self.quantum_energy_j / h

    @property
    def zero_point_displacement_m(self) -> float:
        return float(np.sqrt(hbar / (2 * self.mass_kg * self.angular_frequency_rad_s)))

    @property
    def oscillator_length_m(self) -> float:
        """ell=sqrt(hbar/(M omega))=sqrt(2)*u_zp; x=u/ell."""
        return np.sqrt(2) * self.zero_point_displacement_m

    def xi(self, q_m_inv: float) -> float:
        _finite("q_m_inv", q_m_inv)
        return float(q_m_inv * self.zero_point_displacement_m)

    def energy_losses_j(self, p: int) -> np.ndarray:
        """E_n-E_0=n*hbar*omega, including an elastic line at exactly zero."""
        _integer("p", p)
        return np.arange(p + 1, dtype=np.float64) * self.quantum_energy_j


def resolve_coupling(
    oscillator: HarmonicOscillator,
    q_m_inv: float | None = None,
    eta: float | None = None,
    consistency_rtol: float = 1e-12,
) -> tuple[float, float, float]:
    """Return (Q,xi,eta); eta-only uses Q>=0; unspecified input uses eta=0.1.

    A signed Q may be given with eta if consistent. Check eta relative to its
    own scale, including tiny couplings, rather than accepting a loose absolute
    discrepancy near zero. Q=0 is consistent only with eta=0.
    """
    _finite("consistency_rtol", consistency_rtol, positive=True)
    if eta is not None and (not np.isfinite(eta) or eta < 0):
        raise ValueError("eta must be nonnegative and finite")
    if q_m_inv is not None:
        xi = oscillator.xi(q_m_inv)
        implied_eta = xi * xi
        if not np.isfinite(implied_eta):
            raise ValueError("Q implies a nonfinite eta")
        if eta is not None and not np.isclose(implied_eta, eta, rtol=consistency_rtol, atol=0):
            raise ValueError("inconsistent simultaneous Q and eta specifications")
        return float(q_m_inv), xi, implied_eta
    eta = 0.1 if eta is None else float(eta)
    xi = float(np.sqrt(eta))
    return xi / oscillator.zero_point_displacement_m, xi, eta


def ladder_operators(working_dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """Finite ladder matrices. Their top-state commutator is not the identity."""
    _integer("working_dimension", working_dimension, 1)
    lower = np.zeros((working_dimension, working_dimension), dtype=np.complex128)
    indices = np.arange(working_dimension - 1)
    lower[indices, indices + 1] = np.sqrt(indices + 1)
    return lower, lower.conj().T.copy()


def fock_scattering_matrix(xi: float, working_dimension: int) -> np.ndarray:
    """Compute exp[-i xi (a_D+a_D†)] by SciPy's dense matrix exponential.

    D is a working dimension, not a physical retained cutoff. This finite matrix
    is unitary even when it is too small to approximate low-occupation elements
    of the infinite operator. Converge D before extracting a retained block.
    """
    _finite("xi", xi)
    lower, raise_ = ladder_operators(working_dimension)
    return np.asarray(expm(-1j * xi * (lower + raise_)), dtype=np.complex128)


def normalized_hermite_polynomials(x: np.ndarray, n_max: int) -> np.ndarray:
    """Return h_n=H_n/sqrt(2^n n!), using a normalized three-term recurrence.

    psi_n(u) = h_n(x)*exp(-x^2/2)/(pi^(1/4)*sqrt(ell)), x=u/ell.
    The quadrature weight is exp(-x^2); its integral is sqrt(pi).
    """
    _integer("n_max", n_max)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a finite one-dimensional array")
    values = np.empty((n_max + 1, x.size), dtype=np.float64)
    values[0] = 1
    if n_max:
        values[1] = np.sqrt(2) * x
    for n in range(1, n_max):
        values[n + 1] = np.sqrt(2 / (n + 1)) * x * values[n] - np.sqrt(n / (n + 1)) * values[n - 1]
    return values


def quadrature_scattering_block(xi: float, n_max: int, quadrature_order: int) -> np.ndarray:
    """Evaluate all <m|exp(-i Q u)|n>, m,n<=n_max, by coordinate overlaps.

    Integrate h_m(x) h_n(x) exp[-i sqrt(2) xi x] exp(-x^2)/sqrt(pi).
    The sqrt(2) comes from ell=sqrt(2)*u_zp and is not a wavevector conversion.
    No analytical amplitudes or Fock-basis exponential are used here.
    """
    _finite("xi", xi)
    _integer("n_max", n_max)
    _integer("quadrature_order", quadrature_order, 1)
    nodes, weights = roots_hermite(quadrature_order)
    hermites = normalized_hermite_polynomials(nodes, n_max)
    phase_weights = weights / np.sqrt(np.pi) * np.exp(-1j * np.sqrt(2) * xi * nodes)
    return np.asarray((hermites * phase_weights) @ hermites.T, dtype=np.complex128)


def quadrature_vacuum_amplitudes(xi: float, p: int, quadrature_order: int) -> np.ndarray:
    """Coordinate-space vacuum overlaps n=0..p, without a factorial formula."""
    _finite("xi", xi)
    _integer("p", p)
    _integer("quadrature_order", quadrature_order, 1)
    nodes, weights = roots_hermite(quadrature_order)
    hermites = normalized_hermite_polynomials(nodes, p)
    phase_weights = weights / np.sqrt(np.pi) * np.exp(-1j * np.sqrt(2) * xi * nodes)
    return np.asarray(hermites @ phase_weights, dtype=np.complex128)


def spectral_correlation(amplitudes: np.ndarray, energy_quantum_j: float, times_s: np.ndarray) -> np.ndarray:
    """Unrenormalized sum_n |A_n|^2 exp(-i n omega t) at a physical cutoff."""
    amplitudes = np.asarray(amplitudes, dtype=np.complex128)
    times_s = np.asarray(times_s, dtype=np.float64)
    _finite("energy_quantum_j", energy_quantum_j, positive=True)
    if amplitudes.ndim != 1 or not np.all(np.isfinite(amplitudes)) or not np.all(np.isfinite(times_s)):
        raise ValueError("amplitudes and times must be finite; amplitudes must be a vector")
    n = np.arange(amplitudes.size, dtype=np.float64)
    return np.asarray(np.exp(-1j * times_s[..., None] * (energy_quantum_j / hbar) * n) @ np.abs(amplitudes)**2, dtype=np.complex128)
