"""Exact spectral evolution of d(phi)/dz = i/(2 k) laplacian_perp(phi).

All lengths are metres, energies joules, and angular wavenumbers rad/m.
Wave arrays have shape (..., ny, nx); leading axes are independent waves or
channels. No potential, channel coupling, or longitudinal carrier is included.

NumPy's forward DFT uses exp(-2 pi i (j p / nx + l q / ny)) and its inverse
uses the positive sign with a factor 1/(nx ny). Thus a positive transverse
Fourier mode acquires exp(-i z (kx**2 + ky**2)/(2 k)). The physical coordinates
are centred, but the FFT and frequency arrays remain in unshifted order.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.constants import c, hbar, m_e


def _positive_finite(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def electron_wavenumber(kinetic_energy_j: float) -> float:
    """Relativistic k = sqrt(T (T + 2 m_e c^2))/(hbar c), in rad/m."""
    _positive_finite("kinetic_energy_j", kinetic_energy_j)
    return float(np.sqrt(kinetic_energy_j * (kinetic_energy_j + 2 * m_e * c**2)) / (hbar * c))


@dataclass(frozen=True)
class TransverseGrid:
    """Periodic cell sampled on (ny, nx), with x along the last axis."""

    lx_m: float
    ly_m: float
    nx: int
    ny: int

    def __post_init__(self) -> None:
        _positive_finite("lx_m", self.lx_m)
        _positive_finite("ly_m", self.ly_m)
        for name in ("nx", "ny"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")

    @property
    def shape(self) -> tuple[int, int]:
        return self.ny, self.nx

    @property
    def dx_m(self) -> float:
        return self.lx_m / self.nx

    @property
    def dy_m(self) -> float:
        return self.ly_m / self.ny

    @property
    def pixel_area_m2(self) -> float:
        return self.dx_m * self.dy_m

    @property
    def x_m(self) -> np.ndarray:
        return (np.arange(self.nx, dtype=np.float64) - self.nx // 2) * self.dx_m

    @property
    def y_m(self) -> np.ndarray:
        return (np.arange(self.ny, dtype=np.float64) - self.ny // 2) * self.dy_m

    @property
    def xy_m(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x_m, self.y_m, indexing="xy")


def _wave_array(wave: np.ndarray, grid: TransverseGrid) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.complex128)
    if wave.ndim < 2 or wave.shape[-2:] != grid.shape:
        raise ValueError(f"wave must have trailing shape {grid.shape}, got {wave.shape}")
    return wave


def grid_norm_squared(wave: np.ndarray, grid: TransverseGrid) -> np.ndarray:
    """Return dx dy sum(|wave|^2), separately for each leading index."""
    wave = _wave_array(wave, grid)
    return grid.pixel_area_m2 * np.sum(np.abs(wave) ** 2, axis=(-2, -1), dtype=np.float64)


class FreePropagator:
    """Reusable FFT propagator; all channels share k and evolve independently."""

    def __init__(self, grid: TransverseGrid, wavenumber_m_inv: float):
        _positive_finite("wavenumber_m_inv", wavenumber_m_inv)
        self.grid = grid
        self.wavenumber_m_inv = float(wavenumber_m_inv)
        kx = 2 * np.pi * np.fft.fftfreq(grid.nx, d=grid.dx_m)
        ky = 2 * np.pi * np.fft.fftfreq(grid.ny, d=grid.dy_m)
        self._transverse_k_squared = ky[:, None] ** 2 + kx[None, :] ** 2

    def multiplier(self, distance_m: float) -> np.ndarray:
        """Return exp[-i distance (kx^2+ky^2)/(2 k)] in unshifted order."""
        if not np.isfinite(distance_m):
            raise ValueError("distance_m must be finite")
        return np.exp(-1j * distance_m * self._transverse_k_squared / (2 * self.wavenumber_m_inv))

    def propagate(self, wave: np.ndarray, distance_m: float) -> np.ndarray:
        """Return a complex128 wave without modifying the input; allow negative z."""
        wave = _wave_array(wave, self.grid)
        multiplier = self.multiplier(distance_m)
        if distance_m == 0:
            return wave.copy()
        spectrum = np.fft.fft2(wave, axes=(-2, -1))
        return np.fft.ifft2(spectrum * multiplier, axes=(-2, -1))
