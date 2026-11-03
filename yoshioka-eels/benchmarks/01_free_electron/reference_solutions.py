"""Closed-form references; these do not use the numerical Fourier multiplier."""

import numpy as np
from scipy.special import erfc

from yoshioka import TransverseGrid, grid_norm_squared


def constant_wave(grid: TransverseGrid) -> np.ndarray:
    return np.full(grid.shape, 1 / np.sqrt(grid.lx_m * grid.ly_m), dtype=np.complex128)


def periodic_mode(
    grid: TransverseGrid, mx: int, my: int, k: float, z_m: float = 0
) -> np.ndarray:
    x, y = grid.xy_m
    kx, ky = 2 * np.pi * mx / grid.lx_m, 2 * np.pi * my / grid.ly_m
    # Evaluate the differential-equation eigenvalue directly, without fftfreq.
    return np.exp(1j * (kx * x + ky * y) - 1j * z_m * (kx**2 + ky**2) / (2 * k)) / np.sqrt(
        grid.lx_m * grid.ly_m
    )


def gaussian_wave(grid: TransverseGrid, width_m: float, k: float, z_m: float = 0) -> np.ndarray:
    x, y = grid.xy_m
    denominator = 1 + 1j * z_m / (k * width_m**2)
    return np.asarray(
        np.exp(-(x**2 + y**2) / (2 * width_m**2 * denominator))
        / (np.sqrt(np.pi) * width_m * denominator),
        dtype=np.complex128,
    )


def relative_wave_error(numerical: np.ndarray, exact: np.ndarray, grid: TransverseGrid) -> float:
    """Discrete L2 error, retaining the fixed physical global phase."""
    return float(np.sqrt(np.sum(grid_norm_squared(numerical - exact, grid)) / np.sum(grid_norm_squared(exact, grid))))


def radial_second_moment(wave: np.ndarray, grid: TransverseGrid) -> float:
    x, y = grid.xy_m
    return float(
        grid.pixel_area_m2 * np.sum((x**2 + y**2) * np.abs(wave) ** 2)
        / grid_norm_squared(wave, grid)
    )


def gaussian_resolution_estimates(grid: TransverseGrid, width_m: float, s: float) -> dict:
    """Continuum tails, not a fitted estimate of the numerical solution error.

    The real-space intensity is proportional to exp(-r^2 / width(z)^2).
    The initial spectral power is proportional to exp(-width(0)^2 q^2).
    Spectral magnitudes are conserved by free evolution. erfc avoids cancellation
    when the probability outside the box or Nyquist rectangle is extremely small.
    """
    width_z = width_m * np.sqrt(1 + s**2)
    bx, by = erfc(grid.lx_m / (2 * width_z)), erfc(grid.ly_m / (2 * width_z))
    qx, qy = erfc(width_m * np.pi / grid.dx_m), erfc(width_m * np.pi / grid.dy_m)
    return {
        "boundary_intensity_relative_to_peak": float(np.exp(-(min(grid.lx_m, grid.ly_m) / (2 * width_z)) ** 2)),
        "infinite_domain_probability_outside_box": float(bx + by - bx * by),
        "outside_box_l2_fraction": float(np.sqrt(bx + by - bx * by)),
        "initial_spectral_power_outside_nyquist_rectangle": float(qx + qy - qx * qy),
        "outside_nyquist_l2_fraction": float(np.sqrt(qx + qy - qx * qy)),
        "samples_per_initial_width_x": float(width_m / grid.dx_m),
        "samples_per_initial_width_y": float(width_m / grid.dy_m),
    }
