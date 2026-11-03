"""Probability-preserving diffraction and fixed-pixel elastic observables."""

import numpy as np

from yoshioka import TransverseGrid


def diffraction_probabilities(wave_si: np.ndarray, grid: TransverseGrid) -> np.ndarray:
    """Probability per reciprocal pixel; sum equals the transverse grid norm.

    For the unnormalized forward DFT F, I = Lx Ly |F(phi)|²/(Nx Ny)².
    The incident plane wave therefore has a single zero-angle pixel of one.
    No exit-wave or diffraction-pattern renormalization is performed.
    """
    return grid.lx_m * grid.ly_m / (grid.nx * grid.ny)**2 * np.abs(np.fft.fft2(wave_si))**2


def angular_grid_mrad(grid: TransverseGrid, wavelength_m: float):
    """Use cycles/m and theta = lambda*f, equivalently k_perp/k."""
    theta_x = wavelength_m * np.fft.fftfreq(grid.nx, grid.dx_m) * 1e3
    theta_y = wavelength_m * np.fft.fftfreq(grid.ny, grid.dy_m) * 1e3
    return np.meshgrid(theta_x, theta_y, indexing="xy")


def angular_masks(grid: TransverseGrid, wavelength_m: float) -> dict[str, np.ndarray]:
    tx, ty = angular_grid_mrad(grid, wavelength_m)
    radius = np.hypot(tx, ty)
    return {
        "annulus_0_10_mrad": radius < 10,
        "annulus_10_30_mrad": (radius >= 10) & (radius < 30),
        "annulus_30_50_mrad": (radius >= 30) & (radius <= 50),
        "region_up_to_50_mrad": radius <= 50,
    }


def select_reference_beams(diffraction: np.ndarray, grid: TransverseGrid, wavelength_m: float, count: int = 6) -> list[dict]:
    """Select strong nonzero reference beams once; retain one of each ±g pair."""
    masks = angular_masks(grid, wavelength_m)
    tx, ty = angular_grid_mrad(grid, wavelength_m)
    candidates = []
    for iy, ix in np.argwhere(masks["region_up_to_50_mrad"]):
        h = int(round(np.fft.fftfreq(grid.nx)[ix] * grid.nx))
        k = int(round(np.fft.fftfreq(grid.ny)[iy] * grid.ny))
        if h < 0 or (h == 0 and k <= 0) or diffraction[iy, ix] < 1e-8:
            continue
        candidates.append({
            "h": h, "k": k, "theta_mrad": float(np.hypot(tx[iy, ix], ty[iy, ix])),
            "selection_reference_intensity": float(diffraction[iy, ix]),
        })
    candidates.sort(key=lambda beam: (-beam["selection_reference_intensity"], beam["h"], beam["k"]))
    if len(candidates) < count:
        raise ValueError("not enough nonzero reference beams inside 50 mrad")
    return candidates[:count]


def intensity_observables(diffraction: np.ndarray, grid: TransverseGrid, wavelength_m: float, beams: list[dict]) -> dict[str, float]:
    masks = angular_masks(grid, wavelength_m)
    values = {"transmitted_0_0": float(diffraction[0, 0])}
    for name, mask in masks.items():
        if name.startswith("annulus"):
            values[name] = float(np.sum(diffraction[mask]))
    for beam in beams:
        h, k = beam["h"], beam["k"]
        values[f"beam_{h}_{k}"] = float(diffraction[k % grid.ny, h % grid.nx])
    return values


def compare_observables(ours: dict, reference: dict, relative_tolerance: float = 0.01, weak_threshold: float = 1e-6, absolute_tolerance: float = 1e-8) -> dict:
    """Use absolute probability errors when the reference observable is weak."""
    comparisons = {}
    for name, value in reference.items():
        absolute = abs(ours[name] - value)
        significant = abs(value) >= weak_threshold
        relative = absolute / abs(value) if significant else None
        comparisons[name] = {
            "ours": ours[name], "reference": value, "absolute_error": absolute,
            "relative_error": relative, "significant_reference": significant,
            "passed": bool(relative < relative_tolerance if significant else absolute < absolute_tolerance),
        }
    return {
        "observables": comparisons,
        "max_significant_relative_error": max((entry["relative_error"] for entry in comparisons.values() if entry["significant_reference"]), default=0.0),
        "max_weak_absolute_error": max((entry["absolute_error"] for entry in comparisons.values() if not entry["significant_reference"]), default=0.0),
        "passed": all(entry["passed"] for entry in comparisons.values()),
    }
