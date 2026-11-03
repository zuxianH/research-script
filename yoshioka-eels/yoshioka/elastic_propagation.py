"""Static elastic multislice, using SI units and Benchmark 1 free propagation.

H_el = -e V, so each *integrated* electrostatic slice V_proj (volt metre)
gives t = exp(+i e V_proj / (hbar v)). There is no additional factor of dz.
The conventional step is transmission followed by propagation over that slice.
Leading wave axes remain independent; no off-diagonal channel operators exist.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.constants import c, elementary_charge, hbar, m_e

from .free_propagation import FreePropagator, TransverseGrid, electron_wavenumber, grid_norm_squared


def electron_speed(kinetic_energy_j: float) -> float:
    """Relativistic electron speed in m/s for a positive kinetic energy in J."""
    if not np.isfinite(kinetic_energy_j) or kinetic_energy_j <= 0:
        raise ValueError("kinetic_energy_j must be positive and finite")
    gamma = 1 + kinetic_energy_j / (m_e * c**2)
    return float(c * np.sqrt(1 - gamma**-2))


def interaction_coefficient(kinetic_energy_j: float) -> float:
    """Positive e/(hbar v), in (volt metre)^-1."""
    return float(elementary_charge / (hbar * electron_speed(kinetic_energy_j)))


def antialias_aperture(
    grid: TransverseGrid, cutoff_fraction: float = 0.6666666, taper_fraction: float = 0.01
) -> np.ndarray:
    """Circular mask matching abTEM's configurable antialias convention.

    f_cut = cutoff_fraction / (2 max(dx,dy)); taper = taper_fraction/max(dx,dy).
    Frequencies are cycles/m, unshifted, on (y,x). A cosine taper descends from
    one at f_cut-taper to zero at f_cut. A zero taper gives a strict hard cutoff.
    This is numerical filtering, never a physical absorptive potential.
    """
    if not np.isfinite(cutoff_fraction) or cutoff_fraction <= 0:
        raise ValueError("cutoff_fraction must be positive and finite")
    if not np.isfinite(taper_fraction) or not 0 <= taper_fraction < cutoff_fraction / 2:
        raise ValueError("taper_fraction must be finite and in [0, cutoff_fraction/2)")
    spacing = max(grid.dx_m, grid.dy_m)
    cutoff, taper = cutoff_fraction / (2 * spacing), taper_fraction / spacing
    fx = np.fft.fftfreq(grid.nx, grid.dx_m)
    fy = np.fft.fftfreq(grid.ny, grid.dy_m)
    radius = np.hypot(fy[:, None], fx[None, :])
    if taper == 0:
        return (radius < cutoff).astype(np.float64)
    coordinate = np.clip((radius - cutoff + taper) / taper, 0, 1)
    return 0.5 * (1 + np.cos(np.pi * coordinate))


@dataclass
class ElasticResult:
    wave: np.ndarray
    # Includes the entrance value, then one value after each complete slice.
    # Shape (num_steps+1, ...independent leading wave axes).
    norm_history: np.ndarray


class ElasticPropagator:
    """Conventional transmission-then-propagation multislice without abTEM calls."""

    def __init__(
        self,
        grid: TransverseGrid,
        kinetic_energy_j: float,
        reciprocal_filter: np.ndarray | None = None,
    ):
        self.grid = grid
        self.free = FreePropagator(grid, electron_wavenumber(kinetic_energy_j))
        self.sigma = interaction_coefficient(kinetic_energy_j)
        self.reciprocal_filter = None
        if reciprocal_filter is not None:
            mask = np.asarray(reciprocal_filter)
            if mask.shape != grid.shape or np.iscomplexobj(mask) or not np.all(np.isfinite(mask)):
                raise ValueError("reciprocal_filter must be a finite real array on the transverse grid")
            if np.any(mask < 0) or np.any(mask > 1):
                raise ValueError("reciprocal_filter values must lie in [0,1]")
            self.reciprocal_filter = mask.astype(np.float64, copy=True)

    def _filter(self, wave: np.ndarray) -> np.ndarray:
        if self.reciprocal_filter is None:
            return wave
        return np.fft.ifft2(
            np.fft.fft2(wave, axes=(-2, -1)) * self.reciprocal_filter, axes=(-2, -1)
        )

    def transmission(self, projected_potential_v_m: np.ndarray) -> np.ndarray:
        """Compute and optionally bandlimit t; potential is already integrated."""
        potential = np.asarray(projected_potential_v_m)
        if potential.shape[-2:] != self.grid.shape or potential.ndim < 2:
            raise ValueError("potential must have the grid's trailing (ny,nx) shape")
        if np.iscomplexobj(potential) or not np.all(np.isfinite(potential)):
            raise ValueError("elastic electrostatic potential must be finite and real")
        return self._filter(np.exp(1j * self.sigma * potential.astype(np.float64, copy=False)))

    def propagate(
        self,
        wave: np.ndarray,
        projected_potentials_v_m: np.ndarray,
        slice_thicknesses_m: np.ndarray,
        repetitions: int = 1,
    ) -> ElasticResult:
        """Propagate through a slice stack, optionally repeated exactly along z.

        Repetition is a lossless representation of a static periodic potential,
        not a thermal ensemble. Thicknesses are used only for free propagation.
        With a filter, both transmission functions and propagated waves are
        bandlimited, matching conventional abTEM. With no filter the real-
        potential step is unitary up to floating-point roundoff.
        """
        wave = np.asarray(wave, dtype=np.complex128)
        if wave.ndim < 2 or wave.shape[-2:] != self.grid.shape or not np.all(np.isfinite(wave)):
            raise ValueError("wave must be finite and have the grid's trailing (ny,nx) shape")
        potential = np.asarray(projected_potentials_v_m)
        thicknesses = np.asarray(slice_thicknesses_m, dtype=np.float64)
        if potential.ndim != 3 or potential.shape[0] < 1:
            raise ValueError("projected_potentials_v_m must have shape (nslices,ny,nx)")
        if thicknesses.ndim != 1 or thicknesses.size != potential.shape[0]:
            raise ValueError("one slice thickness is required for each projected potential")
        if not np.all(np.isfinite(thicknesses)) or np.any(thicknesses <= 0):
            raise ValueError("slice thicknesses must be positive and finite")
        if isinstance(repetitions, bool) or not isinstance(repetitions, Integral) or repetitions < 1:
            raise ValueError("repetitions must be a positive integer")
        transmissions = self.transmission(potential)
        current = wave.copy()
        history = [grid_norm_squared(current, self.grid)]
        for _ in range(repetitions):
            for transmission, thickness in zip(transmissions, thicknesses):
                current = self.free.propagate(current * transmission, float(thickness))
                current = self._filter(current)
                history.append(grid_norm_squared(current, self.grid))
        return ElasticResult(current, np.asarray(history, dtype=np.float64))
