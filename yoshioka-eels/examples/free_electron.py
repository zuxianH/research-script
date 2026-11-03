"""Bounded 60 keV free-propagation example; all output paths are explicit."""

import argparse
from pathlib import Path

import numpy as np
from scipy.constants import elementary_charge

from yoshioka import FreePropagator, TransverseGrid, electron_wavenumber, grid_norm_squared
from yoshioka.provenance import json_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    grid = TransverseGrid(8e-9, 8e-9, 64, 64)
    x, y = grid.xy_m
    initial = np.exp(-(x**2 + y**2) / (2 * (0.4e-9)**2)).astype(complex)
    initial /= np.sqrt(grid_norm_squared(initial, grid))
    k0 = electron_wavenumber(60e3 * elementary_charge)
    propagator = FreePropagator(grid, k0)
    distance = 10e-9
    final = propagator.propagate(initial, distance)
    recovered = propagator.propagate(final, -distance)
    norm_error = abs(float(grid_norm_squared(final, grid)) - 1)
    roundtrip_error = float(np.sqrt(grid_norm_squared(recovered - initial, grid)))
    np.savez(args.output / "waves.npz", initial=initial, final=final, x_m=x, y_m=y)
    json_write(args.output / "checks.json", {
        "model": "free_paraxial_electron", "energy_eV": 60000,
        "distance_m": distance, "grid": [64, 64], "k0_rad_per_m": k0,
        "wavelength_m": 2 * np.pi / k0, "norm_absolute_error": norm_error,
        "roundtrip_relative_wave_error": roundtrip_error,
    })
    if max(norm_error, roundtrip_error) > 1e-12:
        raise RuntimeError("Free propagation check exceeded 1e-12")
    print(f"Saved {args.output}; norm error {norm_error:.3g}, roundtrip error {roundtrip_error:.3g}")


if __name__ == "__main__":
    main()
