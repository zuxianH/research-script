"""Benchmark 1: run from the project root with python3 -m benchmarks.free_electron."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shlex
import sys

import numpy as np
import scipy
from scipy.constants import c, elementary_charge, hbar, m_e

from yoshioka import FreePropagator, TransverseGrid, electron_wavenumber, grid_norm_squared
from .reference_solutions import (
    constant_wave,
    gaussian_resolution_estimates,
    gaussian_wave,
    periodic_mode,
    radial_second_moment,
    relative_wave_error,
)


TOLERANCES = {
    "relative_norm_drift": 1e-10,
    "constant_relative_wave_error": 1e-10,
    "fourier_relative_wave_error": 1e-10,
    "composition_relative_wave_error": 1e-10,
    "round_trip_relative_wave_error": 1e-10,
    "gaussian_relative_wave_error": 1e-8,
    "gaussian_relative_moment_error": 1e-8,
    "empty_channel_population": 0.0,
}


@dataclass(frozen=True)
class BenchmarkConfig:
    energy_kev: float = 60.0
    lx_nm: float = 10.0
    ly_nm: float = 10.0
    nx: int = 256
    ny: int = 256
    width_nm: float = 0.2
    distances_in_zd: tuple[float, ...] = (0.0, 0.25, 1.0, 2.0)
    mode_x: int = 3
    mode_y: int = -5
    composition_steps: int = 100
    refinement_grids: tuple[int, ...] = (64, 128, 256)

    def __post_init__(self) -> None:
        for name in ("energy_kev", "width_nm"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        grid = self.grid
        for mode, size in ((self.mode_x, grid.nx), (self.mode_y, grid.ny)):
            if not isinstance(mode, int) or not -(size // 2) <= mode <= (size - 1) // 2:
                raise ValueError("mode indices must be integers within the represented FFT frequencies")
        if not self.distances_in_zd or not all(np.isfinite(self.distances_in_zd)):
            raise ValueError("distances_in_zd must contain finite distances")
        if not isinstance(self.composition_steps, int) or self.composition_steps < 1:
            raise ValueError("composition_steps must be a positive integer")
        if not self.refinement_grids:
            raise ValueError("refinement_grids must not be empty")
        for n in self.refinement_grids:
            TransverseGrid(grid.lx_m, grid.ly_m, n, n)

    @property
    def grid(self) -> TransverseGrid:
        # User-facing keV and nm are converted once at the benchmark boundary.
        return TransverseGrid(self.lx_nm * 1e-9, self.ly_nm * 1e-9, self.nx, self.ny)

    @property
    def energy_j(self) -> float:
        return self.energy_kev * 1e3 * elementary_charge

    @property
    def width_m(self) -> float:
        return self.width_nm * 1e-9


def _norm(wave: np.ndarray, grid: TransverseGrid) -> float:
    return float(np.sum(grid_norm_squared(wave, grid)))


def _drift(wave: np.ndarray, initial: np.ndarray, grid: TransverseGrid) -> float:
    initial_norm = _norm(initial, grid)
    return abs(_norm(wave, grid) - initial_norm) / initial_norm


def _gaussian_case(grid: TransverseGrid, width_m: float, k: float, s: float) -> dict:
    z_m = s * k * width_m**2
    initial = gaussian_wave(grid, width_m, k)
    numerical = FreePropagator(grid, k).propagate(initial, z_m)
    exact = gaussian_wave(grid, width_m, k, z_m)
    exact_moment = width_m**2 * (1 + s**2)
    measured_moment = radial_second_moment(numerical, grid)
    return {
        "s": s,
        "z_m": z_m,
        "relative_wave_error": relative_wave_error(numerical, exact, grid),
        "relative_norm_drift": _drift(numerical, initial, grid),
        "initial_grid_norm": _norm(initial, grid),
        "numerical_grid_norm": _norm(numerical, grid),
        "analytical_grid_norm": _norm(exact, grid),
        "numerical_radial_second_moment_m2": measured_moment,
        "analytical_radial_second_moment_m2": exact_moment,
        "relative_moment_error": abs(measured_moment - exact_moment) / exact_moment,
        "resolution_estimates": gaussian_resolution_estimates(grid, width_m, s),
    }


def run_benchmark(config: BenchmarkConfig) -> dict:
    """Compute all tests and return JSON-compatible data; do not write files."""
    grid, width_m = config.grid, config.width_m
    k = electron_wavenumber(config.energy_j)
    zd_m = k * width_m**2
    propagator = FreePropagator(grid, k)
    initial_constant = constant_wave(grid)
    initial_mode = periodic_mode(grid, config.mode_x, config.mode_y, k)
    initial_gaussian = gaussian_wave(grid, width_m, k)
    # Minimal channel representation: (channel, y, x), no coupling machinery.
    initial_channels = np.zeros((3, *grid.shape), dtype=np.complex128)
    initial_channels[0] = initial_gaussian
    cases = {name: [] for name in ("constant", "fourier_mode", "gaussian", "composition", "round_trip", "channels")}
    all_norm_drifts = []
    for s in config.distances_in_zd:
        z_m = s * zd_m
        for label, initial, exact in (
            ("constant", initial_constant, initial_constant),
            ("fourier_mode", initial_mode, periodic_mode(grid, config.mode_x, config.mode_y, k, z_m)),
        ):
            wave = propagator.propagate(initial, z_m)
            drift = _drift(wave, initial, grid)
            cases[label].append({
                "s": s, "z_m": z_m,
                "relative_wave_error": relative_wave_error(wave, exact, grid),
                "initial_grid_norm": _norm(initial, grid),
                "numerical_grid_norm": _norm(wave, grid),
                "relative_norm_drift": drift,
            })
            all_norm_drifts.append(drift)
        gaussian_case = _gaussian_case(grid, width_m, k, s)
        cases["gaussian"].append(gaussian_case)
        all_norm_drifts.append(gaussian_case["relative_norm_drift"])
        single_step = propagator.propagate(initial_gaussian, z_m)
        multiple_steps = initial_gaussian.copy()
        for _ in range(config.composition_steps):
            multiple_steps = propagator.propagate(multiple_steps, z_m / config.composition_steps)
        composition_drift = _drift(multiple_steps, initial_gaussian, grid)
        cases["composition"].append({
            "s": s, "z_m": z_m, "steps": config.composition_steps,
            "relative_wave_error": relative_wave_error(multiple_steps, single_step, grid),
            "relative_norm_drift": composition_drift,
        })
        recovered = propagator.propagate(single_step, -z_m)
        round_trip_drift = _drift(recovered, initial_gaussian, grid)
        cases["round_trip"].append({
            "s": s, "z_m": z_m,
            "relative_wave_error": relative_wave_error(recovered, initial_gaussian, grid),
            "relative_norm_drift": round_trip_drift,
        })
        channels = propagator.propagate(initial_channels, z_m)
        populations = grid_norm_squared(channels, grid)
        channel_drift = _drift(channels, initial_channels, grid)
        cases["channels"].append({
            "s": s, "z_m": z_m,
            "channel_populations": populations.tolist(),
            "total_grid_norm": float(np.sum(populations)),
            "relative_norm_drift": channel_drift,
            "empty_channels_exactly_zero": bool(np.all(channels[1:] == 0)),
            "inelastic_population": float(np.sum(populations[1:])),
            "incident_relative_wave_error": relative_wave_error(channels[0], single_step, grid),
        })
        all_norm_drifts.extend((composition_drift, round_trip_drift, channel_drift))
    refinement = []
    for n in config.refinement_grids:
        refined_grid = TransverseGrid(grid.lx_m, grid.ly_m, n, n)
        refinement.append({
            "nx": n, "ny": n, "dx_m": refined_grid.dx_m, "dy_m": refined_grid.dy_m,
            "cases": [_gaussian_case(refined_grid, width_m, k, s) for s in config.distances_in_zd],
        })
    maximum = lambda name, field: max(case[field] for case in cases[name])
    summary = {
        "relative_norm_drift": max(all_norm_drifts),
        "constant_relative_wave_error": maximum("constant", "relative_wave_error"),
        "fourier_relative_wave_error": maximum("fourier_mode", "relative_wave_error"),
        "composition_relative_wave_error": maximum("composition", "relative_wave_error"),
        "round_trip_relative_wave_error": maximum("round_trip", "relative_wave_error"),
        "gaussian_relative_wave_error": maximum("gaussian", "relative_wave_error"),
        "gaussian_relative_moment_error": maximum("gaussian", "relative_moment_error"),
        "empty_channel_population": maximum("channels", "inelastic_population"),
    }
    # Positive tolerances are strict upper bounds; the zero-population test is exact.
    checks = {key: (value < TOLERANCES[key] if TOLERANCES[key] > 0 else value == 0) for key, value in summary.items()}
    checks["empty_channels_exactly_zero"] = all(case["empty_channels_exactly_zero"] for case in cases["channels"])
    checks["constant_initial_unit_norm"] = abs(_norm(initial_constant, grid) - 1) < TOLERANCES["relative_norm_drift"]
    checks["fourier_initial_unit_norm"] = abs(_norm(initial_mode, grid) - 1) < TOLERANCES["relative_norm_drift"]
    checks["incident_channel_matches_single_wave"] = maximum("channels", "incident_relative_wave_error") < TOLERANCES["gaussian_relative_wave_error"]
    diagnostics = {
        "max_outside_box_l2_fraction": max(case["resolution_estimates"]["outside_box_l2_fraction"] for case in cases["gaussian"]),
        "initial_outside_nyquist_l2_fraction": cases["gaussian"][0]["resolution_estimates"]["outside_nyquist_l2_fraction"],
    }
    checks["gaussian_boundary_tail_negligible"] = diagnostics["max_outside_box_l2_fraction"] < TOLERANCES["gaussian_relative_wave_error"]
    checks["gaussian_spectral_tail_negligible"] = diagnostics["initial_outside_nyquist_l2_fraction"] < TOLERANCES["gaussian_relative_wave_error"]
    return {
        "schema_version": 1,
        "benchmark": "Benchmark 1: free-electron paraxial propagation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "parameters": {
            "input": asdict(config),
            "si": {"kinetic_energy_j": config.energy_j, "lx_m": grid.lx_m, "ly_m": grid.ly_m,
                   "dx_m": grid.dx_m, "dy_m": grid.dy_m, "width_m": width_m,
                   "wavenumber_m_inv": k, "wavelength_m": 2 * np.pi / k,
                   "zd_m": zd_m, "evaluation_distances_m": [s * zd_m for s in config.distances_in_zd]},
            "channel_count": 3,
        },
        "conventions": {
            "wave_dtype": "complex128", "real_dtype": "float64", "array_axes": "(..., y, x)",
            "fourier_forward_sign": -1, "fourier_inverse_normalization": "1/(nx*ny)",
            "angular_frequencies": "2*pi*fftfreq(N, spacing_m)",
            "propagation_multiplier": "exp[-i*z*(kx^2+ky^2)/(2*k)]",
            "norm": "dx*dy*sum(abs(phi)^2)",
            "relative_wave_error": "sqrt(grid_norm_squared(numerical-exact)/grid_norm_squared(exact))",
            "global_phase_alignment": False, "longitudinal_carrier_included": False,
            "interactions_and_channel_couplings": "zero",
        },
        "tolerances": dict(TOLERANCES), "summary": summary, "checks": checks,
        "resolution_diagnostics": diagnostics, "cases": cases, "grid_refinement": refinement,
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "constants_source": "scipy.constants (installed SciPy CODATA values)",
            "constants_si": {"electron_mass_kg": m_e, "c_m_per_s": c, "hbar_j_s": hbar, "elementary_charge_c": elementary_charge},
        },
    }


def write_figures(results: dict, config: BenchmarkConfig, output: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid, width_m = config.grid, config.width_m
    k = results["parameters"]["si"]["wavenumber_m_inv"]
    propagator = FreePropagator(grid, k)
    initial = gaussian_wave(grid, width_m, k)
    distances = config.distances_in_zd
    cols = min(2, len(distances))
    rows = (len(distances) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.3 * rows), squeeze=False, constrained_layout=True)
    extent = [(grid.x_m[0] - grid.dx_m / 2) * 1e9, (grid.x_m[-1] + grid.dx_m / 2) * 1e9,
              (grid.y_m[0] - grid.dy_m / 2) * 1e9, (grid.y_m[-1] + grid.dy_m / 2) * 1e9]
    vmax = float(np.max(np.abs(initial) ** 2) * 1e-18)
    view_nm = min(0.5 * min(config.lx_nm, config.ly_nm), 3 * config.width_nm * np.sqrt(1 + max(abs(s) for s in distances)**2))
    for ax, s in zip(axes.flat, distances):
        wave = propagator.propagate(initial, s * k * width_m**2)
        im = ax.imshow(np.abs(wave) ** 2 * 1e-18, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax)
        ax.set(title=f"z/z_d = {s:g}", xlabel="x (nm)", ylabel="y (nm)", xlim=(-view_nm, view_nm), ylim=(-view_nm, view_nm))
    for ax in list(axes.flat)[len(distances):]:
        ax.set_visible(False)
    fig.colorbar(im, ax=list(axes.flat)[:len(distances)], label="Intensity (nm$^{-2}$)")
    fig.suptitle("Free Gaussian diffraction — numerical intensity")
    fig.savefig(output / "gaussian_intensities.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for index, s in enumerate(distances):
        z_m = s * k * width_m**2
        numerical = propagator.propagate(initial, z_m)
        exact = gaussian_wave(grid, width_m, k, z_m)
        color = f"C{index % 10}"
        ax.plot(grid.x_m * 1e9, np.abs(numerical[grid.ny // 2]) ** 2 * 1e-18, color=color, label=f"Numerical, z/z_d={s:g}")
        ax.plot(grid.x_m * 1e9, np.abs(exact[grid.ny // 2]) ** 2 * 1e-18, linestyle="none", marker="o", markersize=3, markevery=max(1, grid.nx // 40), color=color, label=f"Analytical, z/z_d={s:g}")
    ax.set(xlabel="x (nm), y = 0", ylabel="Intensity (nm$^{-2}$)", xlim=(-view_nm, view_nm), title="Gaussian profiles: numerical and infinite-domain analytical")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    fig.savefig(output / "gaussian_profiles.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for index, s in enumerate(distances):
        if s == 0:
            continue
        errors = [entry["cases"][index]["relative_wave_error"] for entry in results["grid_refinement"]]
        # Only positive measured errors can be shown on a logarithmic axis.
        points = [(entry["nx"], err) for entry, err in zip(results["grid_refinement"], errors) if err > 0]
        if points:
            sizes, errors = zip(*points)
            ax.semilogy(sizes, errors, "o-", label=f"z/z_d={s:g}")
    ax.axhline(TOLERANCES["gaussian_relative_wave_error"], color="black", linestyle="--", label="Default-grid acceptance limit")
    ax.set(xticks=list(config.refinement_grids), xlabel="N (fixed transverse box, N × N)", ylabel="Relative complex-wave L2 error", title="Spatial sampling reaches the floating-point floor")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, which="both")
    fig.savefig(output / "gaussian_error_vs_grid.png", dpi=180)
    plt.close(fig)
    return ["gaussian_intensities.png", "gaussian_profiles.png", "gaussian_error_vs_grid.png"]


def reproduction_command(config: BenchmarkConfig, output: Path) -> str:
    args = ["python3", "-m", "benchmarks.free_electron", "--energy-kev", str(config.energy_kev),
            "--lx-nm", str(config.lx_nm), "--ly-nm", str(config.ly_nm),
            "--nx", str(config.nx), "--ny", str(config.ny), "--width-nm", str(config.width_nm),
            "--distances-in-zd", *map(str, config.distances_in_zd),
            "--mode-x", str(config.mode_x), "--mode-y", str(config.mode_y),
            "--composition-steps", str(config.composition_steps),
            "--refinement-grids", *map(str, config.refinement_grids), "--output", str(output)]
    return shlex.join(args)


def write_report(results: dict, config: BenchmarkConfig, output: Path) -> None:
    si = results["parameters"]["si"]
    summary = results["summary"]
    lines = [
        "# Benchmark 1: free-electron paraxial propagation", "",
        f"Overall result: **{results['status']}**. All errors below are measured, without global-phase alignment.", "",
        "## Equation, units, and Fourier convention", "",
        r"The envelope obeys $\partial_z\phi=i\nabla_\perp^2\phi/(2k)$, with every potential and channel coupling set to zero.",
        r"The relativistic angular wavenumber is $k=\sqrt{T(T+2m_ec^2)}/(\hbar c)$. There is no longitudinal $e^{ikz}$ carrier.",
        "User inputs convert as 1 nm = 1e-9 m and 1 keV = 1e3 × elementary_charge joules. All propagation and reference calculations use SI units, float64 coordinates, and complex128 waves. Plots convert m to nm and m^-2 to nm^-2 by multiplying intensity by 1e-18.", "",
        r"Arrays have axes (..., y, x). Coordinates are $(j-\lfloor N/2\rfloor)\Delta x$. NumPy's forward DFT has a negative exponential; the inverse has a positive exponential and factor $1/(N_xN_y)$. Frequencies are unshifted $k_x=2\pi\,\mathrm{fftfreq}(N_x,\Delta x)$, similarly for y.",
        r"Each represented mode evolves by $P=\exp[-iz(k_x^2+k_y^2)/(2k)]$. The sampled norm is $\Delta x\Delta y\sum|\phi|^2$, and relative wave error is $\sqrt{\|\phi_{num}-\phi_{exact}\|^2_{grid}/\|\phi_{exact}\|^2_{grid}}$.", "",
        "## Parameters and measured acceptance checks", "",
        f"Energy {config.energy_kev:g} keV; box {config.lx_nm:g} × {config.ly_nm:g} nm; grid {config.nx} × {config.ny}; width {config.width_nm:g} nm; mode ({config.mode_x}, {config.mode_y}).",
        f"k = {si['wavenumber_m_inv']:.12e} rad/m; wavelength = {si['wavelength_m'] * 1e12:.9f} pm; z_d = k w_0² = {si['zd_m'] * 1e9:.9f} nm; dx = {si['dx_m'] * 1e9:.9f} nm; dy = {si['dy_m'] * 1e9:.9f} nm.",
        f"Evaluation distances z/z_d: {', '.join(f'{s:g}' for s in config.distances_in_zd)}. Values in the table are maxima over these distances; norm drift includes all cases, including the {config.composition_steps}-step composition runs.", "",
        "| Quantity | Measured maximum | Required bound | Result |", "|---|---:|---:|---|",
    ]
    for key, value in summary.items():
        limit = TOLERANCES[key]
        bound = f"< {limit:.1e}" if limit else "exactly zero"
        lines.append(f"| {key} | {value:.6e} | {bound} | {'PASS' if results['checks'][key] else 'FAIL'} |")
    lines.extend(["", f"Additional checks: {json.dumps({key: value for key, value in results['checks'].items() if key not in summary}, sort_keys=True)}.", "",
        "The constant reference is 1/sqrt(Lx Ly). The Fourier-mode reference evaluates its phase directly from the integer indices and physical box lengths, independently of fftfreq and the production multiplier. Both initial norms are checked against one.", "",
        r"The independent infinite-domain Gaussian is $\phi=(\sqrt{\pi}w_0)^{-1}(1+is)^{-1}\exp[-r^2/(2w_0^2(1+is))]$, with $s=z/(kw_0^2)$. Its normalized radial second moment is $w_0^2(1+s^2)$. Initial Gaussian samples are not renormalized.", "",
        f"Composition compares one step over z with {config.composition_steps} steps over z/{config.composition_steps}; reversibility applies z then -z. These are algebraic and floating-point consistency checks. The spectral free operator is exact for represented modes, so these are not slice-discretization convergence tests.", "",
        "Three uncoupled channels are stored as a minimal (channel, y, x) array. Only channel zero starts populated. Each channel is transformed on the last two axes; both empty waves and their integrated populations are checked to remain exactly zero, and total norm and the incident channel are checked independently.", "",
        "## Periodic boundaries and spatial sampling", "",
        "The FFT solves a periodic problem; the Gaussian reference solves an infinite-domain problem. No phase adjustment or periodic-image correction is applied to the reference. For intensity proportional to exp(-r²/w(z)²), the outside-box probability is ex + ey - ex ey, where ex = erfc(Lx/[2 w(z)]) and similarly for y. The initial spectral power outside the Nyquist rectangle uses ex = erfc(w0 π/dx), ey = erfc(w0 π/dy). Their square roots give continuum L2 tail fractions; these are resolution diagnostics, not exact bounds on the discrete aliasing error. Spectral power is unchanged by free evolution.", "",
        f"At the configured grid, the maximum outside-box L2 tail fraction is {results['resolution_diagnostics']['max_outside_box_l2_fraction']:.6e}; the initial outside-Nyquist L2 fraction is {results['resolution_diagnostics']['initial_outside_nyquist_l2_fraction']:.6e}.",
        f"Maximum boundary intensity relative to the on-axis peak is {max(case['resolution_estimates']['boundary_intensity_relative_to_peak'] for case in results['cases']['gaussian']):.6e}.", "",
        "| Grid | z/z_d | Relative complex-wave error | Relative moment error | Relative norm drift |", "|---|---:|---:|---:|---:|",
    ])
    for entry in results["grid_refinement"]:
        for case in entry["cases"]:
            lines.append(f"| {entry['nx']} × {entry['ny']} | {case['s']:g} | {case['relative_wave_error']:.6e} | {case['relative_moment_error']:.6e} | {case['relative_norm_drift']:.6e} |")
    lines.extend(["",
        "For the default parameters, the 64² grid has appreciable unresolved Gaussian spectral tails; at 128² the tails approach double-precision roundoff, and 256² is fully resolved. The measured table distinguishes this spatial-sampling transition from propagation-step accuracy. Monotonic improvement is not required once roundoff dominates. A narrower initial Gaussian or a smaller box can fail without any defect in the FFT multiplier; configured runs keep the same acceptance tolerances and report failures explicitly. The refinement table is diagnostic and its coarse grids are not required to pass the default-grid tolerances.", "",
        "## Figures", "",
        "![Initial and propagated intensities](gaussian_intensities.png)", "",
        "![Numerical and analytical profiles](gaussian_profiles.png)", "",
        "![Complex-wave error versus grid size](gaussian_error_vs_grid.png)", "",
        "## Reproduction", "", "Run from the project root:", "", "```bash",
        f"cd {Path(__file__).resolve().parents[1]}", "python3 -m unittest discover -s tests -v",
        reproduction_command(config, output), "```", "",
        "Dependencies: NumPy, SciPy, Matplotlib; automated tests use the standard-library unittest runner. The JSON records input parameters, SI conversions, constants, package versions, individual cases, fixed tolerances, checks, and overall status. The benchmark exits with status 1 when any acceptance check fails.", "",
        "## Scope", "",
        "Passing validates free propagation and the absence of unintended channel mixing. It does not validate electron–phonon scattering, atomic potentials, phonons, inelastic couplings, or EELS normalization. Every channel here shares the incident wavenumber; no general coupled-channel solver is introduced.", "",
    ])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-kev", type=float, default=60)
    parser.add_argument("--lx-nm", type=float, default=10)
    parser.add_argument("--ly-nm", type=float, default=10)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=256)
    parser.add_argument("--width-nm", type=float, default=0.2)
    parser.add_argument("--distances-in-zd", type=float, nargs="+", default=[0, 0.25, 1, 2])
    parser.add_argument("--mode-x", type=int, default=3)
    parser.add_argument("--mode-y", type=int, default=-5)
    parser.add_argument("--composition-steps", type=int, default=100)
    parser.add_argument("--refinement-grids", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/free_electron"))
    args = parser.parse_args(argv)
    try:
        config = BenchmarkConfig(**{key: (tuple(value) if isinstance(value, list) else value) for key, value in vars(args).items() if key != "output"})
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = run_benchmark(config)
    results["artifacts"] = write_figures(results, config, output) + ["results.json", "report.md"]
    results["reproduction_command"] = reproduction_command(config, output)
    import matplotlib
    results["environment"]["matplotlib"] = matplotlib.__version__
    (output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(results, config, output)
    print(f"{results['benchmark']}: {results['status']}")
    for key, value in results["summary"].items():
        print(f"  {key}: {value:.6e}")
    print(f"Results, report, and figures: {output}")
    return 0 if results["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
