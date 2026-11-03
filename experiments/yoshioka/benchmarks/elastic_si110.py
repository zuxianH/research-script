"""Benchmark 2: shared Lobato slices, SI elastic propagation versus abTEM.

Run .venv/bin/python -m benchmarks.elastic_si110 from the project root.
The exit code is nonzero if agreement OR the requested convergence tests fail.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.constants import c, elementary_charge, hbar, m_e

from yoshioka import ElasticPropagator, FreePropagator, TransverseGrid, electron_wavenumber
from yoshioka.elastic_propagation import antialias_aperture
from .abtem_reference import (
    build_shared_slices, export_shared_inputs, load_structure, propagate_reference,
    reference_configuration, source_audit, verify_direct_repeated_build,
)
from .elastic_observables import (
    angular_masks, compare_observables, diffraction_probabilities,
    intensity_observables, select_reference_beams,
)
from .reference_solutions import constant_wave, periodic_mode, relative_wave_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOLERANCES = {
    "relative_exit_wave_error": 1e-6,
    "diffraction_intensity_error_up_to_50_mrad": 1e-5,
    "significant_intensity_relative_error": 0.01,
    "physical_intensity_relative_change": 0.01,
    "weak_reference_probability_threshold": 1e-6,
    "weak_intensity_absolute_error": 1e-8,
    "analytical_wave_error": 1e-10,
    "unfiltered_norm_drift": 1e-10,
    "empty_channel_population": 0.0,
}


@dataclass(frozen=True)
class ElasticConfig:
    structure: str = str(PROJECT_ROOT / "Si_110_temp300.lmp")
    energy_kev: float = 60.0
    repeats: tuple[int, ...] = (13, 26, 52)
    grids: tuple[int, ...] = (64, 128, 256)
    slices_per_repeat: tuple[int, ...] = (8, 16, 32)
    initial_grid: int = 128
    initial_slices: int = 8
    beam_count: int = 6
    extra_refinement: bool = True

    def __post_init__(self):
        if not np.isfinite(self.energy_kev) or self.energy_kev <= 0:
            raise ValueError("energy_kev must be positive and finite")
        for name in ("repeats", "grids", "slices_per_repeat"):
            values = getattr(self, name)
            if not values or any(isinstance(n, bool) or not isinstance(n, int) or n < (8 if name == "grids" else 1) for n in values):
                raise ValueError(f"{name} must contain positive integers (grids >= 8)")
            if tuple(sorted(set(values))) != tuple(values):
                raise ValueError(f"{name} must be unique and increasing")
        if self.initial_grid not in self.grids or self.initial_slices not in self.slices_per_repeat:
            raise ValueError("initial grid and slices must be included in the comparison sets")
        if not isinstance(self.beam_count, int) or self.beam_count < 1:
            raise ValueError("beam_count must be positive")


def _compare(a: dict, b: dict, convergence: bool = False) -> dict:
    return compare_observables(
        a, b,
        relative_tolerance=TOLERANCES["physical_intensity_relative_change" if convergence else "significant_intensity_relative_error"],
        weak_threshold=TOLERANCES["weak_reference_probability_threshold"],
        absolute_tolerance=TOLERANCES["weak_intensity_absolute_error"],
    )


def analytical_regressions(energy_j: float) -> dict:
    grid = TransverseGrid(5.43053e-10, 3.83996e-10, 32, 24)
    thicknesses = np.array([0.31, 0.73, 1.17]) * 1e-10
    initial = constant_wave(grid)
    zero = np.zeros((3, *grid.shape))
    solver = ElasticPropagator(grid, energy_j)
    zero_result = solver.propagate(initial, zero, thicknesses)
    mode = periodic_mode(grid, 2, -3, electron_wavenumber(energy_j))
    zero_mode = solver.propagate(mode, zero, thicknesses)
    free_mode = FreePropagator(grid, electron_wavenumber(energy_j)).propagate(mode, float(thicknesses.sum()))
    voltage, repetitions = 13.7, 7
    potentials = voltage * thicknesses[:, None, None] * np.ones((3, *grid.shape))
    uniform = solver.propagate(initial, potentials, thicknesses, repetitions)
    gamma = 1 + energy_j / (m_e * c**2)
    v_independent = np.sqrt((energy_j / c)**2 + 2 * m_e * energy_j) / (gamma * m_e)
    phase = elementary_charge * voltage * thicknesses.sum() * repetitions / (hbar * v_independent)
    expected = initial * np.exp(1j * phase)
    errors = {
        "zero_potential_plane_wave_error": relative_wave_error(zero_result.wave, initial, grid),
        "zero_potential_benchmark1_mode_error": relative_wave_error(zero_mode.wave, free_mode, grid),
        "uniform_potential_phase_error": relative_wave_error(uniform.wave, expected, grid),
        "uniform_unfiltered_max_norm_drift": float(np.max(np.abs(uniform.norm_history - uniform.norm_history[0]))),
        "uniform_analytical_phase_rad": float(phase),
    }
    errors["passed"] = all(value < TOLERANCES["analytical_wave_error"] for key, value in errors.items() if key.endswith("error") or key.endswith("drift"))
    return errors


def _case_key(n: int, slices: int, repeat: int) -> str:
    return f"n{n}_s{slices}_r{repeat}"


def run_case(shared, repeat: int, n: int, slices: int, energy_j: float, wavelength_m: float, beams, output: Path, primary: bool):
    reference = propagate_reference(shared, repeat, check_driver=primary)
    if beams is None:
        raise ValueError("select beams from the initial reference before running comparison cases")
    grid = shared.grid
    solver = ElasticPropagator(grid, energy_j, antialias_aperture(grid))
    initial = shared.incident_si
    if primary:
        initial = np.zeros((3, *grid.shape), dtype=np.complex128)
        initial[0] = shared.incident_si
    result = solver.propagate(initial, shared.projected_potential_v_m, shared.thicknesses_m, repetitions=repeat)
    wave = result.wave[0] if primary else result.wave
    norm_history = result.norm_history.sum(axis=1) if primary else result.norm_history
    diffraction = diffraction_probabilities(wave, grid)
    region = angular_masks(grid, wavelength_m)["region_up_to_50_mrad"]
    values = intensity_observables(diffraction, grid, wavelength_m, beams)
    ref_values = intensity_observables(reference.diffraction, grid, wavelength_m, beams)
    observables = _compare(values, ref_values)
    wave_error = relative_wave_error(wave, reference.wave_si, grid)
    intensity_error = float(np.sum(np.abs(diffraction - reference.diffraction)[region]) / np.sum(reference.diffraction[region]))
    phase = float(np.angle(np.vdot(reference.wave_si, wave)))
    checks = {
        "exit_wave": wave_error < TOLERANCES["relative_exit_wave_error"],
        "diffraction": intensity_error < TOLERANCES["diffraction_intensity_error_up_to_50_mrad"],
        "intensity_observables": observables["passed"],
    }
    if primary:
        checks["empty_channels_exactly_zero"] = bool(np.all(result.wave[1:] == 0) and np.all(result.norm_history[:, 1:] == 0))
    arrays = {
        "exit_wave_ours_si_yx": wave, "exit_wave_abtem_si_yx": reference.wave_si,
        "exit_intensity_ours_m_inv2": np.abs(wave)**2,
        "exit_intensity_abtem_m_inv2": np.abs(reference.wave_si)**2,
        "diffraction_ours_probability_yx": diffraction,
        "diffraction_abtem_probability_yx": reference.diffraction,
        "norm_history_ours": norm_history, "norm_history_abtem": reference.norm_history,
        "z_edges_m": np.r_[0.0, np.cumsum(np.tile(shared.thicknesses_m, repeat))],
        "comparison_region_up_to_50_mrad": region,
        "extent_m": np.array([grid.lx_m, grid.ly_m]),
    }
    unfiltered_drift = None
    if primary:
        unfiltered = ElasticPropagator(grid, energy_j).propagate(
            shared.incident_si, shared.projected_potential_v_m, shared.thicknesses_m, repetitions=repeat,
        )
        arrays["norm_history_unfiltered"] = unfiltered.norm_history
        arrays["channel_populations_after_exit"] = result.norm_history[-1]
        unfiltered_drift = float(np.max(np.abs(unfiltered.norm_history / unfiltered.norm_history[0] - 1)))
        checks["unfiltered_norm_conserved"] = unfiltered_drift < TOLERANCES["unfiltered_norm_drift"]
    key = _case_key(n, slices, repeat)
    path = output / "waves" / f"{key}.npz"
    np.savez_compressed(path, **arrays)
    record = {
        "key": key, "grid": [n, n], "slices_per_repeat": slices, "z_repeats": repeat,
        "atom_count": 4 * repeat, "num_slices": len(shared.thicknesses_m) * repeat,
        "thickness_m": float(np.sum(shared.thicknesses_m) * repeat),
        "thickness_nm": float(np.sum(shared.thicknesses_m) * repeat * 1e9),
        "primary_comparison": primary,
        "relative_exit_wave_error": wave_error,
        "diffraction_intensity_error_up_to_50_mrad": intensity_error,
        "real_space_intensity_relative_l1_error": float(np.sum(np.abs(np.abs(wave)**2 - np.abs(reference.wave_si)**2)) / np.sum(np.abs(reference.wave_si)**2)),
        "phase_aligned_wave_error_secondary": relative_wave_error(wave * np.exp(-1j * phase), reference.wave_si, grid),
        "relative_phase_rad_secondary": phase,
        "norm_initial": float(norm_history[0]), "norm_exit_ours": float(norm_history[-1]),
        "norm_exit_abtem": float(reference.norm_history[-1]),
        "filter_norm_loss_ours": float(1 - norm_history[-1] / norm_history[0]),
        "filter_norm_loss_abtem": float(1 - reference.norm_history[-1] / reference.norm_history[0]),
        "max_norm_history_absolute_difference": float(np.max(np.abs(norm_history - reference.norm_history))),
        "max_unfiltered_relative_norm_drift": unfiltered_drift,
        "empty_channel_populations": [float(value) for value in result.norm_history[-1, 1:]] if primary else None,
        "observables_ours": values, "observables_abtem": ref_values,
        "observable_comparison": observables,
        "reference_metadata": reference.metadata,
        "diffraction_parseval_error_ours": float(abs(np.sum(diffraction) - norm_history[-1])),
        "diffraction_parseval_error_abtem": float(abs(np.sum(reference.diffraction) - reference.norm_history[-1])),
        "reference_native_diffraction_vs_si_fft_max_difference": float(np.max(np.abs(diffraction_probabilities(reference.wave_si, grid) - reference.diffraction))),
        "artifact": str(path.relative_to(output)),
        "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL",
    }
    return record


def calculate_convergence(cases: list[dict], config: ElasticConfig) -> dict:
    lookup = {entry["key"]: entry for entry in cases}
    convergence = {"grid": [], "slice": []}
    for repeat in config.repeats:
        for kind, sizes in (("grid", config.grids), ("slice", config.slices_per_repeat)):
            for coarse, fine in zip(sizes[:-1], sizes[1:]):
                if kind == "grid":
                    a = lookup[_case_key(coarse, max(config.slices_per_repeat), repeat)]
                    b = lookup[_case_key(fine, max(config.slices_per_repeat), repeat)]
                else:
                    a = lookup[_case_key(max(config.grids), coarse, repeat)]
                    b = lookup[_case_key(max(config.grids), fine, repeat)]
                comparisons = {name: _compare(a[f"observables_{name}"], b[f"observables_{name}"], convergence=True) for name in ("ours", "abtem")}
                convergence[kind].append({
                    "z_repeats": repeat, "thickness_nm": b["thickness_nm"],
                    "coarse": coarse, "fine": fine, "coarse_case": a["key"], "fine_case": b["key"],
                    "finest_requested_pair": fine == max(sizes), "comparisons": comparisons,
                    "max_significant_relative_change": max(value["max_significant_relative_error"] for value in comparisons.values()),
                    "passed": all(value["passed"] for value in comparisons.values()),
                })
    required = [entry for values in convergence.values() for entry in values if entry["finest_requested_pair"]]
    convergence["evaluated"] = len(config.grids) >= 2 and len(config.slices_per_repeat) >= 2
    convergence["passed"] = bool(convergence["evaluated"] and all(entry["passed"] for entry in required))
    return convergence


def _write_json(path: Path, results: dict):
    path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def native_constant_control(shared, repeat: int, energy_j: float, audit: dict, reference_path: Path) -> float:
    """Secondary, unfitted constant-table control; never change primary results."""
    solver = ElasticPropagator(shared.grid, energy_j, antialias_aperture(shared.grid))
    solver.sigma = audit["native_sigma_per_v_angstrom"] * 1e10
    solver.free.wavenumber_m_inv = 2 * np.pi / (audit["native_wavelength_angstrom"] * 1e-10)
    result = solver.propagate(shared.incident_si, shared.projected_potential_v_m, shared.thicknesses_m, repetitions=repeat)
    with np.load(reference_path) as saved:
        return relative_wave_error(result.wave, saved["exit_wave_abtem_si_yx"], shared.grid)


def run_benchmark(config: ElasticConfig, output: Path) -> dict:
    from abtem.antialias import antialias_aperture as native_aperture

    output = output.resolve()
    for subdir in (output, output / "shared_inputs", output / "waves"):
        subdir.mkdir(parents=True, exist_ok=True)
    atoms, verification = load_structure(Path(config.structure))
    energy_ev = config.energy_kev * 1e3
    energy_j = energy_ev * elementary_charge
    wavelength_m = 2 * np.pi / electron_wavenumber(energy_j)
    specimen_arrays = {}
    specimens = []
    for repeat in config.repeats:
        specimen = atoms.repeat((1, 1, repeat))
        specimen_arrays[f"positions_angstrom_r{repeat}"] = specimen.positions
        specimen_arrays[f"cell_angstrom_r{repeat}"] = np.asarray(specimen.cell)
        specimen_arrays[f"atomic_numbers_r{repeat}"] = specimen.numbers
        specimens.append({"repeat": [1, 1, repeat], "atoms": len(specimen), "thickness_nm": float(specimen.cell[2, 2] * 0.1)})
    np.savez_compressed(output / "specimens.npz", **specimen_arrays)
    results = {
        "schema_version": 1, "benchmark": "Benchmark 2: elastic plane-wave diffraction from Si [110]",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "RUNNING",
        "parameters": asdict(config), "structure": verification, "specimens": specimens,
        "tolerances": dict(TOLERANCES), "source_audit": source_audit(energy_ev),
        "analytical_regressions": analytical_regressions(energy_j),
        "shared_configurations": [], "cases": [], "additional_refinement_cases": [],
        "conventions": {
            "equation": "dphi/dz=i*laplacian(phi)/(2*k)-i*H_el*phi/(hbar*v); H_el=-e*V",
            "transmission": "exp(+i*e*V_projected/(hbar*v)); integrated V_projected in V*m",
            "unit_conversions": {"angstrom_to_m": 1e-10, "kev_to_j": "1000*elementary_charge", "V_angstrom_to_V_m": 1e-10},
            "fourier": "numpy forward negative exponential, inverse positive divided by Nx*Ny; k_perp=2*pi*fftfreq",
            "ordering": "bandlimited transmission, then full-slice free propagation and reciprocal-space aperture",
            "reference_planes": "z=0 entrance and exact repeated-cell exit; no half-slice shifts",
            "normalization": "incident probability one; outputs never renormalized; native amplitude 1/(Nx*Ny) converted to SI 1/sqrt(Lx*Ly)",
            "angular_coordinates": "theta=lambda*f=k_perp/k, shared unshifted grid, radians converted to mrad",
            "annuli_mrad": [[0, 10], [10, 30], [30, 50]], "comparison_region_max_mrad": 50,
            "diffraction_crop": "none: abTEM full, parity same, fftshift false, block_direct false, renormalize false",
            "primary_global_phase_alignment": False, "longitudinal_carrier": False,
            "potential_mean_removed": False, "thermal_displacements": False,
            "inelastic_couplings": "absent; three uncoupled channels checked in primary cases",
        },
    }
    order = [(config.initial_grid, config.initial_slices)] + [(n, s) for n in config.grids for s in config.slices_per_repeat if (n, s) != (config.initial_grid, config.initial_slices)]
    start = time.perf_counter()
    with reference_configuration():
        for n, slices in order:
            shared = build_shared_slices(atoms, n, slices, energy_ev)
            if (n, slices) == (max(config.grids), max(config.slices_per_repeat)):
                finest_shared = shared
            fingerprint = hashlib.sha256(shared.native.array.tobytes()).hexdigest()
            shared_file = output / "shared_inputs" / f"n{n}_s{slices}.npz"
            export_shared_inputs(shared, shared_file, config.repeats)
            aperture_error = float(np.max(np.abs(antialias_aperture(shared.grid) - native_aperture(shared.native.gpts, shared.native.sampling, xp=np).T)))
            primary = (n, slices) == (config.initial_grid, config.initial_slices)
            shared_metadata = {
                **shared.metadata, "grid": [n, n], "slices_per_repeat": slices,
                "artifact": str(shared_file.relative_to(output)),
                "native_potential_sha256": fingerprint, "independent_aperture_max_error": aperture_error,
            }
            results["shared_configurations"].append(shared_metadata)
            if primary:
                first = propagate_reference(shared, config.repeats[0])
                beams = select_reference_beams(first.diffraction, shared.grid, wavelength_m, count=config.beam_count)
                results["selected_beams"] = beams
                results["beam_selection"] = f"{config.beam_count} strongest nonzero beams within 50 mrad in the thinnest initial reference, one representative per Friedel pair; fixed thereafter"
                results["direct_repeated_potential_verification"] = verify_direct_repeated_build(shared, atoms, config.repeats[0])
            for repeat in config.repeats:
                record = run_case(shared, repeat, n, slices, energy_j, wavelength_m, results["selected_beams"], output, primary)
                results["cases"].append(record)
                print(f"{record['key']}: {record['status']}; wave={record['relative_exit_wave_error']:.3e}; diffraction={record['diffraction_intensity_error_up_to_50_mrad']:.3e}; norm={record['norm_exit_ours']:.9f}", flush=True)
            shared_metadata["native_potential_unchanged"] = hashlib.sha256(shared.native.array.tobytes()).hexdigest() == fingerprint
            _write_json(output / "results.json", results)
        results["convergence"] = calculate_convergence(results["cases"], config)
        # Independently derived older ASE constants are substituted ONLY in this
        # secondary diagnostic. Primary waves and acceptance numbers are untouched.
        baseline = next(case for case in results["cases"] if case["key"] == _case_key(max(config.grids), max(config.slices_per_repeat), max(config.repeats)))
        shared = finest_shared
        audit = results["source_audit"]
        controlled_error = native_constant_control(shared, max(config.repeats), energy_j, audit, output / baseline["artifact"])
        results["constant_diagnostic"] = {
            "case": baseline["key"], "primary_wave_error": baseline["relative_exit_wave_error"],
            "wave_error_using_independently_reported_native_constants_secondary": controlled_error,
            "description": "secondary control uses the exact abTEM/ASE sigma and wavelength, with no fitted scale or phase; primary SI results retain scipy.constants",
        }
        if config.extra_refinement and results["convergence"]["evaluated"] and not results["convergence"]["passed"]:
            for n, slices, kind in ((max(config.grids), 2 * max(config.slices_per_repeat), "slice"), (2 * max(config.grids), max(config.slices_per_repeat), "grid")):
                shared = build_shared_slices(atoms, n, slices, energy_ev)
                shared_file = output / "shared_inputs" / f"n{n}_s{slices}.npz"
                export_shared_inputs(shared, shared_file, (max(config.repeats),))
                results["shared_configurations"].append({**shared.metadata, "grid": [n, n], "slices_per_repeat": slices, "artifact": str(shared_file.relative_to(output)), "additional_diagnostic": True})
                case = run_case(shared, max(config.repeats), n, slices, energy_j, wavelength_m, results["selected_beams"], output, False)
                comparisons = {name: _compare(baseline[f"observables_{name}"], case[f"observables_{name}"], convergence=True) for name in ("ours", "abtem")}
                case["refinement_kind"] = kind
                case["comparison_to_finest_requested"] = comparisons
                if not case["checks"]["exit_wave"]:
                    case["native_constant_control_wave_error_secondary"] = native_constant_control(shared, max(config.repeats), energy_j, audit, output / case["artifact"])
                results["additional_refinement_cases"].append(case)
                print(f"Extra {case['key']}: wave={case['relative_exit_wave_error']:.3e}; max intensity change={max(item['max_significant_relative_error'] for item in comparisons.values()):.2%}", flush=True)
                _write_json(output / "results.json", results)
    primary = [case for case in results["cases"] if case["primary_comparison"]]
    results["checks"] = {
        "analytical_regressions": results["analytical_regressions"]["passed"],
        "implementation_agreement_all_requested_configurations": all(case["status"] == "PASS" for case in results["cases"]),
        "implementation_agreement_additional_refinements": all(case["status"] == "PASS" for case in results["additional_refinement_cases"]),
        "physical_convergence_finest_requested_grid_and_slices": results["convergence"]["passed"],
        "disabled_channels_exactly_zero": all(case["checks"]["empty_channels_exactly_zero"] for case in primary),
        "unfiltered_si_norm_conserved": all(case["checks"]["unfiltered_norm_conserved"] for case in primary),
        "direct_repeated_potential_verified": results["direct_repeated_potential_verification"]["passed"],
        "independent_antialias_aperture_matches": all(item["independent_aperture_max_error"] < 1e-12 for item in results["shared_configurations"] if not item.get("additional_diagnostic")),
        "shared_native_potentials_unchanged": all(item["native_potential_unchanged"] for item in results["shared_configurations"] if not item.get("additional_diagnostic")),
        "original_structure_file_unchanged": hashlib.sha256(Path(config.structure).read_bytes()).hexdigest() == verification["input_sha256"],
    }
    results["status"] = "PASS" if all(results["checks"].values()) else "FAIL"
    results["runtime_seconds"] = time.perf_counter() - start
    _write_json(output / "results.json", results)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, default=PROJECT_ROOT / "Si_110_temp300.lmp")
    parser.add_argument("--energy-kev", type=float, default=60)
    parser.add_argument("--grids", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--slices-per-repeat", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--repeats", type=int, nargs="+", default=[13, 26, 52])
    parser.add_argument("--initial-grid", type=int, default=128)
    parser.add_argument("--initial-slices", type=int, default=8)
    parser.add_argument("--beam-count", type=int, default=6)
    parser.add_argument("--skip-extra-refinement", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "benchmark_results/elastic_si110")
    args = parser.parse_args(argv)
    try:
        config = ElasticConfig(
            structure=str(args.structure.resolve()), energy_kev=args.energy_kev,
            grids=tuple(args.grids), slices_per_repeat=tuple(args.slices_per_repeat),
            repeats=tuple(args.repeats), initial_grid=args.initial_grid, initial_slices=args.initial_slices,
            beam_count=args.beam_count, extra_refinement=not args.skip_extra_refinement,
        )
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output.resolve()
    results = run_benchmark(config, output)
    from .elastic_outputs import write_figures, write_report
    results["figures"] = write_figures(results, output)
    write_report(results, output)
    _write_json(output / "results.json", results)
    print(f"{results['benchmark']}: {results['status']}")
    print(f"Agreement: {results['checks']['implementation_agreement_all_requested_configurations']}; requested physical convergence: {results['checks']['physical_convergence_finest_requested_grid_and_slices']}")
    print(f"Results: {output}")
    return 0 if results["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
