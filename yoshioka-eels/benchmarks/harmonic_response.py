"""Benchmark 3: exact fixed-Q, T=0 harmonic-oscillator scattering response."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shlex
import sys

import numpy as np
import scipy
from scipy import constants
from scipy.integrate import trapezoid
from scipy.special import roots_hermite

from yoshioka.harmonic_oscillator import (
    HarmonicOscillator, fock_scattering_matrix, ladder_operators,
    quadrature_scattering_block, quadrature_vacuum_amplitudes,
    resolve_coupling, spectral_correlation,
)
from .oscillator_reference import (
    adaptive_cutoff, exact_correlation, omitted_inelastic_fraction,
    omitted_probability, poisson_weights, vacuum_amplitudes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class OscillatorConfig:
    mass_amu: float = 28.085
    energy_mev: float = 60.0
    eta: float | None = None
    q_inv_angstrom: float | None = None
    sweep_etas: tuple[float, ...] = (0.0, 0.01, 0.1, 0.5, 1.0, 4.0)
    physical_cutoffs: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)
    working_dimensions: tuple[int, ...] = (32, 64, 96, 128)
    quadrature_orders: tuple[int, ...] = (32, 64, 128, 256)
    block_n_max: int = 6
    correlation_time_points: int = 257
    fwhm_mev: float = 2.0
    amplitude_tolerance: float = 1e-10
    probability_tolerance: float = 1e-10
    block_tolerance: float = 1e-10
    weight_tolerance: float = 1e-10
    tail_tolerance: float = 1e-12
    moment_tolerance: float = 1e-8
    correlation_tolerance: float = 1e-10
    spectrum_area_tolerance: float = 1e-10
    consistency_rtol: float = 1e-12

    def __post_init__(self):
        oscillator = self.oscillator
        resolve_coupling(oscillator, None if self.q_inv_angstrom is None else self.q_inv_angstrom * 1e10, self.eta, self.consistency_rtol)
        if not self.sweep_etas or any(not np.isfinite(eta) or eta < 0 for eta in self.sweep_etas):
            raise ValueError("sweep_etas must contain nonnegative finite couplings")
        for name, minimum in (("physical_cutoffs", 0), ("working_dimensions", 1), ("quadrature_orders", 1)):
            values = getattr(self, name)
            if not values or any(isinstance(value, bool) or not isinstance(value, int) or value < minimum for value in values):
                raise ValueError(f"{name} must contain integers >= {minimum}")
            if tuple(sorted(set(values))) != tuple(values):
                raise ValueError(f"{name} must be unique and increasing")
        for name in ("amplitude_tolerance", "probability_tolerance", "block_tolerance", "weight_tolerance", "tail_tolerance", "moment_tolerance", "correlation_tolerance", "spectrum_area_tolerance", "fwhm_mev"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.tail_tolerance >= 1:
            raise ValueError("tail_tolerance must be below one")
        if isinstance(self.block_n_max, bool) or not isinstance(self.block_n_max, int) or self.block_n_max < 0:
            raise ValueError("block_n_max must be a nonnegative integer")
        if not isinstance(self.correlation_time_points, int) or self.correlation_time_points < 17:
            raise ValueError("correlation_time_points must be an integer >= 17")

    @property
    def oscillator(self) -> HarmonicOscillator:
        return HarmonicOscillator.from_amu_mev(self.mass_amu, self.energy_mev)


def constants_record() -> dict:
    from scipy.constants import _codata
    import matplotlib

    names = ("atomic mass constant", "Planck constant", "reduced Planck constant", "elementary charge")
    return {
        "configuration": "installed scipy.constants, as documented in Benchmark 2; no ASE constant substitution",
        "codata_release": getattr(_codata, "_current_codata", "see installed SciPy version"),
        "source_module": str(Path(_codata.__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(_codata.__file__).read_bytes()).hexdigest(),
        "values": {name: {"value": constants.physical_constants[name][0], "units": constants.physical_constants[name][1], "standard_uncertainty": constants.physical_constants[name][2]} for name in names},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
    }


def _complex_json(array: np.ndarray) -> list:
    return np.stack((array.real, array.imag), axis=-1).tolist()


def _amplitude_error(numerical, reference) -> float:
    # Absolute vector error, not an error divided by a possibly tiny retained norm.
    return float(np.linalg.norm(numerical - reference))


def retained_statistics(weights: np.ndarray, eta: float, quantum_j: float) -> dict:
    """Raw moments of retained weights, never conditional/renormalized moments.

    second_raw-first_raw^2 is a variance estimator. It equals a variance only
    in the normalized limit; underconverged cutoffs are explicitly labelled.
    """
    n = np.arange(weights.size, dtype=np.float64)
    weight, mean = float(np.sum(weights)), float(n @ weights)
    second = float(n**2 @ weights)
    variance = second - mean**2
    return {
        "retained_weight": weight, "raw_mean_occupation": mean,
        "raw_second_moment": second, "variance_estimate_from_raw_moments": variance,
        "mean_occupation_error_scaled": abs(mean - eta) / max(1, eta),
        "variance_error_scaled": abs(variance - eta) / max(1, eta),
        "raw_mean_energy_loss_j": mean * quantum_j,
        "raw_mean_energy_loss_mev": mean * quantum_j / (1e-3 * constants.elementary_charge),
        "mean_energy_error_scaled": abs(mean - eta) / max(1, eta),
    }


def broaden_spectrum(line_energies_j: np.ndarray, weights: np.ndarray, fwhm_j: float) -> tuple[dict, dict]:
    """Normalized Gaussian broadening in SI; densities are per joule.

    The plotting grid extends eight Gaussian sigmas beyond both endpoint lines.
    Its finite-window area is checked against the retained weight, without any
    rescaling. Samples are separated by at most FWHM/16.
    """
    if not np.isfinite(fwhm_j) or fwhm_j <= 0:
        raise ValueError("fwhm_j must be positive and finite")
    energies = np.asarray(line_energies_j, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if energies.ndim != 1 or energies.size == 0 or weights.shape != energies.shape or not np.all(np.isfinite(energies)) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("line energies and nonnegative weights must be finite vectors of equal length")
    sigma = fwhm_j / (2 * np.sqrt(2 * np.log(2)))
    start, stop = float(np.min(energies) - 8 * sigma), float(np.max(energies) + 8 * sigma)
    points = max(257, int(np.ceil((stop - start) / (fwhm_j / 16))) + 1)
    if points > 2_000_000:
        raise ValueError("spectrum grid exceeds two million points; increase the plotting FWHM")
    grid = np.linspace(start, stop, points, dtype=np.float64)
    elastic, inelastic = np.zeros_like(grid), np.zeros_like(grid)
    for index, (center, weight) in enumerate(zip(energies, weights)):
        profile = weight * np.exp(-0.5 * ((grid - center) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)
        if index == 0:
            elastic += profile
        else:
            inelastic += profile
    density = elastic + inelastic
    retained = float(np.sum(weights))
    integral = float(trapezoid(density, grid))
    arrays = {"energy_grid_j": grid, "total_density_per_j": density, "elastic_density_per_j": elastic, "inelastic_density_per_j": inelastic}
    metrics = {
        "fwhm_j": fwhm_j, "sigma_j": float(sigma), "samples": points,
        "energy_min_j": start, "energy_max_j": stop, "density_units": "J^-1",
        "plotted_density_units": "meV^-1", "density_to_per_mev_factor": 1e-3 * constants.elementary_charge,
        "retained_unbroadened_weight": retained, "integrated_broadened_weight": integral,
        "absolute_area_error": abs(integral - retained),
        "elastic_area_error": abs(float(trapezoid(elastic, grid)) - float(weights[0])),
        "inelastic_area_error": abs(float(trapezoid(inelastic, grid)) - float(np.sum(weights[1:]))),
        "normalization_applied": False,
        "negative_energy_density_interpretation": "Gaussian broadening tails only; all unbroadened transitions have nonnegative energy",
    }
    return arrays, metrics


def _run_case(xi: float, eta: float, q_m_inv: float, key: str, config: OscillatorConfig) -> tuple[dict, dict]:
    oscillator = config.oscillator
    adaptive_p = adaptive_cutoff(eta, config.tail_tolerance)
    p_test = max(max(config.physical_cutoffs), adaptive_p)
    final_d, final_nodes = max(config.working_dimensions), max(config.quadrature_orders)
    if final_d <= max(p_test + 1, config.block_n_max):
        raise ValueError(f"eta={eta:g} needs a working dimension larger than p+1={p_test+1} and block n_max={config.block_n_max}; increase --working-dimensions")
    n_eval = min(final_d - 1, p_test + 16)
    exact = vacuum_amplitudes(xi, n_eval)
    exact_weights = poisson_weights(eta, n_eval)
    matrices = {d: fock_scattering_matrix(xi, d) for d in config.working_dimensions}
    quad = {nodes: quadrature_vacuum_amplitudes(xi, n_eval, nodes) for nodes in config.quadrature_orders}
    blocks = {nodes: quadrature_scattering_block(xi, config.block_n_max, nodes) for nodes in config.quadrature_orders}
    fock_a = matrices[final_d][:n_eval+1, 0]
    quad_a = quad[final_nodes]
    block_size = config.block_n_max + 1
    fock_block, quad_block = matrices[final_d][:block_size, :block_size], blocks[final_nodes]
    times = np.linspace(0, 1 / oscillator.frequency_hz, config.correlation_time_points)
    correlation_exact = exact_correlation(eta, oscillator.angular_frequency_rad_s, times)
    working_study = []
    for d in config.working_dimensions:
        matrix = matrices[d]
        valid_adaptive, valid_test = d > adaptive_p + 1, d > p_test
        block_valid = d >= block_size
        entry = {
            "D": d, "adaptive_p_supported_with_guard": valid_adaptive,
            "all_tested_cutoffs_supported": valid_test,
            "adaptive_amplitude_error": _amplitude_error(matrix[:adaptive_p+1, 0], exact[:adaptive_p+1]) if valid_adaptive else None,
            "all_cutoffs_amplitude_error": _amplitude_error(matrix[:p_test+1, 0], exact[:p_test+1]) if valid_test else None,
            "general_block_error_vs_final_quadrature": _amplitude_error(matrix[:block_size, :block_size], quad_block) if block_valid else None,
            "change_to_largest_D_adaptive": _amplitude_error(matrix[:adaptive_p+1, 0], fock_a[:adaptive_p+1]) if valid_adaptive else None,
        }
        entry["adaptive_converged"] = bool(valid_adaptive and block_valid and entry["adaptive_amplitude_error"] < config.amplitude_tolerance and entry["change_to_largest_D_adaptive"] < config.amplitude_tolerance and entry["general_block_error_vs_final_quadrature"] < config.block_tolerance)
        entry["all_cutoffs_converged"] = bool(entry["adaptive_converged"] and valid_test and entry["all_cutoffs_amplitude_error"] < config.amplitude_tolerance)
        working_study.append(entry)
    quadrature_study = []
    for nodes in config.quadrature_orders:
        entry = {
            "nodes": nodes,
            "adaptive_amplitude_error": _amplitude_error(quad[nodes][:adaptive_p+1], exact[:adaptive_p+1]),
            "all_cutoffs_amplitude_error": _amplitude_error(quad[nodes][:p_test+1], exact[:p_test+1]),
            "change_to_largest_order_adaptive": _amplitude_error(quad[nodes][:adaptive_p+1], quad_a[:adaptive_p+1]),
            "general_block_error_vs_final_fock": _amplitude_error(blocks[nodes], fock_block),
        }
        entry["adaptive_converged"] = bool(entry["adaptive_amplitude_error"] < config.amplitude_tolerance and entry["change_to_largest_order_adaptive"] < config.amplitude_tolerance and entry["general_block_error_vs_final_fock"] < config.block_tolerance)
        entry["all_cutoffs_converged"] = bool(entry["adaptive_converged"] and entry["all_cutoffs_amplitude_error"] < config.amplitude_tolerance)
        quadrature_study.append(entry)
    cutoff_study = []
    for p in sorted(set(config.physical_cutoffs + (adaptive_p,))):
        exact_retained = float(np.sum(exact_weights[:p+1]))
        tail = omitted_probability(eta, p)
        methods = {}
        for name, amplitudes in (("fock", fock_a), ("quadrature", quad_a)):
            weights = np.abs(amplitudes[:p+1])**2
            correlation = spectral_correlation(amplitudes[:p+1], oscillator.quantum_energy_j, times)
            direct_tail = float(np.sum(np.abs(amplitudes[p+1:])**2))
            statistics = retained_statistics(weights, eta, oscillator.quantum_energy_j)
            methods[name] = {
                "amplitude_error": _amplitude_error(amplitudes[:p+1], exact[:p+1]),
                "maximum_absolute_probability_error": float(np.max(np.abs(weights - exact_weights[:p+1]))),
                **statistics,
                "retained_weight_error": abs(statistics["retained_weight"] - exact_retained),
                "missing_weight_from_norm": 1 - statistics["retained_weight"],
                "missing_weight_absolute_error": abs(1 - statistics["retained_weight"] - tail),
                "direct_nonnegative_omitted_sum_through_n_eval": direct_tail,
                "direct_omitted_sum_absolute_error": abs(direct_tail - tail),
                "correlation_max_absolute_error": float(np.max(np.abs(correlation - correlation_exact))),
                "correlation_c0_real": float(correlation[0].real),
            }
        cutoff_study.append({
            "p": p, "adaptive": p == adaptive_p, "exact_retained_weight": exact_retained,
            "exact_omitted_probability": tail,
            "exact_omitted_inelastic_fraction": omitted_inelastic_fraction(eta, p),
            "status": "CONVERGED_PHYSICAL_CUTOFF" if tail < config.tail_tolerance else "DELIBERATELY_UNDERCONVERGED",
            "methods": methods,
        })
    adaptive = next(entry for entry in cutoff_study if entry["p"] == adaptive_p)
    negative_fock = fock_scattering_matrix(-xi, final_d)
    negative_quad = quadrature_scattering_block(-xi, config.block_n_max, final_nodes)
    negative_quad_a = quadrature_vacuum_amplitudes(-xi, p_test, final_nodes)
    nodes, weights = roots_hermite(final_nodes)
    variance_u_quad = float(oscillator.oscillator_length_m**2 * np.sum(weights * nodes**2) / np.sqrt(np.pi))
    lower, raise_ = ladder_operators(final_d)
    coordinate = oscillator.zero_point_displacement_m * (lower + raise_)
    variance_u_fock = float((coordinate @ coordinate)[0, 0].real)
    dw_expected = eta / 2
    dw = {}
    for name, a0, u2 in (("fock", fock_a[0], variance_u_fock), ("quadrature", quad_a[0], variance_u_quad)):
        exponent = float(q_m_inv**2 * u2 / 2)
        dw[name] = {
            "ground_state_displacement_variance_m2": u2, "W0": exponent,
            "W0_absolute_error": abs(exponent - dw_expected),
            "elastic_amplitude_error": float(abs(a0 - np.exp(-exponent))),
            "elastic_weight_error": float(abs(abs(a0)**2 - np.exp(-2 * exponent))),
        }
    block_error = _amplitude_error(fock_block, quad_block)
    reversal_fock = _amplitude_error(negative_fock[:block_size, :block_size], fock_block.conj().T)
    reversal_quad = _amplitude_error(negative_quad, quad_block.conj().T)
    previous_d = config.working_dimensions[-2] if len(config.working_dimensions) >= 2 else None
    previous_nodes = config.quadrature_orders[-2] if len(config.quadrature_orders) >= 2 else None
    d_stable = bool(previous_d is not None and previous_d > p_test and _amplitude_error(matrices[previous_d][:p_test+1, 0], fock_a[:p_test+1]) < config.amplitude_tolerance and _amplitude_error(matrices[previous_d][:block_size, :block_size], fock_block) < config.block_tolerance)
    quad_stable = bool(previous_nodes is not None and _amplitude_error(quad[previous_nodes][:p_test+1], quad_a[:p_test+1]) < config.amplitude_tolerance and _amplitude_error(blocks[previous_nodes], quad_block) < config.block_tolerance)
    recoil_j = constants.hbar**2 * q_m_inv**2 / (2 * oscillator.mass_kg)
    losses = oscillator.energy_losses_j(n_eval)
    checks = {
        "fock_amplitudes": _amplitude_error(fock_a[:p_test+1], exact[:p_test+1]) < config.amplitude_tolerance,
        "quadrature_amplitudes": _amplitude_error(quad_a[:p_test+1], exact[:p_test+1]) < config.amplitude_tolerance,
        "probabilities": all(max(entry["methods"][name]["maximum_absolute_probability_error"] for entry in cutoff_study) < config.probability_tolerance for name in ("fock", "quadrature")),
        "retained_weights_and_tails": all(max(entry["methods"][name][field] for entry in cutoff_study) < config.weight_tolerance for name in ("fock", "quadrature") for field in ("retained_weight_error", "missing_weight_absolute_error", "direct_omitted_sum_absolute_error")),
        "adaptive_tail": adaptive["exact_omitted_probability"] < config.tail_tolerance,
        "adaptive_moments": all(adaptive["methods"][name][field] < config.moment_tolerance for name in ("fock", "quadrature") for field in ("mean_occupation_error_scaled", "variance_error_scaled", "mean_energy_error_scaled")),
        "adaptive_normalization": all(abs(1 - adaptive["methods"][name]["retained_weight"]) < config.weight_tolerance for name in ("fock", "quadrature")),
        "debye_waller_amplitude_and_intensity": all(entry[field] < config.amplitude_tolerance for entry in dw.values() for field in ("W0_absolute_error", "elastic_amplitude_error", "elastic_weight_error")),
        "recoil_identity": abs(recoil_j - eta * oscillator.quantum_energy_j) / (oscillator.quantum_energy_j * max(1, eta)) < config.moment_tolerance,
        "elastic_at_zero_no_gain_lines": bool(losses[0] == 0 and np.all(losses >= 0)),
        "energy_loss_assignments": bool(np.allclose(losses / oscillator.quantum_energy_j, np.arange(losses.size), rtol=0, atol=1e-12)),
        "opposite_Q_vacuum_conjugation": _amplitude_error(negative_fock[:p_test+1, 0], fock_a[:p_test+1].conj()) < config.amplitude_tolerance and _amplitude_error(negative_quad_a, quad_a[:p_test+1].conj()) < config.amplitude_tolerance,
        "general_block": block_error < config.block_tolerance,
        "general_block_opposite_Q_adjoint": max(reversal_fock, reversal_quad) < config.block_tolerance,
        "working_dimension_convergence": d_stable,
        "quadrature_order_convergence": quad_stable,
        "adaptive_correlation": all(adaptive["methods"][name]["correlation_max_absolute_error"] < config.correlation_tolerance for name in ("fock", "quadrature")),
    }
    if eta == 0:
        checks["Q_zero_identity"] = max(_amplitude_error(fock_a[:p_test+1], exact[:p_test+1]), _amplitude_error(quad_a[:p_test+1], exact[:p_test+1])) < config.amplitude_tolerance
    arrays = {
        "occupations": np.arange(n_eval+1), "line_energy_losses_j": losses,
        "exact_amplitudes": exact, "fock_amplitudes": fock_a, "quadrature_amplitudes": quad_a,
        "exact_weights": exact_weights, "fock_weights": np.abs(fock_a)**2, "quadrature_weights": np.abs(quad_a)**2,
        "fock_general_block": fock_block, "quadrature_general_block": quad_block,
        "times_s": times, "correlation_exact": correlation_exact,
        "correlation_fock_adaptive": spectral_correlation(fock_a[:adaptive_p+1], oscillator.quantum_energy_j, times),
        "correlation_quadrature_adaptive": spectral_correlation(quad_a[:adaptive_p+1], oscillator.quantum_energy_j, times),
    }
    record = {
        "key": key, "eta": eta, "xi": xi, "q_m_inv": q_m_inv,
        "adaptive_p": adaptive_p, "largest_physical_cutoff_tested": p_test,
        "tail_evaluation_n_max": n_eval, "exact_tail_beyond_n_eval": omitted_probability(eta, n_eval),
        "final_working_dimension": final_d, "final_quadrature_nodes": final_nodes,
        "first_converged_D_for_adaptive_p": next((row["D"] for row in working_study if row["adaptive_converged"]), None),
        "first_converged_D_for_all_cutoffs": next((row["D"] for row in working_study if row["all_cutoffs_converged"]), None),
        "first_converged_nodes_for_adaptive_p": next((row["nodes"] for row in quadrature_study if row["adaptive_converged"]), None),
        "first_converged_nodes_for_all_cutoffs": next((row["nodes"] for row in quadrature_study if row["all_cutoffs_converged"]), None),
        "fock_amplitude_error_all_cutoffs": _amplitude_error(fock_a[:p_test+1], exact[:p_test+1]),
        "quadrature_amplitude_error_all_cutoffs": _amplitude_error(quad_a[:p_test+1], exact[:p_test+1]),
        "general_block_frobenius_error": block_error,
        "general_block_opposite_Q_adjoint_error": {"fock": reversal_fock, "quadrature": reversal_quad},
        "extracted_fock_block_unitarity_defect_diagnostic_only": float(np.linalg.norm(fock_block.conj().T @ fock_block - np.eye(block_size))),
        "debye_waller": dw,
        "expected_recoil_energy_j": float(recoil_j), "expected_mean_energy_loss_j": eta * oscillator.quantum_energy_j,
        "working_dimension_study": working_study, "quadrature_study": quadrature_study,
        "physical_cutoff_study": cutoff_study,
        "amplitudes_through_p_test_re_im": {"exact": _complex_json(exact[:p_test+1]), "fock": _complex_json(fock_a[:p_test+1]), "quadrature": _complex_json(quad_a[:p_test+1])},
        "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL",
    }
    return record, arrays


def run_benchmark(config: OscillatorConfig, output: Path | None = None) -> dict:
    oscillator = config.oscillator
    q, xi, selected_eta = resolve_coupling(oscillator, None if config.q_inv_angstrom is None else config.q_inv_angstrom * 1e10, config.eta, config.consistency_rtol)
    inputs = [(f"eta_{index}", np.sqrt(eta), eta) for index, eta in enumerate(config.sweep_etas)]
    selected_key = next((key for key, _, eta in inputs if eta == selected_eta and xi >= 0 and config.q_inv_angstrom is None), None)
    if selected_key is None:
        selected_key = "selected_Q"
        inputs.append((selected_key, xi, selected_eta))
    results = {
        "schema_version": 1, "benchmark": "Benchmark 3: exact harmonic-oscillator scattering response at T=0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "parameters": asdict(config),
        "constants": constants_record(), "cases": [], "selected_case": selected_key,
        "physical_parameters_si": {
            "mass_kg": oscillator.mass_kg, "quantum_energy_j": oscillator.quantum_energy_j,
            "omega0_rad_per_s": oscillator.angular_frequency_rad_s, "frequency_hz": oscillator.frequency_hz,
            "period_s": 1 / oscillator.frequency_hz, "u_zp_m": oscillator.zero_point_displacement_m,
            "oscillator_length_m": oscillator.oscillator_length_m,
            "selected_Q_m_inv": q, "selected_Q_inv_angstrom": q * 1e-10, "selected_xi": xi, "selected_eta": selected_eta,
        },
        "conventions": {
            "operator": "exp(-i*Q*u), Q angular m^-1; xi=Q*u_zp; eta=xi^2; no extra 2*pi",
            "units": {"amu_to_kg": constants.atomic_mass, "mev_to_j": 1e-3 * constants.elementary_charge, "inverse_angstrom_to_inverse_m": 1e10, "angular_frequency": "omega=energy/hbar=2*pi*f; f=energy/h"},
            "oscillator_length": "ell=sqrt(hbar/(M*omega))=sqrt(2)*u_zp; x=u/ell",
            "physical_cutoff": "n=0..p; independent of working dimension D",
            "amplitude_error": "sqrt(sum(n=0..p) abs(A_num-A_exact)^2); no phase alignment",
            "block_error": "absolute Frobenius norm of quadrature minus Fock block",
            "matrix_dtype": "complex128", "real_dtype": "float64",
            "statistical_moments": "raw retained moments; variance estimator=second_raw-first_raw^2; no conditional renormalization",
            "correlation": "sum_n retained |A_n|^2 exp(-i*n*omega*t), one period inclusive; no numerical Fourier transform",
            "weights": "dimensionless oscillator spectral weights, excluding atomic form factors and electron-flux factors",
            "broadened_density": "per joule internally, per meV in plots; normalized individual Gaussian kernels; no output renormalization",
        },
    }
    if output is not None:
        output = Path(output).resolve()
        if (output / "results.json").is_file():
            previous = json.loads((output / "results.json").read_text())
            if previous.get("benchmark") != results["benchmark"]:
                raise ValueError("output directory belongs to another benchmark; choose a separate output directory")
        (output / "arrays").mkdir(parents=True, exist_ok=True)
    selected_arrays = None
    for key, coupling, eta in inputs:
        case_q = q if key == "selected_Q" else coupling / oscillator.zero_point_displacement_m
        case, arrays = _run_case(float(coupling), float(eta), float(case_q), key, config)
        if output is not None:
            path = output / "arrays" / f"{key}.npz"
            np.savez_compressed(path, **arrays)
            case["artifact"] = str(path.relative_to(output))
        results["cases"].append(case)
        if key == selected_key:
            selected_arrays = arrays
    selected = next(case for case in results["cases"] if case["key"] == selected_key)
    p = selected["adaptive_p"]
    spectrum_arrays, spectrum_metrics = broaden_spectrum(
        selected_arrays["line_energy_losses_j"][:p+1], selected_arrays["fock_weights"][:p+1], config.fwhm_mev * 1e-3 * constants.elementary_charge,
    )
    results["spectrum"] = spectrum_metrics
    if output is not None:
        np.savez_compressed(output / "broadened_spectrum.npz", **spectrum_arrays)
    # Deliberate mathematical stress example, distinct from user-selected Q.
    diagnostic_p, diagnostic_eta = 2, 4.0
    small = fock_scattering_matrix(np.sqrt(diagnostic_eta), diagnostic_p + 1)
    projected = quadrature_scattering_block(np.sqrt(diagnostic_eta), diagnostic_p, max(config.quadrature_orders))
    diagnostic_exact = vacuum_amplitudes(np.sqrt(diagnostic_eta), diagnostic_p)
    results["premature_truncation_diagnostic"] = {
        "eta": diagnostic_eta, "physical_p": diagnostic_p, "premature_D": diagnostic_p + 1,
        "premature_full_state_weight": float(np.sum(np.abs(small[:, 0])**2)),
        "correct_retained_weight": float(np.sum(np.abs(diagnostic_exact)**2)),
        "premature_vacuum_vector_error": _amplitude_error(small[:, 0], diagnostic_exact),
        "projected_infinite_vs_exponentiated_truncation_block_error": _amplitude_error(projected, small),
        "finite_operator_unitarity_defect": float(np.linalg.norm(small.conj().T @ small - np.eye(diagnostic_p + 1))),
        "correct_projected_block_unitarity_defect": float(np.linalg.norm(projected.conj().T @ projected - np.eye(diagnostic_p + 1))),
        "interpretation": "a normalized finite-D state can have incorrect retained coefficients; never use its unit norm as evidence of converged scattering weights",
    }
    results["checks"] = {
        "numerical_matrix_elements": all(all(case["checks"][name] for name in ("fock_amplitudes", "quadrature_amplitudes", "probabilities", "retained_weights_and_tails", "debye_waller_amplitude_and_intensity", "opposite_Q_vacuum_conjugation", "general_block", "general_block_opposite_Q_adjoint", "working_dimension_convergence", "quadrature_order_convergence")) for case in results["cases"]),
        "adaptive_physical_cutoff_and_moments": all(all(case["checks"][name] for name in ("adaptive_tail", "adaptive_moments", "adaptive_normalization", "adaptive_correlation")) for case in results["cases"]),
        "energy_assignments_and_recoil": all(all(case["checks"][name] for name in ("recoil_identity", "elastic_at_zero_no_gain_lines", "energy_loss_assignments")) for case in results["cases"]),
        "all_case_checks": all(case["status"] == "PASS" for case in results["cases"]),
        "spectral_plotting_area": max(spectrum_metrics[name] for name in ("absolute_area_error", "elastic_area_error", "inelastic_area_error")) < config.spectrum_area_tolerance,
        "premature_truncation_effect_demonstrated": results["premature_truncation_diagnostic"]["premature_vacuum_vector_error"] > 1e-2 and results["premature_truncation_diagnostic"]["finite_operator_unitarity_defect"] < config.block_tolerance,
    }
    results["prior_benchmark_statuses_preserved"] = {}
    for name in ("free_electron", "elastic_si110"):
        path = PROJECT_ROOT / "benchmark_results" / name / "results.json"
        if path.is_file():
            prior = json.loads(path.read_text())
            results["prior_benchmark_statuses_preserved"][name] = prior.get("status")
    results["status"] = "PASS" if all(results["checks"].values()) else "FAIL"
    return results


def reproduction_command(config: OscillatorConfig, output: Path) -> str:
    args = [".venv/bin/python", "-m", "benchmarks.harmonic_response", "--mass-amu", str(config.mass_amu), "--energy-mev", str(config.energy_mev)]
    if config.eta is not None:
        args += ["--eta", str(config.eta)]
    if config.q_inv_angstrom is not None:
        args += ["--q-inv-angstrom", str(config.q_inv_angstrom)]
    for name in ("sweep_etas", "physical_cutoffs", "working_dimensions", "quadrature_orders"):
        args += ["--" + name.replace("_", "-"), *map(str, getattr(config, name))]
    for name in ("block_n_max", "correlation_time_points", "fwhm_mev", "amplitude_tolerance", "probability_tolerance", "block_tolerance", "weight_tolerance", "tail_tolerance", "moment_tolerance", "correlation_tolerance", "spectrum_area_tolerance", "consistency_rtol"):
        args += ["--" + name.replace("_", "-"), str(getattr(config, name))]
    return shlex.join(args + ["--output", str(output)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = OscillatorConfig()
    parser.add_argument("--mass-amu", type=float, default=defaults.mass_amu)
    parser.add_argument("--energy-mev", type=float, default=defaults.energy_mev)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--q-inv-angstrom", type=float, help="signed angular wavevector, not cycles/angstrom")
    for name, dtype in (("sweep_etas", float), ("physical_cutoffs", int), ("working_dimensions", int), ("quadrature_orders", int)):
        parser.add_argument("--" + name.replace("_", "-"), type=dtype, nargs="+", default=getattr(defaults, name))
    for name in ("block_n_max", "correlation_time_points"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("fwhm_mev", "amplitude_tolerance", "probability_tolerance", "block_tolerance", "weight_tolerance", "tail_tolerance", "moment_tolerance", "correlation_tolerance", "spectrum_area_tolerance", "consistency_rtol"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "benchmark_results/harmonic_response")
    args = parser.parse_args(argv)
    kwargs = vars(args).copy()
    output = kwargs.pop("output").resolve()
    for name in ("sweep_etas", "physical_cutoffs", "working_dimensions", "quadrature_orders"):
        kwargs[name] = tuple(kwargs[name])
    try:
        config = OscillatorConfig(**kwargs)
        results = run_benchmark(config, output)
    except ValueError as exc:
        parser.error(str(exc))
    results["reproduction_command"] = reproduction_command(config, output)
    from .oscillator_outputs import write_figures, write_report
    results["figures"] = write_figures(results, output)
    write_report(results, output)
    (output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{results['benchmark']}: {results['status']}")
    for case in results["cases"]:
        tail = next(row["exact_omitted_probability"] for row in case["physical_cutoff_study"] if row["adaptive"])
        print(f"  eta={case['eta']:g}: {case['status']}; adaptive p={case['adaptive_p']}; tail={tail:.3e}; Fock error={case['fock_amplitude_error_all_cutoffs']:.3e}; quadrature error={case['quadrature_amplitude_error_all_cutoffs']:.3e}")
    print(f"Results: {output}")
    return 0 if results["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
