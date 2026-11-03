"""Benchmark 4: finite-temperature fixed-Q harmonic-oscillator response."""

import argparse
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys

import numpy as np
from scipy import constants

from yoshioka.harmonic_oscillator import HarmonicOscillator, fock_scattering_matrix, quadrature_scattering_block, resolve_coupling
from yoshioka.thermal_response import (
    ThermalOscillatorState, broaden_signed_spectrum, final_state_cutoff,
    line_correlation, line_statistics, restrict_lines, transition_sum,
)
from .harmonic_response import constants_record
from .thermal_reference import (
    elastic_and_leading_terms, energy_window, exact_weights, exact_window_tail,
    reference_parameters, thermal_correlation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = "Benchmark 4: finite-temperature harmonic-oscillator scattering response"


@dataclass(frozen=True)
class ThermalConfig:
    mass_amu: float = 28.085
    energy_mev: float = 60.0
    eta: float | None = None
    q_inv_angstrom: float | None = None
    temperature_k: float = 300.0
    sweep_temperatures_k: tuple[float, ...] = (0, 10, 100, 300, 600, 1000)
    sweep_etas: tuple[float, ...] = (0, 0.01, 0.1, 0.5, 1, 4)
    working_dimensions: tuple[int, ...] = (64, 96, 128, 192)
    quadrature_orders: tuple[int, ...] = (32, 64, 128, 256)
    initial_cutoff_study: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 48)
    final_cutoff_study: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 48, 64, 96)
    energy_window_study: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 48)
    correlation_time_points: int = 257
    fwhm_mev: float = 2.0
    initial_tail_tolerance: float = 1e-13
    final_tail_tolerance: float = 1e-12
    energy_tail_tolerance: float = 1e-12
    matrix_tolerance: float = 1e-10
    line_tolerance: float = 1e-10
    summed_line_tolerance: float = 1e-9
    weight_tolerance: float = 1e-10
    moment_tolerance: float = 1e-8
    correlation_tolerance: float = 1e-10
    detailed_balance_atol: float = 1e-10
    detailed_balance_rtol: float = 1e-8
    coherent_tolerance: float = 1e-10
    spectrum_area_tolerance: float = 1e-8
    consistency_rtol: float = 1e-12

    def __post_init__(self):
        resolve_coupling(self.oscillator, None if self.q_inv_angstrom is None else self.q_inv_angstrom * 1e10, self.eta, self.consistency_rtol)
        for name in ("temperature_k",):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        for name in ("sweep_temperatures_k", "sweep_etas"):
            values = getattr(self, name)
            if not values or any(not np.isfinite(value) or value < 0 for value in values) or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique nonnegative finite values")
        for name in ("working_dimensions", "quadrature_orders", "initial_cutoff_study", "final_cutoff_study", "energy_window_study"):
            values = getattr(self, name)
            minimum = 1 if name in ("working_dimensions", "quadrature_orders") else 0
            if not values or any(isinstance(value, bool) or not isinstance(value, int) or value < minimum for value in values) or tuple(sorted(set(values))) != tuple(values):
                raise ValueError(f"{name} must contain increasing unique integers >= {minimum}")
        if len(self.working_dimensions) < 2 or len(self.quadrature_orders) < 2:
            raise ValueError("provide at least two working dimensions and quadrature orders for independent refinement")
        if isinstance(self.correlation_time_points, bool) or not isinstance(self.correlation_time_points, int) or self.correlation_time_points < 17:
            raise ValueError("correlation_time_points must be an integer >=17")
        for field in fields(self):
            if "tolerance" in field.name or field.name in ("detailed_balance_atol", "detailed_balance_rtol", "consistency_rtol", "fwhm_mev"):
                value = getattr(self, field.name)
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"{field.name} must be positive and finite")
                if "tail_tolerance" in field.name and value >= 1:
                    raise ValueError(f"{field.name} must be below one")

    @property
    def oscillator(self):
        return HarmonicOscillator.from_amu_mev(self.mass_amu, self.energy_mev)


def _complex(value):
    return [float(value.real), float(value.imag)]


def detailed_balance(ell, weights, x, atol, rtol, cutoff_bound):
    """Every numerical loss/gain pair is independently extracted from the sum."""
    lookup = dict(zip(map(int, ell), map(float, weights)))
    # A ratio is resolved at the requested relative accuracy only if physical
    # cutoff uncertainty is smaller than rtol times either member of the pair.
    # Every pair, including those below this floor, still receives the mixed
    # absolute/relative acceptance check.
    floor = max(1e-12, 100 * cutoff_bound, cutoff_bound / rtol)
    pairs = []
    for change in range(1, int(max(abs(ell[0]), abs(ell[-1]))) + 1):
        loss, gain = lookup.get(change, 0.0), lookup.get(-change, 0.0)
        expected = 0.0 if x is None else float(np.exp(-change * x) * loss)
        residual = abs(gain - expected)
        scale = atol + rtol * max(abs(gain), abs(expected))
        resolved = bool(x is not None and min(gain, loss) >= floor)
        log_error = float(abs(np.log(gain) - np.log(loss) + change * x)) if resolved else None
        pairs.append({"ell": change, "loss_weight": loss, "gain_weight": gain, "gain_expected_from_loss_for_check_only": expected,
                      "absolute_residual": residual, "mixed_residual": residual / scale,
                      "well_resolved_pair": resolved, "absolute_log_ratio_error": log_error})
    return {"absolute_tolerance": atol, "relative_tolerance": rtol,
            "definition": "abs(W_-ell-exp(-ell*x)*W_ell)/(atol+rtol*max(W_-ell,exp(-ell*x)*W_ell)) <= 1",
            "log_ratio_resolution_floor": floor, "pairs": pairs,
            "maximum_mixed_residual": max((row["mixed_residual"] for row in pairs), default=0.0),
            "maximum_absolute_residual": max((row["absolute_residual"] for row in pairs), default=0.0),
            "maximum_resolved_log_ratio_error": max((row["absolute_log_ratio_error"] for row in pairs if row["well_resolved_pair"]), default=None),
            "resolved_pair_count": sum(row["well_resolved_pair"] for row in pairs),
            "unresolved_pair_policy": "mixed absolute/relative test retained; no ratios reported for underflowed, negligible, or cutoff-limited pairs"}


def _metrics(response, loss_max, gain_max, exact, tail, eta, state, times=None, exact_correlation=None):
    ell, weights = restrict_lines(response.ell, response.weights, loss_max, gain_max)
    difference = np.abs(weights - exact)
    metrics = {
        "maximum_absolute_line_error": float(np.max(difference)),
        "summed_absolute_line_error_in_window": float(np.sum(difference)),
        "summed_absolute_line_error_with_exact_tail_bound": float(np.sum(difference)) + tail["omitted_weight_upper_bound"],
        **line_statistics(ell, weights, eta, state.mean_occupation, state.quantum_energy_j),
        "coherent_amplitude_re_im": _complex(response.coherent_amplitude),
        "coherent_weight": float(abs(response.coherent_amplitude)**2),
        "zero_energy_weight": float(weights[gain_max]),
        "retained_initial_weight": response.retained_initial_weight,
        "final_omitted_weight_residual": response.final_omitted_weight_residual,
        "numerical_weight_outside_reported_window": float(np.sum(response.weights) - np.sum(weights)),
        "numerical_gain_weight_in_window": float(np.sum(weights[ell < 0])),
        "numerical_loss_weight_in_window": float(np.sum(weights[ell > 0])),
    }
    if times is not None:
        correlation = line_correlation(ell, weights, state.quantum_energy_j, times)
        metrics.update(correlation_max_absolute_error=float(np.max(np.abs(correlation - exact_correlation))),
                       correlation_c0_real=float(correlation[0].real), correlation_c0_deficit=float(1 - correlation[0].real))
    return metrics, ell, weights


def _case(operator, state, matrices, quadratures, n_max, config):
    key, eta, xi, q = operator
    oscillator = config.oscillator
    m_max = state.initial_cutoff(config.initial_tail_tolerance)
    probabilities = state.probabilities(m_max)
    d_final, nodes_final = config.working_dimensions[-1], config.quadrature_orders[-1]
    matrix = matrices[d_final]
    loss_max, gain_max, tail = energy_window(eta, state.quantum_energy_j, state.temperature_k, config.energy_tail_tolerance)
    ell = np.arange(-gain_max, loss_max + 1)
    exact = exact_weights(ell, eta, state.quantum_energy_j, state.temperature_k)
    parameters = reference_parameters(eta, state.quantum_energy_j, state.temperature_k)
    exact_elastic = elastic_and_leading_terms(eta, state.quantum_energy_j, state.temperature_k)
    times = np.linspace(0, 1 / oscillator.frequency_hz, config.correlation_time_points)
    correlation_exact = thermal_correlation(times, oscillator.angular_frequency_rad_s, eta, state.quantum_energy_j, state.temperature_k)
    responses = {"fock": transition_sum(matrix, probabilities, n_max), "quadrature": transition_sum(quadratures[nodes_final], probabilities, n_max)}
    methods, reported = {}, {}
    initial_tail = state.omitted_probability(m_max)
    positive_final_tail = float(np.sum(np.abs(matrix[n_max+1:, :m_max+1])**2 @ probabilities))
    cutoff_bound = initial_tail + positive_final_tail + tail["omitted_weight_upper_bound"]
    for name, response in responses.items():
        metrics, _, weights = _metrics(response, loss_max, gain_max, exact, tail, eta, state, times, correlation_exact)
        metrics["coherent_amplitude_error"] = float(abs(response.coherent_amplitude - exact_elastic["coherent_amplitude"]))
        metrics["coherent_weight_error"] = abs(metrics["coherent_weight"] - exact_elastic["coherent_weight"])
        metrics["zero_energy_weight_error"] = abs(metrics["zero_energy_weight"] - exact_elastic["zero_energy_weight"])
        metrics["zero_energy_minus_coherent_weight"] = metrics["zero_energy_weight"] - metrics["coherent_weight"]
        metrics["detailed_balance"] = detailed_balance(response.ell, response.weights, parameters["x"], config.detailed_balance_atol, config.detailed_balance_rtol, cutoff_bound)
        methods[name], reported[name] = metrics, weights

    def study_metrics(m_cut, n_cut, working_matrix=matrix):
        response = transition_sum(working_matrix, state.probabilities(m_cut), n_cut)
        metrics, _, _ = _metrics(response, loss_max, gain_max, exact, tail, eta, state)
        return metrics

    initial_study = []
    for cutoff in sorted(set(config.initial_cutoff_study + (m_max,))):
        if cutoff >= d_final:
            continue
        residual = state.omitted_probability(cutoff)
        initial_study.append({"m_max": cutoff, "exact_initial_tail": residual, "status": "CONVERGED_INITIAL_CUTOFF" if residual < config.initial_tail_tolerance else "DELIBERATELY_UNDERCONVERGED", **study_metrics(cutoff, n_max)})
    final_study = []
    for cutoff in sorted(set(config.final_cutoff_study + (n_max, min(n_max+8, d_final-2)))):
        if cutoff >= d_final:
            continue
        positive_tail = float(np.sum(np.abs(matrix[cutoff+1:, :m_max+1])**2 @ probabilities))
        final_study.append({"n_max": cutoff, "positive_high_basis_final_tail": positive_tail,
                            "status": "CONVERGED_FINAL_CUTOFF" if positive_tail < config.final_tail_tolerance else "DELIBERATELY_UNDERCONVERGED", **study_metrics(m_max, cutoff)})
    working_study = []
    for dimension in config.working_dimensions:
        supported = dimension > max(m_max, n_max) + 1
        row = {"D": dimension, "supported_with_guard": supported}
        if supported:
            response = transition_sum(matrices[dimension], probabilities, n_max)
            _, _, weights = _metrics(response, loss_max, gain_max, exact, tail, eta, state)
            amplitude_change = float(np.linalg.norm((matrices[dimension][:n_max+1, :m_max+1] - matrix[:n_max+1, :m_max+1]) * np.sqrt(probabilities)))
            change = float(np.max(np.abs(weights - reported["fock"])))
            row.update(weighted_amplitude_change=amplitude_change, maximum_line_change=change, **study_metrics(m_max, n_max, matrices[dimension]))
            row["converged"] = amplitude_change < config.matrix_tolerance and change < config.line_tolerance
        else:
            row["converged"] = False
        working_study.append(row)
    quad_study = []
    for nodes in config.quadrature_orders:
        response = transition_sum(quadratures[nodes], probabilities, n_max)
        metrics, _, weights = _metrics(response, loss_max, gain_max, exact, tail, eta, state)
        amplitude_error = float(np.linalg.norm((quadratures[nodes][:n_max+1, :m_max+1] - matrix[:n_max+1, :m_max+1]) * np.sqrt(probabilities)))
        line_change = float(np.max(np.abs(weights - reported["fock"])))
        quad_study.append({"nodes": nodes, "weighted_amplitude_difference_vs_fock": amplitude_error,
                           "maximum_line_difference_vs_fock": line_change, "converged": amplitude_error < config.matrix_tolerance and line_change < config.line_tolerance, **metrics})
    window_study = []
    for radius in sorted(set(config.energy_window_study + (max(loss_max, gain_max),))):
        window_tail = exact_window_tail(radius, radius, eta, state.quantum_energy_j, state.temperature_k)
        _, weights = restrict_lines(responses["fock"].ell, responses["fock"].weights, radius, radius)
        changes = np.arange(-radius, radius + 1)
        window_study.append({"loss_max": radius, "gain_max": radius, "exact_omitted_weight": window_tail["omitted_weight"],
                             "exact_omitted_weight_upper_bound": window_tail["omitted_weight_upper_bound"],
                             "numerical_window_omission": float(np.sum(responses["fock"].weights) - np.sum(weights)),
                             **line_statistics(changes, weights, eta, state.mean_occupation, state.quantum_energy_j)})
    previous_d = working_study[-2]
    d_converged = bool(previous_d["converged"])
    previous_quad = quadratures[config.quadrature_orders[-2]]
    quad_change = float(np.linalg.norm((previous_quad[:n_max+1, :m_max+1] - quadratures[nodes_final][:n_max+1, :m_max+1]) * np.sqrt(probabilities)))
    quads_converged = bool(quad_change < config.matrix_tolerance and quad_study[-1]["converged"] and quad_study[-2]["converged"])
    spectrum_arrays, spectrum = broaden_signed_spectrum(ell, reported["fock"], state.quantum_energy_j, config.fwhm_mev * 1e-3 * constants.elementary_charge)
    checks = {
        "initial_tail": initial_tail < config.initial_tail_tolerance,
        "final_tail": positive_final_tail < config.final_tail_tolerance and all(-config.weight_tolerance < method["final_omitted_weight_residual"] < config.final_tail_tolerance for method in methods.values()),
        "signed_energy_window_tail": tail["omitted_weight_upper_bound"] < config.energy_tail_tolerance,
        "working_dimension_convergence": d_converged,
        "quadrature_convergence": quads_converged,
        "line_weights": all(method["maximum_absolute_line_error"] < config.line_tolerance and method["summed_absolute_line_error_with_exact_tail_bound"] < config.summed_line_tolerance for method in methods.values()),
        "total_weight": all(method["total_weight_error"] < config.weight_tolerance for method in methods.values()),
        "moments": all(method[field] < config.moment_tolerance for method in methods.values() for field in ("mean_error_scaled", "variance_error_scaled", "mean_energy_error_scaled")),
        "correlation": all(method["correlation_max_absolute_error"] < config.correlation_tolerance for method in methods.values()),
        "detailed_balance": all(method["detailed_balance"]["maximum_mixed_residual"] <= 1 for method in methods.values()),
        "coherent_and_zero_energy": all(method[field] < config.coherent_tolerance for method in methods.values() for field in ("coherent_amplitude_error", "coherent_weight_error", "zero_energy_weight_error")),
        "coherent_distinction": all(method["zero_energy_minus_coherent_weight"] > -config.coherent_tolerance for method in methods.values()),
        "energy_origin_and_recoil": bool(ell[gain_max] == 0 and abs(constants.hbar**2 * q**2 / (2 * oscillator.mass_kg) - eta * state.quantum_energy_j) / (state.quantum_energy_j * (abs(eta) if eta else 1)) < config.moment_tolerance),
        "spectrum_area": max(spectrum["area_error_including_window_bound"], *spectrum["component_area_errors"].values()) < config.spectrum_area_tolerance,
    }
    if state.temperature_k == 0:
        checks["T_zero_no_gain"] = all(method["numerical_gain_weight_in_window"] == 0 for method in methods.values())
        checks["T_zero_coherent_equals_zero_energy"] = all(abs(method["zero_energy_minus_coherent_weight"]) < config.coherent_tolerance for method in methods.values())
    net_one = {}
    for change, label in ((1, "loss"), (-1, "gain")):
        exact_one = float(exact_weights(np.array([change]), eta, state.quantum_energy_j, state.temperature_k)[0])
        leading = exact_elastic[f"leading_{label}_one_phonon"]
        numerical_one = {name: float(response.weights[change - int(response.ell[0])]) if response.ell[0] <= change <= response.ell[-1] else 0.0 for name, response in responses.items()}
        net_one[label] = {"exact_net_one_quantum_weight": exact_one, "numerical_net_one_quantum_weight": numerical_one, "leading_one_phonon_weight": leading,
                          "absolute_difference": exact_one - leading,
                          "relative_difference": (exact_one - leading) / exact_one if exact_one > 1e-300 else None}
    exact_gain, exact_gain_bound = exact_window_tail(0, 0, eta, state.quantum_energy_j, state.temperature_k)["gain_tail"], exact_window_tail(0, 0, eta, state.quantum_energy_j, state.temperature_k)["tail_evaluation_error_bound"]
    record = {
        "key": f"{key}_T{state.temperature_k:g}", "operator_key": key, "eta": eta, "xi": xi, "Q_m_inv": q, "temperature_k": state.temperature_k,
        "thermal_parameters": {**parameters, "r": state.r}, "m_max": m_max, "n_max": n_max,
        "working_dimension": d_final, "quadrature_nodes": nodes_final,
        "first_converged_D": next((row["D"] for row in working_study if row["converged"]), None),
        "first_converged_quadrature_nodes": next((row["nodes"] for row in quad_study if row["converged"]), None),
        "loss_max": loss_max, "gain_max": gain_max,
        "exact_initial_omitted_probability": initial_tail, "positive_high_basis_final_tail": positive_final_tail,
        "working_basis_full_column_norm_residual_range": [float(np.min(1 - np.sum(np.abs(matrix[:, :m_max+1])**2, axis=0))), float(np.max(1 - np.sum(np.abs(matrix[:, :m_max+1])**2, axis=0)))],
        "final_column_norm_residuals": {name: response.final_column_norm_residuals.tolist() for name, response in responses.items()},
        "exact_energy_window_tail": tail, "exact_elastic_and_leading_terms": exact_elastic,
        "exact_total_gain_weight": exact_gain, "exact_gain_tail_evaluation_error_bound": exact_gain_bound,
        "net_one_quantum_vs_leading": net_one,
        "methods": methods, "spectrum": spectrum,
        "initial_cutoff_study": initial_study, "final_cutoff_study": final_study,
        "working_dimension_study": working_study, "quadrature_study": quad_study, "energy_window_study": window_study,
        "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL",
    }
    arrays = {
        "ell": ell, "energy_losses_j": ell * state.quantum_energy_j, "exact_weights": exact,
        "fock_weights": reported["fock"], "quadrature_weights": reported["quadrature"],
        "initial_probabilities": probabilities, "times_s": times, "correlation_exact": correlation_exact,
        **{f"correlation_{name}": line_correlation(ell, weights, state.quantum_energy_j, times) for name, weights in reported.items()},
        "full_numerical_ell": responses["fock"].ell, "full_fock_weights": responses["fock"].weights,
        "full_quadrature_weights": responses["quadrature"].weights, **spectrum_arrays,
    }
    return record, arrays


def run_benchmark(config: ThermalConfig, output: Path | None = None) -> dict:
    oscillator = config.oscillator
    q, xi, eta = resolve_coupling(oscillator, None if config.q_inv_angstrom is None else config.q_inv_angstrom * 1e10, config.eta, config.consistency_rtol)
    operators = [(f"eta_{index}", float(value), float(np.sqrt(value)), float(np.sqrt(value) / oscillator.zero_point_displacement_m)) for index, value in enumerate(config.sweep_etas)]
    selected = next((op[0] for op in operators if op[1] == eta and config.q_inv_angstrom is None), None)
    if selected is None:
        selected = "selected_Q"
        operators.append((selected, eta, xi, q))
    temperatures = tuple(sorted(set(config.sweep_temperatures_k + (config.temperature_k,))))
    states = [ThermalOscillatorState(oscillator.quantum_energy_j, temperature) for temperature in temperatures]
    max_initial = max(state.initial_cutoff(config.initial_tail_tolerance) for state in states)
    if max_initial + 16 >= config.working_dimensions[-1]:
        raise ValueError(f"thermal m_max={max_initial} needs a larger working dimension with a guard region")
    if output is not None:
        output = Path(output).resolve()
        if (output / "results.json").is_file() and json.loads((output / "results.json").read_text()).get("benchmark") != BENCHMARK:
            raise ValueError("output directory belongs to another benchmark")
        (output / "arrays").mkdir(parents=True, exist_ok=True)
    constant_data = constants_record()
    value, units, uncertainty = constants.physical_constants["Boltzmann constant"]
    constant_data["values"]["Boltzmann constant"] = {"value": value, "units": units, "standard_uncertainty": uncertainty}
    results = {
        "schema_version": 1, "benchmark": BENCHMARK, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": asdict(config), "constants": constant_data,
        "physical_parameters_si": {"mass_kg": oscillator.mass_kg, "quantum_energy_j": oscillator.quantum_energy_j,
                                   "omega_rad_s": oscillator.angular_frequency_rad_s, "frequency_hz": oscillator.frequency_hz,
                                   "u_zp_m": oscillator.zero_point_displacement_m, "selected_Q_m_inv": q, "selected_eta": eta},
        "conventions": {"units": "SI internally; 1 u=atomic_mass kg, 1 meV=1e-3 e J, 1 inverse angstrom=1e10 inverse metres; omega=2*pi*f",
                        "operator": "exp(-i*Q*u), angular Q; xi=Q*u_zp, eta=xi^2; no extra 2*pi",
                        "energy_loss": "ell*hbar*omega=(n-m)*hbar*omega; positive loss, negative gain, zero elastic",
                        "thermal_average": "sum_m p_m |F_nm|^2, no amplitude averaging and no renormalization",
                        "coherent_amplitude": "sum_m p_m F_mm; its squared magnitude differs from sum_m p_m |F_mm|^2",
                        "moment_scaling": "abs(error)/abs(exact moment) when nonzero; use denominator one when the exact moment vanishes; raw moments, no conditional normalization",
                        "reference": "scaled Bessel/logarithmic evaluation, small-argument positive series; independent positive Poisson convolution tails",
                        "cutoffs": "independent m_max, n_max, working D, signed line window, and quadrature order",
                        "broadening": "normalized Gaussians, per J internally, per meV plotted; no deliberate plotting crop"},
        "optional_propagation": {"status": "NOT_PERFORMED", "reason": "No validated Benchmark 3B channel integrator exists in the inspected repository; the thermal transition sum is not a propagation solution."},
        "selected_operator": selected, "cases": [], "operator_cache": [],
    }
    for operator in operators:
        matrices = {dimension: fock_scattering_matrix(operator[2], dimension) for dimension in config.working_dimensions}
        n_cutoffs = [final_state_cutoff(matrices[config.working_dimensions[-1]], state.probabilities(state.initial_cutoff(config.initial_tail_tolerance)), config.final_tail_tolerance) for state in states]
        if max(n_cutoffs) + 16 >= config.working_dimensions[-1]:
            raise ValueError(f"eta={operator[1]:g} needs a larger final working dimension to resolve its final tail")
        block_max = max(max_initial, max(n_cutoffs)) + 8
        quadratures = {nodes: quadrature_scattering_block(operator[2], block_max, nodes) for nodes in config.quadrature_orders}
        cache_record = {"operator_key": operator[0], "eta": operator[1], "xi": operator[2], "working_dimensions": config.working_dimensions,
                        "quadrature_orders": config.quadrature_orders, "quadrature_block_n_max": block_max, "temperature_cases_reusing_matrices": len(states)}
        if output is not None:
            path = output / "arrays" / f"{operator[0]}_operator.npz"
            np.savez_compressed(path, fock_matrix=matrices[config.working_dimensions[-1]], quadrature_matrix=quadratures[config.quadrature_orders[-1]])
            cache_record["artifact"] = str(path.relative_to(output))
        results["operator_cache"].append(cache_record)
        for state, n_max in zip(states, n_cutoffs):
            case, arrays = _case(operator, state, matrices, quadratures, n_max, config)
            if output is not None:
                path = output / "arrays" / f"{case['key']}.npz"
                np.savez_compressed(path, **arrays)
                case["artifact"] = str(path.relative_to(output))
            results["cases"].append(case)
    results["checks"] = {name: all(case["checks"].get(name, True) for case in results["cases"]) for name in sorted({key for case in results["cases"] for key in case["checks"]})}
    results["prior_benchmark_statuses_preserved"] = {}
    for path in sorted((PROJECT_ROOT / "benchmark_results").glob("*/results.json")):
        if path.parent.name == "thermal_harmonic_response" or (output is not None and path.parent == output):
            continue
        prior = json.loads(path.read_text())
        results["prior_benchmark_statuses_preserved"][path.parent.name] = {"benchmark": prior.get("benchmark"), "status": prior.get("status")}
    results["benchmark_3B_status"] = "NOT_AVAILABLE: no Benchmark 3B implementation or saved result found; no status changed"
    results["status"] = "PASS" if all(results["checks"].values()) else "FAIL"
    return results


def reproduction_command(config, output):
    arguments = [".venv/bin/python", "-m", "benchmarks.thermal_harmonic_response"]
    for field in fields(config):
        value = getattr(config, field.name)
        if value is not None:
            arguments += ["--" + field.name.replace("_", "-"), *map(str, value if isinstance(value, tuple) else (value,))]
    return shlex.join(arguments + ["--output", str(output)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = ThermalConfig()
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        kwargs = {"default": value}
        if isinstance(value, tuple):
            kwargs.update(type=int if field.name in ("working_dimensions", "quadrature_orders", "initial_cutoff_study", "final_cutoff_study", "energy_window_study") else float, nargs="+")
        else:
            kwargs["type"] = int if field.name == "correlation_time_points" else float
        parser.add_argument("--" + field.name.replace("_", "-"), **kwargs)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "benchmark_results/thermal_harmonic_response")
    args = vars(parser.parse_args(argv))
    output = args.pop("output").resolve()
    for field in fields(defaults):
        if isinstance(getattr(defaults, field.name), tuple):
            args[field.name] = tuple(args[field.name])
    try:
        config = ThermalConfig(**args)
        results = run_benchmark(config, output)
    except ValueError as exc:
        parser.error(str(exc))
    results["reproduction_command"] = reproduction_command(config, output)
    from .thermal_outputs import write_figures, write_report
    results["figures"] = write_figures(results, output)
    write_report(results, output)
    (output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(f"{BENCHMARK}: {results['status']}")
    for case in results["cases"]:
        print(f"  T={case['temperature_k']:g} K eta={case['eta']:g}: {case['status']}; m_max={case['m_max']}, n_max={case['n_max']}, ell=[{-case['gain_max']},{case['loss_max']}]; line error={max(row['maximum_absolute_line_error'] for row in case['methods'].values()):.3e}")
    print(f"Optional channel propagation: {results['optional_propagation']['status']}")
    print(f"Results: {output}")
    return 0 if results["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
