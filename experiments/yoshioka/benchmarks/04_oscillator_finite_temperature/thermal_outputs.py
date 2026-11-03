"""Measured scientific report and figures for Benchmark 4."""

from pathlib import Path

import numpy as np
from scipy.constants import elementary_charge


def worst_errors(results):
    methods = [method for case in results["cases"] for method in case["methods"].values()]
    keys = ("maximum_absolute_line_error", "summed_absolute_line_error_with_exact_tail_bound", "total_weight_error",
            "mean_error_scaled", "variance_error_scaled", "correlation_max_absolute_error", "coherent_amplitude_error",
            "coherent_weight_error", "zero_energy_weight_error")
    errors = {key: max(method[key] for method in methods) for key in keys}
    errors["detailed_balance_maximum_mixed_residual"] = max(method["detailed_balance"]["maximum_mixed_residual"] for method in methods)
    errors["detailed_balance_maximum_absolute_residual"] = max(method["detailed_balance"]["maximum_absolute_residual"] for method in methods)
    errors["maximum_resolved_log_ratio_error"] = max((method["detailed_balance"]["maximum_resolved_log_ratio_error"] for method in methods if method["detailed_balance"]["maximum_resolved_log_ratio_error"] is not None), default=None)
    errors["spectrum_area_error_including_window_bound"] = max(case["spectrum"]["area_error_including_window_bound"] for case in results["cases"])
    return errors


def write_figures(results: dict, output: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = results["cases"]
    selected = [case for case in cases if case["operator_key"] == results["selected_operator"]]
    arrays = {}
    for case in cases:
        with np.load(output / case["artifact"]) as data:
            arrays[case["key"]] = {name: data[name] for name in data.files}
    figures = []

    def save(fig, name):
        fig.savefig(output / name, dpi=175)
        plt.close(fig)
        figures.append(name)

    rows = (len(selected) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(14, 4 * rows), squeeze=False, layout="constrained")
    for ax, case in zip(axes.flat, selected):
        data = arrays[case["key"]]
        for method, style in (("exact", "-"), ("fock", "o"), ("quadrature", "x")):
            values = data[f"{method}_weights"]
            valid = values > 0
            ax.semilogy(data["energy_losses_j"][valid] / (1e-3 * elementary_charge), values[valid], style, markersize=4, markerfacecolor="none", label=method)
        ax.axvline(0, color="gray", alpha=0.4)
        positive = data["exact_weights"][data["exact_weights"] > 0]
        floor = min(1e-14, max(1e-40, float(np.min(positive)) / 10))
        ax.set(title=f"T={case['temperature_k']:g} K; gain={case['exact_total_gain_weight']:.2e}", xlabel="Energy loss (meV); gain < 0", ylabel="Dimensionless line weight", ylim=(floor, 1.5))
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(selected):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"Unbroadened thermal oscillator response, η={selected[0]['eta']:g}; no renormalization")
    save(fig, "signed_transition_weights.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for operator in results["operator_cache"]:
        group = [case for case in cases if case["operator_key"] == operator["operator_key"]]
        temperatures = [case["temperature_k"] for case in group]
        residuals = [max(method["detailed_balance"]["maximum_mixed_residual"] for method in case["methods"].values()) for case in group]
        valid = np.array(residuals) > 0
        axes[0].semilogy(np.array(temperatures)[valid], np.array(residuals)[valid], "o-", label=f"η={operator['eta']:g}")
        resolved = [(case["temperature_k"], method["detailed_balance"]["maximum_resolved_log_ratio_error"]) for case in group for method in (case["methods"]["fock"],) if method["detailed_balance"]["maximum_resolved_log_ratio_error"] is not None and method["detailed_balance"]["maximum_resolved_log_ratio_error"] > 0]
        if resolved:
            xx, yy = zip(*resolved)
            axes[1].semilogy(xx, yy, "o-", label=f"η={operator['eta']:g}")
    axes[0].axhline(1, color="black", linestyle="--", label="Acceptance limit")
    axes[0].set(xlabel="Temperature (K)", ylabel="Maximum mixed residual", title="All pairs: absolute + relative tolerance")
    axes[1].set(xlabel="Temperature (K)", ylabel="|ln(W₋ℓ/Wℓ) + ℓx|", title="Well-resolved pairs only; Fock sum")
    for ax in axes:
        ax.grid(alpha=0.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.suptitle("Detailed balance checked on independently calculated gain and loss weights")
    save(fig, "detailed_balance.png")

    operators = results["operator_cache"]
    cols = min(3, len(operators))
    rows = (len(operators) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.6 * rows), squeeze=False, layout="constrained")
    for ax, operator in zip(axes.flat, operators):
        group = [case for case in cases if case["operator_key"] == operator["operator_key"]]
        temperatures = [case["temperature_k"] for case in group]
        for key, label, style in (("zero_energy_weight", "Full zero-energy weight W₀", "C0-"), ("coherent_weight", "|Tr(ρF)|²", "C1--")):
            ax.plot(temperatures, [case["exact_elastic_and_leading_terms"][key] for case in group], style, label=label)
            ax.plot(temperatures, [case["methods"]["fock"][key] for case in group], "o", color=style[:2], markerfacecolor="none", markersize=4)
        ax.set(title=f"η={operator['eta']:g}", xlabel="Temperature (K)", ylabel="Dimensionless weight")
        if operator["eta"] == 0:
            ax.set_ylim(0.99, 1.01)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(operators):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Thermal coherent weight and complete zero-energy-transfer weight")
    save(fig, "coherent_and_zero_energy.png")

    diagnostic = max(cases, key=lambda case: (case["temperature_k"], case["eta"]))
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8), layout="constrained")
    specifications = [
        ("initial_cutoff_study", "m_max", "exact_initial_tail", "Initial cutoff m_max", "Exact omitted initial probability"),
        ("final_cutoff_study", "n_max", "positive_high_basis_final_tail", "Final cutoff n_max", "Positive omitted final weight"),
        ("working_dimension_study", "D", "weighted_amplitude_change", "Working dimension D", "Weighted complex amplitude change"),
        ("energy_window_study", "loss_max", "exact_omitted_weight_upper_bound", "Symmetric signed-window radius", "Exact omitted line-weight bound"),
        ("quadrature_study", "nodes", "weighted_amplitude_difference_vs_fock", "Gauss–Hermite nodes", "Weighted amplitude difference vs Fock"),
    ]
    for ax, (study, xkey, ykey, xlabel, ylabel) in zip(axes.flat, specifications):
        points = [(row[xkey], row[ykey]) for row in diagnostic[study] if row.get(ykey) is not None and row[ykey] > 0]
        if points:
            xx, yy = zip(*points)
            ax.semilogy(xx, yy, "o-")
        target_key = {"initial_cutoff_study": "initial_tail_tolerance", "final_cutoff_study": "final_tail_tolerance", "working_dimension_study": "matrix_tolerance", "energy_window_study": "energy_tail_tolerance", "quadrature_study": "matrix_tolerance"}[study]
        ax.axhline(results["parameters"][target_key], color="black", linestyle="--", label="Target")
        if study == "working_dimension_study":
            ax.set_ylim(1e-17, max(1e-9, max((point[1] for point in points), default=0) * 10))
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.2, which="both")
        ax.legend(fontsize=8)
    data = arrays[diagnostic["key"]]
    fraction = data["times_s"] * results["physical_parameters_si"]["frequency_hz"]
    for method in ("fock", "quadrature"):
        errors = np.abs(data[f"correlation_{method}"] - data["correlation_exact"])
        valid = errors > 0
        axes.flat[5].semilogy(fraction[valid], errors[valid], label=method)
    axes.flat[5].axhline(results["parameters"]["correlation_tolerance"], color="black", linestyle="--", label="Correlation target")
    axes.flat[5].set(xlabel="Time / oscillator period", ylabel="Absolute correlation error")
    axes.flat[5].legend(fontsize=8)
    axes.flat[5].grid(alpha=0.2)
    fig.suptitle(f"Independent convergence studies, η={diagnostic['eta']:g}, T={diagnostic['temperature_k']:g} K; exact zeros omitted on log axes")
    save(fig, "cutoff_convergence.png")

    fig, axes = plt.subplots(len(selected), 1, figsize=(12, max(3, 2.1 * len(selected))), squeeze=False, layout="constrained")
    for ax, case in zip(axes.flat, selected):
        data = arrays[case["key"]]
        energy = data["energy_grid_j"] / (1e-3 * elementary_charge)
        for name, style in (("total", "k-"), ("elastic", "C0--"), ("loss", "C1-"), ("gain", "C2-")):
            density = data[f"{name}_density_per_j"] * (1e-3 * elementary_charge)
            valid = density > 0
            ax.semilogy(energy[valid], density[valid], style, label=name)
        peak = float(np.max(data["total_density_per_j"]) * (1e-3 * elementary_charge))
        ax.set(title=f"T={case['temperature_k']:g} K; 1 − retained line weight = {1-case['spectrum']['retained_line_weight']:.2e}", xlabel="Energy loss (meV)", ylabel="Response (meV$^{-1}$)", ylim=(min(1e-14, peak * 1e-5), peak * 2))
        ax.axvline(0, color="gray", alpha=0.4)
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8, ncol=4)
    fig.suptitle(f"Gaussian-broadened response: η={selected[0]['eta']:g}, FWHM={results['parameters']['fwhm_mev']:g} meV; finite window, no renormalization")
    save(fig, "broadened_spectra.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    group = sorted((case for case in cases if case["temperature_k"] == results["parameters"]["temperature_k"] and case["eta"] > 0), key=lambda case: case["eta"])
    for ax, label in zip(axes, ("loss", "gain")):
        eta_values = [case["eta"] for case in group]
        exact = [case["net_one_quantum_vs_leading"][label]["exact_net_one_quantum_weight"] for case in group]
        leading = [case["net_one_quantum_vs_leading"][label]["leading_one_phonon_weight"] for case in group]
        numerical = [case["net_one_quantum_vs_leading"][label]["numerical_net_one_quantum_weight"]["fock"] for case in group]
        if any(weight > 0 for weight in exact):
            ax.loglog(eta_values, exact, "-", label="Exact net one-quantum line")
            ax.loglog(eta_values, numerical, "o", markerfacecolor="none", label="Numerical transition sum")
            ax.loglog(eta_values, leading, "--", label="Leading one-phonon term")
        else:
            ax.text(0.5, 0.5, f"{label.capitalize()} weights are zero or below numerical range", ha="center", transform=ax.transAxes)
        ax.set(xlabel="η", ylabel="Dimensionless weight", title=f"Net {label} line versus leading term")
        ax.grid(alpha=0.2, which="both")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.suptitle(f"Weak-coupling limit and higher-order combinations, T={results['parameters']['temperature_k']:g} K")
    save(fig, "net_one_quantum_vs_leading.png")
    return figures


def write_report(results: dict, output: Path) -> None:
    config, physical = results["parameters"], results["physical_parameters_si"]
    selected = [case for case in results["cases"] if case["operator_key"] == results["selected_operator"]]
    errors = worst_errors(results)
    maximum_initial_tail = max(case["exact_initial_omitted_probability"] for case in results["cases"])
    maximum_positive_final_tail = max(case["positive_high_basis_final_tail"] for case in results["cases"])
    final_residuals = [method["final_omitted_weight_residual"] for case in results["cases"] for method in case["methods"].values()]
    maximum_window_tail = max(case["exact_energy_window_tail"]["omitted_weight_upper_bound"] for case in results["cases"])
    lines = [
        "# Benchmark 4: finite-temperature harmonic-oscillator response", "",
        f"Overall acceptance: **{results['status']}** across {len(results['cases'])} cases. This is a thermal Fock-state transition sum using the validated Benchmark 3A/T=0 oscillator matrix-element routines. It is not a numerical solution of Yoshioka propagation equations.", "",
        "## Definitions, conventions, and constants", "",
        r"$F(Q)=e^{-iQ\hat u}$, $\hat u=u_{zp}(a+a^\dagger)$, $u_{zp}=\sqrt{\hbar/(2M\omega_0)}$, $\xi=Qu_{zp}$, $\eta=\xi^2$. Q is angular wavevector in m^-1 with no additional $2\pi$. The illustrative mass and oscillator quantum are not a full crystalline Si model; large η values are mathematical stress tests.",
        f"Mass={config['mass_amu']:g} u = {physical['mass_kg']:.12e} kg; quantum={config['energy_mev']:g} meV = {physical['quantum_energy_j']:.12e} J. ω={physical['omega_rad_s']:.12e} rad/s; f={physical['frequency_hz']:.12e} Hz; u_zp={physical['u_zp_m']:.12e} m. Selected η={physical['selected_eta']:g}, Q={physical['selected_Q_m_inv']:.12e} m^-1.",
        "Conversions: 1 u=atomic_mass kg; 1 meV=1e-3 e J; inverse ångströms multiply by 1e10 to give inverse metres. ω=E/hbar=2π f, f=E/h. SI units are used internally; plotted Gaussian densities are per meV, obtained by multiplying per-joule densities by 1e-3 e J/meV.",
        f"Constants follow the installed scipy.constants configuration used in earlier benchmarks: {results['constants']['codata_release']}. Package versions: {results['constants']['environment']}. The complete source path, SHA256, units, and standard uncertainties are in results.json.", "",
        "| Constant | SI value | Units |", "|---|---:|---|",
    ]
    for name, constant in results["constants"]["values"].items():
        lines.append(f"| {name} | {constant['value']:.16e} | {constant['units']} |")
    lines.extend(["", "## Independent numerical and analytical calculations", "",
        r"For T>0, $x=\hbar\omega_0/(k_BT)$, $r=e^{-x}$, $p_m=(1-r)e^{-mx}$ and $\bar n=r/[-\mathrm{expm1}(-x)]$. The normalization factor uses -expm1(-x), avoiding high-temperature cancellation and low-temperature exp(x) overflow. At T=0 the initial state is exactly the vacuum. At positive T, m=1 is retained even if its probability is below the initial-tail target, so representable tiny gain is measured.",
        r"The numerical calculation averages probabilities: $W_\ell=\sum_{n-m=\ell}p_m|F_{nm}|^2$. Both gain and loss are obtained independently; detailed balance is only checked afterward. Energy loss is $(n-m)\hbar\omega_0$: positive for electron loss, negative for gain, zero for the complete zero-energy line. Net occupation change ℓ is distinct from final occupation n. No zero-point energy is assigned to transition energies.",
        "Fock matrices are complex128 scipy.linalg.expm results from yoshioka.harmonic_oscillator. The independent coordinate overlap uses the same validated normalized Hermite recurrence and Gauss–Hermite implementation as Benchmark 3, with x_coord=u/ell_osc and ell_osc=sqrt(2) u_zp. Matrices are cached at fixed eta and reused across temperatures. Complex matrix differences are not phase aligned.",
        r"The independent exact reference has $\lambda_+=\eta(\bar n+1)$ and $\lambda_-=\eta\bar n$, and uses the Skellam modified-Bessel formula. Scaled Bessel functions and log weights avoid overflow. At small argument or Bessel underflow, the leading power is removed and a positive converged series is summed in log space. η=0 is an exact delta line; T=0 is an explicit one-sided Poisson branch. Zero parameters are never passed into a strictly-positive Skellam routine.",
        "Exact signed tails use the nonnegative Poisson convolution P(X−Y>L)=sum_y Pois(y;lambda_minus) SF_Pois(L+y;lambda_plus), with an explicit bound on the omitted outer series. This avoids subtracting a nearly unit CDF. [Scaled Bessel documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ive.html), [Skellam definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skellam.html), [Poisson survival function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html).", "",
        "## Independent cutoffs and measured errors", "",
        f"Initial omitted probability r^(m_max+1) must be below {config['initial_tail_tolerance']:.1e}. Final-state cutoff selection uses a positive tail sum in the largest working basis; the requested sum_m p_m[1−sum_n<=n_max |F_nm|²] is also saved without clipping small negative roundoff. Final-tail target={config['final_tail_tolerance']:.1e}; exact signed-window omitted-weight bound target={config['energy_tail_tolerance']:.1e}.",
        f"Working dimensions {config['working_dimensions']} and quadrature orders {config['quadrature_orders']} are varied independently of both physical cutoffs. Successive largest settings must agree and the final numerical/reference line comparisons must pass. Final evaluations use D={config['working_dimensions'][-1]} and {config['quadrature_orders'][-1]} quadrature nodes. Finite-matrix unitarity alone is not used to establish convergence.",
        f"Largest measured initial omitted probability={maximum_initial_tail:.6e}; largest positive high-basis final tail={maximum_positive_final_tail:.6e}; signed final norm-deficit residual range=[{min(final_residuals):.6e}, {max(final_residuals):.6e}]; largest exact signed-window omitted-weight bound={maximum_window_tail:.6e}.",
        "Neither initial distributions nor retained line weights are renormalized. Raw retained mean and second moment are used; second_raw−mean_raw² is accepted as a variance only in the normalized limit. Mean and variance errors are divided by the magnitude of their respective exact values when nonzero; a denominator of one is used when the exact value vanishes. Low cutoff studies are labelled deliberately underconverged rather than implementation failures.", "",
        "| η | T (K) | m_max | n_max | Signed ℓ window | First adequate D | First adequate nodes |", "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for case in results["cases"]:
        lines.append(f"| {case['eta']:g} | {case['temperature_k']:g} | {case['m_max']} | {case['n_max']} | [{-case['gain_max']}, {case['loss_max']}] | {case['first_converged_D']} | {case['first_converged_quadrature_nodes']} |")
    lines.extend(["", "Maximum measured errors across both independent numerical routes:", "", "| Quantity | Maximum |", "|---|---:|"])
    for name, value in errors.items():
        lines.append(f"| {name} | {'not applicable' if value is None else f'{value:.6e}'} |")
    lines.extend(["", "Separate acceptance checks:", "", "| Criterion | Status |", "|---|---|"])
    for name, passed in results["checks"].items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines.extend(["",
        f"Fixed targets: maximum line error {config['line_tolerance']:.1e}, summed absolute error including exact omitted-window bound {config['summed_line_tolerance']:.1e}, total weight {config['weight_tolerance']:.1e}, scaled moments {config['moment_tolerance']:.1e}, correlation {config['correlation_tolerance']:.1e}, coherent/zero-energy errors {config['coherent_tolerance']:.1e}, broadened area {config['spectrum_area_tolerance']:.1e}. Numerical matrix refinement target={config['matrix_tolerance']:.1e}. No tolerance is relaxed in an underresolved run.", "",
        "## Detailed balance, thermal coherence, and net one-quantum lines", "",
        f"Every positive-ℓ pair in the full retained transition sum is tested with residual abs(W_-ell−exp(-ell*x)W_ell)/[atol+rtol*max(W_-ell,exp(-ell*x)W_ell)], requiring ≤1. atol={config['detailed_balance_atol']:.1e}; rtol={config['detailed_balance_rtol']:.1e}. No gain weights are constructed from this identity. Log-ratio errors are reported only when both weights exceed max(1e-12,100 times the combined physical cutoff bound,cutoff_bound/rtol). This requires that physical cutoff uncertainty be below the requested relative error scale. Negligible, cutoff-limited, and underflowed pairs keep all absolute/mixed acceptance checks without unstable ratios.",
        r"Thermal coherent amplitude is $\mathrm{Tr}(\rho F)=e^{-\eta(\bar n+1/2)}$ and its squared magnitude is $e^{-\eta(2\bar n+1)}$. The complete zero-energy weight is instead $W_0=\sum_m p_m|F_{mm}|^2$. Their equality at T=0 and generally positive finite-T difference are checked. For changes smaller than floating-point resolution, the stable analytical difference and the absolute numerical errors are recorded instead of demanding a resolved relative difference.", "",
        f"Selected η={physical['selected_eta']:g}:", "", "| T (K) | Numerical total gain | Exact total gain | Squared coherent amplitude | Full W₀ | C(0) deficit |", "|---:|---:|---:|---:|---:|---:|",
    ])
    for case in selected:
        lines.append(f"| {case['temperature_k']:g} | {case['methods']['fock']['numerical_gain_weight_in_window']:.9e} | {case['exact_total_gain_weight']:.9e} | {case['methods']['fock']['coherent_weight']:.9e} | {case['methods']['fock']['zero_energy_weight']:.9e} | {case['methods']['fock']['correlation_c0_deficit']:.3e} |")
    lines.extend(["",
        r"Exact net ℓ=±1 lines include higher-order creation/annihilation combinations. Leading one-phonon terms are exp[-η(2nbar+1)]η(nbar+1) and exp[-η(2nbar+1)]η nbar. The following relative corrections demonstrate the weak-η limit; these terms are not identified with exact net-energy lines at arbitrary η. Numerical net lines, exact lines, and leading terms are saved separately.", "",
        f"At T={config['temperature_k']:g} K:", "", "| η | Exact W+1 | Leading loss term | Relative correction | Exact W−1 | Leading gain term |", "|---:|---:|---:|---:|---:|---:|",
    ])
    for case in results["cases"]:
        if case["temperature_k"] != config["temperature_k"]:
            continue
        loss, gain = case["net_one_quantum_vs_leading"]["loss"], case["net_one_quantum_vs_leading"]["gain"]
        correction = "n/a" if loss["relative_difference"] is None else f"{loss['relative_difference']:.6e}"
        lines.append(f"| {case['eta']:g} | {loss['exact_net_one_quantum_weight']:.6e} | {loss['leading_one_phonon_weight']:.6e} | {correction} | {gain['exact_net_one_quantum_weight']:.6e} | {gain['leading_one_phonon_weight']:.6e} |")
    lines.extend(["", "## Correlation and plotting checks", "",
        r"One full oscillator period, endpoints included, is tested against $C_T(t)=\exp[\lambda_+(e^{-i\omega t}-1)+\lambda_-(e^{+i\omega t}-1)]$. The numerical spectral sum uses exp(-iℓωt), including the signed retained energy window. C(0) equals retained weight; each finite-cutoff deficit is reported. No Fourier transform, window leakage, or energy-axis shift is used.",
        f"Gaussian FWHM={config['fwhm_mev']:g} meV is configurable. The broadening grid extends eight sigmas beyond both endpoint lines with spacing ≤FWHM/16. Each Gaussian is normalized analytically; the finite numerical area is checked against the retained unbroadened weight. A two-sided eight-sigma Gaussian omission bound is included. The signed line-window omission is recorded separately. There is no deliberate plotting crop and no spectrum rescaling to unit area. Density units are J^-1 internally and meV^-1 in figures.",
        "Physical gain lines have negative signed energy at positive T. Gaussian tails from the zero-energy/loss peaks can also enter negative energies; broadening is not used for detailed-balance or physical-gain acceptance. The unbroadened weights are primary, and the very small 10 K gain is quantified in the table even when it is below the broadened-plot display floor.", "",
        "## Artifacts, reproduction, and scope", "",
        "Reusable routines: yoshioka/thermal_response.py. Independent reference: benchmarks/thermal_reference.py. results.json contains constants, inputs, all configured cases, raw signed-tail residuals, independent convergence sequences, full detailed-balance pair diagnostics, errors, statuses, and optional-propagation status. arrays/*_operator.npz stores cached final Fock and coordinate matrices; case NPZ files store signed line energies/weights, initial probabilities, correlations, and separate elastic/loss/gain Gaussian densities. No pre-existing benchmark outputs are overwritten.", "",
        "```bash", "cd /home/zuxian/Documents/PhD/Yoshioka", ".venv/bin/python -m unittest discover -s tests -v", ".venv/bin/python -m benchmarks.thermal_harmonic_response", "```", "",
        "The existing NumPy/SciPy/Matplotlib dependencies suffice; no new dependencies were installed. The recorded environment is pinned in requirements-benchmark2.txt. Full configured reproduction command:", "", "```bash", results["reproduction_command"], "```", "",
        f"Optional channel propagation: **{results['optional_propagation']['status']}**. {results['optional_propagation']['reason']}",
        f"Separate Benchmark 3B status: {results['benchmark_3B_status']}. Previous benchmark statuses read without modifying them: {results['prior_benchmark_statuses_preserved']}.",
        "Benchmark 2's physical convergence failure remains unresolved. Passing this benchmark validates finite-temperature harmonic oscillator response and thermal probability averaging. Atomic form factors, flux factors, repeated electron scattering, detector integration, general channel propagation, and finite-temperature material EELS accuracy require separate validation. These dimensionless oscillator weights are not absolute detector probabilities or cross sections.", "",
    ])
    for name in results.get("figures", []):
        lines.extend([f"![{name.removesuffix('.png').replace('_', ' ')}]({name})", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
