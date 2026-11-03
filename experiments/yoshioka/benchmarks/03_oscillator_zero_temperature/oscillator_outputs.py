"""Figures and measured report for the fixed-Q oscillator response benchmark."""

from pathlib import Path

import numpy as np
from scipy.constants import elementary_charge


def _adaptive(case):
    return next(row for row in case["physical_cutoff_study"] if row["adaptive"])


def write_figures(results: dict, output: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = results["cases"]
    selected = next(case for case in cases if case["key"] == results["selected_case"])
    arrays = {}
    for case in cases:
        with np.load(output / case["artifact"]) as data:
            arrays[case["key"]] = {name: data[name] for name in data.files}
    chosen = arrays[selected["key"]]
    figures = []

    def save(fig, name):
        fig.savefig(output / name, dpi=175)
        plt.close(fig)
        figures.append(name)

    cols = min(3, len(cases))
    rows = (len(cases) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.7 * rows), squeeze=False, layout="constrained")
    for ax, case in zip(axes.flat, cases):
        data = arrays[case["key"]]
        last = min(len(data["occupations"]), max(5, case["adaptive_p"] + 3))
        n = data["occupations"][:last]
        for name, style, label in (("exact", "-", "Analytical"), ("fock", "o", "Fock exponential"), ("quadrature", "x", "Coordinate quadrature")):
            weights = data[f"{name}_weights"][:last]
            positive = weights > 0
            ax.semilogy(n[positive], weights[positive], style, markersize=4, markerfacecolor="none", label=label)
        ax.set(title=f"η={case['eta']:g}; adaptive p={case['adaptive_p']}", xlabel="Final oscillator occupation n", ylabel="Oscillator transition weight Pₙ", ylim=(1e-14, 1.5))
        ax.grid(alpha=0.2, which="both")
    for ax in list(axes.flat)[len(cases):]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("T=0 unbroadened transition weights — fixed Q, no renormalization")
    save(fig, "transition_weights.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    elastic = [float(arrays[case["key"]]["fock_weights"][0]) for case in cases]
    inelastic = [float(np.sum(arrays[case["key"]]["fock_weights"][1:case["adaptive_p"]+1])) for case in cases]
    x = np.arange(len(cases))
    axes[0].bar(x, elastic, label="Elastic weight")
    axes[0].bar(x, inelastic, bottom=elastic, label="Retained inelastic weight")
    axes[0].set(xticks=x, xticklabels=[f"{case['eta']:g}" for case in cases], xlabel="η", ylabel="Dimensionless oscillator weight", title="Elastic and inelastic contributions", ylim=(0, 1.06))
    axes[0].legend(fontsize=9)
    with np.load(output / "broadened_spectrum.npz") as spectrum:
        conversion = 1e-3 * elementary_charge
        energy_mev = spectrum["energy_grid_j"] / conversion
        for field, style, label in (("total", "k--", "Total"), ("elastic", "C0-", "Elastic"), ("inelastic", "C1-", "Inelastic")):
            density = spectrum[f"{field}_density_per_j"] * conversion
            positive = density > 0
            axes[1].semilogy(energy_mev[positive], density[positive], style, label=label)
        spectrum_peak = float(np.max(spectrum["total_density_per_j"]) * conversion)
        axes[1].axvspan(float(energy_mev[0]), 0, color="gray", alpha=0.15, label="Broadening tails below zero")
    axes[1].set(xlabel="Energy loss (meV)", ylabel="Oscillator response density (meV$^{-1}$)", title=f"η={selected['eta']:g}, p={selected['adaptive_p']}, FWHM={results['parameters']['fwhm_mev']:g} meV", ylim=(min(1e-14, spectrum_peak * 1e-4), spectrum_peak * 2))
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2, which="both")
    save(fig, "elastic_inelastic_spectrum.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    for index, case in enumerate(cases):
        color, label = f"C{index % 10}", f"η={case['eta']:g}"
        for ax, records, xkey, ykey in (
            (axes[0], case["physical_cutoff_study"], "p", "exact_omitted_probability"),
            (axes[1], case["working_dimension_study"], "D", "adaptive_amplitude_error"),
            (axes[2], case["quadrature_study"], "nodes", "all_cutoffs_amplitude_error"),
        ):
            points = [(row[xkey], row[ykey]) for row in records if row[ykey] is not None and row[ykey] > 0]
            if points:
                xx, yy = zip(*points)
                ax.semilogy(xx, yy, "o-", markersize=4, color=color, label=label)
    axes[0].axhline(results["parameters"]["tail_tolerance"], color="black", linestyle="--", label="Tail target")
    for ax in axes[1:]:
        ax.axhline(results["parameters"]["amplitude_tolerance"], color="black", linestyle="--", label="Amplitude target")
    axes[0].set(xlabel="Physical cutoff p", ylabel="Exact omitted weight Rₚ", title="Physical truncation", ylim=(1e-18, 2))
    axes[1].set(xlabel="Working dimension D", ylabel="Absolute amplitude-vector error", title="Fock basis at fixed adaptive p", ylim=(1e-17, None))
    axes[2].set(xlabel="Gauss–Hermite nodes", ylabel="Absolute amplitude-vector error", title="Quadrature through largest tested p", ylim=(1e-17, None))
    for ax in axes:
        ax.grid(alpha=0.2, which="both")
        ax.legend(fontsize=7)
    fig.suptitle("Independent numerical convergence and physical cutoff convergence; exact zeros omitted on log axes")
    save(fig, "convergence.png")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), layout="constrained")
    fock, quad = chosen["fock_general_block"], chosen["quadrature_general_block"]
    maximum = float(max(np.max(np.abs(fock)), np.max(np.abs(quad))))
    for ax, matrix, title in zip(axes[:2], (fock, quad), ("Fock exponential |Fₘₙ|", "Coordinate quadrature |Fₘₙ|")):
        im = ax.imshow(np.abs(matrix), origin="lower", vmin=0, vmax=maximum, cmap="viridis", interpolation="nearest")
        ax.set(title=title, xlabel="Initial n", ylabel="Final m")
    fig.colorbar(im, ax=list(axes[:2]), label="Matrix-element magnitude")
    im = axes[2].imshow(np.abs(fock - quad), origin="lower", cmap="magma", interpolation="nearest", vmin=0)
    axes[2].set(title="|Fₘₙ(Fock) − Fₘₙ(quad)|", xlabel="Initial n", ylabel="Final m")
    fig.colorbar(im, ax=axes[2], label="Absolute complex-element difference")
    fig.suptitle(f"η={selected['eta']:g}; finite extracted blocks omit transitions outside the block")
    save(fig, "general_matrix_elements.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    time_fraction = chosen["times_s"] / results["physical_parameters_si"]["period_s"]
    exact = chosen["correlation_exact"]
    axes[0].plot(time_fraction, exact.real, "C0-", label="Exact Re C")
    axes[0].plot(time_fraction, exact.imag, "C1-", label="Exact Im C")
    for name, marker in (("fock", "o"), ("quadrature", "x")):
        measured = chosen[f"correlation_{name}_adaptive"]
        axes[0].plot(time_fraction[::16], measured.real[::16], marker, color="C0", markerfacecolor="none", markersize=4, label=f"{name}: Re")
        axes[0].plot(time_fraction[::16], measured.imag[::16], marker, color="C1", markerfacecolor="none", markersize=4, label=f"{name}: Im")
        error = np.abs(measured - exact)
        positive = error > 0
        axes[1].semilogy(time_fraction[positive], error[positive], label=name)
    axes[0].axhline(float(chosen["fock_weights"][0]), color="gray", linestyle="--", label="p=0: elastic-only correlation")
    axes[0].set(xlabel="t / oscillator period", ylabel="Correlation C(t)", title=f"η={selected['eta']:g}, adaptive p={selected['adaptive_p']}")
    axes[1].axhline(results["parameters"]["correlation_tolerance"], color="black", linestyle="--", label="Acceptance target")
    axes[1].set(xlabel="t / oscillator period", ylabel="Absolute error |C_num(t) − C_exact(t)|", title="Spectral sum versus exact correlation")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    save(fig, "correlation.png")
    return figures


def write_report(results: dict, output: Path) -> None:
    parameters, physical = results["parameters"], results["physical_parameters_si"]
    cases = results["cases"]
    selected = next(case for case in cases if case["key"] == results["selected_case"])
    selected_adaptive = _adaptive(selected)
    selected_a0 = complex(*selected["amplitudes_through_p_test_re_im"]["fock"][0])
    maxima = {
        "Fock amplitude-vector error": max(case["fock_amplitude_error_all_cutoffs"] for case in cases),
        "Quadrature amplitude-vector error": max(case["quadrature_amplitude_error_all_cutoffs"] for case in cases),
        "General-block Frobenius error": max(case["general_block_frobenius_error"] for case in cases),
        "Absolute probability error": max(row["methods"][name]["maximum_absolute_probability_error"] for case in cases for row in case["physical_cutoff_study"] for name in ("fock", "quadrature")),
        "Retained-weight error": max(row["methods"][name]["retained_weight_error"] for case in cases for row in case["physical_cutoff_study"] for name in ("fock", "quadrature")),
        "Omitted-weight absolute error": max(row["methods"][name]["direct_omitted_sum_absolute_error"] for case in cases for row in case["physical_cutoff_study"] for name in ("fock", "quadrature")),
        "Adaptive mean error / max(1,eta)": max(_adaptive(case)["methods"][name]["mean_occupation_error_scaled"] for case in cases for name in ("fock", "quadrature")),
        "Adaptive variance error / max(1,eta)": max(_adaptive(case)["methods"][name]["variance_error_scaled"] for case in cases for name in ("fock", "quadrature")),
        "Adaptive correlation max absolute error": max(_adaptive(case)["methods"][name]["correlation_max_absolute_error"] for case in cases for name in ("fock", "quadrature")),
    }
    lines = [
        "# Benchmark 3: exact harmonic-oscillator response at T=0", "",
        f"Overall acceptance: **{results['status']}**. Numerical matrix elements, adaptive physical cutoff/moments, energy assignments/recoil, and spectrum-area checks are recorded separately in `results.json`. Deliberately low-p or under-resolved numerical settings are retained as convergence diagnostics; only independently converged routes are used for final acceptance.", "",
        "## Physical response and units", "",
        r"The benchmark applies $F(Q)=\exp[-iQ\hat u]$ once to the oscillator vacuum. Here $\hat u=u_{zp}(a+a^\dagger)$, $u_{zp}=\sqrt{\hbar/(2M\omega_0)}$, $\xi=Qu_{zp}$ and $\eta=\xi^2$. Q is an angular wavevector; no additional $2\pi$ belongs in the scattering exponential.",
        r"The exact amplitudes are $A_n=e^{-\eta/2}(-i\xi)^n/\sqrt{n!}$ and weights $P_n=e^{-\eta}\eta^n/n!$. Energy losses are $E_n-E_0=n\hbar\omega_0$: the elastic line is at zero and zero-point energy cancels. There are no unbroadened negative-energy transitions at T=0.",
        r"The Debye–Waller exponent is $W_0=\eta/2$. Elastic amplitude $A_0=e^{-W_0}$ and elastic weight $P_0=e^{-2W_0}$ are checked separately using numerical ground-state displacement variance. Mean occupation and variance approach $\eta$, and mean energy loss equals both $\eta\hbar\omega_0$ and $\hbar^2 Q^2/(2M)$.", "",
        f"Illustrative mass = {parameters['mass_amu']:g} u = {physical['mass_kg']:.12e} kg; quantum energy = {parameters['energy_mev']:g} meV = {physical['quantum_energy_j']:.12e} J. Selected η={physical['selected_eta']:.12g}, Q={physical['selected_Q_m_inv']:.12e} m^-1 = {physical['selected_Q_inv_angstrom']:.9f} Å^-1; u_zp={physical['u_zp_m']:.12e} m.",
        f"Angular frequency ω={physical['omega0_rad_per_s']:.12e} rad/s; ordinary frequency f={physical['frequency_hz']:.12e} Hz; period={physical['period_s']:.12e} s. ω=E/hbar=2π f, f=E/h. Inverse ångströms convert to inverse metres by multiplying by 1e10; meV converts to joules by multiplying by 1e-3 times the elementary charge.", "",
        f"For the selected case, numerical W0={selected['debye_waller']['fock']['W0']:.12g}, elastic amplitude A0={selected_a0.real:.12f}{selected_a0.imag:+.3e}i, and elastic weight P0={abs(selected_a0)**2:.12f}. The amplitude and intensity exponents differ by a factor of two.", "",
        "These are dimensionless oscillator transition weights. The delta-function response density has inverse-energy units. Atomic electron-scattering form factors and electron-flux factors are excluded; these are not absolute detector probabilities or cross sections. The mass/energy choice is illustrative, not a complete crystalline Si model.", "",
        "## Constants and independent numerical routes", "",
        f"The installed scipy.constants configuration follows the existing Benchmark 2 convention: {results['constants']['codata_release']}; SciPy {results['constants']['environment']['scipy']}. Actual constants, units, standard uncertainties, source path and source hash are saved in JSON. No ASE constant substitution is used.", "",
        "| Constant | Value | Units |", "|---|---:|---|",
    ]
    for name, item in results["constants"]["values"].items():
        lines.append(f"| {name} | {item['value']:.16e} | {item['units']} |")
    lines.extend(["",
        r"Coordinate quadrature uses $x=u/\ell$ with $\ell=\sqrt{\hbar/(M\omega_0)}=\sqrt2 u_{zp}$. Normalized Hermite factors $h_n=H_n/\sqrt{2^n n!}$ obey $h_0=1$, $h_1=\sqrt2 x$, $h_{n+1}=\sqrt{2/(n+1)}xh_n-\sqrt{n/(n+1)}h_{n-1}$. The overlap is $\sum_j w_j h_m(x_j)h_n(x_j)e^{-i\sqrt2\xi x_j}/\sqrt\pi$, with physicists' Gauss–Hermite weight $e^{-x^2}$. This evaluates normalized oscillator wavefunctions, independently of the Poisson amplitudes. The sqrt(2) is the oscillator-length convention, not a wavevector conversion.",
        "The second route explicitly constructs complex128 ladder matrices and evaluates exp[-i xi (a_D+a_D†)] with scipy.linalg.expm. Neither numerical implementation calls the analytical-amplitude reference. Standard real Hermite oscillator phases are used; complex amplitudes are compared directly with no arbitrary phase alignment.",
        f"Working dimensions tested: {parameters['working_dimensions']}; quadrature orders: {parameters['quadrature_orders']}. Successive high settings are checked for convergence, including the general block m,n=0..{parameters['block_n_max']}. The opposite-Q identities are tested for vacuum amplitudes and general elements. The finite block extracted from the converged infinite-space approximation is not required to be unitary: transitions beyond the block are omitted.",
        "Implementation references: [Gauss–Hermite quadrature](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_hermite.html), [matrix exponential](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html), [Poisson survival function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html).", "",
        "## Measured errors and sufficient cutoffs", "",
        "Amplitude-vector error is sqrt(sum_{n=0}^p |A_num-A_exact|²), an absolute norm. The block error is an absolute Frobenius norm. Tiny probabilities and zero quantities use absolute errors. The table gives the first tested settings meeting the amplitude and block targets; adaptive/all-p denotes the adaptive physical cutoff versus every requested physical cutoff.", "",
        "| η | Adaptive p | Exact omitted weight | D: adaptive / all-p | Nodes: adaptive / all-p |", "|---:|---:|---:|---:|---:|",
    ])
    for case in cases:
        lines.append(f"| {case['eta']:g} | {case['adaptive_p']} | {_adaptive(case)['exact_omitted_probability']:.6e} | {case['first_converged_D_for_adaptive_p']} / {case['first_converged_D_for_all_cutoffs']} | {case['first_converged_nodes_for_adaptive_p']} / {case['first_converged_nodes_for_all_cutoffs']} |")
    lines.extend(["", "Maximum measured errors across the sweep and both numerical routes where applicable:", "", "| Quantity | Measured maximum |", "|---|---:|"])
    for label, value in maxima.items():
        lines.append(f"| {label} | {value:.6e} |")
    lines.extend(["",
        f"Final evaluations use D={max(parameters['working_dimensions'])} and {max(parameters['quadrature_orders'])} quadrature nodes. Limits are amplitude {parameters['amplitude_tolerance']:.1e}, probability {parameters['probability_tolerance']:.1e}, block {parameters['block_tolerance']:.1e}, retained-weight {parameters['weight_tolerance']:.1e}, tail {parameters['tail_tolerance']:.1e}, scaled mean/variance {parameters['moment_tolerance']:.1e}, and correlation {parameters['correlation_tolerance']:.1e}. None is silently relaxed. The adaptive and all-p studies can require different quadrature orders; their measured settings appear above. Once roundoff dominates, increasing order need not improve errors monotonically; the full measured sequences are saved.", "",
        "## Physical cutoff versus working dimension", "",
        f"Physical cutoffs {parameters['physical_cutoffs']} plus each adaptive cutoff are evaluated with converged numerical matrix elements. Exact omitted weight is scipy.stats.poisson.sf(p,eta), not subtraction from a nearly unit cumulative sum. For p≥1 and η>0 the omitted inelastic fraction is R_p/[-expm1(-eta)]. At η=0 it is recorded as zero by convention because there is no inelastic weight; for p=0 the requested inelastic-fraction definition is marked inapplicable.",
        "Neither analytical nor numerical retained weights are renormalized. Raw first/second moments are saved; second_raw-first_raw² is a variance estimator accepted only in the normalized limit. Underconverged low-p values are labelled deliberately underconverged. Numerical omitted-weight sums beyond p are also reported through a stated evaluation ceiling, together with the exact residual beyond that ceiling; tiny tails are assessed with absolute errors. A norm deficit may be a tiny negative number at roundoff and is not clipped or interpreted as negative physical weight.", "",
    ])
    diagnostic = results["premature_truncation_diagnostic"]
    lines.extend([
        f"A deliberate η={diagnostic['eta']:g}, p={diagnostic['physical_p']}, D=p+1={diagnostic['premature_D']} example gives finite-state norm {diagnostic['premature_full_state_weight']:.12f} but amplitude-vector error {diagnostic['premature_vacuum_vector_error']:.6f}. The correct retained weight is {diagnostic['correct_retained_weight']:.12f}. The block difference between P_p exp(-iQu) P_p and exp(-iQ P_p u P_p) is {diagnostic['projected_infinite_vs_exponentiated_truncation_block_error']:.6f}. The latter's unitarity defect is only {diagnostic['finite_operator_unitarity_defect']:.3e}: its unit norm does not establish correct low-occupation coefficients.", "",
        "## Correlation and broadened spectrum", "",
        r"Over one full oscillator period, $C_{exact}(t)=\exp[\eta(e^{-i\omega t}-1)]$ is compared with $\sum_{n=0}^p|A_n|^2 e^{-in\omega t}$. At finite p, C(0) equals the retained weight, with exact deficit R_p; no correction is applied. The pure physical-cutoff correlation error is bounded by R_p because each omitted phase has modulus one. No numerical Fourier transform or window-leakage interpretation is used.",
        f"For the selected case, adaptive p={selected['adaptive_p']}; numerical C(0)={selected_adaptive['methods']['fock']['correlation_c0_real']:.15f}; exact omitted weight={selected_adaptive['exact_omitted_probability']:.6e}.",
        f"The primary benchmark uses unbroadened transition weights. Visualization convolves the retained lines with normalized Gaussian kernels, FWHM={parameters['fwhm_mev']:g} meV and sigma=FWHM/[2 sqrt(2 ln 2)]. The grid extends eight sigmas beyond both endpoint lines and samples at at most FWHM/16. Density is J^-1 internally and converts to meV^-1 by multiplying by {1e-3*elementary_charge:.12e} J/meV.",
        f"The plotted area is {results['spectrum']['integrated_broadened_weight']:.15f} versus retained line weight {results['spectrum']['retained_unbroadened_weight']:.15f}, absolute error {results['spectrum']['absolute_area_error']:.3e}; elastic and inelastic areas are independently checked. No cropped or truncated spectrum is normalized to one. Density below zero is a Gaussian tail, not a T=0 energy-gain transition.", "",
        "## Artifacts, reproduction, and limits", "",
        "`results.json` contains all physical/numerical inputs, constants and sources, unaligned complex amplitudes, errors, numerical convergence studies, physical tails, moments, and separate acceptance checks. `arrays/*.npz` stores line energies, exact and numerical amplitudes and weights, general complex blocks and correlations. `broadened_spectrum.npz` stores the SI energy grid and separate elastic/inelastic density arrays. Figures retain the original weights.", "",
        "```bash", "cd /home/zuxian/Documents/PhD/Yoshioka", ".venv/bin/python -m unittest discover -s tests -v", ".venv/bin/python -m benchmarks.harmonic_response", "```", "",
        "The existing NumPy, SciPy and Matplotlib dependencies suffice. The pinned `requirements-benchmark2.txt` environment reproduces the recorded constants and package versions. For a fresh environment use `python3 -m venv .venv` followed by `.venv/bin/python -m pip install -r requirements-benchmark2.txt`. The full configured reproduction command is:", "", "```bash", results["reproduction_command"], "```", "",
        f"Previous benchmark statuses, read without modifying their results: {results['prior_benchmark_statuses_preserved']}. Benchmark 3 does not resolve Benchmark 2's existing elastic convergence failure.", "",
        "Passing validates oscillator scattering matrix elements and fixed-Q harmonic spectral weights. Poisson weights apply to this single phase-operator action on the vacuum. General finite-thickness electron–atom scattering, repeated electron scattering, detector integration and momentum-dependent prefactors require separate treatment. This benchmark does not validate a general coupled electron–phonon multislice calculation.", "",
    ])
    for name in results.get("figures", []):
        lines.extend([f"![{name.removesuffix('.png').replace('_', ' ')}]({name})", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
