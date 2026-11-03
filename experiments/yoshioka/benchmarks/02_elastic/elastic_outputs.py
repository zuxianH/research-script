"""Measured figures and report for the elastic benchmark; no simulation code."""

from pathlib import Path

import numpy as np


def _primary(results):
    return [case for case in results["cases"] if case["primary_comparison"]]


def write_figures(results: dict, output: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import PercentFormatter

    figures = []

    def save(fig, name):
        fig.savefig(output / name, dpi=170)
        plt.close(fig)
        figures.append(name)

    structure = results["structure"]
    positions = np.array(structure["positions_angstrom"])
    lx, ly, lz = np.diag(structure["cell_angstrom"])
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    points = ax.scatter(positions[:, 0], positions[:, 1], c=positions[:, 2], s=190, cmap="viridis", vmin=0, vmax=lz, edgecolors="black")
    for index, (x, y, _) in enumerate(positions):
        ax.annotate(f"Si {index + 1}", (x, y), xytext=(7, 10), textcoords="offset points")
    ax.set(xlim=(0, lx), ylim=(0, ly), xlabel="Supplied x (Å)", ylabel="Supplied y (Å)", title="Si [110] projected columns — unchanged coordinates", aspect="equal")
    fig.colorbar(points, ax=ax, label="Supplied z (Å), within one repeat")
    save(fig, "projected_atoms.png")

    initial = next(item for item in results["shared_configurations"] if item["grid"][0] == results["parameters"]["initial_grid"] and item["slices_per_repeat"] == results["parameters"]["initial_slices"])
    initial_label = f"{results['parameters']['initial_grid']}² / {results['parameters']['initial_slices']} slices per repeat"
    with np.load(output / initial["artifact"]) as shared:
        potential = shared["potential_v_angstrom_sxy"]
        index = int(np.argmax(np.max(potential, axis=(1, 2))))
        thickness = float(shared["unit_slice_thickness_angstrom"][index])
        nx, ny = potential.shape[-2:]
        # One wrapped display pixel closes the periodic cell at the upper face.
        # Samples stay at j*dx; no physical array is shifted or interpolated.
        real_extent = (-lx / (2 * nx), lx + lx / (2 * nx), -ly / (2 * ny), ly + ly / (2 * ny))
        fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        im = ax.imshow(np.pad(potential[index].T, ((0, 1), (0, 1)), mode="wrap"), origin="lower", extent=real_extent, interpolation="nearest", cmap="magma")
        ax.set(xlabel="x (Å)", ylabel="y (Å)", xlim=(0, lx), ylim=(0, ly), title=f"Finite Lobato projection, slice {index}; Δz={thickness:.6f} Å")
        fig.colorbar(im, ax=ax, label="Integrated electrostatic potential (V Å)")
        save(fig, "potential_slice.png")

    primary = _primary(results)
    data = []
    for case in primary:
        with np.load(output / case["artifact"]) as arrays:
            data.append({name: arrays[name] for name in arrays.files})
    count = len(primary)
    fig, axes = plt.subplots(2, count, figsize=(4.6 * count, 7.5), squeeze=False, layout="constrained")
    vmax = max(float(np.max(item[f"exit_intensity_{name}_m_inv2"])) * 1e-18 for item in data for name in ("ours", "abtem"))
    for col, (case, item) in enumerate(zip(primary, data)):
        for row, name in enumerate(("ours", "abtem")):
            ax = axes[row, col]
            image = np.pad(item[f"exit_intensity_{name}_m_inv2"] * 1e-18, ((0, 1), (0, 1)), mode="wrap")
            im = ax.imshow(image, origin="lower", extent=real_extent, vmin=0, vmax=vmax, cmap="magma", interpolation="nearest")
            ax.set(title=f"{name}: {case['thickness_nm']:.6f} nm", xlabel="x (Å)", ylabel="y (Å)", xlim=(0, lx), ylim=(0, ly))
    fig.colorbar(im, ax=list(axes.flat), label="Exit intensity (nm$^{-2}$)")
    fig.suptitle("Exit intensities — common scale, incident probability one")
    save(fig, "exit_intensities.png")

    wavelength = results["source_audit"]["ours_wavelength_m"]

    def reciprocal_extent(item):
        ny, nx = item["diffraction_ours_probability_yx"].shape
        extent = item["extent_m"]
        dx, dy = wavelength / extent[0] * 1e3, wavelength / extent[1] * 1e3
        return [(-(nx//2) - 0.5) * dx, ((nx-1)//2 + 0.5) * dx, (-(ny//2) - 0.5) * dy, ((ny-1)//2 + 0.5) * dy]

    fig, axes = plt.subplots(2, count, figsize=(4.6 * count, 8.1), squeeze=False, layout="constrained")
    for col, (case, item) in enumerate(zip(primary, data)):
        for row, name in enumerate(("ours", "abtem")):
            ax = axes[row, col]
            im = ax.imshow(np.fft.fftshift(item[f"diffraction_{name}_probability_yx"]), origin="lower", extent=reciprocal_extent(item), norm=LogNorm(1e-8, 1), cmap="magma", interpolation="nearest")
            ax.set(title=f"{name}: {case['thickness_nm']:.6f} nm", xlabel="θx (mrad)", ylabel="θy (mrad)", xlim=(-50, 50), ylim=(-50, 50))
    fig.colorbar(im, ax=list(axes.flat), label="Probability per reciprocal pixel (log scale)")
    fig.suptitle("Diffraction — same full arrays and common scale; view ±50 mrad")
    save(fig, "diffraction_patterns.png")

    real_differences = [(item["exit_intensity_ours_m_inv2"] - item["exit_intensity_abtem_m_inv2"]) * 1e-18 for item in data]
    diffraction_differences = [item["diffraction_ours_probability_yx"] - item["diffraction_abtem_probability_yx"] for item in data]
    fig, axes = plt.subplots(2, count, figsize=(4.6 * count, 8.1), squeeze=False, layout="constrained")
    for row, differences in enumerate((real_differences, diffraction_differences)):
        vmax = max(float(np.max(np.abs(difference))) for difference in differences)
        vmax = max(vmax, np.finfo(float).eps)
        for col, (case, item, difference) in enumerate(zip(primary, data, differences)):
            ax = axes[row, col]
            image = np.pad(difference, ((0, 1), (0, 1)), mode="wrap") if row == 0 else np.fft.fftshift(difference)
            im = ax.imshow(image, origin="lower", extent=real_extent if row == 0 else reciprocal_extent(item), vmin=-vmax, vmax=vmax, cmap="RdBu_r", interpolation="nearest")
            ax.set(title=f"ours − abTEM: {case['thickness_nm']:.6f} nm", xlabel="x (Å)" if row == 0 else "θx (mrad)", ylabel="y (Å)" if row == 0 else "θy (mrad)")
            if row == 1:
                ax.set(xlim=(-50, 50), ylim=(-50, 50))
            else:
                ax.set(xlim=(0, lx), ylim=(0, ly))
        fig.colorbar(im, ax=list(axes[row]), label="Intensity difference (nm$^{-2}$)" if row == 0 else "Probability difference per reciprocal pixel")
    fig.suptitle("Difference maps retain the incident normalization")
    save(fig, "difference_maps.png")

    keys = ["transmitted_0_0"] + [f"beam_{beam['h']}_{beam['k']}" for beam in results["selected_beams"]]
    cols = min(4, len(keys))
    rows = (len(keys) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.2 * rows), squeeze=False, layout="constrained")
    thicknesses = [case["thickness_nm"] for case in primary]
    for ax, key in zip(axes.flat, keys):
        ax.plot(thicknesses, [case["observables_abtem"][key] for case in primary], "-", label="abTEM")
        ax.plot(thicknesses, [case["observables_ours"][key] for case in primary], "o", markerfacecolor="none", label="ours")
        ax.set(title=key.replace("_", " "), xlabel="Exact thickness (nm)", ylabel="Incident probability fraction")
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(keys):]:
        ax.set_visible(False)
    axes[0, 0].legend()
    fig.suptitle(f"Fixed selected beams — initial {initial_label} comparison")
    save(fig, "beam_intensities.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, kind in zip(axes, ("grid", "slice")):
        for repeat in results["parameters"]["repeats"]:
            entries = [entry for entry in results["convergence"][kind] if entry["z_repeats"] == repeat]
            if entries:
                ax.semilogy([entry["fine"] for entry in entries], [max(entry["max_significant_relative_change"], 1e-16) for entry in entries], "o-", label=f"{entries[0]['thickness_nm']:.6f} nm")
        extra = [case for case in results["additional_refinement_cases"] if case["refinement_kind"] == kind]
        for case in extra:
            x = case["grid"][0] if kind == "grid" else case["slices_per_repeat"]
            y = max(value["max_significant_relative_error"] for value in case["comparison_to_finest_requested"].values())
            ax.semilogy([x], [max(y, 1e-16)], "s", markerfacecolor="none", color="C2", label="Thickest: extra diagnostic")
        ax.axhline(0.01, color="black", linestyle="--", label="1% target")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set(xlabel="Fine grid N (vs previous N)" if kind == "grid" else "Fine slices/repeat (vs previous value)", ylabel="Largest selected-observable relative change", title="Grid: finest requested slicing" if kind == "grid" else "Slicing: finest requested grid")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2, which="both")
    fig.suptitle("Physical convergence — includes both implementations")
    save(fig, "convergence.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for index, (case, item) in enumerate(zip(primary, data)):
        color = f"C{index % 10}"
        axes[0].plot(item["z_edges_m"] * 1e9, item["norm_history_ours"], color=color, label=f"{case['thickness_nm']:.6f} nm: ours")
        axes[0].plot(item["z_edges_m"][::max(1, len(item["z_edges_m"])//12)] * 1e9, item["norm_history_abtem"][::max(1, len(item["z_edges_m"])//12)], "o", markersize=3, markerfacecolor="none", color=color)
    axes[0].axhline(1, linestyle="--", color="black", label="Unfiltered unit norm")
    axes[0].set(xlabel="z (nm)", ylabel="Transverse probability", title="Primary norm histories; circles: abTEM")
    for n in results["parameters"]["grids"]:
        cases = [case for case in results["cases"] if case["grid"][0] == n and case["slices_per_repeat"] == max(results["parameters"]["slices_per_repeat"])]
        axes[1].plot([case["thickness_nm"] for case in cases], [case["filter_norm_loss_ours"] for case in cases], "o-", label=f"N={n}")
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes[1].set(xlabel="Exact thickness (nm)", ylabel="Numerical norm loss", title="Finest requested slicing: filter loss decreases with grid spacing")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    save(fig, "norm_histories.png")
    return figures


def write_report(results: dict, output: Path) -> None:
    primary = _primary(results)
    config = results["parameters"]
    audit = results["source_audit"]
    structure = results["structure"]
    all_cases = results["cases"]
    regressions = results["analytical_regressions"]
    lines = [
        "# Benchmark 2: elastic plane-wave diffraction from Si [110]", "",
        f"Overall acceptance: **{results['status']}**. Implementation agreement on the {len(all_cases)} requested shared-input configurations: **{'PASS' if results['checks']['implementation_agreement_all_requested_configurations'] else 'FAIL'}**. Physical convergence of the finest requested grid and slicing: **{'PASS' if results['convergence']['passed'] else 'FAIL'}**.", "",
        f"Incident kinetic energy: {config['energy_kev']:g} keV. Our relativistic k = {2*np.pi/audit['ours_wavelength_m']:.12e} rad/m and interaction coefficient e/(hbar v) = {audit['ours_sigma_per_v_m']:.12e} (V m)^-1. Double precision is used throughout.", "",
        "## Structure and static specimens", "",
        f"The original LAMMPS data file contains four Si atoms and is interpreted in ångströms (ASE `units='metal'`, `atom_style='atomic'`, explicit type 1 → atomic number 14). Cell lengths are 5.43053, 3.83996, 3.83996 Å. Its SHA256 is `{structure['input_sha256']}`; the file is checked unchanged after the run.",
        f"Periodic nearest-neighbor distances are {structure['nearest_neighbor_min_angstrom']:.10f}–{structure['nearest_neighbor_max_angstrom']:.10f} Å with four image-resolved neighbors per atom, consistent with diamond Si. The projected columns have the supplied [110] dumbbell arrangement, separation {structure['projected_dumbbell_separation_angstrom']:.8f} Å. No rotation, strain, relaxation, shift, thermal displacement, or Debye–Waller smearing is applied.", "",
        "| z repeats | Atoms | Actual cell thickness (nm) |", "|---:|---:|---:|",
    ]
    for specimen in results["specimens"]:
        lines.append(f"| {specimen['repeat'][2]} | {specimen['atoms']} | {specimen['thickness_nm']:.6f} |")
    lines.extend(["",
        f"Propagation follows the supplied z axis. Each specimen repeats only along z; the lateral cell stays fixed. The first case is the thinnest specimen at {config['initial_grid']}² and {config['initial_slices']} slices per original repeat. The abTEM finite-potential builder uses periodic atomic-image padding, including at the z faces. This benchmark therefore truncates propagation through a periodic bulk potential at the exact cell faces; it is not a vacuum–surface–vacuum slab calculation.", "",
        "## Equation and matched conventions", "",
        r"The SI envelope satisfies $\partial_z\phi=i\nabla_\perp^2\phi/(2k)-iH_{el}\phi/(\hbar v)$, with $H_{el}=-eV$, $k=\sqrt{T(T+2m_ec^2)}/(\hbar c)$ and $v=c\sqrt{1-(1+T/(m_ec^2))^{-2}}$.",
        r"The integrated slice $V_j^{proj}=\int V\,dz$ is in V m and gives $t_j=\exp[+ieV_j^{proj}/(\hbar v)]$. The sign is positive because the electron charge is negative. The longitudinal carrier is absent.",
        "Conversions are 1 Å = 1e-10 m, 1 keV = 1000 e joules, and 1 V Å = 1e-10 V m. The installed `QuadratureProjectionIntegrals` integrates between slice bounds, and `PotentialArray._transmission_function` multiplies its array directly by sigma without any dz. A regression also verifies that changing the slice-thickness metadata while keeping the exported array fixed leaves the transmission unchanged. Thus the potential arrays are integrated V Å, despite a generic `eV/e` metadata label. Slice thickness is **not multiplied twice**.", "",
        "abTEM arrays use (slice, x, y); our arrays use (slice, y, x), with an explicit transpose. The array origin stays at the supplied x=y=0; no atom or potential array is shifted. Benchmark 1's translation-invariant free operator is reused unchanged, with NumPy's negative-sign unnormalized forward DFT and positive-sign inverse DFT divided by Nx Ny. Angular frequencies are 2π fftfreq, giving exp[-i dz (kx²+ky²)/(2k)].", "",
        "The native normalized abTEM plane wave has amplitude 1/(Nx Ny). Multiplying by Nx Ny / sqrt(Lx Ly), with SI lengths, gives our entrance amplitude 1/sqrt(Lx Ly) and grid probability one. This conversion is applied once. Diffraction probability per pixel is Lx Ly |DFT(phi)|²/(Nx Ny)²; its sum equals the real-space grid norm. Exit waves and patterns are never renormalized independently.", "",
        "Both implementations apply a filtered transmission first, then free propagation through the full slice. Entrance is z=0 and exit is the exact specimen thickness; there are no half-slice reference-plane adjustments. abTEM uses the conventional Fourier algorithm with order=1, transpose=False, conjugate=False. Our solver computes its own transmission and calls only the validated Benchmark 1 free propagator. It contains no abTEM calls.", "",
        "The antialias aperture is independently calculated and checked against abTEM. In cycles/length, cutoff = 0.6666666/[2 max(dx,dy)] and cosine taper width = 0.01/max(dx,dy). Both transmission functions and propagated waves are filtered. This can remove probability; it is numerical loss, not physical absorption. Separate unfiltered real-potential runs check unitarity.", "",
        "Full uncropped native diffraction arrays are exported using max_angle='full', parity='same', fftshift=False, block_direct=False, renormalize=False. Shared angles use θ=λ f=k_perp/k. Detector masks are [0,10), [10,30), [30,50] mrad; the diffraction-error region is θ≤50 mrad. Figures display ±50 mrad using the actual reciprocal pixels and common scales; Bragg spots are not broadened.", "",
        "The perfect periodic cell's distinct finite slices are generated once per configuration. Both solvers reuse exactly those arrays. Shared-input NPZ files store the actual native unit slices, actual dz values, entrance waves, and explicit ordered slice-index maps and z edges for each complete specimen. This lossless representation avoids storing many identical arrays. A direct repeated-atom build verifies it independently:",
        f"relative potential difference = {results['direct_repeated_potential_verification']['relative_potential_error']:.3e}; relative exit-wave difference = {results['direct_repeated_potential_verification']['relative_exit_wave_error']:.3e}. Primary reference loops also agree with abTEM's public Waves.multislice driver to {max(case['reference_metadata']['public_multislice_driver_relative_wave_error'] for case in primary):.3e}.", "",
        "## Measured implementation agreement", "",
        f"Relative complex-wave error uses the discrete L2 norm, retaining the actual global phase. Diffraction error is sum(|I_ours-I_ref|)/sum(I_ref) inside θ≤50 mrad. All primary comparisons below use {config['initial_grid']}² and {config['initial_slices']} slices/repeat (actual dz={3.83996/config['initial_slices']:.9f} Å).", "",
        "| Thickness (nm) | Relative complex-wave error | Diffraction L1 error | Largest significant intensity error | Norm loss |", "|---:|---:|---:|---:|---:|",
    ])
    for case in primary:
        lines.append(f"| {case['thickness_nm']:.6f} | {case['relative_exit_wave_error']:.6e} | {case['diffraction_intensity_error_up_to_50_mrad']:.6e} | {case['observable_comparison']['max_significant_relative_error']:.6e} | {case['filter_norm_loss_ours']:.6e} |")
    lines.extend(["",
        f"Across every requested grid/slice/thickness configuration: maximum wave error {max(case['relative_exit_wave_error'] for case in all_cases):.6e} (target <1e-6), maximum diffraction error {max(case['diffraction_intensity_error_up_to_50_mrad'] for case in all_cases):.6e} (target <1e-5), and maximum significant beam/detector relative error {max(case['observable_comparison']['max_significant_relative_error'] for case in all_cases):.6e} (target <1%). Every initially empty channel in the primary three-channel runs stays exactly zero.",
        f"The uniform-potential analytical phase error is {regressions['uniform_potential_phase_error']:.3e}, zero-potential plane-wave error {regressions['zero_potential_plane_wave_error']:.3e}, and zero-potential nonconstant-mode error versus Benchmark 1 {regressions['zero_potential_benchmark1_mode_error']:.3e}. Maximum unfiltered Si norm drift is {max(case['max_unfiltered_relative_norm_drift'] for case in primary):.3e}.", "",
        "The residual complex-wave discrepancy is investigated without changing primary results. abTEM's energy routines use ASE's default CODATA 2014 constants; our implementation uses installed scipy.constants. Native-vs-ours sigma differs fractionally by " + f"{audit['sigma_relative_difference_native_to_ours']:.6e}, wavelength by {audit['wavelength_relative_difference_native_to_ours']:.6e}. " + f"At {results['constant_diagnostic']['case']}, substituting the independently reported native constants in a secondary control changes the wave error from {results['constant_diagnostic']['primary_wave_error']:.6e} to {results['constant_diagnostic']['wave_error_using_independently_reported_native_constants_secondary']:.6e}. This is a diagnostic with known constants, not a fitted scale or phase. Secondary phase-aligned errors are also recorded in JSON and never used for acceptance.", "",
        "## Physical convergence and failed criteria", "",
        f"The full grid×slice comparison uses grids {config['grids']} and slices/repeat {config['slices_per_repeat']}, always at fixed physical cell and atomic positions. Actual dz values are {', '.join(f'{3.83996/s:.9f}' for s in config['slices_per_repeat'])} Å. Grid convergence holds slicing at {max(config['slices_per_repeat'])}/repeat; slice convergence holds the grid at {max(config['grids'])}². The finest adjacent pair, not agreement between implementations, determines the requested convergence criterion.", "",
        "Selected fixed reciprocal indices (one representative per Friedel pair): " + ", ".join(f"({beam['h']},{beam['k']})" for beam in results["selected_beams"]) + ". They were chosen from the nonzero thinnest initial reference and remain fixed across thickness, grids and slicing.",
        "An observable is significant at reference probability ≥1e-6; weaker or vanishing references use an absolute probability-error threshold of 1e-8 instead of an unstable ratio. Significant intensity agreement and convergence each retain the fixed 1% threshold. No mean potential removal, phase fit, or output renormalization is used to meet a target.", "",
        "| Refinement | Thickness (nm) | Maximum selected-intensity change | Worst reference observable | Result |", "|---|---:|---:|---|---|",
    ])
    for kind in ("grid", "slice"):
        for entry in results["convergence"][kind]:
            if not entry["finest_requested_pair"]:
                continue
            observations = entry["comparisons"]["abtem"]["observables"]
            worst = max(observations, key=lambda name: observations[name]["relative_error"] or 0)
            lines.append(f"| {kind} {entry['coarse']}→{entry['fine']} | {entry['thickness_nm']:.6f} | {100*entry['max_significant_relative_change']:.4f}% | {worst} | {'PASS' if entry['passed'] else 'FAIL'} |")
    if results["additional_refinement_cases"]:
        lines.extend(["", "Additional thick-specimen diagnostic refinements:", "", "| Configuration | Wave error vs abTEM | Shared-input agreement | Intensity change vs finest requested | 1% refinement check |", "|---|---:|---|---:|---|"])
        for case in results["additional_refinement_cases"]:
            comparisons = case["comparison_to_finest_requested"]
            maximum = max(item["max_significant_relative_error"] for item in comparisons.values())
            passed = all(item["passed"] for item in comparisons.values())
            lines.append(f"| {case['key']} | {case['relative_exit_wave_error']:.6e} | {case['status']} | {100*maximum:.4f}% | {'PASS' if passed else 'FAIL'} |")
        for case in results["additional_refinement_cases"]:
            if "native_constant_control_wave_error_secondary" in case:
                lines.extend(["", f"The additional {case['key']} diagnostic exceeds the fixed 1e-6 raw-phase wave-error target. With the independently reported native constants, its secondary control error is {case['native_constant_control_wave_error_secondary']:.6e}. This explains the constant-table sensitivity on the finer high-frequency grid; its primary error and failed agreement flag are retained.", ""])
    lines.extend(["",
        "The largest requested convergence changes at the longer thicknesses are not roundoff or disagreement between solvers: they occur in the independent reference too. The transmitted beam near its diffraction minimum and thickness-dependent dynamical Bragg intensities amplify the remaining discretization changes. Finite core-potential sampling, antialiasing, and the finite transmission/propagation splitting all remain part of this discretized physical calculation. The additional runs measure their refinement sensitivity; no stable continuum result is claimed when the fixed 1% tests fail.", "",
        "| Finest-slice grid | Norm loss at thickest specimen |", "|---:|---:|",
    ])
    for n in config["grids"]:
        case = next(case for case in all_cases if case["grid"][0] == n and case["slices_per_repeat"] == max(config["slices_per_repeat"]) and case["z_repeats"] == max(config["repeats"]))
        lines.append(f"| {n}² | {100*case['filter_norm_loss_ours']:.6f}% |")
    lines.extend(["", "Norm losses are tracked after every slice in the saved NPZ files for both calculations. Their grid dependence and the unfiltered unitary control identify them as numerical filtering effects. They must not be treated as inelastic or absorptive intensity.", "",
        "## Artifacts, reproduction, and scope", "",
        "`results.json` stores all parameters, native and SI dtypes/units, package versions, inspected-source hashes, raw-phase errors, observables, norm losses, all refinement pairs, and individual checks. `specimens.npz` stores all repeated coordinates and cells; `shared_inputs/*.npz` stores actual unit potential arrays and complete specimen slice maps; `waves/*.npz` stores both complex exit waves, real and reciprocal intensities, masks, and per-slice norms.", "",
        "Run from the project root:", "", "```bash",
        "cd /home/zuxian/Documents/PhD/Yoshioka",
        "python3 -m venv .venv",
        ".venv/bin/python -m pip install -r requirements-benchmark2.txt",
        ".venv/bin/python -m unittest discover -s tests -v",
        ".venv/bin/python -m benchmarks.elastic_si110",
        "```", "",
        "The default benchmark returns exit status 1 when any acceptance criterion fails, including unresolved physical convergence. It still writes complete results, figures, and this report. `--help` lists configurable structure, energy, grids, slicing, thickness repeats and output. `--skip-extra-refinement` omits the additional diagnostic runs but retains every requested grid/slice combination and the same tolerances.", "",
        "Installed scientific versions: " + ", ".join(f"{name} {version}" for name, version in audit["versions"].items()) + ". Potential arrays are float64; both exit waves are complex128; native diffraction is float64. No reference source code is patched.", "",
        "Reference documentation: [potentials](https://abtem.github.io/doc/user_guide/walkthrough/potentials.html), [multislice](https://abtem.github.io/doc/user_guide/walkthrough/multislice.html), [antialiasing](https://abtem.github.io/doc/user_guide/appendix/antialiasing.html). Installed-source behavior and measured dtypes take precedence over ambiguous generic unit labels.", "",
        "This stage intentionally shares Lobato potentials. It checks propagation correctness through a specified static potential and does not independently validate atomic-potential generation. Agreement alone does not establish physical convergence. It does not validate phonon scattering, thermal averaging, or EELS spectral normalization.", "",
    ])
    for name in results.get("figures", []):
        lines.extend([f"![{name.replace('_', ' ').removesuffix('.png')}]({name})", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
