"""Figures and scientific report for the independently validated MOD lattice."""

from pathlib import Path

import numpy as np
from scipy.constants import elementary_charge, hbar


def write_outputs(results, unit, reference, fitted, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = []
    def save(fig, name):
        fig.savefig(output/name, dpi=170, bbox_inches="tight");plt.close(fig);figures.append(name)
    vertices = np.array([[0,0,0],[.5,0,0],[.5,.5,0],[0,0,0],[0,0,.5]])
    segments = [np.linspace(a,b,41,endpoint=False) for a,b in zip(vertices[:-1],vertices[1:])]
    points = np.vstack((*segments,vertices[-1:]))
    first = reference.modes(points)[0]*hbar/(1e-3*elementary_charge)
    second = fitted.modes(points)[0]*hbar/(1e-3*elementary_charge)
    fig, axes = plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
    for branch in range(first.shape[1]):
        axes[0].plot(first[:,branch],"k-",linewidth=1,label="Central-displacement reference" if branch==0 else None)
        axes[0].plot(second[:,branch],"C1--",linewidth=.9,label="hiPhive second order" if branch==0 else None)
    axes[0].set(xticks=np.arange(5)*41,xticklabels=["Γ","(½,0,0)","(½,½,0)","Γ","(0,0,½)"],ylabel="Phonon energy (meV)",title="Oriented four-atom cell: twelve folded branches")
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    mesh=20
    q=np.array(list(np.ndindex(mesh,mesh,mesh)),dtype=float)/mesh
    omega,vectors=fitted.modes(q)
    energies=omega*hbar/(1e-3*elementary_charge)
    physical=np.ones(omega.shape,dtype=bool)
    translations=np.tile(np.eye(3),(len(fitted.positions_m),1))*np.repeat(np.sqrt(fitted.masses_kg),3)[:,None]
    translations/=np.linalg.norm(translations,axis=0)
    overlap=np.sum(abs(vectors[0].conj().T@translations)**2,axis=-1)
    physical[0,np.argsort(overlap)[-3:]]=False
    if np.any(energies[physical]<=0):
        raise ValueError('a physical mode is unstable in the DOS mesh')
    bins=np.linspace(0,90,181)
    density,edges=np.histogram(energies[physical],bins=bins)
    density=density/(mesh**3*np.diff(edges))
    results['dos_normalization']={'q_mesh':[mesh]*3,'removed_Gamma_translations':3,
                                 'expected_modes_per_oriented_cell':float(np.sum(physical)/mesh**3),
                                 'integrated_modes_per_oriented_cell':float(np.sum(density*np.diff(edges))),
                                 'units':'modes per oriented four-atom cell per meV'}
    axes[1].step((edges[:-1]+edges[1:])/2,density,where="mid")
    axes[1].set(xlabel="Phonon energy (meV)",ylabel="DOS (modes / oriented cell / meV)",title="Uniform 20³ oriented reciprocal mesh")
    axes[1].grid(alpha=.2)
    save(fig,"dispersion_and_dos.png")
    np.savez_compressed(output/"dispersion_dos.npz",band_q_fractional=points,reference_band_mev=first,fitted_band_mev=second,dos_bin_edges_mev=edges,dos_modes_per_cell_per_mev=density,dos_removed_Gamma_translations=3,dos_expected_modes_per_oriented_cell=np.sum(physical)/mesh**3)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout="constrained")
    for category,xkey,ax in (("training_size","training_independent_pairs",axes[0]),("displacement","sigma_angstrom",axes[1]),("second_order_cutoff","second_order_cutoff_angstrom",axes[2])):
        rows=sorted((row for row in results["fit_study"] if row["category"]==category),key=lambda row:row[xkey])
        ax.semilogy([row[xkey] for row in rows],[row["relative_force_constant_error"] for row in rows],"o-",label="Relative FC error")
        ax.semilogy([row[xkey] for row in rows],[row["heldout_relative_force_rms_error"] for row in rows],"s-",label="Held-out raw MOD force error")
        ax.axhline(results["parameters"]["force_constant_relative_tolerance"],color="k",linestyle="--",label="FC target")
        ax.set(xlabel=xkey.replace("_"," "),ylabel="Relative error");ax.grid(alpha=.2);ax.legend(fontsize=7)
    save(fig,"hiphive_fit_convergence.png")
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.6),layout="constrained")
    for repetitions in results["parameters"]["reference_supercells"]:
        rows=sorted((row for row in results["reference_study"] if tuple(row["supercell"])==tuple(repetitions)),key=lambda row:row["displacement_angstrom"])
        xx=[row["displacement_angstrom"] for row in rows]
        yy=[row["maximum_energy_difference_from_finest_mev"] for row in rows]
        valid=np.array(yy)>0
        axes[0].loglog(np.array(xx)[valid],np.array(yy)[valid],"o-",label=str(tuple(repetitions)))
    axes[0].axhline(results["parameters"]["phonon_energy_tolerance_mev"],color="k",linestyle="--",label="0.1 meV target")
    axes[0].set(xlabel="Central displacement (Å)",ylabel="Max energy difference from finest reference (meV)",title="Force-displacement and force-supercell convergence")
    axes[0].grid(alpha=.2);axes[0].legend(fontsize=8)
    distance=np.linalg.norm(reference.relative_vectors_m,axis=1)*1e10
    strengths=np.linalg.norm(reference.phi_n_per_m,axis=(1,2))/(elementary_charge/1e-20)
    axes[1].semilogy(distance,strengths,"o")
    axes[1].axvline(3.3,color="k",linestyle="--",label="MOD pair cutoff")
    axes[1].set(xlabel="Pair separation (Å)",ylabel="Force-constant block norm (eV/Å²)",title="Measured many-body harmonic range")
    axes[1].grid(alpha=.2);axes[1].legend(fontsize=8)
    save(fig,"force_constant_convergence_and_range.png")
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),layout="constrained")
    rows=results["covariance_study"]
    for t,index in ((0,0),(300,1)):
        traces=[float(np.mean(np.trace(np.array(row["covariance_m2"])[index],axis1=-2,axis2=-1)))*1e20 for row in rows]
        axes[0].plot([row["mesh"][0] for row in rows],traces,"o-",label=f"T={t} K")
    axes[0].set(xlabel="Oriented reciprocal mesh n × n × n",ylabel="Mean trace of displacement covariance (Å²)")
    changes=[row["relative_change_from_previous"] for row in rows[1:]]
    axes[1].semilogy([row["mesh"][0] for row in rows[1:]],changes,"o-")
    axes[1].axhline(results["parameters"]["covariance_relative_tolerance"],color="k",linestyle="--",label="Declared covariance target")
    axes[1].set(xlabel="Mesh n",ylabel="Max relative covariance change")
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,"covariance_convergence.png")
    lines=["# Benchmark 5: harmonic Si dynamics from the supplied Kumagai MOD potential","",f"Overall numerical/model-reproduction status: **{results['status']}**.","",
           "This calculation uses the original Si.tersoff.mod through LAMMPS pair_style tersoff/mod, with metal units, atomic style, and periodic boundaries. No ordinary-Tersoff substitute, MD, electron propagation, or DFT reference is used. The original structure and potential are unchanged.","",
           "## Equilibrium and geometry","",f"Cubic lattice parameter: {results['equilibrium']['lattice_parameter_angstrom']:.12f} Å. Energy: {results['equilibrium']['energy_ev_per_atom']:.12f} eV/atom. Maximum residual force: {results['equilibrium']['maximum_force_ev_per_angstrom']:.3e} eV/Å. Pressure tensor (bar): {results['equilibrium']['pressure_bar_voigt']}.",
           "The supplied orthorhombic orientation and fractional coordinates are preserved. The tiny input rounding mismatch between a and sqrt(2)b is repaired explicitly, then the isotropic cubic scale is optimized by the zero-pressure condition. Separate energy/pressure evaluations at ±1e-4 and ±2e-5 strain verify the minimum. Internal relaxation is unnecessary at the measured force level. The original-to-equilibrium Cartesian transform, atom mapping, stress, energies, and volume checks are saved in equilibrium.json. This new geometry differs from Benchmark 2; none of Benchmark 2's inputs or results is overwritten.","",
           "## Independent force constants and fitting","",
           "Central differences displace each of the four basis atoms along all three Cartesian directions by both signs; 24 independent MOD evaluations per supercell/amplitude are sufficient because exact lattice translations reconstruct every remaining column. Displaced structures are never relaxed. Force constants are Phi_iα,jβ=−dF_iα/du_jβ. Raw and corrected full arrays are saved separately. The correction is P[(H+H^T)/2]P, with P projecting only the three rigid translation vectors. Its norm and phonon changes are recorded; no physical negative eigenvalue is clipped.",
           "The raw non-Hermitian finite-difference matrix is not treated as an exact Hamiltonian. Its permutation and Fourier-Hermiticity residuals are reported, and raw-frequency diagnostics explicitly use its Hermitian part before the translational correction. The final stability check uses the corrected model, removes only the three known Gamma translations, and retains all nonzero-q acoustic modes.","",
           "| Force supercell | Displacement (Å) | Raw permutation residual | Correction norm | Max energy error vs finest (meV) |","|---|---:|---:|---:|---:|"]
    for row in results["reference_study"]:
        lines.append(f"| {row['supercell']} | {row['displacement_angstrom']} | {row['raw_residuals']['permutation']:.3e} | {row['relative_sum_rule_and_permutation_correction']:.3e} | {row['maximum_energy_difference_from_finest_mev']:.6e} |")
    lines.extend(["",f"The measured nonzero harmonic range is {max(row['range']['maximum_nonzero_range_angstrom'] for row in results['reference_study']):.8f} Å, exceeding the 3.3 Å pair cutoff because neighbours can couple through a shared centre. Both reference supercells exceed twice the conservative 6.6 Å many-body bound. Training supercells satisfy the measured FC range and hiPhive cluster-image requirements.",
                  "hiPhive fits only second order: each ClusterSpace cutoff list is [radius], not a list of several expansion orders. Radius, paired training-set size, displacement magnitude, supercell, and ridge setting are independently swept. Positive/negative force pairs isolate the odd response and diagnose even anharmonic contamination. Training/validation seeds are fixed and distinct. Held-out actual MOD forces are evaluated independently; paired and independent finite-displacement predictions are also compared. Small regression error is model reproduction, not improved physical accuracy over MOD.","",
                  "| Category | Training supercell | Pairs | σ (Å) | Radius (Å) | Relative FC error | Held-out force error | Phonon error (meV) |","|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in results["fit_study"]:
        lines.append(f"| {row['category']} | {row['training_supercell']} | {row['training_independent_pairs']} | {row['sigma_angstrom']} | {row['second_order_cutoff_angstrom']} | {row['relative_force_constant_error']:.6e} | {row['heldout_relative_force_rms_error']:.6e} | {row['maximum_phonon_energy_error_mev']:.6e} |")
    lines.extend(["","## Fourier, eigenvectors, and covariance conventions","",
                  "Row real-space vectors A retain the supplied Cartesian x,y,z orientation, with the beam direction z=[110]. Reciprocal rows are B=2πA^-T in angular inverse metres. The Phonopy atomic gauge is D_ab(q)=sum_R Phi(0a,Rb) exp[+iq·(R+tau_b−tau_a)]/sqrt(Ma Mb). Eigenvectors are normalized by sum_iα |e_iα|²=1; dynamical eigenvalues are angular omega². All reusable arrays use metres, kilograms, and N/m. Exported LAMMPS/Phonopy FC arrays use eV/Å². Masses are 28.085 u, converted with the same SciPy atomic mass constant as prior benchmarks.",
                  "Phonopy frequencies are ordinary THz; omega=2πf and energy=h f=hbar omega. Its default native conversion factor and an explicitly selected factor from project SciPy constants are both recorded. The two-atom primitive cell has six branches; the oriented four-atom cell has twelve folded branches. The primitive/oriented cell and atom mappings and the two primitive q cosets are saved. Degenerate eigenspaces are compared using projectors instead of raw phases/rotations.",f"Phonopy validation: {results['phonopy_validation']}","",
                  "Displacement covariance at 0 and 300 K sums hbar/(2M omega) times the normalized eigenvector outer product and coth(hbar omega/(2kBT)), with q weights 1/Nq exactly once. Only Gamma's three explicitly identified rigid translations are omitted; physical low-q acoustic modes are retained. Force supercell size and phonon sampling are independent. W_i(Q,T)=Q^T covariance_i(T) Q/2 is dimensionless, with angular Q in inverse metres.",
                  f"The DOS uses the same explicit Gamma translation identification, without clipping frequencies. Its finite-mesh integral is {results['dos_normalization']['integrated_modes_per_oriented_cell']:.9f} modes per oriented cell, equal to 12−3/20³; no DOS renormalization is applied.","",
                  "| q mesh | Maximum relative covariance change |","|---|---:|"])
    for row in rows:lines.append(f"| {row['mesh']} | {row['relative_change_from_previous']} |")
    lines.extend(["","## Acceptance and reproduction","","| Check | Status |","|---|---|"])
    for name,value in results["checks"].items():lines.append(f"| {name} | {'PASS' if value else 'FAIL'} |")
    lines.extend(["",f"Costs: {results['resource_costs']}. Environment: {results['environment']}. Actual physical constants and source hashes are in results.json. Prior statuses preserved: {results['prior_benchmark_statuses_preserved']}; Benchmark 3B remains unavailable, with no status changed.","",
                  "```bash","cd /home/zuxian/Documents/PhD/Yoshioka",".venv/bin/python -m pip install -r requirements-lattice.txt",".venv/bin/python -m benchmarks.si_lattice_dynamics --config benchmarks/configs/lattice.json",".venv/bin/python -m unittest discover -s tests -v","```","",
                  "manifest.json identifies equilibrium geometry, fitted and independent FC models, the trained hiPhive model, Phonopy-compatible arrays, covariances, units, conventions, checksums, and statuses for Benchmarks 6–7. Valid cached force constants and training forces are reused on reruns. Passing is numerical agreement with the supplied MOD model, not validation against experiment or DFT.",""])
    for name in figures:lines.extend([f"![{name}]({name})",""])
    (output/"report.md").write_text("\n".join(lines))
    return figures
